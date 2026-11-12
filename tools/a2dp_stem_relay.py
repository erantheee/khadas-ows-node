#!/usr/bin/env python3
"""A2DP -> board NPU (stem separation / envelope) -> local playout relay.

This is the listening/integration counterpart of `tools/a2dp_sink_capture.sh`:
it takes the audio the board receives over A2DP, runs it through the board's own
C++ node + NPU worker (`khadas-ows-node` on 127.0.0.1:9100), and plays the
*processed* stream out of a local sink (the USB DAC used for listening).

Signal path (all on the board):

    phone --A2DP--> bluez_input.<mac> --> virtual sink "a2dp_stem_in" (silent)
                                                   |
                                      parec 44.1 kHz float32 stereo
                                                   v
                        this relay ==TCP 9100==> C++ node --> NPU worker
                                                   ^            (--mode npu,
                                                   |             --policy s3,
                                        processed PCM             --output-mode envelope)
                                                   v
                          pacat --> USB DAC sink --> wired earphone

Why the intermediate virtual sink: the A2DP stream must NOT go directly to the
earphone, otherwise the direct + processed copies would mix.  The relay makes a
null sink the default sink, so every new received A2DP stream is linked there
(silent), captures its monitor, and plays only the processed audio to the DAC.

What this is and is not:
  * it is an integration probe: proves the phone->board->NPU->earphone path and
    reports node status (actual_mode / RTF / fallbacks) and relay (RTF, underruns);
  * it is not a product latency claim: add ~1 s capture window + NPU RTT + pre-roll
    on top of the ~0.4 s A2DP leg (typical measured total here is 2.5-4 s);
  * a route guard polls every second and moves any received A2DP stream back onto the
    silent sink (WirePlumber stream-restore can otherwise re-link it to the earphone);
  * the C++ node serves one audio client at a time. While the relay runs, the
    Mac-side five-stem bridge cannot use port 9100.

Usage (board, after tools/a2dp_sink_setup.sh and tools/a2dp_sink_usb_output.sh):

    ~/khadas-ows-node/.venv/bin/python tools/a2dp_stem_relay.py
    # optional: --record /tmp/processed.wav --stats-json /tmp/relay.json
    #           --preroll 1 (lower latency, more underrun risk)
    #           --duration 60   (stop after 60 s)

Exit code 0 = ran and cleaned up; 2 = precondition failure (node/worker/pactl).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<IIQIHHi")
MAGIC = 0x4B4F5753
SCHEMA = 1
SAMPLE_RATE = 44100
CHANNELS = 2
BYTES_PER_FRAME = CHANNELS * 4
HTTP_TIMEOUT = 2.0

_stop = threading.Event()


def _on_signal(signum, frame):  # noqa: ARG001
    _stop.set()


class Float32WavWriter:
    """Minimal streaming WAV writer for 32-bit IEEE-float PCM (format 3)."""

    def __init__(self, path: str):
        self._fh = open(path, "wb")
        self._bytes = 0
        self._fh.write(b"RIFF" + struct.pack("<I", 0) + b"WAVEfmt " + struct.pack(
            "<IHHIIHH", 16, 3, CHANNELS, SAMPLE_RATE, SAMPLE_RATE * BYTES_PER_FRAME,
            BYTES_PER_FRAME, 32) + b"data" + struct.pack("<I", 0))

    def write(self, pcm: bytes) -> None:
        self._fh.write(pcm)
        self._bytes += len(pcm)

    def close(self) -> None:
        self._fh.seek(4)
        self._fh.write(struct.pack("<I", 36 + self._bytes))
        self._fh.seek(40)
        self._fh.write(struct.pack("<I", self._bytes))
        self._fh.close()


def http_get_json(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=HTTP_TIMEOUT) as reply:
        return json.loads(reply.read().decode())


def node_status(port: int) -> dict:
    return http_get_json(port, "/v1/status")


def ensure_npu(port: int) -> dict:
    status = node_status(port)
    if status.get("requested_mode") == "npu":
        return status
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/mode", data=b"npu", method="POST")
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as reply:
        reply.read()
    return node_status(port)


def pctl(*args: str) -> str:
    result = subprocess.run(["pactl", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pactl {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def list_sinks() -> list[tuple[int, str, str]]:
    sinks = []
    for line in pctl("list", "short", "sinks").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            sinks.append((int(parts[0]), parts[1], parts[2]))
    return sinks


def pick_sink(explicit: str | None) -> str:
    sinks = list_sinks()
    if explicit:
        for _idx, name, _in_fmt in sinks:
            if name == explicit:
                return name
        raise RuntimeError(f"sink {explicit!r} not found; sinks: {[s[1] for s in sinks]}")
    for _idx, name, _in_fmt in sinks:
        if name.startswith("alsa_output.usb-"):
            return name
    raise RuntimeError(
        "no USB audio sink found; plug a USB DAC / USB-C earphone, or pass --sink. "
        f"sinks: {[s[1] for s in sinks]}")


def setup_null_sink(name: str) -> tuple[int, str]:
    """Create the silent capture sink, make it default, return (module_id, previous_default)."""
    module = pctl(
        "load-module", "module-null-sink", f"sink_name={name}",
        "sink_properties=device.description=A2DP-Stem-In")
    previous = pctl("get-default-sink")
    pctl("set-default-sink", name)
    return int(module), previous


def move_existing_inputs(sink: str) -> int:
    """Move already-playing streams (e.g. the phone that is already streaming) to the sink."""
    moved = 0
    for line in pctl("list", "short", "sink-inputs").splitlines():
        parts = line.split("\t")
        if parts:
            try:
                pctl("move-sink-input", parts[0], sink)
                moved += 1
            except RuntimeError:
                pass
    return moved


def resolve_pipewire_source(name: str, timeout_s: float) -> str:
    """Wait for the phone's A2DP source node and return its actual node name.

    The numeric suffix of ``bluez_input.<MAC>.<n>`` is assigned by BlueZ/PipeWire and can
    change after the phone reconnects, so the resolver also accepts the MAC prefix (any
    suffix). `pw-record --target NAME` silently falls back to the default source when the
    exact name is missing, so the relay must never capture until this resolves.
    """
    prefix = name.rsplit(".", 1)[0] + "." if name.count(".") >= 2 else name
    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
    while not _stop.is_set() and (deadline is None or time.monotonic() < deadline):
        try:
            # pactl/pipewire-pulse does NOT list bluez_input sources on this board even
            # when the node exists; pw-cli sees the real PipeWire graph.
            listing = subprocess.run(
                ["pw-cli", "ls", "Node"],
                capture_output=True, text=True, timeout=3.0).stdout
        except (OSError, subprocess.SubprocessError):
            listing = ""
        for line in listing.splitlines():
            line = line.strip()
            if line.startswith('node.name = "'):
                node_name = line[len('node.name = "'):-1]
                if node_name == name or node_name.startswith(prefix):
                    return node_name
        time.sleep(1.0)
    raise RuntimeError(f"PipeWire source {name!r} did not appear "
                       f"within {timeout_s:g}s (--start-timeout 0 waits)")


def direct_links(source: str, sink: str, *, connect: bool) -> None:
    """Connect or disconnect only the direct Bluetooth source->sink stereo links."""
    option = "" if connect else "-d"
    for channel in ("FL", "FR"):
        command = ["pw-link"]
        if option:
            command.append(option)
        command.extend((f"{source}:output_{channel}", f"{sink}:playback_{channel}"))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 and connect:
            raise RuntimeError(f"{' '.join(command)}: {result.stderr.strip()}")


def source_watch_loop(source_name: str, stop: threading.Event) -> None:
    """End the session the moment the expected capture node disappears.

    `pw-record --target NAME` silently falls back to the DEFAULT source when NAME is
    dropped mid-session, so a relay that lost the phone used to keep "processing" the
    DMIC array. Poll the graph and tear down so the re-arm loop re-gates on the phone.
    """
    while not stop.is_set():
        try:
            listing = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True,
                                     text=True, timeout=3.0).stdout
        except (OSError, subprocess.SubprocessError):
            listing = ""
        if f'node.name = "{source_name}"' not in listing:
            print("[relay] source node lost; ending session to re-arm", flush=True)
            stop.set()
            return
        stop.wait(2.0)


def route_guard_loop(target_sink: str, stats: dict, stats_lock: threading.Lock,
                     stop: threading.Event) -> None:
    """Keep every received A2DP stream on the silent capture sink.

    WirePlumber's stream-restore remembers the sink of the previous "oowow (codec SBC)"
    stream, so a newly created A2DP stream can be linked straight to the earphone sink
    even though the default sink is the silent one. Moving it is cheap and idempotent.
    """
    target_index = None
    while not stop.is_set():
        try:
            sinks = pctl("list", "short", "sinks").splitlines()
            for line in sinks:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1] == target_sink:
                    target_index = int(parts[0])
                    break
            result = subprocess.run(["pactl", "-f", "json", "list", "sink-inputs"],
                                    capture_output=True, text=True)
            entries = json.loads(result.stdout) if result.stdout.strip() else []
        except Exception:  # noqa: BLE001 - best effort routing
            entries = []
        for entry in entries:
            if target_index is not None and entry.get("sink") == target_index:
                continue  # already on the silent capture sink
            props = entry.get("properties") or {}
            is_a2dp = props.get("device.api") == "bluez5" or "(codec " in props.get("media.name", "")
            if not is_a2dp:
                continue
            try:
                pctl("move-sink-input", str(entry["index"]), target_sink)
            except RuntimeError:
                continue
            with stats_lock:
                stats["a2dp_stream_seen"] = True
                stats["a2dp_moves"] = stats.get("a2dp_moves", 0) + 1
            print(f"[relay] A2DP stream #{entry['index']} moved to {target_sink}", flush=True)
        stop.wait(1.0)


def reader_loop(proc: subprocess.Popen, out: queue.Queue, chunk_bytes: int,
                stats: dict, stats_lock: threading.Lock, stop: threading.Event) -> None:
    buffer = bytearray()
    while not stop.is_set():
        data = proc.stdout.read(chunk_bytes - len(buffer))
        if not data:
            break
        buffer.extend(data)
        if len(buffer) == chunk_bytes:
            try:
                out.put(bytes(buffer), timeout=2.0)
            except queue.Full:
                with stats_lock:
                    stats["dropped_input_chunks"] += 1
            buffer.clear()
    stop.set()


def read_pipe_chunk(proc: subprocess.Popen, size: int) -> bytes | None:
    """Read one exact PCM chunk from a live capture process."""
    data = bytearray()
    while len(data) < size:
        block = proc.stdout.read(size - len(data))
        if not block:
            return None
        data.extend(block)
    return bytes(data)


def processor_loop(args: argparse.Namespace, in_q: queue.Queue, out_q: queue.Queue,
                   stats: dict, stats_lock: threading.Lock,
                   stop: threading.Event) -> None:
    chunk_frames = int(round(SAMPLE_RATE * args.window_ms / 1000.0))
    chunk_bytes = chunk_frames * BYTES_PER_FRAME
    sock = None
    seq = 0
    rtts: list[float] = []
    try:
        sock = socket.create_connection((args.node_host, args.node_port), timeout=5.0)
        sock.settimeout(args.node_timeout)
        while not stop.is_set():
            try:
                payload = in_q.get(timeout=1.0)
            except queue.Empty:
                continue
            header = HEADER.pack(MAGIC, SCHEMA, seq, SAMPLE_RATE, CHANNELS, 0, chunk_frames)
            started = time.perf_counter()
            sock.sendall(header)
            sock.sendall(payload)
            reply = _recv_exact(sock, HEADER.size)
            if reply is None:
                raise ConnectionError("node closed the connection (busy with another client?)")
            magic, schema, r_seq, rate, channels, _reserved, frames = HEADER.unpack(reply)
            if (magic, schema, rate, channels, frames) != (MAGIC, SCHEMA, SAMPLE_RATE, CHANNELS,
                                                           chunk_frames):
                raise ConnectionError(f"bad reply header {reply.hex()}")
            processed = _recv_exact(sock, chunk_bytes)
            if processed is None:
                raise ConnectionError("short reply payload")
            rtt = (time.perf_counter() - started) * 1000.0
            rtts.append(rtt)
            try:
                out_q.put(processed, timeout=5.0)
            except queue.Full:
                with stats_lock:
                    stats["dropped_output_chunks"] += 1
            samples = np.frombuffer(payload, dtype="<f4")
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            seq += 1
            with stats_lock:
                stats["chunks"] = seq
                stats["last_input_peak"] = round(peak, 4)
                if peak > 0.001:
                    stats["captured_audio_seconds"] = (
                        stats.get("captured_audio_seconds", 0.0) + args.window_ms / 1000.0)
                    if "first_audio_at_s" not in stats:
                        stats["first_audio_at_s"] = round(seq * args.window_ms / 1000.0, 1)
                stats["last_rtt_ms"] = rtt
                stats["rtf"] = rtt / (chunk_frames / SAMPLE_RATE) / 1000.0
                stats["rtt_p50_ms"] = sorted(rtts)[len(rtts) // 2]
                stats["rtt_p95_ms"] = sorted(rtts)[min(len(rtts) - 1, int(len(rtts) * 0.95))]
    except Exception as exc:  # noqa: BLE001 - surfaced in stats and exit code
        with stats_lock:
            stats["processor_error"] = f"{type(exc).__name__}: {exc}"
        stop.set()
    finally:
        if sock is not None:
            sock.close()


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        try:
            data = sock.recv(size - len(chunks))
        except socket.timeout:
            return None
        if not data:
            return None
        chunks.extend(data)
    return bytes(chunks)


def player_loop(args: argparse.Namespace, out_q: queue.Queue, pacat: subprocess.Popen,
                writer: "Float32WavWriter | None", stats: dict,
                stats_lock: threading.Lock, stop: threading.Event) -> None:
    chunk_frames = int(round(SAMPLE_RATE * args.window_ms / 1000.0))
    chunk_bytes = chunk_frames * BYTES_PER_FRAME
    subs = max(1, int(args.window_ms // 50))          # 50 ms sub-writes keep the server buffer small
    sub_bytes = chunk_bytes // subs
    sub_duration = (args.window_ms / 1000.0) / subs
    silence = b"\x00" * chunk_bytes
    # pre-roll: wait until enough processed audio is queued to absorb the NPU RTT jitter
    deadline = time.monotonic() + args.preroll_timeout
    while (not stop.is_set() and out_q.qsize() < args.preroll
           and time.monotonic() < deadline):
        time.sleep(0.05)
    with stats_lock:
        stats["preroll_chunks"] = out_q.qsize()
    next_tick = time.monotonic()
    underruns = 0
    chunk_index = 0
    while not stop.is_set():
        try:
            processed = out_q.get(timeout=min(0.5, sub_duration))
        except queue.Empty:
            processed = None
        if processed is None:
            underruns += 1
            with stats_lock:
                stats["underruns"] = underruns
            processed = silence
        if writer is not None:
            writer.write(processed)
        view = memoryview(processed)
        for index in range(subs):
            part = view[index * sub_bytes:(index + 1) * sub_bytes]
            try:
                pacat.stdin.write(part)
                pacat.stdin.flush()
            except (BrokenPipeError, ValueError):
                stop.set()
                return
            next_tick += sub_duration
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()      # fell behind: re-anchor instead of bursting
        with stats_lock:
            stats["played_seconds"] = (chunk_index + 1) * args.window_ms / 1000.0
        chunk_index += 1


def status_loop(args: argparse.Namespace, stats: dict, stats_lock: threading.Lock,
                stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            status = node_status(args.http_port)
            with stats_lock:
                stats["node"] = {
                    "requested_mode": status.get("requested_mode"),
                    "actual_mode": status.get("actual_mode"),
                    "fallback_reason": status.get("fallback_reason"),
                    "packets": status.get("packets"),
                    "fallbacks": status.get("fallbacks"),
                    "last_rtf": status.get("last_rtf"),
                }
            print(f"[relay] chunks={stats.get('chunks', 0)} rtt={stats.get('last_rtt_ms', 0):.0f}ms "
                  f"rtf={stats.get('rtf', 0):.2f} underruns={stats.get('underruns', 0)} "
                  f"node={status.get('actual_mode')} node_rtf={status.get('last_rtf', 0):.2f} "
                  f"fallbacks={status.get('fallbacks')} audio_s={stats.get('captured_audio_seconds', 0):.0f} "
                  f"a2dp_moves={stats.get('a2dp_moves', 0)}", flush=True)
        except Exception:  # noqa: BLE001 - status is best-effort telemetry
            pass
        stop.wait(args.status_interval)


def write_stats(path: str, stats: dict) -> None:
    try:
        Path(path).write_text(json.dumps(stats, indent=2) + "\n")
    except OSError as exc:
        print(f"[relay] cannot write stats: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--node-host", default="127.0.0.1")
    parser.add_argument("--node-port", type=int, default=9100)
    parser.add_argument("--http-port", type=int, default=8787)
    parser.add_argument("--window-ms", type=int, default=1000,
                        help="pipe packet length; 1000 ms keeps ~1 NPU window per packet")
    parser.add_argument("--preroll", type=int, default=2,
                        help="processed chunks queued before playback starts (default 2)")
    parser.add_argument("--preroll-timeout", type=float, default=30.0)
    parser.add_argument("--start-timeout", type=float, default=15.0,
                        help="seconds to wait for non-silent source audio before moving "
                             "the links; 0 = wait until the phone starts playing")
    parser.add_argument("--stall-timeout", type=float, default=20.0,
                        help="tear down and re-arm when no chunk has progressed for this "
                             "long (node restart / dead socket)")
    parser.add_argument("--node-timeout", type=float, default=10.0)
    parser.add_argument("--sink", help="playback sink name (default: first USB sink)")
    parser.add_argument("--null-sink", default="a2dp_stem_in",
                        help="silent capture sink made default during the relay")
    parser.add_argument("--external-source", metavar="MONITOR",
                        help="capture this source instead of creating the null sink")
    parser.add_argument("--pipewire-source", metavar="NODE",
                        help="capture this PipeWire output node directly with pw-record; "
                             "the phone stream is still moved to the silent sink")
    parser.add_argument("--capture-latency-ms", type=int, default=100)
    parser.add_argument("--playback-latency-ms", type=int, default=1200)
    parser.add_argument("--record", metavar="PATH", help="also write the processed stream to a float32 WAV")
    parser.add_argument("--stats-json", metavar="PATH")
    parser.add_argument("--duration", type=float, help="stop after N seconds")
    parser.add_argument("--status-interval", type=float, default=5.0)
    args = parser.parse_args(argv)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    stats: dict = {"chunks": 0, "underruns": 0, "dropped_input_chunks": 0,
                   "dropped_output_chunks": 0, "captured_audio_seconds": 0.0, "a2dp_moves": 0}
    stats_lock = threading.Lock()

    attempt = 0
    while not _stop.is_set():
        attempt += 1
        session_stop = threading.Event()
        _run_session(args, stats, stats_lock, session_stop)
        if _stop.is_set():
            break
        print(f"[relay] session {attempt} ended; re-arming after the phone source returns",
              flush=True)
        time.sleep(2.0)

    with stats_lock:
        stats["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        stats["node_final"] = stats.get("node")
        stats["ok"] = (
            stats.get("chunks", 0) > 0
            and stats.get("underruns", 0) == 0
            and not stats.get("processor_error")
            and (stats.get("node") or {}).get("actual_mode") == "npu"
        )
    if args.stats_json:
        write_stats(args.stats_json, stats)
    print(f"[relay] stopped: chunks={stats.get('chunks')} underruns={stats.get('underruns')} "
          f"node={stats.get('node_final')} ok={stats.get('ok')}", flush=True)
    return 0 if not stats.get("processor_error") else 1


def _run_session(args: argparse.Namespace, stats: dict, stats_lock: threading.Lock,
                 stop: threading.Event) -> None:
    """One capture->node->playback session with teardown. Callers re-arm after it ends."""
    capture = player = None
    module_id = None
    previous_default = None
    writer = None
    direct_unlinked = False
    source = ""
    sink = ""
    try:
        status = ensure_npu(args.http_port)
        print(f"[relay] node requested={status.get('requested_mode')} "
              f"actual={status.get('actual_mode')}", flush=True)

        if args.external_source:
            source = args.external_source
            sink = pick_sink(args.sink)
        elif args.pipewire_source:
            source = resolve_pipewire_source(args.pipewire_source, args.start_timeout)
            args.pipewire_source = source  # teardown restores links on the real node name
            sink = pick_sink(args.sink)
        else:
            if args.null_sink in [name for _i, name, _f in list_sinks()]:
                raise RuntimeError(f"sink {args.null_sink!r} already exists; is a relay already running?")
            module_id, previous_default = setup_null_sink(args.null_sink)
            moved = move_existing_inputs(args.null_sink)
            source = f"{args.null_sink}.monitor"
            sink = pick_sink(args.sink)
            print(f"[relay] silent sink {args.null_sink} (module {module_id}), moved {moved} "
                  f"playing stream(s); playback -> {sink}", flush=True)

        chunk_frames = int(round(SAMPLE_RATE * args.window_ms / 1000.0))
        chunk_bytes = chunk_frames * BYTES_PER_FRAME
        in_q: queue.Queue = queue.Queue(maxsize=4)
        out_q: queue.Queue = queue.Queue(maxsize=16)

        if args.record:
            writer = Float32WavWriter(args.record)
        if args.pipewire_source:
            capture_command = [
                "pw-record", "--target", source, "--format", "f32",
                "--rate", str(SAMPLE_RATE), "--channels", str(CHANNELS),
                "--latency", f"{args.capture_latency_ms}ms", "-",
            ]
        else:
            capture_command = [
                "parec", f"--device={source}", "--format=float32le", f"--rate={SAMPLE_RATE}",
                f"--channels={CHANNELS}", f"--latency-msec={args.capture_latency_ms}",
            ]
        capture = subprocess.Popen(
            capture_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if args.pipewire_source:
            # Keep direct playback clocking the Bluetooth source until capture
            # has observed a genuinely non-silent window. A fixed delay races
            # iOS stream startup and can strand bluez_input in digital silence.
            first_audio = None
            deadline = None if args.start_timeout <= 0 else time.monotonic() + args.start_timeout
            while (not stop.is_set() and not _stop.is_set()
                   and (deadline is None or time.monotonic() < deadline)):
                candidate = read_pipe_chunk(capture, chunk_bytes)
                if candidate is None:
                    break
                samples = np.frombuffer(candidate, dtype="<f4")
                if samples.size and float(np.max(np.abs(samples))) > 0.001:
                    first_audio = candidate
                    break
            if first_audio is None:
                raise RuntimeError(
                    "PipeWire source stayed silent for "
                    f"{args.start_timeout:g}s (--start-timeout 0 waits for playback)")
            in_q.put(first_audio)
            direct_links(source, sink, connect=False)
            direct_unlinked = True
            print("[relay] non-silent capture confirmed; direct Bluetooth links removed",
                  flush=True)
        player = subprocess.Popen(
            ["pacat", "--playback", f"--device={sink}", "--format=float32le",
             f"--rate={SAMPLE_RATE}", f"--channels={CHANNELS}",
             f"--latency-msec={args.playback_latency_ms}"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        threads = [
            threading.Thread(target=reader_loop,
                             args=(capture, in_q, chunk_bytes, stats, stats_lock, stop),
                             daemon=True),
            threading.Thread(target=processor_loop,
                             args=(args, in_q, out_q, stats, stats_lock, stop), daemon=True),
            threading.Thread(target=player_loop,
                             args=(args, out_q, player, writer, stats, stats_lock, stop),
                             daemon=True),
            *( [] if args.pipewire_source else [
                threading.Thread(target=route_guard_loop,
                                 args=(args.null_sink, stats, stats_lock, stop), daemon=True)
            ] ),
            *( [threading.Thread(target=source_watch_loop, args=(source, stop), daemon=True)]
               if args.pipewire_source else [] ),
            threading.Thread(target=status_loop,
                             args=(args, stats, stats_lock, stop), daemon=True),
        ]
        for thread in threads:
            thread.start()

        started = time.monotonic()
        last_chunks = 0
        stall_deadline: float | None = None
        while not stop.is_set() and not _stop.is_set():
            if args.duration and time.monotonic() - started >= args.duration:
                break
            if not any(thread.is_alive() for thread in threads[1:3]):
                break
            with stats_lock:
                now_chunks = int(stats.get("chunks", 0))
            if now_chunks != last_chunks:
                last_chunks = now_chunks
                stall_deadline = None
            elif now_chunks > 0:
                if stall_deadline is None:
                    stall_deadline = time.monotonic() + args.stall_timeout
                elif time.monotonic() >= stall_deadline:
                    print(f"[relay] stalled {args.stall_timeout:g}s at chunk {now_chunks}; "
                          f"tearing down to re-arm", flush=True)
                    break
            time.sleep(0.2)
    except Exception as exc:  # noqa: BLE001
        print(f"[relay] session error: {exc}", file=sys.stderr)
        with stats_lock:
            stats["error"] = str(exc)
    finally:
        stop.set()
        for proc in (capture, player):
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:  # noqa: BLE001
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
        if direct_unlinked:
            try:
                direct_links(source, sink, connect=True)
            except Exception:  # noqa: BLE001
                pass
        if module_id is not None:
            try:
                pctl("unload-module", str(module_id))
                if previous_default:
                    pctl("set-default-sink", previous_default)
            except Exception:  # noqa: BLE001
                pass
        if writer is not None:
            writer.close()



if __name__ == "__main__":
    sys.exit(main())
