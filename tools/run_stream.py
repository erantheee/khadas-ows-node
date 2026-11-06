#!/usr/bin/env python3
"""Realtime-paced PCM streaming probe for khadas-ows-node.

This is an **engineering evidence tool**, not a listening-quality judge. It streams
stereo 44.1 kHz float32 PCM to the C++ node over the project's TCP protocol at a
realtime pace, records round-trip latency / deadline misses / reconnects, and writes
a machine-readable JSON summary whose ``pass`` field is derived only from explicit,
configured timing criteria. It never claims anything about audio quality.

Protocol (see src/audio_core.cpp, python/khadas_ows/worker.py):
    request : <IIQIHHi> header + frames*channels little-endian float32 samples
    reply   : the same header echoed back + frames*channels float32 samples

Realtime semantics: by default the probe is strictly synchronous -- one request is in
flight at a time. A chunk therefore "keeps up" only if its round-trip time is shorter
than the chunk duration; otherwise a strictly synchronous consumer cannot sustain
realtime. ``deadline_misses`` counts chunks whose RTT exceeded ``--chunk-ms``.
When the probe falls behind schedule it catches up for at most ``--max-catchup-chunks``
chunks and then re-anchors its monotonic schedule instead of bursting without bound
(``pacing_resets``).

Exit codes:
    0  all configured criteria met (PASS)
    1  a criterion failed (FAIL), or the run was interrupted
    2  precondition failure (server unreachable, bad input file, unusable WAV/args)

Usage (local passthrough verification on this Mac, server already running):
    .venv/bin/python tools/run_stream.py --mode passthrough --seconds 5 \
        --json /tmp/run-stream.json --output /tmp/passthrough.wav
"""
from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import signal
import socket
import struct
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<IIQIHHi")
MAGIC = 0x4B4F5753
SCHEMA = 1
SAMPLE_RATE = 44100
CHANNELS = 2
REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "tools/run_stream.py"

_STOP = False


def _on_signal(signum, frame):  # noqa: ARG001 - signal handler signature
    global _STOP
    _STOP = True
    print(f"[run_stream] received signal {signum}; finalising evidence...", flush=True)


class PreconditionError(RuntimeError):
    """Bad input or unreachable server: no measurement was possible."""


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- HTTP

def http_request(host: str, port: int, method: str, path: str, body: str | None = None,
                 timeout: float = 3.0) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        payload = body.encode() if body is not None else None
        headers = {}
        if payload is not None:
            headers["Content-Type"] = "text/plain"
            headers["Content-Length"] = str(len(payload))
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def get_status(host: str, port: int, timeout: float = 3.0) -> dict:
    try:
        code, payload = http_request(host, port, "GET", "/v1/status", timeout=timeout)
    except OSError as exc:
        raise PreconditionError(
            f"cannot reach node status endpoint http://{host}:{port}/v1/status ({exc}); "
            "is build/khadas-ows-node running and reachable? no evidence was recorded"
        ) from exc
    if code != 200:
        raise PreconditionError(f"GET /v1/status returned HTTP {code}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PreconditionError(f"GET /v1/status returned non-JSON: {payload[:200]!r}") from exc


def set_mode(host: str, port: int, mode: str, timeout: float = 3.0) -> dict:
    try:
        code, payload = http_request(host, port, "POST", "/v1/mode", body=mode, timeout=timeout)
    except OSError as exc:
        raise PreconditionError(
            f"cannot reach node http://{host}:{port}/v1/mode ({exc}); no evidence was recorded"
        ) from exc
    if code != 200:
        raise PreconditionError(f"POST /v1/mode {mode!r} returned HTTP {code}: {payload[:200]!r}")
    return json.loads(payload)


# ------------------------------------------------------------------------- WAV I/O

def read_wav(path: Path) -> np.ndarray:
    """Read an int16 stereo 44.1 kHz WAV into float32 (frames, 2) in [-1, 1]."""
    if not path.is_file():
        raise PreconditionError(f"input WAV not found: {path}")
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if channels != CHANNELS or rate != SAMPLE_RATE:
        raise PreconditionError(
            f"{path}: expected {CHANNELS} channels @ {SAMPLE_RATE} Hz, got {channels} @ {rate} Hz"
        )
    if width != 2:
        raise PreconditionError(f"{path}: expected 16-bit PCM WAV, got {width * 8}-bit")
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, CHANNELS).astype(np.float32) / 32768.0
    return pcm


def write_wav(path: Path, pcm_float: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(pcm_float, -1.0, 1.0)
    pcm_int16 = (clipped * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm_int16.tobytes())


def generate_fixture(seconds: float) -> np.ndarray:
    """Import the repository fixture generator (tools/make_fixture.py) and synthesize."""
    fixture_path = REPO_ROOT / "tools" / "make_fixture.py"
    if not fixture_path.is_file():
        raise PreconditionError(
            f"--input not given and fixture generator missing at {fixture_path}; "
            "pass --input <wav> instead"
        )
    spec = importlib.util.spec_from_file_location("_khadas_make_fixture", fixture_path)
    if spec is None or spec.loader is None:
        raise PreconditionError(f"cannot import fixture generator {fixture_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pcm_int16 = module.synthesize(seconds)  # deterministic (n, 2) int16
    return pcm_int16.astype(np.float32) / 32768.0


def build_source(input_path: Path | None, duration_s: float) -> tuple[np.ndarray, dict]:
    if input_path is not None:
        pcm = read_wav(input_path)
        source = {"kind": "file", "path": str(input_path)}
    else:
        fixture_seconds = max(10.0, duration_s + 1.0)
        pcm = generate_fixture(fixture_seconds)
        source = {"kind": "generated_fixture", "path": "tools/make_fixture.py",
                  "generated_seconds": fixture_seconds}
    if pcm.shape[0] == 0:
        raise PreconditionError("source audio is empty")
    total_frames = int(round(duration_s * SAMPLE_RATE))
    repeats = int(np.ceil(total_frames / pcm.shape[0]))
    used = np.tile(pcm, (repeats, 1))[:total_frames]
    source.update({"source_frames": int(pcm.shape[0]), "looped": repeats > 1,
                   "repeats": repeats, "frames_streamed_target": total_frames})
    return used, source


# ------------------------------------------------------------------------ socket IO

def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("peer closed the TCP stream mid-packet")
        data.extend(chunk)
    return bytes(data)


class PcmLink:
    """One TCP connection to the node's audio port with explicit reconnect accounting."""

    def __init__(self, host: str, port: int, connect_timeout_s: float, read_timeout_s: float):
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.sock: socket.socket | None = None
        self.bytes_sent = 0
        self.connects = 0

    def connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
        sock.settimeout(self.read_timeout_s)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self.sock = sock
        self.connects += 1

    def close(self, abrupt: bool = False) -> None:
        if self.sock is None:
            return
        if abrupt:
            # SO_LINGER {on=1, seconds=0} makes close() emit RST instead of FIN: this is
            # the "network vanished / process killed" case the node must survive.
            try:
                self.sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
            except OSError:
                pass
        try:
            self.sock.close()
        except OSError:
            pass
        self.sock = None

    def exchange(self, header: bytes, payload: bytes, frames: int) -> tuple[bytes, bytes]:
        if self.sock is None:
            raise ConnectionError("no TCP connection")
        self.sock.sendall(header)
        self.sock.sendall(payload)
        self.bytes_sent += len(header) + len(payload)
        reply_header = recv_exact(self.sock, HEADER.size)
        reply_payload = recv_exact(self.sock, frames * CHANNELS * 4)
        return reply_header, reply_payload


# -------------------------------------------------------------------------- probe

def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def run_probe(args: argparse.Namespace) -> dict:
    global _STOP
    started_at = datetime.now(timezone.utc).isoformat()
    chunk_frames = max(1, int(round(args.chunk_ms / 1000.0 * SAMPLE_RATE)))
    chunk_period_s = chunk_frames / SAMPLE_RATE
    chunk_ms_actual = chunk_period_s * 1000.0

    # --- preconditions: server + input must be real before we claim any measurement
    status_start = get_status(args.host, args.http_port)
    source_pcm, source_info = build_source(args.input, args.seconds)
    total_frames = source_pcm.shape[0]
    n_chunks = total_frames // chunk_frames
    if n_chunks < 1:
        raise PreconditionError(
            f"duration too short for chunk size: {total_frames} frames vs {chunk_frames}"
        )
    target_frames = n_chunks * chunk_frames
    audio_seconds_target = target_frames / SAMPLE_RATE

    if args.mode is not None:
        set_mode(args.host, args.http_port, args.mode)
        status_after_mode = get_status(args.host, args.http_port)
    else:
        status_after_mode = status_start
    requested_mode = status_after_mode.get("requested_mode", args.mode or "passthrough")

    log(f"[run_stream] target {args.host} audio={args.audio_port} http={args.http_port} "
        f"mode={requested_mode!r}")
    log(f"[run_stream] source={source_info['kind']} planned_audio={audio_seconds_target:.2f}s "
        f"chunk={chunk_ms_actual:.1f}ms frames={chunk_frames} chunks={n_chunks}")

    link = PcmLink(args.host, args.audio_port,
                   args.connect_timeout_ms / 1000.0, args.read_timeout_ms / 1000.0)
    try:
        link.connect()
    except OSError as exc:
        raise PreconditionError(
            f"cannot connect to audio tcp://{args.host}:{args.audio_port} ({exc}); "
            "no evidence was recorded"
        ) from exc

    output_wav = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_wav = wave.open(str(args.output), "wb")
        output_wav.setnchannels(CHANNELS)
        output_wav.setsampwidth(2)
        output_wav.setframerate(SAMPLE_RATE)

    rtts: list[float] = []
    errors: list[str] = []
    reconnects = {"forced": 0, "auto": 0}
    recoveries = 0
    retried_chunks = 0
    pending_recovery = False
    pacing_resets = 0
    send_late = 0
    deadline_misses = 0
    chunks_ok = 0
    failed_attempts = 0
    protocol_errors = 0
    non_finite = 0
    bit_exact_checked = 0
    bit_exact_mismatches = 0
    output_frames = 0
    seq = 0
    fatal_reason: str | None = None
    last_forced_at = time.perf_counter()
    last_progress_at = time.perf_counter()
    progress_handle = None
    if args.progress_jsonl is not None:
        args.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
        progress_handle = args.progress_jsonl.open("w", encoding="utf-8")

    def write_progress(force: bool = False) -> None:
        nonlocal last_progress_at
        if progress_handle is None:
            return
        now = time.perf_counter()
        if not force and now - last_progress_at < 1.0:
            return
        last_progress_at = now
        record = {
            "wall_unix": time.time(),
            "elapsed_s": round(now - started_wall_perf, 3),
            "chunks_ok": chunks_ok,
            "chunks_target": n_chunks,
            "last_rtt_ms": round(rtts[-1], 3) if rtts else None,
            "reconnects": dict(reconnects),
            "deadline_misses": deadline_misses,
            "failed_attempts": failed_attempts,
        }
        progress_handle.write(json.dumps(record) + "\n")
        progress_handle.flush()

    started_wall_perf = time.perf_counter()
    # Schedule: target_send_time is anchored to a monotonic clock; catch-up is bounded.
    target_send = started_wall_perf
    try:
        for chunk_index in range(n_chunks):
            if _STOP:
                fatal_reason = "interrupted_by_signal"
                break
            # ---- pacing (bounded catch-up, no unbounded burst)
            now = time.perf_counter()
            if now < target_send:
                time.sleep(target_send - now)
            send_at = time.perf_counter()
            if send_at - target_send > chunk_period_s:
                send_late += 1

            # ---- forced abrupt reconnect (network failure simulation)
            if args.reconnect_every > 0 and (send_at - last_forced_at) >= args.reconnect_every:
                link.close(abrupt=True)
                reconnects["forced"] += 1
                last_forced_at = send_at
                pending_recovery = True
                try:
                    link.connect()
                except OSError as exc:
                    fatal_reason = f"reconnect_failed_after_forced_disconnect: {exc}"
                    break

            frames = chunk_frames
            block = source_pcm[chunk_index * frames:(chunk_index + 1) * frames].astype("<f4")
            header = HEADER.pack(MAGIC, SCHEMA, seq, SAMPLE_RATE, CHANNELS, 0, frames)
            payload = block.tobytes()

            reply_header = reply_payload = None
            while True:
                try:
                    t_send = time.perf_counter()
                    reply_header, reply_payload = link.exchange(header, payload, frames)
                    rtt = (time.perf_counter() - t_send) * 1000.0
                    break
                except (OSError, ConnectionError) as exc:
                    failed_attempts += 1
                    errors.append(f"chunk {chunk_index} attempt: {type(exc).__name__}: {exc}")
                    del errors[:-40]  # keep evidence list bounded
                    link.close()
                    # auto-reconnect loop until the configured deadline
                    deadline = time.perf_counter() + args.auto_reconnect_timeout_s
                    connected = False
                    while time.perf_counter() < deadline and not _STOP:
                        try:
                            link.connect()
                            connected = True
                            reconnects["auto"] += 1
                            pending_recovery = True
                            break
                        except OSError:
                            time.sleep(0.1)
                    if not connected:
                        fatal_reason = (
                            f"unrecoverable_stream_loss: could not reconnect to "
                            f"{args.host}:{args.audio_port} within "
                            f"{args.auto_reconnect_timeout_s:.1f}s"
                        )
                        break
                    retried_chunks += 1
                    if retried_chunks > args.max_retried_chunks:
                        fatal_reason = (
                            f"unrecoverable_stream_loss: {retried_chunks} chunks needed "
                            "retry after TCP errors"
                        )
                        break
                if fatal_reason is not None:
                    break
            if fatal_reason is not None:
                break

            # ---- validate reply framing (protocol correctness is part of the evidence)
            magic, schema, r_seq, sr, ch, _reserved, r_frames = HEADER.unpack(reply_header)
            expected_bytes = frames * CHANNELS * 4
            protocol_error = None
            if magic != MAGIC:
                protocol_error = f"bad magic {magic:#x}"
            elif schema != SCHEMA:
                protocol_error = f"bad schema {schema}"
            elif r_seq != seq:
                protocol_error = f"sequence echo {r_seq} != {seq}"
            elif sr != SAMPLE_RATE or ch != CHANNELS:
                protocol_error = f"format echo {sr}Hz/{ch}ch"
            elif r_frames != frames:
                protocol_error = f"frame count echo {r_frames} != {frames}"
            elif len(reply_payload) != expected_bytes:
                protocol_error = f"payload {len(reply_payload)} != {expected_bytes}"
            if protocol_error is not None:
                protocol_errors += 1
                errors.append(f"chunk {chunk_index}: protocol_error: {protocol_error}")
                del errors[:-40]
                link.close()
                try:
                    link.connect()
                except OSError as exc:
                    fatal_reason = f"unrecoverable_stream_loss: cannot resync after protocol error: {exc}"
                    break
                pending_recovery = True
                target_send += chunk_period_s
                continue

            reply = np.frombuffer(reply_payload, dtype="<f4")
            sent = np.frombuffer(payload, dtype="<f4")
            if not np.isfinite(reply).all():
                non_finite += 1
            if args.mode in (None, "passthrough"):
                bit_exact_checked += 1
                if reply.tobytes() != sent.tobytes():
                    bit_exact_mismatches += 1
            if output_wav is not None:
                merged = reply.reshape(-1, CHANNELS)
                output_wav.writeframes(
                    (np.clip(merged, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()
                )
                output_frames += frames

            chunks_ok += 1
            rtts.append(rtt)
            if pending_recovery:
                recoveries += 1
                pending_recovery = False
            if rtt > chunk_ms_actual:
                deadline_misses += 1
            seq += 1

            # ---- next schedule slot, bounded catch-up
            target_send += chunk_period_s
            behind = time.perf_counter() - target_send
            if behind > args.max_catchup_chunks * chunk_period_s:
                target_send = time.perf_counter()
                pacing_resets += 1
            write_progress()
    finally:
        try:
            write_progress(force=True)  # final liveness record before the handle closes
        finally:
            link.close()
            if output_wav is not None:
                output_wav.close()
            if progress_handle is not None:
                progress_handle.close()

    wall_seconds = time.perf_counter() - started_wall_perf
    status_end = None
    status_error = None
    try:
        status_end = get_status(args.host, args.http_port)
    except PreconditionError as exc:  # server died during the run: that is evidence
        status_error = str(exc)

    # ---------------------------------------------------------------- criteria
    misses_total = deadline_misses + retried_chunks
    miss_ratio = misses_total / n_chunks if n_chunks else 1.0
    failures: list[str] = []
    warnings: list[str] = []
    if fatal_reason is not None:
        failures.append(fatal_reason)
    if chunks_ok == 0:
        failures.append("no successful chunks were completed")
    if protocol_errors:
        failures.append(f"{protocol_errors} protocol error(s) in replies")
    if non_finite:
        failures.append(f"{non_finite} reply chunk(s) contained non-finite samples")
    if len(rtts) < 20:
        warnings.append(f"short run: latency percentiles from only {len(rtts)} samples")
    sorted_rtts = sorted(rtts)
    p50 = percentile(sorted_rtts, 0.50)
    p95 = percentile(sorted_rtts, 0.95)
    p99 = percentile(sorted_rtts, 0.99)
    rtt_max = sorted_rtts[-1] if sorted_rtts else 0.0
    if rtts and p95 > args.max_p95_ms:
        failures.append(f"p95 latency {p95:.2f}ms > --max-p95-ms {args.max_p95_ms:.2f}ms")
    if miss_ratio > args.max_miss_ratio:
        failures.append(
            f"deadline miss ratio {miss_ratio:.4f} > --max-miss-ratio {args.max_miss_ratio:.4f} "
            f"({misses_total}/{n_chunks} chunks)"
        )
    if requested_mode not in (None, "passthrough") and status_end is not None:
        actual = status_end.get("actual_mode")
        if actual != requested_mode and not args.allow_fallback:
            failures.append(
                f"requested mode {requested_mode!r} not engaged: actual_mode={actual!r} "
                f"fallback_reason={status_end.get('fallback_reason', '')!r} "
                "(pass --allow-fallback only if fallback is the expected engineering result)"
            )
    if args.require_bit_exact and bit_exact_mismatches:
        failures.append(f"{bit_exact_mismatches}/{bit_exact_checked} passthrough chunks not bit-exact")
    if status_error is not None:
        warnings.append(f"final status poll failed: {status_error}")
    if chunks_ok < n_chunks:
        failures.append(f"only {chunks_ok}/{n_chunks} chunks were streamed")
    if _STOP:
        failures.append("probe interrupted before completion")

    fallbacks_delta = None
    rejected_delta = None
    for key, delta_key in (("fallbacks", "fallbacks_delta"), ("rejected", "rejected_delta")):
        try:
            start_value = int(status_start.get(key, 0))
            end_value = int(status_end.get(key, 0))
            value = end_value - start_value
        except (TypeError, ValueError):
            value = None
        if key == "fallbacks":
            fallbacks_delta = value
        else:
            rejected_delta = value

    audio_seconds_streamed = chunks_ok * chunk_frames / SAMPLE_RATE
    rtf_estimate = ((sum(rtts) / 1000.0) / audio_seconds_streamed
                    if audio_seconds_streamed > 0 else None)
    realtime_ratio = audio_seconds_streamed / wall_seconds if wall_seconds > 0 else None
    if audio_seconds_streamed < audio_seconds_target - 1e-9:
        warnings.append(
            f"streamed {audio_seconds_streamed:.2f}s of audio < planned "
            f"{audio_seconds_target:.2f}s (interrupted/short run)"
        )
    if realtime_ratio is not None and realtime_ratio < 0.95:
        warnings.append(
            f"realtime ratio {realtime_ratio:.3f} < 0.95: wall clock exceeded streamed audio "
            "duration (strictly synchronous protocol could not sustain realtime)"
        )
    if bit_exact_mismatches:
        warnings.append(
            f"passthrough reply differed bit-exactly from request in "
            f"{bit_exact_mismatches}/{bit_exact_checked} chunks "
            "(expected if the policy/media-gain stage is active)"
        )
    if fallbacks_delta:
        warnings.append(f"node fallback counter increased by {fallbacks_delta} during the run")

    verdict = "PASS" if not failures else "FAIL"
    report = {
        "schema": 1,
        "tool": TOOL,
        "started_at": started_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "audio_port": args.audio_port,
        "http_port": args.http_port,
        "requested_mode": requested_mode,
        "source": source_info,
        "chunk_ms": round(chunk_ms_actual, 4),
        "chunk_frames": chunk_frames,
        "duration_requested_s": args.seconds,
        "audio_seconds_target": round(audio_seconds_target, 4),
        "audio_seconds_streamed": round(audio_seconds_streamed, 4),
        "wall_seconds": round(wall_seconds, 4),
        "chunks_target": n_chunks,
        "chunks_ok": chunks_ok,
        "failed_attempts": failed_attempts,
        "retried_chunks": retried_chunks,
        "protocol_errors": protocol_errors,
        "non_finite_chunks": non_finite,
        "latency_ms": {
            "p50": round(p50, 4),
            "p95": round(p95, 4),
            "p99": round(p99, 4),
            "max": round(rtt_max, 4),
            "mean": round(sum(rtts) / len(rtts), 4) if rtts else None,
            "samples": len(rtts),
        },
        "deadline_misses": deadline_misses,
        "miss_ratio": round(miss_ratio, 6),
        "send_late_chunks": send_late,
        "pacing_resets": pacing_resets,
        "reconnects": {"forced": reconnects["forced"], "auto": reconnects["auto"]},
        "recoveries": recoveries,
        "bytes_sent": link.bytes_sent,
        "rtf_estimate": round(rtf_estimate, 6) if rtf_estimate is not None else None,
        "rtf_estimate_note": "client-side estimate: sum(RTT-s of successful chunks) / audio seconds; "
                             "includes TCP + server scheduling, not pure inference time",
        "realtime_ratio": round(realtime_ratio, 6) if realtime_ratio is not None else None,
        "passthrough_bit_exact": {
            "checked": bit_exact_checked,
            "mismatches": bit_exact_mismatches,
            "all_exact": bit_exact_mismatches == 0,
        },
        "status_start": status_start,
        "status_after_mode": status_after_mode,
        "status_end": status_end,
        "status_poll_error": status_error,
        "fallbacks_delta": fallbacks_delta,
        "rejected_delta": rejected_delta,
        "criteria": {
            "max_p95_ms": args.max_p95_ms,
            "max_miss_ratio": args.max_miss_ratio,
            "max_catchup_chunks": args.max_catchup_chunks,
            "require_bit_exact": bool(args.require_bit_exact),
            "allow_fallback": bool(args.allow_fallback),
            "reconnect_every_s": args.reconnect_every,
        },
        "output_wav": str(args.output) if args.output else None,
        "output_frames": output_frames,
        "errors": errors,
        "failures": failures,
        "warnings": warnings,
        "pass": not failures,
        "verdict": verdict,
        "listening_quality": "NOT ASSESSED by this tool; quality conclusions require human "
                             "blind listening (tools/listening_ab.py).",
    }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- human summary
    log("")
    log("=== run_stream summary =====================================================")
    log(f"target      : {args.host} audio={args.audio_port} http={args.http_port} "
        f"mode={requested_mode}")
    log(f"audio       : {audio_seconds_streamed:.2f}s/{audio_seconds_target:.2f}s streamed "
        f"in {wall_seconds:.2f}s wall "
        f"(realtime_ratio={realtime_ratio:.3f})")
    log(f"chunks      : ok={chunks_ok}/{n_chunks} failed_attempts={failed_attempts} "
        f"protocol_errors={protocol_errors}")
    log(f"latency ms  : p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} max={rtt_max:.2f}")
    log(f"deadlines   : misses={deadline_misses} miss_ratio={miss_ratio:.4f} "
        f"(rtt > {chunk_ms_actual:.1f}ms), send_late={send_late}, pacing_resets={pacing_resets}")
    log(f"reconnects  : forced={reconnects['forced']} auto={reconnects['auto']} "
        f"recoveries={recoveries} retried_chunks={retried_chunks}")
    if status_end is not None:
        log(f"status      : requested={status_end.get('requested_mode')} "
            f"actual={status_end.get('actual_mode')} "
            f"fallback_reason={status_end.get('fallback_reason')!r} "
            f"fallbacks_delta={fallbacks_delta} rejected_delta={rejected_delta}")
    if bit_exact_checked:
        log(f"passthrough : bit-exact {bit_exact_checked - bit_exact_mismatches}/{bit_exact_checked} chunks")
    for warning in warnings:
        log(f"WARNING     : {warning}")
    for failure in failures:
        log(f"FAILURE     : {failure}")
    log(f"RESULT      : {verdict}")
    log("NOTE        : signals/timing only; audio quality is a human listening decision.")
    if args.json is not None:
        log(f"json        : {args.json}")
    if args.output:
        log(f"output wav  : {args.output} ({output_frames} frames)")
    log("=" * 78)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime-paced PCM streaming probe for the khadas-ows-node TCP protocol. "
                    "Engineering evidence only: it does not judge audio quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=None,
                        help="input 16-bit stereo 44.1 kHz WAV; omitted => generate the repo fixture")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--audio-port", type=int, default=9100)
    parser.add_argument("--http-port", type=int, default=8787)
    parser.add_argument("--mode", choices=("passthrough", "cpu", "npu"), default=None,
                        help="requested node mode via POST /v1/mode; omitted => passthrough")
    parser.add_argument("--seconds", "--duration", dest="seconds", type=float, default=10.0,
                        help="seconds of audio to stream (input is looped if shorter)")
    parser.add_argument("--chunk-ms", type=float, default=20.0,
                        help="audio chunk duration; one synchronous request per chunk")
    parser.add_argument("--output", type=Path, default=None,
                        help="write received PCM to this WAV (successful chunks only)")
    parser.add_argument("--json", type=Path, default=None, help="write the JSON summary here")
    parser.add_argument("--progress-jsonl", type=Path, default=None,
                        help="append machine-readable progress lines (used by tools/soak_edge2.py)")
    parser.add_argument("--reconnect-every", type=float, default=0.0,
                        help="every N seconds abruptly RST-close and reconnect (0 disables)")
    parser.add_argument("--read-timeout-ms", type=float, default=5000.0,
                        help="per-chunk reply timeout before treating the reply as lost")
    parser.add_argument("--connect-timeout-ms", type=float, default=3000.0)
    parser.add_argument("--auto-reconnect-timeout-s", type=float, default=10.0,
                        help="keep retrying a broken TCP stream for this long before failing")
    parser.add_argument("--max-retried-chunks", type=int, default=3,
                        help="more retried chunks than this => unrecoverable stream loss")
    parser.add_argument("--max-p95-ms", type=float, default=None,
                        help="fail if p95 RTT exceeds this (default: one chunk duration)")
    parser.add_argument("--max-miss-ratio", type=float, default=0.02,
                        help="fail if deadline misses + failed attempts exceed this ratio")
    parser.add_argument("--max-catchup-chunks", type=int, default=2,
                        help="bounded catch-up: re-anchor schedule if this far behind")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="do not fail when a requested cpu/npu mode is not engaged")
    parser.add_argument("--require-bit-exact", action="store_true",
                        help="fail if passthrough replies are not bit-identical to requests")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seconds <= 0:
        print("[run_stream] --seconds must be > 0", file=sys.stderr)
        return 2
    if args.chunk_ms <= 0:
        print("[run_stream] --chunk-ms must be > 0", file=sys.stderr)
        return 2
    if args.mode is None:
        # keep whatever the node has requested; passthrough is the safe default for local runs
        args.mode = "passthrough"
    if args.max_p95_ms is None:
        args.max_p95_ms = args.chunk_ms
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        report = run_probe(args)
    except PreconditionError as exc:
        print(f"[run_stream] FAIL (precondition): {exc}", file=sys.stderr)
        print("[run_stream] no timing evidence was recorded; this is not a pass.", file=sys.stderr)
        return 2
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
