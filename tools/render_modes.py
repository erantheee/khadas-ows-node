#!/usr/bin/env python3
"""Render one real-music excerpt through every node mode, for listening and evidence.

This is the "play a song and listen" workflow:

    # 1) pick an excerpt of a real song (any 44.1 kHz stereo WAV)
    # 2) render passthrough / CPU / NPU renditions of the same excerpt
    .venv/bin/python tools/render_modes.py --input ~/Music/song.wav --seconds 45 --start 30

    # 3) blind-listen the three renditions (interactive)
    .venv/bin/python tools/listening_ab.py --render out/renditions --out out/blind-<ts>

`passthrough` is the reference excerpt exactly as the node would forward it, `cpu` is
rendered through the local PyTorch worker, `npu` through the Edge2 NPU worker over the
network. The tool is an engineering harness: it does not judge quality and does not
loudness-match (listening_ab.py does that before playback).

Every rendition is written as 44.1 kHz / 2 ch / 16-bit WAV with identical length so
listening_ab.py can loudness-match and blind them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 44100
CHANNELS = 2


def log(message: str) -> None:
    print(f"[render] {message}", flush=True)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def normalize_excerpt(input_path: Path, output: Path, start_s: float, seconds: float) -> None:
    """Cut/resample to 44.1 kHz stereo 16-bit. passthrough must be exactly this file."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found; install it (brew install ffmpeg) to cut/resample")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if start_s > 0:
        command += ["-ss", f"{start_s:.3f}"]
    command += ["-t", f"{seconds:.3f}", "-i", str(input_path),
                "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-sample_fmt", "s16", str(output)]
    result = run(command)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {result.stderr.strip()}")
    log(f"reference excerpt: {output} ({wav_duration(output):.2f}s)")


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def wav_rms(path: Path) -> float:
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return float(np.sqrt((pcm ** 2).mean()))


def http_json(url: str, data: bytes | None = None, timeout: float = 3.0) -> dict:
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_http(host: str, port: int, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            return http_json(f"http://{host}:{port}/v1/status")
        except (urllib.error.URLError, OSError) as exc:  # noqa: PERF203
            last_error = exc
            time.sleep(0.1)
    raise SystemExit(f"node on {host}:{port} not ready: {last_error}")


def port_free(port: int) -> bool:
    probe = socket.socket()
    probe.settimeout(0.3)
    try:
        return probe.connect_ex(("127.0.0.1", port)) != 0
    finally:
        probe.close()


def wait_worker(port: int, timeout_s: float = 120.0, log_path: Path | None = None) -> None:
    """Wait until the worker listens. Torch import + model load + graph warmup takes seconds,
    so connecting before this point silently falls back to passthrough for every packet."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        probe = socket.socket()
        probe.settimeout(0.5)
        try:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        finally:
            probe.close()
        time.sleep(0.2)
    tail = ""
    if log_path is not None and log_path.exists():
        tail = "\n" + "\n".join(log_path.read_text(errors="replace").splitlines()[-8:])
    raise SystemExit(f"worker on 127.0.0.1:{port} not ready within {timeout_s:.0f}s{tail}")


def render_local_cpu(excerpt: Path, output: Path, chunk_frames: int) -> dict:
    node_binary = ROOT / "build" / "khadas-ows-node"
    if not node_binary.exists():
        raise SystemExit(f"node binary missing: {node_binary} (cmake -S . -B build && cmake --build build)")
    checkpoint = ROOT / "models" / "candidate-v0.pt"
    if not checkpoint.exists():
        raise SystemExit("models/candidate-v0.pt missing; CPU rendition needs the checkpoint")
    for port in (38787, 39100, 19201):
        if not port_free(port):
            raise SystemExit(f"port {port} already in use; stop the leftover process first")

    env = {
        "KHADAS_AUDIO_PORT": "39100",
        "KHADAS_HTTP_PORT": "38787",
        "KHADAS_CPU_PORT": "19201",
        "KHADAS_NPU_PORT": "19202",
        "KHADAS_CLIENT_TIMEOUT_MS": "15000",
        "KHADAS_WORKER_TIMEOUT_MS": "8000",
        "PYTHONPATH": str(ROOT / "python"),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
    }
    node_log = output.parent / "cpu-node.log"
    worker_log = output.parent / "cpu-worker.log"
    node = subprocess.Popen([str(node_binary)], stdout=open(node_log, "w"), stderr=subprocess.STDOUT, env=env)
    worker = subprocess.Popen(
        [sys.executable, "-m", "khadas_ows.worker", "--mode", "cpu", "--model", str(checkpoint),
         "--port", "19201"],
        stdout=open(worker_log, "w"), stderr=subprocess.STDOUT, env=env)
    try:
        wait_http("127.0.0.1", 38787)
        wait_worker(19201, log_path=worker_log)
        http_json("http://127.0.0.1:38787/v1/mode", data=b"cpu")
        started = time.time()
        result = run([sys.executable, str(ROOT / "tools" / "pcm_client.py"), str(excerpt), str(output),
                      "--host", "127.0.0.1", "--port", "39100", "--chunk", str(chunk_frames)])
        if result.returncode != 0:
            raise SystemExit(f"CPU pcm_client failed: {result.stderr.strip()}")
        status = http_json("http://127.0.0.1:38787/v1/status")
        elapsed = time.time() - started
        if status.get("actual_mode") != "cpu":
            raise SystemExit(
                f"CPU mode did not engage (actual={status.get('actual_mode')}, "
                f"reason={status.get('fallback_reason')}, fallbacks={status.get('fallbacks')}); "
                f"the rendered file is not CPU output. Worker log: {worker_log}")
    finally:
        for process in (worker, node):
            process.send_signal(signal.SIGTERM)
        for process in (worker, node):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    return {"actual_mode": status.get("actual_mode"), "fallbacks": status.get("fallbacks"),
            "wall_s": round(elapsed, 2), "server_rtf": status.get("last_rtf"),
            "node_log": str(node_log), "worker_log": str(worker_log)}


def render_remote_npu(excerpt: Path, output: Path, board: str, audio_port: int, http_port: int,
                      chunk_frames: int) -> dict:
    status = wait_http(board, http_port, timeout_s=8.0)
    if status.get("actual_mode") != "npu" and "npu" not in str(status.get("requested_mode")):
        log(f"board currently requested={status.get('requested_mode')}; asking for npu")
    http_json(f"http://{board}:{http_port}/v1/mode", data=b"npu")
    started = time.time()
    result = run([sys.executable, str(ROOT / "tools" / "pcm_client.py"), str(excerpt), str(output),
                  "--host", board, "--port", str(audio_port), "--chunk", str(chunk_frames)])
    if result.returncode != 0:
        raise SystemExit(f"NPU pcm_client failed: {result.stderr.strip()}")
    elapsed = time.time() - started
    after = wait_http(board, http_port)
    if after.get("actual_mode") != "npu":
        raise SystemExit(f"board did not run npu mode (actual={after.get('actual_mode')}, "
                         f"reason={after.get('fallback_reason')}); the rendered file is not NPU output")
    return {"actual_mode": after.get("actual_mode"), "fallbacks": after.get("fallbacks"),
            "server_rtf": after.get("last_rtf"), "model_hash": after.get("model_hash"),
            "wall_s": round(elapsed, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="source audio (any format ffmpeg reads)")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "listening-material" / "real-music")
    parser.add_argument("--start", type=float, default=30.0, help="excerpt start seconds (default 30)")
    parser.add_argument("--seconds", type=float, default=45.0, help="excerpt length seconds (default 45)")
    parser.add_argument("--modes", default="passthrough,cpu,npu",
                        help="subset of passthrough,cpu,npu (default all)")
    parser.add_argument("--board", default="172.16.20.18", help="Edge2 host for the npu rendition")
    parser.add_argument("--audio-port", type=int, default=9100)
    parser.add_argument("--http-port", type=int, default=8787)
    parser.add_argument("--chunk-ms", type=float, default=4000.0,
                        help="packet size; the fixed-shape NPU worker needs one 4 s window per packet")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - {"passthrough", "cpu", "npu"}
    if unknown:
        raise SystemExit(f"unknown modes: {sorted(unknown)}")

    out = args.out.resolve()
    renditions = out / "renditions"
    renditions.mkdir(parents=True, exist_ok=True)
    reference = renditions / "passthrough.wav"
    normalize_excerpt(args.input, reference, args.start, args.seconds)
    chunk_frames = int(round(args.chunk_ms / 1000.0 * SAMPLE_RATE))
    report: dict = {"input": str(args.input), "excerpt": {"start_s": args.start, "seconds": args.seconds},
                    "chunk_frames": chunk_frames, "renditions": {}}

    if "cpu" in modes:
        log("rendering CPU rendition (local PyTorch worker) ...")
        report["renditions"]["cpu"] = render_local_cpu(reference, renditions / "cpu.wav", chunk_frames)
        log(f"CPU: {report['renditions']['cpu']['actual_mode']} in "
            f"{report['renditions']['cpu']['wall_s']}s")

    if "npu" in modes:
        log(f"rendering NPU rendition (Edge2 {args.board}) ...")
        report["renditions"]["npu"] = render_remote_npu(reference, renditions / "npu.wav", args.board,
                                                        args.audio_port, args.http_port, chunk_frames)
        log(f"NPU: {report['renditions']['npu']['actual_mode']} in "
            f"{report['renditions']['npu']['wall_s']}s")

    for name, path in (("passthrough", reference), ("cpu", renditions / "cpu.wav"), ("npu", renditions / "npu.wav")):
        if path.exists():
            report.setdefault("files", {})[name] = {"path": str(path), "duration_s": round(wav_duration(path), 3),
                                                    "rms": round(wav_rms(path), 5)}
    (out / "render-report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"renditions: {renditions}")
    log(f"report: {out / 'render-report.json'}")
    log("next: .venv/bin/python tools/listening_ab.py --render "
        f"{renditions} --out {out}/blind-$(date +%Y%m%d-%H%M%S)")


if __name__ == "__main__":
    main()
