#!/usr/bin/env python3
"""Offline environment-decision A/B through the envelope worker path.

Runs the same six-stem envelope chain as the board worker on a WAV clip, once per
environment label (plus an optional env-off baseline), and writes one processed WAV and
a JSON summary per run. Intended for real captures (e.g. the phone A2DP source monitor)
so the live relay is never restarted; the live and offline paths share
`StreamingWindowManager`.

Board example:
    .venv/bin/python tools/env_offline_ab.py --input /tmp/clip.wav \
        --backend rknnlite --model artifacts/rk-fused.rknn \
        --outdir /tmp/env-ab --env quiet,street,speech --baseline

Local example (no NPU):
    .venv/bin/python tools/env_offline_ab.py --input clip.wav --backend torch \
        --model models/candidate-v0.pt --outdir /tmp/env-ab --env street --baseline

The clip must be 44.1 kHz stereo 16-bit; only full windows are processed (trailing
partial window is dropped), so use a clip length that is a multiple of 4.005 s.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from khadas_ows import worker as W  # noqa: E402

ENV_EVIDENCE: dict[str, dict] = {
    "quiet": {"context": "quiet", "environment": "QUIET", "event": "quiet"},
    "office": {"context": "other", "environment": "NOISY", "event": "quiet"},
    "street": {"context": "street", "environment": "NOISY", "event": "quiet"},
    "speech": {"context": "near_voice", "environment": "VOICE", "event": "speech"},
    "babble": {"context": "babble", "environment": "NOISY", "event": "quiet"},
    "wind": {"context": "wind", "environment": "NOISY", "event": "wind"},
    "keyboard": {"context": "other", "environment": "NOISY", "event": "keyboard"},
}


def read_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (44100, 2, 2):
            raise SystemExit("input must be 44100 Hz stereo 16-bit WAV")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, 2) / 32768.0


def write_wav(path: str | Path, data: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes((np.clip(data, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def band_shares(mono: np.ndarray, rate: int = 44100) -> dict[str, float]:
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / rate)
    total = float(power.sum()) + 1e-20

    def share(lo: float, hi: float) -> float:
        return round(100.0 * float(power[(freqs >= lo) & (freqs < hi)].sum() / total), 1)

    return {"low": share(20, 200), "mid": share(300, 3000), "high": share(3000, 12000)}


def run(manager, pcm: np.ndarray, chunk: int, refresh=None) -> np.ndarray:
    blocks = []
    for start in range(0, pcm.shape[0], chunk):
        block = pcm[start:start + chunk]
        if block.shape[0] < chunk:
            break  # full windows only: no zero-padded tail in an A/B render
        if refresh is not None:
            refresh()  # static evidence ages out of the 2.5 s TTL; keep it fresh
        out = manager.process(block.reshape(-1).astype(np.float32), 2)
        blocks.append(out.reshape(-1, 2))
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 2), dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="44.1 kHz stereo 16-bit WAV")
    parser.add_argument("--outdir", default="/tmp/env-ab")
    parser.add_argument("--env", default="quiet,street,speech,babble,wind,keyboard")
    parser.add_argument("--baseline", action="store_true", help="also render env-decision=off")
    parser.add_argument("--backend", default="stub", choices=("stub", "torch", "rknnlite"))
    parser.add_argument("--model", help="torch .pt or RKNN .rknn for the chosen backend")
    parser.add_argument("--window-frames", type=int, default=345)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pcm = read_wav(args.input)
    chunk = int(args.window_frames) * 512
    backend = W.make_backend(args.backend, args.model, None)
    summary = {
        "input": str(args.input),
        "backend": args.backend,
        "model": args.model,
        "seconds_input": round(pcm.shape[0] / 44100, 2),
        "seconds_processed": round((pcm.shape[0] // chunk) * chunk / 44100, 2),
        "runs": [],
    }

    def one(label: str, evidence: dict, env_decision: str) -> None:
        payload = dict(evidence)
        payload["band_dbfs"] = [-60.0] * 24
        tmp = Path(tempfile.mkdtemp()) / "evidence.json"
        tmp.write_text(json.dumps(payload))
        manager = W.StreamingWindowManager(
            backend, output_mode="envelope", policy="roles", listening_mode="auto",
            env_decision=env_decision, evidence_file=str(tmp), output_gain="unity",
            window_frames=args.window_frames)
        output = run(manager, pcm, chunk, refresh=lambda: tmp.touch())
        write_wav(outdir / f"{label}.wav", output)
        mono = output.mean(axis=1)
        rms = 20.0 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-12)
        entries = manager.role_history
        last = entries[-1] if entries else {}
        summary["runs"].append({
            "label": label,
            "env": last.get("environment"),
            "env_windows": {env: sum(1 for e in entries if e["environment"] == env)
                            for env in sorted({e["environment"] for e in entries})},
            "mode": last.get("mode"),
            "windows": len(entries),
            "rms_dbfs": round(float(rms), 2),
            "bands": band_shares(mono),
            "target_db_mean": {k: round(float(np.mean([e["target_gain_db"][k] for e in entries])), 2)
                               for k in last.get("target_gain_db", {})},
            "applied_db_mean": {k: round(float(np.mean(
                [e["applied_gain_db_last_frame"][k] for e in entries])), 2)
                for k in last.get("applied_gain_db_last_frame", {})},
        })

    if args.baseline:
        one("baseline", {"context": "quiet", "environment": "QUIET", "event": "quiet"}, "off")
    for env in [item.strip() for item in args.env.split(",") if item.strip()]:
        if env not in ENV_EVIDENCE:
            raise SystemExit(f"unknown env {env!r}; expected one of {sorted(ENV_EVIDENCE)}")
        one(env, ENV_EVIDENCE[env], "on")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
