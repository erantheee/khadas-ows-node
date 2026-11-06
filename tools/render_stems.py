#!/usr/bin/env python3
"""Render the four separated stems of one excerpt as individual WAVs (CPU model, offline).

This is for *listening to the separation itself* (what each estimated stem sounds like,
where the leakage is). The product path does not play stems; it uses them for role-aware
envelope modulation (see `tools/render_modes.py --worker-args ... --output-mode envelope`).

    .venv/bin/python tools/render_stems.py --input /tmp/qq-capture.wav --out docs/listening-material/qq-music/stems

Outputs: `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`, `mixture.wav` (reference copy),
`roles.json` (per-window dominant roles + stem RMS, i.e. what the strategy would see).
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from khadas_ows import strategy  # noqa: E402
from khadas_ows.model import load_legacy_checkpoint  # noqa: E402
from khadas_ows.worker import (HOP, MODEL_FRAMES, SAMPLE_RATE, istft, stft)  # noqa: E402

STEMS = strategy.STEMMS


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == SAMPLE_RATE and handle.getnchannels() == 2, "need 44.1 kHz stereo"
        raw = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return pcm.reshape(-1, 2).T


def write_wav(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(signal, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((clipped.T.reshape(-1) * 32767.0).round().astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="models/candidate-v0.pt")
    args = parser.parse_args()

    mix = read_wav(args.input)
    model = load_legacy_checkpoint(args.model)
    stems = {stem: np.zeros_like(mix) for stem in STEMS}
    window = MODEL_FRAMES * HOP
    roles: list[dict] = []
    tracker = strategy.RoleTracker()
    for start in range(0, mix.shape[-1], window):
        chunk = mix[..., start:start + window]
        if chunk.shape[-1] < HOP * 8:
            break
        spectrum = stft(chunk)
        magnitude = np.abs(spectrum)
        frames = magnitude.shape[-1]
        padded = np.zeros((1, 2, 1025, MODEL_FRAMES), dtype=np.float32)
        used = min(MODEL_FRAMES, frames)
        padded[0, :, :, :used] = magnitude[:, :, :used]
        with torch.no_grad():
            masks = model(torch.from_numpy(padded)).numpy()[0]
        stats = {}
        for index, stem in enumerate(STEMS):
            estimate = istft(masks[index, :, :used].astype(np.complex64) * spectrum[:, :, :used],
                             chunk.shape[-1])
            stems[stem][..., start:start + estimate.shape[-1]] = estimate
            rms = float(np.sqrt((np.asarray(estimate, dtype=np.float64) ** 2).mean()))
            stats[stem] = 20.0 * np.log10(max(rms, 1e-12))
        roles.append({"t_s": round(start / SAMPLE_RATE, 2),
                      "stem_rms_db": {stem: round(stats[stem], 2) for stem in STEMS},
                      "roles": tracker.update(stats)})
        print(f"window {start // window}: {roles[-1]}", flush=True)

    out = args.out.resolve()
    write_wav(out / "mixture.wav", mix)
    for stem in STEMS:
        write_wav(out / f"{stem}.wav", stems[stem])
    (out / "roles.json").write_text(json.dumps({"input": str(args.input), "windows": roles}, indent=2) + "\n")
    print(f"stems written to {out}")


if __name__ == "__main__":
    main()
