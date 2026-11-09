#!/usr/bin/env python3
"""Deterministic calibration set for RKNN INT8 conversion (engineering probe only).

The MaskNetwork input is a real magnitude spectrogram [1, 2, 1025, 345]. RKNN
quantisation needs representative inputs, so this script builds a fixed set of
magnitude spectrograms from deterministic synthetic mixtures with the same STFT
geometry as the runtime worker (n_fft=2048, hop=512, centre=True, Hann window).

The mixtures are engineering signals (harmonic music bed, drum transients, speech-like
formant bursts, noise, near-silence) and are *not* a claim about listening quality or a
real calibration corpus. Every entry is reproducible from the seed.

Usage:
    python tools/make_calibration.py --output-dir models/calib
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SR = 44100
N_FFT = 2048
HOP = 512
DEFAULT_FRAMES = 345  # 4 s window at hop 512, matching the default exported ONNX shape
SEED = 20261106


def _noise(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(0, 1, (2, n))


def _music(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for channel in range(2):
        root = 110.0 * 2 ** (rng.integers(0, 12) / 12)
        for harmonic in range(1, 9):
            out[channel] += (0.5 / harmonic) * np.sin(
                2 * np.pi * root * harmonic * t + rng.uniform(0, 2 * np.pi)
            )
        for _ in range(6):
            at = rng.integers(0, max(1, n - SR // 4))
            length = SR // 8
            local = np.arange(length) / SR
            out[channel, at : at + length] += 0.8 * np.sin(
                2 * np.pi * (60 + 50 * rng.random()) * local
            ) * np.exp(-local * 20)
    return out


def _speech_like(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    f0 = 110 + 60 * rng.random()
    for channel in range(2):
        formant = 0.6 * np.sin(2 * np.pi * f0 * t)
        formant += 0.25 * np.sin(2 * np.pi * f0 * 2 * t + 0.4)
        formant += 0.15 * np.sin(2 * np.pi * f0 * 3.3 * t + 1.1)
        syllable = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6.28))
        out[channel] = formant * syllable
    return out


def _near_silence(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(0, 1e-3, (2, n))


def magnitude(audio: np.ndarray, frames: int = DEFAULT_FRAMES) -> np.ndarray:
    x = torch.from_numpy(audio.astype(np.float32))
    window = torch.hann_window(N_FFT)
    spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True)
    mag = spec.abs()[:, :, :frames].numpy().astype(np.float32)
    if mag.shape[-1] < frames:  # defensive; a 4 s packet always yields the full window
        mag = np.pad(mag, ((0, 0), (0, 0), (0, frames - mag.shape[-1])))
    return mag[None, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="STFT frames per calibration sample (default 345 = 4 s)")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    builders = [_music, _speech_like, _noise, _near_silence]
    entries = []
    for index in range(args.count):
        builder = builders[index % len(builders)]
        audio = builder(rng, 4 * SR)
        scale = 10 ** rng.uniform(-1.0, 0.2)
        mag = magnitude(audio * scale, args.frames)
        path = output / f"calib_{index:02d}.npy"
        np.save(path, mag)
        entries.append(
            {
                "file": path.name,
                "shape": list(mag.shape),
                "sha256": hashlib.sha256(mag.tobytes()).hexdigest(),
                "dtype": str(mag.dtype),
            }
        )
    dataset = output / "dataset.txt"
    dataset.write_text("".join(f"{entry['file']}\n" for entry in entries))
    manifest = {
        "schema": 1,
        "purpose": "RKNN INT8 calibration for MaskNetwork magnitude input",
        "probe_only": True,
        "stft": {"n_fft": N_FFT, "hop": HOP, "center": True, "window": "hann"},
        "frames": args.frames,
        "seed": SEED,
        "count": len(entries),
        "entries": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "count": len(entries), "shape": list(magnitude(_music(rng, 4 * SR), args.frames).shape)}))


if __name__ == "__main__":
    main()
