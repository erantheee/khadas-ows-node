#!/usr/bin/env python3
"""Deterministic 10 s stereo 44.1 kHz engineering fixture.

This is a *test signal*, not listening material: harmonic chord progression with
drum-like transients and a noise floor. It exists so tests, the soak harness and the
network probe can run without shipping binary audio in the repository.

Usage:
    python tools/make_fixture.py --output /tmp/fixture-10s.wav [--seconds 10]
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

SR = 44100


def _chord_notes(root_hz: float) -> list[float]:
    # Root, major third, fifth, octave.
    return [root_hz, root_hz * 2 ** (4 / 12), root_hz * 2 ** (7 / 12), root_hz * 2.0]


def synthesize(seconds: float = 10.0, seed: int = 20261106) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(seconds * SR))
    t = np.arange(n) / SR
    out = np.zeros((n, 2), dtype=np.float64)

    # Chord bed: four chords, two bars each at 120 BPM (2 s per chord).
    roots = [220.0, 174.61, 146.83, 196.0]
    chord_len = 2 * SR
    for chord_index, root in enumerate(roots):
        start = chord_index * chord_len
        stop = min(start + chord_len, n)
        if start >= n:
            break
        local_t = t[start:stop] - t[start]
        # Crossfade the chord entry so the fixture has no clicks.
        envelope = np.minimum(1.0, local_t * 40.0) * np.exp(-local_t / 1.6)
        voice = np.zeros(stop - start)
        for note_index, hz in enumerate(_chord_notes(root)):
            vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 5.2 * local_t + note_index)
            for harmonic in range(1, 6):
                voice += (0.12 / harmonic) * np.sin(
                    2 * np.pi * hz * harmonic * local_t * vibrato
                    + rng.uniform(0, 2 * np.pi)
                )
        out[start:stop, 0] += 0.35 * envelope * voice
        out[start:stop, 1] += 0.32 * envelope * np.roll(voice, 23)

    # Drum bed at 120 BPM: kick on 1/3, snare on 2/4, hats on eighths.
    beat = 60 / 120
    for k in range(int(seconds / beat)):
        at = k * beat
        start = int(at * SR)
        if start >= n:
            break
        if k % 4 in (0, 2):  # kick
            length = int(0.28 * SR)
            local = np.arange(length) / SR
            sweep = 110 * np.exp(-local * 28) + 42
            hit = np.sin(2 * np.pi * np.cumsum(sweep) / SR) * np.exp(-local * 11)
            stop = min(start + length, n)
            out[start:stop, 0] += 0.5 * hit[: stop - start]
            out[start:stop, 1] += 0.5 * hit[: stop - start]
        if k % 4 in (1, 3):  # snare
            length = int(0.18 * SR)
            local = np.arange(length) / SR
            hit = rng.normal(0, 1, length) * np.exp(-local * 30)
            hit = np.convolve(hit, np.ones(3) / 3, mode="same")
            stop = min(start + length, n)
            out[start:stop, 0] += 0.22 * hit[: stop - start]
            out[start:stop, 1] += 0.20 * hit[: stop - start]
        for half in (0.0, beat / 2):  # hats
            hat_start = start + int(half * SR)
            length = int(0.05 * SR)
            local = np.arange(length) / SR
            hit = rng.normal(0, 1, length) * np.exp(-local * 90)
            stop = min(hat_start + length, n)
            if hat_start < n:
                out[hat_start:stop, 0] += 0.10 * hit[: stop - hat_start]
                out[hat_start:stop, 1] += 0.12 * hit[: stop - hat_start]

    # Broadband floor so denormals and silent frames do not make tests unrealistically easy.
    out += rng.normal(0, 1e-4, out.shape)
    out = np.clip(out, -1.0, 1.0)
    return (out * 32767.0).astype("<i2")


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    write_wav(Path(args.output), synthesize(args.seconds))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
