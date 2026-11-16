"""Fast drum-transient EQ for the relay playback path.

Detects kick transients from the low-band short-time energy slope and applies
a short dry/wet-mixed shape: low punch (+dB shelf below ~120 Hz) plus a short
mid dip (~1 kHz) that lets the kick through. Attack ~20 ms, release ~250 ms,
all per-sample ramped inside each 50 ms block — no block-edge clicks.

Everything else stays untouched: vocals / keys / cymbals / MIDI / bass keep
going through the six-stem envelope on the slow path. The drum transient is
"obvious" enough to solve instantly; the rest gets the envelope treatment.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100


class Biquad:
    """RBJ cookbook biquad (direct form I transposed), fixed coefficients."""

    def __init__(self, kind: str, f0: float, q: float, db_gain: float,
                 sr: int = SAMPLE_RATE) -> None:
        a_gain = 10.0 ** (db_gain / 40.0)
        w0 = 2.0 * np.pi * f0 / sr
        cos_w, sin_w = np.cos(w0), np.sin(w0)
        alpha = sin_w / (2.0 * q)
        if kind == "lowshelf":
            s = 2.0 * np.sqrt(a_gain) * alpha
            b0 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_w + s)
            b1 = 2.0 * a_gain * ((a_gain - 1.0) - (a_gain + 1.0) * cos_w)
            b2 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_w - s)
            a0 = (a_gain + 1.0) + (a_gain - 1.0) * cos_w + s
            a1 = -2.0 * ((a_gain - 1.0) + (a_gain + 1.0) * cos_w)
            a2 = (a_gain + 1.0) + (a_gain - 1.0) * cos_w - s
        elif kind == "peaking":
            b0 = 1.0 + alpha * a_gain
            b1 = -2.0 * cos_w
            b2 = 1.0 - alpha * a_gain
            a0 = 1.0 + alpha / a_gain
            a1 = -2.0 * cos_w
            a2 = 1.0 - alpha / a_gain
        elif kind == "lowpass":
            b0 = (1.0 - cos_w) / 2.0
            b1 = 1.0 - cos_w
            b2 = (1.0 - cos_w) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w
            a2 = 1.0 - alpha
        else:
            raise ValueError(f"unknown biquad kind {kind!r}")
        self.b = np.array([b0, b1, b2], dtype=np.float64) / a0
        self.a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        self.z = np.zeros(2, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float64)
        b0, b1, b2 = self.b
        a1, a2 = self.a[1], self.a[2]
        z0, z1 = self.z[0], self.z[1]
        for i in range(x.shape[0]):
            w = x[i] - a1 * z0 - a2 * z1
            y[i] = b0 * w + b1 * z0 + b2 * z1
            z1 = z0
            z0 = w
        self.z[0], self.z[1] = z0, z1
        return y


class DrumEnhancer:
    """Kick transient detector + dry/wet-mixed low punch and mid dip."""

    ATTACK_SAMPLES = int(0.020 * SAMPLE_RATE)   # 20 ms
    RELEASE_SAMPLES = int(0.250 * SAMPLE_RATE)  # 250 ms
    REFRACTORY_SAMPLES = int(0.180 * SAMPLE_RATE)  # no re-trigger within 180 ms
    TRIGGER_DB = 10.0                           # low-band energy jump in dB

    def __init__(self, boost_db: float = 3.0, cut_db: float = -1.5,
                 sr: int = SAMPLE_RATE) -> None:
        self.boost_db = float(boost_db)
        self.cut_db = float(cut_db)
        self.sr = sr
        # energy probe: low band below ~150 Hz
        self.probe = Biquad("lowpass", 150.0, 0.707, 0.0, sr)
        # fixed shape filters; dry/wet mix implements the dynamic gain
        self.low_shelf = Biquad("lowshelf", 120.0, 0.707, self.boost_db, sr)
        self.mid_peak = Biquad("peaking", 1000.0, 1.0, self.cut_db, sr)
        self.avg = 0.0
        self.ramp = 0.0  # 0..1 shape amount
        self.events = 0
        self.refractory = 0

    def process_block(self, block: np.ndarray) -> np.ndarray:
        x = block.astype(np.float64, copy=False)
        probe = self.probe.process(x)
        energy = float(np.sqrt(np.mean(probe ** 2)) + 1e-12)
        if self.avg <= 0.0:
            self.avg = energy
        else:
            self.avg = 0.95 * self.avg + 0.05 * energy
        ratio_db = 20.0 * np.log10((energy + 1e-12) / (self.avg + 1e-12))
        if self.refractory > 0:
            self.refractory -= x.shape[0]
        triggered = ratio_db > self.TRIGGER_DB and self.refractory <= 0
        if triggered:
            self.refractory = self.REFRACTORY_SAMPLES
        target = 1.0 if triggered else 0.0
        if triggered:
            self.events += 1
        n = x.shape[0]
        if target > self.ramp:
            step = 1.0 / max(self.ATTACK_SAMPLES, 1)
            ramp = np.clip(np.arange(n) * step + self.ramp, 0.0, target)
        else:
            step = 1.0 / max(self.RELEASE_SAMPLES, 1)
            ramp = np.clip(self.ramp - np.arange(n) * step, target, 1.0)
        self.ramp = float(ramp[-1])
        if float(self.ramp) < 0.02 and target == 0.0 and float(ramp[-1]) < 0.02:
            return block
        wet_low = self.low_shelf.process(x)
        wet_mid = self.mid_peak.process(wet_low)
        out = x + (wet_mid - x) * ramp
        return out.astype(block.dtype, copy=False)


def main() -> int:
    """Self-test: synthetic kick train in noise; prints detected events."""
    import math
    rng = np.random.default_rng(7)
    sr = SAMPLE_RATE
    total = int(sr * 4.0)
    sig = rng.normal(0, 0.01, total)
    kick_period = int(sr * 0.5)
    for start in range(0, total, kick_period):
        t = np.arange(int(sr * 0.08))
        kick = np.exp(-t / (sr * 0.012)) * np.sin(2 * math.pi * 55 * t / sr)
        sig[start:start + len(kick)] += kick * 0.9
    enh = DrumEnhancer(boost_db=3.0, cut_db=-1.5)
    block = int(sr * 0.05)
    for start in range(0, total, block):
        enh.process_block(sig[start:start + block])
    print(f"self-test: blocks processed, kick events detected = {enh.events} "
          f"(expect ~8)")
    return 0 if enh.events >= 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
