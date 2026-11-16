"""S3-style smart tuning on the Edge2 (the board is the tuning engineer).

Target architecture (user, 2026-11-09):

    Mac      capture system audio only  ->  PCM to the board
    Edge2    separation (NPU) + noise perception (DMIC 24-band) + song style/content
             + 24-band envelope decisions + masking matrix   <- the "smart tuner"
    Mac      play the processed PCM

The S3 firmware used to do the decision part (`main/controller.cpp`, D39/D41/D48/
D51/D54).  Its executor then applied the 24-band envelope plus the D56 content x
ambient masking matrix (`systemeq.swift`).  This module moves both of those
layers onto the board, driven by the board's own DMIC evidence and by the
separated spectrum, so none of the tuning has to live on the Mac anymore.

Inputs (per ~1 s packet):
    ambient_bands  DMIC 24-band Goertzel dBFS (noise), /tmp/dmic-latest.json
    level_dbfs     DMIC broadband dBFS
    event          quiet/speech/keyboard/wind/knock/friction
    voice          voice likelihood 0..1
    conversation   node policy state (IDLE/ANTICIPATING/ENGAGED/RELEASING)
    content_bands  24 band dB of the music (from the separated spectrum)
    stem_shares    per-stem energy share (style hint from separation)

Outputs: mode, device suggestion, applied bands (device x depth + style +
masking, clamped), masking bands, exposure, style tag.  Fail-open: missing or
stale evidence -> FLAT with zero gains; the caller must still return audio.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.request

import numpy as np

BAND_COUNT = 24
#: 分轨轨数（vocals/drums/bass/other）——逐轨包络就是按这 4 条分别算/分别执行。
STEM_COUNT = 4
STEM_NAMES = ("vocals", "drums", "bass", "other")
BAND_CENTERS_HZ = np.array([
    70.7, 141.4, 244.9, 346.4, 451.7, 566.8, 696.5, 841.7,
    996.8, 1171.3, 1370.0, 1595.0, 1854.7, 2154.1, 2503.3,
    2916.3, 3414.7, 4036.1, 4829.1, 5824.1, 7020.0, 8552.8,
    10677.1, 13638.2,
])

# ── D39 content curve (same constants as main/controller.cpp) ─────────────────
SONG_EQ_CONTENT_DB = np.array([
    2.5, 3.0, 0.0, -1.8, -1.8, -1.4, -0.6, -0.5,
    0.0, 0.5, 1.0, 1.2, 1.7, 2.1, 2.3, 2.3,
    1.9, 1.5, 1.3, 1.3, 1.3, 2.6, 2.6, 1.9,
])
SONG_EQ_MASK_K = np.array([
    0.65, 0.65, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.60, 0.60, 0.60, 0.45, 0.45, 0.45, 0.45, 0.45,
    0.45, 0.45, 0.60, 0.60, 0.60, 0.85, 0.85, 0.85,
])
DEVICE_LIMIT_DB = 4.0
EXPOSURE_QUIET_DBFS = -64.0
EXPOSURE_SPAN_DB = 18.0
BAND_FLOOR_DBFS = -75.0
BAND_SPAN_DB = 40.0

# ── D54 mode routing / D51 preview gate ──────────────────────────────────────
QUIET_DBFS = -66.0
LOUD_DBFS = -40.0
PREVIEW_CONFIRM_MS = 500.0
VOICE_MODE_ON_MS = 300.0
VOICE_MODE_OFF_MS = 500.0
LEVEL_ATTACK_MS = 250.0
LEVEL_RELEASE_MS = 250.0

# ── D41 background-voice masking (demo semantics) ────────────────────────────
VOICE_FLOOR = 0.55
VOICE_FULL = 0.75
EVI_BROADBAND_DB = 1.0
EVI_EMPHASIS_DB = 4.0
EVI_FIRST_BAND = 8
EVI_LAST_BAND = 17

# ── D48 noise / keyboard masking ─────────────────────────────────────────────
NOISE_MASK_BROADBAND_DB = 0.3
NOISE_MASK_EMPHASIS_DB = 1.0
NOISE_MASK_FIRST_BAND = 11
NOISE_MASK_LAST_BAND = 17
NOISE_MASK_FLOOR_DBFS = -54.0
NOISE_MASK_SPAN_DB = 20.0
KEYBOARD_MASK_BROADBAND_DB = 0.5
KEYBOARD_MASK_EMPHASIS_DB = 1.5
KEYBOARD_FIRST_BAND = 8
KEYBOARD_LAST_BAND = 16
KEYBOARD_MASK_FLOOR_DBFS = -52.0
KEYBOARD_MASK_SPAN_DB = 18.0
MASK_ATTACK_MS = 400.0
MASK_RELEASE_MS = 1200.0
KEYBOARD_ATTACK_MS = 250.0
KEYBOARD_RELEASE_MS = 2600.0

# ── D56 content x ambient masking matrix (same formulas as systemeq.swift) ───
MASKING_BUDGET_MID_DB = 5.0
MASKING_BUDGET_LOW_DB = 2.4
MASKING_BUDGET_HIGH_DB = 3.0
MASKING_FLOOR_MID_DB = 3.6
MASKING_FLOOR_LOW_DB = 1.6
MASKING_FLOOR_HIGH_DB = 2.6
MASKING_EXPOSURE_FLOOR_DBFS = -64.0
MASKING_EXPOSURE_SPAN_DB = 18.0
MASKING_TICK_S = 0.05
MASKING_ATTACK_ALPHA = 0.18
MASKING_RELEASE_ALPHA = 0.05

# ── application layer ────────────────────────────────────────────────────────
DEFAULT_DEPTH = 1.3          # basic panel depth; aggressive passes 2.5
DEFAULT_MASKING_DOSE = 1.0
MAX_POSITIVE_DB = 3.0        # D54: basic positive-gain ceiling 3 dB (aggressive 4)
MIN_APPLIED_DB = -12.0
OUTPUT_PEAK_LIMIT = 0.995
EVIDENCE_MAX_AGE_S = 2.5

MODE_FLAT = "FLAT"
MODE_SONG_EQ = "SONG_EQ"
MODE_EXTERNAL_VOICE = "EXTERNAL_VOICE_INTENT"
MODE_SAFETY_HOLD = "SAFETY_HOLD"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def slew(current: float, target: float, dt: float, attack: float, release: float) -> float:
    tau = attack if target > current else release
    alpha = 1.0 - math.exp(-max(dt, 0.0) / max(tau, 1e-6))
    return current + (target - current) * alpha


def masking_band_target(index: int, noise_relative: float, song_relative: float,
                        exposure: float, dose: float, transient_risk: float) -> float:
    """D56 per-band target (same formula as systemeq.swift maskingBandTarget)."""
    noise_prominence = clamp01((noise_relative + 18.0) / 18.0)
    song_presence = clamp01((song_relative + 24.0) / 24.0)
    deficit = clamp01((noise_relative - song_relative - 3.0) / 15.0)
    overlap = min(noise_prominence, song_presence)
    content_presence = clamp01((song_relative + 18.0) / 6.0)
    need = 0.45 + 0.55 * deficit
    budget = (MASKING_BUDGET_MID_DB if 8 <= index <= 16
              else (MASKING_BUDGET_LOW_DB if index <= 7 else MASKING_BUDGET_HIGH_DB)) * dose
    transient_scale = max(0.35, 1.0 - 0.75 * clamp01(transient_risk))
    if noise_relative - song_relative < 3.0:
        return 0.0
    exposure_scale = clamp01((exposure - 0.35) / 0.45)
    floor_db = (MASKING_FLOOR_MID_DB if 8 <= index <= 16
                else (MASKING_FLOOR_LOW_DB if index <= 7 else MASKING_FLOOR_HIGH_DB))
    audible = floor_db * dose * exposure_scale / transient_scale * content_presence
    shaped = exposure * max(0.25, overlap) * need * budget * 1.8 * content_presence
    return float(min(budget, max(shaped, audible)))


class BandMap:
    """FFT bins <-> 24 log-spaced bands (analysis + synthesis)."""

    def __init__(self, n_fft: int = 2048, sample_rate: int = 44100) -> None:
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        log_freqs = np.log2(np.maximum(freqs, 1e-3))
        log_centers = np.log2(BAND_CENTERS_HZ)
        pos = np.interp(log_freqs, log_centers, np.arange(BAND_COUNT), left=0.0,
                        right=float(BAND_COUNT - 1))
        self.low = np.floor(pos).astype(np.int32)
        self.high = np.minimum(self.low + 1, BAND_COUNT - 1)
        self.frac = (pos - self.low).astype(np.float32)
        nearest = np.abs(log_freqs[:, None] - log_centers[None, :]).argmin(axis=1)
        self.band_power_matrix = np.zeros((BAND_COUNT, len(freqs)), np.float32)
        self.band_power_matrix[nearest, np.arange(len(freqs))] = 1.0

    def gains_to_bins(self, gains_db: np.ndarray) -> np.ndarray:
        gains = np.asarray(gains_db, dtype=np.float64)
        return (gains[self.low] * (1.0 - self.frac)
                + gains[self.high] * self.frac).astype(np.float32)

    def band_power(self, power_spectrum: np.ndarray) -> np.ndarray:
        return self.band_power_matrix @ np.asarray(power_spectrum, dtype=np.float64)


def style_offsets(content_bands: np.ndarray,
                  stem_shares: "dict[str, float] | None") -> tuple[np.ndarray, str]:
    """Song style features -> small 24-band offsets (v1, within +-1 dB)."""
    offsets = np.zeros(BAND_COUNT, dtype=np.float64)
    power = 10.0 ** (np.asarray(content_bands, dtype=np.float64) / 10.0)
    total = float(power.sum()) + 1e-12
    low_share = float(power[:8].sum()) / total
    high_share = float(power[17:].sum()) / total
    if low_share >= 0.55:
        offsets[:8] = -0.5          # bass-heavy: do not stack more low boost
        style = "bass_heavy"
    elif high_share >= 0.30:
        offsets[17:] = -0.8         # dense: 6 kHz+ trim against extra brightness
        style = "dense"
    else:
        style = "balanced"
    if stem_shares:
        vocals = float(stem_shares.get("vocals", 0.0))
        drums = float(stem_shares.get("drums", 0.0))
        if vocals >= 0.35:
            offsets[11:17] += 0.3
            style = f"{style}+vocal"
        if drums >= 0.40:
            offsets[:8] -= 0.2
            style = f"{style}+rhythm"
    return np.clip(offsets, -1.0, 1.0), style


class EvidenceReader:
    """Reads /tmp/dmic-latest.json (board DMIC evidence) with a staleness gate."""

    def __init__(self, path: str = "/tmp/dmic-latest.json", max_age_s: float = EVIDENCE_MAX_AGE_S):
        self.path = path
        self.max_age_s = max_age_s

    def read(self) -> "dict | None":
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return None
        if time.time() - mtime > self.max_age_s:
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        bands = data.get("band_dbfs")
        if not isinstance(bands, list) or len(bands) != BAND_COUNT:
            return None
        return data


class NodeStatusReader:
    """Reads the node conversation policy state (same board, loopback).

    MUST be non-blocking: the node is a single-threaded poll loop, so while it
    waits for this worker to process the current audio packet it cannot answer
    an HTTP status request.  A synchronous GET inside the worker would stall
    the audio path for the request timeout on every packet (measured 300+ ms
    spikes, enough to starve the output ring).  A daemon refreshes the cached
    value in the background; the audio thread only reads the cache.
    """

    def __init__(self, url: str = "http://127.0.0.1:8787/v1/status",
                 interval_s: float = 1.0):
        self.url = url
        self.interval_s = interval_s
        self._value: dict = {}
        self._lock = threading.Lock()
        self._started = False

    def _loop(self) -> None:
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=0.4) as response:
                    value = json.loads(response.read().decode("utf-8"))
                if isinstance(value, dict):
                    with self._lock:
                        self._value = value
            except (OSError, ValueError):
                pass
            time.sleep(self.interval_s)

    def read(self) -> dict:
        if not self._started:
            self._started = True
            threading.Thread(target=self._loop, name="node-status", daemon=True).start()
        with self._lock:
            return dict(self._value)


class BoardTuner:
    """S3 decision layer + D56 masking, running on the board."""

    def __init__(self, depth: float = DEFAULT_DEPTH,
                 masking_dose: float = DEFAULT_MASKING_DOSE,
                 max_positive_db: float = MAX_POSITIVE_DB,
                 band_map: "BandMap | None" = None,
                 evidence: "EvidenceReader | None" = None,
                 node_status: "NodeStatusReader | None" = None) -> None:
        self.depth = float(depth)
        self.masking_dose = float(masking_dose)
        self.max_positive_db = float(max_positive_db)
        self.band_map = band_map or BandMap()
        self.evidence = evidence or EvidenceReader()
        self.node_status = node_status or NodeStatusReader()
        self.level: float | None = None
        self.voice_mode = False
        self.voice_on_ms = 0.0
        self.voice_off_ms = 0.0
        self.anticipating_ms = 0.0
        self.noise_mask_depth = 0.0
        self.keyboard_mask_depth = 0.0
        self.masking_db = np.zeros(BAND_COUNT, dtype=np.float64)
        self.device_db = np.zeros(BAND_COUNT, dtype=np.float64)
        self.applied_db = np.zeros(BAND_COUNT, dtype=np.float64)
        #: 逐轨 24 段包络（4×24，含 650/850 ms slew）：audio 路径按它分别处理 4 条 stem。
        self.stem_db = np.zeros((STEM_COUNT, BAND_COUNT), dtype=np.float64)
        self.last: dict = {"mode": MODE_FLAT, "style": "unknown", "stale": True}

    # ── content / style features from the separated spectrum ─────────────────
    def content_features(self, spectrum: np.ndarray, masks: np.ndarray):
        """spectrum [2,1025,T], masks [1,4,1025,T] -> (content dB[24], shares, per-stem dB[4,24])."""
        power = np.abs(spectrum).astype(np.float64) ** 2          # [2,b,T]
        power_sum = power.sum(axis=0)                             # [b,T]
        band_power = self.band_map.band_power(power_sum.mean(axis=1))
        content_db = 10.0 * np.log10(np.maximum(band_power, 1e-18))
        masks_power = np.asarray(masks[0], dtype=np.float64) ** 2  # [4,b,T]
        totals = (masks_power * power_sum[None, :, :]).sum(axis=(1, 2)) + 1e-18
        shares = {name: float(totals[i] / totals.sum())
                  for i, name in enumerate(STEM_NAMES)}
        # 逐轨内容谱：每轨自己的 24 段电平。逐轨包络按"该轨在这个频段有没有内容"门控，
        # 空频段不会被硬塞增益（否则分离残差/空段会被抬出怪声）。
        stem_db = np.zeros((STEM_COUNT, BAND_COUNT), dtype=np.float64)
        for stem in range(STEM_COUNT):
            stem_bins = (masks_power[stem] * power_sum).sum(axis=1)
            stem_bands = self.band_map.band_power(stem_bins)
            stem_db[stem] = 10.0 * np.log10(np.maximum(stem_bands, 1e-18))
        return content_db, shares, stem_db

    def stale(self) -> dict:
        self.device_db[:] = 0.0
        self.masking_db[:] = 0.0
        self.applied_db[:] = 0.0
        self.stem_db[:] = 0.0
        self.voice_mode = False
        self.level = None
        self.last = {"mode": MODE_FLAT, "applied_db": self.applied_db.copy(),
                     "device_db": self.device_db.copy(), "masking_db": self.masking_db.copy(),
                     "stem_db": self.stem_db.copy(),
                     "style": "stale", "exposure": 0.0, "stale": True}
        return self.last

    def step(self, *, level_dbfs: "float | None", ambient_bands, event: "str | None",
             voice: float, conversation: "str | None", content_bands: np.ndarray,
             stem_shares: "dict[str, float] | None", dt_s: float,
             stem_content_bands: "np.ndarray | None" = None) -> dict:
        dt = max(0.05, min(2.0, float(dt_s)))
        if level_dbfs is None or not isinstance(level_dbfs, (int, float)) or \
                not math.isfinite(float(level_dbfs)):
            return self.stale()
        level = float(level_dbfs)
        self.level = level if self.level is None else slew(
            self.level, level, dt, LEVEL_ATTACK_MS / 1000.0, LEVEL_RELEASE_MS / 1000.0)
        voice = clamp01(voice if isinstance(voice, (int, float)) else 0.0)
        event = str(event or "quiet")
        state = str(conversation or "NONE").upper()

        # D51: raw ANTICIPATING needs 500 ms of accumulated preview before flattening
        if state == "ANTICIPATING":
            self.anticipating_ms += dt * 1000.0
        else:
            self.anticipating_ms = 0.0
        conversation_active = state == "ENGAGED" or (
            state == "ANTICIPATING" and self.anticipating_ms >= PREVIEW_CONFIRM_MS)

        voice_candidate = voice >= VOICE_FLOOR or event == "speech"
        if voice_candidate:
            self.voice_on_ms += dt * 1000.0
            self.voice_off_ms = 0.0
        else:
            self.voice_off_ms += dt * 1000.0
            self.voice_on_ms = 0.0
        if not self.voice_mode and self.voice_on_ms >= VOICE_MODE_ON_MS:
            self.voice_mode = True
        elif self.voice_mode and self.voice_off_ms >= VOICE_MODE_OFF_MS:
            self.voice_mode = False

        if conversation_active or self.level < QUIET_DBFS:
            mode = MODE_FLAT
        elif self.voice_mode:
            mode = MODE_EXTERNAL_VOICE
        elif self.level > LOUD_DBFS:
            mode = MODE_SAFETY_HOLD
        else:
            mode = MODE_SONG_EQ

        # D48 masking depths (updated every packet so they decay across modes)
        noise_target = clamp01((self.level - NOISE_MASK_FLOOR_DBFS) / NOISE_MASK_SPAN_DB)
        keyboard_target = (clamp01((self.level - KEYBOARD_MASK_FLOOR_DBFS) / KEYBOARD_MASK_SPAN_DB)
                           if event == "keyboard" else 0.0)
        self.noise_mask_depth = slew(self.noise_mask_depth, noise_target, dt,
                                     MASK_ATTACK_MS / 1000.0, MASK_RELEASE_MS / 1000.0)
        self.keyboard_mask_depth = slew(self.keyboard_mask_depth, keyboard_target, dt,
                                        KEYBOARD_ATTACK_MS / 1000.0, KEYBOARD_RELEASE_MS / 1000.0)

        ambient = None
        if ambient_bands is not None and len(ambient_bands) == BAND_COUNT:
            values = np.asarray(ambient_bands, dtype=np.float64)
            if np.isfinite(values).all() and float(values.max()) > -119.0:
                ambient = values

        exposure = clamp01((self.level - EXPOSURE_QUIET_DBFS) / EXPOSURE_SPAN_DB)
        offsets, style = style_offsets(content_bands, stem_shares)
        device = np.zeros(BAND_COUNT, dtype=np.float64)
        if mode in (MODE_SONG_EQ, MODE_EXTERNAL_VOICE, MODE_SAFETY_HOLD):
            if ambient is not None:
                dominance = np.clip((ambient - BAND_FLOOR_DBFS) / BAND_SPAN_DB, 0.0, 1.0)
            else:
                dominance = np.full(BAND_COUNT, exposure)
            curve = SONG_EQ_CONTENT_DB * exposure * (1.0 - SONG_EQ_MASK_K * dominance)
            curve = np.clip(curve, -DEVICE_LIMIT_DB, DEVICE_LIMIT_DB)
            if mode == MODE_EXTERNAL_VOICE:
                mask_voice = clamp01((voice - VOICE_FLOOR) / (VOICE_FULL - VOICE_FLOOR))
                curve = curve + EVI_BROADBAND_DB * mask_voice
                curve[EVI_FIRST_BAND:EVI_LAST_BAND + 1] += EVI_EMPHASIS_DB * mask_voice
            elif mode == MODE_SAFETY_HOLD:
                curve = np.minimum(curve, 0.0)
            else:
                curve = curve + NOISE_MASK_BROADBAND_DB * self.noise_mask_depth
                curve[NOISE_MASK_FIRST_BAND:NOISE_MASK_LAST_BAND + 1] += \
                    NOISE_MASK_EMPHASIS_DB * self.noise_mask_depth
                curve = curve + KEYBOARD_MASK_BROADBAND_DB * self.keyboard_mask_depth
                curve[KEYBOARD_FIRST_BAND:KEYBOARD_LAST_BAND + 1] += \
                    KEYBOARD_MASK_EMPHASIS_DB * self.keyboard_mask_depth
            device = np.clip(curve + offsets, -DEVICE_LIMIT_DB, DEVICE_LIMIT_DB)
        self.device_db = device

        # D56 content x ambient masking matrix (only off FLAT with valid ambient spectrum)
        masking_target = np.zeros(BAND_COUNT, dtype=np.float64)
        masking_exposure = clamp01(
            (self.level - MASKING_EXPOSURE_FLOOR_DBFS) / MASKING_EXPOSURE_SPAN_DB)
        if mode != MODE_FLAT and ambient is not None:
            noise_rel = ambient - float(np.max(ambient))
            content_rel = np.asarray(content_bands, dtype=np.float64)
            content_rel = content_rel - float(np.max(content_rel))
            for i in range(BAND_COUNT):
                masking_target[i] = masking_band_target(
                    i, float(noise_rel[i]), float(content_rel[i]),
                    masking_exposure, self.masking_dose, 0.0)
        if mode != MODE_FLAT:
            attack = 1.0 - (1.0 - MASKING_ATTACK_ALPHA) ** (dt / MASKING_TICK_S)
            release = 1.0 - (1.0 - MASKING_RELEASE_ALPHA) ** (dt / MASKING_TICK_S)
            for i in range(BAND_COUNT):
                alpha = attack if masking_target[i] > self.masking_db[i] else release
                self.masking_db[i] += (masking_target[i] - self.masking_db[i]) * alpha
        else:
            self.masking_db[:] = 0.0

        applied = np.clip(device * self.depth + self.masking_db,
                          MIN_APPLIED_DB, self.max_positive_db)
        self.applied_db = applied

        # ── 逐轨 24 段包络（4 条独立曲线，650/850 ms slew）───────────────────
        # 用同一条 S3 执行曲线，但按"该轨在这个频段有没有内容"门控：每轨只在自己的
        # 频段里被补偿/掩蔽，空频段不硬塞增益。这样才能真正按 4 轨分别调节，而不是
        # 把同一条曲线贴在混音上。
        if stem_content_bands is not None and mode != MODE_FLAT:
            applied_curve = np.clip(device * self.depth + self.masking_db,
                                    MIN_APPLIED_DB, self.max_positive_db)
            for stem in range(STEM_COUNT):
                content = np.asarray(stem_content_bands[stem], dtype=np.float64)
                if content.size != BAND_COUNT:
                    continue
                peak = float(np.max(content))
                presence = np.clip((content - (peak - 24.0)) / 24.0, 0.0, 1.0)
                target_curve = applied_curve * presence
                for band in range(BAND_COUNT):
                    self.stem_db[stem, band] = slew(
                        self.stem_db[stem, band], float(target_curve[band]),
                        dt, 0.65, 0.85)
        else:
            for stem in range(STEM_COUNT):
                for band in range(BAND_COUNT):
                    self.stem_db[stem, band] = slew(
                        self.stem_db[stem, band], 0.0, dt, 0.65, 0.85)

        self.last = {
            "mode": mode,
            "style": style,
            "applied_db": applied.copy(),
            "device_db": device.copy(),
            "masking_db": self.masking_db.copy(),
            "stem_db": self.stem_db.copy(),
            "exposure": exposure,
            "stale": False,
            "voice": voice,
            "level_dbfs": self.level,
            "noise_mask_depth": self.noise_mask_depth,
            "keyboard_mask_depth": self.keyboard_mask_depth,
            "anticipating_ms": self.anticipating_ms,
        }
        return self.last
