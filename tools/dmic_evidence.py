#!/usr/bin/env python3
"""Edge2 PDM DMIC -> conversation evidence producer (v2, on-board sensing).

The Node exposes `POST /v1/evidence`; this process is its ears.  It captures
the on-board PDM microphone array at 48 kHz (the firmware's analysis rate),
computes the same 24 log-spaced band levels with a Goertzel bank, runs the
sound-event rules ported from `main/sound_event.hpp` (quiet/speech/keyboard/
wind + hold state machine), classifies the environment, and posts one
control-rate evidence frame every 100 ms.

Event semantics carried to the policy:
  * `speech` sets `conversation_cue`/`voice_likelihood` (the existing
    conversation path starts its predicted duck);
  * `keyboard`/`wind` set `event_blocked` (neutral in the confirmation window,
    exactly like the firmware's event filtering);
  * `quiet` leaves everything neutral.

`own_voice` stays 0.0: there is no reliable self-voice detector on this board.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import wave
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

CHANNELS = 2
SAMPLE_WIDTH = 2
RATE = 48_000
FRAME_MS = 100
# 固件口径（features.py）：每 100 ms 决策一次，但分析帧只有 2560 样本（53.3 ms）。
# 用 4800 样本帧会让 Goertzel 每 bin 的噪声能量系统性偏低，模型输入变成分布外。
ANALYSIS_SAMPLES = 2560
CHUNK_SAMPLES = RATE * FRAME_MS // 1000  # 4800 samples read per tick
BAND_CENTER_HZ = np.array([
    70.7, 141.4, 244.9, 346.4, 451.7, 566.8, 696.5, 841.7,
    996.8, 1171.3, 1370.0, 1595.0, 1854.7, 2154.1, 2503.3,
    2916.3, 3414.7, 4036.1, 4829.1, 5824.1, 7020.0, 8552.8,
    10677.1, 13638.2,
], dtype=np.float64)
BAND_COUNT = 24
VOCAL_BANDS = (BAND_CENTER_HZ >= 300.0) & (BAND_CENTER_HZ <= 4000.0)
WINDOW_FRAMES = 10
FLOOR_DOWN = 0.25
FLOOR_UP = 0.002
FLOOR_INIT_FRAMES = 10
EVENT_CLASSES = ("quiet", "speech", "keyboard", "wind", "artifact")
HOLD_KEYBOARD_S = 0.8
HOLD_WIND_S = 2.0
HOLD_SPEECH_S = 0.8
HOLD_ARTIFACT_S = 2.0
# Rule thresholds — same values as main/sound_event.hpp Config.
KEYBOARD_LOW_SHARE = 0.55
KEYBOARD_STD_MID = 4.0
KEYBOARD_CENT = 600.0
KEYBOARD_SLOPE = -1.35
KEYBOARD_CENT_ESCAPE = 800.0
WIND_SLOPE = -1.45
WIND_VOCAL_PEAK = 0.45
WIND_D_LO = 16.0
WIND_RANGE_LO = 14.0
SPEECH_VOCAL_PEAK = 0.45
# 9-class listening context model (ContextNet MLP 92-32-16-9, v0.2-s3), constants
# mirrored from experiments/listening-model/scripts/features.py.
GATE_FLOOR_DBFS = -72.0
GATE_FULL_DBFS = -50.0
NOISE_FLATNESS_CEILING = 0.45
MAX_VOCAL_PEAKINESS = 0.72
MIN_VOCAL_RATIO = 0.34
VOCAL_RATIO_SPAN = 0.36
FLOOR_DBFS = -95.0
MIN_EXCITED_VOCAL_BANDS = 5
EXCITED_BAND_SOFTNESS = 2.0
FLATNESS_SOFTNESS = 0.10
PEAK_SOFTNESS = 0.08
CONTEXT_WINDOW_FRAMES = 20
DEFAULT_MODEL_PATH = "models/listening_v0.2.json"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def frame_voice_likelihood(vocal_ratio: float, peakiness: float, flatness: float,
                           excited: float, level_dbfs: float) -> float:
    band_term = clamp01((vocal_ratio - MIN_VOCAL_RATIO) / VOCAL_RATIO_SPAN)
    peak_term = clamp01((MAX_VOCAL_PEAKINESS - peakiness) / PEAK_SOFTNESS)
    noise_term = clamp01((NOISE_FLATNESS_CEILING - flatness) / FLATNESS_SOFTNESS)
    spread_term = clamp01((excited - (MIN_EXCITED_VOCAL_BANDS - EXCITED_BAND_SOFTNESS))
                          / EXCITED_BAND_SOFTNESS)
    level_gate = clamp01((level_dbfs - GATE_FLOOR_DBFS) / (GATE_FULL_DBFS - GATE_FLOOR_DBFS))
    return band_term * peak_term * noise_term * spread_term * level_gate


def periodicity(mono: np.ndarray, rate: int) -> float:
    """Normalized autocorrelation peak over 2-12 ms lags (F0 83-500 Hz).

    Edge2 DMIC measured 2026-11-09: near speech 0.35-0.42, blowing 0.18-0.25,
    room noise 0.28-0.33.  This is the feature the CoreS3 rule set never had;
    on this mic vocal_peak does not separate speech from wind.
    """
    x = mono - mono.mean()
    lo = max(1, int(rate * 0.002))
    hi = min(len(x) - 1, int(rate * 0.012))
    step = max(1, rate // 6000)
    best = 0.0
    for lag in range(lo, hi + 1, step):
        a = x[:-lag]
        b = x[lag:]
        denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b))) + 1e-12
        value = float(np.dot(a, b)) / denom
        if value > best:
            best = value
    return best


def band_flatness(band_dbfs: np.ndarray) -> float:
    """Flatness of the 300-4000 Hz vocal bands only (1/f room tilt excluded)."""
    power = np.power(10.0, band_dbfs[VOCAL_BANDS] / 10.0)
    return float(np.exp(np.mean(np.log(power + 1e-20))) / (np.mean(power) + 1e-20))


def edge2_voice(pitch: float, vflat: float, level_dbfs: float) -> float:
    """Speech likelihood for the Edge2 PDM array (2026-11-09 calibration).

    Replaces the CoreS3 product formula: near-field speech carries strong LF
    (mouth proximity) so vocal_ratio collapses and the old formula inverted the
    answer (near speech 0.16, blowing 0.38).  Pitch periodicity + vocal-band
    flatness + level gate separate the classes on this mic.
    """
    pitch_term = clamp01((pitch - 0.22) / 0.10)
    tone_term = clamp01((0.18 - vflat) / 0.05)
    level_gate = clamp01((level_dbfs + 66.0) / 8.0)
    return pitch_term * tone_term * level_gate


class ListeningModel:
    """ContextNet MLP 92-32-16-9; weights ported verbatim from main/listening_model.h."""

    def __init__(self, path: str) -> None:
        data = json.load(open(path, encoding="utf-8"))
        if data["feature_count"] != 92 or data["class_count"] < 2:
            raise SystemExit("unexpected listening model shape")
        self.classes = list(data["classes"])
        self.w1 = np.array(data["w1"], dtype=np.float64).reshape(data["hidden1"], 92)
        self.b1 = np.array(data["b1"], dtype=np.float64)
        self.w2 = np.array(data["w2"], dtype=np.float64).reshape(data["hidden2"], data["hidden1"])
        self.b2 = np.array(data["b2"], dtype=np.float64)
        self.w3 = np.array(data["w3"], dtype=np.float64).reshape(data["class_count"], data["hidden2"])
        self.b3 = np.array(data["b3"], dtype=np.float64)

    def predict(self, vector: np.ndarray) -> tuple[str, float]:
        h1 = np.maximum(self.w1 @ vector + self.b1, 0.0)
        h2 = np.maximum(self.w2 @ h1 + self.b2, 0.0)
        logits = self.w3 @ h2 + self.b3
        shifted = logits - logits.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        index = int(np.argmax(probs))
        return self.classes[index], float(probs[index])


def context_window_vector(frames: list[dict]) -> np.ndarray:
    """92-dim window vector, mirrored from features.py::window_vector (motion=None, when=None)."""
    bands = np.stack([frame["band_dbfs"] for frame in frames])
    times = np.array([frame["t"] for frame in frames], dtype=np.float64)
    level = np.array([frame["broadband"] for frame in frames], dtype=np.float64)
    vl = np.array([frame["voice"] for frame in frames], dtype=np.float64)
    n = len(frames)
    band_mean = bands.mean(axis=0)
    band_std = bands.std(axis=0)
    if n >= 2:
        t = times - times.mean()
        denom = float(np.sum(t * t)) or 1.0
        band_slope = (bands * t[:, None]).sum(axis=0) / denom
    else:
        band_slope = np.zeros(BAND_COUNT)
    active = vl > 0.55
    transitions = float(np.count_nonzero(active[1:] != active[:-1])) if n > 1 else 0.0
    steps = float(np.count_nonzero(np.abs(np.diff(level)) > 3.0)) if n > 1 else 0.0
    flatness = np.array([frame["flatness"] for frame in frames])
    peakiness = np.array([frame["peakiness"] for frame in frames])
    vocal_ratio = np.array([frame["vocal_ratio"] for frame in frames])
    excited = np.array([frame["excited"] for frame in frames])
    hour_frac, dow = 0.5, 2  # features.py when=None baseline (12:00, Wednesday)
    return np.concatenate([
        band_mean, band_std, band_slope,
        [level.mean(), level.std(), level.min(), level.max()],
        [vl.mean(), vl.std(), float(active.mean()), transitions],
        [flatness.mean(), peakiness.mean(), vocal_ratio.mean(), excited.mean()],
        [steps],
        [0.0, 0.0],
        [math.sin(2 * math.pi * hour_frac), math.cos(2 * math.pi * hour_frac),
         math.sin(2 * math.pi * dow / 7.0), math.cos(2 * math.pi * dow / 7.0),
         1.0 if dow >= 5 else 0.0],
    ])


class ContinuousWavWriter:
    """Continuously write fixed-length WAV segments (labels added afterwards)."""

    def __init__(self, directory: str, rate: int, channels: int, segment_seconds: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rate = rate
        self.channels = channels
        self.segment_frames = rate * segment_seconds
        self._handle = None
        self._frames = 0
        self._started = 0.0
        self._open()

    def _open(self) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.directory / f"{stamp}-seg.wav"
        self._handle = wave.open(str(path), "wb")
        self._handle.setnchannels(self.channels)
        self._handle.setsampwidth(2)
        self._handle.setframerate(self.rate)
        self._frames = 0
        self._started = time.monotonic()

    def write(self, pcm: bytes) -> None:
        if self._handle is None:
            return
        self._handle.writeframesraw(pcm)
        self._frames += len(pcm) // (self.channels * 2)
        if self._frames >= self.segment_frames:
            self._close()

    def _close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        except OSError:
            pass
        self._handle = None

    def close(self) -> None:
        self._close()


class AudioBroadcaster:
    """16 kHz mono S16LE PCM over TCP, one 100 ms block per broadcast."""

    def __init__(self, port: int) -> None:
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", port))
        self.server.listen(4)
        self.server.settimeout(1.0)
        threading.Thread(target=self._accept_loop, name="audio-accept", daemon=True).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                client, _ = self.server.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self._clients.append(client)
            except OSError:
                continue

    def broadcast(self, data: bytes) -> None:
        with self._lock:
            dead = []
            for client in self._clients:
                try:
                    client.sendall(data)
                except OSError:
                    dead.append(client)
            for client in dead:
                try:
                    client.close()
                except OSError:
                    pass
                self._clients.remove(client)


def frame_bytes(frames: int) -> int:
    return frames * CHANNELS * SAMPLE_WIDTH


def read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def mic_coherence(block: np.ndarray, rate: int) -> tuple[float, float]:
    """Inter-channel correlation + delay for the PDM pair.

    A near-field talker reaches both mics coherently with a small, stable delay;
    a street/diffuse field decorrelates.  This is the Edge2 counterpart of the
    firmware's mic_diff / mic_coherence fields.
    """
    if block.shape[1] < 2:
        return 0.0, 0.0
    left = block[:, 0].astype(np.float64)
    right = block[:, 1].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    left_energy = float(np.sqrt(np.sum(left * left))) + 1e-12
    right_energy = float(np.sqrt(np.sum(right * right))) + 1e-12
    corr = float(np.sum(left * right) / (left_energy * right_energy))
    # Delay via cross-correlation near zero lag (+-0.5 ms).
    max_lag = max(1, int(rate * 0.0005))
    step = 8  # 128 us resolution at 48 kHz; enough to separate near/far
    best_lag, best_value = 0, -1.0
    for lag in range(-max_lag, max_lag + 1, step):
        if lag >= 0:
            value = float(np.sum(left[lag:] * right[: len(right) - lag]))
        else:
            value = float(np.sum(left[: len(left) + lag] * right[-lag:]))
        if value > best_value:
            best_lag, best_value = lag, value
    return corr, best_lag / float(rate)


def impulse_class(mono: np.ndarray, rate: int) -> tuple[str, float]:
    """Short-impact vs sustained-rub detection from the 10 ms envelope.

    Measured on Edge2 DMIC (2026-11-09): knock/friction both show env_trans
    > 11 dB and > 50 spikes per 100 ms frame, while keyboard/wind show < 6.
    Duration separates them: a knock is one short burst, friction keeps the
    envelope high for many 10 ms cells.
    """
    cell = max(1, rate // 100)
    usable = (len(mono) // cell) * cell
    if usable < cell * 4:
        return "none", 0.0
    cells = mono[:usable].reshape(-1, cell).astype(np.float64)
    env = np.sqrt(np.mean(cells * cells, axis=1)) + 1e-9
    env_db = 20.0 * np.log10(env)
    median = float(np.median(env_db))
    hi = env_db > median + 12.0
    spikes = int(np.count_nonzero(hi))
    if spikes < 1:
        return "none", float(env_db.max() - median)
    # longest consecutive run of high cells (10 ms each)
    longest = run = 0
    for flag in hi:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return "impulse", float(env_db.max() - median)


def goertzel_bands(mono: np.ndarray, freqs: np.ndarray, rate: int) -> np.ndarray:
    """24 log-spaced band levels, same formula as the firmware's goertzel_dbfs()."""
    n = len(mono)
    window = np.hanning(n)
    weighted = mono.astype(np.float64) * window
    seconds = np.arange(n) / float(rate)
    phases = np.exp(-2j * np.pi * np.outer(freqs, seconds))
    spectrum = phases @ weighted
    power = np.maximum(spectrum.real ** 2 + spectrum.imag ** 2, 1e-12)
    amplitude = 2.0 * np.sqrt(power) / max(n, 1)
    return 20.0 * np.log10(np.maximum(amplitude, 1e-6))


def frame_descriptors(band_dbfs: np.ndarray) -> tuple[float, float, float, float, float, float]:
    power = np.power(10.0, band_dbfs / 10.0)
    total = max(float(power.sum()), 1e-20)
    centroid = float((power * BAND_CENTER_HZ).sum() / total)
    flatness = float(np.exp(np.log(np.maximum(power, 1e-20)).mean()) / (total / BAND_COUNT))
    vocal_power = power[VOCAL_BANDS]
    vocal_sum = max(float(vocal_power.sum()), 1e-20)
    vocal_ratio = float(vocal_sum / total)
    vocal_peak = float(vocal_power.max() / vocal_sum)
    index = np.arange(BAND_COUNT, dtype=np.float64)
    centered = index - index.mean()
    slope = float((centered * (band_dbfs - band_dbfs.mean())).sum() / (centered * centered).sum())
    low_share = float(power[:5].sum() / total)
    return centroid, flatness, vocal_ratio, vocal_peak, slope, low_share


class FloorTracker:
    def __init__(self) -> None:
        self.base: np.ndarray | None = None
        self.level = -120.0
        self._init: list[np.ndarray] = []
        self._levels: list[float] = []

    def step(self, band_dbfs: np.ndarray, broadband_dbfs: float, valid: bool) -> None:
        if not valid:
            return
        if self.base is None:
            self._init.append(band_dbfs.astype(np.float64))
            self._levels.append(float(broadband_dbfs))
            if len(self._init) >= FLOOR_INIT_FRAMES:
                self.base = np.median(np.stack(self._init), axis=0)
                self.level = float(np.median(self._levels))
            return
        delta = band_dbfs - self.base
        rate = np.where(delta < 0.0, FLOOR_DOWN, FLOOR_UP)
        self.base = self.base + rate * delta
        dlevel = float(broadband_dbfs) - self.level
        self.level += (FLOOR_DOWN if dlevel < 0.0 else FLOOR_UP) * dlevel

    @property
    def ready(self) -> bool:
        return self.base is not None


class EventStateMachine:
    def __init__(self) -> None:
        self.state = "quiet"
        self.hold_until = -1.0

    def update(self, raw: str, now_s: float) -> str:
        if self.state == "artifact" and raw != "artifact":
            self.hold_until = -1.0
        if raw == "keyboard":
            self.state = "keyboard"
            self.hold_until = now_s + HOLD_KEYBOARD_S
        elif raw == "wind":
            if not (self.state == "keyboard" and now_s <= self.hold_until):
                self.state = "wind"
                self.hold_until = now_s + HOLD_WIND_S
        elif raw == "speech":
            if now_s > self.hold_until:
                self.state = "speech"
                self.hold_until = now_s + HOLD_SPEECH_S
        elif raw == "quiet":
            if now_s > self.hold_until:
                self.state = "quiet"
        elif raw == "artifact":
            self.state = "artifact"
            self.hold_until = now_s + HOLD_ARTIFACT_S
        return self.state


def window_features(base: np.ndarray, base_level: float,
                    frames: list[dict]) -> dict[str, float]:
    bands = np.stack([frame["band_dbfs"] for frame in frames])
    rel = bands - base
    d_lo = float(rel[:, :5].mean())
    d_mid = float(rel[:, 5:16].mean())
    d_hi = float(rel[:, 16:].mean())
    env_lo = bands[:, :5].mean(axis=1)
    env_mid = bands[:, 5:16].mean(axis=1)
    cursor = np.zeros(7)
    for frame in frames:
        centroid, flatness, vocal_ratio, vocal_peak, slope, low_share = frame["desc"]
        cursor += np.array([centroid, flatness, vocal_ratio, frame["broadband"], vocal_peak,
                            slope, low_share])
    cursor /= len(frames)
    return {
        "d_lo": d_lo,
        "d_mid": d_mid,
        "d_hi": d_hi,
        "mid_lo": d_mid - d_lo,
        "hi_mid": d_hi - d_mid,
        "cent": float(cursor[0]),
        "flat": float(cursor[1]),
        "vr": float(cursor[2]),
        "std_lo": float(env_lo.std()),
        "range_lo": float(env_lo.max() - env_lo.min()),
        "std_mid": float(env_mid.std()),
        "level_rel": float(cursor[3] - base_level),
        "vocal_peak": float(cursor[4]),
        "slope": float(cursor[5]),
        "low_share": float(cursor[6]),
    }


def raw_classify(features: dict[str, float], voice: float = 0.0,
                 pitch: float = 0.0, vflat: float = 0.0) -> str:
    """Edge2-calibrated event rules (2026-11-09, second calibration).

    Measured on the board's own DMIC clips:
      * near speech: periodicity 0.35-0.42, vocal-band flatness 0.06-0.14
      * blowing:     periodicity 0.18-0.25, flatness 0.22-0.24, LF share > 0.7
      * quiet room:  level_rel < 5 dB, flatness 0.20-0.24
    Speech and wind are gated by level_rel > 6 dB so the floor never becomes an
    event.  Keyboard/knock/friction keep the 2026-11-09 first calibration (they
    are transient detectors; they no longer override speech/wind).
    """
    if features["level_rel"] > 4.0 and pitch >= 0.30 and vflat <= 0.16 and voice >= 0.5:
        return "speech"
    if (features["level_rel"] > 8.0 and pitch < 0.26 and vflat > 0.18
            and features["low_share"] > 0.70):
        return "wind"
    if features["d_mid"] > 5.0 and features["std_mid"] > 6.0:
        return "keyboard"
    if features["d_hi"] > 9.0 and features["std_mid"] < 8.0:
        return "knock"
    if features["d_mid"] < -10.0 or (features["std_mid"] > 4.5 and features["d_mid"] < -5.0):
        return "friction"
    return "quiet"


def environment_label(event: str, level_dbfs: float) -> str:
    if event == "wind":
        return "NOISY"
    if level_dbfs >= -35.0:
        return "LOUD"
    if event == "speech":
        return "VOICE"
    if level_dbfs >= -52.0:
        return "NOISY"
    return "QUIET"


def post_evidence(url: str, frame: dict) -> bool:
    event = frame["event"]
    payload = {
        "speech_active": event == "speech",
        "voice_likelihood": round(float(frame["voice_likelihood"]), 4),
        # 对话让位用逐帧语音似然，而不是事件状态机的 speech 标签：状态机有 0.8 s
        # hold，街道/电视里的偶发人声会被放大成连续 speech（2026-11-09 现场：
        # 播街道声时媒体被压到 60.7%）。voice_likelihood 是乘积，街道声实测 0.04，
        # 近场说话 0.73，用它做门控不会误让位。
        # 2026-11-10 现场："出现说话媒体就被压低、除非我参与对话否则应该做掩蔽"。
        # 环境里的 speech ≠ 用户参与对话。对话让位是 **Mac 侧门**（声纹 self / LLM 确认，
        # D42）才做的动作；板端给 conversation_cue 会让任何背景人声把 policy 顶到 ENGAGED
        # → 媒体压 60% + 包络压平（掩蔽反而停了），而且键盘/风事件会阻止 ENGAGED 释放。
        # 板端只提供 noise/voice/event 证据：voice 驱动 D41 掩蔽，对话让位交给 Mac。
        "conversation_cue": 0.0,
        "own_voice": 0.0,
        "level_dbfs": frame["level_dbfs"],
        "event_blocked": event in ("keyboard", "wind"),
        "dialogue_veto": False,
        "event": event,
        "environment": frame["environment"],
        "context": frame["context"],
        "context_score": frame["context_score"],
        "dt_ms": FRAME_MS,
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=1.0) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="hw:2,0", help="ALSA capture device")
    parser.add_argument("--rate", type=int, default=RATE)
    parser.add_argument("--channels", type=int, default=CHANNELS)
    parser.add_argument("--frame-ms", type=int, default=FRAME_MS)
    parser.add_argument("--evidence-url", default="http://127.0.0.1:8787/v1/evidence")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 = run forever")
    parser.add_argument("--dry-run", action="store_true", help="print frames, do not post")
    parser.add_argument("--status-file", default="/tmp/dmic-latest.json",
                        help="write the latest frame here for other consumers")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="9-class listening context model (JSON ported from listening_model.h)")
    parser.add_argument("--audio-port", type=int, default=9101,
                        help="TCP port for the 16 kHz mono PCM stream (0 = disabled)")
    parser.add_argument("--record-dir", default="",
                        help="continuously write 60 s WAV segments here (labels after the fact)")
    args = parser.parse_args()
    chunk_samples = args.rate * args.frame_ms // 1000
    analysis_samples = min(ANALYSIS_SAMPLES, chunk_samples)
    chunk = frame_bytes(chunk_samples)
    if chunk_samples < 512:
        raise SystemExit("frame too short")

    command = ["arecord", "-D", args.device, "-f", "S16_LE", "-r", str(args.rate),
               "-c", str(args.channels), "-t", "raw", "--quiet", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    recorder = None
    if args.record_dir:
        recorder = ContinuousWavWriter(args.record_dir, args.rate, args.channels, 60)
    broadcaster = None
    if args.audio_port > 0 and args.rate % 3 == 0:
        try:
            broadcaster = AudioBroadcaster(args.audio_port)
        except OSError:
            broadcaster = None
    tracker = FloorTracker()
    machine = EventStateMachine()
    model = None
    try:
        model = ListeningModel(args.model)
    except (OSError, ValueError, KeyError):
        model = None  # context stays "unknown" instead of faking a label
    history: list[dict] = []
    impulse_frames: list[bool] = []
    impulse_hold = "none"
    impulse_hold_until = 0.0
    started = time.monotonic()
    posted = 0
    failures = 0
    try:
        while True:
            if args.seconds > 0 and time.monotonic() - started >= args.seconds:
                break
            raw = read_exact(process.stdout, chunk)
            if len(raw) < chunk:
                break
            block = np.frombuffer(raw, dtype="<i2").reshape(-1, args.channels)
            mono_full = (block.mean(axis=1).astype(np.float32) / 32768.0)
            if recorder is not None:
                recorder.write(block.astype("<i2", copy=False).tobytes())
            if broadcaster is not None:
                pcm16 = (mono_full.reshape(-1, 3).mean(axis=1) * 32767.0).astype("<i2").tobytes()
                broadcaster.broadcast(pcm16)
            mono = mono_full[-analysis_samples:]
            rms = float(np.sqrt(np.mean(mono * mono)) + 1e-12)
            level_dbfs = 20.0 * math.log10(rms)
            clipped = bool(np.any(np.abs(mono) >= 0.999))
            valid = not clipped and math.isfinite(level_dbfs)
            band_dbfs = goertzel_bands(mono, BAND_CENTER_HZ, args.rate)
            # Use the full 100 ms chunk, not the 53 ms analysis frame: a 10 ms
            # impact is averaged away by the shorter window (measured 15 vs 25 hits).
            impulse_raw, impulse_score = impulse_class(mono_full, args.rate)
            impulse_frames.append(impulse_raw == "impulse")
            del impulse_frames[:-20]
            impulse = "none"
            if impulse_raw == "impulse":
                # Rate over the last 2 s separates repeated rubbing from isolated
                # impacts: measured on 2026-11-09 clips, friction ~0.5, knock ~0.06.
                recent = impulse_frames[-20:]
                rate = sum(recent) / len(recent)
                impulse = "friction" if rate >= 0.5 else "knock"
            if impulse_raw != "none":
                recent_rate = sum(impulse_frames[-20:]) / max(1, len(impulse_frames[-20:]))
                print(f"impulse {impulse_raw} -> {impulse} rate={recent_rate:.2f} "
                      f"score={impulse_score:.1f} level={level_dbfs:.1f}",
                      file=sys.stderr, flush=True)
            coherence, mic_delay = mic_coherence(block[-analysis_samples:], args.rate)
            descriptors = frame_descriptors(band_dbfs)
            tracker.step(band_dbfs, level_dbfs, valid)
            now_s = time.monotonic() - started
            quiet_ref = -95.0
            excited = float(np.count_nonzero(band_dbfs[VOCAL_BANDS] > quiet_ref))
            flatness = float(clamp01(descriptors[1]))
            peakiness = float(descriptors[3])
            vocal_ratio = float(descriptors[2])
            pitch = periodicity(mono, args.rate)
            vflat = band_flatness(band_dbfs)
            voice = edge2_voice(pitch, vflat, level_dbfs)
            history.append({"band_dbfs": band_dbfs, "broadband": level_dbfs, "valid": valid,
                            "t": now_s, "voice": voice, "flatness": flatness,
                            "peakiness": peakiness, "vocal_ratio": vocal_ratio,
                            "excited": excited, "desc": descriptors,
                            "pitch": pitch, "vflat": vflat})
            del history[:-CONTEXT_WINDOW_FRAMES]
            event = "quiet"
            recent = history[-WINDOW_FRAMES:]
            if tracker.ready and len(recent) == WINDOW_FRAMES and all(f["valid"] for f in recent):
                features = window_features(tracker.base, tracker.level, recent)
                pitch_med = float(np.median([f["pitch"] for f in recent]))
                vflat_med = float(np.median([f["vflat"] for f in recent]))
                event = machine.update(
                    raw_classify(features, voice, pitch_med, vflat_med), now_s)
            elif not valid:
                event = machine.update("artifact", now_s)
            if impulse in ("knock", "friction"):
                impulse_hold = impulse
                impulse_hold_until = now_s + 1.5
            elif now_s > impulse_hold_until:
                impulse_hold = "none"
            # Keyboard typing also fires the transient detector; keyboard wins so
            # typing is not reported as knock/friction (2026-11-09 field report).
            # 2026-11-09 second calibration: transient detector may fill a silent
            # frame, but never relabel a speech/wind frame (blowing gusts fire it).
            if impulse_hold != "none" and event not in ("keyboard", "speech", "wind"):
                event = impulse_hold
            environment = environment_label(event, level_dbfs)
            context = "unknown"
            context_score = 0.0
            vector = None
            if model is not None and len(history) == CONTEXT_WINDOW_FRAMES and all(
                    f["valid"] for f in history):
                vector = context_window_vector(history)
                # Rule layer first: sound-event labels are physical and mic-robust, while the
                # v0.2-s3 model was trained on CoreS3 scenes and drifts on Edge2 PDM data.
                # v0.5-edge2 has a native `quiet` class (Edge2 clips), so the model
                # owns all five contexts; no rule fallback needed.
                context, context_score = model.predict(vector)
            frame = {
                "event": event,
                "environment": environment,
                "context": context,
                "context_score": round(context_score, 4),
                "vector": [round(float(value), 4) for value in vector] if vector is not None else None,
                "voice_likelihood": round(voice, 4),
                "impulse": impulse,
                "impulse_score": round(impulse_score, 1),
                "mic_coherence": round(coherence, 3),
                "mic_delay_us": round(mic_delay * 1e6, 1),
                "level_dbfs": round(level_dbfs, 2),
                "band_dbfs": [round(float(value), 1) for value in band_dbfs],
                "posted": posted,
                "post_failures": failures,
                "ts_ms": int(now_s * 1000),
            }
            if args.dry_run:
                print(json.dumps(frame, ensure_ascii=False), flush=True)
            elif post_evidence(args.evidence_url, frame):
                posted += 1
            else:
                failures += 1
            try:
                tmp_path = args.status_file + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    json.dump(frame, handle, ensure_ascii=False)
                os.replace(tmp_path, args.status_file)
            except OSError:
                pass
    finally:
        if recorder is not None:
            recorder.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
    if not args.dry_run:
        print(json.dumps({"event": "stopped", "posted": posted, "post_failures": failures}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
