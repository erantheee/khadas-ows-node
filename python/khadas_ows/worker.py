#!/usr/bin/env python3
"""CPU/NPU worker for the fixed-shape MaskNetwork boundary (engineering probe).

Streaming contract
------------------
The model is exported with a fixed input shape ``[1, 2, 1025, 345]``: 345 STFT
frames of a 4.0 s window at n_fft=2048, hop=512, 44.1 kHz.  An arbitrary PCM
packet cannot be fed to the model directly, so :class:`StreamingWindowManager`
slices the packet's STFT frames into windows of 345 frames, zero-pads the final
window, and runs exactly one fixed-shape inference per window.

Latency: the first output block is only complete after a full window has been
observed, i.e. ``345 * 512 / 44100 = 4.005 s``.  That is acceptable for this
throughput probe but is **not** a product latency claim; real-time separation
needs a smaller fixed window and a re-exported model.

The reply always has exactly as many frames as the request: masks are assembled
on the packet's own frame grid, applied to the complex STFT, and reconstructed
with one masked ISTFT (itself an overlap-add), trimmed to the packet length.  The
returned stem is peak-matched to the source packet -- an engineering probe
convention, not a gain or loudness policy.

Backends
--------
``--backend torch``     CPU PyTorch checkpoint (default for ``--mode cpu``)
``--backend rknnlite``  RKNN-Toolkit-Lite2 on an RK3588/Edge2 board (default for ``--mode npu``)
``--backend stub``      deterministic in-process backend for tests; no torch/rknnlite needed

If the rknnlite backend cannot be imported or initialised, the worker prints a
clear error and exits nonzero so the C++ core falls back to passthrough.  No
worker exit code should ever be interpreted as evidence of on-board NPU success;
that requires ``actual_mode=npu`` and a board soak test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import strategy as role_strategy
from . import tuning as board_tuning

HEADER = struct.Struct("<IIQIHHi")
MAGIC = 0x4B4F5753
SAMPLE_RATE = 44100
N_FFT = 2048
HOP = 512
FREQ_BINS = N_FFT // 2 + 1
MODEL_CHANNELS = 2
MODEL_FRAMES = 345  # fixed time axis of the exported model
STEMS = 4  # masks[0] = vocals for this probe
WINDOW_SECONDS = MODEL_FRAMES * HOP / SAMPLE_RATE  # 4.005 s


class WorkerError(RuntimeError):
    """Any unrecoverable worker configuration/processing failure."""


class BackendUnavailable(WorkerError):
    """The requested model backend is not installed on this host."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _periodic_hann(n_fft: int = N_FFT) -> np.ndarray:
    """Hann window matching ``torch.hann_window(n_fft)`` (periodic, not symmetric)."""
    indices = np.arange(n_fft, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * indices / n_fft)).astype(np.float32)


_WINDOW = _periodic_hann()


def stft(x: np.ndarray) -> np.ndarray:
    """Centre-padded STFT matching ``torch.stft(center=True)`` geometry.

    ``x`` is ``[channels, samples]``; returns complex64 ``[channels, 1025, frames]``
    with ``frames = 1 + samples // HOP`` (so 4.0 s -> 345 frames).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise WorkerError(f"stft expects [channels, samples], got shape {x.shape}")
    channels, samples = x.shape
    pad = N_FFT // 2
    if samples >= N_FFT:
        padded = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    else:
        # Packets shorter than one FFT frame cannot reflect-pad; zero-pad instead.
        padded = np.pad(x, ((0, 0), (pad, pad)), mode="constant")
    frames = 1 + samples // HOP
    frame_offsets = HOP * np.arange(frames, dtype=np.int64)
    indices = frame_offsets[:, None] + np.arange(N_FFT, dtype=np.int64)[None, :]
    windowed = padded[:, indices] * _WINDOW  # [channels, frames, N_FFT]
    spectrum = np.fft.rfft(windowed, axis=-1)  # [channels, frames, 1025]
    return spectrum.transpose(0, 2, 1).astype(np.complex64)


def istft(spectrum: np.ndarray, length: int) -> np.ndarray:
    """Inverse of :func:`stft` (centre=True), trimmed to exactly ``length`` samples."""
    spectrum = np.asarray(spectrum)
    if spectrum.ndim != 3:
        raise WorkerError(f"istft expects [channels, 1025, frames], got shape {spectrum.shape}")
    channels, bins, frames = spectrum.shape
    if bins != FREQ_BINS:
        raise WorkerError(f"istft expects {FREQ_BINS} frequency bins, got {bins}")
    frames_time = np.fft.irfft(spectrum.transpose(0, 2, 1).astype(np.complex128), n=N_FFT, axis=-1)
    frames_time *= _WINDOW  # synthesis window; analysis window was applied in stft()
    total = (frames - 1) * HOP + N_FFT
    signal = np.zeros((channels, total), dtype=np.float64)
    weight = np.zeros(total, dtype=np.float64)
    window_sq = _WINDOW.astype(np.float64) ** 2
    for index in range(frames):
        start = index * HOP
        signal[:, start : start + N_FFT] += frames_time[:, index, :]
        weight[start : start + N_FFT] += window_sq
    signal /= np.maximum(weight, 1e-10)[None, :]
    signal = signal[:, N_FFT // 2 : N_FFT // 2 + length]
    if signal.shape[-1] < length:  # defensive; geometry above always covers `length`
        signal = np.pad(signal, ((0, 0), (0, length - signal.shape[-1])))
    return signal.astype(np.float32)


# --------------------------------------------------------------------------- #
# Model backends
# --------------------------------------------------------------------------- #
def _canonical_masks(value, window_frames: int) -> np.ndarray:
    """Normalise a backend output to float32 ``[1, STEMS, 1025, window_frames]``."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    expected = (1, STEMS, FREQ_BINS, window_frames)
    if array.shape == expected:
        return array
    if array.shape == (1, FREQ_BINS, window_frames, STEMS):
        return np.ascontiguousarray(array.transpose(0, 3, 1, 2))
    raise WorkerError(f"model returned mask shape {array.shape}; expected {expected}")


class StubBackend:
    """Deterministic analytic mask backend. Used by tests; never claims model quality."""

    name = "stub"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, magnitude: np.ndarray) -> np.ndarray:
        magnitude = np.asarray(magnitude, dtype=np.float32)
        frames_used = int(np.count_nonzero(np.any(magnitude[0] != 0.0, axis=(0, 1))))
        self.calls.append({"shape": tuple(magnitude.shape), "frames_used": frames_used})
        # Fixed, input-independent priors shaped over frequency: pure numpy, deterministic,
        # softmax-normalised so masks sum to 1 along the stem axis.
        frequency = np.arange(FREQ_BINS, dtype=np.float32)[None, :]  # [1, 1025]
        stems = np.arange(STEMS, dtype=np.float32)[:, None]  # [4, 1]
        priors = np.array([1.2, 0.2, -0.6, -0.8], dtype=np.float32)[:, None]  # [4, 1]
        logits = priors + 0.2 * np.sin(frequency / 53.0 + stems)  # [4, 1025]
        weights = np.exp(logits - logits.max(axis=0, keepdims=True))
        masks = (weights / weights.sum(axis=0, keepdims=True)).astype(np.float32)  # [4, 1025]
        return np.broadcast_to(masks[None, :, :, None], (1, STEMS, FREQ_BINS, magnitude.shape[-1])).copy()


class TorchBackend:
    """CPU PyTorch checkpoint backend (``--mode cpu`` default)."""

    name = "torch"

    def __init__(self, model_path: str) -> None:
        import torch
        from .model import load_legacy_checkpoint

        self._torch = torch
        self._model = load_legacy_checkpoint(model_path)

    def run(self, magnitude: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            masks = self._model(self._torch.from_numpy(np.ascontiguousarray(magnitude)))
        return masks.numpy()


class RknnLiteBackend:
    """RKNN-Toolkit-Lite2 backend for the Edge2 NPU (``--mode npu`` default)."""

    name = "rknnlite"

    def __init__(self, model_path: str) -> None:
        from rknnlite.api import RKNNLite

        self._rknn = RKNNLite()
        result = self._rknn.load_rknn(model_path)
        if result != 0:
            raise BackendUnavailable(
                f"rknnlite.load_rknn({model_path!r}) failed with return code {result}; "
                "the .rknn artifact or the board's RKNPU driver/runtime is not usable"
            )
        try:
            result = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        except TypeError:  # older RKNNLite builds do not accept core_mask
            result = self._rknn.init_runtime()
        if result != 0:
            raise BackendUnavailable(
                f"rknnlite.init_runtime failed with return code {result}; "
                "check that the Edge2 kernel driver and librknnrt versions match RKNN-Toolkit-Lite2"
            )
        # Warm the graph once so the first *audio* packet is not charged the one-off
        # runtime/driver initialisation cost (measured ~0.5-3.5 s on an Edge2, which
        # would otherwise look like a worker timeout and force a passthrough fallback).
        warmup = np.zeros((1, MODEL_CHANNELS, FREQ_BINS, MODEL_FRAMES), dtype=np.float32)
        self.run(warmup)

    def run(self, magnitude: np.ndarray) -> np.ndarray:
        payload = np.ascontiguousarray(magnitude, dtype=np.float32)
        try:
            outputs = self._rknn.inference(inputs=[payload], data_format=["nchw"])
        except TypeError:  # older RKNNLite builds have no data_format keyword
            outputs = self._rknn.inference(inputs=[payload])
        if not outputs:
            raise WorkerError("rknnlite.inference returned no outputs")
        return outputs[0]


def _verify_model_hash(model_path: str | None, expected_sha256: str | None) -> str | None:
    if not expected_sha256:
        return None
    if not model_path:
        raise WorkerError("--model-sha256 / KHADAS_OWS_MODEL_SHA256 given without --model")
    if not Path(model_path).is_file():
        raise WorkerError(f"model not found: {model_path}")
    actual = sha256_file(model_path)
    if actual.lower() != expected_sha256.strip().lower():
        raise WorkerError(f"model sha256 mismatch for {model_path}: expected {expected_sha256} got {actual}")
    return actual


def make_backend(name: str, model_path: str | None, expected_sha256: str | None = None) -> object:
    """Backend factory. Tests inject ``stub``; hardware uses ``torch``/``rknnlite``."""
    if name == "stub":
        _verify_model_hash(model_path, expected_sha256)  # still honour an explicit hash request
        return StubBackend()
    if name == "torch":
        _verify_model_hash(model_path, expected_sha256)
        if not model_path:
            raise WorkerError("--backend torch requires --model <checkpoint.pt>")
        if not Path(model_path).is_file():
            raise WorkerError(f"model not found: {model_path}")
        try:
            return TorchBackend(model_path)
        except WorkerError:
            raise
        except Exception as exc:  # noqa: BLE001 - checkpoint/torch loading errors are user-facing
            raise WorkerError(f"failed to load the CPU checkpoint {model_path}: {exc}") from exc
    if name == "rknnlite":
        # Import first: on hosts without rknnlite (macOS dev, x86_64 CI) this must fail
        # fast and clearly so the C++ core falls back to passthrough.
        try:
            import rknnlite.api  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - any import/init failure is a clean fallback
            raise BackendUnavailable(
                "NPU backend unavailable: RKNN-Toolkit-Lite2 ('rknnlite') is not installed "
                "on this host. rknnlite ships as a Linux aarch64 wheel for the Edge2 board; "
                f"underlying import error: {exc!r}. The CPU path (--mode cpu) still works."
            ) from exc
        _verify_model_hash(model_path, expected_sha256)
        if not model_path:
            raise WorkerError("--backend rknnlite requires --model <model.rknn>")
        if not Path(model_path).is_file():
            raise WorkerError(f"model not found: {model_path}")
        try:
            return RknnLiteBackend(model_path)
        except WorkerError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"failed to initialise the RKNN-Lite2 runtime with {model_path}: {exc}") from exc
    raise WorkerError(f"unknown backend {name!r}; expected one of torch, rknnlite, stub")


# --------------------------------------------------------------------------- #
# Fixed-shape streaming window manager
# --------------------------------------------------------------------------- #
class StreamingWindowManager:
    """Run a fixed-shape [1,2,1025,345] model over arbitrary-length packets."""

    def __init__(
        self,
        backend,
        window_frames: int = MODEL_FRAMES,
        window_hop: int | None = None,
        output_gain: str = "peak",
        output_mode: str = "stem",
        scenario: str = "quiet",
        scenario_start_s: float = 0.0,
        scenario_end_s: float = 1e9,
        strategy_strength: float = 0.6,
        policy: str = "s3",
        evidence_file: str = "/tmp/dmic-latest.json",
        node_status_url: str = "http://127.0.0.1:8787/v1/status",
        eq_depth: float = board_tuning.DEFAULT_DEPTH,
        masking_dose: float = board_tuning.DEFAULT_MASKING_DOSE,
        max_positive_db: float = board_tuning.MAX_POSITIVE_DB,
    ) -> None:
        self.backend = backend
        if output_gain not in ("peak", "unity"):
            raise WorkerError(f"output_gain must be 'peak' or 'unity', got {output_gain!r}")
        if output_mode not in ("stem", "envelope"):
            raise WorkerError(f"output_mode must be 'stem' or 'envelope', got {output_mode!r}")
        if scenario not in role_strategy.SCENARIOS:
            raise WorkerError(f"unknown scenario {scenario!r}; expected {role_strategy.SCENARIOS}")
        self.output_gain = output_gain
        self.output_mode = output_mode
        self.scenario = scenario
        self.scenario_start_s = float(scenario_start_s)
        self.scenario_end_s = float(scenario_end_s)
        self.strategy_strength = float(strategy_strength)
        self.role_tracker = role_strategy.RoleTracker()
        self.envelope = role_strategy.EnvelopeFollower()
        if policy not in ("roles", "s3"):
            raise WorkerError(f"policy must be 'roles' or 's3', got {policy!r}")
        self.policy = policy
        self.evidence_reader = board_tuning.EvidenceReader(evidence_file)
        self.tuner = board_tuning.BoardTuner(
            depth=eq_depth, masking_dose=masking_dose,
            max_positive_db=max_positive_db,
            node_status=board_tuning.NodeStatusReader(node_status_url))
        self.frame_gains = np.zeros(board_tuning.BAND_COUNT, dtype=np.float64)
        self.last_tuning: dict = {"mode": "FLAT", "stale": True}
        # 10 Hz control loop: the tuner decision cadence is decoupled from the 1 s audio
        # packet cadence, so a noise/event change reaches the Mac in ~100 ms instead of
        # waiting for the next 0.9 s hop (user feedback: change latency must match S3).
        self.latest_features: tuple | None = None
        self._features_lock = threading.Lock()
        self._control_started = False
        self.samples_processed = 0
        self.role_history: list[dict] = []
        self.window_frames = int(window_frames)
        self.window_hop = int(window_hop if window_hop is not None else window_frames)
        if self.window_frames < 1:
            raise WorkerError(f"window_frames must be >= 1, got {self.window_frames}")
        if not 1 <= self.window_hop <= self.window_frames:
            raise WorkerError(
                f"window_hop must be in [1, window_frames={self.window_frames}], got {self.window_hop}"
            )
        self.calls: list[dict] = []

    @property
    def latency_seconds(self) -> float:
        return self.window_frames * HOP / SAMPLE_RATE

    def window_starts(self, total_frames: int) -> list[int]:
        return list(range(0, max(total_frames, 1), self.window_hop))

    def masks(self, magnitude: np.ndarray) -> np.ndarray:
        """Assemble ``[1,4,1025,T]`` masks for a ``[channels,1025,T]`` magnitude packet."""
        magnitude = np.asarray(magnitude, dtype=np.float32)
        if magnitude.ndim != 3 or magnitude.shape[0] != MODEL_CHANNELS or magnitude.shape[1] != FREQ_BINS:
            raise WorkerError(
                f"expected magnitude [2,{FREQ_BINS},T], got {tuple(magnitude.shape)}"
            )
        total_frames = magnitude.shape[-1]
        accumulated = np.zeros((STEMS, FREQ_BINS, total_frames), dtype=np.float32)
        weights = np.zeros(total_frames, dtype=np.float32)
        for start in self.window_starts(total_frames):
            window = np.zeros((1, MODEL_CHANNELS, FREQ_BINS, self.window_frames), dtype=np.float32)
            used = min(self.window_frames, total_frames - start)
            window[0, :, :, :used] = magnitude[:, :, start : start + used]
            masks = _canonical_masks(self.backend.run(window), self.window_frames)
            if not np.isfinite(masks).all():
                raise WorkerError(f"model backend returned non-finite masks (window start {start})")
            accumulated[:, :, start : start + used] += masks[0, :, :, :used]
            weights[start : start + used] += 1.0
            self.calls.append(
                {
                    "start": start,
                    "frames_used": used,
                    "window_shape": list(window.shape),
                    "backend": getattr(self.backend, "name", type(self.backend).__name__),
                }
            )
        if np.any(weights == 0):
            raise WorkerError("internal error: uncovered frames in the mask timeline")
        accumulated /= weights[None, None, :]
        return accumulated[None, ...]

    def process(self, pcm: np.ndarray, channels: int) -> np.ndarray:
        """Process one interleaved PCM packet; returns the same number of frames."""
        pcm = np.asarray(pcm, dtype=np.float32)
        if channels < 1:
            raise WorkerError(f"channels must be >= 1, got {channels}")
        frames = pcm.size // channels
        if frames == 0:
            return np.zeros(0, dtype="<f4")
        if channels != MODEL_CHANNELS:
            raise WorkerError(f"the fixed-shape model expects {MODEL_CHANNELS} channels, got {channels}")
        source = pcm[: frames * channels].reshape(-1, channels).T.copy()  # [channels, frames]
        spectrum = stft(source)  # complex64 [2,1025,T]
        magnitude = np.abs(spectrum)
        masks = self.masks(magnitude)  # [1,4,1025,T]
        if self.output_mode == "envelope":
            if self.policy == "s3":
                output = self._s3_output(spectrum, masks, frames)
            else:
                output = self._envelope_output(spectrum, masks, frames)
            self.samples_processed += frames
            return np.ascontiguousarray(output, dtype="<f4").reshape(-1)
        vocals_spec = masks[0, 0].astype(np.complex64) * spectrum
        vocals = istft(vocals_spec, frames)  # [2,frames]
        if self.output_gain == "unity":
            # No per-packet level manipulation: the extracted stem keeps the model's own
            # scale, so the 4 s window boundaries cannot introduce level jumps. Use this for
            # listening/analysis; 'peak' is the historical probe convention.
            output = vocals.T
        else:
            source_peak = float(np.max(np.abs(source)))
            vocal_peak = float(np.max(np.abs(vocals)))
            scale = source_peak / vocal_peak if vocal_peak > 1e-9 else 0.0
            output = (vocals * scale).T
        return np.ascontiguousarray(output, dtype="<f4").reshape(-1)


    def _s3_output(self, spectrum: np.ndarray, masks: np.ndarray, pcm_frames: int) -> np.ndarray:
        """Board-side smart tuner: DMIC noise x separated-content style -> 24-band envelope.

        D39 content curve + D41/D48 masking decisions (sources: main/controller.cpp)
        plus the D56 content x ambient masking matrix (source: systemeq.swift), all
        applied as a time-varying 24-band gain on the remixed music.  The Mac sends
        PCM and plays the result; no tuning decision happens on the Mac.
        """
        stft_frames = int(masks.shape[-1])
        content_bands, shares = self.tuner.content_features(spectrum, masks)
        with self._features_lock:
            self.latest_features = (content_bands, shares)
        self.start_control_loop()
        target = np.asarray(self.tuner.applied_db, dtype=np.float64)
        dt_frame = HOP / SAMPLE_RATE
        gains_db = np.zeros((stft_frames, board_tuning.BAND_COUNT), dtype=np.float64)
        current = self.frame_gains.copy()
        for frame_index in range(stft_frames):
            current = np.array([
                board_tuning.slew(cur, tgt, dt_frame, 0.65, 0.85)
                for cur, tgt in zip(current, target)])
            gains_db[frame_index] = current
        self.frame_gains = current
        bin_linear = np.zeros((stft_frames, FREQ_BINS), dtype=np.float32)
        for frame_index in range(stft_frames):
            bin_linear[frame_index] = 10.0 ** (
                self.tuner.band_map.gains_to_bins(gains_db[frame_index]) / 20.0)
        remixed = istft(spectrum * bin_linear.T.astype(np.complex64), pcm_frames)
        peak = float(np.max(np.abs(remixed)))
        if peak > board_tuning.OUTPUT_PEAK_LIMIT:
            remixed = remixed * (board_tuning.OUTPUT_PEAK_LIMIT / peak)
        return remixed.T

    def _publish_tuning(self, result: dict) -> None:
        target = np.asarray(result["applied_db"], dtype=np.float64)
        self.last_tuning = {
            "mode": result.get("mode", "FLAT"),
            "style": result.get("style", "?"),
            "stale": bool(result.get("stale")),
            "applied_peak_db": round(float(np.max(np.abs(target))), 2),
            "applied_lifted": int(np.sum(target > 0.3)),
            "applied_cut": int(np.sum(target < -0.3)),
            "masking_peak_db": round(float(np.max(result["masking_db"])), 2),
            "exposure": round(float(result.get("exposure", 0.0)), 3),
            "band_db": [round(float(v), 2) for v in target],
            "device_db": [round(float(v), 2)
                          for v in np.asarray(result.get("device_db", target))],
            "masking_db": [round(float(v), 2)
                           for v in np.asarray(result.get("masking_db", np.zeros(24)))],
        }
        try:
            path = Path("/tmp/tuning-latest.json")
            writing = path.with_suffix(".writing")
            writing.write_text(json.dumps({"ts_ms": int(time.time() * 1000), **self.last_tuning},
                                          ensure_ascii=False), encoding="utf-8")
            os.replace(writing, path)
        except OSError:
            pass

    def _control_loop(self) -> None:
        """10 Hz: DMIC evidence + latest content features -> tuner -> telemetry."""
        while True:
            try:
                with self._features_lock:
                    features = self.latest_features
                if features is not None:
                    content_bands, shares = features
                    evidence = self.tuner.evidence.read() or {}
                    status = self.tuner.node_status.read()
                    result = self.tuner.step(
                        level_dbfs=evidence.get("level_dbfs"),
                        ambient_bands=evidence.get("band_dbfs"),
                        event=evidence.get("event"),
                        voice=float(evidence.get("voice_likelihood") or 0.0),
                        conversation=status.get("policy_state"),
                        content_bands=content_bands,
                        stem_shares=shares,
                        dt_s=0.1,
                    )
                    self._publish_tuning(result)
            except Exception:  # noqa: BLE001 - control loop must never kill audio
                pass
            time.sleep(0.1)

    def start_control_loop(self) -> None:
        if self._control_started:
            return
        self._control_started = True
        threading.Thread(target=self._control_loop, name="tuning-control", daemon=True).start()

    def _envelope_output(self, spectrum: np.ndarray, masks: np.ndarray, pcm_frames: int) -> np.ndarray:
        """Stems -> dominant roles -> scenario gains -> envelope-shaped remix.

        `pcm_frames` is the number of time-domain samples in the packet; the STFT frame
        count (masks.shape[-1]) is what the per-frame gain envelope walks over.
        """
        stft_frames = int(masks.shape[-1])
        # 1) per-stem estimated energy for the role decision (one ISTFT per stem)
        stats_db: dict[str, float] = {}
        for index, stem in enumerate(role_strategy.STEMMS):
            stem_signal = istft(masks[0, index].astype(np.complex64) * spectrum, pcm_frames)
            rms = float(np.sqrt((np.asarray(stem_signal, dtype=np.float64) ** 2).mean()))
            stats_db[stem] = 20.0 * np.log10(max(rms, 1e-12))
        roles = self.role_tracker.update(stats_db)
        # 2) scenario gain targets (contrast-only: attenuate competitors, never boost)
        t0 = self.samples_processed / SAMPLE_RATE
        active = role_strategy.scenario_active(t0, self.scenario_start_s, self.scenario_end_s)
        if active:
            raw = role_strategy.role_profile(roles, self.scenario)
            target = role_strategy.contrast_only_profile(raw, self.strategy_strength)
        else:
            target = {stem: 0.0 for stem in role_strategy.STEMMS}
        # 3) envelope-shaped per-frame gains (attack 400 ms / release 1200 ms)
        gain_frames = np.zeros((len(role_strategy.STEMMS), stft_frames), dtype=np.float32)
        for frame_index in range(stft_frames):
            applied = self.envelope.step(target, HOP / SAMPLE_RATE)
            gain_frames[:, frame_index] = [applied[stem] for stem in role_strategy.STEMMS]
        linear = (10.0 ** (gain_frames / 20.0)).astype(np.float32)  # [4, T]
        weights = np.einsum("st,sft->ft", linear, masks[0])  # [1025,T]
        if weights.shape != spectrum.shape[-2:]:
            raise WorkerError(f"envelope weights shape {weights.shape} != spectrum {spectrum.shape[-2:]}")
        remixed_spec = spectrum * weights.astype(np.complex64)
        remixed = istft(remixed_spec, pcm_frames)  # [2,pcm_frames]
        self.role_history.append({
            "t_s": round(t0, 3),
            "scenario": self.scenario if active else "quiet",
            "roles": roles,
            "stem_rms_db": {stem: round(stats_db[stem], 2) for stem in role_strategy.STEMMS},
            "target_gain_db": {stem: round(target[stem], 2) for stem in role_strategy.STEMMS},
            "applied_gain_db_last_frame": {stem: round(float(gain_frames[i, -1]), 2)
                                           for i, stem in enumerate(role_strategy.STEMMS)},
        })
        return remixed.T


def process(pcm: np.ndarray, channels: int, backend: str = "torch", model: str | None = None) -> np.ndarray:
    """Convenience one-shot helper used by tests: build a backend and process a packet."""
    backend_name = backend or "torch"
    manager = StreamingWindowManager(make_backend(backend_name, model))
    return manager.process(pcm, channels)


# --------------------------------------------------------------------------- #
# TCP protocol (unchanged wire format)
# --------------------------------------------------------------------------- #
def exact(conn: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise EOFError
        data.extend(chunk)
    return bytes(data)


def serve(manager: StreamingWindowManager, mode: str, backend_name: str, model: str | None, port: int) -> int:
    model_sha = sha256_file(model) if model and Path(model).is_file() else None
    print(
        f"{mode} worker backend={backend_name} ready port={port} "
        f"model={model or '-'} sha256={model_sha or '-'} "
        f"window={manager.window_frames}x{HOP} ({manager.latency_seconds:.3f}s) "
        f"gain={manager.output_gain} mode={manager.output_mode} scenario={manager.scenario} "
        "evidence=software-only",
        flush=True,
    )
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("127.0.0.1", port))
        server.listen()
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    raw = exact(conn, HEADER.size)
                    magic, schema, sequence, rate, channels, _reserved, frames = HEADER.unpack(raw)
                    if magic != MAGIC or schema != 1 or rate != SAMPLE_RATE or channels != MODEL_CHANNELS:
                        continue
                    if frames <= 0:
                        conn.sendall(raw)
                        continue
                    pcm = np.frombuffer(exact(conn, frames * channels * 4), dtype="<f4").copy()
                    started = time.time()
                    output = manager.process(pcm, channels)
                    elapsed = time.time() - started
                    if manager.output_mode == "envelope" and manager.policy == "s3":
                        meta = manager.last_tuning
                        print(
                            f"packet seq={sequence} frames={frames} "
                            f"rtf={elapsed / (frames / SAMPLE_RATE):.3f} "
                            f"mode={meta.get('mode')} style={meta.get('style')} "
                            f"applied_peak={meta.get('applied_peak_db')} "
                            f"lifted={meta.get('applied_lifted')} "
                            f"cut={meta.get('applied_cut')} "
                            f"mask_peak={meta.get('masking_peak_db')} "
                            f"exposure={meta.get('exposure')}"
                            + (" STALE" if meta.get('stale') else ""),
                            flush=True,
                        )
                    elif manager.output_mode == "envelope" and manager.role_history:
                        last = manager.role_history[-1]
                        print(
                            f"packet seq={sequence} frames={frames} "
                            f"rtf={elapsed / (frames / SAMPLE_RATE):.3f} "
                            f"scenario={last['scenario']} roles={last['roles']} "
                            f"target_db={last['target_gain_db']} "
                            f"applied_db={last['applied_gain_db_last_frame']}",
                            flush=True,
                        )
                    else:
                        print(
                            f"packet seq={sequence} frames={frames} rtf={elapsed / (frames / SAMPLE_RATE):.3f}",
                            flush=True,
                        )
                    conn.sendall(raw + output.tobytes())
                except (EOFError, OSError, ValueError, RuntimeError) as exc:
                    print(f"worker error: {exc}", flush=True)
    finally:
        server.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fixed-shape CPU/NPU worker (engineering probe). See module docstring for evidence scope.",
    )
    parser.add_argument("--mode", choices=("cpu", "npu"), required=True, help="cpu=PyTorch checkpoint, npu=RKNN-Lite2")
    parser.add_argument("--model", help=".pt checkpoint for cpu, .rknn for npu; optional for --backend stub")
    parser.add_argument(
        "--backend",
        choices=("torch", "rknnlite", "stub"),
        help="model backend; defaults to torch for --mode cpu and rknnlite for --mode npu",
    )
    parser.add_argument(
        "--model-sha256",
        help="expected sha256 of --model (also read from KHADAS_OWS_MODEL_SHA256); refuse to run on mismatch",
    )
    parser.add_argument("--window-frames", type=int, default=MODEL_FRAMES, help=f"fixed model window (default {MODEL_FRAMES})")
    parser.add_argument("--window-hop", type=int, default=None, help="frames between windows (default = window-frames)")
    parser.add_argument(
        "--output-mode",
        choices=("stem", "envelope"),
        default=None,
        help="stem = separated vocal stem (probe); envelope = role-aware per-stem gains for a "
             "noise scenario + attack/release envelope, remixed into music (product-path proxy). "
             "Default: $KHADAS_OWS_OUTPUT_MODE or 'stem'.",
    )
    parser.add_argument("--scenario", choices=role_strategy.SCENARIOS, default=None,
                        help="noise scenario for --output-mode envelope (default quiet)")
    parser.add_argument("--scenario-start-ms", type=float, default=0.0,
                        help="envelope-mode scenario window start (default 0)")
    parser.add_argument("--scenario-end-ms", type=float, default=1e9,
                        help="envelope-mode scenario window end (default whole stream)")
    parser.add_argument("--strategy-strength", type=float, default=0.6,
                        help="contrast-only strength for envelope mode (0..1, default 0.6)")
    parser.add_argument(
        "--output-gain",
        choices=("peak", "unity"),
        default=None,
        help="peak = historical probe convention (per-packet peak matching, can pump at window "
             "boundaries); unity = keep the model's own scale (use for listening). "
             "Default: $KHADAS_OWS_OUTPUT_GAIN or 'peak'.",
    )
    parser.add_argument(
        "--policy",
        choices=("roles", "s3"),
        default=None,
        help="envelope tuning policy: s3 = board-side S3 decision layer + D56 masking "
             "(default); roles = legacy role-contrast probe. "
             "Default: $KHADAS_OWS_POLICY or 's3'.",
    )
    parser.add_argument("--evidence-file", default="/tmp/dmic-latest.json",
                        help="board DMIC evidence JSON from tools/dmic_evidence.py")
    parser.add_argument("--node-status-url", default="http://127.0.0.1:8787/v1/status",
                        help="loopback node status (conversation policy state)")
    parser.add_argument("--eq-depth", type=float, default=board_tuning.DEFAULT_DEPTH,
                        help="device-suggestion depth applied on board (basic 1.3)")
    parser.add_argument("--masking-dose", type=float,
                        default=board_tuning.DEFAULT_MASKING_DOSE,
                        help="D56 masking dose (basic 1.0 / aggressive 1.4)")
    parser.add_argument("--max-positive-db", type=float,
                        default=board_tuning.MAX_POSITIVE_DB,
                        help="positive-gain ceiling (D54: 3 dB basic / 4 dB aggressive)")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)

    backend_name = args.backend or ("torch" if args.mode == "cpu" else "rknnlite")
    output_gain = args.output_gain or os.environ.get("KHADAS_OWS_OUTPUT_GAIN") or "peak"
    output_mode = args.output_mode or os.environ.get("KHADAS_OWS_OUTPUT_MODE") or "stem"
    scenario = args.scenario or os.environ.get("KHADAS_OWS_SCENARIO") or "quiet"
    policy = args.policy or os.environ.get("KHADAS_OWS_POLICY") or "s3"
    if output_gain not in ("peak", "unity"):
        print(f"error: invalid output gain {output_gain!r} (expected peak|unity)", file=sys.stderr)
        return 2
    expected_sha = args.model_sha256 or os.environ.get("KHADAS_OWS_MODEL_SHA256")
    port = args.port or (9201 if args.mode == "cpu" else 9202)
    try:
        backend = make_backend(backend_name, args.model, expected_sha)
        manager = StreamingWindowManager(
            backend, window_frames=args.window_frames, window_hop=args.window_hop,
            output_gain=output_gain, output_mode=output_mode, scenario=scenario,
            scenario_start_s=args.scenario_start_ms / 1000.0,
            scenario_end_s=args.scenario_end_ms / 1000.0,
            strategy_strength=args.strategy_strength,
            policy=policy,
            evidence_file=args.evidence_file,
            node_status_url=args.node_status_url,
            eq_depth=args.eq_depth,
            masking_dose=args.masking_dose,
            max_positive_db=args.max_positive_db)
    except WorkerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return serve(manager, args.mode, backend_name, args.model, port)


if __name__ == "__main__":
    sys.exit(main())
