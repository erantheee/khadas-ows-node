"""Fixed-shape streaming worker tests (software behavior only, no board claims)."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from khadas_ows import worker as W  # noqa: E402

SR = W.SAMPLE_RATE
CHANNELS = W.MODEL_CHANNELS


def sine_packet(seconds: float, amplitude: float = 0.5, frequency: float = 440.0) -> np.ndarray:
    frames = int(round(seconds * SR))
    time = np.arange(frames) / SR
    left = (amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)
    right = (0.8 * left).astype(np.float32)
    return np.stack([left, right], axis=1).reshape(-1).copy()


def test_stft_istft_roundtrip_and_frame_geometry():
    rng = np.random.default_rng(1)
    source = rng.standard_normal((2, SR)).astype(np.float32)
    spectrum = W.stft(source)
    # 1.0 s -> 1 + 44100 // 512 = 87 frames; the 4 s model window is 345 frames.
    assert spectrum.shape == (2, W.FREQ_BINS, 87)
    assert W.stft(np.zeros((2, 4 * SR), dtype=np.float32)).shape == (2, W.FREQ_BINS, 345)
    reconstructed = W.istft(spectrum, SR)
    assert reconstructed.shape == (2, SR)
    assert float(np.max(np.abs(reconstructed - source))) < 1e-4


def test_window_manager_uses_fixed_model_shape_and_zero_padding():
    backend = W.StubBackend()
    manager = W.StreamingWindowManager(backend)
    assert manager.latency_seconds == pytest.approx(345 * 512 / SR)
    manager.process(sine_packet(1.0), CHANNELS)
    assert len(backend.calls) == 1
    assert backend.calls[0]["shape"] == (1, 2, W.FREQ_BINS, 345)
    assert backend.calls[0]["frames_used"] == 87  # 1 s is zero-padded up to 345 frames

    backend_4s = W.StubBackend()
    W.StreamingWindowManager(backend_4s).process(sine_packet(4.0), CHANNELS)
    assert len(backend_4s.calls) == 1
    assert backend_4s.calls[0]["frames_used"] == 345

    backend_10s = W.StubBackend()
    W.StreamingWindowManager(backend_10s).process(sine_packet(10.0), CHANNELS)
    assert [call["frames_used"] for call in backend_10s.calls] == [345, 345, 172]


def test_reply_length_equals_request_length():
    for seconds in (1.0, 4.0):
        packet = sine_packet(seconds)
        output = W.StreamingWindowManager(W.StubBackend()).process(packet, CHANNELS)
        assert output.size == packet.size
        assert np.isfinite(output).all()


def test_output_is_finite_at_matched_peak():
    packet = sine_packet(2.0, amplitude=0.25)
    output = W.StreamingWindowManager(W.StubBackend()).process(packet, CHANNELS)
    assert np.isfinite(output).all()
    assert output.size == packet.size
    assert float(np.max(np.abs(output))) == pytest.approx(float(np.max(np.abs(packet))), abs=1e-3)
    assert float(np.max(np.abs(output))) > 0.0


def test_stub_backend_is_deterministic():
    packet = sine_packet(1.0)
    first = W.StreamingWindowManager(W.StubBackend()).process(packet, CHANNELS)
    second = W.StreamingWindowManager(W.StubBackend()).process(packet, CHANNELS)
    assert np.array_equal(first, second)


def test_window_hop_controls_model_calls():
    backend = W.StubBackend()
    manager = W.StreamingWindowManager(backend, window_hop=115)
    output = manager.process(sine_packet(4.0), CHANNELS)  # 345 frames -> starts 0, 115, 230
    assert [call["start"] for call in manager.calls] == [0, 115, 230]
    assert output.size == 4 * SR * CHANNELS


def test_npu_mode_without_rknnlite_fails_clearly(tmp_path):
    if importlib.util.find_spec("rknnlite") is not None:
        pytest.skip("rknnlite is installed on this host; the import-failure path does not apply")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "khadas_ows.worker",
            "--mode",
            "npu",
            "--model",
            str(tmp_path / "missing.rknn"),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "python"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "error:" in combined
    assert "rknnlite" in combined
    assert "npu backend unavailable" in combined


def test_model_hash_mismatch_is_rejected(tmp_path):
    model = tmp_path / "weights.bin"
    model.write_bytes(b"not a real checkpoint")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "khadas_ows.worker",
            "--mode",
            "cpu",
            "--backend",
            "stub",
            "--model",
            str(model),
            "--model-sha256",
            "0" * 64,
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "python"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "sha256 mismatch" in (result.stdout + result.stderr).lower()


def test_cpu_torch_backend_loads_checkpoint_and_runs(tmp_path):
    torch = pytest.importorskip("torch")
    from khadas_ows.model import MaskNetwork

    torch.manual_seed(7)
    checkpoint = tmp_path / "mask-network.pt"
    torch.save(MaskNetwork().state_dict(), checkpoint)
    backend = W.make_backend("torch", str(checkpoint))
    # A small probe tensor exercises the real checkpoint/architecture wiring cheaply;
    # the full 345-frame path is covered by the stub and parity tests.
    masks = np.asarray(backend.run(np.random.default_rng(3).random((1, 2, 33, 9), dtype=np.float32)))
    assert masks.shape == (1, W.STEMS, 33, 9)
    assert np.isfinite(masks).all()


def test_cpu_mode_missing_checkpoint_fails_clearly(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "khadas_ows.worker",
            "--mode",
            "cpu",
            "--model",
            str(tmp_path / "missing.pt"),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "python"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "model not found" in (result.stdout + result.stderr).lower()
