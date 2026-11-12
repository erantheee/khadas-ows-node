"""Fixed-shape streaming worker tests (software behavior only, no board claims)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
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


def test_envelope_roles_policy_runs_six_stem_decision_on_four_stem_model():
    """Six-stem strategy schema (Mac parity) on the 4-stem candidate model."""
    backend = W.StubBackend()
    manager = W.StreamingWindowManager(
        backend, output_mode="envelope", policy="roles",
        scenario="broadband", strategy_strength=0.6)
    output = manager.process(sine_packet(1.0), CHANNELS)
    assert output.shape == (2 * SR,)
    assert np.isfinite(output).all()
    assert manager.role_history, "envelope path must record role history"
    entry = manager.role_history[-1]
    # decision layer runs the six-stem role map...
    assert set(entry["roles"]) == {"lead", "rhythm", "low", "bed"}
    # ...while execution stays on the four stems the model actually emits
    assert set(entry["target_gain_db"]) == {"vocals", "drums", "bass", "other"}
    # guitar/piano are silent on this model, so drums always owns rhythm
    assert entry["roles"]["rhythm"] == "drums"
    assert entry["roles"]["lead"] in {"vocals", "other"}
    # default listening mode without conversation evidence = "boost"
    assert entry["mode"] == "boost"
    assert set(entry["target_gain_db"]) == {"vocals", "drums", "bass", "other"}
    assert entry["content_type"] in {"music", "podcast", "video"}


def _write_evidence(path, payload, age_s: float = 0.0):
    path.write_text(json.dumps(payload), encoding="utf-8")
    if age_s:
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))


def test_environment_decision_shapes_envelope_targets(tmp_path):
    evidence = tmp_path / "evidence.json"
    _write_evidence(evidence, {"context": "street", "environment": "NOISY",
                               "event": "quiet", "band_dbfs": [0.0] * 24})
    # quiet input keeps the own-playback guard from demoting the street state
    packet = sine_packet(1.0, amplitude=0.0005)
    reference = W.StreamingWindowManager(
        W.StubBackend(), output_mode="envelope", policy="roles",
        evidence_file=str(evidence), env_decision="off")
    reference.process(packet, CHANNELS)
    shaped = W.StreamingWindowManager(
        W.StubBackend(), output_mode="envelope", policy="roles",
        evidence_file=str(evidence), env_decision="on")
    shaped.process(packet, CHANNELS)
    off_entry = reference.role_history[-1]
    on_entry = shaped.role_history[-1]
    assert off_entry["environment"] == "quiet"
    assert on_entry["environment"] == "street"
    assert on_entry["target_gain_db"]["drums"] < off_entry["target_gain_db"]["drums"]
    assert on_entry["target_gain_db"]["bass"] < off_entry["target_gain_db"]["bass"]
    assert on_entry["target_gain_db"]["vocals"] == off_entry["target_gain_db"]["vocals"]


def test_environment_fails_open_when_evidence_is_stale(tmp_path):
    evidence = tmp_path / "evidence.json"
    _write_evidence(evidence, {"context": "near_voice", "band_dbfs": [0.0] * 24}, age_s=60)
    manager = W.StreamingWindowManager(
        W.StubBackend(), output_mode="envelope", policy="roles",
        evidence_file=str(evidence))
    manager.process(sine_packet(1.0), CHANNELS)
    entry = manager.role_history[-1]
    assert entry["environment"] == "quiet"
    assert entry["mode"] == "boost"


def test_board_speech_evidence_masks_instead_of_ducking(tmp_path):
    """2026-11-10 decision: board speech feeds contrast, the Mac owns ducking."""
    evidence = tmp_path / "evidence.json"
    _write_evidence(evidence, {"context": "near_voice", "environment": "VOICE",
                               "event": "speech", "speech_active": True,
                               "band_dbfs": [0.0] * 24})
    manager = W.StreamingWindowManager(
        W.StubBackend(), output_mode="envelope", policy="roles",
        evidence_file=str(evidence))
    manager.process(sine_packet(1.0, amplitude=0.0005), CHANNELS)
    entry = manager.role_history[-1]
    assert entry["environment"] == "speech"
    assert entry["mode"] == "boost"                       # no duck without the Mac cue
    assert entry["target_gain_db"]["drums"] < 0.0         # contrast applied instead


def test_mac_conversation_cue_still_ducks(tmp_path):
    evidence = tmp_path / "evidence.json"
    _write_evidence(evidence, {"context": "quiet", "conversation_cue": 1.0,
                               "band_dbfs": [0.0] * 24})
    manager = W.StreamingWindowManager(
        W.StubBackend(), output_mode="envelope", policy="roles",
        evidence_file=str(evidence))
    manager.process(sine_packet(1.0), CHANNELS)
    entry = manager.role_history[-1]
    assert entry["mode"] == "duck"
    assert entry["target_gain_db"]["drums"] == -1.2


def test_media_guard_demotes_content_env_states_during_playback(tmp_path):
    """Open-ear leakage: speech/babble/street must not pump the mix while media plays."""
    evidence = tmp_path / "evidence.json"
    _write_evidence(evidence, {"context": "street", "environment": "NOISY",
                               "event": "quiet", "band_dbfs": [0.0] * 24})
    manager = W.StreamingWindowManager(
        W.StubBackend(), output_mode="envelope", policy="roles",
        evidence_file=str(evidence), env_media_guard="on")
    manager.process(sine_packet(1.0, amplitude=0.5), CHANNELS)   # hold 2 windows
    manager.process(sine_packet(1.0, amplitude=0.5), CHANNELS)
    entry = manager.role_history[-1]
    assert entry["media_active"] is True
    assert entry["environment"] == "quiet"                        # street demoted


def test_media_guard_keeps_physical_env_states_during_playback(tmp_path):
    evidence = tmp_path / "evidence.json"
    _write_evidence(evidence, {"context": "other", "environment": "NOISY",
                               "event": "wind", "band_dbfs": [0.0] * 24})
    manager = W.StreamingWindowManager(
        W.StubBackend(), output_mode="envelope", policy="roles",
        evidence_file=str(evidence), env_media_guard="on")
    manager.process(sine_packet(1.0, amplitude=0.5), CHANNELS)
    manager.process(sine_packet(1.0, amplitude=0.5), CHANNELS)
    entry = manager.role_history[-1]
    assert entry["environment"] == "wind"


def test_media_guard_off_keeps_street_during_playback(tmp_path):
    evidence = tmp_path / "evidence.json"
    _write_evidence(evidence, {"context": "street", "environment": "NOISY",
                               "event": "quiet", "band_dbfs": [0.0] * 24})
    manager = W.StreamingWindowManager(
        W.StubBackend(), output_mode="envelope", policy="roles",
        evidence_file=str(evidence), env_media_guard="off")
    manager.process(sine_packet(1.0, amplitude=0.5), CHANNELS)
    assert manager.role_history[-1]["environment"] == "street"
