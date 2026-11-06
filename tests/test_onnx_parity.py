"""Tiny-model PyTorch vs ONNXRuntime parity (no binary artifacts required).

This gives the regular ci.yml job a real numeric ONNX-vs-PyTorch check even though
the published 0.55M-parameter candidate and its checkpoint are not in Git.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from khadas_ows.model import MaskNetwork  # noqa: E402

MAX_ABS_TOLERANCE = 1e-4


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_torch_onnx_parity_tiny_model(tmp_path):
    torch.manual_seed(20261106)
    model = MaskNetwork(base=8, layers=1).eval()
    sample = torch.rand(1, 2, 33, 9)
    with torch.no_grad():
        reference = model(sample).numpy()

    onnx_path = tmp_path / "tiny-mask-network.onnx"
    torch.onnx.export(
        model,
        sample,
        onnx_path,
        input_names=["magnitude"],
        output_names=["masks"],
        opset_version=18,
        dynamic_axes=None,
    )
    assert onnx_path.is_file()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"magnitude": sample.numpy()})[0]

    assert output.shape == reference.shape == (1, 4, 33, 9)
    assert np.isfinite(output).all()
    max_abs = float(np.max(np.abs(output - reference)))
    assert max_abs <= MAX_ABS_TOLERANCE, f"max abs error {max_abs:.3e} > {MAX_ABS_TOLERANCE:.0e}"
    assert _cosine(output, reference) >= 0.999999
    # Contract: softmax stem masks sum to 1 per (frequency, time) bin.
    assert float(np.max(np.abs(output.sum(axis=1) - 1.0))) <= 1e-5
