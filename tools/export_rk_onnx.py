#!/usr/bin/env python3
"""Export the RK-targeted six-stem model (log1p input, 4 layers x 96 ch) to ONNX.

Training model: `tools/stem-ab-training/train_rk.py::RKNet` (waveform -> waveform).
ONNX boundary for the NPU: magnitude [1,2,1025,345] -> masks [1,6,1025,345],
with log1p normalisation inside the graph and fp16-rounding simulation off.

Usage:
    PYTHONPATH=python python tools/export_rk_onnx.py \
        --checkpoint ../../cores3-noise-adapt-demo/work/stem-ab/runs/six_rk/rk_net.pt \
        --output models/six/rk-6s.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
TRAIN_DIR = Path.home() / "Documents/Codex/2026-10-13/cores3-noise-adapt-demo/work/stem-ab"
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(TRAIN_DIR))

from train_rk import RKNet  # noqa: E402

INPUT_SHAPE = (1, 2, 1025, 345)


class RKExport(torch.nn.Module):
    """Magnitude -> masks graph used by the NPU (log1p in-graph, no fp16 simulation)."""

    def __init__(self, rk: RKNet) -> None:
        super().__init__()
        self.rk = rk
        self.rk.fp16_sim = False
        self.rk.eval()

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        mag = torch.log1p(magnitude)
        h = mag
        for layer in self.rk.blocks:
            h = layer(h)
        h = self.rk.gate(h)
        return torch.softmax(self.rk.mask(h), dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stems", type=int, default=6)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--output", default="models/six/rk-6s.onnx")
    args = ap.parse_args()

    rk = RKNet(n_stems=args.stems, base=args.base, layers=args.layers, fp16_sim=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    rk.load_state_dict(state, strict=True)
    model = RKExport(rk).eval()
    count = sum(p.numel() for p in rk.parameters())

    dummy = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(output), opset_version=18,
                      input_names=["magnitude"], output_names=["masks"])
    reference = model(dummy).detach().numpy()
    import onnxruntime as ort
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"magnitude": dummy.numpy()})[0]
    max_abs = float(np.max(np.abs(reference - onnx_out)))
    manifest = {
        "stems": args.stems, "base": args.base, "layers": args.layers,
        "params": count, "input_mode": "log1p",
        "input_shape": list(INPUT_SHAPE), "output_shape": [1, args.stems, 1025, 345],
        "opset": 18, "torch_vs_onnx_max_abs": max_abs,
    }
    (output.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2))
    print(f"exported {output} params={count} max_abs={max_abs:.3e}")


if __name__ == "__main__":
    main()
