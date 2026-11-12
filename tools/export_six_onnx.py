#!/usr/bin/env python3
"""Export the six-stem network body to fixed-shape ONNX (Edge2 boundary).

Training model: waveform -> waveform (`work/stem-ab/band_scnet.py`, n_stems=6).
ONNX boundary used on Edge2: magnitude [1,2,1025,345] -> masks [1,6,1025,345]
(same as the four-stem candidate; STFT/ISTFT stay outside the model).

Usage:
    PYTHONPATH=python python tools/export_six_onnx.py \
        --checkpoint ../../cores3-noise-adapt-demo/work/stem-ab/runs/six_stem/band_split_six.pt \
        --output models/six/candidate-6s.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from khadas_ows.model import MaskNetwork  # noqa: E402

INPUT_SHAPE = (1, 2, 1025, 345)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="band_split_six.pt from train_six.py")
    ap.add_argument("--stems", type=int, default=6)
    ap.add_argument("--no-softmax", action="store_true",
                    help="export logits; caller applies fp32 softmax (RKNN fp16 fidelity)")
    ap.add_argument("--input-scale", type=float, default=16.0,
                    help="must match train_six.py --input_scale (fp16 headroom)")
    ap.add_argument("--output", default="models/six/candidate-6s.onnx")
    args = ap.parse_args()

    model = MaskNetwork(stems=args.stems, input_scale=args.input_scale,
                        apply_softmax=not args.no_softmax)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    body = {k: v for k, v in state.items() if k.startswith(("blocks.", "gate.", "mask."))}
    model.load_state_dict(body, strict=True)
    model.eval()
    count = sum(p.numel() for p in model.parameters())

    dummy = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(output), opset_version=18,
        input_names=["magnitude"], output_names=["masks"])
    reference = model(dummy).detach().numpy()

    import onnxruntime as ort
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"magnitude": dummy.numpy()})[0]
    max_abs = float(np.max(np.abs(reference - onnx_out)))
    cosine = float(np.dot(reference.ravel(), onnx_out.ravel())
                   / (np.linalg.norm(reference.ravel()) * np.linalg.norm(onnx_out.ravel()) + 1e-12))
    manifest = {
        "stems": args.stems,
        "input_scale": args.input_scale,
        "post_softmax": not args.no_softmax,
        "params": count,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input_shape": list(INPUT_SHAPE),
        "output_shape": [1, args.stems, 1025, 345],
        "opset": 18,
        "torch_vs_onnx_max_abs": max_abs,
        "torch_vs_onnx_cosine": cosine,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"exported {output} params={count} max_abs={max_abs:.3e} cosine={cosine:.8f}")


if __name__ == "__main__":
    main()
