#!/usr/bin/env python3
"""ONNX integrity check and PyTorch-vs-ONNXRuntime parity (engineering probe).

Evidence scope
--------------
A pass here proves that the published ONNX artifact loads with its external-data
sidecar, exercises the expected fixed input/output boundary, and (when a source
checkpoint is supplied) numerically matches the PyTorch model within 1e-4.  It
does not prove RKNN conversion or Edge2 NPU execution; those are separate steps.

Checks
------
1. Every external-data reference is present beside the ``.onnx`` file; hashes are
   recorded for the ``.onnx`` and its sidecar.
2. ``SHA256SUMS`` / an export manifest, when supplied, are compared against the
   computed hashes.
3. ``onnx.checker`` accepts the fully loaded model; opset and tensor shapes match
   the expected boundary.
4. ONNXRuntime inference on a seeded fixed input returns ``[1,4,1025,345]``,
   finite masks that sum to 1 per bin.
5. With ``--checkpoint``: PyTorch vs ONNXRuntime max-abs error <= 1e-4 (the
   default; override with --max-abs).

Usage
-----
    PYTHONPATH=python python tools/verify_onnx.py \
        --onnx artifacts/candidate-v0.onnx \
        --sha256sums artifacts/SHA256SUMS \
        --report reports/candidate-v0.onnx.json

    PYTHONPATH=python python tools/verify_onnx.py \
        --onnx models/candidate-v0.onnx --checkpoint models/candidate-v0.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

INPUT_SHAPE = (1, 2, 1025, 345)
OUTPUT_SHAPE = (1, 4, 1025, 345)
DEFAULT_OPSET = 18
MASK_SUM_ATOL = 1e-4


class VerifyError(RuntimeError):
    """Raised for any failed integrity or parity check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_of(value_info) -> list[int]:
    return [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]


def _parse_sha256sums(path: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise VerifyError(f"malformed SHA256SUMS line: {line!r}")
        digest, filename = parts[0].lower(), parts[-1]
        sums[Path(filename).name] = digest
    return sums


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def run(args: argparse.Namespace) -> dict:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - declared project/test dependency
        raise VerifyError(f"onnx and onnxruntime are required for this check: {exc}") from exc

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise VerifyError(f"ONNX model not found: {onnx_path}")

    report: dict = {
        "schema": 1,
        "purpose": "ONNX integrity and PyTorch/ONNXRuntime parity (engineering probe)",
        "evidence_scope": {
            "proven_by_this_report": "the published ONNX artifact loads with its external data, has the expected boundary, and passes a deterministic ONNXRuntime forward pass (plus torch parity when --checkpoint is supplied)",
            "not_proven": ["RKNN conversion", "on-board Edge2 NPU execution", "listening quality"],
        },
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "checks": {},
    }

    # 1. External-data references must resolve beside the .onnx file.
    bare = onnx.load_model(str(onnx_path), load_external_data=False)
    external_files = []
    for tensor in bare.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        location = next((entry.value for entry in tensor.external_data if entry.key == "location"), None)
        if not location:
            raise VerifyError(f"external-data initializer {tensor.name!r} has no location entry")
        data_path = onnx_path.parent / location
        if not data_path.is_file():
            raise VerifyError(
                f"missing external data {data_path}; the ONNX artifact is incomplete "
                "(download candidate-v0.onnx.data beside candidate-v0.onnx)"
            )
        if not any(entry["location"] == location for entry in external_files):
            external_files.append(
                {"location": location, "bytes": data_path.stat().st_size, "sha256": sha256_file(data_path)}
            )
    report["external_data"] = external_files

    # 2. Optional published hash manifests.
    hashes = {onnx_path.name: report["onnx_sha256"]}
    for entry in external_files:
        hashes[Path(entry["location"]).name] = entry["sha256"]
    if args.sha256sums:
        sums_path = Path(args.sha256sums)
        if not sums_path.is_file():
            raise VerifyError(f"SHA256SUMS file not found: {sums_path}")
        sums = _parse_sha256sums(sums_path)
        compared = {}
        for name, expected in sums.items():
            if name in hashes:
                compared[name] = hashes[name] == expected
                if not compared[name]:
                    raise VerifyError(f"SHA256 mismatch for {name}: published {expected} got {hashes[name]}")
        if not compared:
            raise VerifyError(f"SHA256SUMS lists {sorted(sums)} but none match the ONNX artifacts {sorted(hashes)}")
        report["checks"]["sha256sums"] = {"path": str(sums_path), "matched": compared}
        print(f"sha256sums: ok ({', '.join(compared)})")

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            raise VerifyError(f"export manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        for field, local_name in (
            ("onnx_sha256", onnx_path.name),
            ("onnx_external_data_sha256", Path(external_files[0]["location"]).name if external_files else None),
        ):
            expected = manifest.get(field)
            if expected is None or local_name is None:
                continue
            if hashes.get(local_name) != expected:
                raise VerifyError(f"manifest {field} mismatch: manifest {expected} vs local {hashes.get(local_name)}")
        report["checks"]["manifest"] = {"path": str(manifest_path), "matched": True}
        print(f"manifest: ok ({manifest_path})")

    # 3. Full graph check and boundary validation.
    model = onnx.load_model(str(onnx_path), load_external_data=True)
    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001
        raise VerifyError(f"ONNX checker rejected {onnx_path}: {exc}") from exc

    opset = {opset.domain or "ai.onnx": int(opset.version) for opset in model.opset_import}
    input_shapes = {value.name: _shape_of(value) for value in model.graph.input}
    output_shapes = {value.name: _shape_of(value) for value in model.graph.output}
    report["opset"] = opset
    report["input_shapes"] = input_shapes
    report["output_shapes"] = output_shapes
    if list(input_shapes.values()) != [list(INPUT_SHAPE)]:
        raise VerifyError(f"unexpected ONNX inputs {input_shapes}; expected {list(INPUT_SHAPE)}")
    if list(output_shapes.values()) != [list(OUTPUT_SHAPE)]:
        raise VerifyError(f"unexpected ONNX outputs {output_shapes}; expected {list(OUTPUT_SHAPE)}")
    default_domain_opset = opset.get("ai.onnx") or opset.get("")
    if default_domain_opset is not None and default_domain_opset != args.expect_opset:
        raise VerifyError(f"expected opset {args.expect_opset}, model uses {default_domain_opset}")
    report["checks"]["graph"] = {"checker": "ok", "opset": default_domain_opset, "boundary": "ok"}

    # 4. ONNXRuntime reference forward pass on a seeded fixed input.
    rng = np.random.default_rng(args.seed)
    sample = rng.random(INPUT_SHAPE, dtype=np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    reference = session.run(None, {session.get_inputs()[0].name: sample})[0].astype(np.float32)
    if tuple(reference.shape) != OUTPUT_SHAPE:
        raise VerifyError(f"ONNXRuntime output shape {reference.shape}; expected {OUTPUT_SHAPE}")
    if not np.isfinite(reference).all():
        raise VerifyError("ONNXRuntime output contains non-finite values")
    mask_sum_error = float(np.abs(reference.sum(axis=1) - 1.0).max())
    if mask_sum_error > MASK_SUM_ATOL:
        raise VerifyError(f"mask sums deviate from 1 by {mask_sum_error:.3e} (> {MASK_SUM_ATOL})")
    report["checks"]["onnxruntime"] = {
        "input_seed": args.seed,
        "output_shape": list(reference.shape),
        "finite": True,
        "mask_sum_max_abs_error": mask_sum_error,
        "mask_sum_atol": MASK_SUM_ATOL,
        "providers": session.get_providers(),
    }
    print(f"onnxruntime: ok shape={tuple(reference.shape)} mask_sum_err={mask_sum_error:.2e}")

    # 5. Optional PyTorch-vs-ONNXRuntime numeric parity.
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            raise VerifyError(f"checkpoint not found: {checkpoint}")
        try:
            import torch
            from khadas_ows.model import load_legacy_checkpoint
        except ImportError as exc:
            raise VerifyError(
                f"PyTorch parity requested but torch/khadas_ows are unavailable ({exc}); "
                "run with PYTHONPATH=python and install torch"
            ) from exc
        model_pt = load_legacy_checkpoint(str(checkpoint))
        with torch.no_grad():
            torch_output = model_pt(torch.from_numpy(sample)).numpy().astype(np.float32)
        if tuple(torch_output.shape) != OUTPUT_SHAPE:
            raise VerifyError(f"PyTorch output shape {torch_output.shape}; expected {OUTPUT_SHAPE}")
        max_abs = float(np.abs(torch_output - reference).max())
        mean_abs = float(np.abs(torch_output - reference).mean())
        cosine = _cosine(torch_output, reference)
        if max_abs > args.max_abs:
            raise VerifyError(
                f"PyTorch/ONNXRuntime parity failed: max_abs={max_abs:.3e} > {args.max_abs:.3e} "
                f"(mean_abs={mean_abs:.3e}, cosine={cosine:.8f})"
            )
        report["checks"]["torch_parity"] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "cosine_similarity": cosine,
            "threshold_max_abs": args.max_abs,
        }
        print(f"torch parity: ok max_abs={max_abs:.3e} cosine={cosine:.8f}")
    else:
        report["checks"]["torch_parity"] = {"skipped": "no --checkpoint supplied"}

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the published ONNX artifact and optional torch parity.")
    parser.add_argument("--onnx", required=True, help="ONNX model path (external-data sidecar beside it)")
    parser.add_argument("--checkpoint", help="optional PyTorch checkpoint for torch-vs-onnxruntime parity")
    parser.add_argument("--sha256sums", help="optional SHA256SUMS file to verify against")
    parser.add_argument("--manifest", help="optional export manifest JSON to verify hashes against")
    parser.add_argument("--report", help="write the JSON report to this path")
    parser.add_argument("--seed", type=int, default=20261106, help="RNG seed for the fixed test input")
    parser.add_argument("--expect-opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--max-abs", type=float, default=1e-4, help="torch-vs-ORT max abs tolerance")
    args = parser.parse_args(argv)

    try:
        report = run(args)
    except VerifyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote report {report_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
