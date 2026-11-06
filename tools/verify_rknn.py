#!/usr/bin/env python3
"""RKNN-Toolkit2 simulator parity against the ONNX reference (engineering probe).

Evidence scope
--------------
This tool runs the exported ``.rknn`` model in RKNN-Toolkit2's **PC simulator**
(``RKNN.init_runtime()`` with no target), compares it with the ONNXRuntime
reference on one deterministic fixed-size input, and records the numbers.

A pass proves:
  * the ``.rknn`` artifact loads in the toolkit,
  * fixed-shape inference returns the expected ``[1,4,1025,345]`` tensor,
  * simulator numerics track ONNX within the documented thresholds.

A pass does **not** prove on-board Edge2/RK3588 NPU execution, real-time
throughput, or listening quality.  On-board execution needs RKNN-Toolkit-Lite2
on the board itself and is still unverified.

Thresholds
----------
* float16 (non-quantised build): cosine >= 0.999 and max-abs error <= 1e-2.
* int8: cosine >= 0.98 (max-abs error is reported but not gated: quantisation
  changes absolute mask values by design; a higher max-abs error is expected).
* masks sum to 1 per bin: float16 atol 1e-2, int8 atol 5e-2 (dequantisation
  granularity of the output tensor; this is a sanity check, not a quality claim).

Feeding the boundary
--------------------
The RKNN/ONNX boundary is NCHW ``[1,2,1025,345]``.  RKNN-Toolkit2's ``inference``
takes a ``data_format`` hint for 4-D inputs; rather than assume one calling
convention this tool probes the plausible variants and keeps the one whose output
best matches the ONNX reference, recording the winning convention in the report.

Usage
-----
    PYTHONPATH=python python tools/verify_rknn.py \
        --rknn models/candidate-v0.fp.rknn \
        --onnx artifacts/candidate-v0.onnx \
        --report reports/candidate-v0.fp.parity.json \
        --summary reports/candidate-v0.fp.summary.md

    PYTHONPATH=python python tools/verify_rknn.py \
        --rknn models/candidate-v0.int8.rknn \
        --onnx artifacts/candidate-v0.onnx --quantized \
        --report reports/candidate-v0.int8.parity.json
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
STEMS = 4
DEFAULT_SEED = 20261106

THRESHOLDS = {
    "float16": {"cosine_min": 0.999, "max_abs_max": 1e-2, "mask_sum_atol": 1e-2},
    "int8": {"cosine_min": 0.98, "max_abs_max": None, "mask_sum_atol": 5e-2},
}


class VerifyError(RuntimeError):
    """Raised when simulator parity cannot be established."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def toolkit_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("rknn-toolkit2")
    except Exception:  # pragma: no cover - only on hosts without the toolkit
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def _canonical_output(value) -> tuple[np.ndarray | None, str]:
    """Normalise a simulator output array to [1,4,1025,345] if possible."""
    array = np.asarray(value, dtype=np.float32)
    if array.shape == OUTPUT_SHAPE:
        return array, "nchw"
    if array.shape == (OUTPUT_SHAPE[0], INPUT_SHAPE[2], INPUT_SHAPE[3], STEMS):
        return array.transpose(0, 3, 1, 2), "nhwc->nchw"
    if array.shape == (STEMS, INPUT_SHAPE[2], INPUT_SHAPE[3]):
        return array[None, ...], "chw->nchw"
    return None, f"unexpected shape {array.shape}"


def _onnx_reference(onnx_path: Path, sample: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: sample})[0].astype(np.float32)


def probe_simulator(rknn, sample: np.ndarray, reference: np.ndarray) -> list[dict]:
    """Try plausible input-format conventions and score each against the reference."""
    variants = [
        ("nchw", sample, ["nchw"]),
        ("default", sample, None),
        ("nhwc", np.ascontiguousarray(sample.transpose(0, 2, 3, 1)), ["nhwc"]),
    ]
    attempts: list[dict] = []
    for name, payload, data_format in variants:
        attempt = {"variant": name, "data_format": data_format, "input_shape": list(payload.shape)}
        try:
            kwargs = {} if data_format is None else {"data_format": data_format}
            outputs = rknn.inference(inputs=[np.ascontiguousarray(payload)], **kwargs)
        except Exception as exc:  # noqa: BLE001 - record and try the next convention
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            print(f"simulator input variant {name!r} failed: {attempt['error']}")
            attempts.append(attempt)
            continue
        if not outputs:
            attempt["error"] = "simulator returned no outputs"
            attempts.append(attempt)
            continue
        canonical, layout = _canonical_output(outputs[0])
        attempt["output_shape"] = list(np.asarray(outputs[0]).shape)
        attempt["layout"] = layout
        if canonical is None:
            attempt["error"] = f"could not canonicalise output layout ({layout})"
            attempts.append(attempt)
            continue
        if not np.isfinite(canonical).all():
            attempt["error"] = "non-finite simulator output"
            attempts.append(attempt)
            continue
        attempt["cosine_similarity"] = _cosine(canonical, reference)
        attempt["max_abs_error"] = float(np.abs(canonical - reference).max())
        attempt["mean_abs_error"] = float(np.abs(canonical - reference).mean())
        attempt["mask_sum_max_abs_error"] = float(np.abs(canonical.sum(axis=1) - 1.0).max())
        print(
            f"simulator input variant {name!r}: cosine={attempt['cosine_similarity']:.8f} "
            f"max_abs={attempt['max_abs_error']:.3e}"
        )
        attempts.append(attempt)
    return attempts


def markdown_summary(report: dict) -> str:
    precision = report["precision"]
    metrics = report["metrics"]
    verdict = "PASS" if report["passed"] else "FAIL"
    return "\n".join(
        [
            f"### RKNN simulator parity ({precision}) — {verdict}",
            "",
            "| metric | value | threshold |",
            "| --- | --- | --- |",
            f"| input convention | `{report['input_convention']}` | — |",
            f"| output shape | `{metrics['output_shape']}` | `[1,4,1025,345]` |",
            f"| cosine similarity vs ONNX | {metrics['cosine_similarity']:.8f} | "
            f">= {report['thresholds']['cosine_min']} |",
            f"| max abs error vs ONNX | {metrics['max_abs_error']:.3e} | "
            f"{'<= ' + format(report['thresholds']['max_abs_max'], '.0e') if report['thresholds']['max_abs_max'] is not None else 'reported only'} |",
            f"| mean abs error vs ONNX | {metrics['mean_abs_error']:.3e} | — |",
            f"| max \\|sum(masks)-1\\| | {metrics['mask_sum_max_abs_error']:.3e} | "
            f"<= {report['thresholds']['mask_sum_atol']:.0e} |",
            f"| rknn sha256 | `{report['rknn_sha256']}` | — |",
            "",
            "Simulator only: this does **not** prove on-board Edge2 NPU execution.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict:
    rknn_path = Path(args.rknn)
    onnx_path = Path(args.onnx)
    if not rknn_path.is_file():
        raise VerifyError(f"RKNN model not found: {rknn_path}")
    if not onnx_path.is_file():
        raise VerifyError(f"ONNX reference not found: {onnx_path}")

    try:
        from rknn.api import RKNN
    except Exception as exc:  # noqa: BLE001 - missing or broken toolkit install
        raise VerifyError(
            "rknn-toolkit2 is not importable in this environment "
            f"({exc!r}). Official wheels exist for Linux x86_64/aarch64 only; "
            "run this tool inside .github/workflows/rknn.yml."
        ) from exc

    precision = "int8" if args.quantized else "float16"
    thresholds = THRESHOLDS[precision]

    rng = np.random.default_rng(args.seed)
    sample = rng.random(INPUT_SHAPE, dtype=np.float32)
    report: dict = {
        "schema": 1,
        "purpose": "RKNN-Toolkit2 simulator parity vs ONNX reference (engineering probe)",
        "evidence_scope": {
            "proven_by_this_report": "the .rknn artifact loads and runs in the RKNN-Toolkit2 PC simulator and tracks the ONNX reference numerically",
            "not_proven": [
                "on-board Edge2 / RK3588 NPU execution (requires RKNN-Toolkit-Lite2 on the board)",
                "real-time throughput on the board",
                "listening / separation quality",
            ],
        },
        "precision": precision,
        "quantized": bool(args.quantized),
        "toolkit_version": toolkit_version(),
        "rknn": str(rknn_path),
        "rknn_sha256": sha256_file(rknn_path),
        "rknn_bytes": rknn_path.stat().st_size,
        "onnx_reference": str(onnx_path),
        "onnx_reference_sha256": sha256_file(onnx_path),
        "simulator": True,
        "simulator_note": "RKNN.init_runtime() with no target runs RKNN-Toolkit2's PC simulator",
        "input": {"shape": list(INPUT_SHAPE), "seed": args.seed, "sha256": hashlib.sha256(sample.tobytes()).hexdigest()},
        "thresholds": thresholds,
    }

    reference = _onnx_reference(onnx_path, sample)
    if tuple(reference.shape) != OUTPUT_SHAPE:
        raise VerifyError(f"ONNX reference output shape {reference.shape}; expected {OUTPUT_SHAPE}")
    report["onnx_reference_output_shape"] = list(reference.shape)

    rknn = RKNN(verbose=bool(args.verbose))
    try:
        result = rknn.load_rknn(str(rknn_path))
        if result != 0:
            raise VerifyError(f"RKNN.load_rknn failed with return code {result}")
        # No target argument => RKNN-Toolkit2 simulator (not a board).
        result = rknn.init_runtime()
        if result != 0:
            raise VerifyError(f"RKNN.init_runtime (simulator) failed with return code {result}")

        attempts = probe_simulator(rknn, sample, reference)
        report["input_attempts"] = attempts
        usable = [attempt for attempt in attempts if "cosine_similarity" in attempt]
        if not usable:
            raise VerifyError(
                "simulator inference produced no evaluable output for any input convention; "
                f"attempts={json.dumps(attempts)}"
            )
        best = max(usable, key=lambda attempt: attempt["cosine_similarity"])
        report["input_convention"] = best["variant"]
        report["metrics"] = {
            "output_shape": best["output_shape"],
            "canonical_output_shape": list(OUTPUT_SHAPE),
            "cosine_similarity": best["cosine_similarity"],
            "max_abs_error": best["max_abs_error"],
            "mean_abs_error": best["mean_abs_error"],
            "mask_sum_max_abs_error": best["mask_sum_max_abs_error"],
        }
    finally:
        try:
            rknn.release()
        except Exception:  # noqa: BLE001 - release must not mask the real failure
            pass

    metrics = report["metrics"]
    failures = []
    if metrics["output_shape"] != list(OUTPUT_SHAPE):
        failures.append(f"output shape {metrics['output_shape']} != {list(OUTPUT_SHAPE)}")
    if metrics["cosine_similarity"] < thresholds["cosine_min"]:
        failures.append(
            f"cosine {metrics['cosine_similarity']:.8f} < {thresholds['cosine_min']} (max_abs {metrics['max_abs_error']:.3e})"
        )
    if thresholds["max_abs_max"] is not None and metrics["max_abs_error"] > thresholds["max_abs_max"]:
        failures.append(f"max_abs {metrics['max_abs_error']:.3e} > {thresholds['max_abs_max']:.0e}")
    if metrics["mask_sum_max_abs_error"] > thresholds["mask_sum_atol"]:
        failures.append(
            f"mask sum error {metrics['mask_sum_max_abs_error']:.3e} > {thresholds['mask_sum_atol']:.0e}"
        )
    report["checks_passed"] = not failures
    report["passed"] = not failures
    report["failures"] = failures
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify an RKNN model in the toolkit simulator against ONNX.")
    parser.add_argument("--rknn", required=True, help=".rknn artifact to load in the simulator")
    parser.add_argument("--onnx", required=True, help="ONNX reference model")
    parser.add_argument("--quantized", action="store_true", help="use INT8 parity thresholds")
    parser.add_argument("--report", help="write the JSON parity report here")
    parser.add_argument("--summary", help="write a markdown table (for $GITHUB_STEP_SUMMARY) here")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for the fixed test input")
    parser.add_argument("--verbose", action="store_true", help="pass verbose=True to RKNN-Toolkit2")
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
    summary = markdown_summary(report)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary + "\n")
        print(f"wrote summary {summary_path}")
    print(summary)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print("error: simulator parity FAILED: " + "; ".join(report["failures"]), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
