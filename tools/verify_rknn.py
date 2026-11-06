#!/usr/bin/env python3
"""RKNN-Toolkit2 simulator parity against the ONNX reference (engineering probe).

Evidence scope
--------------
RKNN-Toolkit2 cannot run an exported ``.rknn`` file on the PC simulator: a model
loaded with ``RKNN.load_rknn`` requires a physical target
(``E init_runtime: RKNN model that loaded by 'load_rknn' not support inference on
the simulator, please set 'target' first``).  Running the exported artifact itself
is therefore **on-board** verification and remains unverified until an Edge2 is
available.

The legitimate pre-device gate is the one the toolkit itself recommends
("use 'load_xxx' & 'build' instead of 'load_rknn'"): rebuild the *same* ONNX graph
with the *same* conversion flags in this process (``load_onnx`` -> ``config`` ->
``build`` -> ``init_runtime()``, whose default target is the simulator), then run
fixed-shape inference and compare with the ONNXRuntime reference.  ``export_rknn``
writes exactly this built graph, so the rebuilt export is cross-checked against the
``.rknn`` artifact hash produced by ``tools/convert_rknn.py``.

A pass proves:
  * the ONNX graph converts and builds for ``rk3588`` with the same flags as the
    exported artifact,
  * the built graph runs in the RKNN-Toolkit2 PC simulator,
  * simulator numerics track the ONNX reference within the documented thresholds.

A pass does **not** prove on-board Edge2/RK3588 NPU execution, real-time
throughput, or listening quality.

Thresholds
----------
* float16 (non-quantised build): cosine >= 0.999 and max-abs error <= 1e-2.
* int8: cosine >= 0.98 (max-abs error is reported but not gated: quantisation
  changes absolute mask values by design).
* masks sum to 1 per bin: float16 atol 1e-2, int8 atol 5e-2 (output dequantisation
  granularity; a sanity check, not a quality claim).

Usage
-----
    PYTHONPATH=python python tools/verify_rknn.py \
        --onnx artifacts/candidate-v0.onnx --no-quant \
        --rknn models/candidate-v0.fp.rknn \
        --report reports/candidate-v0.fp.parity.json \
        --summary reports/candidate-v0.fp.summary.md

    PYTHONPATH=python python tools/verify_rknn.py \
        --onnx artifacts/candidate-v0.onnx \
        --dataset models/calib/dataset.txt --quantized \
        --rknn models/candidate-v0.int8.rknn \
        --report reports/candidate-v0.int8.parity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# This script lives beside convert_rknn.py; reuse its ONNX external-data handling
# so the verified graph is prepared exactly like the exported artifact.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_rknn import ConversionError, prepare_onnx, sha256_file, toolkit_version  # noqa: E402

INPUT_SHAPE = (1, 2, 1025, 345)
OUTPUT_SHAPE = (1, 4, 1025, 345)
STEMS = 4
DEFAULT_SEED = 20261106
DEFAULT_TARGET = "rk3588"

THRESHOLDS = {
    "float16": {"cosine_min": 0.999, "max_abs_max": 1e-2, "mask_sum_atol": 1e-2},
    "int8": {"cosine_min": 0.98, "max_abs_max": None, "mask_sum_atol": 5e-2},
}

EXPORTED_RKNN_NOTE = (
    "RKNN.load_rknn artifacts cannot run in the PC simulator (toolkit error: "
    "'not support inference on the simulator, please set target first'); running the "
    "exported .rknn file itself requires RK3588/Edge2 hardware and remains unverified. "
    "This report therefore verifies the identical in-process build graph and records "
    "whether the rebuilt export matches the exported artifact byte-for-byte."
)


class VerifyError(RuntimeError):
    """Raised when simulator parity cannot be established."""


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
    rebuilt = report.get("rebuilt_export") or {}
    artifact = report.get("exported_artifact") or {}
    match = rebuilt.get("matches_artifact_sha256")
    match_text = "n/a" if match is None else ("yes (byte-identical)" if match else "no")
    max_abs_threshold = (
        f"<= {report['thresholds']['max_abs_max']:.0e}"
        if report["thresholds"]["max_abs_max"] is not None
        else "reported only"
    )
    return "\n".join(
        [
            f"### RKNN simulator parity ({precision}) — {verdict}",
            "",
            "| metric | value | threshold |",
            "| --- | --- | --- |",
            f"| verification method | in-process `load_onnx`→`build`→`init_runtime()` | — |",
            f"| input convention | `{report['input_convention']}` | — |",
            f"| output shape | `{metrics['output_shape']}` | `[1,4,1025,345]` |",
            f"| cosine similarity vs ONNX | {metrics['cosine_similarity']:.8f} | "
            f">= {report['thresholds']['cosine_min']} |",
            f"| max abs error vs ONNX | {metrics['max_abs_error']:.3e} | {max_abs_threshold} |",
            f"| mean abs error vs ONNX | {metrics['mean_abs_error']:.3e} | — |",
            f"| max \\|sum(masks)-1\\| | {metrics['mask_sum_max_abs_error']:.3e} | "
            f"<= {report['thresholds']['mask_sum_atol']:.0e} |",
            f"| exported artifact sha256 | `{artifact.get('sha256', 'n/a')}` | — |",
            f"| rebuilt export matches artifact | {match_text} | — |",
            "",
            "Simulator parity is verified on the in-process built graph (the same graph "
            "`export_rknn` writes). The exported `.rknn` file itself cannot be loaded by "
            "the PC simulator: that requires RK3588/Edge2 hardware, so **on-board NPU "
            "execution remains unverified**.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict:
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise VerifyError(f"ONNX model not found: {onnx_path}")
    quantized = bool(args.quantized)
    precision = "int8" if quantized else "float16"
    thresholds = THRESHOLDS[precision]

    dataset: Path | None = None
    dataset_info = None
    if quantized:
        if not args.dataset:
            raise VerifyError("--quantized requires --dataset <dataset.txt> (same calibration set as the conversion)")
        dataset = Path(args.dataset)
        if not dataset.is_file():
            raise VerifyError(f"calibration dataset not found: {dataset}")
        entries = [line.strip() for line in dataset.read_text().splitlines() if line.strip()]
        missing = [entry for entry in entries if not (dataset.parent / entry).is_file()]
        if not entries or missing:
            raise VerifyError(f"calibration dataset {dataset} is empty or has missing entries: {missing[:5]}")
        dataset_info = {"path": str(dataset), "entries": len(entries), "sha256": sha256_file(dataset)}
    elif args.dataset:
        raise VerifyError("--dataset only applies to INT8 (--quantized); pass --no-quant without a dataset")

    exported_artifact = None
    if args.rknn:
        artifact_path = Path(args.rknn)
        if not artifact_path.is_file():
            raise VerifyError(f"exported .rknn artifact not found: {artifact_path}")
        exported_artifact = {
            "path": str(artifact_path),
            "sha256": sha256_file(artifact_path),
            "bytes": artifact_path.stat().st_size,
        }

    try:
        from rknn.api import RKNN
    except Exception as exc:  # noqa: BLE001 - missing or broken toolkit install
        raise VerifyError(
            "rknn-toolkit2 is not importable in this environment "
            f"({exc!r}). Official wheels exist for Linux x86_64/aarch64 only; "
            "run this tool inside .github/workflows/rknn.yml."
        ) from exc

    work_dir = Path(tempfile.mkdtemp(prefix="rknn-verify-"))
    report: dict = {
        "schema": 1,
        "purpose": "RKNN-Toolkit2 simulator parity vs ONNX reference (engineering probe)",
        "verification_method": "in_process_build_and_simulator",
        "verification_method_note": (
            "RKNN-Toolkit2 cannot run a load_rknn() artifact on the PC simulator; the "
            "in-process load_onnx->build->init_runtime() path runs the graph that "
            "export_rknn writes and is the toolkit-recommended pre-device check."
        ),
        "evidence_scope": {
            "proven_by_this_report": [
                "the fixed-shape ONNX graph builds for the requested target with the same flags as the exported artifact",
                "the built graph runs in the RKNN-Toolkit2 PC simulator and tracks the ONNX reference numerically",
            ],
            "not_proven": [
                "loading/running the exported .rknn file itself (requires RK3588/Edge2 hardware)",
                "on-board Edge2 / RK3588 NPU execution",
                "real-time throughput on the board",
                "listening / separation quality",
            ],
        },
        "precision": precision,
        "quantized": quantized,
        "target": args.target,
        "toolkit_version": toolkit_version(),
        "source_onnx": str(onnx_path),
        "source_onnx_sha256": sha256_file(onnx_path),
        "dataset": dataset_info,
        "exported_artifact": exported_artifact,
        "exported_rknn_loadable_in_simulator": False,
        "exported_rknn_note": EXPORTED_RKNN_NOTE,
        "simulator": {"init_runtime_target": None, "note": "RKNN.init_runtime() with no target runs the PC simulator"},
        "thresholds": thresholds,
    }

    try:
        report.update(prepare_onnx(onnx_path, work_dir))
        inlined = Path(report["inlined_onnx"])
        rng = np.random.default_rng(args.seed)
        sample = rng.random(INPUT_SHAPE, dtype=np.float32)
        report["input"] = {
            "shape": list(INPUT_SHAPE),
            "seed": args.seed,
            "sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
        }

        reference = _onnx_reference(onnx_path, sample)
        if tuple(reference.shape) != OUTPUT_SHAPE:
            raise VerifyError(f"ONNX reference output shape {reference.shape}; expected {OUTPUT_SHAPE}")
        report["onnx_reference_output_shape"] = list(reference.shape)

        rebuilt_export = work_dir / f"candidate-v0.{precision}.rebuilt.rknn"
        rknn = RKNN(verbose=bool(args.verbose))
        try:
            started = time.time()
            for step, function, kwargs in (
                ("config", rknn.config, {"target_platform": args.target}),
                ("load_onnx", rknn.load_onnx, {"model": str(inlined)}),
                (
                    "build",
                    rknn.build,
                    {"do_quantization": quantized, "dataset": str(dataset) if dataset is not None else None},
                ),
            ):
                try:
                    result = function(**kwargs)
                except Exception as exc:  # noqa: BLE001
                    raise VerifyError(f"RKNN step '{step}' raised: {exc}") from exc
                if result != 0:
                    raise VerifyError(f"RKNN step '{step}' failed with return code {result}")
            report["build_time_sec"] = round(time.time() - started, 3)
            result = rknn.export_rknn(str(rebuilt_export))
            if result != 0:
                raise VerifyError(f"RKNN.export_rknn (rebuilt graph) failed with return code {result}")
            rebuilt = {
                "sha256": sha256_file(rebuilt_export),
                "bytes": rebuilt_export.stat().st_size,
                "matches_artifact_sha256": None,
            }
            if exported_artifact is not None:
                rebuilt["matches_artifact_sha256"] = rebuilt["sha256"] == exported_artifact["sha256"]
            report["rebuilt_export"] = rebuilt

            # No target argument => RKNN-Toolkit2 simulator.
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
    finally:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)

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
    parser = argparse.ArgumentParser(
        description="Verify the RKNN-Toolkit2 built graph in the PC simulator against ONNX.",
    )
    parser.add_argument("--onnx", required=True, help="ONNX model the artifact was converted from")
    parser.add_argument("--rknn", help="exported .rknn artifact to cross-check (hash/size); not loadable in the simulator")
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"RKNN target platform (default: {DEFAULT_TARGET})")
    precision = parser.add_mutually_exclusive_group(required=True)
    precision.add_argument("--no-quant", action="store_true", help="build float16 like the fp16 conversion")
    precision.add_argument("--quantized", action="store_true", help="build INT8 like the int8 conversion")
    parser.add_argument("--dataset", help="calibration dataset.txt used by the INT8 conversion")
    parser.add_argument("--report", help="write the JSON parity report here")
    parser.add_argument("--summary", help="write a markdown table (for $GITHUB_STEP_SUMMARY) here")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for the fixed test input")
    parser.add_argument("--verbose", action="store_true", help="pass verbose=True to RKNN-Toolkit2")
    args = parser.parse_args(argv)

    try:
        report = run(args)
    except (VerifyError, ConversionError) as exc:
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
