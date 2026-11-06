#!/usr/bin/env python3
"""Harden the fixed-shape ONNX -> RKNN conversion (RKNN-Toolkit2, engineering probe).

Evidence scope
--------------
A successful run of this tool proves that the fixed-shape ONNX model can be
converted into a ``.rknn`` binary for the requested target platform.  It does
**not** prove on-board Edge2 NPU execution, real-time throughput on RK3588, or
acceptable listening quality.  Those require RKNN-Toolkit-Lite2 on an actual
board and are tracked separately (see AGENTS.md and docs/handoff.md).

Why this runs on a Linux x86_64 runner
--------------------------------------
Official ``rknn-toolkit2`` wheels are published for Linux x86_64/aarch64 only.
The development machine is macOS arm64, so the conversion runs in the
``rknn-convert`` GitHub Actions workflow (ubuntu-22.04, Python 3.11).

External-data ONNX
------------------
The published model is ``candidate-v0.onnx`` plus ``candidate-v0.onnx.data``.
RKNN-Toolkit2's ONNX front-end is not relied on for external-data resolution:
this tool validates every external-data reference with the ``onnx`` package and
re-saves one self-contained ONNX file in a scratch directory before handing it
to the toolkit.

Usage
-----
    PYTHONPATH=python python tools/convert_rknn.py \
        --onnx artifacts/candidate-v0.onnx \
        --output models/candidate-v0.fp.rknn --no-quant \
        --report reports/candidate-v0.fp.convert.json

    PYTHONPATH=python python tools/convert_rknn.py \
        --onnx artifacts/candidate-v0.onnx \
        --output models/candidate-v0.int8.rknn \
        --dataset models/calib/dataset.txt \
        --report reports/candidate-v0.int8.convert.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TARGET = "rk3588"
DEFAULT_INPUT_SHAPE = [1, 2, 1025, 345]
DEFAULT_OUTPUT_SHAPE = [1, 4, 1025, 345]


class ConversionError(RuntimeError):
    """Raised for any conversion failure; main() turns it into a nonzero exit."""


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
    except Exception:  # pragma: no cover - only observable on non-RKNN hosts
        return None


def _shape_of(value_info) -> list[int]:
    dims = value_info.type.tensor_type.shape.dim
    return [int(dim.dim_value) for dim in dims]


def prepare_onnx(onnx_path: Path, work_dir: Path) -> dict:
    """Validate the (possibly external-data) ONNX and re-save a self-contained copy.

    Returns a dict of provenance facts for the conversion report.
    """
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - onnx is a project dependency
        raise ConversionError(f"the 'onnx' package is required to inspect the model: {exc}") from exc

    if not onnx_path.is_file():
        raise ConversionError(f"ONNX model not found: {onnx_path}")

    # 1. Inspect external-data references without loading the weights.  (The checker
    # needs the sidecar files present before it will accept the graph, so resolve
    # every reference by hand first and only then run the full check.)
    bare = onnx.load_model(str(onnx_path), load_external_data=False)

    references: dict[str, int] = {}
    for tensor in bare.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        location = next((entry.value for entry in tensor.external_data if entry.key == "location"), None)
        if not location:
            raise ConversionError(f"external-data initializer {tensor.name!r} has no location entry")
        references[location] = references.get(location, 0) + 1

    external_files = []
    for location, count in sorted(references.items()):
        data_path = onnx_path.parent / location
        if not data_path.is_file():
            raise ConversionError(
                f"ONNX external data file missing: {data_path} "
                f"(the runtime needs it beside {onnx_path.name}; "
                f"download it from the models-v0 release asset)"
            )
        external_files.append(
            {
                "location": location,
                "initializers": count,
                "bytes": data_path.stat().st_size,
                "sha256": sha256_file(data_path),
            }
        )

    # 2. Load with weights, check the full graph, and inline the weights into one file.
    model = onnx.load_model(str(onnx_path), load_external_data=True)
    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001
        raise ConversionError(f"ONNX checker rejected the model with external data loaded: {exc}") from exc

    work_dir.mkdir(parents=True, exist_ok=True)
    inlined = work_dir / f"{onnx_path.stem}.inlined.onnx"
    onnx.save_model(model, str(inlined), save_as_external_data=False)
    if inlined.with_suffix(inlined.suffix + ".data").exists() or inlined.with_suffix(".data").exists():
        raise ConversionError(f"failed to inline external data; a data sidecar was still written for {inlined}")

    opset = {opset.domain or "ai.onnx": int(opset.version) for opset in model.opset_import}
    input_shapes = {value.name: _shape_of(value) for value in model.graph.input}
    output_shapes = {value.name: _shape_of(value) for value in model.graph.output}

    facts = {
        "source_onnx": str(onnx_path),
        "source_onnx_sha256": sha256_file(onnx_path),
        "source_external_data": external_files,
        "inlined_onnx": str(inlined),
        "inlined_onnx_sha256": sha256_file(inlined),
        "opset": opset,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
    }
    expected_input = list(DEFAULT_INPUT_SHAPE)
    expected_output = list(DEFAULT_OUTPUT_SHAPE)
    actual_inputs = list(input_shapes.values())
    actual_outputs = list(output_shapes.values())
    if actual_inputs != [expected_input] or actual_outputs != [expected_output]:
        raise ConversionError(
            "unexpected ONNX boundary: "
            f"inputs={input_shapes} outputs={output_shapes}; "
            f"expected input {expected_input} and output {expected_output}"
        )
    return facts


def _rknn_step(rknn, name: str, function, *args, **kwargs) -> None:
    """Call one RKNN API step and raise a step-tagged error if it fails."""
    try:
        result = function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - toolkit raises plain exceptions on bad models
        raise ConversionError(f"RKNN step '{name}' raised: {exc}") from exc
    if result != 0:
        raise ConversionError(f"RKNN step '{name}' failed with return code {result} (see toolkit log above)")


def convert(args: argparse.Namespace) -> dict:
    onnx_path = Path(args.onnx)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.no_quant and args.dataset:
        raise ConversionError("--dataset enables INT8 quantisation and conflicts with --no-quant; pick one")
    if not args.no_quant and not args.dataset:
        raise ConversionError(
            "INT8 conversion requires --dataset <dataset.txt> (from tools/make_calibration.py); "
            "pass --no-quant for a float (fp16) build"
        )

    report: dict = {
        "schema": 1,
        "purpose": "RKNN conversion report (engineering probe)",
        "evidence_scope": {
            "proven_by_this_report": "the fixed-shape ONNX model was converted into a target .rknn binary",
            "not_proven": [
                "on-board Edge2 / RK3588 NPU execution (requires RKNN-Toolkit-Lite2 on the board)",
                "real-time throughput on the board",
                "listening / separation quality",
            ],
        },
        "target": args.target,
        "quantized": not args.no_quant,
        "precision": "int8" if not args.no_quant else "float16",
        "toolkit_version": toolkit_version(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    work_dir = Path(tempfile.mkdtemp(prefix="rknn-convert-"))
    report["work_dir"] = str(work_dir)
    try:
        report.update(prepare_onnx(onnx_path, work_dir))
        inlined = Path(report["inlined_onnx"])

        dataset: Path | None = None
        if args.dataset:
            dataset = Path(args.dataset)
            if not dataset.is_file():
                raise ConversionError(f"calibration dataset not found: {dataset}")
            entries = [line.strip() for line in dataset.read_text().splitlines() if line.strip()]
            if not entries:
                raise ConversionError(f"calibration dataset is empty: {dataset}")
            missing = [entry for entry in entries if not (dataset.parent / entry).is_file()]
            if missing:
                raise ConversionError(f"calibration entries missing next to {dataset}: {missing[:5]}")
            report["dataset"] = {
                "path": str(dataset),
                "entries": len(entries),
                "sha256": sha256_file(dataset),
                "files": entries,
            }
        else:
            report["dataset"] = None

        try:
            from rknn.api import RKNN
        except Exception as exc:  # noqa: BLE001 - missing or broken toolkit install
            raise ConversionError(
                "rknn-toolkit2 is not importable in this environment "
                f"({exc!r}). Official wheels exist for Linux x86_64/aarch64 only; "
                "run this tool inside .github/workflows/rknn.yml (ubuntu-22.04, Python 3.11)."
            ) from exc

        started = time.time()
        rknn = RKNN(verbose=bool(args.verbose))
        try:
            _rknn_step(rknn, "config", rknn.config, target_platform=args.target)
            _rknn_step(rknn, "load_onnx", rknn.load_onnx, model=str(inlined))
            _rknn_step(
                rknn,
                "build",
                rknn.build,
                do_quantization=not args.no_quant,
                dataset=str(dataset) if dataset is not None else None,
            )
            _rknn_step(rknn, "export_rknn", rknn.export_rknn, str(output))
        finally:
            try:
                rknn.release()
            except Exception:  # noqa: BLE001 - release must not mask the real failure
                pass
        report["build_time_sec"] = round(time.time() - started, 3)

        if not output.is_file():
            raise ConversionError(f"export_rknn reported success but {output} does not exist")
        report["rknn"] = str(output)
        report["rknn_sha256"] = sha256_file(output)
        report["rknn_bytes"] = output.stat().st_size
        report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return report
    finally:
        # The published model stays beside the release assets; the scratch tree is disposable.
        if not args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
            report["work_dir"] = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert the fixed-shape ONNX model to RKNN (engineering probe).",
    )
    parser.add_argument("--onnx", required=True, help="published ONNX model; its .onnx.data must sit beside it")
    parser.add_argument("--output", required=True, help="output .rknn path")
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"RKNN target platform (default: {DEFAULT_TARGET})")
    parser.add_argument("--dataset", help="calibration dataset.txt from tools/make_calibration.py; enables INT8")
    parser.add_argument("--no-quant", action="store_true", help="build a float16 (non-quantised) model")
    parser.add_argument("--report", help="write the JSON conversion report to this path")
    parser.add_argument("--verbose", action="store_true", help="pass verbose=True to RKNN-Toolkit2")
    parser.add_argument("--keep-work-dir", action="store_true", help="keep the inlined-ONNX scratch directory")
    args = parser.parse_args(argv)

    try:
        report = convert(args)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - last-resort clarity for CI logs
        print(f"error: unexpected conversion failure: {exc}", file=sys.stderr)
        raise

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote report {report_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
