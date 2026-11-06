# RKNN conversion verification — pending

Status: conversion has not run yet on this checkout. This file is filled from the
`rknn-convert` GitHub Actions run (`workflow_dispatch`, ubuntu-22.04 x86_64) together with
the model parity evidence in `tools/verify_model_parity.py`.

Evidence chain used by this repository (each step proves only its own step):

1. checkpoint audit — `tools/verify_model_parity.py` (strict load + bit-identical masks).
2. ONNX export — `tools/export_onnx.py` + `tools/verify_onnx.py` (torch vs onnxruntime).
3. RKNN conversion + simulator parity — CI `rknn-convert` workflow artifacts.
4. Edge2 NPU execution — `docs/edge2-runbook.md` (requires the physical board).
5. Listening acceptance — `docs/listening-template.md` + `tools/listening_ab.py` (human only).
