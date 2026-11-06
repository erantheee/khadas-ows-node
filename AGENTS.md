# AGENTS.md — khadas-ows-node

Read `docs/handoff.md` and `docs/proposal.md` before changing code.

## Scope

This repository is the minimal computer↔Khadas Edge2 PCM and RKNN engineering probe. Do not add Bluetooth relay, Edge2 DMIC, mobile capture, production hardware claims or automatic listening-quality judgments without a separate decision.

## Evidence language

- Automated tests prove software behavior only.
- ONNX export does not prove RKNN conversion.
- RKNN conversion does not prove Edge2 NPU execution.
- NPU execution does not prove acceptable listening quality.
- Listening acceptance is recorded by a human in `docs/listening-template.md`.

## Verification

```bash
cmake -S . -B build && cmake --build build --parallel
ctest --test-dir build --output-on-failure
.venv/bin/pytest -q
```

