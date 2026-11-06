# AGENTS.md — khadas-ows-node

Read `docs/handoff.md` (current state, board run facts, next steps) and `docs/proposal.md`
(frozen scope) before changing code. `docs/rknn-verification.md` records the RKNN/NPU
evidence chain; `docs/edge2-runbook.md` is the board acceptance procedure.

## Scope

This repository is the minimal computer↔Khadas Edge2 PCM and RKNN engineering probe. Do not
add Bluetooth relay, Edge2 DMIC capture, mobile capture, production hardware claims or
automatic listening-quality judgments without a separate decision.

## Evidence language (do not overstate)

- Automated tests prove software behaviour only.
- ONNX export does not prove RKNN conversion.
- RKNN conversion does not prove Edge2 NPU execution; simulator parity is not NPU parity.
- NPU execution does not prove acceptable listening quality.
- Listening acceptance is recorded by a human (`docs/listening-template.md`,
  `tools/listening_ab.py`); no metric, SDR or AI substitutes for it.
- The realtime claim uses the **fp16** `.rknn`; the INT8 artifact is converted and verified
  but measured slower than real-time on this board. Do not relabel that as a success.

## Verification

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
ctest --test-dir build --output-on-failure        # 5/5: golden, failure matrix, server e2e
.venv/bin/pytest -q                                # model/worker/ONNX-parity/network
```

CI: `.github/workflows/ci.yml` (build+test and ASan/UBSan) and `.github/workflows/rknn.yml`
(x86_64 RKNN conversion + simulator parity, `workflow_dispatch`).

## Runtime knobs worth knowing

- `KHADAS_AUDIO_PORT` (9100), `KHADAS_HTTP_PORT` (8787), `KHADAS_CPU_PORT` (9201),
  `KHADAS_NPU_PORT` (9202).
- `KHADAS_CLIENT_TIMEOUT_MS` (10000): accepted-client idle timeout. Must exceed the largest
  inter-packet gap — the fixed-shape NPU stream sends one 4.005 s packet every 4 s.
- `KHADAS_WORKER_TIMEOUT_MS` (1500; board runs 6000): per-call worker timeout before the
  bit-exact passthrough fallback.
- `KHADAS_EVIDENCE_TTL_MS` (2000): stale `/v1/evidence` fail-open release of the media duck.
- NPU worker packets must be ≈4 s (one fixed-shape window per packet); use
  `--chunk-ms 4000` and `soak_edge2.py --max-unrecovered-seconds 15` for board runs.

## Board access

The physical Edge2 used for the 2026-11-06 run is `khadas@172.16.20.18` (Khadas Ubuntu 24.04,
aarch64). The SSH key used locally is not in the repository; deployments are driven by
`tools/deploy_edge2.sh`. Never commit board credentials or model binaries (`.gitignore` covers
`*.pt/*.onnx/*.rknn/*.wav`).
