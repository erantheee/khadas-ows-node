# RKNN conversion and NPU verification — 2026-11-06

This file records the RKNN evidence chain end to end. Each step proves only its own step.

## 1. Checkpoint audit (reproduced by `tools/verify_model_parity.py`)

- Legacy definition: `work/stem-ab/band_scnet.py::BandSplitNet(n_stems=4, base=128, layers=6)`
  from `erantheee/cores3-noise-adapt-demo@f896df7` (not vendored; pass `--legacy-root`).
- `state_dict(..., strict=True)`: **missing = [], unexpected = []**.
- Masks and reconstructed stems from the new `khadas_ows.model.MaskNetwork` compared with the
  legacy `forward()` on a seeded 2 s stereo input: **mask max abs error 0.0, stem max abs
  error 0.0**, mask-sum max abs error 1.8e-7.
- Parameter count 550,816; checkpoint SHA256 `0989b293…5de` matches
  `models/quarantine/candidate-v0.manifest.json`.
- Evidence file: [`evidence/checkpoint-parity.json`](evidence/checkpoint-parity.json).

## 2. ONNX export and PyTorch/ONNXRuntime parity

- `tools/export_onnx.py` → `magnitude [1,2,1025,345]` → `masks [1,4,1025,345]`, opset 18,
  external data in `candidate-v0.onnx.data`.
- ONNX SHA256 `ef6399c76b41ebffaedfbd7e5fdc9f0e28ab60a5954d7bcfdfa46b0b75a9f1d7`;
  external-data SHA256 `df36fcb7…4f3`.
- `tools/verify_onnx.py --checkpoint models/candidate-v0.pt`:
  torch vs onnxruntime **max abs error 2.38e-7**, mean 1.66e-8, cosine 0.9999999999999967,
  mask-sum error 1.19e-7.
- Evidence file: [`evidence/onnx-parity.json`](evidence/onnx-parity.json).

## 3. ONNX → RKNN conversion (GitHub-hosted x86_64 runner)

RKNN-Toolkit2 publishes Linux x86_64/aarch64 wheels only, so conversion runs in
`.github/workflows/rknn.yml` (`workflow_dispatch`, ubuntu-22.04, Python 3.11,
`rknn-toolkit2==2.5.0` from PyPI, CPU-only torch for the calibration generator).

Green run: **https://github.com/erantheee/khadas-ows-node/actions/runs/42186101812** — success
in 5m01s on 2026-11-06, including ONNX artifact verification, deterministic calibration
generation, both conversions and both simulator parity checks.

| artifact | size | sha256 | build | simulator parity vs ONNXRuntime |
| --- | --- | --- | --- | --- |
| `candidate-v0.fp.rknn` (fp16, non-quantised) | 25,278,331 B | `af92ce83…6a6f` | 15.2 s | cosine **0.9999999842**, max abs 4.19e-4, mean 5.0e-5 → PASS |
| `candidate-v0.int8.rknn` (8-entry deterministic calibration set) | 9,249,288 B | `e77dd0fe…499a` | 65.1 s | cosine **0.99992929**, max abs 1.076e-1 (reported only), mean 3.94e-3 → PASS |

Method note: `load_rknn` + `init_runtime()` cannot run on the PC simulator (the toolkit requires
a target for that path), so `tools/verify_rknn.py` rebuilds the identical graph in-process
(`load_onnx → config → build → init_runtime()` without target) with the same ONNX/target/flags/
calibration set and compares that graph against onnxruntime. The reports state explicitly that
this is simulator parity on the built graph and that the exported `.rknn` itself can only be
loaded/verified on RK3588 hardware. Evidence files: [`evidence/rknn/`](evidence/rknn/).

Build warnings recorded by the toolkit (not fatal): 6 conv layers fall back from 3-core to
single-core mode; INT8 build has 2 `Transpose` ops that fall back to CPU because the tensor
shape exceeds the toolkit limit. The INT8 runtime penalty is visible in §4.

## 4. Edge2 on-board NPU execution (physical board)

Board: Khadas Edge2, RK3588S2, Ubuntu 24.04.3 LTS, kernel 6.1.118 aarch64, Python 3.12.3,
RKNPU kernel driver **v0.9.8**, `librknnrt` **2.5.0**, RKNN-Toolkit-Lite2 **2.5.0**.

- Cross-architecture build + tests: `cmake` + `ctest` on the board → **5/5 pass**
  ([`evidence/board-ctest.txt`](evidence/board-ctest.txt)).
- `load_rknn` rc=0, `init_runtime(core_mask=NPU_CORE_AUTO)` rc=0 for both artifacts;
  runtime reports `target: RKNPU v2, target platform: rk3588, framework: ONNX`.
- Steady-state inference on a 4.005 s window (10 repetitions after warmup):
  - fp16: min 969.5 ms / p50 973.4 ms / max 1031.2 ms → **RTF ≈ 0.24**, real-time.
  - INT8: min 4061.6 ms / p50 4284.1 ms / max 4378.7 ms → **RTF ≈ 1.07**, not real-time.
- NPU utilisation sampled from `/sys/kernel/debug/rknpu/load` during sustained fp16 inference:
  `Core0: 71–74 %` (single-core mode), i.e. the graph really runs on the NPU.
- Service path (node + rknnlite worker over the real TCP protocol): worker logs
  `packet seq=0 frames=176400 rtf=0.276…0.290` per 4 s packet; `/v1/status` converges to
  `requested_mode=npu, actual_mode=npu, fallbacks=0, rejected=0`.
- Evidence file: [`evidence/board-npu-bench.json`](evidence/board-npu-bench.json).

**Artifact choice:** the realtime claim uses the **fp16** `.rknn`. INT8 was converted and
numerically verified, but on this board it is slower than fp16 and misses real-time, so it is
reported as a negative engineering result rather than hidden.

## 5. Real-time stream and 30-minute soak

- 60 s stream, `npu` mode, 4 s packets: 15/15 chunks, p50 1.69 s / p95 1.92 s RTT,
  0 deadline misses, 0 reconnects, `actual_mode=npu`, `fallbacks_delta=0`
  ([`evidence/stream-npu-60s.json`](evidence/stream-npu-60s.json)).
- 30-minute soak: see the report named in the handoff/README (final run) and
  [`evidence/soak-attempt1-aborted.json`](evidence/soak-attempt1-aborted.json) for a first
  aborted attempt. The first attempt was killed by the soak monitor after 73 s because the
  default “no successful chunk for 5 s” threshold is calibrated for 100 ms packets, not for
  the 4 s NPU packet cadence, and because heavy host-side work (PyTorch parity/CPU rendering)
  ran concurrently on the Mac. The second run uses `--max-unrecovered-seconds 15` and an
  otherwise idle host; the criteria were not loosened for chunk coverage, mode engagement,
  fallbacks, realtime ratio or thermal evidence.

## 6. What this chain still does not prove

- On-board NPU **product latency**: the 4.005 s window is a throughput probe.
- Separation/listening quality: the candidate's MUSDB SDR is ≈ 0 dB (vocals −0.16, drums 0.06,
  bass −0.30, other 1.02); the listening judgement is recorded separately by a human.
- INT8 real-time (it is not), multi-core NPU utilisation (conversion falls back to single core).
