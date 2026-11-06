# Khadas OWS Node — minimal engineering proposal

Approved plan: "Khadas Edge2 极简迁移计划". First version answers one engineering question —
can the ESP32/CoreS3 processing core and the lightweight separation candidate be carried by
the chosen Khadas Edge2 (RK3588S2) in a network PCM relay, with a safe fallback? — and keeps
listening acceptance deliberately separate.

## Frozen scope (first version)

- Computer ↔ Edge2 network PCM relay (44.1 kHz stereo float32, small binary header).
- Modes `passthrough` / `cpu` / `npu`; minimal `GET /v1/status`, `POST /v1/mode` and
  `POST /v1/evidence` interface with requested/actual mode, model hash, buffer watermark,
  RTF, fallback reason, policy state.
- Ported, platform-independent C++ conversation policy (legacy 500 ms preview / 800 ms
  confirmation / 2500 ms hold semantics, event-frame neutrality, dialogue veto) with the
  legacy W1 golden timeline; smoothed media duck; fail-open on stale evidence.
- Fixed-shape model boundary: stream STFT / complex-mask application / ISTFT on the CPU side,
  ONNX/RKNN only for the real-valued `[1,2,1025,345]` → four-mask network body.
- PyTorch → ONNX → RKNN conversion with simulator parity, on a supported x86_64 runner.
- Hard rule: CPU and passthrough paths always remain; the NPU must never be a single point of
  failure; every failure path returns bit-exact passthrough with a visible reason.

Explicitly **out of scope** for this first version: phone-direct/Bluetooth relay, Edge2 DMIC
re-calibration and perception integration, full ESP32 peripheral/UI migration, production
hardware form factor, model retraining/quality acceptance.

## Implemented

- C++ audio core with ring buffer, bounded buffering, poll-based TCP/HTTP servers, graceful
  SIGTERM shutdown, configurable ports/timeouts.
- Faithful legacy conversation state machine + audio-path duck gain + evidence ingestion API.
- CPU (PyTorch) and NPU (RKNN-Toolkit-Lite2) workers with a fixed-shape streaming window
  manager, same-length replies, explicit backend-unavailable errors and model-hash checking.
- Conversion/verification tooling (`tools/convert_rknn.py`, `tools/verify_rknn.py`,
  `tools/verify_onnx.py`, `tools/verify_model_parity.py`) and a `workflow_dispatch` CI job.
- Measurement/acceptance tooling: PCM client, real-time stream probe, 30-minute soak driver
  with thermal sampling, idempotent Edge2 deploy script, human blind A/B tool.
- Tests: C++ golden/failure-matrix/server suites (5 executables) and Python model/worker/
  ONNX-parity/network tests (13 tests); CI also builds with ASan/UBSan.

## Validated (2026-11-06)

- Legacy checkpoint ↔ new model definition: strict load, bit-identical masks/stems.
- ONNX ↔ PyTorch: max abs error 2.38e-7.
- ONNX → RKNN (x86_64 CI): fp16 and INT8 artifacts + simulator parity (fp16 cos 0.99999998).
- **On-board Edge2 NPU:** `actual_mode=npu`, fp16 ≈ 0.97 s per 4.005 s window (RTF ≈ 0.24),
  NPU core busy 71–74 %; 60 s real-time network stream passes with 0 fallbacks/reconnects.
- 30-minute soak: engineering verdict recorded in `docs/soak-report-npu-20261106.json`.

## Not yet validated / open

- Human listening acceptance — the only valid judge; tooling is prepared, no conclusion exists.
- Separation quality of the candidate: measured MUSDB SDR ≈ 0 dB, so “deployment success,
  quality not accepted” is the expected outcome and must not be re-labelled.
- INT8 real-time and multi-core NPU utilisation: INT8 is currently ~4× slower than fp16 on the
  board (CPU Transpose fallbacks) and the conversion falls back to single-core for 6 conv layers.
- Product latency: the 4.005 s window is a throughput probe; real-time separation needs a
  smaller fixed window and a re-export/re-trained model, plus an async node↔worker protocol.

## Success language

“NPU engineering success” means: the fixed-shape ONNX model converts to RKNN, runs on the
Edge2 NPU in real time and returns finite, correctly shaped, numerically consistent masks,
while every failure path falls back safely. It does **not** mean the separation sounds good.
Listening quality is decided by a human and recorded separately in
`docs/listening-template.md`.
