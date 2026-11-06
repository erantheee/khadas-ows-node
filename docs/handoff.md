# Handoff — khadas-ows-node (2026-11-06)

Read `README.md` for the architecture and `docs/proposal.md` for the frozen scope.
This file is the current state: what was verified, how, on which objects, and what is next.

## TL;DR

The first-version engineering question of the approved plan — *does the lightweight
separation model convert to RKNN and run on the Edge2 NPU in real time with a safe
fallback?* — is **answered yes on a physical Edge2**, with the fp16 artifact, while the
INT8 artifact is converted/verified but **not** real-time on this board. Separation quality
is unchanged and poor (≈0 dB SDR); the listening conclusion is still the human listener's.

## Board and access used for the on-board run

- Khadas Edge2 (RK3588S2), Ubuntu 24.04.3 LTS, kernel 6.1.118 aarch64, Python 3.12.3,
  7.9 GB RAM, RKNPU kernel driver v0.9.8, `librknnrt` 2.5.0, RKNN-Toolkit-Lite2 2.5.0.
- ssh target used: `khadas@172.16.20.18` (LAN). The key used locally is
  `~/.ssh/khadas_dsh_ed25519`; **no credentials are committed to this repository**.
- Board install lives at `~/khadas-ows-node` (`build/`, `.venv/`, `artifacts/`, `logs/` are
  all git-ignored / board-local).

## Verified on 2026-11-06 (each row is backed by a file or a CI run)

| # | Claim | Evidence |
| --- | --- | --- |
| 1 | New `MaskNetwork` is the unchanged legacy candidate: strict load, masks and stems bit-identical, 550,816 params, checkpoint SHA matches quarantine manifest | `tools/verify_model_parity.py` → `docs/evidence/checkpoint-parity.json` |
| 2 | C++ golden + failure-matrix suites pass on macOS **and** on the Edge2 (aarch64) | 5/5 CTest, conversation W1 golden 67 assertions (0 false confirmations), failure matrix 141, server e2e 19; `docs/evidence/board-ctest.txt` |
| 3 | Conversation policy is the faithful legacy port and is wired into the audio path (duck gain, fail-open on stale evidence) | `include/khadas_ows/conversation.hpp`, `src/conversation.cpp`, `tests/conversation_golden_test.cpp`, `tests/failure_matrix_test.cpp` |
| 4 | ONNX export is numerically identical to PyTorch | torch vs onnxruntime max abs 2.38e-7, cosine 1.0 → `docs/evidence/onnx-parity.json` |
| 5 | ONNX → RKNN conversion works on a supported x86_64 host | CI [run 42186101812](https://github.com/erantheee/khadas-ows-node/actions/runs/42186101812) green (5m01s), fp16 + INT8 artifacts, simulator parity → `docs/rknn-verification.md`, `docs/evidence/rknn/` |
| 6 | **Edge2 on-board NPU execution** | `actual_mode=npu`; fp16 ≈ 0.97 s per 4.005 s window (RTF ≈ 0.24), NPU core 0 71–74 % busy; INT8 ≈ 4.3 s (RTF ≈ 1.07, not real-time) → `docs/evidence/board-npu-bench.json` |
| 7 | Real-time network stream in `npu` mode | 60 s, 15/15 chunks, p95 1.92 s, 0 misses/fallbacks/reconnects → `docs/evidence/stream-npu-60s.json` |
| 8 | 30-minute soak in `npu` mode | **PASS** — 450/450 chunks, 1800.0 s audio, realtime ratio 1.0009, RTT p50 1.60 s / p95 1.91 s / p99 2.20 s, 0 misses/fallbacks/reconnects, `actual_mode=npu` 1806/1806 samples, SoC max 53.6 °C, 0 throttle events → `docs/soak-report-npu-20261106.json` (an earlier aborted attempt is kept as `docs/evidence/soak-attempt1-aborted.json` and explained in `docs/rknn-verification.md` §5) |
| 9 | Human blind listening | **NOT RUN.** Material and tooling prepared: `docs/listening-material/`, `tools/listening_ab.py` |

## Room state at handoff (board)

The board currently runs the verified services left over from this session (clean them up or
reuse them):

```text
node      ./build/khadas-ows-node            audio TCP 9100, HTTP 8787
worker    .venv/bin/python -m khadas_ows.worker --mode npu --model artifacts/candidate-v0.fp.rknn
env       KHADAS_MODEL_HASH=<fp16 sha> KHADAS_CLIENT_TIMEOUT_MS=15000 KHADAS_WORKER_TIMEOUT_MS=6000
logs      ~/khadas-ows-node/logs/{server.log,worker-npu.log}
stop      pkill -x khadas-ows-node ; pkill -f "khadas_ows.worker"
```

## How to reproduce

Local (Mac) verification:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
ctest --test-dir build --output-on-failure          # 5/5
.venv/bin/pytest -q                                  # 13 tests
.venv/bin/python tools/verify_model_parity.py --legacy-root <cores3-noise-adapt-demo>
.venv/bin/python tools/verify_onnx.py --onnx models/candidate-v0.onnx --checkpoint models/candidate-v0.pt
```

RKNN conversion (must run on Linux x86_64 — the Mac cannot run RKNN-Toolkit2):

```bash
gh workflow run rknn.yml --repo erantheee/khadas-ows-node --ref main
gh run watch <run-id> --repo erantheee/khadas-ows-node --exit-status
gh run download <run-id> --repo erantheee/khadas-ows-node --dir /tmp/rknn-out
```

Board deploy + acceptance: `docs/edge2-runbook.md` (bootstrap, Lite2 wheel, deploy,
`run_stream.py`, `soak_edge2.py`, thermal evidence, failure semantics).

Human listening (interactive terminal required; the tool refuses to fake it):

```bash
.venv/bin/python tools/listening_ab.py \
  --render docs/listening-material/renditions \
  --out docs/listening-material/blind-$(date +%Y%m%d-%H%M%S) \
  --build-hash 80ff171
```

## Important integration lessons from the board run

1. **4 s packet cadence vs 1 s socket timeout.** The C++ node's accepted-client idle timeout
   was hard-coded to 1000 ms; a fixed-shape NPU stream sends one 4 s packet every 4 s, so the
   node closed the connection between packets and the client reported “unrecoverable stream
   loss”. Fixed in `80ff171` via `KHADAS_CLIENT_TIMEOUT_MS` (default 10 s).
2. **Worker timeout margin.** The per-call worker timeout was hard-coded to 1500 ms while the
   fp16 inference is ~1.0–1.2 s; it is now `KHADAS_WORKER_TIMEOUT_MS` (default 1500 ms, board
   runs 6000 ms). Explicit `process(..., timeout_ms)` values still win, so failure-path tests
   stay deterministic.
3. **Warm-up.** The first inference after `init_runtime()` costs seconds; the rknnlite backend
   now warms the graph once at startup so the first audio packet is not misread as a timeout.
4. **Soak silence threshold scales with packet size.** `soak_edge2.py` defaults to “no
   successful chunk for 5 s = unrecoverable”, which is correct for 100 ms packets and too tight
   for 4 s packets; use `--max-unrecovered-seconds 15` with `--chunk-ms 4000` (and do not run
   heavy work on the driver host during the soak).
5. **INT8 is not automatically faster.** The INT8 artifact is ~4× slower than fp16 on this
   board (CPU `Transpose` fallbacks for oversized tensors), so the realtime claim is fp16-only.

## Listening status (2026-11-06, first human session)

- The user listened to the real-music material (live QQ Music capture): CPU and NPU renditions
  are indistinguishable, separation is acceptable to the listener overall, **the drums stem
  sounds abnormal** (measured: −45 dBFS vs −11.7 dBFS mixture, 5.6 kHz centroid, drums SDR
  ≈ 0 dB, estimate 23–34 dB below truth; stem order verified by a 4×4 SDR matrix; checkpoint
  trained only 6/20 epochs — see `docs/evidence/stem-drums-analysis-20261106.json`).
- Accepted outcome for this version (user): "以后再说吧，主要是能分开，你也能调包络" —
  separation is demonstrable and the envelope-modulation chain is implemented and adjustable;
  the drums/model-quality fix (continued training) is **deferred** and was not started.
- The envelope chain is implemented in `python/khadas_ows/strategy.py` +
  `worker.py --output-mode envelope`. Tunable knobs: `--scenario
  quiet|broadband|keyboard|wind|babble`, `--strategy-strength 0..1`, `--scenario-start-ms /
  --scenario-end-ms`; role targets (`SCENARIO_TARGETS`), role mapping (`ROLE_MAP`), hysteresis
  (`RoleTracker`: 2 dB / 2 windows) and envelope times (`EnvelopeFollower`: 400 ms attack /
  1200 ms release) are plain constants in `strategy.py`.
- Rendered material (git-ignored WAVs): `docs/listening-material/qq-music/stems/` (single
  stems) and `docs/listening-material/qq-music-envelope-{broadband,keyboard,broadband-strong}/`
  (envelope-modulated music + blind sets); see `docs/listening-material/README.md`.

## Next steps (in priority order)

1. **Human listening** — first session done (see above); remaining re-listening after any
   model/strategy change optional.
2. **Model quality (deferred by user, "以后再说")**: the candidate is undertrained (6/20
   epochs) and drums is unusable; options are (a) resume training from the checkpoint with
   drums-weighted loss (~1.5–2.5 h on MPS), or (b) train a 2.59M-class Band-SCNet from scratch
   (industry precedent SDR 7.79 dB, longer and needs an NPU RTF re-check). Do not start without
   the user's go-ahead.
3. **Product-latency model**: re-export/re-train with a much smaller fixed window (e.g. 32–64
   frames) and re-measure NPU RTF and stability; the current 4.005 s window is a throughput
   probe only.
4. **INT8 root cause**: remove/absorb the oversized `Transpose` (layout-friendly export, or
   keep the mask head in CPU-friendly form) to get the INT8 speed advantage back; also
   investigate the single-core fallback for 6 conv layers.
5. **Async/pipelined node↔worker protocol**: the current protocol is strictly synchronous, so
   one 4 s window blocks the audio path for ~1–2 s; a pipelined design is required for real
   product latency (and would keep the p99 honest).
6. Deferred by the approved plan: Edge2 DMIC re-calibration and perception integration
   (Phase 2B), Bluetooth/phone relay topology, full ESP32 peripheral migration, production
   hardware and model training decisions.

## Evidence index

```
docs/evidence/checkpoint-parity.json      legacy↔new model bit-parity
docs/evidence/onnx-parity.json            PyTorch ↔ ONNXRuntime
docs/evidence/rknn/*.json                 conversion + simulator parity reports (CI)
docs/evidence/board-npu-bench.json        Edge2 NPU latency / NPU load / INT8 caveat
docs/evidence/board-ctest.txt             C++ suite on the board (aarch64)
docs/evidence/stream-npu-60s.json         60 s real-time stream, npu mode
docs/evidence/soak-attempt1-aborted.json  first soak attempt (aborted; cause documented)
docs/soak-report-npu-20261106.json        30-minute soak verdict (final run)
docs/listening-material/                  blinded renditions + instructions
```
