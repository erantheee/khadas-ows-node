# khadas-ows-node

Minimal Khadas Edge2 network-audio relay + lightweight stem-separation probe. This
repository ports the portable processing core of the previous ESP32/CoreS3 demo to the
board that was chosen as the more serious hardware candidate (Khadas Edge2, RK3588S2,
6 TOPS NPU), and asks one narrow question: **can the lightweight separation model run on
the Edge2 NPU in real time, with a safe fallback, without pretending that deployment
equals listening quality?**

Engineering success and listening acceptance are deliberately separate. Read
[`docs/handoff.md`](docs/handoff.md) for the current state and
[`docs/proposal.md`](docs/proposal.md) for the frozen scope.

## Current status (2026-11-06)

| Evidence | Result |
| --- | --- |
| Checkpoint is the unchanged legacy candidate | PASS — `strict=True` load, masks/stems bit-identical (max abs error 0.0), 550,816 params (`tools/verify_model_parity.py`) |
| C++ core behaviour (golden + failure matrix) | PASS — 5/5 CTest on macOS and on the Edge2: conversation W1 golden 67 assertions, failure matrix 141, server e2e 19 |
| ONNX export + PyTorch/ONNXRuntime parity | PASS — input `[1,2,1025,345]`, output `[1,4,1025,345]`, opset 18, max abs error 2.4e-7, cosine 1.0 |
| ONNX → RKNN conversion (x86_64 CI, RKNN-Toolkit2 2.5.0) | PASS — fp16 and INT8 `.rknn` produced; simulator parity vs ONNX: fp16 cos 0.99999998 / max abs 4.2e-4, INT8 cos 0.99993 |
| **Edge2 on-board NPU execution** | **PASS — `actual_mode=npu` on a physical Edge2**; fp16 model ~0.97 s per 4.005 s window (RTF ≈ 0.24), NPU core 0 measured 71–74 % busy during inference |
| 60 s real-time network stream in `npu` mode | PASS — 15/15 chunks, p95 RTT 1.92 s, 0 deadline misses, 0 fallbacks, 0 reconnects |
| 30-minute board soak (realtime input, NPU mode) | **PASS** — 450/450 chunks, 1800.0 s audio, realtime ratio 1.0009, RTT p50 1.60 s / p95 1.91 s / p99 2.20 s, 0 deadline misses, 0 fallbacks, 0 reconnects, `actual_mode=npu` on 1806/1806 status samples, max SoC temp 53.6 °C, 0 throttle events → [`docs/soak-report-npu-20261106.json`](docs/soak-report-npu-20261106.json) |
| Human blind listening (passthrough / CPU / NPU) | **NOT RUN — belongs to the human listener**; tooling and blinded material prepared (`tools/listening_ab.py`, `docs/listening-material/`) |

## Architecture

```
computer (player / capture)                Khadas Edge2 (RK3588S2)
┌─────────────────────┐   TCP PCM        ┌──────────────────────────────────────────────┐
│ tools/pcm_client.py │ ───────────────▶ │ C++ core  (src/audio_core.cpp)               │
│ tools/run_stream.py │  44.1 kHz stereo │  • ring buffer + mode switch                 │
│ tools/soak_edge2.py │   float32 +      │  • passthrough / cpu / npu                   │
│                     │   packet header  │  • worker timeout → bit-exact fallback       │
│                     │ ◀─────────────── │  • conversation policy → smoothed duck gain  │
└─────────────────────┘                  │  • HTTP /v1/status /v1/mode /v1/evidence     │
                                         └───────────────┬──────────────────────────────┘
                                                         │ loopback TCP (one packet at a time)
                                                         ▼
                                         ┌──────────────────────────────────────────────┐
                                         │ Python worker (python/khadas_ows/worker.py)  │
                                         │  CPU: PyTorch checkpoint                     │
                                         │  NPU: RKNN-Toolkit-Lite2 (RK3588)            │
                                         │  STFT(2048/512, Hann) → fixed-shape masks    │
                                         │  → masked ISTFT, peak-matched, same length   │
                                         └──────────────────────────────────────────────┘
```

Key boundaries:

- The C++ core never blocks passthrough on Python: a missing, hung or non-finite worker
  produces a **bit-exact passthrough** reply and a visible `fallback_reason` in `/v1/status`.
- The neural body is the unchanged training architecture. Stream STFT / complex mask
  application / ISTFT live outside the model, so ONNX/RKNN only ever sees a fixed-shape
  real-valued magnitude tensor `[1,2,1025,345]` and returns four softmax masks.
- The conversation policy is the faithful port of the ESP32 state machine (500 ms preview
  duck, 800 ms confirmation, 2500 ms hold, event-frame neutrality, dialogue veto). It is
  driven by control-rate evidence frames over `POST /v1/evidence` and applies a smoothed
  media duck; stale evidence fails open to idle/100 % (no media held down).

## Build and test

```bash
# C++ core + golden/failure tests (macOS or Edge2 aarch64)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
ctest --test-dir build --output-on-failure

# Python model/worker/network tests
python3.11 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q
```

## Run locally (Mac/CPU)

```bash
./build/khadas-ows-node                        # audio TCP 9100, HTTP 8787
PYTHONPATH=python .venv/bin/python -m khadas_ows.worker --mode cpu --model models/candidate-v0.pt
curl -X POST --data cpu http://127.0.0.1:8787/v1/mode
.venv/bin/python tools/pcm_client.py in.wav out.wav     # CPU output
.venv/bin/python tools/make_fixture.py --output /tmp/fixture.wav
```

Ports and timeouts are configurable: `KHADAS_AUDIO_PORT`, `KHADAS_HTTP_PORT`,
`KHADAS_CPU_PORT`, `KHADAS_NPU_PORT`, `KHADAS_CLIENT_TIMEOUT_MS` (default 10 000),
`KHADAS_WORKER_TIMEOUT_MS` (default 1500), `KHADAS_EVIDENCE_TTL_MS` (default 2000).

## HTTP interface

| Endpoint | Meaning |
| --- | --- |
| `GET /v1/status` | requested/actual mode, fallback reason + counters, model hash, buffer watermark, last RTF, policy state / media %, evidence staleness |
| `POST /v1/mode` | body `passthrough` \| `cpu` \| `npu` |
| `POST /v1/evidence` | one control-rate evidence frame (JSON: `speech_active`, `voice_likelihood`, `conversation_cue`, `own_voice`, `level_dbfs`, `event_blocked`, `dialogue_veto`, `elapsed_ms`) |

## Model boundary and numbers

- Input ONNX: `magnitude` `[1,2,1025,345]` float32 (4.005 s window at 44.1 kHz).
- Output ONNX: `masks` `[1,4,1025,345]` float32, softmax over stems.
- Mac CPU worker: last RTF ≈ 0.44 per 4 s window (PyTorch, this Mac).
- Edge2 NPU, fp16 `.rknn`: ≈ 0.97 s per 4.005 s window → RTF ≈ 0.24, NPU busy.
- Edge2 NPU, INT8 `.rknn`: ≈ 4.1–4.4 s per window (RTF ≈ 1.07) — **not real-time** on this
  board: the INT8 build inserts CPU `Transpose` fallbacks for oversized tensors. The realtime
  claim therefore uses the fp16 artifact and says so.
- The 4.005 s window is a **throughput probe**, not a product latency claim. Real-time
  separation needs a smaller fixed window and a re-exported/re-trained model.

## Edge2 deployment and acceptance

```bash
# 1. convert/verify ONNX->RKNN on a GitHub-hosted x86_64 runner (workflow_dispatch)
gh workflow run rknn.yml --repo erantheee/khadas-ows-node --ref main
gh run download <run-id> --dir /tmp/rknn-out

# 2. provision + deploy + start node/worker on the board
tools/deploy_edge2.sh --host khadas@<board-ip> \
  --rknn /tmp/rknn-out/rknn-artifacts/models/candidate-v0.fp.rknn \
  --lite2-wheel /path/to/rknn_toolkit_lite2-2.5.0-cp312-...-aarch64.whl

# 3. realtime stream + 30-minute soak
.venv/bin/python tools/run_stream.py --host <board-ip> --mode npu --chunk-ms 4000 --seconds 60
.venv/bin/python tools/soak_edge2.py --host <board-ip> --mode npu --minutes 30 \
  --chunk-ms 4000 --max-unrecovered-seconds 15 --ssh khadas@<board-ip>
```

Full board steps, failure semantics and debugging commands: [`docs/edge2-runbook.md`](docs/edge2-runbook.md).

## Human listening (only the human decides)

`tools/listening_ab.py` loudness-matches (BS.1770) the three renditions, randomises blind
labels, helps playback and records scores; it never scores quality itself. Prepared material
and instructions: [`docs/listening-material/README.md`](docs/listening-material/README.md).
The candidate model's measured SDR is around 0 dB (no meaningful separation), so a
“deployment success, quality not accepted” conclusion is the expected outcome.

## Honest boundaries

- Automated tests prove software behaviour only; ONNX export does not prove RKNN conversion;
  conversion does not prove NPU execution; NPU execution does not prove listening quality.
- INT8 is not the realtime artifact on this board (see above); only fp16 was measured real-time.
- Edge2 DMIC capture, Bluetooth relay, phone-direct topology, production hardware and
  full ESP32 peripheral migration are **out of scope** for this first version.
- Thermal/throttling evidence is collected by the soak only when `--ssh` is supplied; a soak
  without it says “throttling NOT verified” in its own report.

## Repository layout

```
src/ include/          C++ audio core, conversation policy port, HTTP/TCP servers
python/khadas_ows/     fixed-shape streaming worker (torch / rknnlite / stub backends)
tools/                 convert/verify ONNX+RKNN, pcm client, stream probe, soak, deploy,
                       listening A/B, deterministic fixture and calibration generators
tests/                 C++ golden/failure-matrix/server tests + Python model/worker/network tests
docs/                  handoff, proposal, RKNN verification, Edge2 runbook, listening material,
                       recorded evidence (docs/evidence/)
.github/workflows/     ci.yml (build/test + ASan/UBSan) and rknn.yml (x86_64 conversion)
```
