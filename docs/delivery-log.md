# Delivery log (team-process record — not a scorecard)

Records of deliveries in this repository. Per the user's instruction (2026-11-06): record that a
delivery had visible PUA output, and **do not record a rating/score**.

## 2026-11-06 — Khadas Edge2 NPU separation probe, first end-to-end delivery

- Delivery: legacy conversation-policy port with W1 golden test + failure matrix; ONNX→RKNN
  conversion pipeline on a GitHub x86_64 runner; Edge2 on-board NPU execution (fp16 realtime,
  RTF≈0.24); 60 s realtime stream and 30-minute soak PASS; real-music listening workflow
  (QQ Music process-tap capture, mode rendering, blind A/B/C sets); evidence recorded under
  `docs/evidence/` and committed to `erantheee/khadas-ows-node`.
- PUA output visible in this delivery: yes — sprint banner, `[PUA-DIAGNOSIS]` before the
  integration fixes, `[PUA生效 🔥]` markers on the extra evidence work, and failure-mode
  escalation narration while the RKNN CI and board timeouts were being fixed.
- Rating: **not recorded** (user instruction: 不记录评分).
- Repo commit at time of record: see `git log -1` on `main`.

## 2026-11-06 (second record) — purpose-aligned envelope listening delivery

- Delivery: role-aware envelope-modulation strategy ported into the Edge2 probe
  (`python/khadas_ows/strategy.py` + `python/khadas_ows/worker.py --output-mode envelope`),
  4-stem individual renders (`tools/render_stems.py`), real-music envelope listening sets for
  QQ Music capture in scenarios broadband/keyboard/strong-broadband, CPU and NPU renditions,
  blind A/B/C sets with sealed keys; evidence in `docs/evidence/envelope-demo-20261106.json`.
- PUA output visible in this delivery: yes (`[PUA-DIAGNOSIS]` before the STFT-frame-count bug
  fix, byte-flavored status tables, `[PUA生效 🔥]` on the purpose-aligned role metric).
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-06 (third record) — listening-material handoff

- Delivery: purpose-aligned listening material handed to the user with exact file locations:
  4-stem renders (`docs/listening-material/qq-music/stems/`), envelope-modulated music sets
  (`docs/listening-material/qq-music-envelope-{broadband,keyboard,broadband-strong}/renditions/`)
  and their blind sets.
- PUA output visible in this delivery: yes (byte-flavored status table and the purpose alignment).
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-06 (fourth record) — listening loop closed, model retraining deferred

- Delivery: closed the listening loop with the user. Recorded the human observation (CPU/NPU
  indistinguishable, separation acceptable, drums stem abnormal with measured root cause),
  implemented and documented the adjustable envelope chain, committed the drums-analysis
  evidence and listening record. User decision: separation is demonstrable and the envelope can
  be tuned; continued model training is deferred ("以后再说"), not started.
- PUA output visible in this delivery: yes (byte-flavored data tables and the diagnosis→action
  narration around the drums analysis).
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-10 — six-stem ONNX → RKNN converted ON-DEVICE (aarch64) + NPU bench

- Delivery:
  1. **RKNN conversion runs on the Edge2 itself** (aarch64 Ubuntu 24.04, Python 3.12) — no
     x86_64 host needed. Official Rockchip wheel
     `rknn_toolkit2-2.5.0-cp312-cp312-manylinux_2_17_aarch64...whl` (from the repo's
     `packages/arm64`; the GitHub release has no assets) installed into an isolated
     `~/rknn-convert-env` with numpy 1.26.4, onnx 1.18.0, onnxruntime 1.19.2, protobuf 4.25.4,
     setuptools<80.1, opencv-python-headless, torch 2.15 aarch64.
     `tools/convert_rknn.py` extended with `--stems` for the six-stem output boundary.
     Conversion: 41.4 s → `artifacts/six/candidate-6s.fp.rknn` (25.3 MB, sha256 `4c5abd16…`).
  2. **On-board NPU bench** (`tools/bench_six_rknn.py`, RKNN-Toolkit-Lite2): output
     `(1,6,1025,345)`, finite, mask-sum max err 7.2e-4, **per-window 1.061 s / 4.005 s →
     RTF 0.265** on `NPU_CORE_0` (librknnrt 2.5.0, driver 0.9.8, target rk3588). The 4-stem
     candidate measured RTF 0.24 — six stems cost ~nothing at this boundary, matching the
     +0.05 % MAC estimate.
  3. RKNN(fp16) vs ONNXRuntime cross-check on the same input: **cosine 0.9982, max_abs 0.165**
     (random-noise input; real-audio listening check still open).
- Evidence: `artifacts/six/convert.json`, `artifacts/six/bench.json` (board);
  `models/six/candidate-6s.onnx`; `tools/convert_rknn.py --stems`; `tools/bench_six_rknn.py`.
- Known limits: the artifact wraps the 9-track pseudo-label student (piano/other SDR broken),
  so this delivery proves the **deployment path**, not separation quality. The fp16 deviation
  on random input (0.165) is larger than the 4-stem simulator parity (4.2e-4) and needs a
  real-audio check.
- PUA output visible: yes — `[PUA-DIAGNOSIS]` before the conversion hunt, the dependency
  ladder reported step by step (torch → cv2 → pkg_resources → onnxruntime → protobuf), and
  "deployment path ≠ quality" stated instead of dressed up as success.
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-10 (second record) — on-device RKNN path works, but fp16 fidelity is NOT acceptable (P0)

- Delivery:
  1. Six-stem RKNN converted **on the board** (aarch64, rknn-toolkit2 2.5.0) and benchmarked:
     RTF 0.265, output `(1,6,1025,345)`, finite, mask-sum err 7.2e-4 — the deployment path works.
  2. **P0 finding: RKNN fp16 does not reproduce ONNX numerically on real-audio magnitudes.**
     Same input (`real_mag.npy`, real music STFT window, 0…82 range): torch↔ONNX **max_abs
     1.1e-6 / cosine 1.0** (export correct), but **RKNN↔ONNX max_abs 0.999 / cosine 0.904**.
     Scaled input (÷80): 0.225 / 0.995. Random-unit input: 0.165 / 0.998.
     `--opt-level 0` conversion produced a bit-identical artifact — not an optimizer-level issue.
  3. Consequence: the 4-stem artifact (RTF 0.24) was never parity-checked against ONNX on real
     audio; the same fp16 deviation likely affects it, masked so far by the candidate's ≈0 dB
     SDR. Earlier CPU/NPU listening comparisons cannot rule this out.
  4. Started a 40-track train-split pseudo-label expansion (teacher RTF 0.11–0.15 on an idle
     machine; earlier 0.6–1.3 was load-bound) for the next training round.
- Planned fix (needs retrain): normalize the model input (log1p or fixed-scale magnitude) so
  activations stay inside fp16's comfortable range; re-export → re-convert → re-check parity
  (target cosine >0.999) → listen. Also fix SDR evaluation to ignore near-silent teacher targets
  (many htdemucs_6s outputs are −70…−90 dB on tracks without piano/guitar).
- PUA output visible: yes — diagnostic gate, explicit "deployment path ≠ numerical fidelity"
  call-out, and the reproducible parity matrix (raw / scaled / random / opt0) instead of
  accepting a green conversion as success.
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-11 — RK-targeted retrain: NPU fidelity still FAILS; CPU path is real-time

- Delivery:
  1. Regenerated 13/40 train-split pseudo labels (generation was slowed by host load; stopped to
     free MPS for training), then trained the RK-targeted recipe (`train_rk.py`: log1p input,
     4 layers × 96 ch = **199,418 params**, fp16-rounding simulation during training, mask-TV
     regulariser, 12 epochs): final SDR `{vocals 1.84, drums 0.26, bass 3.82, guitar −8.09,
     piano −0.78, other 0.17}`. The 13-track set is a pipeline probe, not a quality claim.
  2. Exported `models/six/rk-6s.onnx` (log1p in-graph; torch↔ONNX 5.96e-8) and converted it on
     the board: `artifacts/rk/rk-6s.fp.rknn`.
  3. **NPU acceptance FAILED**: 8-window real-audio parity RKNN vs ONNX cosine 0.928
     (max_abs 0.753); per-stem 2 s energy error **mean 3.50 dB / max 14.5 dB** (target <0.5 dB;
     the previous 6-layer model measured 3.88 dB). The fp16-rounding training did **not** make the
     model robust to RKNN's numerical error — all precision modes (fp16, INT8 w8a8 + real
     calibration, w16a16i_dfp, fp32 unsupported) remain unusable.
  4. **However, ONNX Runtime CPU on the Edge2 runs the 199K model in real time: 3.03 s per
     4.005 s window → RTF 0.755**, with fp32 numerics (torch↔ONNX 1e-6, no envelope distortion).
     The CPU path is therefore "correct numbers + real time on RK3588-class SoC".
- Evidence: `work/stem-ab/runs/six_rk/rk_net.pt`, board `artifacts/rk/rk-6s.fp.rknn` +
  convert report, parity/energy measurements above; listening renders under
  `~/Desktop/cpu_path_20261111/` (original / CPU vocals / CPU drums / 6-stem mix).
- Reading: for the cost-down goal the live option is **small model + CPU** (199K params,
  RTF 0.755 on RK3588S; a cheaper SoC needs a smaller MLP or window). The NPU route stays
  blocked pending a different toolchain or model structure — do not re-label fp16 conversion
  success as fidelity success.
- PUA output visible: yes — acceptance gate stated before the run (energy error <0.5 dB), FAIL
  reported with numbers, and the CPU real-time measurement run immediately after instead of
  declaring the NPU route dead.
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-11 (correction) — NPU fidelity was an input-layout BUG, not fp16 precision

- Root cause (found via the RKNN PC simulator): `rknnlite` interprets float inputs as
  **NHWC** (`data_format` defaults to `'nhwc'`). `RknnLiteBackend.run()` passed the NCHW
  `[1,2,1025,345]` tensor straight through, so the NPU saw a silently transposed input.
  Every previous "NPU fidelity failure" (cosine 0.93 / energy error 1.5–3.5 dB, all precision
  modes, the "envelope distortion" listening) was **measuring a corrupted input**, not
  RKNN's numerical precision.
- Fix (deployed to the board): `RknnLiteBackend.run()` transposes NCHW -> NHWC before
  inference (`data_format=["nhwc"]`); `tools/bench_six_rknn.py` likewise.
- Re-measured with the fix on the same artifact and same real-audio windows:
  - worker backend vs ONNX: mask **cosine 0.9998 / max_abs 0.0125** (was 0.928 / 0.75)
  - per-stem 2 s energy error: **mean 0.043 dB / max 0.076 dB** (was 3.50 dB) → **PASS**
  - NPU RTF (199K six-stem RK model): **0.161** (0.645 s per 4.005 s window)
- Consequences:
  1. The NPU route is viable; the 2026-11-10 "RKNN fp16 FAIL" conclusions above are
     **superseded** (they measured transposed input).
  2. Historical NPU evidence (4-stem candidate-v0 soak/stream/CPU-vs-NPU listening) also ran
     with the wrong layout — re-validate before any product claim.
  3. Cost-down: the 199K model has 6× real-time headroom on the 6 TOPS NPU, so a 1–2 TOPS
     tier NPU remains plausible for the next hardware step.
- PUA output visible: yes — instead of re-tuning training (which was the wrong lever), the
  per-depth/per-op probe matrix localized the error to the input boundary, the PC simulator
  produced the decisive error message, and the fix was verified end-to-end with numbers.
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-11 (second record) — six-stem RK model integrated into the NPU worker

- Delivery:
  1. Worker now handles both model widths: `_canonical_masks` accepts 4/6-stem outputs
     (NCHW or NHWC), `masks()` sizes the accumulator from the first window, `_envelope_output`
     executes on the model's own stem count (6 stems direct, 4 stems folded), `EnvelopeFollower`
     is parameterised by stem list, and `--policy s3` rejects six-stem masks with a clear error.
  2. Board: `artifacts/rk-fused.rknn` (199K params, log1p input) deployed; worker restarted with
     `--model artifacts/rk-fused.rknn --window-frames 345 --policy roles`; start scripts
     (`~/restart_board_worker.sh`, `~/start_demo.sh` hash) updated so the watchdog boots into the
     six-stem configuration.
  3. Verified end-to-end on the board:
     - worker log: six-stem decisions `roles={lead:vocals, rhythm:guitar, low:bass, bed:guitar}`,
       applied `{vocals 0, drums −2.4, bass −1.2, guitar −0.9, piano −2.4, other −2.4}` dB
     - `run_stream --chunk-ms 4005`: PASS, `actual_mode=npu`, RTF 0.33
     - 6 s music clip through `pcm_client --chunk 176400`: **zero fallbacks**, packets 13,
       last_rtf 0.456; output vs original corr 0.9996, RMS −0.55 dB, bass band −0.8 dB
       (role gains applied) — no distortion.
- Evidence: `logs/worker-npu-rk6.log`, board `/v1/status`, A/B renders `~/Desktop/rk6_npu_ab/`,
  worker/strategy changes committed.
- Known limits: model quality still weak (13-track probe); 4 s window = 4 s pipeline latency
  (throughput probe, not product latency); `ring_overflow` appears when the client chunk cadence
  does not match the 4 s window.
- PUA output visible: yes — integrated against live board state, chased the fallback reason
  (`worker_protocol` → mask-shape guard; `ring_overflow` → chunk mismatch) instead of calling the
  start a success, and reported end-to-end numbers.
- Rating: **not recorded** (user instruction: 不记录评分).
