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

## 2026-11-11 (third record) — streaming overlap implemented; seam diagnosis was a false alarm

- Implemented: `--stream-overlap-frames` (default 0 = legacy). Each packet is processed together
  with the previous packet's tail; two window reconstructions overlap and cross-fade with a
  raised-cosine fade; output length == input length. Deployed to the board
  (`--stream-overlap-frames 172`), worker RTF 0.29 -> 0.87 (two model windows per packet).
- **Correction to the previous record**: the "4 s packet seams |diff| 1000 vs mean 274" was a
  false alarm. Per-boundary comparison against the *original* track shows those positions are
  drum transients of the music itself (original |diff| 1128–1164 at 1234800 / 1411200 /
  2469600 samples); the processed output is ~0.9x of the original at every boundary (pure gain
  scaling). There is no seam artifact on this material — the streaming change is infrastructure
  for a future small-window mode, not a fix for an observed defect.
- The real residual: fitted gain g = 0.8905 (−1.01 dB); after removing that gain, residual noise
  is **−28.7 dB relative to signal**, and in the quietest 20% frames the independent noise is
  **−13.1 dB relative to signal** — audible as a "slight noise floor". Its source is the
  time-varying spectral gain application (envelope modulation multiplies per-frame per-bin
  gains, which smears transients and adds texture), not NPU numerics (ORT vs NPU render corr
  1.000000).
- Candidate fixes: (a) silence/low-level gating (bypass below a threshold), (b) slower
  attack/release, (c) smoother gain interpolation across frequency.
- PUA output visible: yes — corrected my own seam diagnosis using the original-signal baseline
  instead of shipping a "fix" for a nonexistent defect; reported the residual decomposition
  (gain vs independent noise) with numbers.
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-11 (fourth record) — envelope parameter sweep: offline gains are small on-board

- Sweep (offline, Dukou 60 s + local ONNX masks, 8 configs varying attack/release, gain smoothing,
  gating): best = slow envelope 800/2000 ms + 5-frame gain smoothing + low-level gate
  (floor −45 / full −25 dB re window peak). Offline quiet-passage residual improved
  −17.8 → −18.2 dB and per-STFT-frame gain ripple halved (2.37 → 1.31).
- Implemented as worker flags: `--attack-ms`, `--release-ms`, `--gain-smooth-frames`
  (plus the existing `--gate-floor-db/--gate-full-db`), wired into the envelope follower and a
  moving-average smoother on the per-frame gains.
- On-board verification (Dukou 60 s, NPU, zero fallbacks):
  - default 400/1200: residual −18.8 dB, quiet-passage −18.5 dB, gain ripple 0.86 dB
  - optimal 800/2000 + smooth 5 + gate: residual −18.9 dB, quiet-passage −18.8 dB, ripple 0.87 dB
  → the offline gain mostly does not survive on-board (0.3 dB quiet-passage improvement, ripple
  unchanged at 1 s granularity). The dominant on-board difference from the original is the
  −1 dB role gain itself, not added noise.
- Reading: the audible "slight noise floor" is the time-varying spectral gain's residual
  (−18.8 dB relative to signal, mostly the intentional −1 dB attenuation plus modulation), not a
  numerical or reconstruction artifact (both measured clean: −138 dB reconstruction, 1.000000
  ORT↔NPU). Parameter tuning has hit diminishing returns; further reduction needs a different
  processing structure (static-gain default with scenario-triggered ramps), not more parameter
  search.
- PUA output visible: yes — swept, deployed, measured on-board, reported that the offline
  improvement largely vanished rather than claiming the tuning win.
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-11 (fifth record) — listening modes approved by the user and landed on the board

- User acceptance (A/B with the Apple Music capture, 2026-11-11): **`01_am_duck_50pct` = approved**
  ("可以了，非常好了"). That is the duck profile: `{drums −1.2, bass −0.6, guitar −0.45,
  piano −1.2, other −1.2}` dB (half of the old ducking strength); the boost candidate is
  `{vocals +0.5, drums +0.8, bass +0.8}` dB. Product form confirmed: **boost by default for music,
  duck when a conversation is detected** (scenario switching).
- Source finding that framed the A/B: the user-provided Dukou WAV is **8 dB louder, has heavier
  low end and a thinner 500 Hz–2 kHz band** than the Apple Music capture (RMS 6037 vs 2426; mid
  share 13.9% vs 20.7%). Both versions have the same (limited) 12 kHz+ content — that is the
  1996 recording, not a file defect. The earlier "big noise floor" in `00_orig` was a **rendering
  bug on our side** (int16 double-scaling → 99.9% clipping), fixed and verified sample-identical
  to the source (max diff 0, SHA256 equal).
- Implementation:
  - `strategy.BOOST_PROFILE` / `strategy.DUCK_PROFILE` (approved maps above).
  - Worker flags `--listening-mode auto|boost|duck`, `--boost-gains/--duck-gains` (JSON override).
    `auto` reads the board's DMIC evidence (`conversation_cue` or `speech_active`, with the
    existing evidence TTL) and picks duck vs boost; envelope smoothing handles the transition.
  - Board startup (deployed): `--listening-mode auto --attack-ms 800 --release-ms 2000
    --gate-floor-db -100 --gate-full-db -90` (gate effectively off = the accepted render).
  - Fixed two board-side integration bugs found during bring-up: a missing line-continuation in
    `restart_board_worker.sh` (worker silently failed to start) and a stale `role_history["scenario"]`
    key in the packet log (KeyError crashed the worker on the first packet).
- Board verification (Dukou 60 s, NPU, chunk 176640):
  - boost mode: log `mode=boost target_db={vocals 0.5, drums 0.8, bass 0.8, ...}`; output vs the
    same-source ONNX render corr **0.99959**
  - duck mode: log `mode=duck target_db={drums −1.2, bass −0.6, guitar −0.45, piano −1.2,
    other −1.2}`; output vs the same-source render corr **0.99986**
  - node `actual_mode=npu`, zero fallbacks during the runs.
- Known: render files were peak-normalised during A/B preparation (board output is not), so A/B
  files differ in absolute level but match in waveform (corr ≥0.9995); A/B listening files should
  be level-matched before the next round.
- PUA output visible: yes — user acceptance recorded verbatim, the source-vs-Apple-Music
  measurement separated file defects from processing defects, and both integration bugs were
  chased down on the live board rather than left as "restart didn't work".
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-11 (sixth record) — content-type profiles: platform-agnostic tuning layer

- User requirement (2026-11-11): "平台调音能力需要不同" — different playback platforms (QQ Music /
  Apple Music / podcast / video apps) need different tuning; both link topologies must be supported
  (iPhone -> relay box over A2DP AND Mac/PC host with process tap).
- Architecture landed:
  1. **Content-type profiles** (`strategy.PODCAST_PROFILE`, `VIDEO_PROFILE`, plus the approved
     `BOOST_PROFILE` for music): music = boost, podcast = voice-forward
     `{vocals +0.8, drums -0.5, bass -0.8, guitar -0.5, piano +0.5}`, video = dialogue-forward
     `{vocals +0.3, drums -1.0, bass -1.0, guitar -0.5}`.
  2. **Relay-side auto detection** (works when the platform is unknown, e.g. A2DP sink):
     band-energy heuristic on the incoming stream — `low(20–200 Hz) < 0.20 and mid(300–3000 Hz) >
     0.40` -> podcast; `low > 0.30` -> music; else video; one-window first decision + two-window
     hysteresis. Verified: Dukou -> music, TTS speech -> podcast, mixed field recording -> video.
  3. **Upstream override**: the evidence file's `content_type` field takes precedence when a
     host (Mac/PC process tap) knows the source; manual modes still available via
     `--listening-mode boost|duck`.
  4. **Priority chain**: explicit mode > conversation (duck) > content profile. Envelope
     smoothing (800/2000 ms) covers all switches.
- Board verification: Dukou 60 s processed with `content_type=music` -> boost targets
  `{vocals 0.5, drums 0.8, bass 0.8}`; a 20 s mixed recording detected as `video` -> video
  targets `{vocals 0.3, drums -1.0, bass -1.0, guitar -0.5}`; `actual_mode=npu`, no fallbacks.
- Still open: (a) real platform matrix test (QQ/Apple Music/podcast/video app via the Mac tap),
  (b) environment-scene layer (quiet/broadband/keyboard/wind/babble) on top of content types,
  (c) classification accuracy on real platform audio (heuristic thresholds are first-cut).
- PUA output visible: yes — built the platform-agnostic layer because the link cannot always know
  the source app, verified detection on three real audio kinds, and stated the accuracy caveat
  instead of calling the heuristic "done".
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-11 (seventh record) — real-platform matrix: content-type detection validated

- Captured 25 s from four real sources through the Core Audio tap (voice prompts guided the user):
  QQ Music, NetEase Cloud Music, Apple Podcasts, and a Chrome video page.
- Detection results (band-energy heuristic, one-window first decision + hysteresis):
  - QQ Music: low 0.909 / mid 0.071 -> **music** (boost profile)
  - NetEase Cloud: low 0.688 / mid 0.165 -> **music**
  - Apple Podcasts: low 0.127 / mid 0.609 -> **podcast** (voice-forward profile)
  - Chrome video: low 0.083 / mid 0.491 -> **podcast** (this clip was voice-dominant/口播; a
    music video or film would land in music/video instead)
- User accepted the result ("OK了"). Agreed framing: the classifier dimension is **content
  character (voice-dominant / music-dominant / mixed)**, not "platform" — platform names are only
  used for explicit overrides from a host that knows the source app.
- Chrome specifics learned: its audio is emitted by the `Google Chrome Helper (...) --type=utility
  --utility-sub-type=audio.mojom.AudioService` process, so the tap must target that pid
  (`--target-pid`); `--bundle com.google.Chrome` taps the browser process which produces no audio.
- Deliverables: A/B renders for each platform under `~/Desktop/platform_ab/`
  (`<platform>_orig.wav` vs `<platform>_<content>.wav`).
- PUA output visible: yes — matrix run end-to-end with real audio, detection results reported with
  the ambiguity (video-as-speech) called out instead of hidden, user accepted.
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-11 (eighth record) — rap classification verified; stem-energy classifier rejected

- Captured a typical rap track (25 s from QQ Music). Band-energy rule: `low(20–200 Hz)=0.89`,
  `mid=0.04` -> **music** (boost profile, drums/bass preserved) — the feared misclassification into
  podcast/video does **not** happen for beat-driven rap, because the kick/bass carries strong low-end.
- Counter-check of the proposed "stem-energy" classifier (using the six-stem masks): **rejected**.
  On podcast/voice content the model (trained on music) splits the voice across bass/guitar:
  podcast mask shares were `vocals 0.45 / bass 0.25 / guitar 0.22`, i.e. a stem-energy rule would
  classify pure speech as music. The band-energy heuristic stays the classifier for now; note this
  also documents a model-quality limit (OOD on speech/podcast material).
- Residual risk documented: a minimal-beat rap with weak low end could still drift toward
  podcast/video; fixing that properly needs a rhythmic-modulation feature (low-band energy
  periodicity), not further threshold nudges.
- PUA output visible: yes — tested the edge case the user asked about with a real rap sample,
  and rejected my own proposed classifier once the data showed it fails on speech.
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-12 — Edge six-stem algorithm committed and pushed to khadas main

- Repo attribution correction: the edge-side six-stem algorithm belongs to `khadas-ows-node`;
  the cores3 checkout's `tools/stem-ab-training/` is the training-side producer (weights), not the
  on-device algorithm.
- New commit `a57dc4e` (10 files): `model.py` RKNN boundary (`input_scale`, optional logits with
  fp32 softmax on the CPU side); `convert_rknn.py` six-stem conversion controls
  (`--stems/--float-dtype/--quant-dtype/--opt-level`); `export_rk_onnx.py` (RKNet log1p export +
  onnxruntime parity); `export_six_onnx.py` (`--no-softmax`); five `models/six/*.manifest.json`
  provenance files; `uv.lock` adds onnxruntime to the test extra.
- Pushed `origin/main` `964ebe7 -> a57dc4e` (21 commits): includes `a978c33` (six-stem RK model in
  the worker: 4/6-stem canonical masks + six-stem envelope execution), `effe055` (NCHW->NHWC),
  plus `1ebb5af`/`29d6ffd` which were already present on `six-stem-onboard` / `edge2-a2dp-sink`.
- Verification: `pytest tests/test_model.py tests/test_strategy.py tests/test_worker_streaming.py`
  -> 26 passed; `git ls-remote origin main` = `a57dc4e`; `git branch -r --contains a978c33` shows
  `origin/main`.
- Residual risks recorded (not fixed): (a) weights are gitignored — `rk-6s.onnx` and the board's
  `artifacts/rk-fused.rknn` are not archived anywhere in git; (b) `export_rk_onnx.py` hardcodes
  `TRAIN_DIR` to the local cores3 training checkout, so khadas is not self-contained (the RKNet
  class lives in the unpushed cores3 `train_rk.py`); (c) main now carries un-accepted 11-09..11-12
  commits, so it is no longer the verified deploy baseline (`964ebe7` was) and no tag marks the
  verified point; (d) `six-stem-onboard` / `edge2-a2dp-sink` are behind main.
- PUA output visible: yes — ⚫百度味 research framing, `[PUA-DIAGNOSIS]` before acting, corrected
  the earlier mis-attribution instead of re-justifying it, and evidence tables (`--contains`,
  `ls-remote`, pytest) instead of prose claims.
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-12 (second record) — board live-state audit + training-asset map (read-only)

- Scope: answer two questions with evidence — (a) is the Edge2 currently running the
  environment/envelope algorithm, (b) where the previously trained artifacts live. No code
  changes, no restart, no model swap.
- Board state (verified by ssh, 2026-11-12 09:57–09:58 UTC):
  - worker PID 2401 runs `--mode npu --model artifacts/rk-fused.rknn --window-frames 345
    --output-mode envelope --policy roles --listening-mode auto --attack-ms 800 --release-ms 2000
    --gate-floor-db -100 --gate-full-db -90 --eq-depth 1.3 --masking-dose 1.0 --max-positive-db 3.0`.
  - envelope chain is in the path and shaping gains: banner `mode=envelope scenario=quiet`;
    packets up to seq=143 carry non-zero `target_db`/`applied_db`; last packet 09:47:22 UTC.
  - idle since then: `/v1/status` `packets=144`, `engaged=false`, `policy_state=IDLE`,
    `ambient_intent=false`; relay PID 2789 socket to 127.0.0.1:9100 in CLOSE-WAIT; phone A2DP
    source absent (only the Anker headset sink connected).
  - environment layer: the roles path never reads `scenario`/`environment`; DMIC `context=horn
    (0.91)` and `dmic_environment=QUIET` only reach `/v1/status` and `tools/board_display.py`.
    Board `worker.py`/`strategy.py` sha256 equal the local checkout.
- Training-asset map (all present, none deleted): `models/six/candidate-6s-final.onnx` (2.1 MB)
  + board `artifacts/rk-fused.rknn` (19 MB) — live; `models/candidate-v0.pt` +
  `models/rknn/candidate-v0.{fp,int8}.rknn` (35 MB) — archived; `models/listening_v0.2.json`
  (52 KB, 9 classes) — live in dmic_evidence; Edge2-retrained v0.3/v0.4/v0.5 in cores3
  `experiments/listening-model/artifacts/` (26 MB; v0.5 val acc 0.8448);
  `experiments/scene-model/artifacts/v0.1-s3-scene.json` (32 KB, telemetry-only S3 head);
  `experiments/event-classifier/event_lr_v1.json` (5 KB, 0.721, contaminated labels);
  cores3 `experiments/sound-events` (1.2 GB) + `ear-context` (405 MB); data caches
  `work/stem-ab` (62 GB) + `work/tse-p0` (2.3 GB).
- Findings raised (not fixed): (1) board `models/model_v0.5.json` is referenced by no code and the
  service starts without `--model`, so the live context model is v0.2-s3 — proven by
  `context=horn`, a class absent from v0.5's 5-class schema; (2) `experiments/event-classifier` is
  untracked (`git ls-files experiments` empty); (3) the running worker carries `--eq-depth`,
  `--masking-dose`, `--max-positive-db`, absent from `~/restart_board_worker.sh`, so a watchdog
  restart changes behaviour.
- Verification: process cmdlines and `/proc/*/environ`; `/v1/status`; worker log banner + packet
  lines; `ss -tnp` (9100 CLOSE-WAIT); `sha256sum` local↔board; `git ls-files experiments`; JSON
  class lists; `du -sh`.
- PUA output visible: yes — Alibaba-flavored narration (conclusion first, closed-loop wording),
  two evidence tables, `[PUA生效 🔥]` on the v0.5-not-wired and classifier-untracked findings;
  zero attempted-but-unaccepted experiments, so no pressure escalation.
- Rating: **not recorded** (user instruction: 不记录评分).

## 2026-11-12 (third record) — all Khadas content committed and pushed

- Instruction: "属于khadas的内容全部上git"; no rating recorded.
- Delivery: every uncommitted Khadas item is now in git on branch
  `feature/phone-bluetooth-headset-relay` (4 commits, pushed to origin):
  - `4acfce5` `feat(a2dp)`: dual-role relay prototype — WirePlumber drop-in
    (a2dp_sink + a2dp_source only), explicit-Profile connect script, native PipeWire
    capture relay, `live_play.py` local-player path, runbook with the 2026-11-12 boundary.
  - `cec04fd` `fix(dmic)`: Edge2-calibrated voice likelihood (periodicity + vocal-band
    flatness + level gate) and event rules; `conversation_cue` pinned to 0.0 because
    dialogue ducking is a Mac-side decision after the 2026-11-10 field report.
  - `0ee2510` `feat(experiments)`: learned DMIC event classifier v1 — pipeline proof only
    (0.721 temporal-split accuracy; wind 21% / keyboard 38% recall on contaminated labels;
    not wired to the live path). Repo root now derived from `__file__`, no absolute path.
  - `e9b3817` `docs`: 11-12 board live-state audit + training-asset map.
- Verification: `git status` clean; `pytest -q tests/test_model.py tests/test_strategy.py
  tests/test_tuning.py tests/test_worker_streaming.py` -> 37 passed; `git push -u origin
  feature/phone-bluetooth-headset-relay` -> `* [new branch]`, upstream set;
  `sh -n` + `py_compile` on every new script before staging.
- Deliberately not added (repo `.gitignore` = generated engineering artifacts):
  `models/candidate-v0.pt` (2.1 MB), `models/rknn/*.rknn` (33 MB), `models/six/*.onnx(.data)`
  (9.4 MB), `models/calib/*.npy` (22 MB). Archiving weights needs an explicit decision
  (~70 MB of history); the 11-12 audit entry already records that gap.
- Carried risk: `main` is untouched (still `3e934da`); this work exists only on the feature
  branch — merging to main is a separate acceptance decision.
- PUA output visible: yes — Alibaba-flavored narration, `[PUA-DIAGNOSIS]` before staging,
  secret/absolute-path scan before the first commit, commits grouped by workstream,
  test + push evidence instead of prose claims.
- Rating: **not recorded** (user instruction: 不记录评分).
