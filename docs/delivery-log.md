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
