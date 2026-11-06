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
