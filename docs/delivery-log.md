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
