# Human listening record

Engineering status and listening status are independent: a soak PASS (`docs/soak-report-*.json`)
does not imply listening acceptance, and a listening PASS does not imply the board sustained
realtime streaming. The conclusion below belongs to the human listener; tooling only provides
loudness matching, blind randomisation, playback help and record keeping.

Use the tool (see `docs/edge2-runbook.md` §10):

```bash
.venv/bin/python tools/listening_ab.py --render docs/listening-renders \
    --out docs/listening-records --build-hash "<git sha + rknn sha256>"
.venv/bin/python tools/listening_ab.py --self-test   # machinery check only, not a quality test
```

## Session record — 2026-11-06 (first human listening, QQ Music capture)

- Listener observation (verbatim, user): "NPU、CPU其实都还好，分轨可以，但好像鼓分出来的怪怪的"
- Source material: 45 s live QQ Music capture (Core Audio process tap); named stems in
  `docs/listening-material/qq-music/stems/`, envelope-modulated sets in
  `docs/listening-material/qq-music-envelope-{broadband,keyboard,broadband-strong}/`
- Build: main `d27f1bd` (NPU fp16 rknn sha256 `af92ce83…6a6f`)
- Observed: CPU and NPU renditions are indistinguishable (measured -66..-70 dB relative
  difference, as expected); separation is acceptable to the listener overall; the **drums
  stem sounds abnormal** (measured: -45 dBFS vs -11.7 dBFS mixture, 5.6 kHz centroid,
  62 % of energy in the top 2 % samples -> mostly faint high-frequency leakage).
- Root-cause evidence: `docs/evidence/stem-drums-analysis-20261106.json` (drums SDR ~0 dB,
  drums estimate 23-34 dB below truth, stem order verified correct, checkpoint trained only
  6 of 20 planned epochs).
- Human conclusion (user, 2026-11-06): "以后再说吧，主要是能分开，你也能调包络" — the
  accepted outcome for this version is: **separation works well enough to be listened to and
  the envelope-modulation chain is implemented/adjustable**; the drums-stem quality issue is
  recorded and its fix (continued training) is **deferred by the user**. No score is recorded
  and no continued-training run was started.

## Session record template

- Date / listener:
- Build SHA / model hash (rknn sha256):
- Headphones / volume / environment:
- Renders source (run commands / JSON evidence):
- Randomisation seed + order key (revealed only after judgment):
- Record location: `docs/listening-records/<timestamp>/` (`A.wav/B.wav/C.wav`, `scores.csv`,
  `key.json`, `summary.md`)
- Level matching: ITU-R BS.1770 integrated LUFS, attenuation only to the quietest rendition
  (levels listed in `key.json`; matching is not a quality metric)

## Blind scores (fill from the tool's `scores.csv`, blind labels first)

- A: leakage / pumping / artifacts / preference:
- B: leakage / pumping / artifacts / preference:
- C: leakage / pumping / artifacts / preference:

## Key (revealed after judgment)

- A = :
- B = :
- C = :

## Human conclusion

- Verdict: pass / fail / continue investigation
- Notes (what was audible, which condition was preferred and why, caveats):
- Listener signature / date:

> The tool never fills in or infers this conclusion. Do not replace it with an AI judgement,
> SDR metric or loudness number; BS.1770 matching only equalises level.
