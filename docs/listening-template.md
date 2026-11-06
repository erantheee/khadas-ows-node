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

## Session record

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
