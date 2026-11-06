# Human blind listening material (2026-11-06)

This directory supports the listening part of the acceptance. **Nothing here judges quality.**
The conclusion belongs to the human listener and is recorded in `docs/listening-template.md`
(or in the record directory created by the tool).

## What is here

`renditions/` (git-ignored WAVs, regenerable):

| file | what it is |
| --- | --- |
| `passthrough.wav` | the node in `passthrough` mode (bit-exact copy of the input) |
| `cpu.wav` | the same input through the CPU/PyTorch worker on this Mac |
| `npu.wav` | the same input through the Edge2 NPU worker (fp16 `.rknn`, 4 s packets) |

The current material uses the deterministic engineering fixture from
`tools/make_fixture.py` (harmonic chords + drum-like transients + noise floor). It exercises
the pipeline, but it is **not music** — for a real listening judgement, re-render the three
files from a track you care about (see “Rendering your own material” below).

## Running the blind test (interactive terminal required)

```bash
cd <repo>
.venv/bin/python tools/listening_ab.py \
  --render docs/listening-material/renditions \
  --out docs/listening-material/blind-$(date +%Y%m%d-%H%M%S) \
  --build-hash 80ff171
```

The tool loudness-matches the three renditions (ITU-R BS.1770 integrated LUFS, gain-only),
assigns A/B/C blindly with a seeded random permutation, plays them with `afplay` (macOS) or
`aplay` (Linux), collects leakage / pumping / artifacts / preference scores, and only reveals
the key at the end. Use `--no-play` if you want to play the files yourself.

The important question is not “which one is loudest” but: does the processed audio leak,
pump, warble or add artifacts, and would you accept it for music playback? A “fail” is a
legitimate outcome — the candidate model's objective separation quality is ≈ 0 dB SDR.

## Rendering your own material

```bash
# 1. input: any 44.1 kHz stereo 16-bit WAV
# 2. run the node in each mode and record the reply with the PCM client
./build/khadas-ows-node                       # local node (passthrough/cpu)
PYTHONPATH=python .venv/bin/python -m khadas_ows.worker --mode cpu --model models/candidate-v0.pt
curl -X POST --data passthrough http://127.0.0.1:8787/v1/mode
.venv/bin/python tools/pcm_client.py song.wav /tmp/renditions/passthrough.wav

curl -X POST --data cpu http://127.0.0.1:8787/v1/mode
.venv/bin/python tools/pcm_client.py song.wav /tmp/renditions/cpu.wav

# 3. for the NPU rendition run against the board (docs/edge2-runbook.md), then:
#    .venv/bin/python tools/pcm_client.py song.wav /tmp/renditions/npu.wav \
#        --host <board-ip> --chunk 176400
```

Keep the three WAVs identical in length and sample rate; the tool handles level matching.

## Recording the conclusion

Interactive mode writes `scores.csv`, `key.json` (revealed at the end) and `summary.md` into
the `--out` directory. Summarise the human conclusion in `docs/listening-template.md` —
“pass / fail / continue investigation” — and never substitute an automated metric for it.
