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

## Real-music sets prepared with this session's pipeline

Real music is what matters for listening; the synthetic fixture only exercised the plumbing.
Generated on 2026-11-06 (all WAVs git-ignored, regenerate any time):

| set | source | how it was made |
| --- | --- | --- |
| `music/angels-in-amplifiers-im-alright/` | MUSDB18-HQ test track (real music, 40 s excerpt) | `render_modes.py` → passthrough / CPU / NPU |
| `music/buitraker-revo-x/` | MUSDB18-HQ test track (40 s excerpt) | same |
| `music/little-chicagos-finest-my-own/` | MUSDB18-HQ test track (40 s excerpt) | same |
| `qq-music/` | **live capture of QQ Music playback** (45 s, Core Audio process tap) | `tools/listen_qqmusic.sh` |

Each set has `renditions/` (named files) and `blind-set/` (loudness-matched, blind labels
`A/B/C` + sealed `key.json` + `presentation.txt` + scores template). Quick listen:

```bash
cd docs/listening-material/qq-music/blind-set
afplay A.wav            # then B.wav, C.wav  (level-matched, mapping hidden)
cat key.json            # only AFTER you have judged
```

What the three renditions are: `passthrough` = the exact excerpt the node forwards,
`cpu`/`npu` = the worker's separation output (the extracted vocal stem, peak-matched per
packet by the engineering-probe convention). So the blind test tells you about leakage,
pumping and artifacts of the separation output against the original mix — not about a
"music enhancement" product.

### Capture your own music app (macOS)

```bash
# capture what QQ Music is playing, render all modes, build the blind set
sh tools/listen_qqmusic.sh                    # 45 s; starts capture and waits for playback
sh tools/listen_qqmusic.sh --seconds 60 --bundle com.netease.163music --name netease
```

Direct pieces: `tools/capture_qqmusic.py` (tap capture, `--launch` opens the app),
`tools/render_modes.py` (three renditions of one excerpt),
`tools/prepare_blind.py` (level matching + blinded A/B/C without an interactive terminal).

## Envelope-modulated listening (the actual purpose of separation)

The consumer never hears a stem. The product chain is: stems -> dominant roles
(`lead/rhythm/low/bed`, hysteresis) -> per-stem gains for the current noise scenario
(contrast-only, attenuation only) -> 400 ms attack / 1200 ms release envelope -> remixed
music. The Edge2 probe now implements this (`python/khadas_ows/strategy.py`,
`worker.py --output-mode envelope`), and these sets were rendered from the live QQ Music
capture (45 s, scenario window **8-30 s**, so 0-8 s is untouched and 30-45 s is the release
tail):

| set | scenario | strength | what it is |
| --- | --- | --- | --- |
| `qq-music-envelope-broadband/` | broadband noise | 0.6 | vocals -2.4 / drums -0.9 / bass -1.2 dB in the event window |
| `qq-music-envelope-keyboard/` | keyboard clicks | 0.6 | keyboard profile (rhythm protected more than low) |
| `qq-music-envelope-broadband-strong/` | broadband noise | 1.0 | stronger contrast, if 0.6 is too subtle to hear |
| `qq-music/stems/` | — | — | individual `vocals/drums/bass/other/mixture` estimates for listening to the separation itself |

Each envelope set has `renditions/` (passthrough = untouched excerpt, cpu = CPU-stem-driven
envelope output, npu = Edge2-NPU-stem-driven envelope output) and `blind-set/` (level-matched
A/B/C + sealed `key.json`). CPU and NPU renditions differ by ~-66..-70 dB relative (same
weights, fp16 rounding), so a blind test cannot separate them -- that is expected and is
recorded, not hidden.

```bash
cd docs/listening-material/qq-music-envelope-broadband/blind-set
afplay A.wav; afplay B.wav; afplay C.wav   # order in presentation.txt; level matched
# compare named renditions / stems directly:
afplay ../renditions/passthrough.wav ../renditions/cpu.wav
afplay ../../qq-music/stems/vocals.wav ../../qq-music/stems/drums.wav
```

Re-render any scenario/strength yourself:

```bash
ARGS="--output-mode envelope --scenario wind --scenario-start-ms 8000 --scenario-end-ms 30000 --strategy-strength 0.6"
.venv/bin/python tools/render_modes.py --input /tmp/qq-music-capture.wav --start 0 --seconds 45 \
  --out docs/listening-material/qq-music-envelope-wind --output-gain unity --worker-args "$ARGS"
# for the npu rendition restart the board worker with the same $ARGS first (see docs/edge2-runbook.md)
.venv/bin/python tools/render_stems.py --input /tmp/qq-music-capture.wav \
  --out docs/listening-material/qq-music/stems
```

What to judge (purpose-aligned): does the envelope-modulated music stay pleasant while the
competing roles drop? Any pumping/zipper at the 8 s attack and 30 s release? Does the
separation leakage (audible in `stems/`) make the remix worse than plain volume ducking? The
engineering status of the role information itself is in `docs/handoff.md` / the evidence JSON.

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
