#!/usr/bin/env bash
# One-command listening workflow for QQ Music (or any tapped player):
#   1. capture N seconds of what the player is actually playing (Core Audio process tap)
#   2. render passthrough / CPU / NPU renditions of exactly that excerpt
#   3. loudness-match + blind-label them into a playable A/B/C set
#
# Usage:
#   sh tools/listen_qqmusic.sh                    # 45 s, QQ Music, output under docs/listening-material/qq-music
#   sh tools/listen_qqmusic.sh --seconds 60 --bundle com.netease.163music
#
# The capture step waits (up to 180 s) for the player to produce audio: start the script,
# press play in the app, done. The original audio stays audible (no mute) while recording.
#
# Afterwards:
#   - play A/B/C in docs/listening-material/<name>/blind-set/presentation.txt order
#     (macOS: afplay A.wav  — level-matched, mapped blind)
#   - write your scores, only then open key.json
#   - for the recorded protocol: .venv/bin/python tools/listening_ab.py --render <renditions> --out <record-dir>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SECONDS_TO_CAPTURE=45
WAIT_AUDIO=180
NAME="qq-music"
BUNDLE="com.tencent.QQMusicMac"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seconds) SECONDS_TO_CAPTURE="$2"; shift 2 ;;
    --wait-audio) WAIT_AUDIO="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --bundle) BUNDLE="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

OUT_DIR="docs/listening-material/$NAME"
CAPTURE="/tmp/${NAME}-capture.wav"

echo "== 1/3 capture: play a song in the music app now (waiting up to ${WAIT_AUDIO}s) =="
.venv/bin/python tools/capture_qqmusic.py \
  --seconds "$SECONDS_TO_CAPTURE" --wait-audio "$WAIT_AUDIO" \
  --bundle "$BUNDLE" --out "$CAPTURE"

echo
echo "== 2/3 render passthrough / CPU / NPU (excerpt length ${SECONDS_TO_CAPTURE}s) =="
.venv/bin/python tools/render_modes.py \
  --input "$CAPTURE" --start 0 --seconds "$SECONDS_TO_CAPTURE" --out "$OUT_DIR"

echo
echo "== 3/3 build blind A/B/C set (loudness matched, mapping sealed) =="
.venv/bin/python tools/prepare_blind.py \
  --render "$OUT_DIR/renditions" --out "$OUT_DIR/blind-set" \
  --build-hash "git $(git rev-parse --short HEAD)"

echo
echo "== done =="
echo "listen : cd $OUT_DIR/blind-set && afplay A.wav   (order: $(cat "$OUT_DIR/blind-set/presentation.txt" | tail -n +2 | tr '\n' ' '))"
echo "reveal : only after judging -> cat $OUT_DIR/blind-set/key.json"
echo "record : .venv/bin/python tools/listening_ab.py --render $OUT_DIR/renditions --out $OUT_DIR/blind-record"
