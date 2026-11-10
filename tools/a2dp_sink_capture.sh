#!/bin/sh
# Capture what the board's A2DP sink is receiving.
#
# PipeWire exposes the received stream as `bluez_input.<source-mac>` and (by
# default WirePlumber policy) links it to the default sink.  That sink's monitor
# is the received PCM, so recording the monitor is the simplest stable tap:
#   bluez_input.<MAC>  ->  default sink (HDMI on the current Edge2 image)  ->  <sink>.monitor
#
# Usage (on the board, as user khadas, after tools/a2dp_sink_setup.sh):
#   sh tools/a2dp_sink_capture.sh --seconds 10 --out /tmp/a2dp.wav
#   sh tools/a2dp_sink_capture.sh --probe            # 3 s, prints peak/RMS
#   sh tools/a2dp_sink_capture.sh --source <name>    # explicit source/monitor name
#
# Notes:
#   * parec needs ~1.5-2 s to connect before the first sample arrives, so the
#     script records `seconds + 3` and trims to the requested length afterwards.
#   * the monitor only carries audio while a source device is streaming to the
#     box; silence while no phone/PC is playing is expected (digital zeros).
set -eu

SECONDS_TO_CAP=10
OUT=/tmp/a2dp-capture-$(date +%Y%m%d-%H%M%S).wav
PROBE=0
SRC=""

while [ $# -gt 0 ]; do
    case "$1" in
        --seconds) SECONDS_TO_CAP="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --source) SRC="$2"; shift 2 ;;
        --probe) PROBE=1; SECONDS_TO_CAP=3; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
command -v pactl >/dev/null 2>&1 || { echo "pactl missing: run tools/a2dp_sink_setup.sh first" >&2; exit 1; }

if [ -z "$SRC" ]; then
    sink="$(pactl get-default-sink 2>/dev/null || true)"
    [ -n "$sink" ] || sink="$(pactl list short sinks 2>/dev/null | awk 'NR==1{print $2}')"
    [ -n "$sink" ] || { echo "no PipeWire sink found" >&2; exit 1; }
    SRC="${sink}.monitor"
fi

has_src=0
for s in $(pactl list short sources 2>/dev/null | awk '{print $2}'); do
    [ "$s" = "$SRC" ] && has_src=1
done
if [ "$has_src" = 0 ]; then
    echo "source '$SRC' not found; available sources:" >&2
    pactl list short sources >&2
    exit 1
fi

RAW="$(dirname "$OUT")/.a2dp-raw-$$.pcm"
echo "recording from $SRC (requested ${SECONDS_TO_CAP}s) ..."
# SIGTERM after seconds+3: parec consumes ~1.5-2 s before its first sample
timeout "$((SECONDS_TO_CAP + 3))" parec --device="$SRC" \
        --format=s16le --rate=48000 --channels=2 --file-format=raw "$RAW" 2>/dev/null || true

python3 - "$RAW" "$OUT" "$SECONDS_TO_CAP" "$PROBE" <<'PY'
import array, math, os, sys, wave

raw_path, out_path, want_s, probe = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] == "1"
rate, ch = 48000, 2
data = array.array("h")
try:
    with open(raw_path, "rb") as f:
        data.frombytes(f.read())
except FileNotFoundError:
    print("capture failed: no PCM written", file=sys.stderr)
    sys.exit(1)
finally:
    try:
        os.unlink(raw_path)
    except OSError:
        pass

n = min(len(data) // ch, int(want_s * rate))
if n <= 0:
    print("capture failed: no samples (is an A2DP source streaming?)", file=sys.stderr)
    sys.exit(1)

seg = data[: n * ch]
with wave.open(out_path, "wb") as w:
    w.setnchannels(ch)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(seg.tobytes())

peak = max(abs(min(seg)), abs(max(seg))) / 32768.0
mean_sq = sum(v * v for v in seg) / len(seg)
rms = math.sqrt(mean_sq) / 32768.0
dbfs = 20 * math.log10(rms) if rms > 0 else -180.0
print(f"wrote {out_path}: {n/rate:.2f}s (requested {want_s:.0f}s) "
      f"peak={peak:.3f} rms={dbfs:.1f} dBFS")
if probe:
    print("digital silence (no A2DP stream)" if dbfs < -120 else "audio present")
PY
