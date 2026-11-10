#!/bin/sh
# Select a USB audio device as the board's default output (wired earphone via
# USB DAC / USB-C headset).  The Edge2 has no usable analog codec (the ES8316
# node in the device tree is not populated: probe logs "ES8316 doesn't exist"),
# so a USB DAC is the way to listen to what the board receives over A2DP.
#
# Once this is set, WirePlumber links each new received A2DP stream
# (bluez_input.<mac>) to this default sink automatically, so:
#   phone --A2DP--> board --USB--> wired earphone
# works with no extra routing.  The default sink is saved in WirePlumber state.
#
# Usage (board, as user khadas):  sh tools/a2dp_sink_usb_output.sh
set -eu

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
command -v pactl >/dev/null 2>&1 || { echo "pactl missing: run tools/a2dp_sink_setup.sh first" >&2; exit 1; }

USB=$(pactl list short sinks 2>/dev/null | awk '/\talsa_output\.usb-/{print $2; exit}')
if [ -z "$USB" ]; then
    echo "no USB audio sink found; sinks:" >&2
    pactl list short sinks >&2
    exit 1
fi

pactl set-default-sink "$USB"
pactl set-sink-mute "$USB" 0
pactl set-sink-volume "$USB" 85%
echo "default output -> $USB"
pactl get-default-sink
echo "plug any wired earphone into that USB DAC; phone A2DP audio is routed here automatically"
