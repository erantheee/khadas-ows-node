#!/bin/sh
# Establish the tested single-controller dual-A2DP topology in the required order.
# Usage: tools/a2dp_phone_headset_connect.sh PHONE_MAC HEADSET_MAC
set -eu

PHONE_MAC="${1:?phone Bluetooth MAC required}"
HEADSET_MAC="${2:?headset Bluetooth MAC required}"
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
DROPIN_DIR="$HOME/.config/wireplumber/bluetooth.lua.d"

mkdir -p "$DROPIN_DIR"
install -m 0644 "$ROOT/config/wireplumber/51-bluez-dual-a2dp.lua" \
  "$DROPIN_DIR/51-bluez-dual-a2dp.lua"

bluetoothctl disconnect "$PHONE_MAC" >/dev/null 2>&1 || true
bluetoothctl disconnect "$HEADSET_MAC" >/dev/null 2>&1 || true
systemctl --user restart wireplumber
sleep 4

# Remote headset is an Audio Sink (0x110B); remote phone is an Audio Source (0x110A).
busctl call org.bluez "/org/bluez/hci0/dev_$(printf '%s' "$HEADSET_MAC" | tr : _)" \
  org.bluez.Device1 ConnectProfile s 0000110b-0000-1000-8000-00805f9b34fb
sleep 4
busctl call org.bluez "/org/bluez/hci0/dev_$(printf '%s' "$PHONE_MAC" | tr : _)" \
  org.bluez.Device1 ConnectProfile s 0000110a-0000-1000-8000-00805f9b34fb
sleep 4

HEADSET_SINK="$(pactl list sinks short | awk -v mac="$(printf '%s' "$HEADSET_MAC" | tr : _)" \
  '$2 ~ mac { print $2; exit }')"
test -n "$HEADSET_SINK" || { echo "headset A2DP sink was not created" >&2; exit 1; }
pactl set-default-sink "$HEADSET_SINK"

echo "headset_sink=$HEADSET_SINK"
bluetoothctl devices Connected
