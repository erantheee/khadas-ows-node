#!/bin/sh
# A2DP sink bring-up for the Khadas Edge2 relay-box prototype.
#
# Board facts (verified 2026-11-10): AP6275P combo (Broadcom BCM43752), BT over
# HCI-UART, HCI 5.1, br/edr + le -> Classic Bluetooth / A2DP is possible; Ubuntu
# 24.04.3, kernel 6.1.118, BlueZ 5.72.  See docs/a2dp-sink-20261110.md.
#
# What this does (idempotent, safe to re-run after a reboot):
#   1. installs PipeWire + WirePlumber + the BlueZ media plugin (A2DP endpoints
#      only exist while a media provider registers them),
#   2. turns the adapter into an A/V loudspeaker and keeps it discoverable,
#   3. runs a persistent NoInputNoOutput pairing agent (`bt-agent`) so a phone
#      can pair without a screen/keyboard on the box,
#   4. enables the user audio session to survive SSH/reboots (linger),
#   5. verifies that the adapter advertises "Audio Sink" (A2DP sink endpoint).
#
# Run on the board as user `khadas` (sudo asks for the password):
#   sh tools/a2dp_sink_setup.sh
#
# Undo (back to a plain Bluetooth adapter):
#   sudo systemctl disable --now bt-agent.service
#   sudo sed -i 's/^Class = .*/#Class = 0x000100/; s/^DiscoverableTimeout = .*/#DiscoverableTimeout = 180/;
#                s/^PairableTimeout = .*/#PairableTimeout = 180/' /etc/bluetooth/main.conf
#   sudo systemctl restart bluetooth
set -eu

if [ "$(id -u)" = 0 ]; then
    echo "run as the board user (e.g. khadas), not root: the script uses sudo internally" >&2
    echo "and the PipeWire/WirePlumber session is a systemd --user service." >&2
    exit 1
fi

BT_ADDR="$(bluetoothctl show 2>/dev/null | awk -F'[()]' '/Controller/{print $1}' | awk '{print $2}')"
[ -n "$BT_ADDR" ] || { echo "no Bluetooth controller found" >&2; exit 1; }

echo "== 1/5 packages =="
pkgs="pipewire pipewire-pulse wireplumber libspa-0.2-bluetooth pulseaudio-utils bluez-tools"
missing=""
for p in $pkgs; do
    dpkg -s "$p" >/dev/null 2>&1 || missing="$missing $p"
done
if [ -n "$missing" ]; then
    echo "installing:$missing"
    sudo apt-get update -o Acquire::ForceIPv4=true -qq || true
    # shellcheck disable=SC2086
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $missing
else
    echo "already installed: $pkgs"
fi

echo "== 2/5 adapter profile (A/V loudspeaker, always discoverable) =="
sudo cp -n /etc/bluetooth/main.conf /etc/bluetooth/main.conf.bak-a2dp 2>/dev/null || true
sudo sed -i -e 's/^#\?Class = .*/Class = 0x240414/' \
            -e 's/^#\?DiscoverableTimeout = .*/DiscoverableTimeout = 0/' \
            -e 's/^#\?PairableTimeout = .*/PairableTimeout = 0/' /etc/bluetooth/main.conf
grep -E '^(Class|DiscoverableTimeout|PairableTimeout)' /etc/bluetooth/main.conf

echo "== 3/5 auto-accept pairing agent =="
sudo tee /etc/systemd/system/bt-agent.service >/dev/null <<'EOF'
[Unit]
Description=Bluetooth auto-accept pairing agent (NoInputNoOutput) for the A2DP sink
After=bluetooth.service
Requires=bluetooth.service

[Service]
Type=simple
ExecStart=/usr/bin/bt-agent -c NoInputNoOutput
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
# an ad-hoc `bt-agent -d` from manual bring-up would fight the service for the agent slot
pkill -x bt-agent 2>/dev/null || true
sudo systemctl daemon-reload
sudo systemctl enable --now bt-agent.service
sudo systemctl restart bluetooth
sleep 3

echo "== 4/5 user audio session survives logout/reboot =="
sudo loginctl enable-linger "$(id -un)"
# NOTE: HFP/HSP roles must stay enabled (WirePlumber default).  With A2DP-only
# roles macOS 26 does not publish the box as an audio device ("All profiles not
# connected, delay publishing") -- verified 2026-11-10.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
systemctl --user enable --now pipewire.socket pipewire-pulse.socket wireplumber.service

echo "== 5/5 verify =="
sleep 2
bluetoothctl show | grep -E 'Name:|Class|Powered:|Discoverable:|Pairable:'
bluetoothctl show | grep -q 'Audio Sink' \
    || { echo "FAIL: adapter does not advertise Audio Sink (no media endpoint registered?)" >&2; exit 1; }
systemctl --user is-active pipewire wireplumber pipewire-pulse
systemctl is-active bt-agent.service
echo "OK: A2DP sink ready; adapter $BT_ADDR; pair 'Khadas' from the source device,"
echo "    then capture with: sh tools/a2dp_sink_capture.sh --seconds 10"

# Optional DMIC coexistence guard (only needed if dmic_evidence starts failing
# with EBUSY while PipeWire probes the PDM array).  Card name from the PDM node
# in the Edge2 device tree; add it as a WirePlumber drop-in and restart
# wireplumber if the demo logs `arecord: Device or resource busy`:
#   mkdir -p ~/.config/wireplumber/main.lua.d
#   cat > ~/.config/wireplumber/main.lua.d/51-disable-pdm-mic.lua <<'LUA'
#   table.insert(alsa_monitor.rules, {
#     matches = { { { "device.name", "matches", "alsa_card.platform-fe4b0000.pdm" } } },
#     apply_properties = { ["device.disabled"] = true },
#   })
#   LUA
