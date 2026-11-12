# Phone → Edge2 NPU → Bluetooth headset

This prototype uses the Edge2's single Bluetooth controller in two concurrent
Classic Bluetooth roles:

```text
phone --A2DP--> Edge2 --six-stem NPU envelope--> A2DP headset
```

## Required connection order

1. Pair both devices once.
2. Connect the headset explicitly as remote Audio Sink (`0x110B`).
3. Connect the phone explicitly as remote Audio Source (`0x110A`).
4. Select `Khadas` as the phone's playback device and start non-silent audio.
5. Start `a2dp_stem_relay.py` only after its native PipeWire capture observes
   non-silence. It removes only the direct stereo links and restores them on exit.

```sh
tools/a2dp_phone_headset_connect.sh PHONE_MAC HEADSET_MAC

.venv/bin/python tools/a2dp_stem_relay.py \
  --pipewire-source bluez_input.PHONE_MAC_WITH_UNDERSCORES.2 \
  --sink bluez_output.HEADSET_MAC_WITH_UNDERSCORES.1
```

The numeric suffixes are assigned by BlueZ/PipeWire and must be checked with
`pw-cli ls Node` or `pactl list sinks short` after reconnecting.

## Verified boundary (2026-11-12)

- iPhone input and Anker AeroClip 2 output both negotiated SBC at 44.1 kHz stereo.
- A 35-second live relay observation showed continuous non-silent input,
  `actual_mode=npu`, RTF about 0.69–0.80, zero relay underruns and zero NPU fallbacks.
- The current path has seconds of latency and is a prototype, not a product latency result.
- Generic BlueZ auto-connect can select the headset's wrong role; explicit Profile UUIDs
  are required. HFP/HSP are disabled in the supplied WirePlumber drop-in.
