-- Phone -> Khadas (A2DP sink) and Khadas -> headset (A2DP source).
-- HFP/HSP are intentionally disabled so WirePlumber cannot select the headset
-- call profile before the high-fidelity A2DP profile.
bluez_monitor.properties = {
  ["bluez5.roles"] = "[ a2dp_sink a2dp_source ]",
  ["bluez5.default.rate"] = 44100,
  ["bluez5.default.channels"] = 2,
}
