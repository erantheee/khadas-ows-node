# Edge2 availability recheck — 2026-11-09

This is a fresh operational recheck, not a replacement for the recorded 30-minute soak.

- Discovery: `khadas.local` resolved to `172.16.20.18`; ICMP and SSH port 22 succeeded.
- Board identity: hostname `Khadas`, Ubuntu 24.04.3 LTS, aarch64, kernel 6.1.118.
- Running services: C++ node plus the RKNN-Lite NPU worker using
  `artifacts/candidate-v0.fp.rknn`.
- Model SHA-256 reported by the node and verified on disk:
  `af92ce83e2a5a9666e9a2c090151c76b39770b891777f2aa3036a3a1fcdb6a6f`.
- Fresh network stream: generated 44.1 kHz stereo fixture, NPU mode, 20 seconds,
  five 4-second chunks. Result: 5/5 successful, zero deadline misses, zero fallbacks,
  zero reconnects; RTT p50 1606.50 ms, p95 1679.91 ms, max 1693.32 ms.
- Post-run status: `requested_mode=npu`, `actual_mode=npu`, 577 cumulative packets,
  zero cumulative fallbacks, `last_rtf=0.290495`.
- Thermal zones were 36.1–37.9 °C after the run.
- Recent server/worker logs contained no runtime failure. RKNN printed its documented
  static-shape dynamic-range query warning; it did not prevent inference.

This verifies that the previously deployed NPU path was reachable and operational on
2026-11-09. It does not add a new listening-quality conclusion.
