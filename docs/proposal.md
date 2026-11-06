# Khadas OWS Node — minimal engineering proposal

## Implemented in this repository

- C++ TCP PCM core with bounded buffering, passthrough, CPU/NPU worker modes and automatic passthrough fallback.
- Minimal `GET /v1/status` and `POST /v1/mode` interface.
- Fixed-shape mask network boundary, legacy checkpoint loader, ONNX export, RKNN conversion script and CPU/RKNN-Lite workers.
- Automated core/model contract tests and a human listening record template.

## Not yet validated

- The quarantined lightweight checkpoint has not passed listening quality acceptance.
- RKNN conversion and Edge2 NPU execution require the Rockchip toolkit/runtime and target board.
- Network PCM transport is an engineering probe, not a final phone or Bluetooth product topology.
- No claim is made about target-speaker enhancement, ANC, DOA, beamforming or production power/size.

## Success language

“NPU engineering success” means: fixed-shape ONNX converts to RKNN, runs on Edge2, returns finite correctly shaped masks in real time, and the system falls back safely on failure. It does not mean the separation sounds good. Listening quality is decided by a human and recorded separately.

