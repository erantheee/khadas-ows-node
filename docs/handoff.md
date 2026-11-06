# Handoff — 2026-11-06

## Current state

- Private repository: `https://github.com/erantheee/khadas-ows-node` (`main`).
- GitHub Actions run `42181954927` passed on 2026-11-06: CMake/CTest and Python/Pytest green.
- Local C++ build passes 2/2 CTest tests.
- Python model/network tests pass 2/2.
- CPU end-to-end smoke passed on a 2-second 44.1 kHz stereo fixture; observed RTF was about 0.36 on the development Mac. This is not an Edge2 result or listening acceptance.
- Candidate checkpoint SHA256: `0989b29332a3fe710be85f9c168691b11460e8b8a16df35f3c4cafccaeb745de`.
- Fixed-shape ONNX export passed ONNX checker: input `[1,2,1025,345]`, output `[1,4,1025,345]`, opset 18, finite masks, mask sum max error `1.19e-7`.
- RKNN conversion is not run: RKNN-Toolkit2 v2.5.0 has Linux aarch64/x86_64 wheels, not macOS arm64; no reachable Edge2 or Docker runtime was available.

## Next commands on a supported Linux conversion host / Edge2

```bash
sh tools/bootstrap_edge2.sh
PYTHONPATH=python .venv/bin/python tools/export_onnx.py --checkpoint models/candidate-v0.pt --output models/candidate-v0.onnx
PYTHONPATH=python .venv/bin/python tools/convert_rknn.py --onnx models/candidate-v0.onnx --output models/candidate-v0.rknn
PYTHONPATH=python .venv/bin/python -m khadas_ows.worker --mode npu --model models/candidate-v0.rknn
```

Then start `build/khadas-ows-node`, request `npu`, stream the fixed PCM fixture, verify `actual_mode=npu`, and run the 30-minute soak. Record listening separately; do not infer it from metrics.
