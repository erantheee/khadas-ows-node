# khadas-ows-node

Minimal Khadas Edge2 network-audio and RKNN separation probe. Engineering success and listening acceptance are deliberately separate.

## Build and test

```bash
cmake -S . -B build && cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3.11 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q
```

## Run locally

```bash
./build/khadas-ows-node
curl http://127.0.0.1:8787/v1/status
curl -X POST --data passthrough http://127.0.0.1:8787/v1/mode
python tools/pcm_client.py input-44k-stereo.wav output.wav
```

CPU/NPU modes require their independent workers:

```bash
python -m khadas_ows.worker --mode cpu --model models/candidate-v0.pt
python -m khadas_ows.worker --mode npu --model models/candidate-v0.rknn
```

The candidate checkpoint is not committed. Verify its SHA256 against `models/quarantine/candidate-v0.manifest.json` before use.

## Edge2

```bash
sh tools/bootstrap_edge2.sh
PYTHONPATH=python .venv/bin/python tools/convert_rknn.py \
  --onnx models/candidate-v0.onnx --output models/candidate-v0.rknn
```

RKNN conversion requires a supported Linux aarch64/x86_64 RKNN-Toolkit2 environment. NPU execution additionally requires a board image whose RKNPU driver matches RKNN-Toolkit-Lite2. Do not report NPU success until `npu` becomes the `actual_mode` and a board soak test completes.

