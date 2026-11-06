#!/usr/bin/env bash
set -euo pipefail
if [[ $(uname -m) != aarch64 ]]; then echo "expected Edge2 aarch64" >&2; exit 2; fi
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential cmake python3 python3-venv python3-pip curl ca-certificates
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
echo "Base runtime ready. Install the RKNN-Toolkit-Lite2 wheel matching the board image/RKNPU driver before NPU mode."

