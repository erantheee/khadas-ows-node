#!/usr/bin/env python3
"""Benchmark the six-stem RKNN artifact on the board (RKNN-Toolkit-Lite2).

Measures load/init, the [1,6,1025,345] output boundary, finite masks, mask-sum
consistency (the network ends in a softmax over stems), and per-window latency
(RTF against the 4.005 s window).

Usage (board):
    .venv/bin/python tools/bench_six_rknn.py --rknn artifacts/six/candidate-6s.fp.rknn \
        --report artifacts/six/bench.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from rknnlite.api import RKNNLite

    t0 = time.monotonic()
    rknn = RKNNLite()
    if rknn.load_rknn(args.rknn) != 0:
        raise SystemExit("load_rknn failed")
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        raise SystemExit("init_runtime failed")
    load_s = time.monotonic() - t0

    rng = np.random.default_rng(args.seed)
    x = (rng.standard_normal((1, 2, 1025, 345)).astype(np.float32) * 0.5)
    for _ in range(2):
        rknn.inference(inputs=[x])
    times = []
    for _ in range(args.iters):
        t = time.monotonic()
        out = rknn.inference(inputs=[x])
        times.append(time.monotonic() - t)
    masks = np.asarray(out[0])
    window = 345 * 512 / 44100
    mean_s = float(np.mean(times))
    report = {
        "rknn": args.rknn,
        "load_init_s": round(load_s, 2),
        "output_shape": list(masks.shape),
        "finite": bool(np.isfinite(masks).all()),
        "mask_sum_max_err": float(np.max(np.abs(masks.sum(axis=1) - 1.0))),
        "per_window_s": round(mean_s, 3),
        "per_window_min_s": round(float(np.min(times)), 3),
        "per_window_max_s": round(float(np.max(times)), 3),
        "rtf": round(mean_s / window, 3),
        "iters": args.iters,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        open(args.report, "w").write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
