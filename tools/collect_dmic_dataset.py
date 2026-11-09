#!/usr/bin/env python3
"""Collect labeled Edge2 DMIC clips on a fixed timeline.

Labels come from the operator's actions, not from a model: the script just
records what the room did during each block.  Output is 48 kHz stereo S16LE
(the firmware analysis rate) plus a labels CSV, so the same clips can drive
rule-threshold calibration and a listening-model retrain later.

Usage on the board:

    .venv/bin/python tools/collect_dmic_dataset.py --out ~/khadas-ows-node/记录/dmic-dataset

The timeline below is printed up front; the operator follows it.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# (label, seconds) — keep blocks short; long blocks encourage sloppy actions.
TIMELINE = [
    ("quiet", 20),
    ("speech", 20),
    ("keyboard", 20),
    ("wind", 20),
]


def record_block(device: str, rate: int, channels: int, seconds: int,
                 target: Path) -> None:
    command = ["arecord", "-D", device, "-f", "S16_LE", "-r", str(rate),
               "-c", str(channels), "-d", str(seconds), str(target)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="hw:2,0")
    parser.add_argument("--rate", type=int, default=48_000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--out", default=str(Path.home() / "记录" / "dmic-dataset"))
    parser.add_argument("--lead-in", type=float, default=10.0,
                        help="seconds before the first block (operator prep)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    print(f"output: {out}", flush=True)
    print("timeline:", json.dumps(TIMELINE, ensure_ascii=False), flush=True)
    print(f"lead-in {args.lead_in:g}s", flush=True)
    for remaining in range(int(args.lead_in), 0, -1):
        print(f"  starting in {remaining}s", flush=True)
        time.sleep(1.0)

    for index, (label, seconds) in enumerate(TIMELINE):
        target = out / f"{stamp}-{index:02d}-{label}.wav"
        print(f"[{index}] {label}: recording {seconds}s -> {target.name}", flush=True)
        started = time.time()
        try:
            record_block(args.device, args.rate, args.channels, seconds, target)
        except subprocess.CalledProcessError as exc:
            print(f"  FAILED: {exc.stderr.decode('utf-8', 'replace')[:200]}", flush=True)
            rows.append({"index": index, "label": label, "seconds": seconds,
                         "file": target.name, "status": "failed"})
            continue
        rows.append({"index": index, "label": label, "seconds": seconds,
                     "file": target.name, "status": "ok",
                     "started_at": time.strftime("%H:%M:%S", time.localtime(started))})
        print(f"  done ({seconds}s)", flush=True)

    csv_path = out / f"{stamp}-labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "label", "seconds",
                                                    "file", "status", "started_at"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"labels: {csv_path}", flush=True)
    ok = sum(1 for row in rows if row.get("status") == "ok")
    print(f"finished: {ok}/{len(rows)} blocks recorded", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
