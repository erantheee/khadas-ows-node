#!/usr/bin/env python3
"""Append one line per second of relay + node health for chop localization.

Reads the node /v1/status (live) and parses the relay's latest stats line from
logs/relay.log (relay prints every ~5 s). Run from the repo root:

    .venv/bin/python tools/audio_watch.py >> logs/audio-watch.log 2>&1 &

When the user reports a chop, the per-second timeline shows whether underruns,
node fallbacks or packets froze at that moment — i.e. whether the drop is inside
our pipeline or upstream of it.
"""
import json
import re
import time
import urllib.request
from pathlib import Path

STATUS_URL = "http://127.0.0.1:8787/v1/status"
RELAY_LOG = "logs/relay.log"


def read_status() -> dict:
    try:
        with urllib.request.urlopen(STATUS_URL, timeout=2) as reply:
            return json.loads(reply.read())
    except Exception:  # noqa: BLE001 - telemetry only
        return {}


def read_relay_line() -> str:
    try:
        with open(RELAY_LOG, "r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 4096))
            lines = handle.read().splitlines()
            for line in reversed(lines):
                if line.startswith("[relay] chunks="):
                    return line
    except OSError:
        pass
    return ""


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "logs" / "audio-watch.log"
    with open(out, "a", encoding="utf-8") as handle:
        while True:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            status = read_status()
            relay = read_relay_line()
            chunks = re.search(r"chunks=(\d+)", relay)
            underruns = re.search(r"underruns=(\d+)", relay)
            audio_s = re.search(r"audio_s=([\d.]+)", relay)
            node_fb = re.search(r"fallbacks=(\d+)", relay)
            line = (
                f"{ts} "
                f"chunks={chunks.group(1) if chunks else '?'} "
                f"audio_s={audio_s.group(1) if audio_s else '?'} "
                f"underruns={underruns.group(1) if underruns else '?'} "
                f"node_pkts={status.get('packets', '?')} "
                f"node_fb={status.get('fallbacks', node_fb.group(1) if node_fb else '?')} "
                f"rtf={status.get('last_rtf', '?')}"
            )
            handle.write(line + "\n")
            handle.flush()
            time.sleep(1)


if __name__ == "__main__":
    main()
