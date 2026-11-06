#!/usr/bin/env python3
"""Capture what a running music app is playing (QQ Music by default) into a WAV file.

macOS only: uses the Core Audio process-tap recorder in `tools/mac/qqmusic_tap_record.swift`
(compiled on first use). The capture is read-only for the player: it taps that app's
process mixdown, leaves the original audible (MUTE=2 mutes it while recording if needed)
and writes a 44.1 kHz stereo 16-bit WAV suitable for `tools/render_modes.py`.

Typical use:

    # start QQ Music and press play yourself, then:
    .venv/bin/python tools/capture_qqmusic.py --seconds 45 --out /tmp/qqmusic-excerpt.wav

    # or let the script start the app and wait for you to press play:
    .venv/bin/python tools/capture_qqmusic.py --launch --wait-audio 180 \
        --seconds 45 --out /tmp/qqmusic-excerpt.wav

    # then render the three modes and prepare a blind set:
    .venv/bin/python tools/render_modes.py --input /tmp/qqmusic-excerpt.wav \
        --start 0 --seconds 45 --out docs/listening-material/qq-music
    .venv/bin/python tools/prepare_blind.py \
        --render docs/listening-material/qq-music/renditions \
        --out docs/listening-material/qq-music/blind-set

Other players work too: `--bundle com.netease.163music` for NetEase Cloud Music.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWIFT_SOURCE = ROOT / "tools" / "mac" / "qqmusic_tap_record.swift"
CACHE = ROOT / ".cache" / "mac-tools"
BINARY = CACHE / "qqmusic_tap_record"


def log(message: str) -> None:
    print(f"[capture] {message}", flush=True)


def build_recorder(force: bool = False) -> Path:
    if BINARY.exists() and not force and BINARY.stat().st_mtime >= SWIFT_SOURCE.stat().st_mtime:
        return BINARY
    swiftc = shutil.which("swiftc")
    if swiftc is None:
        raise SystemExit("swiftc not found: install Xcode Command Line Tools (xcode-select --install)")
    CACHE.mkdir(parents=True, exist_ok=True)
    log("compiling Core Audio tap recorder ...")
    result = subprocess.run(
        [swiftc, "-O", str(SWIFT_SOURCE), "-o", str(BINARY),
         "-framework", "CoreAudio", "-framework", "AVFoundation", "-framework", "AppKit"],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"swiftc failed:\n{result.stderr.strip()}")
    return BINARY


def error_message(code: int, bundle: str) -> str:
    if code == 2:
        return (f"target app not running ({bundle}). Open it (or pass --launch), press play, "
                "then run the capture again.")
    if code == 12:
        return ("the app is not producing audio. Press play in the app, then run the capture "
                "again (or increase --wait-audio).")
    return (f"capture failed with exit code {code}; see the Swift recorder output above "
            "(permissions for audio capture may need to be enabled for this terminal/app).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("/tmp/qqmusic-excerpt.wav"),
                        help="output WAV (44.1 kHz stereo 16-bit)")
    parser.add_argument("--seconds", type=float, default=45.0, help="seconds to record after audio is detected")
    parser.add_argument("--wait-audio", type=float, default=180.0,
                        help="how long to wait for the app to actually play audio")
    parser.add_argument("--bundle", default="com.tencent.QQMusicMac",
                        help="bundle id to tap (default QQ Music)")
    parser.add_argument("--launch", action="store_true", help="open the app first (macOS `open -a`)")
    parser.add_argument("--mute", choices=("0", "2"), default="0",
                        help="0 = keep the app audible (default); 2 = mute the app while recording")
    parser.add_argument("--target-pid", type=int, default=None, help="test hook: tap a concrete pid")
    parser.add_argument("--keep-raw", action="store_true", help="keep the raw float32 capture next to the WAV")
    parser.add_argument("--rebuild", action="store_true", help="force recompiling the Swift recorder")
    args = parser.parse_args()

    if platform.system() != "Darwin":
        raise SystemExit("this capture path is macOS-only (Core Audio process tap)")

    binary = build_recorder(force=args.rebuild)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".f32.raw")
    if raw.exists():
        raw.unlink()

    app_name = {"com.tencent.QQMusicMac": "QQMusic", "com.netease.163music": "NeteaseMusic"}.get(args.bundle)
    if args.launch and app_name:
        log(f"opening {app_name} ...")
        subprocess.run(["open", "-a", app_name], check=False)
        time.sleep(1.5)

    environment = {
        "OUT": str(raw),
        "SECONDS": str(args.seconds),
        "WAIT_AUDIO_SECONDS": str(args.wait_audio),
        "MUTE": args.mute,
        "BUNDLE": args.bundle,
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
    }
    if args.target_pid is not None:
        environment["TARGET_PID"] = str(args.target_pid)
    log(f"recording up to {args.seconds:.0f}s after audio starts (waiting up to {args.wait_audio:.0f}s) ...")
    result = subprocess.run([str(binary)], env=environment, capture_output=True, text=True)
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            log(f"recorder: {line}")
    if result.returncode != 0:
        raise SystemExit(error_message(result.returncode, args.bundle))

    sidecar = Path(str(raw) + ".json")
    if not sidecar.exists():
        raise SystemExit(f"recorder produced no sidecar metadata at {sidecar}")
    meta = json.loads(sidecar.read_text())
    sample_rate = int(meta["sample_rate"])
    channels = int(meta["channels"])
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found (brew install ffmpeg) — needed to convert the tap capture")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels), "-i", str(raw),
               "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", str(out)]
    converted = subprocess.run(command, capture_output=True, text=True)
    if converted.returncode != 0:
        raise SystemExit(f"ffmpeg conversion failed: {converted.stderr.strip()}")
    if not args.keep_raw:
        raw.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)

    with wave.open(str(out), "rb") as handle:
        frames = handle.getnframes()
        duration = frames / handle.getframerate()
    if duration < 1.0:
        raise SystemExit(f"captured only {duration:.2f}s; nothing usable was recorded")
    log(f"wrote {out} ({duration:.2f}s, 44100 Hz, stereo, 16-bit)")
    log("next:")
    log(f"  .venv/bin/python tools/render_modes.py --input {out} --start 0 --seconds "
        f"{min(duration, 45):.0f} --out docs/listening-material/qq-music")
    log("  .venv/bin/python tools/prepare_blind.py --render docs/listening-material/qq-music/renditions "
        "--out docs/listening-material/qq-music/blind-set")


if __name__ == "__main__":
    main()
