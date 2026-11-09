#!/usr/bin/env python3
"""Edge2 full-screen status page rendered straight to /dev/fb0.

The board's DP output is the user's only monitor, and Ubuntu here has no
X/Wayland desktop.  This page is what they see instead of switching between
the Mac panel and a board console: DMIC level, environment, the conversation
policy state, media percentage and the NPU path, all from the local node
`/v1/status` (the same evidence the Mac panel consumes).

Run on the board:

    ~/khadas-ows-node/.venv/bin/python tools/board_display.py

It switches to a free VT (default tty2) when it can, then redraws at 2.5 Hz.
Ctrl+C stops it and the console comes back.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
BG = (11, 18, 32, 255)
PANEL = (17, 27, 45, 255)
TEXT = (226, 232, 240, 255)
LABEL = (148, 163, 184, 255)
GREEN = (34, 197, 94, 255)
YELLOW = (234, 179, 8, 255)
ORANGE = (249, 115, 22, 255)
BLUE = (56, 189, 248, 255)
RED = (239, 68, 68, 255)

STATE_COLOR = {
    "IDLE": GREEN,
    "ANTICIPATING": YELLOW,
    "ENGAGED": ORANGE,
    "RELEASING": BLUE,
}
STATE_LABEL = {
    "IDLE": "IDLE",
    "ANTICIPATING": "ANTICIPATING",
    "ENGAGED": "ENGAGED",
    "RELEASING": "RELEASING",
}
ENV_COLOR = {
    "QUIET": LABEL,
    "NOISY": YELLOW,
    "VOICE": ORANGE,
    "LOUD": RED,
}


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = os.path.join(FONT_DIR, name)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def read_status(url: str, timeout: float = 1.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def read_int(path: str, default: int) -> int:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return default


def screen_size(fb: str) -> tuple[int, int]:
    try:
        with open("/sys/class/graphics/fb0/virtual_size", "r", encoding="utf-8") as handle:
            raw = handle.read().strip().split(",")
        width, height = int(raw[0]), int(raw[1])
    except (OSError, ValueError, IndexError):
        width, height = 1920, 1080
    return width, height


def bar(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fraction: float,
        color: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    fraction = max(0.0, min(1.0, fraction))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=(30, 41, 59, 255))
    if fraction <= 0.0:
        return
    filled = int((x1 - x0) * fraction)
    if filled >= 24:
        draw.rounded_rectangle((x0, y0, x0 + filled, y1), radius=12, fill=color)
    else:
        draw.rectangle((x0, y0, x0 + max(filled, 4), y1), fill=color)


def render(width: int, height: int, status: dict | None, status_url: str) -> Image.Image:
    image = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title = load_font("DejaVuSans-Bold.ttf", 46)
    label = load_font("DejaVuSans.ttf", 30)
    small = load_font("DejaVuSans.ttf", 24)
    small36 = load_font("DejaVuSans.ttf", 30)
    mono = load_font("DejaVuSansMono.ttf", 26)
    huge = load_font("DejaVuSans-Bold.ttf", 170)
    huge_media = load_font("DejaVuSans-Bold.ttf", 140)
    medium = load_font("DejaVuSans-Bold.ttf", 56)

    draw.text((64, 42), "KHADAS EDGE2 · OWS NODE", font=title, fill=TEXT)
    draw.text((width - 460, 52), time.strftime("%Y-%m-%d %H:%M:%S"), font=mono, fill=LABEL)
    draw.line((64, 116, width - 64, 116), fill=(51, 65, 85, 255), width=3)

    if status is None:
        draw.text((64, 200), "NODE OFFLINE", font=medium, fill=RED)
        draw.text((64, 300), status_url, font=mono, fill=LABEL)
        draw.text((64, height - 90), "retrying every 0.4 s · Ctrl+C stops the page", font=small, fill=LABEL)
        return image

    level = float(status.get("dmic_level_dbfs", -120.0))
    voice = float(status.get("dmic_voice_likelihood", 0.0))
    event = str(status.get("dmic_event") or "quiet")
    context = str(status.get("dmic_context") or "unknown")
    context_score = float(status.get("dmic_context_score") or 0.0)
    policy = str(status.get("policy_state") or "IDLE")
    media = float(status.get("media_percent", 100.0))
    ambient = bool(status.get("ambient_intent", False))
    evidence = int(status.get("evidence_frames") or 0)
    packets = int(status.get("packets") or 0)
    fallbacks = int(status.get("fallbacks") or 0)
    actual = str(status.get("actual_mode") or "?")
    rtf = float(status.get("last_rtf") or 0.0)
    if level >= -35.0:
        environment = "LOUD"
    elif voice >= 0.45:
        environment = "VOICE"
    elif level >= -52.0:
        environment = "NOISY"
    else:
        environment = "QUIET"
    state_color = STATE_COLOR.get(policy, LABEL)

    # Column 1 — DMIC (x 64..664)
    x = 64
    draw.text((x, 150), "DMIC LEVEL", font=label, fill=LABEL)
    draw.text((x, 190), f"{level:+.1f}", font=huge, fill=TEXT)
    draw.text((x + 532, 300), "dBFS", font=small36, fill=LABEL)
    bar(draw, (x, 400, x + 540, 450), (level + 70.0) / 70.0, GREEN if level < -35 else RED)
    draw.text((x, 480), f"VOICE LIKELIHOOD {voice:.2f}", font=small, fill=LABEL)
    bar(draw, (x, 525, x + 540, 555), voice, ORANGE if voice >= 0.45 else LABEL)
    draw.text((x, 570), "ENVIRONMENT", font=label, fill=LABEL)
    draw.text((x, 610), environment, font=medium, fill=ENV_COLOR.get(environment, LABEL))
    draw.text((x, 690), f"EVENT  {event.upper()}", font=label, fill=TEXT)
    draw.text((x, 730), f"EVIDENCE FRAMES {evidence}", font=mono, fill=LABEL)

    # Column 2 — conversation policy (x 684..1284)
    x = 684
    draw.text((x, 150), "CONVERSATION POLICY", font=label, fill=LABEL)
    draw.text((x, 190), STATE_LABEL.get(policy, policy), font=medium, fill=state_color)
    draw.text((x, 290), "MEDIA DUCK TARGET", font=label, fill=LABEL)
    draw.text((x, 330), f"{media:.1f}%", font=huge_media, fill=TEXT)
    bar(draw, (x, 540, x + 540, 590), media / 100.0, GREEN if media > 95 else ORANGE)
    draw.text((x, 630), "AMBIENT INTENT", font=label, fill=LABEL)
    draw.text((x, 670), ("ON" if ambient else "off"), font=medium, fill=GREEN if ambient else LABEL)

    # Column 3 — NPU path (x 1304..1904)
    x = 1304
    draw.text((x, 150), "NPU PATH", font=label, fill=LABEL)
    draw.text((x, 190), actual.upper(), font=huge, fill=GREEN if actual == "npu" else LABEL)
    draw.text((x, 420), f"RTF            {rtf:.3f}", font=mono, fill=TEXT)
    draw.text((x, 462), f"PACKETS        {packets}", font=mono, fill=TEXT)
    draw.text((x, 504), f"FALLBACKS      {fallbacks}", font=mono, fill=RED if fallbacks else TEXT)
    draw.text((x, 546), f"EVIDENCE TTL   {status.get('evidence_ttl_ms', '?')} ms", font=mono, fill=LABEL)
    draw.text((x, 588), f"EVIDENCE FRS   {evidence}", font=mono, fill=LABEL)
    draw.text((x, 630), "CONTEXT", font=label, fill=LABEL)
    draw.text((x, 672), f"{context.upper()} {context_score:.2f}" if context != "unknown" else "—",
              font=medium, fill=TEXT)

    # Footer
    draw.line((64, height - 120, width - 64, height - 120), fill=(51, 65, 85, 255), width=2)
    footer = (f"{status_url}  ·  DMIC evidence drives the conversation policy  ·  "
              f"pol={policy}  ·  media={media:.1f}%  ·  Ctrl+C stops the page")
    draw.text((64, height - 96), footer, font=small, fill=LABEL)
    return image


def switch_vt(vt: int) -> None:
    if vt <= 0:
        return
    try:
        subprocess.run(["sudo", "-n", "chvt", str(vt)], check=False, timeout=3,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-url", default="http://127.0.0.1:8787/v1/status")
    parser.add_argument("--fb", default="/dev/fb0")
    parser.add_argument("--vt", type=int, default=2, help="VT to switch to, 0 = stay")
    parser.add_argument("--interval", type=float, default=0.4)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output", help="write a PNG instead of the framebuffer (for checking)")
    args = parser.parse_args()

    width, height = screen_size(args.fb)
    stride = read_int("/sys/class/graphics/fb0/stride", width * 4)
    if stride != width * 4:
        print(f"unsupported framebuffer stride {stride} for width {width}", file=sys.stderr)
        return 1
    switch_vt(args.vt)
    started = time.monotonic()
    frames = 0
    handle = None
    try:
        if args.output:
            image = render(width, height, read_status(args.status_url), args.status_url)
            image.save(args.output)
            print(f"wrote {args.output}")
            return 0
        handle = open(args.fb, "wb", buffering=0)
        while True:
            if args.seconds > 0 and time.monotonic() - started >= args.seconds:
                break
            image = render(width, height, read_status(args.status_url), args.status_url)
            handle.seek(0)
            handle.write(image.tobytes("raw", "BGRA"))
            frames += 1
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if handle is not None:
            handle.close()
    if frames:
        print(f"frames={frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
