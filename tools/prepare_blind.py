#!/usr/bin/env python3
"""Prepare a blind A/B/C listening set without an interactive terminal.

`listening_ab.py` records the human's scores interactively (by design — a tool must not
invent a listening conclusion). Sometimes you just want to *listen* first: this helper
loudness-matches the renditions, blinds them, writes playable WAVs plus a **sealed** key,
and prints nothing that reveals the mapping.

    .venv/bin/python tools/prepare_blind.py \
        --render docs/listening-material/real-music/renditions \
        --out docs/listening-material/real-music/blind-$(date +%Y%m%d-%H%M%S)

Then listen to A/B/C in `presentation.txt` order, and only afterwards open `key.json`
(or run the interactive tool if you want the recorded protocol).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_listening_module():
    """Reuse the loudness/blinding helpers so both tools agree exactly."""
    spec = importlib.util.spec_from_file_location("listening_ab", ROOT / "tools" / "listening_ab.py")
    if spec is None or spec.loader is None:
        raise SystemExit("cannot import tools/listening_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", type=Path, help="directory of named renditions (passthrough/cpu/npu)")
    parser.add_argument("--a", type=Path)
    parser.add_argument("--b", type=Path)
    parser.add_argument("--c", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="blind-set output directory")
    parser.add_argument("--seed", type=int, default=None, help="deterministic blinding seed (hidden in key.json)")
    parser.add_argument("--build-hash", default="", help="build/model identifier recorded in the key")
    args = parser.parse_args()

    module = load_listening_module()
    explicit = [(name, path) for name, path in (("a", args.a), ("b", args.b), ("c", args.c)) if path]
    if args.render is not None and explicit:
        raise SystemExit("use either --render DIR or --a/--b/--c, not both")
    if explicit:
        if len(explicit) < 2:
            raise SystemExit("at least two renditions are required")
        pairs = [(path.stem, path) for _, path in explicit]
    elif args.render is not None:
        wavs = sorted(Path(args.render).glob("*.wav"))
        if len(wavs) < 2:
            raise SystemExit(f"need at least two .wav files in {args.render}")
        pairs = [(path.stem, path) for path in wavs[:3]]
    else:
        raise SystemExit("pass --render DIR or --a/--b/--c")

    renditions: dict[str, object] = {}
    sample_rate = None
    lengths = {}
    for name, path in pairs:
        signal, rate = module.read_wav(path)
        if sample_rate is None:
            sample_rate = rate
        if rate != sample_rate:
            raise SystemExit(f"{path}: sample rate {rate} differs from {sample_rate}")
        renditions[name] = signal
        lengths[name] = signal.shape[0]
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"renditions differ in length: {lengths}; cut them to the same excerpt")

    measured, gains, matched, target = module.match_loudness(renditions, sample_rate)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 31)
    mapping = module.build_blind_mapping(list(renditions), seed)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    label_files = {}
    for label, condition in mapping["mapping"].items():
        path = out / f"{label}.wav"
        module.write_wav(path, matched[condition], sample_rate)
        label_files[label] = path

    order = mapping["presentation_order"]
    (out / "presentation.txt").write_text(
        "Play in this order (mapping is sealed in key.json):\n" + "".join(f"{label}.wav\n" for label in order),
        encoding="utf-8")
    key = {
        "schema": 1,
        "purpose": "blind human listening set (loudness-matched, randomised labels)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_hash": args.build_hash,
        "sample_rate": sample_rate,
        "duration_s": round(lengths[list(lengths)[0]] / sample_rate, 3),
        "target_lufs": round(target, 2),
        "measured_lufs": {name: round(value, 2) for name, value in measured.items()},
        "gains_db": {name: round(value, 3) for name, value in gains.items()},
        "seed": seed,
        "mapping": mapping["mapping"],
        "presentation_order": mapping["presentation_order"],
        "files": {label: {"path": str(path), "sha256": sha256(path)} for label, path in label_files.items()},
        "source_renditions": {name: {"path": str(path), "sha256": sha256(path)} for name, path in pairs},
        "revealed": False,
        "note": "Do not open this file until after you have listened and written down your judgement. "
                "It contains the A/B/C -> condition mapping.",
    }
    (out / "key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    (out / "scores-template.csv").write_text(
        "label,presentation_index,leakage_1to5,pumping_1to5,artifacts_1to5,preference_1to5,notes\n"
        + "".join(f"{label},{index + 1},,,,,  \n" for index, label in enumerate(order)),
        encoding="utf-8")
    (out / "README.md").write_text(
        "# Blind listening set\n\n"
        "Play the files in the order in `presentation.txt` (macOS: `afplay A.wav`). "
        "Loudness is matched to the quietest rendition with attenuation only, so no file is boosted.\n\n"
        "Write down, for each label: leakage (1-5), pumping (1-5), artifacts (1-5), preference (1-5), notes "
        "(template: `scores-template.csv`). Only then open `key.json` to see which condition each label was.\n\n"
        "A strong tool can compare; only a human can decide. Separate engineering status from listening status.\n",
        encoding="utf-8")

    print(f"[prepare_blind] blind set: {out}")
    print(f"[prepare_blind] renditions: {', '.join(name for name, _ in pairs)}")
    print(f"[prepare_blind] loudness target {target:.2f} LUFS (attenuation-only gains: "
          + ", ".join(f"{name} {gain:+.2f} dB" for name, gain in gains.items()) + ")")
    print(f"[prepare_blind] play order: {' -> '.join(order)}; mapping sealed in key.json")
    print("[prepare_blind] listen first, judge, then reveal key.json")


if __name__ == "__main__":
    main()
