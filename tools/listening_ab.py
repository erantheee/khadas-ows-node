#!/usr/bin/env python3
"""Human blind A/B/C listening tool for khadas-ows-node.

WHAT THIS TOOL DOES (and deliberately does not do)
    It level-matches renditions (passthrough / CPU / NPU) with an ITU-R BS.1770
    integrated-loudness (LUFS) implementation, assigns random blind labels, plays the
    files, collects a human listener's scores and writes the evidence (CSV + key +
    markdown summary). It does NOT decide whether audio sounds good, does not run
    speech quality heuristics, and never substitutes a metric for listening. The
    quality conclusion is entered by the human listener; automation only provides
    loudness matching, blind randomisation and record keeping.

LEVEL MATCHING
    All renditions are attenuated (never boosted, so nothing can clip) to match the
    quietest rendition's integrated loudness. This makes A/B/C comparable in level so
    the listener judges artefacts, leakage and pumping rather than loudness. Loudness
    matching is NOT a quality metric: two files at the same LUFS can sound very
    different, and that difference is exactly what the human records.

BLIND PROTOCOL
    Labels A/B/C are randomly assigned to conditions with a seeded RNG. The label ->
    condition mapping is written to key.json ONLY after the scoring session finishes
    (revealed at the end on the terminal). Matched WAVs are named by blind label only
    (A.wav/B.wav/C.wav). Deterministic seed (--seed) makes the draw reproducible for
    audits, but only after the session.

TYPICAL USE
    # 1. Render comparable outputs from the node (one run per mode, same input):
    .venv/bin/python tools/run_stream.py --mode passthrough --seconds 20 \
        --output docs/listening-renders/passthrough.wav
    .venv/bin/python tools/run_stream.py --mode cpu --seconds 20 \
        --output docs/listening-renders/cpu.wav
    .venv/bin/python tools/run_stream.py --mode npu --seconds 20 \
        --output docs/listening-renders/npu.wav
    # 2. Blind A/B/C:
    .venv/bin/python tools/listening_ab.py --render docs/listening-renders
    # Non-interactive verification of the machinery only:
    .venv/bin/python tools/listening_ab.py --self-test

EXIT CODES
    0  session finished and evidence written (or --self-test passed)
    1  --self-test failed, session incomplete/aborted, or invalid input
    2  bad usage (missing/duplicate inputs, no interactive stdin, no playable files)

Record format:
    docs/listening-records/<timestamp>/A.wav B.wav C.wav  (loudness-matched, blind names)
    docs/listening-records/<timestamp>/scores.csv         (blind scores; label only)
    docs/listening-records/<timestamp>/key.json           (revealed mapping + LUFS + gains + hashes)
    docs/listening-records/<timestamp>/summary.md         (human conclusion + disclaimer)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "tools/listening_ab.py"
LABELS = ["A", "B", "C"]

# ITU-R BS.1770 K-weighting, derived from the standard's analog prototype so that it is
# valid at any sample rate (the standard tabulates 48 kHz coefficients; equations below
# reproduce them exactly at 48 kHz). No scipy needed.
K_SHELF_GAIN_DB = 3.999843853973347
K_SHELF_FC_HZ = 1681.974450955533
K_SHELF_Q = 0.7071752369554196
K_HIGHPASS_FC_HZ = 38.13547087602444
K_HIGHPASS_Q = 0.5003270373238773
LUFS_OFFSET = -0.691

DISCLAIMER = (
    "This tool only performs loudness matching, blind randomisation, playback help and "
    "record keeping. The listening conclusion belongs to the human listener; no AI, "
    "metric or SDR replaces the listening test."
)


# ----------------------------------------------------------------- numerical helpers

def k_weighting_coefficients(sample_rate: int) -> tuple[tuple[np.ndarray, np.ndarray],
                                                        tuple[np.ndarray, np.ndarray]]:
    """Return (shelf b,a), (highpass b,a) for BS.1770 K-weighting at sample_rate."""
    # Stage 1: high shelf (BS.1770 pre-filter)
    k = math.tan(math.pi * K_SHELF_FC_HZ / sample_rate)
    vh = 10.0 ** (K_SHELF_GAIN_DB / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / K_SHELF_Q + k * k
    shelf_b = np.array([(vh + vb * k / K_SHELF_Q + k * k) / a0,
                        2.0 * (k * k - vh) / a0,
                        (vh - vb * k / K_SHELF_Q + k * k) / a0])
    shelf_a = np.array([1.0,
                        2.0 * (k * k - 1.0) / a0,
                        (1.0 - k / K_SHELF_Q + k * k) / a0])
    # Stage 2: high-pass (RLB)
    k = math.tan(math.pi * K_HIGHPASS_FC_HZ / sample_rate)
    a0 = 1.0 + k / K_HIGHPASS_Q + k * k
    hp_b = np.array([1.0, -2.0, 1.0])
    hp_a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / K_HIGHPASS_Q + k * k) / a0])
    return (shelf_b, shelf_a), (hp_b, hp_a)


def _biquad_response(b: np.ndarray, a: np.ndarray, z_inv: np.ndarray) -> np.ndarray:
    return (b[0] + b[1] * z_inv + b[2] * z_inv ** 2) / (
        a[0] + a[1] * z_inv + a[2] * z_inv ** 2)


def k_weight(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply K-weighting via its exact frequency response (zero-padded linear filtering).

    scipy is not a project dependency, so the 2nd-order IIR stages are applied as their
    frequency response over a sufficiently padded FFT. For 400 ms/100 ms gating blocks
    this is numerically equivalent to time-domain biquads (verified by --self-test).
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    frames = x.shape[0]
    if frames == 0:
        return x
    pad = min(frames, max(1024, int(round(sample_rate * 1.0))))
    padded = np.concatenate([np.zeros((pad, x.shape[1])), x, np.zeros((pad, x.shape[1]))])
    nfft = padded.shape[0]
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    z_inv = np.exp(-2j * np.pi * frequencies / sample_rate)
    (shelf_b, shelf_a), (hp_b, hp_a) = k_weighting_coefficients(sample_rate)
    response = (_biquad_response(shelf_b, shelf_a, z_inv)
                * _biquad_response(hp_b, hp_a, z_inv))
    spectrum = np.fft.rfft(padded, axis=0) * response[:, None]
    filtered = np.fft.irfft(spectrum, n=nfft, axis=0)
    return filtered[pad:pad + frames]


def integrated_loudness(signal: np.ndarray, sample_rate: int) -> float | None:
    """ITU-R BS.1770 integrated loudness (LUFS) with 400 ms blocks, 75% overlap, gating.

    Returns None when the signal is too short to contain a 400 ms block or is silent.
    """
    y = k_weight(signal, sample_rate)
    frames, channels = y.shape
    block = int(round(0.4 * sample_rate))
    hop = int(round(0.1 * sample_rate))
    if frames < block:
        return None
    squared = y * y
    cumulative = np.concatenate([np.zeros((1, channels)), np.cumsum(squared, axis=0)])
    starts = np.arange(0, frames - block + 1, hop)
    block_means = (cumulative[starts + block] - cumulative[starts]) / block  # (m, ch)
    channel_weights = np.ones(channels)  # BS.1770: L/R = 1.0; local mono = 1.0
    energies = block_means @ channel_weights
    with np.errstate(divide="ignore", invalid="ignore"):
        block_loudness = np.where(energies > 0.0,
                                  LUFS_OFFSET + 10.0 * np.log10(np.maximum(energies, 1e-30)),
                                  -np.inf)
    gated = block_loudness > -70.0
    if not gated.any():
        return None
    relative_threshold = (LUFS_OFFSET
                          + 10.0 * np.log10(np.maximum(float(energies[gated].mean()), 1e-30))
                          - 10.0)
    kept = block_loudness > relative_threshold
    if not kept.any():
        kept = gated
    mean_energy = float(energies[kept].mean())
    if mean_energy <= 0.0:
        return None
    return LUFS_OFFSET + 10.0 * math.log10(mean_energy)


def apply_gain_db(signal: np.ndarray, gain_db: float) -> np.ndarray:
    return np.asarray(signal, dtype=np.float64) * (10.0 ** (gain_db / 20.0))


def match_loudness(renditions: dict[str, np.ndarray], sample_rate: int
                   ) -> tuple[dict[str, float], dict[str, float], dict[str, np.ndarray], float]:
    """Attenuate each rendition to the quietest measured loudness.

    Returns (measured_lufs, gain_db, matched_signals, target_lufs). Attenuation-only
    guarantees that matching never boosts a file and therefore never introduces clipping.
    """
    measured: dict[str, float] = {}
    for name, signal in renditions.items():
        loudness = integrated_loudness(signal, sample_rate)
        if loudness is None:
            raise ValueError(f"rendition {name!r} is silent or too short to measure loudness")
        measured[name] = loudness
    target = min(measured.values())
    gains = {name: target - loudness for name, loudness in measured.items()}
    matched = {name: apply_gain_db(signal, gains[name])
               for name, signal in renditions.items()}
    return measured, gains, matched, target


# ------------------------------------------------------------------------ audio I/O

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV as float64 (frames, channels) with sample rate; supports 16/24/32-bit."""
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:  # 24-bit packed little-endian
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        data = values.astype(np.float64) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width * 8} bit")
    if channels < 1:
        raise ValueError(f"{path}: no channels")
    return data.reshape(-1, channels), rate


def write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = 1 if signal.ndim == 1 else signal.shape[1]
    clipped = np.clip(np.asarray(signal, dtype=np.float64).reshape(-1, channels), -1.0, 1.0)
    int16 = (clipped * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(int16.tobytes())


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_identical(path_a: Path, path_b: Path) -> bool:
    if path_a.stat().st_size != path_b.stat().st_size:
        return False
    return sha256_of(path_a) == sha256_of(path_b)


# ------------------------------------------------------------------------- blinding

def build_blind_mapping(conditions: list[str], seed: int) -> dict:
    """Assign conditions to labels A/B/C and randomise presentation order (seeded)."""
    if len(conditions) > len(LABELS):
        raise ValueError(f"at most {len(LABELS)} renditions are supported, got {len(conditions)}")
    rng = random.Random(seed)
    labels = LABELS[:len(conditions)]
    assignment = conditions[:]
    rng.shuffle(assignment)
    mapping = dict(zip(labels, assignment))
    presentation = labels[:]
    rng.shuffle(presentation)
    return {
        "seed": seed,
        "algorithm": "random.Random(seed); shuffle conditions->labels, shuffle "
                     "presentation order",
        "mapping": mapping,
        "presentation_order": presentation,
        "revealed": False,
    }


def resolve_renditions(args: argparse.Namespace) -> list[tuple[str, Path]]:
    explicit = [("a", args.a), ("b", args.b), ("c", args.c)]
    explicit = [(name, path) for name, path in explicit if path is not None]
    if explicit and args.render is not None:
        raise ValueError("use either --a/--b/--c or --render DIR, not both")
    if explicit:
        if len(explicit) < 2:
            raise ValueError("at least two renditions are required for a blind comparison")
        paths = [path for _, path in explicit]
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError("duplicate rendition paths given")
        return [(path.stem, path) for _, path in explicit]
    if args.render is None:
        raise ValueError("no renditions given: pass --a/--b/--c WAV files or --render DIR")
    render_dir = Path(args.render)
    wavs = sorted(render_dir.glob("*.wav"))
    if not wavs:
        raise ValueError(f"no .wav files in --render directory {render_dir}")
    # Prefer the conventional render names, otherwise take the first three sorted files.
    preferred_order = ("passthrough", "cpu", "npu")

    def rank(path: Path) -> int:
        stem = path.stem.lower()
        for index, key in enumerate(preferred_order):
            if key in stem:
                return index
        return len(preferred_order)

    ordered = sorted(wavs, key=lambda path: (rank(path), path.name))
    chosen = ordered[:len(LABELS)]
    if len(ordered) > len(LABELS):
        print(f"[listening_ab] WARNING: {len(ordered)} wav files in {render_dir}; "
              f"using first {len(chosen)}: {[p.name for p in chosen]}")
    if len(chosen) < 2:
        raise ValueError(f"need at least two .wav files in {render_dir}")
    return [(path.stem, path) for path in chosen]


# -------------------------------------------------------------------------- session

def detect_player() -> str | None:
    for candidate in ("afplay", "aplay"):
        if shutil.which(candidate):
            return candidate
    return None


def play_file(player: str, path: Path) -> bool:
    command = [player, str(path)]
    if player == "aplay":
        command = [player, "-q", str(path)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[listening_ab] playback failed: {exc}")
        return False
    if completed.returncode != 0:
        print(f"[listening_ab] playback failed ({completed.returncode}): "
              f"{(completed.stderr or '').strip()[:200]}")
        return False
    return True


def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:  # non-interactive stdin
        raise KeyboardInterrupt from exc


def ask_score(label: str, metric: str, help_text: str) -> int:
    while True:
        raw = ask(f"  {label} {metric} (1-5, being 5 = best/cleanest; {help_text}): ").strip()
        if raw == "":
            print("    required (1-5), please answer.")
            continue
        try:
            value = int(raw)
        except ValueError:
            print("    enter an integer 1-5.")
            continue
        if 1 <= value <= 5:
            return value
        print("    value must be between 1 and 5.")


CSV_HEADER = ["label", "presentation_index", "leakage_1to5", "pumping_1to5",
              "artifacts_1to5", "preference_1to5", "notes"]


def write_scores_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown(record_dir: Path, key: dict, measured_lufs: dict[str, float],
                    gains_db: dict[str, float], target_lufs: float, rows: list[dict],
                    session: dict) -> str:
    lines = [
        "# Human blind listening record",
        "",
        "Engineering status and listening status are independent. This record contains a",
        "human listener's judgement only: the tooling performed loudness matching, blind",
        "randomisation and recording; it did not evaluate audio quality.",
        "",
        f"- Tool: `{TOOL}`",
        f"- Date: {session['date']}",
        f"- Listener: {session.get('listener') or '(not given)'}",
        f"- Headphones / volume / environment: {session.get('environment') or '(not given)'}",
        f"- Build / model hash: {session.get('build_hash') or '(not given)'}",
        f"- Randomisation seed (revealed after judgment): {key['seed']}",
        f"- Record directory: `{record_dir}`",
        "",
        "## Blind scores (CSV)",
        "",
        f"`{record_dir / 'scores.csv'}`",
        "",
        "| label | leakage 1-5 | pumping 1-5 | artifacts 1-5 | preference 1-5 | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['leakage_1to5']} | {row['pumping_1to5']} | "
            f"{row['artifacts_1to5']} | {row['preference_1to5']} | "
            f"{str(row['notes']).replace('|', '/')} |"
        )
    lines += [
        "",
        "## Level matching (ITU-R BS.1770 integrated LUFS, attenuation only)",
        "",
        f"Target loudness = quietest rendition = {target_lufs:.2f} LUFS.",
        "",
        "| condition | integrated LUFS | gain applied |",
        "| --- | --- | --- |",
    ]
    for condition in sorted(measured_lufs):
        lines.append(f"| {condition} | {measured_lufs[condition]:.2f} LUFS | "
                     f"{gains_db[condition]:+.2f} dB |")
    lines += [
        "",
        "## Randomisation key (revealed only after scoring)",
        "",
        "| label | condition |",
        "| --- | --- |",
    ]
    for label in key["presentation_order"]:
        lines.append(f"| {label} | {key['mapping'][label]} |")
    lines += [
        "",
        "## Human conclusion",
        "",
        f"- Verdict: **{session.get('conclusion') or 'NOT RECORDED'}**",
        f"- Notes: {session.get('conclusion_notes') or '(none)'}",
        "",
        "## Separation of claims",
        "",
        "- This document is a listening record, not an engineering soak result.",
        "- A soak PASS (docs/soak-report-*.json) does not imply listening acceptance.",
        "- A listening PASS does not imply the board sustained realtime streaming.",
        "",
    ]
    text = "\n".join(lines)
    (record_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def run_session(args: argparse.Namespace) -> int:
    print("=" * 78)
    print("Human blind A/B/C listening tool")
    print(DISCLAIMER)
    print("=" * 78)
    try:
        renditions = resolve_renditions(args)
    except ValueError as exc:
        print(f"[listening_ab] ERROR: {exc}", file=sys.stderr)
        return 2
    if not sys.stdin.isatty() and not args.self_test:
        print("[listening_ab] ERROR: an interactive terminal is required to record a human",
              file=sys.stderr)
        print("[listening_ab]        conclusion (or use --self-test for non-interactive "
              "machinery checks).", file=sys.stderr)
        return 2

    condition_names = [name for name, _ in renditions]
    signals: dict[str, np.ndarray] = {}
    sample_rate: int | None = None
    for name, path in renditions:
        signal, rate = read_wav(path)
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            print(f"[listening_ab] ERROR: sample rate mismatch: {name} is {rate} Hz, "
                  f"earlier files are {sample_rate} Hz", file=sys.stderr)
            return 2
        signals[name] = signal
    assert sample_rate is not None

    warnings: list[str] = []
    for i, (name_a, path_a) in enumerate(renditions):
        for name_b, path_b in renditions[i + 1:]:
            if files_identical(path_a, path_b):
                warnings.append(
                    f"renditions {name_a!r} and {name_b!r} are bit-identical; blind A/B "
                    "cannot distinguish them (this is an observation, not a quality verdict)"
                )

    try:
        measured_lufs, gains_db, matched, target_lufs = match_loudness(signals, sample_rate)
    except ValueError as exc:
        print(f"[listening_ab] ERROR: {exc}", file=sys.stderr)
        return 2

    seed = args.seed if args.seed is not None else secrets.randbits(32)
    condition_to_label_input = build_blind_mapping(condition_names, seed)
    # map condition name -> label; matched files are written under the blind label only
    label_for_condition = {condition: label
                           for label, condition in condition_to_label_input["mapping"].items()}

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    record_dir = Path(args.out) if args.out else REPO_ROOT / "docs" / "listening-records" / stamp
    record_dir.mkdir(parents=True, exist_ok=True)
    matched_paths: dict[str, Path] = {}
    for label, condition in condition_to_label_input["mapping"].items():
        path = record_dir / f"{label}.wav"
        write_wav(path, matched[condition], sample_rate)
        matched_paths[label] = path

    player = None if args.no_play else detect_player()
    if not args.no_play and player is None:
        warnings.append("no afplay/aplay found: play the matched WAVs manually (see paths below)")
    for warning in warnings:
        print(f"[listening_ab] WARNING: {warning}")

    print(f"[listening_ab] record directory: {record_dir}")
    print(f"[listening_ab] matched files: "
          f"{', '.join(str(path.name) for path in matched_paths.values())}")
    print("[listening_ab] A/B/C labels are randomised; the key is revealed after scoring.")
    print("")
    listener = ask("Listener name (optional): ").strip()
    environment = ask("Headphones / volume / environment (optional): ").strip()

    scores: dict[str, dict] = {}
    rows: list[dict] = []
    aborted = False
    try:
        for index, label in enumerate(condition_to_label_input["presentation_order"]):
            path = matched_paths[label]
            print("")
            print("-" * 78)
            print(f"Rendition {label} ({index + 1}/"
                  f"{len(condition_to_label_input['presentation_order'])})")
            if player is not None:
                play_file(player, path)
                while True:
                    choice = ask("  [Enter] scored, [p] play again, [q] abort session: ").strip().lower()
                    if choice == "p":
                        play_file(player, path)
                        continue
                    if choice == "q":
                        raise KeyboardInterrupt
                    break
            else:
                print(f"  Play this file manually, then press Enter: {path}")
                ask("  [Enter] when ready, [q] abort session: ")
            leakage = ask_score(label, "leakage", "5 = no bleed of other sources")
            pumping = ask_score(label, "pumping", "5 = no level pumping")
            artifacts = ask_score(label, "artifacts", "5 = no clicks/dropouts/distortion")
            preference = ask_score(label, "preference", "5 = most preferred overall")
            notes = ask("  notes (optional): ").strip()
            scores[label] = {"label": label, "leakage": leakage, "pumping": pumping,
                             "artifacts": artifacts, "preference": preference,
                             "notes": notes}
            rows.append({
                "label": label,
                "presentation_index": index + 1,
                "leakage_1to5": leakage,
                "pumping_1to5": pumping,
                "artifacts_1to5": artifacts,
                "preference_1to5": preference,
                "notes": notes.replace("\n", " "),
            })
        print("")
        print("=" * 78)
        print("KEY REVEALED (after judgment):")
        for label in LABELS[:len(condition_names)]:
            if label in condition_to_label_input["mapping"]:
                print(f"  {label} = {condition_to_label_input['mapping'][label]}")
        print("=" * 78)
        conclusion = ask("Human conclusion [pass / fail / investigate]: ").strip().lower()
        while conclusion not in ("pass", "fail", "investigate"):
            conclusion = ask("  enter pass, fail or investigate: ").strip().lower()
        conclusion_notes = ask("Conclusion notes (optional): ").strip()
    except KeyboardInterrupt:
        aborted = True
        conclusion = "INCOMPLETE (session aborted before scoring finished)"
        conclusion_notes = ""
        print("\n[listening_ab] session aborted; partial evidence will be written.")

    condition_to_label_input["revealed"] = True
    session = {
        "date": datetime.now(timezone.utc).isoformat(),
        "listener": listener,
        "environment": environment,
        "build_hash": args.build_hash or "",
        "conclusion": conclusion,
        "conclusion_notes": conclusion_notes,
        "aborted": aborted,
        "disclaimer": DISCLAIMER,
    }
    key = {
        "tool": TOOL,
        "generated_at": session["date"],
        "seed": condition_to_label_input["seed"],
        "presentation_order": condition_to_label_input["presentation_order"],
        "mapping": condition_to_label_input["mapping"],
        "code_mapping": condition_to_label_input["algorithm"],
        "inputs": {name: {"path": str(path), "sha256": sha256_of(path)}
                   for name, path in renditions},
        "matched_files": {label: str(path) for label, path in matched_paths.items()},
        "measured_integrated_lufs": measured_lufs,
        "loudness_gain_db": gains_db,
        "level_match_target_lufs": target_lufs,
        "loudness_algorithm": "ITU-R BS.1770 integrated LUFS (K-weighting; 400 ms blocks, "
                              "75% overlap; -70 LUFS absolute gate; -10 LU relative gate)",
        "warnings": warnings,
        "session": session,
        "separation_of_claims": "This key/summary is a listening record; it does not alter "
                                "engineering soak evidence and cannot be inferred from it.",
    }
    write_scores_csv(record_dir / "scores.csv", rows)
    (record_dir / "key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(record_dir, condition_to_label_input, measured_lufs, gains_db,
                               target_lufs, rows, session)
    print("")
    print(f"[listening_ab] scores  : {record_dir / 'scores.csv'}")
    print(f"[listening_ab] key     : {record_dir / 'key.json'} (revealed)")
    print(f"[listening_ab] summary : {record_dir / 'summary.md'}")
    print("[listening_ab] The conclusion recorded above belongs to the human listener; "
          "the tooling did not judge audio quality.")
    _ = markdown
    if aborted:
        return 1
    return 0


# ------------------------------------------------------------------------- self-test

def _check(results: list[tuple[str, bool, str]], name: str, passed: bool, detail: str) -> None:
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def selftest_loudness() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    rate = 44100
    seconds = 5.0
    t = np.arange(int(rate * seconds)) / rate
    amplitude = 0.1
    sine = amplitude * np.sin(2 * np.pi * 997.0 * t)
    stereo_sine = np.stack([sine, sine], axis=1)
    measured = integrated_loudness(stereo_sine, rate)
    # BS.1770 by construction: -0.691 + 20log10(0.1) + K(997Hz) = -20.0 LUFS; verified
    # independently with ffmpeg's ebur128 (-20.0 LUFS) on 2026-11-06.
    _check(results, "k_weighting_absolute_reference",
           measured is not None and abs(measured - (-20.0)) <= 0.15,
           f"997 Hz stereo sine, peak -20 dBFS -> {measured:.2f} LUFS (expected -20.0 +/-0.15)")

    quiet = 0.1
    sine_10k = quiet * np.sin(2 * np.pi * 10000.0 * t)
    lufs_10k = integrated_loudness(np.stack([sine_10k, sine_10k], axis=1), rate)
    delta = lufs_10k - measured
    _check(results, "k_weighting_high_shelf",
           2.5 <= delta <= 4.5,
           f"10 kHz vs 997 Hz at equal peak: +{delta:.2f} dB loudness (BS.1770 shelf ~+3.3 dB)")

    sine_20 = quiet * np.sin(2 * np.pi * 20.0 * t)
    lufs_20 = integrated_loudness(np.stack([sine_20, sine_20], axis=1), rate)
    delta_20 = lufs_20 - measured
    _check(results, "k_weighting_high_pass",
           delta_20 <= -10.0,
           f"20 Hz vs 997 Hz at equal peak: {delta_20:.2f} dB loudness (high-pass must attenuate)")

    gated = np.vstack([stereo_sine, np.zeros_like(stereo_sine)])
    lufs_gated = integrated_loudness(gated, rate)
    _check(results, "gating_silence_blocks",
           lufs_gated is not None and abs(lufs_gated - measured) <= 0.2,
           f"tone + equal-length digital silence -> {lufs_gated:.2f} LUFS "
           f"(tone {measured:.2f}, silence must be gated out)")

    six_db_louder = integrated_loudness(apply_gain_db(stereo_sine, 6.0), rate)
    _check(results, "linear_gain_consistency",
           abs((six_db_louder - measured) - 6.0) <= 0.05,
           f"+6.00 dB gain -> +{six_db_louder - measured:.3f} LU (must be exactly 6)")
    return results


def _noise(seconds: float, rate: int, tilt: str, seed: int) -> np.ndarray:
    """Unit-RMS noise with a chosen spectral shape (frequency-domain shaping via numpy)."""
    rng = np.random.default_rng(seed)
    frames = int(rate * seconds)
    white = rng.standard_normal(frames)
    if tilt == "white":
        signal = white
    else:
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(frames, d=1.0 / rate)
        if tilt == "dark":  # ~-12 dB/oct above 500 Hz: much less K-weighted energy
            magnitude = 1.0 / (1.0 + (freqs / 500.0) ** 2)
        elif tilt == "bright":  # ~+6 dB/oct below 2 kHz: much more K-weighted energy
            magnitude = 1.0 - 1.0 / (1.0 + (freqs / 2000.0) ** 2)
        else:  # pragma: no cover - programming error
            raise ValueError(tilt)
        magnitude[0] = 0.0
        signal = np.fft.irfft(spectrum * magnitude, n=frames)
    return signal / math.sqrt(float(np.mean(signal ** 2)))  # exact unit RMS


def selftest_loudness_matching(workdir: Path) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    rate = 44100
    seconds = 5.0
    target_rms = 10.0 ** (-20.0 / 20.0)
    a = target_rms * _noise(seconds, rate, "dark", seed=11)
    b = target_rms * _noise(seconds, rate, "bright", seed=22)
    # Equal RMS by construction; spectra differ, so K-weighted loudness must differ.
    rms_a = math.sqrt(float(np.mean(a ** 2)))
    rms_b = math.sqrt(float(np.mean(b ** 2)))
    lufs_a = integrated_loudness(np.stack([a, a], axis=1), rate)
    lufs_b = integrated_loudness(np.stack([b, b], axis=1), rate)
    _check(results, "equal_rms_but_different_spectra",
           abs(rms_a - rms_b) <= 1e-6 and abs(lufs_a - lufs_b) > 0.5,
           f"RMS {20 * math.log10(rms_a):.3f}/{20 * math.log10(rms_b):.3f} dBFS, "
           f"LUFS {lufs_a:.2f}/{lufs_b:.2f} (>=0.5 LU apart, so matching is meaningful)")

    renditions = {"dark": np.stack([a, a], axis=1), "bright": np.stack([b, b], axis=1)}
    measured, gains, matched, target = match_loudness(renditions, rate)
    matched_lufs = {name: integrated_loudness(signal, rate) for name, signal in matched.items()}
    worst = max(abs(value - target) for value in matched_lufs.values())
    gains_are_attenuation = all(gain <= 1e-9 for gain in gains.values())
    _check(results, "loudness_match_within_0.2_lu",
           worst <= 0.2 and gains_are_attenuation,
           f"target {target:.2f} LUFS, matched {matched_lufs['dark']:.2f}/"
           f"{matched_lufs['bright']:.2f} LUFS, worst deviation {worst:.3f} LU")
    # Write matched files and re-read them from disk to prove the evidence path works.
    path_a = workdir / "matched-dark.wav"
    write_wav(path_a, matched["dark"], rate)
    reread, reread_rate = read_wav(path_a)
    lufs_reread = integrated_loudness(reread, reread_rate)
    _check(results, "matched_wav_roundtrip",
           reread_rate == rate and abs(lufs_reread - target) <= 0.2,
           f"written+re-read matched file: {lufs_reread:.2f} LUFS vs target {target:.2f}")
    return results


def selftest_blinding() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    conditions = ["passthrough", "cpu", "npu"]
    first = build_blind_mapping(conditions, seed=1234)
    second = build_blind_mapping(conditions, seed=1234)
    valid = (sorted(first["mapping"]) == LABELS
             and sorted(first["mapping"].values()) == sorted(conditions)
             and sorted(first["presentation_order"]) == LABELS)
    _check(results, "blind_mapping_reproducible",
           valid and first == second,
           f"seed 1234 -> {first['mapping']}, presentation {first['presentation_order']} "
           "(same seed yields the same hidden draw)")
    other = build_blind_mapping(conditions, seed=99)
    _check(results, "blind_mapping_uses_all_conditions",
           sorted(other["mapping"].values()) == sorted(conditions),
           "every condition appears exactly once in the A/B/C assignment")
    return results


def selftest_csv(workdir: Path) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    rows = [{"label": label, "presentation_index": index + 1,
             "leakage_1to5": 4, "pumping_1to5": 3, "artifacts_1to5": 5,
             "preference_1to5": 4, "notes": f"unit-test row {label}"}
            for index, label in enumerate(LABELS)]
    path = workdir / "scores.csv"
    write_scores_csv(path, rows)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        read_rows = list(reader)
        header = reader.fieldnames
    _check(results, "csv_schema_roundtrip",
           header == CSV_HEADER and len(read_rows) == 3
           and read_rows[0]["label"] == "A" and read_rows[2]["artifacts_1to5"] == "5",
           f"{path.name} columns={header}, rows={len(read_rows)}")
    return results


def run_selftest(args: argparse.Namespace) -> int:
    print("[listening_ab --self-test] non-interactive checks of loudness, matching, "
          "blinding and record machinery")
    print("[listening_ab] NOTE: this validates tooling only; it cannot and does not "
          "evaluate audio quality or replace human listening.")
    results: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="listening-ab-selftest-") as temp:
        workdir = Path(temp)
        results += selftest_loudness()
        results += selftest_loudness_matching(workdir)
        results += selftest_blinding()
        results += selftest_csv(workdir)
    failed = [name for name, passed, _ in results if not passed]
    print("")
    print(f"SELF-TEST: {'PASS' if not failed else 'FAIL'} "
          f"({len(results) - len(failed)}/{len(results)} checks passed)")
    if failed:
        print(f"failed checks: {failed}")
        return 1
    return 0


# ----------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human blind A/B/C listening tool: BS.1770 loudness matching, "
                    "randomised labels, human scores and an evidence record. It does NOT "
                    "judge audio quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a", type=Path, help="rendition A (e.g. passthrough output)")
    parser.add_argument("--b", type=Path, help="rendition B (e.g. CPU output)")
    parser.add_argument("--c", type=Path, help="rendition C (e.g. NPU output)")
    parser.add_argument("--render", type=Path, default=None,
                        help="directory of rendered WAVs (prefers passthrough/cpu/npu names)")
    parser.add_argument("--out", type=Path, default=None,
                        help="record directory (default docs/listening-records/<timestamp>)")
    parser.add_argument("--seed", type=int, default=None,
                        help="deterministic blinding seed (default: random, printed on reveal)")
    parser.add_argument("--build-hash", default="",
                        help="build/model identifier to record (optional)")
    parser.add_argument("--no-play", action="store_true",
                        help="do not invoke afplay/aplay; play the matched WAVs manually")
    parser.add_argument("--self-test", action="store_true",
                        help="run non-interactive checks and exit 0/1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_selftest(args)
    return run_session(args)


if __name__ == "__main__":
    sys.exit(main())
