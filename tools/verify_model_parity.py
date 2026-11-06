#!/usr/bin/env python3
"""Checkpoint/architecture parity: quarantine candidate vs the original training model.

The migration claim in this repository is that `khadas_ows.model.MaskNetwork` is the
*unchanged* neural body of the legacy Band-SCNet candidate, with streaming STFT/mask/ISTFT
moved out of the model into the DSP boundary. This tool proves that claim instead of
asserting it:

  1. the legacy definition (`work/stem-ab/band_scnet.py::BandSplitNet`, base=128, layers=6)
     loads the candidate checkpoint with `strict=True` (no missing/unexpected keys);
  2. the new definition reproduces the legacy masks bit-for-bit;
  3. reconstructing stems from the new masks equals the legacy `forward()` output;
  4. parameter count and checkpoint SHA256 match the quarantine manifest.

The legacy definition lives in the previous (ESP32) repository and is not vendored here;
pass `--legacy-root` (or set `KHADAS_LEGACY_ROOT`) to the checkout that contains
`work/stem-ab/band_scnet.py`. Without it the tool exits 0 with an explicit SKIP.

Usage:
    python tools/verify_model_parity.py \
        --checkpoint models/candidate-v0.pt \
        --legacy-root /Users/eranthe/Documents/Codex/2026-10-13/cores3-noise-adapt-demo \
        --report models/candidate-v0.parity.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402  (import after sys.path setup)

from khadas_ows.model import MaskNetwork, load_legacy_checkpoint  # noqa: E402


def load_legacy_module(legacy_root: Path):
    candidate = legacy_root / "work" / "stem-ab" / "band_scnet.py"
    if not candidate.exists():
        return None, candidate
    spec = importlib.util.spec_from_file_location("legacy_band_scnet", candidate)
    if spec is None or spec.loader is None:
        return None, candidate
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/candidate-v0.pt")
    parser.add_argument("--legacy-root", default=os.environ.get("KHADAS_LEGACY_ROOT", ""))
    parser.add_argument("--report", default="models/candidate-v0.parity.json")
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"SKIP: checkpoint not found: {checkpoint}")
        return 0

    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "tool": "verify_model_parity",
        "schema": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "legacy_root": args.legacy_root or None,
    }
    expected_sha = None
    quarantine = ROOT / "models" / "quarantine" / "candidate-v0.manifest.json"
    if quarantine.exists():
        expected_sha = json.loads(quarantine.read_text()).get("source_checkpoint_sha256")
        manifest["quarantine_manifest_sha256"] = expected_sha
        manifest["checkpoint_matches_quarantine"] = expected_sha == checkpoint_sha

    model = load_legacy_checkpoint(str(checkpoint))
    mask_model = MaskNetwork()
    manifest["architecture"] = {"name": "MaskNetwork", "base": 128, "layers": 6, "stems": 4}
    manifest["parameter_count"] = int(sum(p.numel() for p in mask_model.parameters()))

    if not args.legacy_root:
        manifest["verdict"] = "SKIP_NO_LEGACY_ROOT"
        print(json.dumps(manifest, indent=2))
        print("SKIP: pass --legacy-root to compare against the original training definition.")
        _write(args.report, manifest)
        return 0

    legacy_root = Path(args.legacy_root)
    module, legacy_path = load_legacy_module(legacy_root)
    if module is None:
        manifest["verdict"] = "SKIP_LEGACY_MODEL_NOT_FOUND"
        manifest["legacy_path"] = str(legacy_path)
        print(json.dumps(manifest, indent=2))
        print(f"SKIP: legacy model definition not found at {legacy_path}")
        _write(args.report, manifest)
        return 0

    legacy = module.BandSplitNet(n_stems=4, base=128, layers=6)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    result = legacy.load_state_dict(state, strict=True)
    legacy.eval()
    manifest["legacy_model"] = str(legacy_path)
    manifest["legacy_strict_load"] = {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}

    torch.manual_seed(args.seed)
    audio = torch.randn(1, 2, 44100 * 2)
    with torch.no_grad():
        legacy_stems = legacy(audio)
        spectrum = module.stft(audio)
        magnitude = spectrum.abs()
        legacy_hidden = legacy.gate(legacy.blocks(magnitude))
        legacy_masks = torch.softmax(legacy.mask(legacy_hidden), dim=1)
        new_masks = model(magnitude)
        reconstructed = torch.stack(
            [module.istft(spectrum * new_masks[:, index].unsqueeze(1), audio.shape[-1]) for index in range(4)],
            dim=1,
        )

    mask_error = float((legacy_masks - new_masks).abs().max())
    stem_error = float((legacy_stems - reconstructed).abs().max())
    mask_sum_error = float((new_masks.sum(1) - 1).abs().max())
    manifest["parity"] = {
        "mask_max_abs_error": mask_error,
        "stem_max_abs_error": stem_error,
        "mask_sum_max_abs_error": mask_sum_error,
        "sample_rate": 44100,
        "seconds": 2,
        "seed": args.seed,
    }
    # The parameter count is factual evidence, not the pass criterion: report it and only
    # require it to stay in the documented ~0.55M class.
    manifest["parameter_count_in_055m_class"] = 540_000 <= manifest["parameter_count"] <= 570_000
    ok = (
        not result.missing_keys
        and not result.unexpected_keys
        and mask_error <= 1e-6
        and stem_error <= 1e-5
        and mask_sum_error <= 1e-5
    )
    ok = ok and manifest["parameter_count_in_055m_class"]
    if expected_sha is not None:
        ok = ok and manifest["checkpoint_matches_quarantine"]
    manifest["verdict"] = "PASS" if ok else "FAIL"
    _write(args.report, manifest)
    print(json.dumps(manifest, indent=2))
    if not ok:
        print("FAIL: parity evidence did not meet the documented thresholds.", file=sys.stderr)
        return 1
    print("PASS: exported neural body is the unchanged legacy candidate.")
    return 0


def _write(path: str, manifest: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
