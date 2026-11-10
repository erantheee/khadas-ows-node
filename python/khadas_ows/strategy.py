"""Role-aware envelope-modulation strategy (port of the legacy stem-remix policy).

The consumer of separation is not the stem itself. The product chain is:

    stems -> dominant musical roles -> per-stem gains for the current noise scenario
          -> time-domain envelope shaping (attack / hold / release) -> remixed music

This module ports the parts of the legacy `tools/stem_remix/stem_remix.py` that the
listening/envelope output needs. Two schemas coexist:

* **Strategy schema (6 stems)** — `STEMMS6` / `ROLE_MAP6` / `PROFILES` mirror the Mac
  `stem_remix.py` six-stem role policy (vocals/drums/bass/guitar/piano/other). This is
  what the decision layer runs on.
* **Execution schema (4 stems)** — `STEMMS` is what the current candidate model emits.
  The model has no guitar/piano masks, so those stems enter the decision as silence
  (`SILENT_DB`) and their gains fold back into `other` at execution time
  (`fold_to_execution`, max semantics). With a 6-stem model the same strategy code runs
  unchanged and the fold becomes an identity on the wider schema.

Scenario targets are the legacy `role_profile()` values for the S3-representable
environments. `contrast_only_profile` keeps the top role at 0 dB and only attenuates
competitors, because boosting estimated stems amplifies separation leakage and eats
headroom.
"""
from __future__ import annotations

STEMMS = ("vocals", "drums", "bass", "other")
ROLE_MAP: dict[str, tuple[str, ...]] = {
    "lead": ("vocals", "other"),
    "rhythm": ("drums",),
    "low": ("bass", "drums"),
    "bed": ("other",),
}
SCENARIOS = ("quiet", "broadband", "keyboard", "wind", "babble")
# Legacy role_profile() targets (dB) for each S3-representable environment.
SCENARIO_TARGETS: dict[str, dict[str, float]] = {
    "quiet": {},
    "broadband": {"lead": 3.0, "rhythm": 1.5, "low": 1.0},
    "keyboard": {"lead": 3.0, "rhythm": 2.0, "low": 0.5},
    "wind": {"lead": 2.5, "rhythm": 1.5, "low": -1.0},
    "babble": {"lead": 3.5, "rhythm": 1.0, "low": 0.0},
}
BASE_GAIN_DB = -1.0
#: Level used for stems the current model cannot observe (guitar/piano). Must lose
#: every dominant-role comparison so the 4-stem execution schema degrades cleanly.
SILENT_DB = -240.0

# --- Six-stem strategy schema (macOS stem_remix.py parity) -------------------
#: Decision-layer stems. The candidate model emits four; see module docstring.
STEMMS6 = ("vocals", "drums", "bass", "guitar", "piano", "other")
ROLE_MAP6: dict[str, tuple[str, ...]] = {
    "lead": ("vocals", "guitar", "piano", "other"),
    "rhythm": ("drums", "guitar", "piano"),
    "low": ("bass", "drums"),
    "bed": ("other", "guitar", "piano"),
}
#: Legacy explicit per-role gain tables (`PROFILES` in Mac stem_remix.py), in dB.
PROFILES: dict[str, dict[str, float]] = {
    "neutral_rebuild": {stem: 0.0 for stem in STEMMS6},
    "lead_protect": {
        "vocals": 3.0, "drums": -1.0, "bass": -1.0,
        "guitar": 1.5, "piano": 1.5, "other": 0.5,
    },
    "rhythm_protect": {
        "vocals": 0.5, "drums": 2.5, "bass": 1.5,
        "guitar": 0.5, "piano": -0.5, "other": -1.0,
    },
    "low_protect": {
        "vocals": 0.5, "drums": 1.0, "bass": 3.0,
        "guitar": -0.5, "piano": -0.5, "other": -1.0,
    },
}


def dominant_roles(stats_db: dict[str, float]) -> dict[str, str]:
    """Pick the strongest candidate for every role from per-stem RMS (dB)."""
    return {role: max(candidates, key=lambda stem: stats_db[stem])
            for role, candidates in ROLE_MAP.items()}


class RoleTracker:
    """Hysteresis over per-window dominant roles so small energy changes do not pump.

    Legacy `smooth_role_windows`: a challenger must lead the incumbent by `margin_db`
    for `hold_windows` consecutive windows before the role flips.

    `role_map` defaults to the four-stem execution map; pass `ROLE_MAP6` for the
    six-stem strategy schema. Stems absent from `stats_db` (guitar/piano on the current
    candidate model) score `SILENT_DB`, so they can never win a role.
    """

    def __init__(self, margin_db: float = 2.0, hold_windows: int = 2,
                 role_map: dict[str, tuple[str, ...]] | None = None) -> None:
        self.margin_db = float(margin_db)
        self.hold_windows = int(hold_windows)
        self.role_map = dict(role_map or ROLE_MAP)
        self.roles: dict[str, str] | None = None
        self._held = {role: 0 for role in self.role_map}

    def _strongest(self, stats_db: dict[str, float]) -> dict[str, str]:
        return {role: max(candidates, key=lambda stem: float(stats_db.get(stem, SILENT_DB)))
                for role, candidates in self.role_map.items()}

    def update(self, stats_db: dict[str, float]) -> dict[str, str]:
        if self.roles is None:
            self.roles = self._strongest(stats_db)
            return dict(self.roles)
        strongest = self._strongest(stats_db)
        for role in self.role_map:
            challenger, owner = strongest[role], self.roles[role]
            if challenger == owner:
                self._held[role] += 1
                continue
            lead_db = (float(stats_db.get(challenger, SILENT_DB))
                       - float(stats_db.get(owner, SILENT_DB)))
            if lead_db >= self.margin_db and self._held[role] >= self.hold_windows:
                self.roles[role] = challenger
                self._held[role] = 0
            else:
                self._held[role] += 1
        return dict(self.roles)


def role_profile(roles: dict[str, str], scenario: str) -> dict[str, float]:
    """Map time-local roles + scenario to per-stem gains (legacy role_profile)."""
    if scenario not in SCENARIO_TARGETS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    if scenario == "quiet":
        return {stem: 0.0 for stem in STEMMS}
    gains = {stem: BASE_GAIN_DB for stem in STEMMS}
    for role, target in SCENARIO_TARGETS[scenario].items():
        stem = roles[role]
        gains[stem] = max(gains[stem], target)
    return gains


def contrast_only_profile(gains_db: dict[str, float], strength: float = 0.6,
                          risk: float = 0.0) -> dict[str, float]:
    """Keep the top role at 0 dB, only attenuate competitors (legacy behaviour)."""
    strength = min(max(float(strength), 0.0), 1.0)
    risk = min(max(float(risk), 0.0), 1.0)
    top = max(gains_db.values(), default=0.0)
    scale = strength * (1.0 - risk)
    return {stem: min(0.0, (gains_db[stem] - top) * scale) for stem in STEMMS}


class EnvelopeFollower:
    """Per-stem gain envelope: attack when going down (more attenuation), release up.

    Legacy live build used 400 ms attack / 1200 ms release for masking depth. One state
    per stem, updated per STFT frame; holds the scenario gain while the scenario is active
    and releases to 0 dB when it ends.
    """

    def __init__(self, attack_ms: float = 400.0, release_ms: float = 1200.0) -> None:
        self.attack_tau = max(float(attack_ms), 1.0) / 1000.0
        self.release_tau = max(float(release_ms), 1.0) / 1000.0
        self.gains = {stem: 0.0 for stem in STEMMS}

    def step(self, targets: dict[str, float], dt_s: float) -> dict[str, float]:
        for stem in STEMMS:
            target = float(targets.get(stem, 0.0))
            current = self.gains[stem]
            tau = self.attack_tau if abs(target) > abs(current) else self.release_tau
            alpha = 1.0 - pow(2.718281828459045, -max(dt_s, 0.0) / tau)
            self.gains[stem] = current + (target - current) * alpha
        return dict(self.gains)


def scenario_active(t_s: float, start_s: float, end_s: float) -> bool:
    return start_s <= t_s < end_s


# --------------------------------------------------------------------------- #
# Six-stem strategy helpers (Mac stem_remix.py parity; decision layer only)
# --------------------------------------------------------------------------- #
def dominant_roles6(stats_db: dict[str, float]) -> dict[str, str]:
    """As `dominant_roles`, on the six-stem role map."""
    return {role: max(candidates, key=lambda stem: float(stats_db.get(stem, SILENT_DB)))
            for role, candidates in ROLE_MAP6.items()}


def role_profile6(roles: dict[str, str], scenario: str) -> dict[str, float]:
    """Six-stem `role_profile` (legacy Mac stanza + role map)."""
    if scenario not in SCENARIO_TARGETS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    if scenario == "quiet":
        return {stem: 0.0 for stem in STEMMS6}
    gains = {stem: BASE_GAIN_DB for stem in STEMMS6}
    for role, target in SCENARIO_TARGETS[scenario].items():
        stem = roles[role]
        gains[stem] = max(gains[stem], target)
    return gains


def automatic_profile6(stats_db: dict[str, float]) -> dict[str, float]:
    """Legacy `automatic_profile`: protect strongest lead/rhythm/low, duck rest by 1 dB."""
    roles = dominant_roles6(stats_db)
    gains = {stem: BASE_GAIN_DB for stem in STEMMS6}
    gains[roles["lead"]] = 3.0
    gains[roles["rhythm"]] = max(gains[roles["rhythm"]], 1.5)
    gains[roles["low"]] = max(gains[roles["low"]], 1.0)
    return gains


def contrast_only_profile6(gains_db: dict[str, float], strength: float = 0.6,
                           risk: float = 0.0) -> dict[str, float]:
    """Keep the top stem at 0 dB, only attenuate competitors (six-stem schema)."""
    strength = min(max(float(strength), 0.0), 1.0)
    risk = min(max(float(risk), 0.0), 1.0)
    top = max((float(gains_db.get(stem, 0.0)) for stem in STEMMS6), default=0.0)
    scale = strength * (1.0 - risk)
    return {stem: min(0.0, (float(gains_db.get(stem, 0.0)) - top) * scale)
            for stem in STEMMS6}


def remix_risk(crest_db: float, active_stems: int, vocal_side_ratio: float) -> float:
    """Legacy `remix_risk`: transients are only dangerous when dense or wet."""
    def clamp(value: float) -> float:
        return min(max(value, 0.0), 1.0)
    transient = clamp((crest_db - 11.0) / 9.0)
    density = clamp((active_stems - 2.0) / 4.0)
    wet_width = clamp((vocal_side_ratio - 0.12) / 0.38)
    interaction = transient * max(density, wet_width)
    return clamp(0.55 * interaction + 0.30 * density + 0.15 * wet_width)


def active_stem_count(stats_db: dict[str, float], floor_db: float = -60.0) -> int:
    """Count stems above an absolute level (legacy `active_stems` for `remix_risk`)."""
    return sum(1 for level in stats_db.values() if float(level) > floor_db)


def fold_to_execution(gains6: dict[str, float]) -> dict[str, float]:
    """Fold six-stem strategy gains onto the four-stem execution schema.

    The current candidate model has no guitar/piano masks: their content sits inside
    `other`. Execution must not attenuate `other` below what guitar/piano protection
    asks for, hence max() (all six-stem targets are attenuation-only at this point).
    With a six-stem model this fold is the identity on the wider schema.
    """
    folded = {stem: float(gains6.get(stem, 0.0)) for stem in STEMMS}
    # Missing guitar/piano must not change `other` (identity fold for 4-stem inputs).
    folded["other"] = max(folded["other"], float(gains6.get("guitar", SILENT_DB)),
                          float(gains6.get("piano", SILENT_DB)))
    return folded
