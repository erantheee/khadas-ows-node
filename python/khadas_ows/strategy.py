"""Role-aware envelope-modulation strategy (port of the legacy stem-remix policy).

The consumer of separation is not the stem itself. The product chain is:

    stems -> dominant musical roles -> per-stem gains for the current noise scenario
          -> time-domain envelope shaping (attack / hold / release) -> remixed music

This module ports the parts of the legacy `tools/stem_remix/stem_remix.py` that the
listening/envelope output needs, with four stems (vocals/drums/bass/other) because the
six-stem model is not real-time on the current candidate. The rolle mapping follows the
legacy ROLE_MAP with `guitar`/`piano` folded into `other`:

    lead   in {vocals, other}
    rhythm in {drums}
    low    in {bass, drums}
    bed    in {other}

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


def dominant_roles(stats_db: dict[str, float]) -> dict[str, str]:
    """Pick the strongest candidate for every role from per-stem RMS (dB)."""
    return {role: max(candidates, key=lambda stem: stats_db[stem])
            for role, candidates in ROLE_MAP.items()}


class RoleTracker:
    """Hysteresis over per-window dominant roles so small energy changes do not pump.

    Legacy `smooth_role_windows`: a challenger must lead the incumbent by `margin_db`
    for `hold_windows` consecutive windows before the role flips.
    """

    def __init__(self, margin_db: float = 2.0, hold_windows: int = 2) -> None:
        self.margin_db = float(margin_db)
        self.hold_windows = int(hold_windows)
        self.roles: dict[str, str] | None = None
        self._held = {role: 0 for role in ROLE_MAP}

    def update(self, stats_db: dict[str, float]) -> dict[str, str]:
        if self.roles is None:
            self.roles = dominant_roles(stats_db)
            return dict(self.roles)
        strongest = dominant_roles(stats_db)
        for role in ROLE_MAP:
            challenger, owner = strongest[role], self.roles[role]
            if challenger == owner:
                self._held[role] += 1
                continue
            lead_db = stats_db[challenger] - stats_db[owner]
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
