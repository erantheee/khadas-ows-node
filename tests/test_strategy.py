"""Role-aware envelope strategy: role hysteresis, scenario profiles, envelope timing."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from khadas_ows import strategy


def test_dominant_roles_pick_strongest_candidate():
    stats = {"vocals": -20.0, "drums": -18.0, "bass": -25.0, "other": -22.0}
    roles = strategy.dominant_roles(stats)
    assert roles == {"lead": "vocals", "rhythm": "drums", "low": "drums", "bed": "other"}


def test_role_hysteresis_needs_margin_and_hold():
    tracker = strategy.RoleTracker(margin_db=2.0, hold_windows=2)
    first = {"vocals": -20.0, "drums": -18.0, "bass": -25.0, "other": -22.0}
    assert tracker.update(first)["lead"] == "vocals"
    # other must lead vocals by 2 dB for 2 consecutive windows before the role flips
    flip_a = {"vocals": -20.0, "drums": -18.0, "bass": -25.0, "other": -17.0}
    # legacy smooth_role_windows: held counter must reach hold_windows before a flip,
    # so the challenger needs hold_windows + 1 consecutive qualifying windows.
    assert tracker.update(flip_a)["lead"] == "vocals"
    assert tracker.update(flip_a)["lead"] == "vocals"
    assert tracker.update(flip_a)["lead"] == "other"    # now flips
    # a small lead (< margin) never flips even after many windows
    close = {"vocals": -20.0, "drums": -18.0, "bass": -25.0, "other": -19.0}
    for _ in range(5):
        assert tracker.update(close)["lead"] == "other"


def test_scenario_profiles_and_contrast_only_are_attenuation_only():
    roles = {"lead": "vocals", "rhythm": "drums", "low": "bass", "bed": "other"}
    quiet = strategy.role_profile(roles, "quiet")
    assert all(gain == 0.0 for gain in quiet.values())
    broadband = strategy.role_profile(roles, "broadband")
    assert broadband["vocals"] == 3.0 and broadband["drums"] == 1.5 and broadband["bass"] == 1.0
    wind = strategy.role_profile(roles, "wind")
    assert wind["bass"] == -1.0  # do not spend headroom pushing low end into wind
    contrast = strategy.contrast_only_profile(broadband, strength=0.6)
    assert max(contrast.values()) == 0.0
    assert all(value <= 0.0 for value in contrast.values())
    assert contrast["vocals"] == 0.0
    deep = strategy.contrast_only_profile(broadband, strength=1.0)
    assert deep["drums"] < contrast["drums"]  # stronger strategy = more contrast


def test_envelope_attack_is_slower_than_release_but_both_move():
    follower = strategy.EnvelopeFollower(attack_ms=400.0, release_ms=1200.0)
    target = {"vocals": 0.0, "drums": -6.0, "bass": -6.0, "other": -6.0}
    after_one = follower.step(target, 0.1)   # 100 ms
    assert -6.0 < after_one["drums"] < 0.0
    for _ in range(100):                     # 10 s -> should be essentially at target
        follower.step(target, 0.1)
    assert abs(follower.gains["drums"] + 6.0) < 0.05
    # release back toward 0 dB must also converge
    for _ in range(200):
        follower.step({stem: 0.0 for stem in strategy.STEMMS}, 0.1)
    assert abs(follower.gains["drums"]) < 0.05


def test_scenario_window():
    assert strategy.scenario_active(5.0, 2.0, 10.0)
    assert not strategy.scenario_active(1.0, 2.0, 10.0)
    assert not strategy.scenario_active(10.0, 2.0, 10.0)


# --------------------------------------------------------------------------- #
# Six-stem strategy schema (Mac stem_remix.py parity; folded onto 4-stem execution)
# --------------------------------------------------------------------------- #
SILENT = strategy.SILENT_DB
SIX = strategy.STEMMS6


def _six_stats_without_guitar_piano():
    stats = {"vocals": -20.0, "drums": -18.0, "bass": -25.0, "other": -22.0}
    return {stem: stats.get(stem, SILENT) for stem in SIX}


def test_profiles_match_mac_stem_remix_values():
    assert strategy.PROFILES["lead_protect"]["vocals"] == 3.0
    assert strategy.PROFILES["lead_protect"]["drums"] == -1.0
    assert strategy.PROFILES["lead_protect"]["guitar"] == 1.5
    assert strategy.PROFILES["rhythm_protect"]["drums"] == 2.5
    assert strategy.PROFILES["low_protect"]["bass"] == 3.0
    assert all(value == 0.0 for value in strategy.PROFILES["neutral_rebuild"].values())


def test_six_stem_roles_degrade_when_guitar_piano_are_silent():
    roles = strategy.dominant_roles6(_six_stats_without_guitar_piano())
    # guitar/piano sit at SILENT_DB and can never win a role
    assert roles["lead"] == "vocals"
    assert roles["rhythm"] == "drums"
    assert roles["low"] == "drums"
    assert roles["bed"] == "other"


def test_six_stem_role_tracker_needs_margin_and_hold_for_guitar():
    tracker = strategy.RoleTracker(role_map=strategy.ROLE_MAP6)
    base = _six_stats_without_guitar_piano()
    assert tracker.update(base)["lead"] == "vocals"
    loud_guitar = dict(base, guitar=0.0)  # 20 dB above vocals
    assert tracker.update(loud_guitar)["lead"] == "vocals"   # held=1
    assert tracker.update(loud_guitar)["lead"] == "vocals"   # held=2
    assert tracker.update(loud_guitar)["lead"] == "guitar"   # flips after hold_windows


def test_role_profile6_matches_legacy_environment_targets():
    roles = {"lead": "vocals", "rhythm": "drums", "low": "bass", "bed": "other"}
    quiet = strategy.role_profile6(roles, "quiet")
    assert all(gain == 0.0 for gain in quiet.values())
    broadband = strategy.role_profile6(roles, "broadband")
    assert broadband["vocals"] == 3.0
    assert broadband["drums"] == 1.5
    assert broadband["bass"] == 1.0
    wind = strategy.role_profile6(roles, "wind")
    assert wind["bass"] == -1.0


def test_contrast_only_profile6_is_attenuation_only():
    roles = {"lead": "guitar", "rhythm": "drums", "low": "bass", "bed": "other"}
    raw = strategy.role_profile6(roles, "broadband")
    out = strategy.contrast_only_profile6(raw, strength=0.6)
    assert max(out.values()) == 0.0
    assert all(value <= 0.0 for value in out.values())
    assert out["guitar"] == 0.0  # protected lead, folded into other afterwards


def test_fold_to_execution_merges_guitar_piano_into_other():
    gains6 = {"vocals": -2.0, "drums": -3.0, "bass": -4.0,
              "guitar": -1.0, "piano": 0.0, "other": -2.5}
    folded = strategy.fold_to_execution(gains6)
    assert set(folded) == set(strategy.STEMMS)
    assert folded["vocals"] == -2.0
    assert folded["drums"] == -3.0
    assert folded["bass"] == -4.0
    assert folded["other"] == 0.0  # max(other, guitar, piano)
    # no guitar/piano information -> identity on the four-stem map
    ident = strategy.fold_to_execution(
        {"vocals": -1.0, "drums": -1.0, "bass": -1.0, "other": -1.0})
    assert ident == {"vocals": -1.0, "drums": -1.0, "bass": -1.0, "other": -1.0}


def test_automatic_profile6_protects_strongest_roles():
    stats = {"vocals": -20.0, "drums": -18.0, "bass": -25.0, "other": -22.0,
             "guitar": SILENT, "piano": SILENT}
    gains = strategy.automatic_profile6(stats)
    assert gains["vocals"] == 3.0   # lead
    assert gains["drums"] == 1.5    # rhythm (also strongest low candidate)
    assert gains["bass"] == -1.0    # not a strongest role
    assert gains["other"] == -1.0
    assert gains["guitar"] == -1.0  # silent and unselected


def test_remix_risk_is_bounded_and_monotone():
    assert strategy.remix_risk(-5.0, 1, 0.0) == 0.0
    assert strategy.remix_risk(30.0, 6, 1.0) == 1.0
    calm = strategy.remix_risk(12.0, 2, 0.0)
    dense = strategy.remix_risk(20.0, 5, 0.3)
    assert 0.0 <= calm < dense <= 1.0


def test_active_stem_count_uses_absolute_floor():
    stats = {"vocals": -20.0, "drums": SILENT, "bass": -70.0,
             "guitar": SILENT, "piano": SILENT, "other": -30.0}
    assert strategy.active_stem_count(stats) == 2


def test_environment_from_evidence_priority():
    assert strategy.environment_from_evidence(None) == "unknown"
    assert strategy.environment_from_evidence({}) == "unknown"
    assert strategy.environment_from_evidence({"context": "near_voice"}) == "speech"
    assert strategy.environment_from_evidence({"environment": "VOICE"}) == "speech"
    assert strategy.environment_from_evidence({"event": "wind", "environment": "NOISY"}) == "wind"
    assert strategy.environment_from_evidence({"event": "keyboard"}) == "keyboard"
    assert strategy.environment_from_evidence({"context": "babble", "environment": "LOUD"}) == "babble"
    assert strategy.environment_from_evidence({"context": "horn"}) == "office"
    assert strategy.environment_from_evidence({"environment": "LOUD"}) == "office"
    assert strategy.environment_from_evidence({"context": "quiet"}) == "quiet"


def test_environment_tracker_hysteresis_and_fail_open():
    tracker = strategy.EnvironmentTracker(hold_windows=2, fast_windows=1)
    assert tracker.update("quiet") == "quiet"
    assert tracker.update("street") == "quiet"    # one window is not enough
    assert tracker.update("street") == "street"   # second consecutive window wins
    assert tracker.update("speech") == "speech"   # active talker enters fast
    assert tracker.update("quiet") == "speech"    # release still needs the hold window
    assert tracker.update("quiet") == "quiet"
    assert tracker.update("unknown") == "quiet"   # missing evidence fails open
    assert tracker.source == "unknown"


def test_apply_environment_is_attenuation_only():
    music = strategy.BOOST_PROFILE
    quiet = strategy.apply_environment(music, "quiet")
    assert quiet == {stem: music[stem] for stem in strategy.STEMMS6}
    street = strategy.apply_environment(music, "street")
    assert street["vocals"] == music["vocals"]
    assert street["drums"] == pytest.approx(music["drums"] - 1.5)
    assert street["bass"] == pytest.approx(music["bass"] - 2.0)
    assert all(street[stem] <= music[stem] for stem in strategy.STEMMS6)
    deep = strategy.apply_environment({stem: -8.0 for stem in strategy.STEMMS6}, "speech")
    assert min(deep.values()) == strategy.ENV_TARGET_FLOOR_DB
