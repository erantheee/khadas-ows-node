"""Role-aware envelope strategy: role hysteresis, scenario profiles, envelope timing."""
import sys
from pathlib import Path

import numpy as np

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
