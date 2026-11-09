"""Board-side S3 tuner: mode routing, D39 curve, D41/D48 masking, D56 matrix, fail-open."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from khadas_ows import tuning as T


class FixedEvidence(T.EvidenceReader):
    def __init__(self, data):
        super().__init__(path="/nonexistent-evidence.json")
        self.data = data

    def read(self):
        return self.data


class FixedStatus(T.NodeStatusReader):
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


def make_tuner(depth=1.3, level=-42.0, ambient=None, event="quiet", voice=0.02,
               conversation="IDLE", content=None, shares=None):
    ambient = ambient if ambient is not None else [-95.0] * T.BAND_COUNT
    content = content if content is not None else np.linspace(-45.0, -25.0, T.BAND_COUNT)
    evidence = {"level_dbfs": level, "band_dbfs": list(ambient),
                "event": event, "voice_likelihood": voice}
    return T.BoardTuner(depth=depth,
                        evidence=FixedEvidence(evidence),
                        node_status=FixedStatus({"policy_state": conversation})), content, shares


def step_many(tuner, content, shares=None, steps=6, dt_s=1.0):
    result = None
    for _ in range(steps):
        result = tuner.step(
            level_dbfs=tuner.evidence.read()["level_dbfs"],
            ambient_bands=tuner.evidence.read()["band_dbfs"],
            event=tuner.evidence.read()["event"],
            voice=tuner.evidence.read()["voice_likelihood"],
            conversation=tuner.node_status.read().get("policy_state"),
            content_bands=content,
            stem_shares=shares,
            dt_s=dt_s,
        )
    return result


def test_quiet_room_is_flat():
    tuner, content, _ = make_tuner(level=-70.0)
    result = step_many(tuner, content)
    assert result["mode"] == T.MODE_FLAT
    assert np.max(np.abs(result["applied_db"])) <= 0.05


def test_steady_noise_uses_d39_shape():
    tuner, content, _ = make_tuner(level=-42.0)
    result = step_many(tuner, content)
    assert result["mode"] == T.MODE_SONG_EQ
    device = result["device_db"]
    assert device[1] > 0.3, "low-frequency content prior must be positive"
    assert device[3] < -0.3 and device[4] < -0.3, "346-842 Hz de-mud must be negative"
    assert device[13] > 0.3, "presence hump must be positive"
    assert np.max(np.abs(device)) <= T.DEVICE_LIMIT_DB + 1e-6
    assert np.max(np.abs(result["applied_db"])) <= T.MAX_POSITIVE_DB + 1e-6
    assert result["applied_db"].min() >= T.MIN_APPLIED_DB - 1e-6


def test_loud_room_is_safety_hold_negative_only():
    tuner, content, _ = make_tuner(level=-25.0)
    result = step_many(tuner, content)
    assert result["mode"] == T.MODE_SAFETY_HOLD
    assert np.max(result["device_db"]) <= 0.05, "safety hold may not boost"
    assert result["device_db"].min() < 0, "de-mud cuts survive safety hold"


def test_background_voice_enters_evi_and_masks_presence():
    tuner, content, _ = make_tuner(level=-42.0, voice=0.9, event="speech")
    result = step_many(tuner, content, steps=4)
    assert result["mode"] == T.MODE_EXTERNAL_VOICE
    assert result["device_db"][12] > 2.0, "background voice masks 1-4 kHz (D41)"


def test_confirmed_dialogue_is_flat():
    tuner, content, _ = make_tuner(level=-42.0, conversation="ENGAGED")
    result = step_many(tuner, content)
    assert result["mode"] == T.MODE_FLAT
    assert np.max(np.abs(result["applied_db"])) <= 0.05


def test_anticipating_needs_preview_confirmation():
    """D51: raw ANTICIPATING must not flatten before 500 ms of preview."""
    tuner, content, _ = make_tuner(level=-42.0, conversation="ANTICIPATING")
    brief = step_many(tuner, content, steps=2, dt_s=0.2)   # 400 ms
    assert brief["mode"] != T.MODE_FLAT
    confirmed = step_many(tuner, content, steps=2, dt_s=0.2)  # 800 ms total
    assert confirmed["mode"] == T.MODE_FLAT


def test_masking_matrix_fills_noise_masked_band():
    """Noise dominates band 10 while the song has a dip there -> D56 boosts that band."""
    ambient = [-72.0] * T.BAND_COUNT
    ambient[10] = -40.0
    content = np.full(T.BAND_COUNT, -30.0)
    content[10] = -42.0  # relative -12 dB vs max; noise at band10 is +32 dB above it
    tuner, content, _ = make_tuner(level=-42.0, ambient=ambient, content=content)
    result = step_many(tuner, content, steps=6)
    assert result["mode"] == T.MODE_SONG_EQ
    assert result["masking_db"][10] > 0.5, "masked band must be compensated"
    assert result["masking_db"][22] < 0.5, "unmasked high band must not be forced up"


def test_masking_matrix_stays_off_without_gap():
    # Noise only prominent in band 5, where the song also has energy -> no gap anywhere.
    ambient = np.full(T.BAND_COUNT, -95.0)
    ambient[5] = -40.0
    content = np.full(T.BAND_COUNT, -30.0)
    content[5] = -20.0
    tuner, content, _ = make_tuner(level=-42.0, ambient=ambient.tolist(), content=content)
    result = step_many(tuner, content, steps=6)
    assert float(np.max(result["masking_db"])) < 0.5, "no gap -> no invented presence"


def test_style_bass_heavy_and_shares():
    content = np.full(T.BAND_COUNT, -60.0)
    content[0:6] = -20.0
    offsets, style = T.style_offsets(content, None)
    assert style == "bass_heavy"
    assert offsets[0] < 0, "bass-heavy songs must not stack more low boost"
    _, vocal_style = T.style_offsets(np.full(T.BAND_COUNT, -40.0),
                                     {"vocals": 0.5, "drums": 0.1, "bass": 0.2, "other": 0.2})
    assert "vocal" in vocal_style


def test_stale_evidence_fails_open():
    tuner = T.BoardTuner(evidence=FixedEvidence(None),
                         node_status=FixedStatus({"policy_state": "IDLE"}))
    result = tuner.step(level_dbfs=None, ambient_bands=None, event="quiet", voice=0.0,
                        conversation="IDLE", content_bands=np.zeros(T.BAND_COUNT),
                        stem_shares=None, dt_s=1.0)
    assert result["mode"] == T.MODE_FLAT
    assert result["stale"] is True
    assert np.max(np.abs(result["applied_db"])) == 0.0


def test_band_map_round_trip():
    band_map = T.BandMap()
    flat = np.zeros(T.BAND_COUNT)
    bins = band_map.gains_to_bins(flat)
    assert bins.shape == (1025,)
    assert np.allclose(bins, 0.0)
    boosted = flat.copy()
    boosted[10] = 6.0
    bins = band_map.gains_to_bins(boosted)
    center_hz = T.BAND_CENTERS_HZ[10]
    bin_index = int(round(center_hz * 2048 / 44100))
    assert abs(bins[bin_index] - 6.0) < 0.6, "band 10 gain must land at its center bin"
    power = np.zeros(1025)
    power[bin_index] = 1.0
    bands = band_map.band_power(power)
    assert int(np.argmax(bands)) == 10
