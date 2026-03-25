"""Tests for the mix analyzer module."""
import numpy as np
import pytest
import soundfile as sf

from ml.analysis.mix_analyzer import analyze_mix
from ml.models.mix_profile import MixProfile

SR = 44100


@pytest.fixture(scope="session")
def mix_fixture(tmp_path_factory):
    """
    3-second stereo mix layering kick pattern + bass sine + noise hat + pad.
    Designed to exercise all analysis paths.
    """
    duration = 3.0
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = np.random.default_rng(99)

    # --- Kick pattern: ~120 BPM = 0.5s between hits ---
    kick = np.zeros(n, dtype=np.float64)
    kick_interval = int(0.5 * SR)  # 120 BPM
    for start in range(0, n, kick_interval):
        seg_len = min(int(0.15 * SR), n - start)
        t_seg = np.linspace(0, 0.15, seg_len, endpoint=False)
        freq = 150 * np.exp(-30 * t_seg) + 40
        phase = 2 * np.pi * np.cumsum(freq) / SR
        env = np.exp(-20 * t_seg)
        kick[start: start + seg_len] += 0.7 * env * np.sin(phase)

    # --- Bass: sustained 80 Hz sine ---
    bass = 0.3 * np.sin(2 * np.pi * 80 * t)

    # --- Hi-hat pattern: noise bursts every 0.25s ---
    hat = np.zeros(n, dtype=np.float64)
    hat_interval = int(0.25 * SR)
    for start in range(0, n, hat_interval):
        seg_len = min(int(0.04 * SR), n - start)
        t_seg = np.linspace(0, 0.04, seg_len, endpoint=False)
        env = np.exp(-80 * t_seg)
        hat[start: start + seg_len] += 0.2 * env * rng.standard_normal(seg_len)

    # --- Pad: stacked detuned sines with slow attack ---
    pad_env = 1.0 - np.exp(-2 * t)
    pad = np.zeros(n, dtype=np.float64)
    for f in [220, 330, 440.3, 554]:
        pad += np.sin(2 * np.pi * f * t) / 4.0
    pad *= 0.25 * pad_env

    # --- Mix down to stereo with slight L/R offset for width ---
    mono = (kick + bass + hat + pad).astype(np.float32)
    # Normalize
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = mono / peak * 0.9
    left = mono.copy()
    right = np.roll(mono, int(0.002 * SR)).copy()
    # Add slight stereo difference
    right += 0.05 * rng.standard_normal(n).astype(np.float32)
    stereo = np.stack([left, right], axis=-1).astype(np.float32)

    path = tmp_path_factory.mktemp("mix") / "test_mix.wav"
    sf.write(str(path), stereo, SR)
    return str(path)


@pytest.fixture(scope="session")
def mix_profile(mix_fixture):
    """Run analyze_mix once and cache the result."""
    return analyze_mix(mix_fixture)


class TestMixAnalyzer:
    """Tests for analyze_mix function."""

    def test_returns_mix_profile(self, mix_profile):
        assert isinstance(mix_profile, MixProfile)

    def test_bpm_detected(self, mix_profile):
        assert mix_profile.analysis.bpm > 0

    def test_key_detected(self, mix_profile):
        assert isinstance(mix_profile.analysis.key, str)
        assert len(mix_profile.analysis.key) > 0

    def test_duration_correct(self, mix_profile):
        assert abs(mix_profile.analysis.duration - 3.0) < 0.1

    def test_spectral_occupancy_shape(self, mix_profile):
        occ = mix_profile.spectral_occupancy
        assert len(occ.bands) == 10
        assert occ.time_frames > 0
        assert len(occ.occupancy_matrix) == 10
        assert len(occ.occupancy_matrix[0]) == occ.time_frames
        assert len(occ.mean_by_band) == 10

    def test_stereo_width_range(self, mix_profile):
        sw = mix_profile.stereo_width
        assert len(sw.width_by_band) == 10
        for w in sw.width_by_band:
            assert 0.0 <= w <= 1.0
        assert 0.0 <= sw.overall_width <= 1.0

    def test_source_roles_populated(self, mix_profile):
        expected_keys = {
            "kick", "snare_clap", "hats_tops", "bass", "lead",
            "chord_support", "pad", "vocal_texture", "fx_transitions", "ambience",
        }
        assert set(mix_profile.source_roles.roles.keys()) == expected_keys
        for v in mix_profile.source_roles.roles.values():
            assert 0.0 <= v <= 1.0

    def test_density_map_populated(self, mix_profile):
        assert len(mix_profile.density_map) == 16
        for d in mix_profile.density_map:
            assert 0.0 <= d <= 1.0

    def test_section_energy_populated(self, mix_profile):
        assert len(mix_profile.analysis.section_energy) == 8
        for e in mix_profile.analysis.section_energy:
            assert 0.0 <= e <= 1.0

    def test_loudness_reasonable(self, mix_profile):
        lufs = mix_profile.analysis.loudness_lufs
        assert -60 <= lufs <= 0
