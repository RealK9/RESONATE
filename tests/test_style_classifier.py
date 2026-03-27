"""Tests for the style classifier module."""
import numpy as np
import pytest

from ml.analysis.style_classifier import StyleClassifier
from ml.models.mix_profile import (
    MixLevelAnalysis,
    MixProfile,
    SourceRolePresence,
    SpectralOccupancy,
    StereoWidth,
    StyleCluster,
)

# All 14 expected clusters
EXPECTED_CLUSTERS = {
    "2010s_edm_drop", "2020s_melodic_house", "2000s_pop_chorus", "1990s_boom_bap",
    "modern_trap", "modern_drill", "melodic_techno", "afro_house", "cinematic",
    "lo_fi_chill", "dnb", "ambient", "r_and_b", "pop_production",
}


@pytest.fixture(scope="session")
def mix_profile_for_style() -> MixProfile:
    """
    Build a realistic MixProfile by hand (no audio file needed).
    Values resemble a 128 BPM melodic house track.
    """
    analysis = MixLevelAnalysis(
        bpm=126.0,
        bpm_confidence=0.85,
        key="A minor",
        key_confidence=0.75,
        tonal_center=440.0,
        harmonic_density=0.52,
        duration=30.0,
        loudness_lufs=-14.0,
        loudness_range=6.0,
        peak=0.95,
        dynamic_range=10.0,
        section_energy=[0.4, 0.5, 0.7, 0.9, 1.0, 0.8, 0.6, 0.5],
    )

    # 10-band spectral occupancy — moderate energy spread, emphasis on mids
    mean_by_band = [0.15, 0.20, 0.30, 0.45, 0.50, 0.40, 0.25, 0.15, 0.08, 0.04]
    spectral_occupancy = SpectralOccupancy(
        bands=[
            "sub", "bass", "low_mid", "mid", "upper_mid",
            "presence", "brilliance", "air", "ultra_high", "ceiling",
        ],
        time_frames=30,
        occupancy_matrix=[
            [e + np.random.default_rng(i).normal(0, 0.02)] * 30
            for i, e in enumerate(mean_by_band)
        ],
        mean_by_band=mean_by_band,
    )

    stereo_width = StereoWidth(
        bands=spectral_occupancy.bands,
        width_by_band=[0.1, 0.15, 0.3, 0.5, 0.6, 0.65, 0.7, 0.6, 0.4, 0.3],
        overall_width=0.58,
        correlation=0.72,
    )

    source_roles = SourceRolePresence(roles={
        "kick": 0.7, "snare_clap": 0.5, "hats_tops": 0.6, "bass": 0.65,
        "lead": 0.4, "chord_support": 0.5, "pad": 0.55, "vocal_texture": 0.3,
        "fx_transitions": 0.2, "ambience": 0.35,
    })

    # Density map with moderate variation
    rng = np.random.default_rng(42)
    density_map = list(np.clip(0.5 + 0.15 * rng.standard_normal(16), 0.0, 1.0))

    return MixProfile(
        filepath="/tmp/test_melodic_house.wav",
        filename="test_melodic_house.wav",
        analysis=analysis,
        spectral_occupancy=spectral_occupancy,
        stereo_width=stereo_width,
        source_roles=source_roles,
        density_map=density_map,
    )


@pytest.fixture(scope="session")
def style_result(mix_profile_for_style) -> StyleCluster:
    """Run the classifier once and cache."""
    classifier = StyleClassifier()
    return classifier.classify(mix_profile_for_style)


class TestStyleClassifier:
    """Tests for StyleClassifier.classify()."""

    def test_returns_style_cluster(self, style_result):
        assert isinstance(style_result, StyleCluster)

    def test_probabilities_sum_near_one(self, style_result):
        total = sum(style_result.cluster_probabilities.values())
        assert abs(total - 1.0) < 1e-6, f"Sum of probabilities = {total}"

    def test_all_clusters_present(self, style_result):
        assert set(style_result.cluster_probabilities.keys()) == EXPECTED_CLUSTERS

    def test_primary_cluster_is_string(self, style_result):
        assert isinstance(style_result.primary_cluster, str)
        assert len(style_result.primary_cluster) > 0
        assert style_result.primary_cluster in EXPECTED_CLUSTERS

    def test_era_estimate_present(self, style_result):
        assert isinstance(style_result.era_estimate, str)
        assert len(style_result.era_estimate) > 0

    def test_probabilities_in_range(self, style_result):
        for name, prob in style_result.cluster_probabilities.items():
            assert 0.0 <= prob <= 1.0, f"{name} probability {prob} out of range"
