"""Tests for Phase 3 reference profiles — StylePrior, ReferenceCorpus,
ReferenceProfileBuilder, and DefaultPriors."""
from __future__ import annotations

import json

import pytest

from ml.analysis.reference_profiles import (
    ALL_CLUSTERS,
    DefaultPriors,
    ReferenceProfileBuilder,
)
from ml.models.mix_profile import (
    MixLevelAnalysis,
    MixProfile,
    SourceRolePresence,
    SpectralOccupancy,
    StereoWidth,
    StyleCluster,
)
from ml.models.reference_profile import ReferenceCorpus, StylePrior


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BAND_NAMES = [
    "sub", "bass", "low_mid", "mid", "upper_mid",
    "presence", "brilliance", "air", "ultra_high", "ceiling",
]

_ROLE_NAMES = [
    "kick", "snare_clap", "hats_tops", "bass", "lead",
    "chord_support", "pad", "vocal_texture", "fx_transitions", "ambience",
]


def _make_mix_profile(
    cluster: str = "modern_trap",
    spectral: list[float] | None = None,
    width: list[float] | None = None,
    roles: dict[str, float] | None = None,
    density_map: list[float] | None = None,
    section_energy: list[float] | None = None,
    harmonic_density: float = 0.4,
) -> MixProfile:
    """Build a minimal but valid MixProfile for testing."""
    if spectral is None:
        spectral = [0.5] * 10
    if width is None:
        width = [0.4] * 10
    if roles is None:
        roles = {r: 0.5 for r in _ROLE_NAMES}
    if density_map is None:
        density_map = [0.5] * 16
    if section_energy is None:
        section_energy = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5]

    return MixProfile(
        filepath="/tmp/test.wav",
        filename="test.wav",
        analysis=MixLevelAnalysis(
            bpm=140.0,
            bpm_confidence=0.9,
            key="C minor",
            key_confidence=0.85,
            harmonic_density=harmonic_density,
            section_energy=section_energy,
        ),
        spectral_occupancy=SpectralOccupancy(
            bands=_BAND_NAMES,
            time_frames=16,
            mean_by_band=spectral,
        ),
        stereo_width=StereoWidth(
            bands=_BAND_NAMES,
            width_by_band=width,
            overall_width=0.5,
        ),
        source_roles=SourceRolePresence(roles=roles),
        style=StyleCluster(
            cluster_probabilities={cluster: 0.8},
            primary_cluster=cluster,
            era_estimate="2020s",
        ),
        density_map=density_map,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDefaultPriors:
    """Tests for DefaultPriors.get_corpus()."""

    def test_default_priors_has_all_clusters(self) -> None:
        corpus = DefaultPriors.get_corpus()
        for cluster in ALL_CLUSTERS:
            assert cluster in corpus.priors, f"Missing cluster: {cluster}"
        assert len(corpus.priors) == 14

    def test_style_prior_fields_populated(self) -> None:
        corpus = DefaultPriors.get_corpus()
        for name, prior in corpus.priors.items():
            assert prior.cluster_name == name
            assert prior.target_spectral_mean, f"{name}: empty spectral_mean"
            assert prior.target_spectral_std, f"{name}: empty spectral_std"
            assert prior.typical_roles, f"{name}: empty typical_roles"
            assert prior.target_width_by_band, f"{name}: empty width_by_band"
            assert prior.section_lift_pattern, f"{name}: empty section_lift"
            assert prior.common_complements, f"{name}: empty common_complements"
            assert prior.confidence > 0.0, f"{name}: zero confidence"

    def test_spectral_mean_length(self) -> None:
        corpus = DefaultPriors.get_corpus()
        for name, prior in corpus.priors.items():
            assert len(prior.target_spectral_mean) == 10, (
                f"{name}: spectral_mean has {len(prior.target_spectral_mean)} bands, expected 10"
            )
            assert len(prior.target_spectral_std) == 10, (
                f"{name}: spectral_std has {len(prior.target_spectral_std)} bands, expected 10"
            )
            assert len(prior.target_width_by_band) == 10, (
                f"{name}: width_by_band has {len(prior.target_width_by_band)} bands, expected 10"
            )

    def test_roles_dict_populated(self) -> None:
        corpus = DefaultPriors.get_corpus()
        for name, prior in corpus.priors.items():
            for role in _ROLE_NAMES:
                assert role in prior.typical_roles, (
                    f"{name}: missing role '{role}'"
                )
                val = prior.typical_roles[role]
                assert 0.0 <= val <= 1.0, (
                    f"{name}: role '{role}' confidence {val} out of [0,1]"
                )


class TestCorpusRoundtrip:
    """Tests for to_dict / from_dict / to_json round-trip."""

    def test_corpus_roundtrip(self) -> None:
        original = DefaultPriors.get_corpus()
        d = original.to_dict()

        # Verify it's JSON-serializable
        json_str = json.dumps(d)
        d_parsed = json.loads(json_str)

        restored = ReferenceCorpus.from_dict(d_parsed)

        assert set(restored.priors.keys()) == set(original.priors.keys())
        assert restored.version == original.version
        assert restored.total_references == original.total_references

        for name in original.priors:
            orig = original.priors[name]
            rest = restored.priors[name]
            assert rest.cluster_name == orig.cluster_name
            assert rest.target_spectral_mean == pytest.approx(orig.target_spectral_mean)
            assert rest.target_spectral_std == pytest.approx(orig.target_spectral_std)
            assert rest.typical_roles == pytest.approx(orig.typical_roles)
            assert rest.target_density_mean == pytest.approx(orig.target_density_mean)
            assert rest.target_density_range == pytest.approx(orig.target_density_range)
            assert rest.target_width_by_band == pytest.approx(orig.target_width_by_band)
            assert rest.target_overall_width == pytest.approx(orig.target_overall_width)
            assert rest.section_lift_pattern == pytest.approx(orig.section_lift_pattern)
            assert rest.common_complements == orig.common_complements
            assert rest.confidence == pytest.approx(orig.confidence)


class TestReferenceProfileBuilder:
    """Tests for ReferenceProfileBuilder."""

    def test_builder_adds_reference(self) -> None:
        builder = ReferenceProfileBuilder()
        profile = _make_mix_profile(cluster="modern_trap")
        builder.add_reference(profile)

        assert "modern_trap" in builder.cluster_names
        assert builder.total_references == 1

    def test_builder_builds_corpus(self) -> None:
        builder = ReferenceProfileBuilder()

        # Add 3 profiles with different spectral values
        for i in range(3):
            spectral = [0.3 + i * 0.1] * 10
            width = [0.2 + i * 0.1] * 10
            profile = _make_mix_profile(
                cluster="melodic_techno",
                spectral=spectral,
                width=width,
                harmonic_density=0.3 + i * 0.1,
            )
            builder.add_reference(profile)

        corpus = builder.build_corpus()
        prior = corpus.get_prior("melodic_techno")

        assert prior is not None
        assert prior.cluster_name == "melodic_techno"
        assert prior.reference_count == 3
        assert prior.confidence > 0.0

        # Mean of [0.3, 0.4, 0.5] = 0.4 for each band
        assert prior.target_spectral_mean[0] == pytest.approx(0.4, abs=0.01)
        # Std should be > 0 (there is variance)
        assert prior.target_spectral_std[0] > 0.0
        # Width mean of [0.2, 0.3, 0.4] = 0.3
        assert prior.target_width_by_band[0] == pytest.approx(0.3, abs=0.01)
        # Tonal complexity = mean of [0.3, 0.4, 0.5] = 0.4
        assert prior.tonal_complexity == pytest.approx(0.4, abs=0.01)

    def test_builder_empty_cluster(self) -> None:
        builder = ReferenceProfileBuilder()
        # Add to one cluster but ask about another
        profile = _make_mix_profile(cluster="modern_trap")
        builder.add_reference(profile)

        corpus = builder.build_corpus()
        assert corpus.get_prior("ambient") is None

    def test_builder_cluster_override(self) -> None:
        builder = ReferenceProfileBuilder()
        profile = _make_mix_profile(cluster="modern_trap")
        builder.add_reference(profile, cluster_override="lo_fi_chill")

        assert "lo_fi_chill" in builder.cluster_names
        assert "modern_trap" not in builder.cluster_names

    def test_builder_no_cluster_skips(self) -> None:
        builder = ReferenceProfileBuilder()
        # Profile with empty cluster and no override -> should be skipped
        profile = _make_mix_profile(cluster="")
        builder.add_reference(profile)

        assert builder.total_references == 0
