"""
Tests for NeedsEngine -- the core diagnostic module of RESONATE.

Each fixture creates a MixProfile designed to trigger specific needs,
allowing us to verify that the engine correctly identifies spectral gaps,
missing roles, dynamic problems, spatial issues, and arrangement weaknesses.
"""
from __future__ import annotations

import pytest

from ml.models.mix_profile import (
    MixLevelAnalysis,
    MixProfile,
    NeedOpportunity,
    SourceRolePresence,
    SpectralOccupancy,
    StereoWidth,
    StyleCluster,
)
from ml.analysis.needs_engine import (
    NeedsEngine,
    VALID_CATEGORIES,
    VALID_POLICIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BAND_NAMES = [
    "sub", "bass", "low_mid", "mid", "upper_mid",
    "presence", "brilliance", "air", "ultra_high", "ceiling",
]


def _make_profile(
    *,
    mean_by_band: list[float] | None = None,
    roles: dict[str, float] | None = None,
    primary_cluster: str = "modern_trap",
    overall_width: float = 0.5,
    width_by_band: list[float] | None = None,
    density_map: list[float] | None = None,
    section_energy: list[float] | None = None,
    dynamic_range: float = 10.0,
) -> MixProfile:
    """Build a MixProfile with controllable defaults for testing."""
    if mean_by_band is None:
        mean_by_band = [0.5] * 10
    if roles is None:
        roles = {
            "kick": 0.6, "snare_clap": 0.5, "hats_tops": 0.5,
            "bass": 0.5, "lead": 0.4, "chord_support": 0.4,
            "pad": 0.3, "vocal_texture": 0.3, "fx_transitions": 0.2,
            "ambience": 0.3,
        }
    if width_by_band is None:
        width_by_band = [0.1, 0.1, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3]
    if density_map is None:
        density_map = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6,
                       0.5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.6, 0.4]
    if section_energy is None:
        section_energy = [0.3, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3]

    return MixProfile(
        filepath="/test/mix.wav",
        filename="mix.wav",
        analysis=MixLevelAnalysis(
            bpm=140.0,
            bpm_confidence=0.9,
            key="C minor",
            key_confidence=0.8,
            tonal_center=261.63,
            harmonic_density=0.4,
            duration=30.0,
            loudness_lufs=-14.0,
            loudness_range=6.0,
            peak=0.95,
            dynamic_range=dynamic_range,
            section_energy=section_energy,
        ),
        spectral_occupancy=SpectralOccupancy(
            bands=list(BAND_NAMES),
            time_frames=100,
            occupancy_matrix=[],
            mean_by_band=mean_by_band,
        ),
        stereo_width=StereoWidth(
            bands=list(BAND_NAMES),
            width_by_band=width_by_band,
            overall_width=overall_width,
            correlation=0.7,
        ),
        source_roles=SourceRolePresence(roles=roles),
        style=StyleCluster(
            cluster_probabilities={primary_cluster: 0.8},
            primary_cluster=primary_cluster,
            era_estimate="2020s",
        ),
        needs=[],
        density_map=density_map,
    )


# ---------------------------------------------------------------------------
# Targeted fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sparse_top_end_mix() -> MixProfile:
    """High-frequency bands far below the style norm for modern_trap."""
    # modern_trap norms for air/brilliance/ultra_high: ~0.55, 0.50, 0.50
    return _make_profile(
        mean_by_band=[0.80, 0.70, 0.30, 0.35, 0.40, 0.45, 0.10, 0.05, 0.05, 0.02],
        primary_cluster="modern_trap",
    )


@pytest.fixture
def overcrowded_mids_mix() -> MixProfile:
    """Upper-mid and presence bands far above style norm."""
    # modern_trap norms for upper_mid/presence: ~0.40, 0.45
    return _make_profile(
        mean_by_band=[0.80, 0.70, 0.30, 0.35, 0.85, 0.90, 0.50, 0.55, 0.50, 0.35],
        primary_cluster="modern_trap",
    )


@pytest.fixture
def no_rhythm_mix() -> MixProfile:
    """Zero kick/snare confidence in a rhythmic style."""
    return _make_profile(
        roles={
            "kick": 0.0, "snare_clap": 0.0, "hats_tops": 0.0,
            "bass": 0.5, "lead": 0.4, "chord_support": 0.4,
            "pad": 0.3, "vocal_texture": 0.3, "fx_transitions": 0.2,
            "ambience": 0.3,
        },
        primary_cluster="modern_trap",
    )


@pytest.fixture
def narrow_mix() -> MixProfile:
    """Very low stereo width (essentially mono)."""
    return _make_profile(
        overall_width=0.03,
        width_by_band=[0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02],
    )


@pytest.fixture
def flat_arrangement_mix() -> MixProfile:
    """Uniform density and flat section energy -- no movement."""
    return _make_profile(
        density_map=[0.50] * 16,
        section_energy=[0.50] * 8,
    )


@pytest.fixture
def well_balanced_mix() -> MixProfile:
    """A mix that closely matches the pop_production norm."""
    # pop_production norms: [0.35, 0.45, 0.50, 0.55, 0.55, 0.55, 0.50, 0.45, 0.35, 0.20]
    return _make_profile(
        mean_by_band=[0.36, 0.46, 0.50, 0.54, 0.54, 0.54, 0.49, 0.44, 0.34, 0.20],
        primary_cluster="pop_production",
        roles={
            "kick": 0.6, "snare_clap": 0.5, "hats_tops": 0.5,
            "bass": 0.5, "lead": 0.5, "chord_support": 0.5,
            "pad": 0.4, "vocal_texture": 0.4, "fx_transitions": 0.2,
            "ambience": 0.4,
        },
        overall_width=0.5,
        width_by_band=[0.08, 0.08, 0.3, 0.4, 0.5, 0.55, 0.55, 0.5, 0.4, 0.3],
        density_map=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8,
                     0.6, 0.5, 0.6, 0.7, 0.8, 0.9, 0.7, 0.4],
        section_energy=[0.3, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3],
        dynamic_range=10.0,
    )


# ---------------------------------------------------------------------------
# Engine instance
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> NeedsEngine:
    return NeedsEngine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNeedsEngineBasics:
    """Core contract: returns a list of NeedOpportunity objects."""

    def test_returns_list_of_needs(self, engine: NeedsEngine, sparse_top_end_mix: MixProfile):
        result = engine.diagnose(sparse_top_end_mix)
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, NeedOpportunity)

    def test_need_fields_valid(self, engine: NeedsEngine, sparse_top_end_mix: MixProfile):
        """Every need has a valid category, non-empty description, severity 0-1, valid policy."""
        result = engine.diagnose(sparse_top_end_mix)
        for need in result:
            assert need.category in VALID_CATEGORIES, (
                f"Invalid category: {need.category}"
            )
            assert len(need.description) > 0, "Description must be non-empty"
            assert 0.0 <= need.severity <= 1.0, (
                f"Severity out of range: {need.severity}"
            )
            assert need.recommendation_policy in VALID_POLICIES, (
                f"Invalid policy: {need.recommendation_policy}"
            )

    def test_policies_are_valid(self, engine: NeedsEngine):
        """All needs from various profiles use valid policies."""
        profiles = [
            _make_profile(mean_by_band=[0.80, 0.70, 0.30, 0.35, 0.40, 0.45, 0.10, 0.05, 0.05, 0.02]),
            _make_profile(overall_width=0.03),
            _make_profile(density_map=[0.50] * 16, section_energy=[0.50] * 8),
        ]
        for p in profiles:
            for need in engine.diagnose(p):
                assert need.recommendation_policy in VALID_POLICIES


class TestSpectralDiagnosis:
    """Spectral gap detection."""

    def test_sparse_top_end_detected(self, engine: NeedsEngine, sparse_top_end_mix: MixProfile):
        needs = engine.diagnose(sparse_top_end_mix)
        spectral_needs = [n for n in needs if n.category == "spectral"]
        descriptions = " ".join(n.description.lower() for n in spectral_needs)
        assert "top-end too sparse" in descriptions, (
            f"Expected 'top-end too sparse' in spectral needs, got: {[n.description for n in spectral_needs]}"
        )

    def test_overcrowded_mids_detected(self, engine: NeedsEngine, overcrowded_mids_mix: MixProfile):
        needs = engine.diagnose(overcrowded_mids_mix)
        spectral_needs = [n for n in needs if n.category == "spectral"]
        descriptions = " ".join(n.description.lower() for n in spectral_needs)
        assert "upper mids overcrowded" in descriptions, (
            f"Expected 'upper mids overcrowded', got: {[n.description for n in spectral_needs]}"
        )


class TestRoleDiagnosis:
    """Missing role detection."""

    def test_missing_rhythm_detected(self, engine: NeedsEngine, no_rhythm_mix: MixProfile):
        needs = engine.diagnose(no_rhythm_mix)
        role_needs = [n for n in needs if n.category == "role"]
        descriptions = " ".join(n.description.lower() for n in role_needs)
        assert "weak attack support" in descriptions, (
            f"Expected 'weak attack support', got: {[n.description for n in role_needs]}"
        )

    def test_no_hats_detected_in_rhythmic_style(self, engine: NeedsEngine, no_rhythm_mix: MixProfile):
        """The no_rhythm_mix also has zero hats_tops."""
        needs = engine.diagnose(no_rhythm_mix)
        role_needs = [n for n in needs if n.category == "role"]
        descriptions = " ".join(n.description.lower() for n in role_needs)
        assert "rhythmic sparkle" in descriptions


class TestSpatialDiagnosis:
    """Spatial issues detection."""

    def test_narrow_width_detected(self, engine: NeedsEngine, narrow_mix: MixProfile):
        needs = engine.diagnose(narrow_mix)
        spatial_needs = [n for n in needs if n.category == "spatial"]
        descriptions = " ".join(n.description.lower() for n in spatial_needs)
        assert "too narrow" in descriptions, (
            f"Expected 'too narrow', got: {[n.description for n in spatial_needs]}"
        )


class TestArrangementDiagnosis:
    """Arrangement weakness detection."""

    def test_flat_arrangement_detected(self, engine: NeedsEngine, flat_arrangement_mix: MixProfile):
        needs = engine.diagnose(flat_arrangement_mix)
        arr_needs = [n for n in needs if n.category == "arrangement"]
        descriptions = " ".join(n.description.lower() for n in arr_needs)
        assert "no movement" in descriptions or "static" in descriptions, (
            f"Expected arrangement issue about flatness, got: {[n.description for n in arr_needs]}"
        )


class TestBalancedMix:
    """A well-balanced mix should trigger fewer needs."""

    def test_balanced_mix_fewer_needs(
        self,
        engine: NeedsEngine,
        well_balanced_mix: MixProfile,
        sparse_top_end_mix: MixProfile,
    ):
        balanced_needs = engine.diagnose(well_balanced_mix)
        sparse_needs = engine.diagnose(sparse_top_end_mix)
        assert len(balanced_needs) < len(sparse_needs), (
            f"Balanced mix ({len(balanced_needs)} needs) should have fewer "
            f"needs than sparse mix ({len(sparse_needs)} needs)"
        )


class TestSeverityOrdering:
    """More extreme deficiencies should produce higher severity."""

    def test_severity_ordering(self, engine: NeedsEngine):
        """A mix with extreme deficits should have higher max severity than a mild one."""
        # Mild deficit: slightly below norm
        mild = _make_profile(
            mean_by_band=[0.80, 0.70, 0.30, 0.35, 0.40, 0.45, 0.35, 0.35, 0.30, 0.20],
            primary_cluster="modern_trap",
        )
        # Extreme deficit: far below norm
        extreme = _make_profile(
            mean_by_band=[0.80, 0.70, 0.30, 0.35, 0.40, 0.45, 0.02, 0.01, 0.01, 0.01],
            primary_cluster="modern_trap",
        )
        mild_needs = engine.diagnose(mild)
        extreme_needs = engine.diagnose(extreme)

        # Filter to spectral needs about top end
        mild_top = [
            n for n in mild_needs
            if n.category == "spectral" and "top-end" in n.description.lower()
        ]
        extreme_top = [
            n for n in extreme_needs
            if n.category == "spectral" and "top-end" in n.description.lower()
        ]

        # The extreme case must have a top-end need
        assert len(extreme_top) > 0, "Extreme deficit should trigger top-end need"

        # If both have the need, extreme should be higher severity
        if mild_top and extreme_top:
            assert extreme_top[0].severity >= mild_top[0].severity, (
                f"Extreme ({extreme_top[0].severity}) should be >= "
                f"mild ({mild_top[0].severity})"
            )
