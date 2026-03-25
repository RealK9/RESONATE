"""
Tests for Reranker -- Phase 4 candidate scoring and ranking.

Uses the same _SAMPLE_DEFS pattern and _make_sample helper from the
candidate generator tests.  Mix profiles are created with specific
spectral occupancy data for spectral scoring tests.
"""
from __future__ import annotations

import pytest

from ml.models.mix_profile import (
    MixLevelAnalysis,
    MixProfile,
    NeedOpportunity,
    SourceRolePresence,
    SpectralOccupancy,
    StyleCluster,
)
from ml.models.sample_profile import (
    CoreDescriptors,
    HarmonicDescriptors,
    PredictedLabels,
    SampleProfile,
    SpectralDescriptors,
    TransientDescriptors,
)
from ml.models.recommendation import Recommendation, ScoringBreakdown
from ml.models.reference_profile import ReferenceCorpus, StylePrior
from ml.recommendation.reranker import Reranker


# ---------------------------------------------------------------------------
# Sample library fixture -- reuses the same pattern as candidate_generator
# ---------------------------------------------------------------------------

_SAMPLE_DEFS: list[dict] = [
    # Kicks
    {"fp": "/lib/kick_01.wav", "role": "kick", "tonal": False, "cr": 0.7,
     "role_conf": 0.85, "centroid": 100.0, "onset_rate": 2.0},
    {"fp": "/lib/kick_02.wav", "role": "kick", "tonal": False, "cr": 0.8,
     "role_conf": 0.9, "centroid": 120.0, "onset_rate": 2.0},
    # Snares
    {"fp": "/lib/snare_01.wav", "role": "snare", "tonal": False, "cr": 0.6,
     "role_conf": 0.8, "centroid": 2500.0, "onset_rate": 2.0},
    {"fp": "/lib/snare_02.wav", "role": "snare", "tonal": False, "cr": 0.75,
     "role_conf": 0.85, "centroid": 2800.0, "onset_rate": 2.0},
    # Hats
    {"fp": "/lib/hat_01.wav", "role": "hat", "tonal": False, "cr": 0.7,
     "role_conf": 0.88, "centroid": 10000.0, "onset_rate": 4.0},
    # Bass -- tonal
    {"fp": "/lib/bass_01.wav", "role": "bass", "tonal": True, "cr": 0.8,
     "key": "C", "role_conf": 0.9, "centroid": 150.0, "onset_rate": 2.0},
    {"fp": "/lib/bass_02.wav", "role": "bass", "tonal": True, "cr": 0.7,
     "key": "G", "role_conf": 0.85, "centroid": 180.0, "onset_rate": 2.0},
    {"fp": "/lib/bass_03.wav", "role": "bass", "tonal": True, "cr": 0.65,
     "key": "F#", "role_conf": 0.8, "centroid": 160.0, "onset_rate": 2.0},
    # Leads -- tonal
    {"fp": "/lib/lead_01.wav", "role": "lead", "tonal": True, "cr": 0.75,
     "key": "C", "role_conf": 0.82, "centroid": 3000.0, "onset_rate": 3.0},
    {"fp": "/lib/lead_02.wav", "role": "lead", "tonal": True, "cr": 0.5,
     "key": "Eb", "role_conf": 0.7, "centroid": 3500.0, "onset_rate": 3.0},
    # Pads -- tonal
    {"fp": "/lib/pad_01.wav", "role": "pad", "tonal": True, "cr": 0.8,
     "key": "C", "role_conf": 0.85, "centroid": 1500.0, "onset_rate": 0.5},
    {"fp": "/lib/pad_02.wav", "role": "pad", "tonal": True, "cr": 0.7,
     "key": "Dm", "role_conf": 0.78, "centroid": 1200.0, "onset_rate": 0.4},
    # Textures
    {"fp": "/lib/texture_01.wav", "role": "texture", "tonal": False, "cr": 0.55,
     "role_conf": 0.75, "centroid": 5000.0, "onset_rate": 1.0},
    # FX
    {"fp": "/lib/fx_01.wav", "role": "fx", "tonal": False, "cr": 0.6,
     "role_conf": 0.7, "centroid": 6000.0, "onset_rate": 1.5},
    # Vocals -- tonal
    {"fp": "/lib/vocal_01.wav", "role": "vocal", "tonal": True, "cr": 0.85,
     "key": "C", "role_conf": 0.92, "centroid": 2000.0, "onset_rate": 3.0},
    # Low quality pad
    {"fp": "/lib/pad_lowq.wav", "role": "pad", "tonal": True, "cr": 0.05,
     "key": "C", "role_conf": 0.5, "centroid": 1400.0, "onset_rate": 0.3},
]


def _make_sample(d: dict) -> SampleProfile:
    """Create a SampleProfile from a compact definition dict."""
    fp = d["fp"]
    filename = fp.rsplit("/", 1)[-1]
    key = d.get("key", "")
    chroma: list[float] = []
    pitch_conf = 0.0
    if key:
        note_names = [
            "C", "Db", "D", "Eb", "E", "F",
            "F#", "G", "Ab", "A", "Bb", "B",
        ]
        root = key.rstrip("m")
        chroma = [0.1] * 12
        if root in note_names:
            chroma[note_names.index(root)] = 0.9
        pitch_conf = 0.8

    return SampleProfile(
        filepath=fp,
        filename=filename,
        core=CoreDescriptors(duration=0.5, sample_rate=44100, channels=1),
        spectral=SpectralDescriptors(centroid=d.get("centroid", 0.0)),
        harmonic=HarmonicDescriptors(
            chroma_profile=chroma,
            pitch_confidence=pitch_conf,
        ),
        transients=TransientDescriptors(
            onset_rate=d.get("onset_rate", 0.0),
        ),
        labels=PredictedLabels(
            role=d["role"],
            role_confidence=d.get("role_conf", 0.8),
            tonal=d["tonal"],
            commercial_readiness=d["cr"],
        ),
    )


# ---------------------------------------------------------------------------
# Mix profile helpers
# ---------------------------------------------------------------------------

_10_BANDS = [
    "sub", "bass", "low_mid", "mid", "upper_mid",
    "presence", "brilliance", "air", "ultra_high", "nyquist",
]


def _mix_with_spectral(
    *,
    key: str = "",
    bpm: float = 120.0,
    mean_by_band: list[float] | None = None,
    style: str = "modern_trap",
    needs: list[NeedOpportunity] | None = None,
) -> MixProfile:
    """Create a MixProfile with specific spectral occupancy."""
    if mean_by_band is None:
        mean_by_band = [0.5] * 10  # Neutral occupancy.

    return MixProfile(
        filepath="/mixes/test_mix.wav",
        filename="test_mix.wav",
        analysis=MixLevelAnalysis(
            key=key,
            key_confidence=0.9 if key else 0.0,
            bpm=bpm,
            bpm_confidence=0.9 if bpm > 0 else 0.0,
        ),
        spectral_occupancy=SpectralOccupancy(
            bands=_10_BANDS[:len(mean_by_band)],
            time_frames=100,
            occupancy_matrix=[],
            mean_by_band=mean_by_band,
        ),
        style=StyleCluster(primary_cluster=style),
        source_roles=SourceRolePresence(roles={
            "kick": 0.1,
            "snare_clap": 0.1,
            "bass": 0.6,
        }),
        needs=needs or [],
    )


def _default_needs() -> list[NeedOpportunity]:
    return [
        NeedOpportunity(
            category="role",
            description="Weak attack -- kick and snare presence is very low",
            severity=0.9,
            recommendation_policy="fill_missing_role",
        ),
        NeedOpportunity(
            category="role",
            description="No harmonic layer -- needs pad or lead",
            severity=0.6,
            recommendation_policy="reduce_emptiness",
        ),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def all_samples() -> list[SampleProfile]:
    return [_make_sample(d) for d in _SAMPLE_DEFS]


@pytest.fixture
def reranker() -> Reranker:
    return Reranker()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReranker:
    """Test suite for the Reranker scoring pipeline."""

    def test_rerank_returns_recommendations(self, all_samples, reranker):
        """rerank() returns a sorted list of Recommendation objects."""
        mix = _mix_with_spectral()
        needs = _default_needs()
        results = reranker.rerank(all_samples, mix, needs)

        assert isinstance(results, list)
        assert len(results) == len(all_samples)
        assert all(isinstance(r, Recommendation) for r in results)
        # Verify descending sort.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_bounded(self, all_samples, reranker):
        """All composite scores and breakdown components are in [0, 1]."""
        mix = _mix_with_spectral(key="C", bpm=120.0)
        needs = _default_needs()
        results = reranker.rerank(all_samples, mix, needs)

        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of bounds for {r.filepath}"
            bd = r.breakdown
            assert 0.0 <= bd.need_fit <= 1.0
            assert 0.0 <= bd.role_fit <= 1.0
            assert 0.0 <= bd.spectral_complement <= 1.0
            assert 0.0 <= bd.tonal_compatibility <= 1.0
            assert 0.0 <= bd.rhythmic_compatibility <= 1.0
            assert 0.0 <= bd.style_prior_fit <= 1.0
            assert 0.0 <= bd.quality_prior <= 1.0
            assert 0.0 <= bd.user_preference <= 1.0
            assert 0.0 <= bd.masking_penalty <= 1.0
            assert 0.0 <= bd.redundancy_penalty <= 1.0

    def test_high_need_fit_ranked_higher(self, reranker):
        """Sample filling a high-severity need ranks above one filling a low-severity need."""
        # enhance_groove maps to: hat, kick, snare, clap
        high_need = NeedOpportunity(
            category="role",
            description="Critical: needs groove",
            severity=0.95,
            recommendation_policy="enhance_groove",
        )
        # support_transition maps to: fx, texture
        low_need = NeedOpportunity(
            category="role",
            description="Minor: could use transition FX",
            severity=0.2,
            recommendation_policy="support_transition",
        )

        # Kick addresses enhance_groove (sev 0.95) but NOT support_transition.
        # FX addresses support_transition (sev 0.2) but NOT enhance_groove.
        kick = _make_sample({
            "fp": "/lib/kick_test.wav", "role": "kick", "tonal": False,
            "cr": 0.7, "role_conf": 0.8, "centroid": 100.0, "onset_rate": 2.0,
        })
        fx = _make_sample({
            "fp": "/lib/fx_test.wav", "role": "fx", "tonal": False,
            "cr": 0.7, "role_conf": 0.8, "centroid": 5000.0, "onset_rate": 1.0,
        })

        mix = _mix_with_spectral()
        results = reranker.rerank([kick, fx], mix, [high_need, low_need])

        assert results[0].filepath == "/lib/kick_test.wav"
        assert results[0].breakdown.need_fit > results[1].breakdown.need_fit

    def test_quality_affects_ranking(self, reranker):
        """Higher quality sample ranks above lower quality, all else equal."""
        high_q = _make_sample({
            "fp": "/lib/kick_hq.wav", "role": "kick", "tonal": False,
            "cr": 0.95, "role_conf": 0.8, "centroid": 100.0, "onset_rate": 2.0,
        })
        low_q = _make_sample({
            "fp": "/lib/kick_lq.wav", "role": "kick", "tonal": False,
            "cr": 0.3, "role_conf": 0.8, "centroid": 100.0, "onset_rate": 2.0,
        })

        mix = _mix_with_spectral()
        needs = [NeedOpportunity(
            category="role", description="Needs kick",
            severity=0.8, recommendation_policy="fill_missing_role",
        )]
        results = reranker.rerank([high_q, low_q], mix, needs)

        assert results[0].filepath == "/lib/kick_hq.wav"
        assert results[0].breakdown.quality_prior > results[1].breakdown.quality_prior

    def test_tonal_compatibility_scoring(self, reranker):
        """Same-key sample scores higher tonal compatibility than distant-key sample."""
        same_key = _make_sample({
            "fp": "/lib/bass_C.wav", "role": "bass", "tonal": True,
            "key": "C", "cr": 0.7, "role_conf": 0.8, "centroid": 150.0,
            "onset_rate": 2.0,
        })
        distant_key = _make_sample({
            "fp": "/lib/bass_Fsharp.wav", "role": "bass", "tonal": True,
            "key": "F#", "cr": 0.7, "role_conf": 0.8, "centroid": 150.0,
            "onset_rate": 2.0,
        })

        mix = _mix_with_spectral(key="C")
        needs = [NeedOpportunity(
            category="role", description="Needs bass",
            severity=0.8, recommendation_policy="reduce_emptiness",
        )]
        results = reranker.rerank([same_key, distant_key], mix, needs)

        # Find each result by filepath.
        same_result = next(r for r in results if r.filepath == "/lib/bass_C.wav")
        dist_result = next(r for r in results if r.filepath == "/lib/bass_Fsharp.wav")

        assert same_result.breakdown.tonal_compatibility == 1.0
        assert dist_result.breakdown.tonal_compatibility < 0.5
        assert same_result.score > dist_result.score

    def test_redundancy_penalty(self, reranker):
        """Selecting two same-role samples penalizes the second."""
        kick_a = _make_sample({
            "fp": "/lib/kick_a.wav", "role": "kick", "tonal": False,
            "cr": 0.7, "role_conf": 0.8, "centroid": 100.0, "onset_rate": 2.0,
        })
        kick_b = _make_sample({
            "fp": "/lib/kick_b.wav", "role": "kick", "tonal": False,
            "cr": 0.7, "role_conf": 0.8, "centroid": 100.0, "onset_rate": 2.0,
        })

        mix = _mix_with_spectral()
        needs = [NeedOpportunity(
            category="role", description="Needs kick",
            severity=0.8, recommendation_policy="fill_missing_role",
        )]

        # First pass: no already_selected.
        results_first = reranker.rerank([kick_b], mix, needs, already_selected=[])
        # Second pass: kick_a already selected.
        results_second = reranker.rerank([kick_b], mix, needs, already_selected=[kick_a])

        assert results_first[0].breakdown.redundancy_penalty == 0.0
        assert results_second[0].breakdown.redundancy_penalty == pytest.approx(0.3)
        assert results_first[0].score > results_second[0].score

    def test_empty_candidates_returns_empty(self, reranker):
        """Empty candidates list returns empty results."""
        mix = _mix_with_spectral()
        needs = _default_needs()
        results = reranker.rerank([], mix, needs)
        assert results == []

    def test_spectral_complement_scoring(self, reranker):
        """Sample in under-occupied band scores higher spectral complement."""
        # Band 0 (sub) is heavily occupied, band 8 (ultra_high) is empty.
        mean_by_band = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

        # Low-centroid sample (falls in occupied sub/bass bands).
        low_sample = _make_sample({
            "fp": "/lib/bass_low.wav", "role": "bass", "tonal": False,
            "cr": 0.7, "role_conf": 0.8, "centroid": 80.0, "onset_rate": 2.0,
        })
        # High-centroid sample (falls in under-occupied high band).
        high_sample = _make_sample({
            "fp": "/lib/hat_high.wav", "role": "hat", "tonal": False,
            "cr": 0.7, "role_conf": 0.8, "centroid": 13000.0, "onset_rate": 4.0,
        })

        mix = _mix_with_spectral(mean_by_band=mean_by_band)
        needs = [NeedOpportunity(
            category="role", description="Needs elements",
            severity=0.5, recommendation_policy="fill_missing_role",
        )]
        results = reranker.rerank([low_sample, high_sample], mix, needs)

        low_result = next(r for r in results if r.filepath == "/lib/bass_low.wav")
        high_result = next(r for r in results if r.filepath == "/lib/hat_high.wav")

        assert high_result.breakdown.spectral_complement > low_result.breakdown.spectral_complement

    def test_masking_penalty(self, reranker):
        """Sample in already-occupied band gets masking penalty."""
        # Band 1 (bass, 60-250 Hz) is highly occupied.
        mean_by_band = [0.3, 0.95, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

        # Bass sample with centroid in the occupied bass band.
        bass_sample = _make_sample({
            "fp": "/lib/bass_mask.wav", "role": "bass", "tonal": False,
            "cr": 0.7, "role_conf": 0.8, "centroid": 150.0, "onset_rate": 2.0,
        })
        # Hat sample with centroid in an unoccupied band.
        hat_sample = _make_sample({
            "fp": "/lib/hat_clear.wav", "role": "hat", "tonal": False,
            "cr": 0.7, "role_conf": 0.8, "centroid": 10000.0, "onset_rate": 4.0,
        })

        mix = _mix_with_spectral(mean_by_band=mean_by_band)
        needs = [NeedOpportunity(
            category="role", description="Needs elements",
            severity=0.5, recommendation_policy="fill_missing_role",
        )]
        results = reranker.rerank([bass_sample, hat_sample], mix, needs)

        bass_result = next(r for r in results if r.filepath == "/lib/bass_mask.wav")
        hat_result = next(r for r in results if r.filepath == "/lib/hat_clear.wav")

        assert bass_result.breakdown.masking_penalty > 0.0
        assert hat_result.breakdown.masking_penalty == 0.0

    def test_custom_weights(self, reranker):
        """Custom weights affect final composite scores."""
        # Create a reranker that heavily weights quality_prior.
        heavy_quality = Reranker(weights={"eta": 0.80, "alpha": 0.01})

        high_q = _make_sample({
            "fp": "/lib/pad_hq.wav", "role": "pad", "tonal": False,
            "cr": 0.95, "role_conf": 0.5, "centroid": 1500.0, "onset_rate": 0.5,
        })
        low_q = _make_sample({
            "fp": "/lib/pad_lq.wav", "role": "pad", "tonal": False,
            "cr": 0.2, "role_conf": 0.95, "centroid": 1500.0, "onset_rate": 0.5,
        })

        mix = _mix_with_spectral()
        needs = [NeedOpportunity(
            category="role", description="Needs pad",
            severity=0.8, recommendation_policy="reduce_emptiness",
        )]

        # With heavy quality weighting, high_q should win despite lower role_conf.
        results = heavy_quality.rerank([high_q, low_q], mix, needs)
        assert results[0].filepath == "/lib/pad_hq.wav"

        # With default weights (quality is only 0.10), the difference should
        # be smaller or even reversed (role_fit + need_fit dominate).
        results_default = reranker.rerank([high_q, low_q], mix, needs)
        # The quality gap should have less impact with default weights.
        score_gap_custom = results[0].score - results[1].score
        score_gap_default = abs(
            next(r for r in results_default if r.filepath == "/lib/pad_hq.wav").score
            - next(r for r in results_default if r.filepath == "/lib/pad_lq.wav").score
        )
        assert score_gap_custom > score_gap_default
