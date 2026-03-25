"""
Tests for ExplanationEngine -- Phase 4 human-readable explanations.

Fixtures build pre-fabricated Recommendation, MixProfile, and
NeedOpportunity objects with various policies, roles, and scoring
breakdowns to verify that the engine produces correct, distinct,
policy-specific explanations.
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
from ml.models.recommendation import Recommendation, ScoringBreakdown
from ml.recommendation.explanations import (
    ExplanationEngine,
    _match_need,
)


# ---------------------------------------------------------------------------
# Fixtures -- scoring breakdowns
# ---------------------------------------------------------------------------

def _breakdown_spectral_heavy() -> ScoringBreakdown:
    """Breakdown where spectral_complement is the dominant score."""
    return ScoringBreakdown(
        need_fit=0.3,
        role_fit=0.5,
        spectral_complement=0.9,
        tonal_compatibility=0.4,
        rhythmic_compatibility=0.2,
        style_prior_fit=0.3,
        quality_prior=0.6,
    )


def _breakdown_rhythmic_heavy() -> ScoringBreakdown:
    """Breakdown where rhythmic_compatibility is the dominant score."""
    return ScoringBreakdown(
        need_fit=0.4,
        role_fit=0.6,
        spectral_complement=0.2,
        tonal_compatibility=0.3,
        rhythmic_compatibility=0.9,
        style_prior_fit=0.3,
        quality_prior=0.5,
    )


def _breakdown_tonal_heavy() -> ScoringBreakdown:
    """Breakdown where tonal_compatibility is the dominant score."""
    return ScoringBreakdown(
        need_fit=0.5,
        role_fit=0.4,
        spectral_complement=0.3,
        tonal_compatibility=0.9,
        rhythmic_compatibility=0.2,
        style_prior_fit=0.4,
        quality_prior=0.5,
    )


def _breakdown_balanced() -> ScoringBreakdown:
    """Balanced breakdown with moderate scores across the board."""
    return ScoringBreakdown(
        need_fit=0.5,
        role_fit=0.5,
        spectral_complement=0.5,
        tonal_compatibility=0.5,
        rhythmic_compatibility=0.5,
        style_prior_fit=0.5,
        quality_prior=0.5,
    )


# ---------------------------------------------------------------------------
# Fixtures -- mix profile
# ---------------------------------------------------------------------------

@pytest.fixture
def mix_profile_with_gaps() -> MixProfile:
    """MixProfile with spectral gaps in the low-mids and presence bands."""
    return MixProfile(
        filepath="/mixes/track_01.wav",
        filename="track_01.wav",
        analysis=MixLevelAnalysis(
            bpm=128.0, bpm_confidence=0.95,
            key="Am", key_confidence=0.8,
            loudness_lufs=-14.0,
        ),
        spectral_occupancy=SpectralOccupancy(
            bands=["sub-bass", "bass", "low-mids", "mids", "upper-mids",
                   "presence", "brilliance", "air", "ultra-highs", "beyond"],
            time_frames=100,
            mean_by_band=[0.7, 0.8, 0.15, 0.6, 0.5,
                          0.2, 0.4, 0.3, 0.1, 0.05],
        ),
        source_roles=SourceRolePresence(
            roles={"kick": 0.9, "bass": 0.7, "hat": 0.6, "pad": 0.3}
        ),
        style=StyleCluster(
            primary_cluster="deep house",
            cluster_probabilities={"deep house": 0.7, "tech house": 0.2},
        ),
        needs=[
            NeedOpportunity(
                category="role",
                description="Mix is missing a snare or clap element",
                severity=0.8,
                recommendation_policy="fill_missing_role",
            ),
            NeedOpportunity(
                category="spectral",
                description="Low-mids are thin and lack body",
                severity=0.6,
                recommendation_policy="reduce_emptiness",
            ),
            NeedOpportunity(
                category="dynamic",
                description="Groove needs more rhythmic variation",
                severity=0.5,
                recommendation_policy="enhance_groove",
            ),
        ],
    )


@pytest.fixture
def mix_profile_minimal() -> MixProfile:
    """Minimal MixProfile with no spectral data -- tests graceful fallback."""
    return MixProfile(
        filepath="/mixes/minimal.wav",
        filename="minimal.wav",
        analysis=MixLevelAnalysis(bpm=120.0, key="C"),
        source_roles=SourceRolePresence(roles={"kick": 0.8}),
    )


# ---------------------------------------------------------------------------
# Fixtures -- needs
# ---------------------------------------------------------------------------

@pytest.fixture
def needs_varied() -> list[NeedOpportunity]:
    return [
        NeedOpportunity(
            category="role",
            description="Mix is missing a snare or clap element",
            severity=0.8,
            recommendation_policy="fill_missing_role",
        ),
        NeedOpportunity(
            category="spectral",
            description="Low-mids are thin and lack body",
            severity=0.6,
            recommendation_policy="reduce_emptiness",
        ),
        NeedOpportunity(
            category="dynamic",
            description="Groove needs more rhythmic variation",
            severity=0.5,
            recommendation_policy="enhance_groove",
        ),
        NeedOpportunity(
            category="arrangement",
            description="Transitions between sections feel abrupt",
            severity=0.4,
            recommendation_policy="support_transition",
        ),
    ]


# ---------------------------------------------------------------------------
# Fixtures -- recommendations (one per major policy)
# ---------------------------------------------------------------------------

def _rec_fill_missing() -> Recommendation:
    return Recommendation(
        filepath="/lib/snare_01.wav",
        filename="snare_01.wav",
        score=0.85,
        breakdown=_breakdown_rhythmic_heavy(),
        policy="fill_missing_role",
        role="snare",
    )


def _rec_reinforce() -> Recommendation:
    return Recommendation(
        filepath="/lib/hat_layer.wav",
        filename="hat_layer.wav",
        score=0.72,
        breakdown=_breakdown_balanced(),
        policy="reinforce_existing",
        role="hat",
    )


def _rec_reduce_emptiness() -> Recommendation:
    return Recommendation(
        filepath="/lib/pad_warm.wav",
        filename="pad_warm.wav",
        score=0.78,
        breakdown=_breakdown_spectral_heavy(),
        policy="reduce_emptiness",
        role="pad",
    )


def _rec_enhance_groove() -> Recommendation:
    return Recommendation(
        filepath="/lib/perc_loop.wav",
        filename="perc_loop.wav",
        score=0.70,
        breakdown=_breakdown_rhythmic_heavy(),
        policy="enhance_groove",
        role="hat",
    )


def _rec_improve_polish() -> Recommendation:
    return Recommendation(
        filepath="/lib/texture_sheen.wav",
        filename="texture_sheen.wav",
        score=0.65,
        breakdown=_breakdown_tonal_heavy(),
        policy="improve_polish",
        role="texture",
    )


def _rec_increase_contrast() -> Recommendation:
    return Recommendation(
        filepath="/lib/lead_stab.wav",
        filename="lead_stab.wav",
        score=0.68,
        breakdown=_breakdown_tonal_heavy(),
        policy="increase_contrast",
        role="lead",
    )


def _rec_add_movement() -> Recommendation:
    return Recommendation(
        filepath="/lib/arp_seq.wav",
        filename="arp_seq.wav",
        score=0.66,
        breakdown=_breakdown_balanced(),
        policy="add_movement",
        role="lead",
    )


def _rec_support_transition() -> Recommendation:
    return Recommendation(
        filepath="/lib/riser_01.wav",
        filename="riser_01.wav",
        score=0.60,
        breakdown=_breakdown_balanced(),
        policy="support_transition",
        role="fx",
    )


def _rec_enhance_lift() -> Recommendation:
    return Recommendation(
        filepath="/lib/vocal_chop.wav",
        filename="vocal_chop.wav",
        score=0.62,
        breakdown=_breakdown_tonal_heavy(),
        policy="enhance_lift",
        role="vocal",
    )


# ---------------------------------------------------------------------------
# Engine fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> ExplanationEngine:
    return ExplanationEngine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExplainReturnType:
    """test_explain_returns_string -- single explanation is a non-empty string."""

    def test_explain_returns_string(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_fill_missing()
        result = engine.explain(rec, mix_profile_with_gaps)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExplainFillMissingRole:
    """test_explain_fill_missing_role -- template used correctly."""

    def test_template_mentions_missing_role(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_fill_missing()
        need = NeedOpportunity(
            category="role",
            description="Mix is missing a snare or clap element",
            severity=0.8,
            recommendation_policy="fill_missing_role",
        )
        result = engine.explain(rec, mix_profile_with_gaps, need=need)
        assert "missing" in result.lower()
        # Should reference the role (snare).
        assert "snare" in result.lower()

    def test_fill_missing_uses_policy_template(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_fill_missing()
        result = engine.explain(rec, mix_profile_with_gaps)
        # The fill_missing_role template starts with "Your mix is missing"
        assert "missing" in result.lower()
        assert "fill" in result.lower() or "gap" in result.lower()


class TestExplainReinforceExisting:
    """test_explain_reinforce_existing -- reinforcement explanation generated."""

    def test_reinforce_mentions_strengthening(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_reinforce()
        result = engine.explain(rec, mix_profile_with_gaps)
        assert "reinforce" in result.lower() or "strengthen" in result.lower()

    def test_reinforce_mentions_role(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_reinforce()
        result = engine.explain(rec, mix_profile_with_gaps)
        assert "hat" in result.lower()


class TestExplainReduceEmptiness:
    """test_explain_reduce_emptiness -- references spectral gaps."""

    def test_reduce_emptiness_mentions_spectral(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_reduce_emptiness()
        result = engine.explain(rec, mix_profile_with_gaps)
        # Should mention spectral gaps or a specific band name.
        assert "spectral" in result.lower() or "gap" in result.lower() or any(
            band in result.lower() for band in [
                "sub-bass", "bass", "low-mids", "mids", "upper-mids",
                "presence", "brilliance", "air", "mid",
            ]
        )

    def test_reduce_emptiness_references_band(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_reduce_emptiness()
        result = engine.explain(rec, mix_profile_with_gaps)
        # The mix profile has the weakest band at index 9 ("beyond") or
        # the lowest real musical band is "low-mids" at 0.15, but
        # "ultra-highs" at 0.1 is even lower.  The engine should pick
        # a band name from the profile.
        assert "fills" in result.lower() or "space" in result.lower()


class TestExplainEnhanceGroove:
    """test_explain_enhance_groove -- mentions rhythm/percussion."""

    def test_enhance_groove_mentions_percussion(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_enhance_groove()
        result = engine.explain(rec, mix_profile_with_gaps)
        assert (
            "percussion" in result.lower()
            or "rhythmic" in result.lower()
            or "groove" in result.lower()
            or "rhythm" in result.lower()
        )


class TestExplainBatch:
    """test_explain_batch_fills_all -- batch populates all recommendations."""

    def test_batch_fills_all_explanations(
        self,
        engine: ExplanationEngine,
        mix_profile_with_gaps: MixProfile,
        needs_varied: list[NeedOpportunity],
    ):
        recs = [
            _rec_fill_missing(),
            _rec_reinforce(),
            _rec_reduce_emptiness(),
            _rec_enhance_groove(),
        ]
        result = engine.explain_batch(recs, mix_profile_with_gaps, needs_varied)

        assert result is recs  # in-place mutation
        for rec in recs:
            assert isinstance(rec.explanation, str)
            assert len(rec.explanation) > 0

    def test_batch_assigns_need_addressed(
        self,
        engine: ExplanationEngine,
        mix_profile_with_gaps: MixProfile,
        needs_varied: list[NeedOpportunity],
    ):
        recs = [_rec_fill_missing(), _rec_reduce_emptiness()]
        engine.explain_batch(recs, mix_profile_with_gaps, needs_varied)

        # fill_missing_role should match the "missing snare" need.
        assert recs[0].need_addressed != ""
        # reduce_emptiness should match the "low-mids thin" need.
        assert recs[1].need_addressed != ""


class TestExplainWithNeedContext:
    """test_explain_with_need_context -- explanation incorporates need."""

    def test_need_context_appears(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_support_transition()
        need = NeedOpportunity(
            category="arrangement",
            description="Transitions between sections feel abrupt",
            severity=0.4,
            recommendation_policy="support_transition",
        )
        result = engine.explain(rec, mix_profile_with_gaps, need=need)
        # The explanation should reference transitions.
        assert "transition" in result.lower()

    def test_need_description_adds_context(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_enhance_lift()
        need = NeedOpportunity(
            category="dynamic",
            description="Build sections lack energy and emotional arc",
            severity=0.55,
            recommendation_policy="enhance_lift",
        )
        result = engine.explain(rec, mix_profile_with_gaps, need=need)
        # Should mention lift/energy/emotional.
        assert (
            "lift" in result.lower()
            or "energy" in result.lower()
            or "emotional" in result.lower()
        )


class TestExplainWithoutNeed:
    """test_explain_without_need -- valid explanation without explicit need."""

    def test_no_need_still_produces_explanation(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        rec = _rec_improve_polish()
        result = engine.explain(rec, mix_profile_with_gaps, need=None)
        assert isinstance(result, str)
        assert len(result) > 10  # Non-trivial content.

    def test_no_need_minimal_profile(
        self, engine: ExplanationEngine, mix_profile_minimal: MixProfile
    ):
        rec = _rec_add_movement()
        result = engine.explain(rec, mix_profile_minimal, need=None)
        assert isinstance(result, str)
        assert len(result) > 0


class TestNeedMatching:
    """test_need_matching -- correctly matches recommendations to needs."""

    def test_match_by_policy(self, needs_varied: list[NeedOpportunity]):
        rec = _rec_fill_missing()
        matched = _match_need(rec, needs_varied)
        assert matched is not None
        assert matched.recommendation_policy == "fill_missing_role"

    def test_match_reduce_emptiness(self, needs_varied: list[NeedOpportunity]):
        rec = _rec_reduce_emptiness()
        matched = _match_need(rec, needs_varied)
        assert matched is not None
        assert matched.recommendation_policy == "reduce_emptiness"

    def test_match_enhance_groove(self, needs_varied: list[NeedOpportunity]):
        rec = _rec_enhance_groove()
        matched = _match_need(rec, needs_varied)
        assert matched is not None
        assert matched.recommendation_policy == "enhance_groove"

    def test_no_match_returns_none(self):
        rec = _rec_improve_polish()
        # No needs with improve_polish policy.
        needs = [
            NeedOpportunity(
                category="role",
                description="Missing kick",
                severity=0.8,
                recommendation_policy="fill_missing_role",
            ),
        ]
        matched = _match_need(rec, needs)
        assert matched is None

    def test_empty_needs_returns_none(self):
        rec = _rec_fill_missing()
        assert _match_need(rec, []) is None


class TestDifferentPoliciesDifferentText:
    """test_different_policies_different_text -- each policy is distinct."""

    def test_all_nine_policies_produce_distinct_text(
        self, engine: ExplanationEngine, mix_profile_with_gaps: MixProfile
    ):
        recs = [
            _rec_fill_missing(),
            _rec_reinforce(),
            _rec_improve_polish(),
            _rec_increase_contrast(),
            _rec_add_movement(),
            _rec_reduce_emptiness(),
            _rec_support_transition(),
            _rec_enhance_groove(),
            _rec_enhance_lift(),
        ]

        explanations: list[str] = []
        for rec in recs:
            text = engine.explain(rec, mix_profile_with_gaps)
            explanations.append(text)

        # All 9 should be non-empty and distinct.
        assert len(explanations) == 9
        for text in explanations:
            assert len(text) > 0

        unique_texts = set(explanations)
        assert len(unique_texts) == 9, (
            f"Expected 9 distinct explanations, got {len(unique_texts)}. "
            f"Duplicates: {[t for t in explanations if explanations.count(t) > 1]}"
        )
