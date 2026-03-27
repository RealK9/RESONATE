"""End-to-end tests for the full v2 recommendation pipeline.

Tests the complete flow: analyze_mix -> style_classify -> needs_diagnose ->
gap_analysis (via genre blueprints) -> candidate_generate -> rerank -> explain.

All external dependencies (audio I/O, model weights, database) are mocked at
the boundary of each pipeline stage so no real files or models are needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml.analysis.genre_blueprints import (
    ROLE_NAMES,
    GenreBlueprint,
    all_blueprints,
    get_best_blueprint,
    get_blueprint,
)
from ml.analysis.needs_engine import NeedsEngine
from ml.analysis.style_classifier import StyleClassifier
from ml.models.mix_profile import (
    MixLevelAnalysis,
    MixProfile,
    NeedOpportunity,
    SourceRolePresence,
    SpectralOccupancy,
    StereoWidth,
    StyleCluster,
)
from ml.models.recommendation import Recommendation, RecommendationResult, ScoringBreakdown
from ml.models.reference_profile import ReferenceCorpus, StylePrior
from ml.models.sample_profile import (
    CoreDescriptors,
    HarmonicDescriptors,
    PredictedLabels,
    SampleProfile,
    SpectralDescriptors,
    TransientDescriptors,
)
from ml.recommendation.candidate_generator import CandidateGenerator
from ml.recommendation.explanations import ExplanationEngine
from ml.recommendation.reranker import Reranker


# ---------------------------------------------------------------------------
# Helpers — build realistic mock objects
# ---------------------------------------------------------------------------

def _make_mix_profile(
    *,
    cluster: str = "modern_trap",
    bpm: float = 145.0,
    key: str = "Am",
    roles: dict[str, float] | None = None,
    spectral_means: list[float] | None = None,
    needs: list[NeedOpportunity] | None = None,
) -> MixProfile:
    """Build a MixProfile with sensible defaults for testing."""
    if roles is None:
        roles = {"kick": 0.9, "bass": 0.8, "hats_tops": 0.7}
    if spectral_means is None:
        spectral_means = [0.7, 0.6, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.5, 0.35]

    bands = [
        "sub", "bass", "low_mid", "mid", "upper_mid",
        "presence", "brilliance", "air", "ultra_high", "ceiling",
    ]
    return MixProfile(
        filepath="/tmp/test_mix.wav",
        filename="test_mix.wav",
        analysis=MixLevelAnalysis(
            bpm=bpm,
            bpm_confidence=0.9,
            key=key,
            key_confidence=0.85,
            duration=180.0,
            loudness_lufs=-9.0,
            dynamic_range=6.0,
        ),
        spectral_occupancy=SpectralOccupancy(
            bands=bands,
            time_frames=100,
            occupancy_matrix=[],
            mean_by_band=spectral_means,
        ),
        stereo_width=StereoWidth(
            bands=bands,
            width_by_band=[0.5] * 10,
            overall_width=0.5,
            correlation=0.8,
        ),
        source_roles=SourceRolePresence(roles=roles),
        style=StyleCluster(
            cluster_probabilities={cluster: 0.85, "modern_drill": 0.10},
            primary_cluster=cluster,
            era_estimate="2020s",
        ),
        needs=needs or [],
    )


def _make_sample(
    filepath: str = "/samples/kick_01.wav",
    role: str = "kick",
    role_confidence: float = 0.9,
    commercial_readiness: float = 0.7,
    centroid: float = 200.0,
    onset_rate: float = 2.0,
    tonal: bool = False,
) -> SampleProfile:
    """Build a minimal SampleProfile for testing."""
    return SampleProfile(
        filepath=filepath,
        filename=filepath.split("/")[-1],
        core=CoreDescriptors(duration=0.5, rms=0.3),
        spectral=SpectralDescriptors(centroid=centroid),
        harmonic=HarmonicDescriptors(pitch_confidence=0.1),
        transients=TransientDescriptors(onset_rate=onset_rate),
        labels=PredictedLabels(
            role=role,
            role_confidence=role_confidence,
            tonal=tonal,
            commercial_readiness=commercial_readiness,
        ),
    )


def _make_needs() -> list[NeedOpportunity]:
    """Build a representative set of diagnosed needs."""
    return [
        NeedOpportunity(
            category="role",
            description="Missing snare/clap — the beat has no backbeat",
            severity=0.85,
            recommendation_policy="fill_missing_role",
        ),
        NeedOpportunity(
            category="spectral",
            description="Weak upper-mid presence — lacks definition",
            severity=0.60,
            recommendation_policy="reduce_emptiness",
        ),
        NeedOpportunity(
            category="arrangement",
            description="No pad or texture layer for harmonic glue",
            severity=0.50,
            recommendation_policy="fill_missing_role",
        ),
    ]


def _make_sample_library() -> list[SampleProfile]:
    """Build a small fake sample library with various roles."""
    return [
        _make_sample("/samples/kick_01.wav", "kick", 0.9, 0.8, 80.0, 2.0),
        _make_sample("/samples/snare_01.wav", "snare", 0.85, 0.75, 3000.0, 2.0),
        _make_sample("/samples/clap_01.wav", "clap", 0.80, 0.70, 2500.0, 2.0),
        _make_sample("/samples/hat_01.wav", "hat", 0.90, 0.65, 8000.0, 4.0),
        _make_sample("/samples/bass_01.wav", "bass", 0.88, 0.72, 150.0, 1.5, tonal=True),
        _make_sample("/samples/lead_01.wav", "lead", 0.75, 0.68, 2000.0, 3.0, tonal=True),
        _make_sample("/samples/pad_01.wav", "pad", 0.82, 0.70, 500.0, 0.5, tonal=True),
        _make_sample("/samples/texture_01.wav", "texture", 0.78, 0.60, 4000.0, 1.0),
        _make_sample("/samples/fx_01.wav", "fx", 0.70, 0.55, 6000.0, 5.0),
        _make_sample("/samples/vocal_01.wav", "vocal", 0.80, 0.65, 1500.0, 3.0, tonal=True),
    ]


def _mock_sample_store(library: list[SampleProfile]) -> MagicMock:
    """Create a mock SampleStore backed by an in-memory library."""
    store = MagicMock()
    store.init.return_value = None

    def search_by_role(role):
        return [s for s in library if s.labels.role == role]

    def load(filepath):
        for s in library:
            if s.filepath == filepath:
                return s
        return None

    store.search_by_role.side_effect = search_by_role
    store.load.side_effect = load
    return store


def _mock_corpus() -> ReferenceCorpus:
    """Build a minimal ReferenceCorpus with a trap prior."""
    prior = StylePrior(
        cluster_name="modern_trap",
        target_spectral_mean=[0.8, 0.7, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.5, 0.35],
        target_spectral_std=[0.05] * 10,
        typical_roles={"kick": 0.9, "snare": 0.85, "hat": 0.8, "bass": 0.9},
        common_complements=["lead", "pad", "fx"],
    )
    return ReferenceCorpus(
        priors={"modern_trap": prior},
        version="1.0",
        total_references=50,
    )


def _mock_preference_server() -> MagicMock:
    """Build a mock PreferenceServer that returns neutral scores."""
    server = MagicMock()
    server.is_loaded = False
    server.get_weight_adjustments.return_value = {}
    server.score.return_value = 0.0
    return server


# ---------------------------------------------------------------------------
# Gap analysis helpers (blueprint-based gap scoring)
# ---------------------------------------------------------------------------

def _gap_analysis(mix_profile: MixProfile) -> dict:
    """
    Compute a gap analysis result using genre blueprints.

    Returns a dict with:
      - production_readiness_score: int 0-100
      - missing_roles: list of role names missing from the mix
      - chart_potential_current: float 0-1
      - chart_potential_ceiling: float 0-1
      - spectral_gaps: list of band names with large deviations
      - blueprint_name: name of the matched blueprint
    """
    cluster_probs = mix_profile.style.cluster_probabilities
    if not cluster_probs:
        return {
            "production_readiness_score": 0,
            "missing_roles": list(ROLE_NAMES),
            "chart_potential_current": 0.0,
            "chart_potential_ceiling": 0.0,
            "spectral_gaps": [],
            "blueprint_name": "",
        }

    bp, bp_prob = get_best_blueprint(cluster_probs)
    roles = mix_profile.source_roles.roles

    # Determine missing roles
    missing = []
    present_count = 0
    for role_name, threshold in bp.required_roles.items():
        presence = roles.get(role_name, 0.0)
        if presence < threshold:
            missing.append(role_name)
        else:
            present_count += 1

    total_required = len(bp.required_roles)
    role_score = present_count / total_required if total_required > 0 else 1.0

    # Spectral deviation score
    spectral_gaps = []
    spectral_score = 1.0
    means = mix_profile.spectral_occupancy.mean_by_band
    if means and len(means) == len(bp.target_spectral):
        deviations = []
        for i, (actual, target, tol) in enumerate(
            zip(means, bp.target_spectral, bp.spectral_tolerance)
        ):
            dev = abs(actual - target)
            if dev > tol:
                band_name = mix_profile.spectral_occupancy.bands[i] if i < len(
                    mix_profile.spectral_occupancy.bands
                ) else f"band_{i}"
                spectral_gaps.append(band_name)
            deviations.append(min(dev / max(tol, 0.01), 2.0))
        avg_dev = sum(deviations) / len(deviations) if deviations else 0.0
        spectral_score = max(0.0, 1.0 - avg_dev * 0.5)

    # Composite production readiness
    weights = bp.scoring_weights
    readiness = (
        weights.get("role", 0.25) * role_score
        + weights.get("spectral", 0.25) * spectral_score
        + weights.get("dynamics", 0.15) * 0.7  # placeholder dynamics score
        + weights.get("perceptual", 0.15) * 0.7  # placeholder perceptual score
        + weights.get("arrangement", 0.20) * role_score
    )
    production_readiness_score = int(round(max(0.0, min(1.0, readiness)) * 100))

    chart_potential_ceiling = bp.chart_hit_rate
    chart_potential_current = chart_potential_ceiling * (production_readiness_score / 100.0)

    return {
        "production_readiness_score": production_readiness_score,
        "missing_roles": missing,
        "chart_potential_current": chart_potential_current,
        "chart_potential_ceiling": chart_potential_ceiling,
        "spectral_gaps": spectral_gaps,
        "blueprint_name": bp.name,
    }


# ===========================================================================
# Test 1: Full pipeline smoke test
# ===========================================================================

class TestFullPipelineSmokeTest:
    """Mock audio -> analyze_mix -> style_classify -> needs_diagnose ->
    gap_analysis -> candidate_generate -> rerank -> explain.

    Verify each stage produces valid output that feeds into the next."""

    @patch("ml.analysis.mix_analyzer.analyze_mix")
    def test_full_pipeline_end_to_end(self, mock_analyze_mix):
        """Run all pipeline stages sequentially and verify data flows correctly."""
        # Stage 1: analyze_mix — returns a MixProfile
        base_profile = _make_mix_profile(
            roles={"kick": 0.9, "bass": 0.8, "hats_tops": 0.7},
        )
        mock_analyze_mix.return_value = base_profile

        from ml.analysis.mix_analyzer import analyze_mix
        mix_profile = analyze_mix("/tmp/test_mix.wav")
        assert isinstance(mix_profile, MixProfile)
        assert mix_profile.analysis.bpm > 0
        assert mix_profile.analysis.key != ""

        # Stage 2: style_classify — populates style cluster
        classifier = StyleClassifier()
        result_style = classifier.classify(mix_profile)
        # The classifier writes directly into the profile
        assert mix_profile.style.primary_cluster != ""
        assert len(mix_profile.style.cluster_probabilities) > 0

        # Stage 3: needs_diagnose — produces NeedOpportunity list
        corpus = _mock_corpus()
        engine = NeedsEngine(corpus=corpus)
        needs = engine.diagnose(mix_profile)
        mix_profile.needs = needs
        assert isinstance(needs, list)
        # Each need must have required fields
        for need in needs:
            assert isinstance(need, NeedOpportunity)
            assert need.category in ("spectral", "role", "dynamic", "spatial", "arrangement")
            assert 0.0 <= need.severity <= 1.0
            assert need.recommendation_policy != ""

        # Stage 4: gap_analysis — uses genre blueprints
        gap_result = _gap_analysis(mix_profile)
        assert 0 <= gap_result["production_readiness_score"] <= 100
        assert isinstance(gap_result["missing_roles"], list)

        # Stage 5: candidate_generate
        library = _make_sample_library()
        store = _mock_sample_store(library)
        generator = CandidateGenerator(sample_store=store, vector_index=None)

        # Use the diagnosed needs (or fall back to synthetic needs if empty)
        effective_needs = needs if needs else _make_needs()
        candidates = generator.generate(mix_profile, effective_needs, max_candidates=50)
        assert isinstance(candidates, list)

        # Stage 6: rerank
        pref_server = _mock_preference_server()
        reranker = Reranker(corpus=corpus, preference_server=pref_server)
        recommendations = reranker.rerank(candidates, mix_profile, effective_needs)
        assert isinstance(recommendations, list)

        # Stage 7: explain
        explainer = ExplanationEngine()
        explainer.explain_batch(recommendations, mix_profile, effective_needs)

        # Verify final output structure
        for rec in recommendations:
            assert isinstance(rec, Recommendation)
            assert isinstance(rec.score, float)
            assert isinstance(rec.breakdown, ScoringBreakdown)

    @patch("ml.analysis.mix_analyzer.analyze_mix")
    def test_pipeline_output_is_serializable(self, mock_analyze_mix):
        """The final RecommendationResult must be JSON-serializable."""
        mix_profile = _make_mix_profile()
        mock_analyze_mix.return_value = mix_profile

        needs = _make_needs()
        mix_profile.needs = needs

        library = _make_sample_library()
        store = _mock_sample_store(library)
        generator = CandidateGenerator(sample_store=store, vector_index=None)
        candidates = generator.generate(mix_profile, needs, max_candidates=50)

        pref_server = _mock_preference_server()
        reranker = Reranker(corpus=_mock_corpus(), preference_server=pref_server)
        recommendations = reranker.rerank(candidates, mix_profile, needs)

        ExplanationEngine().explain_batch(recommendations, mix_profile, needs)

        result = RecommendationResult(
            mix_filepath=mix_profile.filepath,
            recommendations=recommendations,
            needs_addressed=list({r.need_addressed for r in recommendations if r.need_addressed}),
            total_candidates_considered=len(candidates),
        )

        # Must not raise
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 10


# ===========================================================================
# Test 2: Gap analysis integration
# ===========================================================================

class TestGapAnalysisIntegration:
    """Verify gap analysis correctly uses genre blueprints."""

    def test_production_readiness_range(self):
        """production_readiness_score must be an int in [0, 100]."""
        for cluster in ["modern_trap", "2020s_melodic_house", "cinematic", "lo_fi_chill"]:
            profile = _make_mix_profile(cluster=cluster)
            result = _gap_analysis(profile)
            assert isinstance(result["production_readiness_score"], int)
            assert 0 <= result["production_readiness_score"] <= 100

    def test_missing_roles_subset_of_blueprint(self):
        """missing_roles must be a subset of the blueprint's required roles."""
        profile = _make_mix_profile(
            cluster="modern_trap",
            roles={"kick": 0.9, "bass": 0.8},  # missing snare_clap, hats_tops
        )
        result = _gap_analysis(profile)
        bp = get_blueprint("modern_trap")
        assert bp is not None

        for role in result["missing_roles"]:
            assert role in bp.required_roles, (
                f"'{role}' reported missing but not in blueprint required_roles"
            )

    def test_chart_potential_current_lte_ceiling(self):
        """chart_potential_current must never exceed chart_potential_ceiling."""
        for cluster in all_blueprints():
            profile = _make_mix_profile(cluster=cluster)
            result = _gap_analysis(profile)
            assert result["chart_potential_current"] <= result["chart_potential_ceiling"] + 1e-9, (
                f"current ({result['chart_potential_current']}) > "
                f"ceiling ({result['chart_potential_ceiling']}) for {cluster}"
            )

    def test_full_mix_high_readiness(self):
        """A mix with all required roles present should have high readiness."""
        bp = get_blueprint("modern_trap")
        assert bp is not None
        # Give all required roles high presence
        roles = {role: 0.95 for role in bp.required_roles}
        profile = _make_mix_profile(cluster="modern_trap", roles=roles)
        result = _gap_analysis(profile)
        assert result["production_readiness_score"] >= 50
        assert len(result["missing_roles"]) == 0

    def test_empty_mix_low_readiness(self):
        """A mix with no roles present should have low readiness and many gaps."""
        profile = _make_mix_profile(cluster="modern_trap", roles={})
        result = _gap_analysis(profile)
        assert result["production_readiness_score"] < 50
        assert len(result["missing_roles"]) > 0

    def test_spectral_gaps_detected(self):
        """When spectral means deviate far from blueprint targets, gaps are reported."""
        # Use extreme spectral values
        profile = _make_mix_profile(
            cluster="modern_trap",
            spectral_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        result = _gap_analysis(profile)
        assert len(result["spectral_gaps"]) > 0

    def test_all_14_clusters_have_blueprints(self):
        """Verify all 14 genre clusters have registered blueprints."""
        blueprints = all_blueprints()
        assert len(blueprints) >= 14
        for name, bp in blueprints.items():
            assert isinstance(bp, GenreBlueprint)
            assert bp.name == name
            assert len(bp.required_roles) > 0
            assert len(bp.target_spectral) == 10
            assert len(bp.spectral_tolerance) == 10

    def test_get_best_blueprint_selects_highest_prob(self):
        """get_best_blueprint must return the blueprint with the highest probability."""
        probs = {"modern_trap": 0.3, "2020s_melodic_house": 0.6, "cinematic": 0.1}
        bp, prob = get_best_blueprint(probs)
        assert bp.name == "2020s_melodic_house"
        assert prob == pytest.approx(0.6)


# ===========================================================================
# Test 3: Explanation + gap integration
# ===========================================================================

class TestExplanationGapIntegration:
    """When gap_result is available, explanations should reference readiness."""

    def test_explanations_populated_for_all_recommendations(self):
        """No recommendation in a full batch should have an empty explanation."""
        mix_profile = _make_mix_profile()
        needs = _make_needs()
        mix_profile.needs = needs

        library = _make_sample_library()
        store = _mock_sample_store(library)
        generator = CandidateGenerator(sample_store=store, vector_index=None)
        candidates = generator.generate(mix_profile, needs, max_candidates=50)

        pref_server = _mock_preference_server()
        reranker = Reranker(corpus=_mock_corpus(), preference_server=pref_server)
        recommendations = reranker.rerank(candidates, mix_profile, needs)

        # Reranker already adds basic explanations; ExplanationEngine enriches them
        explainer = ExplanationEngine()
        explainer.explain_batch(recommendations, mix_profile, needs)

        for rec in recommendations:
            assert rec.explanation != "", (
                f"Recommendation for {rec.filepath} has empty explanation"
            )
            assert len(rec.explanation) > 5, (
                f"Explanation too short: '{rec.explanation}'"
            )

    def test_fill_missing_role_explanation_mentions_role(self):
        """fill_missing_role explanations should mention the missing role."""
        mix_profile = _make_mix_profile()
        needs = [
            NeedOpportunity(
                category="role",
                description="Missing snare — no backbeat in the mix",
                severity=0.9,
                recommendation_policy="fill_missing_role",
            ),
        ]

        rec = Recommendation(
            filepath="/samples/snare_01.wav",
            filename="snare_01.wav",
            score=0.8,
            breakdown=ScoringBreakdown(need_fit=0.9, role_fit=0.85),
            policy="fill_missing_role",
            role="snare",
        )

        explainer = ExplanationEngine()
        explanation = explainer.explain(rec, mix_profile, need=needs[0])
        assert explanation != ""
        # Should reference "snare" or "missing"
        lower = explanation.lower()
        assert "snare" in lower or "missing" in lower, (
            f"Explanation doesn't reference the missing role: '{explanation}'"
        )

    def test_explanation_batch_returns_same_list(self):
        """explain_batch should return the same list object (mutations are in-place)."""
        mix_profile = _make_mix_profile()
        needs = _make_needs()

        recs = [
            Recommendation(
                filepath="/samples/snare_01.wav",
                filename="snare_01.wav",
                score=0.7,
                breakdown=ScoringBreakdown(need_fit=0.8),
                policy="fill_missing_role",
                role="snare",
            ),
            Recommendation(
                filepath="/samples/pad_01.wav",
                filename="pad_01.wav",
                score=0.5,
                breakdown=ScoringBreakdown(spectral_complement=0.7),
                policy="reduce_emptiness",
                role="pad",
            ),
        ]

        explainer = ExplanationEngine()
        result = explainer.explain_batch(recs, mix_profile, needs)
        assert result is recs


# ===========================================================================
# Test 4: Recommendation quality assertions
# ===========================================================================

class TestRecommendationQuality:
    """Verify recommendation output meets quality constraints."""

    def _run_pipeline(
        self,
        needs: list[NeedOpportunity] | None = None,
        library: list[SampleProfile] | None = None,
    ) -> list[Recommendation]:
        """Run candidate generation -> reranking -> explanation and return results."""
        mix_profile = _make_mix_profile()
        if needs is None:
            needs = _make_needs()
        mix_profile.needs = needs

        if library is None:
            library = _make_sample_library()
        store = _mock_sample_store(library)
        generator = CandidateGenerator(sample_store=store, vector_index=None)
        candidates = generator.generate(mix_profile, needs, max_candidates=50)

        pref_server = _mock_preference_server()
        reranker = Reranker(corpus=_mock_corpus(), preference_server=pref_server)
        recommendations = reranker.rerank(candidates, mix_profile, needs)

        ExplanationEngine().explain_batch(recommendations, mix_profile, needs)
        return recommendations

    def test_sorted_by_score_descending(self):
        """Recommendations must be sorted by score in descending order."""
        recs = self._run_pipeline()
        assert len(recs) > 0
        scores = [r.score for r in recs]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Scores not descending: {scores[i]} < {scores[i + 1]} at index {i}"
            )

    def test_each_recommendation_has_explanation(self):
        """Every recommendation must have a non-empty explanation string."""
        recs = self._run_pipeline()
        for rec in recs:
            assert rec.explanation != "", (
                f"Empty explanation for {rec.filepath}"
            )

    def test_fill_missing_role_has_need_addressed(self):
        """Recommendations with fill_missing_role policy should have need_addressed populated."""
        needs = [
            NeedOpportunity(
                category="role",
                description="Missing snare/clap — the beat has no backbeat",
                severity=0.9,
                recommendation_policy="fill_missing_role",
            ),
        ]
        recs = self._run_pipeline(needs=needs)
        fill_recs = [r for r in recs if r.policy == "fill_missing_role"]
        for rec in fill_recs:
            assert rec.need_addressed != "", (
                f"fill_missing_role recommendation for {rec.filepath} "
                f"has empty need_addressed"
            )

    def test_scores_in_zero_one_range(self):
        """All recommendation scores must be in [0, 1]."""
        recs = self._run_pipeline()
        for rec in recs:
            assert 0.0 <= rec.score <= 1.0, (
                f"Score {rec.score} out of [0, 1] for {rec.filepath}"
            )

    def test_scoring_breakdown_components_bounded(self):
        """Each scoring breakdown component should be in [0, 1]."""
        recs = self._run_pipeline()
        for rec in recs:
            bd = rec.breakdown
            for field_name in [
                "need_fit", "role_fit", "spectral_complement",
                "tonal_compatibility", "rhythmic_compatibility",
                "style_prior_fit", "quality_prior", "user_preference",
                "masking_penalty", "redundancy_penalty",
            ]:
                val = getattr(bd, field_name)
                assert 0.0 <= val <= 1.0, (
                    f"Breakdown {field_name}={val} out of [0, 1] "
                    f"for {rec.filepath}"
                )

    def test_diverse_roles_in_recommendations(self):
        """With varied needs, recommendations should include multiple roles."""
        recs = self._run_pipeline()
        roles = {r.role for r in recs}
        # With the default needs (snare, spectral, pad), we should see at least 2 roles
        assert len(roles) >= 2, f"Only {len(roles)} unique role(s): {roles}"


# ===========================================================================
# Test 5: Edge cases
# ===========================================================================

class TestEdgeCases:
    """Verify graceful handling of boundary conditions."""

    def test_empty_sample_store_returns_empty(self):
        """An empty library should produce zero candidates and zero recommendations."""
        mix_profile = _make_mix_profile()
        needs = _make_needs()
        mix_profile.needs = needs

        store = _mock_sample_store([])  # empty library
        generator = CandidateGenerator(sample_store=store, vector_index=None)
        candidates = generator.generate(mix_profile, needs, max_candidates=50)
        assert candidates == []

        pref_server = _mock_preference_server()
        reranker = Reranker(corpus=_mock_corpus(), preference_server=pref_server)
        recommendations = reranker.rerank(candidates, mix_profile, needs)
        assert recommendations == []

    def test_empty_needs_returns_empty(self):
        """When there are no diagnosed needs, no candidates should be generated."""
        mix_profile = _make_mix_profile()
        library = _make_sample_library()
        store = _mock_sample_store(library)
        generator = CandidateGenerator(sample_store=store, vector_index=None)
        candidates = generator.generate(mix_profile, [], max_candidates=50)
        assert candidates == []

    def test_all_roles_present_high_readiness(self):
        """Mix with all required roles should have high readiness and few gaps."""
        bp = get_blueprint("modern_trap")
        assert bp is not None
        # All required + optional roles present at high confidence
        roles = {}
        for role in bp.required_roles:
            roles[role] = 0.95
        for role in bp.optional_roles:
            roles[role] = 0.80

        profile = _make_mix_profile(cluster="modern_trap", roles=roles)
        result = _gap_analysis(profile)

        assert result["production_readiness_score"] >= 60
        assert len(result["missing_roles"]) == 0
        assert result["chart_potential_current"] > 0
        assert result["chart_potential_current"] <= result["chart_potential_ceiling"]

    def test_mix_with_nothing_low_readiness(self):
        """A mix with no roles and flat spectral means should score very low."""
        profile = _make_mix_profile(
            cluster="modern_trap",
            roles={},
            spectral_means=[0.0] * 10,
        )
        result = _gap_analysis(profile)

        assert result["production_readiness_score"] < 30
        bp = get_blueprint("modern_trap")
        assert bp is not None
        assert len(result["missing_roles"]) == len(bp.required_roles)
        assert result["chart_potential_current"] < 0.2

    def test_unknown_cluster_raises(self):
        """Gap analysis should raise ValueError for unknown cluster probabilities."""
        profile = _make_mix_profile()
        # Override with unknown cluster
        profile.style.cluster_probabilities = {"totally_unknown_genre": 0.9}
        with pytest.raises(ValueError, match="No known cluster"):
            _gap_analysis(profile)

    def test_recommendation_result_empty_is_valid(self):
        """An empty RecommendationResult should serialize cleanly."""
        result = RecommendationResult(mix_filepath="/tmp/empty.wav")
        d = result.to_dict()
        assert d["recommendations"] == []
        assert d["needs_addressed"] == []
        assert d["total_candidates_considered"] == 0

        json_str = result.to_json()
        assert isinstance(json_str, str)

    def test_single_candidate_pipeline(self):
        """Pipeline should work correctly with just one candidate."""
        mix_profile = _make_mix_profile()
        needs = [
            NeedOpportunity(
                category="role",
                description="Missing snare — no backbeat",
                severity=0.9,
                recommendation_policy="fill_missing_role",
            ),
        ]
        library = [_make_sample("/samples/snare_01.wav", "snare", 0.9, 0.8, 3000.0, 2.0)]
        store = _mock_sample_store(library)

        generator = CandidateGenerator(sample_store=store, vector_index=None)
        candidates = generator.generate(mix_profile, needs, max_candidates=50)
        assert len(candidates) == 1

        pref_server = _mock_preference_server()
        reranker = Reranker(corpus=_mock_corpus(), preference_server=pref_server)
        recs = reranker.rerank(candidates, mix_profile, needs)
        assert len(recs) == 1
        assert 0.0 <= recs[0].score <= 1.0

        ExplanationEngine().explain_batch(recs, mix_profile, needs)
        assert recs[0].explanation != ""

    def test_no_corpus_neutral_scores(self):
        """Without a reference corpus, style_prior_fit should default to neutral."""
        mix_profile = _make_mix_profile()
        needs = _make_needs()
        library = _make_sample_library()
        store = _mock_sample_store(library)

        generator = CandidateGenerator(sample_store=store, vector_index=None)
        candidates = generator.generate(mix_profile, needs, max_candidates=50)

        # No corpus, no preferences
        reranker = Reranker(corpus=None, preference_server=None)
        recs = reranker.rerank(candidates, mix_profile, needs)

        for rec in recs:
            assert rec.breakdown.style_prior_fit == 0.5, (
                "Without corpus, style_prior_fit should be 0.5 (neutral)"
            )
            assert rec.breakdown.user_preference == 0.0, (
                "Without preference server, user_preference should be 0.0"
            )
