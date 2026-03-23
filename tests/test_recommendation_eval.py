"""Tests for backend.ml.evaluation.recommendation_eval — recommendation quality metrics."""
from __future__ import annotations

import math

import pytest

from backend.ml.evaluation.recommendation_eval import (
    AcceptanceReport,
    DiversityReport,
    PrecisionAtKReport,
    RecommendationEval,
    WinRateReport,
)
from backend.ml.models.preference import FeedbackEvent, PreferencePair
from backend.ml.models.recommendation import Recommendation, RecommendationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(filepath: str, role: str = "kick", score: float = 1.0) -> Recommendation:
    return Recommendation(filepath=filepath, role=role, score=score)


def _result(mix: str, recs: list[Recommendation]) -> RecommendationResult:
    return RecommendationResult(mix_filepath=mix, recommendations=recs)


# ---------------------------------------------------------------------------
# evaluate_precision_at_k
# ---------------------------------------------------------------------------

class TestEvaluatePrecisionAtK:
    def test_perfect_precision(self):
        """All top-k recommendations are in the accepted set."""
        evaluator = RecommendationEval()

        recs = [_rec(f"s{i}.wav") for i in range(5)]
        result = _result("mix.wav", recs)
        gt = {"mix.wav": {f"s{i}.wav" for i in range(5)}}

        report = evaluator.evaluate_precision_at_k([result], gt, k_values=(1, 3, 5))

        assert report.per_k[1] == 1.0
        assert report.per_k[3] == 1.0
        assert report.per_k[5] == 1.0

    def test_partial_precision(self):
        """Only some top-k are in the accepted set."""
        evaluator = RecommendationEval()

        # Recommendations: s0, s1, s2, s3, s4
        # Accepted: s0, s2, s4  (3 out of 5)
        recs = [_rec(f"s{i}.wav") for i in range(5)]
        result = _result("mix.wav", recs)
        gt = {"mix.wav": {"s0.wav", "s2.wav", "s4.wav"}}

        report = evaluator.evaluate_precision_at_k([result], gt, k_values=(1, 3, 5))

        # P@1: s0 is accepted -> 1/1 = 1.0
        assert report.per_k[1] == 1.0
        # P@3: s0 and s2 accepted out of {s0, s1, s2} -> 2/3
        assert report.per_k[3] == pytest.approx(2 / 3, abs=0.001)
        # P@5: 3 of 5 accepted -> 0.6
        assert report.per_k[5] == 0.6

    def test_no_relevant_items(self):
        """No recommended items are in the accepted set."""
        evaluator = RecommendationEval()

        recs = [_rec(f"s{i}.wav") for i in range(5)]
        result = _result("mix.wav", recs)
        gt = {"mix.wav": {"other1.wav", "other2.wav"}}

        report = evaluator.evaluate_precision_at_k([result], gt, k_values=(1, 3, 5))

        assert report.per_k[1] == 0.0
        assert report.per_k[3] == 0.0
        assert report.per_k[5] == 0.0

    def test_multiple_queries_averaged(self):
        """Precision is averaged across multiple mixes."""
        evaluator = RecommendationEval()

        result_a = _result("a.wav", [_rec("s0.wav"), _rec("s1.wav")])
        result_b = _result("b.wav", [_rec("s2.wav"), _rec("s3.wav")])

        gt = {
            "a.wav": {"s0.wav", "s1.wav"},  # P@1=1.0
            "b.wav": {"s3.wav"},              # P@1=0.0
        }

        report = evaluator.evaluate_precision_at_k(
            [result_a, result_b], gt, k_values=(1,)
        )

        # Average of 1.0 and 0.0
        assert report.per_k[1] == 0.5

    def test_per_query_detail(self):
        evaluator = RecommendationEval()
        result = _result("mix.wav", [_rec("s0.wav")])
        gt = {"mix.wav": {"s0.wav"}}

        report = evaluator.evaluate_precision_at_k([result], gt, k_values=(1,))

        assert len(report.per_query_detail) == 1
        assert report.per_query_detail[0]["mix_filepath"] == "mix.wav"


# ---------------------------------------------------------------------------
# evaluate_acceptance_rate
# ---------------------------------------------------------------------------

class TestEvaluateAcceptanceRate:
    def test_all_accepted(self):
        """Every recommendation has a positive feedback event."""
        evaluator = RecommendationEval()

        result = _result("mix.wav", [
            _rec("s0.wav", role="kick"),
            _rec("s1.wav", role="snare"),
        ])

        feedback = [
            FeedbackEvent(mix_filepath="mix.wav", sample_filepath="s0.wav", action="keep"),
            FeedbackEvent(mix_filepath="mix.wav", sample_filepath="s1.wav", action="drag"),
        ]

        report = evaluator.evaluate_acceptance_rate([result], feedback)

        assert report.acceptance_rate == 1.0
        assert report.total_recommended == 2
        assert report.total_accepted == 2

    def test_partial_acceptance(self):
        """Only some recommendations have positive feedback."""
        evaluator = RecommendationEval()

        result = _result("mix.wav", [
            _rec("s0.wav", role="kick"),
            _rec("s1.wav", role="snare"),
            _rec("s2.wav", role="hat"),
        ])

        feedback = [
            FeedbackEvent(mix_filepath="mix.wav", sample_filepath="s0.wav", action="keep"),
            # s1 has no feedback -> not accepted
            # s2 has a discard -> not positive
            FeedbackEvent(mix_filepath="mix.wav", sample_filepath="s2.wav", action="discard"),
        ]

        report = evaluator.evaluate_acceptance_rate([result], feedback)

        assert report.acceptance_rate == pytest.approx(1 / 3, abs=0.001)
        assert report.total_accepted == 1

    def test_per_role_rate(self):
        """Per-role acceptance rates are computed correctly."""
        evaluator = RecommendationEval()

        result = _result("mix.wav", [
            _rec("s0.wav", role="kick"),
            _rec("s1.wav", role="kick"),
            _rec("s2.wav", role="snare"),
        ])

        feedback = [
            FeedbackEvent(mix_filepath="mix.wav", sample_filepath="s0.wav", action="keep"),
            # s1 not accepted
        ]

        report = evaluator.evaluate_acceptance_rate([result], feedback)

        assert report.per_role_rate["kick"] == 0.5     # 1 of 2
        assert report.per_role_rate["snare"] == 0.0     # 0 of 1

    def test_rate_action_needs_rating_ge_3(self):
        """A 'rate' action with rating < 3 is not considered positive."""
        evaluator = RecommendationEval()

        result = _result("mix.wav", [_rec("s0.wav")])

        feedback_low = [
            FeedbackEvent(
                mix_filepath="mix.wav", sample_filepath="s0.wav",
                action="rate", rating=2,
            ),
        ]
        report = evaluator.evaluate_acceptance_rate([result], feedback_low)
        assert report.total_accepted == 0

        feedback_high = [
            FeedbackEvent(
                mix_filepath="mix.wav", sample_filepath="s0.wav",
                action="rate", rating=4,
            ),
        ]
        report = evaluator.evaluate_acceptance_rate([result], feedback_high)
        assert report.total_accepted == 1

    def test_empty_feedback(self):
        evaluator = RecommendationEval()
        result = _result("mix.wav", [_rec("s0.wav")])
        report = evaluator.evaluate_acceptance_rate([result], [])
        assert report.acceptance_rate == 0.0


# ---------------------------------------------------------------------------
# evaluate_preference_win_rate
# ---------------------------------------------------------------------------

class TestEvaluatePreferenceWinRate:
    def test_all_wins(self):
        """Model scores preferred > rejected for every pair -> win_rate = 1.0."""
        evaluator = RecommendationEval()

        pairs = [
            PreferencePair(preferred_filepath="a.wav", rejected_filepath="b.wav"),
            PreferencePair(preferred_filepath="c.wav", rejected_filepath="d.wav"),
        ]
        scores = {"a.wav": 0.9, "b.wav": 0.3, "c.wav": 0.8, "d.wav": 0.2}

        report = evaluator.evaluate_preference_win_rate(pairs, scores)

        assert report.win_rate == 1.0
        assert report.wins == 2
        assert report.losses == 0
        assert report.ties == 0

    def test_all_losses(self):
        """Model scores preferred < rejected for every pair -> win_rate = 0."""
        evaluator = RecommendationEval()

        pairs = [
            PreferencePair(preferred_filepath="a.wav", rejected_filepath="b.wav"),
        ]
        scores = {"a.wav": 0.1, "b.wav": 0.9}

        report = evaluator.evaluate_preference_win_rate(pairs, scores)

        assert report.win_rate == 0.0
        assert report.losses == 1

    def test_ties(self):
        """Equal scores within threshold -> tie."""
        evaluator = RecommendationEval()

        pairs = [
            PreferencePair(preferred_filepath="a.wav", rejected_filepath="b.wav"),
        ]
        scores = {"a.wav": 0.5, "b.wav": 0.5}

        report = evaluator.evaluate_preference_win_rate(pairs, scores)

        assert report.win_rate == 0.0
        assert report.ties == 1
        assert report.wins == 0

    def test_mixed_outcomes(self):
        """Mix of wins, losses, ties."""
        evaluator = RecommendationEval()

        pairs = [
            PreferencePair(preferred_filepath="a.wav", rejected_filepath="b.wav"),  # win
            PreferencePair(preferred_filepath="c.wav", rejected_filepath="d.wav"),  # loss
            PreferencePair(preferred_filepath="e.wav", rejected_filepath="f.wav"),  # tie
        ]
        scores = {
            "a.wav": 0.9, "b.wav": 0.1,
            "c.wav": 0.2, "d.wav": 0.8,
            "e.wav": 0.5, "f.wav": 0.5,
        }

        report = evaluator.evaluate_preference_win_rate(pairs, scores)

        assert report.total_pairs == 3
        assert report.wins == 1
        assert report.losses == 1
        assert report.ties == 1
        assert report.win_rate == pytest.approx(1 / 3, abs=0.001)

    def test_missing_scores_skipped(self):
        """Pairs where a score is missing are not evaluated."""
        evaluator = RecommendationEval()

        pairs = [
            PreferencePair(preferred_filepath="a.wav", rejected_filepath="b.wav"),
            PreferencePair(preferred_filepath="c.wav", rejected_filepath="d.wav"),
        ]
        # Only a.wav and b.wav have scores
        scores = {"a.wav": 0.9, "b.wav": 0.3}

        report = evaluator.evaluate_preference_win_rate(pairs, scores)

        assert report.total_pairs == 1
        assert report.wins == 1


# ---------------------------------------------------------------------------
# evaluate_diversity
# ---------------------------------------------------------------------------

class TestEvaluateDiversity:
    def test_homogeneous_recommendations(self):
        """All recommendations have the same role -> entropy = 0."""
        evaluator = RecommendationEval()

        recs = [_rec(f"s{i}.wav", role="kick") for i in range(5)]
        result = _result("mix.wav", recs)

        report = evaluator.evaluate_diversity([result], k=5)

        assert report.role_entropy == 0.0

    def test_heterogeneous_recommendations(self):
        """All recommendations have distinct roles -> high entropy."""
        evaluator = RecommendationEval()

        roles = ["kick", "snare", "hat", "bass", "lead"]
        recs = [_rec(f"s{i}.wav", role=roles[i]) for i in range(5)]
        result = _result("mix.wav", recs)

        report = evaluator.evaluate_diversity([result], k=5)

        expected_entropy = math.log2(5)  # ~2.322
        assert report.role_entropy == pytest.approx(expected_entropy, abs=0.01)

    def test_entropy_difference(self):
        """Heterogeneous set has strictly higher entropy than homogeneous."""
        evaluator = RecommendationEval()

        homo_recs = [_rec(f"s{i}.wav", role="kick") for i in range(5)]
        homo_result = _result("homo.wav", homo_recs)

        hetero_roles = ["kick", "snare", "hat", "bass", "lead"]
        hetero_recs = [_rec(f"s{i}.wav", role=hetero_roles[i]) for i in range(5)]
        hetero_result = _result("hetero.wav", hetero_recs)

        homo_report = evaluator.evaluate_diversity([homo_result], k=5)
        hetero_report = evaluator.evaluate_diversity([hetero_result], k=5)

        assert hetero_report.role_entropy > homo_report.role_entropy
        assert hetero_report.mean_diversity_score > homo_report.mean_diversity_score

    def test_spectral_spread(self):
        """Spectral centroids are used to compute spread."""
        evaluator = RecommendationEval()

        recs = [_rec(f"s{i}.wav", role="kick") for i in range(3)]
        result = _result("mix.wav", recs)

        centroids = {"s0.wav": 200.0, "s1.wav": 1000.0, "s2.wav": 3000.0}

        report = evaluator.evaluate_diversity([result], spectral_centroids=centroids, k=3)

        assert report.spectral_spread > 0.0

    def test_empty_results(self):
        evaluator = RecommendationEval()
        report = evaluator.evaluate_diversity([], k=5)
        assert report.role_entropy == 0.0
        assert report.mean_diversity_score == 0.0
