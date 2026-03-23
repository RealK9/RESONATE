"""Tests for backend.ml.evaluation.mix_eval — mix-level evaluation metrics."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.ml.evaluation.mix_eval import (
    MixAnalysisEval,
    NeedInferenceReport,
    RoleDetectionReport,
    StyleClassificationReport,
)
from backend.ml.models.mix_profile import (
    MixProfile,
    NeedOpportunity,
    SourceRolePresence,
    StyleCluster,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mix_with_roles(roles: dict[str, float], filepath: str = "mix.wav") -> MixProfile:
    """Build a MixProfile with given source role confidences."""
    return MixProfile(
        filepath=filepath,
        source_roles=SourceRolePresence(roles=roles),
    )


# ---------------------------------------------------------------------------
# evaluate_role_detection
# ---------------------------------------------------------------------------

class TestEvaluateRoleDetection:
    def test_perfect_detection(self):
        """All ground-truth roles are detected, no false positives."""
        evaluator = MixAnalysisEval()

        items = [
            (_mix_with_roles({"kick": 0.9, "bass": 0.8, "lead": 0.7}), ["kick", "bass", "lead"]),
        ]

        report = evaluator.evaluate_role_detection(items, threshold=0.15)

        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1 == 1.0

    def test_missing_roles(self):
        """Some ground-truth roles fall below threshold -> recall < 1.0."""
        evaluator = MixAnalysisEval()

        # bass is below threshold 0.15
        items = [
            (_mix_with_roles({"kick": 0.9, "bass": 0.05, "lead": 0.8}), ["kick", "bass", "lead"]),
        ]

        report = evaluator.evaluate_role_detection(items, threshold=0.15)

        # detected: kick, lead (2 of 3 GT roles)
        assert report.recall == pytest.approx(2 / 3, abs=0.001)
        # precision: 2 detected, 2 are in GT -> 1.0
        assert report.precision == 1.0

    def test_false_positives(self):
        """Detected roles not in ground truth -> precision < 1.0."""
        evaluator = MixAnalysisEval()

        # GT only has kick, but profile also detects "pad" and "vocal_texture"
        items = [
            (
                _mix_with_roles({"kick": 0.9, "pad": 0.5, "vocal_texture": 0.4}),
                ["kick"],
            ),
        ]

        report = evaluator.evaluate_role_detection(items, threshold=0.15)

        assert report.recall == 1.0  # kick is detected
        # precision: 1 TP / (1 TP + 2 FP) = 1/3
        assert report.precision == pytest.approx(1 / 3, abs=0.001)

    def test_multiple_mixes(self):
        """Aggregate metrics across multiple mixes."""
        evaluator = MixAnalysisEval()

        items = [
            (_mix_with_roles({"kick": 0.9, "snare": 0.8}), ["kick", "snare"]),
            (_mix_with_roles({"bass": 0.7, "lead": 0.6}), ["bass", "lead"]),
        ]

        report = evaluator.evaluate_role_detection(items, threshold=0.15)

        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1 == 1.0

    def test_per_role_detail(self):
        """Per-role detail is populated correctly."""
        evaluator = MixAnalysisEval()

        items = [
            (_mix_with_roles({"kick": 0.9, "snare": 0.8}), ["kick"]),
        ]

        report = evaluator.evaluate_role_detection(items, threshold=0.15)

        assert "kick" in report.per_role_detail
        assert report.per_role_detail["kick"]["true_positive"] == 1.0
        # snare is a false positive
        assert "snare" in report.per_role_detail
        assert report.per_role_detail["snare"]["false_positive"] == 1.0

    def test_empty_items(self):
        evaluator = MixAnalysisEval()
        report = evaluator.evaluate_role_detection([], threshold=0.15)
        assert report.precision == 0.0
        assert report.recall == 0.0


# ---------------------------------------------------------------------------
# evaluate_need_inference
# ---------------------------------------------------------------------------

class TestEvaluateNeedInference:
    def test_perfect_diagnosis(self):
        """Engine diagnoses exactly the ground-truth deficiencies."""
        evaluator = MixAnalysisEval()

        mix = MixProfile(filepath="mix.wav")
        gt_deficiencies = ["spectral", "role"]

        mock_engine = MagicMock()
        mock_engine.diagnose.return_value = [
            NeedOpportunity(category="spectral", severity=0.8),
            NeedOpportunity(category="role", severity=0.6),
        ]

        report = evaluator.evaluate_need_inference(
            [(mix, gt_deficiencies)],
            engine=mock_engine,
        )

        assert report.recall == 1.0
        assert report.false_positive_rate == 0.0
        assert report.true_positives == 2
        assert report.false_negatives == 0
        assert report.false_positives == 0

    def test_missed_deficiency(self):
        """Engine misses one ground-truth deficiency -> recall < 1.0."""
        evaluator = MixAnalysisEval()

        mix = MixProfile(filepath="mix.wav")
        gt_deficiencies = ["spectral", "role", "dynamic"]

        mock_engine = MagicMock()
        mock_engine.diagnose.return_value = [
            NeedOpportunity(category="spectral", severity=0.8),
            NeedOpportunity(category="role", severity=0.6),
            # "dynamic" is missed
        ]

        report = evaluator.evaluate_need_inference(
            [(mix, gt_deficiencies)],
            engine=mock_engine,
        )

        assert report.recall == pytest.approx(2 / 3, abs=0.001)
        assert report.false_negatives == 1

    def test_false_positives_in_diagnosis(self):
        """Engine diagnoses extra categories not in GT -> FP rate > 0."""
        evaluator = MixAnalysisEval()

        mix = MixProfile(filepath="mix.wav")
        gt_deficiencies = ["spectral"]

        mock_engine = MagicMock()
        mock_engine.diagnose.return_value = [
            NeedOpportunity(category="spectral", severity=0.8),
            NeedOpportunity(category="spatial", severity=0.4),
        ]

        report = evaluator.evaluate_need_inference(
            [(mix, gt_deficiencies)],
            engine=mock_engine,
        )

        assert report.recall == 1.0
        assert report.false_positives == 1
        # FP rate = FP / (TP + FP) = 1 / 2 = 0.5
        assert report.false_positive_rate == 0.5

    def test_multiple_mixes(self):
        """Aggregate counts across multiple mixes."""
        evaluator = MixAnalysisEval()

        mixes = [
            (MixProfile(filepath="a.wav"), ["spectral"]),
            (MixProfile(filepath="b.wav"), ["role"]),
        ]

        mock_engine = MagicMock()
        mock_engine.diagnose.side_effect = [
            [NeedOpportunity(category="spectral", severity=0.8)],
            [NeedOpportunity(category="role", severity=0.6)],
        ]

        report = evaluator.evaluate_need_inference(mixes, engine=mock_engine)

        assert report.recall == 1.0
        assert report.true_positives == 2

    def test_empty_gt(self):
        """No GT deficiencies — any diagnosis is a false positive."""
        evaluator = MixAnalysisEval()

        mock_engine = MagicMock()
        mock_engine.diagnose.return_value = [
            NeedOpportunity(category="spectral", severity=0.5),
        ]

        report = evaluator.evaluate_need_inference(
            [(MixProfile(), [])],
            engine=mock_engine,
        )

        assert report.recall == 0.0
        assert report.false_positives == 1


# ---------------------------------------------------------------------------
# evaluate_style_classification
# ---------------------------------------------------------------------------

class TestEvaluateStyleClassification:
    def test_perfect_top1(self):
        """Classifier's primary cluster matches GT -> top1 = top3 = 1.0."""
        evaluator = MixAnalysisEval()

        mix = MixProfile(filepath="mix.wav")
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = StyleCluster(
            primary_cluster="modern_trap",
            cluster_probabilities={
                "modern_trap": 0.8,
                "boom_bap": 0.1,
                "lo_fi": 0.05,
                "drill": 0.03,
                "ambient": 0.02,
            },
        )

        report = evaluator.evaluate_style_classification(
            [(mix, "modern_trap")],
            classifier=mock_classifier,
        )

        assert report.top1_accuracy == 1.0
        assert report.top3_accuracy == 1.0
        assert len(report.errors) == 0

    def test_top3_hit_but_not_top1(self):
        """GT style is in top-3 but not the primary -> top1 < top3."""
        evaluator = MixAnalysisEval()

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = StyleCluster(
            primary_cluster="drill",
            cluster_probabilities={
                "drill": 0.5,
                "modern_trap": 0.3,  # GT is here — 2nd place
                "boom_bap": 0.1,
                "lo_fi": 0.05,
                "ambient": 0.05,
            },
        )

        report = evaluator.evaluate_style_classification(
            [(MixProfile(filepath="mix.wav"), "modern_trap")],
            classifier=mock_classifier,
        )

        assert report.top1_accuracy == 0.0
        assert report.top3_accuracy == 1.0
        assert len(report.errors) == 1
        assert report.errors[0][1] == "modern_trap"  # GT style in error record

    def test_complete_miss(self):
        """GT style not in top 3 at all."""
        evaluator = MixAnalysisEval()

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = StyleCluster(
            primary_cluster="drill",
            cluster_probabilities={
                "drill": 0.5,
                "boom_bap": 0.3,
                "lo_fi": 0.1,
                "modern_trap": 0.05,  # GT is 4th
                "ambient": 0.05,
            },
        )

        report = evaluator.evaluate_style_classification(
            [(MixProfile(filepath="mix.wav"), "modern_trap")],
            classifier=mock_classifier,
        )

        assert report.top1_accuracy == 0.0
        assert report.top3_accuracy == 0.0

    def test_multiple_mixes_averaged(self):
        """Accuracy is averaged across mixes."""
        evaluator = MixAnalysisEval()

        mock_classifier = MagicMock()
        # First mix: top1 hit
        # Second mix: top1 miss, top3 hit
        mock_classifier.classify.side_effect = [
            StyleCluster(
                primary_cluster="drill",
                cluster_probabilities={"drill": 0.9, "trap": 0.05, "lo_fi": 0.05},
            ),
            StyleCluster(
                primary_cluster="lo_fi",
                cluster_probabilities={"lo_fi": 0.5, "boom_bap": 0.3, "drill": 0.2},
            ),
        ]

        items = [
            (MixProfile(filepath="a.wav"), "drill"),
            (MixProfile(filepath="b.wav"), "boom_bap"),
        ]

        report = evaluator.evaluate_style_classification(items, classifier=mock_classifier)

        # 1 of 2 correct for top1
        assert report.top1_accuracy == 0.5
        # Both correct for top3
        assert report.top3_accuracy == 1.0
