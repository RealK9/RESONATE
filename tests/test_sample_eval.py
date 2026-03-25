"""Tests for backend.ml.evaluation.sample_eval — sample-level evaluation metrics."""
from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml.evaluation.sample_eval import (
    ClassificationReport,
    KeyEstimationReport,
    RetrievalCoherenceReport,
    SampleAnalysisEval,
    StabilityReport,
    _cof_distance,
)
from ml.models.sample_profile import (
    Embeddings,
    PredictedLabels,
    SampleProfile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(filepath: str, role: str, embedding: list[float]) -> SampleProfile:
    """Build a minimal SampleProfile with role and embedding."""
    return SampleProfile(
        filepath=filepath,
        labels=PredictedLabels(role=role),
        embeddings=Embeddings(clap_general=embedding),
    )


# ---------------------------------------------------------------------------
# ClassificationReport dataclass
# ---------------------------------------------------------------------------

class TestClassificationReport:
    def test_creation_defaults(self):
        report = ClassificationReport(accuracy=0.9)
        assert report.accuracy == 0.9
        assert report.per_class == {}
        assert report.confusion_pairs == []

    def test_creation_full(self):
        per_class = {"kick": {"precision": 1.0, "recall": 0.8, "f1": 0.89, "support": 5.0}}
        confusion = [("kick", "snare", "file1.wav")]
        report = ClassificationReport(
            accuracy=0.8,
            per_class=per_class,
            confusion_pairs=confusion,
        )
        assert report.accuracy == 0.8
        assert report.per_class["kick"]["precision"] == 1.0
        assert len(report.confusion_pairs) == 1


# ---------------------------------------------------------------------------
# evaluate_role_classification
# ---------------------------------------------------------------------------

class TestEvaluateRoleClassification:
    def test_perfect_classification(self):
        """Mocked classifier returns the correct role every time -> accuracy 1.0."""
        evaluator = SampleAnalysisEval()

        items = [
            ("kick.wav", "kick"),
            ("snare.wav", "snare"),
            ("hat.wav", "hat"),
        ]

        mock_classifier = MagicMock()
        # Return the ground-truth role for each file
        mock_classifier.classify.side_effect = [
            ("kick", 0.99),
            ("snare", 0.95),
            ("hat", 0.90),
        ]

        report = evaluator.evaluate_role_classification(items, mock_classifier)

        assert report.accuracy == 1.0
        assert len(report.confusion_pairs) == 0
        for role in ("kick", "snare", "hat"):
            assert report.per_class[role]["precision"] == 1.0
            assert report.per_class[role]["recall"] == 1.0
            assert report.per_class[role]["f1"] == 1.0

    def test_partial_misclassification(self):
        """Classifier confuses some roles -> accuracy < 1.0 and confusion pairs populated."""
        evaluator = SampleAnalysisEval()

        items = [
            ("kick1.wav", "kick"),
            ("kick2.wav", "kick"),
            ("snare1.wav", "snare"),
            ("snare2.wav", "snare"),
        ]

        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = [
            ("kick", 0.9),    # correct
            ("snare", 0.6),   # WRONG: kick predicted as snare
            ("snare", 0.8),   # correct
            ("kick", 0.5),    # WRONG: snare predicted as kick
        ]

        report = evaluator.evaluate_role_classification(items, mock_classifier)

        assert report.accuracy == 0.5
        assert len(report.confusion_pairs) == 2

        # kick: TP=1, FP=1(snare2 predicted kick), FN=1(kick2 predicted snare)
        assert report.per_class["kick"]["precision"] == 0.5
        assert report.per_class["kick"]["recall"] == 0.5

    def test_empty_items(self):
        evaluator = SampleAnalysisEval()
        mock_classifier = MagicMock()
        report = evaluator.evaluate_role_classification([], mock_classifier)
        assert report.accuracy == 0.0
        assert report.per_class == {}

    def test_all_wrong(self):
        evaluator = SampleAnalysisEval()
        items = [("a.wav", "kick"), ("b.wav", "snare")]
        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = [("snare", 0.9), ("kick", 0.9)]

        report = evaluator.evaluate_role_classification(items, mock_classifier)
        assert report.accuracy == 0.0
        assert len(report.confusion_pairs) == 2


# ---------------------------------------------------------------------------
# evaluate_key_estimation
# ---------------------------------------------------------------------------

class TestEvaluateKeyEstimation:
    @patch("soundfile.read", return_value=(np.zeros((44100, 1), dtype=np.float32), 44100))
    @patch("backend.ml.analysis.mix_analyzer._detect_key")
    @patch("backend.ml.evaluation.sample_eval.analyze_sample")
    def test_exact_matches(self, mock_analyze, mock_detect_key, mock_sf_read):
        """All predictions match ground truth exactly -> exact_accuracy = 1.0."""
        evaluator = SampleAnalysisEval()

        mock_analyze.return_value = MagicMock()

        items = [
            ("file1.wav", "C major"),
            ("file2.wav", "A minor"),
        ]
        mock_detect_key.side_effect = [
            ("C major", 0.9, 261.6),
            ("A minor", 0.85, 220.0),
        ]

        report = evaluator.evaluate_key_estimation(items)

        assert report.exact_accuracy == 1.0
        assert report.close_accuracy == 1.0
        assert len(report.errors) == 0

    @patch("soundfile.read", return_value=(np.zeros((44100, 1), dtype=np.float32), 44100))
    @patch("backend.ml.analysis.mix_analyzer._detect_key")
    @patch("backend.ml.evaluation.sample_eval.analyze_sample")
    def test_close_but_not_exact(self, mock_analyze, mock_detect_key, mock_sf_read):
        """Prediction is one step on circle of fifths -> close but not exact."""
        evaluator = SampleAnalysisEval()

        mock_analyze.return_value = MagicMock()

        # C major vs G major: distance 1 on circle of fifths -> close match
        items = [("file.wav", "C major")]
        mock_detect_key.return_value = ("G major", 0.7, 392.0)

        report = evaluator.evaluate_key_estimation(items)

        assert report.exact_accuracy == 0.0
        assert report.close_accuracy == 1.0
        assert len(report.errors) == 1

    @patch("soundfile.read", return_value=(np.zeros((44100, 1), dtype=np.float32), 44100))
    @patch("backend.ml.analysis.mix_analyzer._detect_key")
    @patch("backend.ml.evaluation.sample_eval.analyze_sample")
    def test_far_key(self, mock_analyze, mock_detect_key, mock_sf_read):
        """Prediction is far on circle of fifths -> neither exact nor close."""
        evaluator = SampleAnalysisEval()

        mock_analyze.return_value = MagicMock()

        # C major vs F# major: distance 6 on circle of fifths
        items = [("file.wav", "C major")]
        mock_detect_key.return_value = ("F# major", 0.5, 370.0)

        report = evaluator.evaluate_key_estimation(items)

        assert report.exact_accuracy == 0.0
        assert report.close_accuracy == 0.0
        assert len(report.errors) == 1

    @patch("backend.ml.evaluation.sample_eval.analyze_sample")
    def test_exception_handled(self, mock_analyze):
        """If analysis raises, key defaults to empty -> not exact, not close."""
        evaluator = SampleAnalysisEval()
        mock_analyze.side_effect = RuntimeError("boom")

        items = [("file.wav", "C major")]
        report = evaluator.evaluate_key_estimation(items)

        assert report.exact_accuracy == 0.0
        assert len(report.errors) == 1


# ---------------------------------------------------------------------------
# _cof_distance helper
# ---------------------------------------------------------------------------

class TestCofDistance:
    def test_same_key(self):
        assert _cof_distance("C major", "C major") == 0

    def test_adjacent(self):
        assert _cof_distance("C major", "G major") == 1
        assert _cof_distance("C major", "F major") == 1

    def test_opposite(self):
        # C to F# = 6 steps
        assert _cof_distance("C major", "F# major") == 6

    def test_unknown_root(self):
        assert _cof_distance("Cb major", "C major") == 99
        assert _cof_distance("", "C major") == 99


# ---------------------------------------------------------------------------
# evaluate_stability
# ---------------------------------------------------------------------------

class TestEvaluateStability:
    @patch("backend.ml.evaluation.sample_eval.analyze_sample")
    def test_deterministic_analysis(self, mock_analyze):
        """When analyze_sample returns identical results, stability = 1.0."""
        evaluator = SampleAnalysisEval()

        deterministic_profile = SampleProfile(
            filepath="test.wav",
            core=MagicMock(
                duration=1.0, sample_rate=44100, channels=1,
                rms=0.3, lufs=-14.0, peak=0.8, crest_factor=2.5,
                attack_time=0.01, decay_time=0.1, sustain_level=0.5,
            ),
        )
        # to_dict must return the same dict every call
        fixed_dict = {
            "filepath": "test.wav",
            "core": {"duration": 1.0, "rms": 0.3, "peak": 0.8},
            "spectral": {"centroid": 2000.0, "flatness": 0.5},
        }
        deterministic_profile.to_dict = MagicMock(return_value=fixed_dict)
        mock_analyze.return_value = deterministic_profile

        report = evaluator.evaluate_stability("test.wav", n_runs=3)

        assert report.stability_score == 1.0
        assert report.unstable_fields == []

    @patch("backend.ml.evaluation.sample_eval.analyze_sample")
    def test_nondeterministic_analysis(self, mock_analyze):
        """When one field varies across runs, stability < 1.0."""
        evaluator = SampleAnalysisEval()

        call_count = [0]

        def make_profile(*args, **kwargs):
            call_count[0] += 1
            profile = MagicMock()
            # vary the 'rms' field across runs
            profile.to_dict.return_value = {
                "core": {"duration": 1.0, "rms": 0.3 + call_count[0] * 0.01},
                "spectral": {"centroid": 2000.0},
            }
            return profile

        mock_analyze.side_effect = make_profile

        report = evaluator.evaluate_stability("test.wav", n_runs=3)

        assert report.stability_score < 1.0
        assert "core.rms" in report.unstable_fields


# ---------------------------------------------------------------------------
# evaluate_retrieval_coherence
# ---------------------------------------------------------------------------

class TestEvaluateRetrievalCoherence:
    def test_perfect_clustering(self):
        """Same-role profiles have identical embeddings -> coherence = 1.0."""
        evaluator = SampleAnalysisEval()

        # Two clusters: kick (embedding near [1,0,...]) and hat (near [0,1,...])
        dim = 16
        kick_emb = np.zeros(dim)
        kick_emb[0] = 1.0
        hat_emb = np.zeros(dim)
        hat_emb[1] = 1.0

        profiles = [
            _make_profile("kick1.wav", "kick", (kick_emb + np.random.default_rng(1).normal(0, 0.01, dim)).tolist()),
            _make_profile("kick2.wav", "kick", (kick_emb + np.random.default_rng(2).normal(0, 0.01, dim)).tolist()),
            _make_profile("kick3.wav", "kick", (kick_emb + np.random.default_rng(3).normal(0, 0.01, dim)).tolist()),
            _make_profile("hat1.wav", "hat", (hat_emb + np.random.default_rng(4).normal(0, 0.01, dim)).tolist()),
            _make_profile("hat2.wav", "hat", (hat_emb + np.random.default_rng(5).normal(0, 0.01, dim)).tolist()),
            _make_profile("hat3.wav", "hat", (hat_emb + np.random.default_rng(6).normal(0, 0.01, dim)).tolist()),
        ]

        report = evaluator.evaluate_retrieval_coherence(profiles, k=3, embedding_field="clap_general")

        assert report.coherence_score == 1.0
        assert report.total_queries == 6
        assert report.coherent_hits == 6
        assert report.per_role_coherence["kick"] == 1.0
        assert report.per_role_coherence["hat"] == 1.0

    def test_random_embeddings_lower_coherence(self):
        """Random embeddings with random roles -> coherence likely < 1.0."""
        evaluator = SampleAnalysisEval()
        rng = np.random.default_rng(42)
        dim = 16
        roles = ["kick", "snare", "hat", "bass"]

        profiles = []
        for i in range(20):
            role = roles[i % len(roles)]
            emb = rng.normal(0, 1, dim).tolist()
            profiles.append(_make_profile(f"sample_{i}.wav", role, emb))

        report = evaluator.evaluate_retrieval_coherence(profiles, k=3, embedding_field="clap_general")

        # With random embeddings, perfect coherence is highly unlikely
        assert report.total_queries == 20
        # Just verify it's a valid number
        assert 0.0 <= report.coherence_score <= 1.0

    def test_too_few_profiles(self):
        """Fewer than 2 valid profiles -> zero coherence report."""
        evaluator = SampleAnalysisEval()
        profiles = [_make_profile("only.wav", "kick", [1.0, 0.0, 0.0])]

        report = evaluator.evaluate_retrieval_coherence(profiles, k=3, embedding_field="clap_general")

        assert report.coherence_score == 0.0
        assert report.total_queries == 0

    def test_empty_embedding_skipped(self):
        """Profiles with empty embeddings are skipped."""
        evaluator = SampleAnalysisEval()
        profiles = [
            _make_profile("a.wav", "kick", []),
            _make_profile("b.wav", "hat", []),
        ]

        report = evaluator.evaluate_retrieval_coherence(profiles, k=3, embedding_field="clap_general")

        assert report.coherence_score == 0.0
        assert report.total_queries == 0
