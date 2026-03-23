"""Tests for backend.ml.evaluation.system_eval — system-level performance benchmarks."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.ml.evaluation.system_eval import (
    FallbackReport,
    LatencyReport,
    QuerySpeedReport,
    SystemEval,
)
from backend.ml.models.sample_profile import SampleProfile
from backend.ml.retrieval.vector_index import VectorIndex


# ---------------------------------------------------------------------------
# benchmark_analysis_latency
# ---------------------------------------------------------------------------

class TestBenchmarkAnalysisLatency:
    @patch("backend.ml.evaluation.system_eval.analyze_sample")
    def test_timing_structure(self, mock_analyze):
        """Verify the latency report has valid timing fields."""
        evaluator = SystemEval()

        mock_analyze.return_value = SampleProfile(filepath="test.wav")

        filepaths = ["a.wav", "b.wav", "c.wav"]
        report = evaluator.benchmark_analysis_latency(filepaths)

        assert report.total_samples == 3
        assert report.mean_ms > 0.0
        assert report.median_ms > 0.0
        assert report.p95_ms > 0.0
        assert report.max_ms >= report.mean_ms
        assert len(report.per_file) == 3

        # Each per_file entry is (filepath, latency_ms)
        for fp, lat in report.per_file:
            assert fp in filepaths
            assert lat >= 0.0

    @patch("backend.ml.evaluation.system_eval.analyze_sample")
    def test_empty_filepaths(self, mock_analyze):
        evaluator = SystemEval()
        report = evaluator.benchmark_analysis_latency([])
        assert report.total_samples == 0
        assert report.per_file == []

    @patch("backend.ml.evaluation.system_eval.analyze_sample")
    def test_exception_still_records_latency(self, mock_analyze):
        """Even if analyze_sample raises, latency is still recorded."""
        evaluator = SystemEval()
        mock_analyze.side_effect = RuntimeError("boom")

        report = evaluator.benchmark_analysis_latency(["bad.wav"])

        assert report.total_samples == 1
        assert report.mean_ms > 0.0
        assert len(report.per_file) == 1


# ---------------------------------------------------------------------------
# benchmark_query_speed
# ---------------------------------------------------------------------------

class TestBenchmarkQuerySpeed:
    def test_real_faiss_index(self):
        """Build a real FAISS VectorIndex with random vectors and verify QPS > 0."""
        evaluator = SystemEval()

        dim = 64
        n_vectors = 100
        rng = np.random.default_rng(42)

        index = VectorIndex(dim=dim)
        for i in range(n_vectors):
            vec = rng.standard_normal(dim).astype(np.float32)
            index.add(f"sample_{i}.wav", vec)

        # Build query vectors
        query_vectors = [rng.standard_normal(dim).astype(np.float32) for _ in range(20)]

        report = evaluator.benchmark_query_speed(index, query_vectors, k=5)

        assert report.queries_per_second > 0.0
        assert report.mean_latency_ms > 0.0
        assert report.total_queries == 20

    def test_empty_queries(self):
        evaluator = SystemEval()
        index = VectorIndex(dim=16)
        report = evaluator.benchmark_query_speed(index, [], k=5)
        assert report.total_queries == 0
        assert report.queries_per_second == 0.0

    def test_single_query(self):
        evaluator = SystemEval()
        dim = 16
        index = VectorIndex(dim=dim)
        rng = np.random.default_rng(99)
        for i in range(10):
            index.add(f"s{i}.wav", rng.standard_normal(dim).astype(np.float32))

        query = rng.standard_normal(dim).astype(np.float32)
        report = evaluator.benchmark_query_speed(index, [query], k=3)

        assert report.total_queries == 1
        assert report.queries_per_second > 0.0


# ---------------------------------------------------------------------------
# evaluate_fallback_success
# ---------------------------------------------------------------------------

class TestEvaluateFallbackSuccess:
    @patch("backend.ml.analysis.perceptual_descriptors.extract_perceptual_descriptors")
    @patch("backend.ml.analysis.transient_descriptors.extract_transient_descriptors")
    @patch("backend.ml.analysis.harmonic_descriptors.extract_harmonic_descriptors")
    @patch("backend.ml.analysis.spectral_descriptors.extract_spectral_descriptors")
    @patch("backend.ml.analysis.core_descriptors.extract_core_descriptors")
    def test_all_stages_succeed(
        self, mock_core, mock_spectral, mock_harmonic, mock_transient, mock_perceptual,
    ):
        """All extractors succeed -> overall success rate = 1.0."""
        evaluator = SystemEval()

        # All return successfully (mock default return)
        report = evaluator.evaluate_fallback_success(["a.wav", "b.wav"], skip_embeddings=True)

        assert report.overall_success_rate == 1.0
        for stage_name, stage_data in report.per_stage.items():
            assert stage_data["rate"] == 1.0
            assert stage_data["success"] == 2.0
            assert stage_data["fail"] == 0.0

    @patch("backend.ml.analysis.perceptual_descriptors.extract_perceptual_descriptors")
    @patch("backend.ml.analysis.transient_descriptors.extract_transient_descriptors")
    @patch("backend.ml.analysis.harmonic_descriptors.extract_harmonic_descriptors")
    @patch("backend.ml.analysis.spectral_descriptors.extract_spectral_descriptors")
    @patch("backend.ml.analysis.core_descriptors.extract_core_descriptors")
    def test_some_stages_fail(
        self, mock_core, mock_spectral, mock_harmonic, mock_transient, mock_perceptual,
    ):
        """Some extractors raise -> per-stage rates reflect failures."""
        evaluator = SystemEval()

        # core and spectral succeed; harmonic raises for all files
        mock_harmonic.side_effect = RuntimeError("harmonic extraction failed")
        # transient succeeds; perceptual raises
        mock_perceptual.side_effect = RuntimeError("perceptual failed")

        report = evaluator.evaluate_fallback_success(
            ["a.wav", "b.wav", "c.wav"],
            skip_embeddings=True,
        )

        # 3 files x 5 stages = 15 total attempts
        # core: 3 ok, spectral: 3 ok, harmonic: 0 ok (3 fail),
        # transient: 3 ok, perceptual: 0 ok (3 fail)
        # Total: 9 success / 15
        assert report.overall_success_rate == pytest.approx(9 / 15, abs=0.001)

        assert report.per_stage["core"]["rate"] == 1.0
        assert report.per_stage["spectral"]["rate"] == 1.0
        assert report.per_stage["harmonic"]["rate"] == 0.0
        assert report.per_stage["harmonic"]["fail"] == 3.0
        assert report.per_stage["transient"]["rate"] == 1.0
        assert report.per_stage["perceptual"]["rate"] == 0.0

    @patch("backend.ml.analysis.perceptual_descriptors.extract_perceptual_descriptors")
    @patch("backend.ml.analysis.transient_descriptors.extract_transient_descriptors")
    @patch("backend.ml.analysis.harmonic_descriptors.extract_harmonic_descriptors")
    @patch("backend.ml.analysis.spectral_descriptors.extract_spectral_descriptors")
    @patch("backend.ml.analysis.core_descriptors.extract_core_descriptors")
    def test_mixed_per_file_failures(
        self, mock_core, mock_spectral, mock_harmonic, mock_transient, mock_perceptual,
    ):
        """An extractor fails for some files but not others."""
        evaluator = SystemEval()

        def core_sometimes_fails(filepath):
            if filepath == "bad.wav":
                raise RuntimeError("core failed for bad.wav")
            return MagicMock()

        mock_core.side_effect = core_sometimes_fails

        report = evaluator.evaluate_fallback_success(
            ["good.wav", "bad.wav"],
            skip_embeddings=True,
        )

        assert report.per_stage["core"]["success"] == 1.0
        assert report.per_stage["core"]["fail"] == 1.0
        assert report.per_stage["core"]["rate"] == 0.5

    @patch("backend.ml.analysis.perceptual_descriptors.extract_perceptual_descriptors")
    @patch("backend.ml.analysis.transient_descriptors.extract_transient_descriptors")
    @patch("backend.ml.analysis.harmonic_descriptors.extract_harmonic_descriptors")
    @patch("backend.ml.analysis.spectral_descriptors.extract_spectral_descriptors")
    @patch("backend.ml.analysis.core_descriptors.extract_core_descriptors")
    def test_embeddings_stage_included(
        self, mock_core, mock_spectral, mock_harmonic, mock_transient, mock_perceptual,
    ):
        """When skip_embeddings=False and no manager -> embeddings stage fails."""
        evaluator = SystemEval()

        report = evaluator.evaluate_fallback_success(
            ["a.wav"],
            skip_embeddings=False,
            embedding_manager=None,
        )

        # Embeddings stage should be present and fail (no manager)
        assert "embeddings" in report.per_stage
        assert report.per_stage["embeddings"]["fail"] == 1.0
        assert report.per_stage["embeddings"]["rate"] == 0.0

    @patch("backend.ml.analysis.perceptual_descriptors.extract_perceptual_descriptors")
    @patch("backend.ml.analysis.transient_descriptors.extract_transient_descriptors")
    @patch("backend.ml.analysis.harmonic_descriptors.extract_harmonic_descriptors")
    @patch("backend.ml.analysis.spectral_descriptors.extract_spectral_descriptors")
    @patch("backend.ml.analysis.core_descriptors.extract_core_descriptors")
    def test_embeddings_with_manager_success(
        self, mock_core, mock_spectral, mock_harmonic, mock_transient, mock_perceptual,
    ):
        """When skip_embeddings=False and manager is provided, embeddings stage runs."""
        evaluator = SystemEval()

        mock_emb_manager = MagicMock()
        mock_emb_manager.extract_all.return_value = {"clap_general": [0.1] * 512}

        report = evaluator.evaluate_fallback_success(
            ["a.wav"],
            skip_embeddings=False,
            embedding_manager=mock_emb_manager,
        )

        assert "embeddings" in report.per_stage
        assert report.per_stage["embeddings"]["success"] == 1.0
        assert report.per_stage["embeddings"]["rate"] == 1.0

    @patch("backend.ml.analysis.perceptual_descriptors.extract_perceptual_descriptors")
    @patch("backend.ml.analysis.transient_descriptors.extract_transient_descriptors")
    @patch("backend.ml.analysis.harmonic_descriptors.extract_harmonic_descriptors")
    @patch("backend.ml.analysis.spectral_descriptors.extract_spectral_descriptors")
    @patch("backend.ml.analysis.core_descriptors.extract_core_descriptors")
    def test_skip_embeddings_excludes_stage(
        self, mock_core, mock_spectral, mock_harmonic, mock_transient, mock_perceptual,
    ):
        """When skip_embeddings=True, no 'embeddings' key in per_stage."""
        evaluator = SystemEval()

        report = evaluator.evaluate_fallback_success(["a.wav"], skip_embeddings=True)

        assert "embeddings" not in report.per_stage
