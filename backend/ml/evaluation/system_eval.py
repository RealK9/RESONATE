"""
System-level performance benchmarks for RESONATE.

Measures analysis latency, ingestion throughput, vector query speed, and
fallback success rates across the pipeline stages.  All timing uses
``time.perf_counter()`` for sub-millisecond accuracy.
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.ml.pipeline.batch_processor import BatchProcessor
from backend.ml.pipeline.ingestion import analyze_sample
from backend.ml.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LatencyReport:
    """Per-sample analysis latency statistics."""

    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    total_samples: int = 0

    per_file: list[tuple[str, float]] = field(default_factory=list)
    """List of (filepath, latency_ms) for every sample analyzed."""


@dataclass
class ThroughputReport:
    """Batch ingestion throughput."""

    samples_per_second: float = 0.0
    total_samples: int = 0
    total_seconds: float = 0.0


@dataclass
class QuerySpeedReport:
    """Vector index query performance."""

    queries_per_second: float = 0.0
    mean_latency_ms: float = 0.0
    total_queries: int = 0


@dataclass
class FallbackReport:
    """Per-stage success/failure rates across a set of files."""

    overall_success_rate: float = 0.0
    """Fraction of (file, stage) pairs that succeeded."""

    per_stage: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-stage breakdown: {stage_name: {success, fail, rate}}."""


# ---------------------------------------------------------------------------
# Analysis stage names — match the try/except blocks in ingestion.py
# ---------------------------------------------------------------------------

_ANALYSIS_STAGES: list[str] = [
    "core",
    "spectral",
    "harmonic",
    "transient",
    "perceptual",
    "embeddings",
]


# ---------------------------------------------------------------------------
# SystemEval
# ---------------------------------------------------------------------------


class SystemEval:
    """Benchmark harness for RESONATE's system-level performance."""

    # ------------------------------------------------------------------
    # 1. Analysis latency
    # ------------------------------------------------------------------

    def benchmark_analysis_latency(
        self,
        filepaths: list[str],
        skip_embeddings: bool = True,
        embedding_manager: object | None = None,
    ) -> LatencyReport:
        """Time ``analyze_sample()`` on each file and report latency stats.

        Parameters
        ----------
        filepaths:
            List of audio file paths to analyze.
        skip_embeddings:
            Whether to skip embedding extraction (default ``True`` for
            faster benchmarking).
        embedding_manager:
            Optional :class:`EmbeddingManager` to pass through when
            ``skip_embeddings`` is ``False``.

        Returns
        -------
        LatencyReport
            Mean, median, p95, max latency in milliseconds, and per-file
            detail.
        """
        latencies_ms: list[float] = []
        per_file: list[tuple[str, float]] = []

        for fp in filepaths:
            t0 = time.perf_counter()
            try:
                analyze_sample(
                    fp,
                    skip_embeddings=skip_embeddings,
                    embedding_manager=embedding_manager,
                )
            except Exception:
                logger.warning("Analysis raised for %s; still recording latency", fp)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            per_file.append((fp, round(elapsed_ms, 3)))

        if not latencies_ms:
            return LatencyReport(total_samples=0, per_file=[])

        sorted_latencies = sorted(latencies_ms)
        p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)

        return LatencyReport(
            mean_ms=round(statistics.mean(latencies_ms), 3),
            median_ms=round(statistics.median(latencies_ms), 3),
            p95_ms=round(sorted_latencies[p95_index], 3),
            max_ms=round(max(latencies_ms), 3),
            total_samples=len(latencies_ms),
            per_file=per_file,
        )

    # ------------------------------------------------------------------
    # 2. Ingestion throughput
    # ------------------------------------------------------------------

    def benchmark_ingestion_throughput(
        self,
        directory: str,
        skip_embeddings: bool = True,
        embedding_manager: object | None = None,
        max_workers: int = 4,
    ) -> ThroughputReport:
        """Time :class:`BatchProcessor` on a directory and report throughput.

        Parameters
        ----------
        directory:
            Path to a directory containing audio files.
        skip_embeddings:
            Whether to skip embedding extraction.
        embedding_manager:
            Optional :class:`EmbeddingManager` instance.
        max_workers:
            Thread pool size for the batch processor.

        Returns
        -------
        ThroughputReport
            Samples per second, total samples, and total wall-clock time.
        """
        processor = BatchProcessor(
            skip_embeddings=skip_embeddings,
            embedding_manager=embedding_manager,
            db_path=None,
            max_workers=max_workers,
        )

        # Discover files first so we can report count even if processing fails.
        files = processor.discover_audio_files(directory)
        total = len(files)

        t0 = time.perf_counter()
        try:
            result = processor.process_directory(directory, source="benchmark")
            total = result.get("total", total)
        except Exception:
            logger.exception("Batch processing raised during throughput benchmark")
        t1 = time.perf_counter()

        elapsed = t1 - t0
        sps = total / elapsed if elapsed > 0 else 0.0

        return ThroughputReport(
            samples_per_second=round(sps, 3),
            total_samples=total,
            total_seconds=round(elapsed, 3),
        )

    # ------------------------------------------------------------------
    # 3. Index query speed
    # ------------------------------------------------------------------

    def benchmark_query_speed(
        self,
        index: VectorIndex,
        query_vectors: list[np.ndarray],
        k: int = 10,
    ) -> QuerySpeedReport:
        """Time a set of nearest-neighbor queries against a VectorIndex.

        Parameters
        ----------
        index:
            A populated :class:`VectorIndex` to query against.
        query_vectors:
            List of query embedding vectors (each a 1-D numpy array).
        k:
            Number of nearest neighbors to retrieve per query.

        Returns
        -------
        QuerySpeedReport
            Queries per second, mean latency, and total queries executed.
        """
        latencies_ms: list[float] = []

        for q_vec in query_vectors:
            t0 = time.perf_counter()
            try:
                index.search(q_vec, k=k)
            except Exception:
                logger.warning("Query raised; still recording latency")
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        if not latencies_ms:
            return QuerySpeedReport(total_queries=0)

        total_time_s = sum(latencies_ms) / 1000.0
        qps = len(latencies_ms) / total_time_s if total_time_s > 0 else 0.0
        mean_lat = statistics.mean(latencies_ms)

        return QuerySpeedReport(
            queries_per_second=round(qps, 3),
            mean_latency_ms=round(mean_lat, 3),
            total_queries=len(latencies_ms),
        )

    # ------------------------------------------------------------------
    # 4. Fallback success rate
    # ------------------------------------------------------------------

    def evaluate_fallback_success(
        self,
        filepaths: list[str],
        skip_embeddings: bool = False,
        embedding_manager: object | None = None,
    ) -> FallbackReport:
        """Run analysis on each file and track per-stage success/failure.

        Each analysis stage (core, spectral, harmonic, transient, perceptual,
        embeddings) is attempted individually.  A stage "succeeds" if it does
        not raise an exception for a given file.

        Parameters
        ----------
        filepaths:
            Audio file paths to test.
        skip_embeddings:
            Whether to skip the embedding stage entirely.  When ``True``,
            the embeddings stage is excluded from the report.
        embedding_manager:
            Optional :class:`EmbeddingManager` to use for the embedding
            stage.

        Returns
        -------
        FallbackReport
            Overall success rate and per-stage breakdown with success count,
            failure count, and success rate.
        """
        # Lazy imports for stage extractors — avoids import errors if a
        # particular extractor has unmet dependencies in the test env.
        from backend.ml.analysis.core_descriptors import extract_core_descriptors
        from backend.ml.analysis.spectral_descriptors import extract_spectral_descriptors
        from backend.ml.analysis.harmonic_descriptors import extract_harmonic_descriptors
        from backend.ml.analysis.transient_descriptors import extract_transient_descriptors
        from backend.ml.analysis.perceptual_descriptors import extract_perceptual_descriptors

        stage_extractors: dict[str, object] = {
            "core": extract_core_descriptors,
            "spectral": extract_spectral_descriptors,
            "harmonic": extract_harmonic_descriptors,
            "transient": extract_transient_descriptors,
            "perceptual": extract_perceptual_descriptors,
        }

        stages_to_test = list(stage_extractors.keys())
        if not skip_embeddings:
            stages_to_test.append("embeddings")

        # Counters: stage -> {success, fail}
        counters: dict[str, dict[str, int]] = {
            stage: {"success": 0, "fail": 0} for stage in stages_to_test
        }

        for fp in filepaths:
            for stage_name in stages_to_test:
                if stage_name == "embeddings":
                    # Embedding stage requires an embedding_manager.
                    if embedding_manager is None:
                        counters[stage_name]["fail"] += 1
                        continue
                    try:
                        embedding_manager.extract_all(fp)
                        counters[stage_name]["success"] += 1
                    except Exception:
                        counters[stage_name]["fail"] += 1
                else:
                    extractor = stage_extractors[stage_name]
                    try:
                        extractor(fp)
                        counters[stage_name]["success"] += 1
                    except Exception:
                        counters[stage_name]["fail"] += 1

        # Build report
        total_success = 0
        total_attempts = 0
        per_stage: dict[str, dict[str, float]] = {}

        for stage_name in stages_to_test:
            s = counters[stage_name]["success"]
            f = counters[stage_name]["fail"]
            total = s + f
            rate = s / total if total > 0 else 0.0
            per_stage[stage_name] = {
                "success": float(s),
                "fail": float(f),
                "rate": round(rate, 4),
            }
            total_success += s
            total_attempts += total

        overall = total_success / total_attempts if total_attempts > 0 else 0.0

        return FallbackReport(
            overall_success_rate=round(overall, 4),
            per_stage=per_stage,
        )
