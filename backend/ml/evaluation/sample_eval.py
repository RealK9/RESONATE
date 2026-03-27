"""
Sample-level evaluation metrics for RESONATE's analysis pipeline.

Measures role classification accuracy, pitch/key estimation accuracy,
descriptor determinism (stability), and embedding retrieval coherence.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ml.classifiers.role_classifier import RoleClassifier
from ml.models.sample_profile import SampleProfile
from ml.pipeline.ingestion import analyze_sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circle-of-fifths ordering for "close match" key evaluation
# ---------------------------------------------------------------------------

_CIRCLE_OF_FIFTHS: list[str] = [
    "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F",
]

# Build lookup: key root -> position on the circle
_COF_INDEX: dict[str, int] = {k: i for i, k in enumerate(_CIRCLE_OF_FIFTHS)}


def _cof_distance(key_a: str, key_b: str) -> int:
    """Return the minimum distance on the circle of fifths between two keys.

    Keys are expected in the format ``"C major"`` or ``"A minor"``.  Only the
    root note is used for distance calculation; the mode (major/minor) is
    compared separately.
    """
    root_a = key_a.split()[0] if key_a else ""
    root_b = key_b.split()[0] if key_b else ""
    idx_a = _COF_INDEX.get(root_a)
    idx_b = _COF_INDEX.get(root_b)
    if idx_a is None or idx_b is None:
        return 99  # unknown root -> never close
    raw = abs(idx_a - idx_b)
    return min(raw, 12 - raw)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClassificationReport:
    """Results of role-classification evaluation."""

    accuracy: float
    """Fraction of items where predicted role == ground truth role."""

    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-class metrics: {role: {precision, recall, f1, support}}."""

    confusion_pairs: list[tuple[str, str, str]] = field(default_factory=list)
    """List of (true_role, predicted_role, filepath) for every misclassification."""


@dataclass
class KeyEstimationReport:
    """Results of pitch/key estimation evaluation."""

    exact_accuracy: float
    """Fraction of items where predicted key exactly matches ground truth."""

    close_accuracy: float
    """Fraction of items where predicted key is within +/-1 on the circle of fifths."""

    errors: list[tuple[str, str, str]] = field(default_factory=list)
    """List of (filepath, true_key, predicted_key) for every non-exact match."""


@dataclass
class StabilityReport:
    """Results of descriptor determinism evaluation."""

    stability_score: float
    """Fraction of numeric descriptor fields that are identical across all runs (0-1)."""

    unstable_fields: list[str] = field(default_factory=list)
    """Names of fields whose values differed between runs."""


@dataclass
class RetrievalCoherenceReport:
    """Results of embedding-based nearest-neighbor retrieval evaluation."""

    coherence_score: float
    """Fraction of queries whose nearest neighbor shares the same role."""

    per_role_coherence: dict[str, float] = field(default_factory=dict)
    """Per-role coherence scores."""

    total_queries: int = 0
    """Number of queries evaluated."""

    coherent_hits: int = 0
    """Number of queries whose nearest neighbor shares the same role."""


# ---------------------------------------------------------------------------
# SampleAnalysisEval
# ---------------------------------------------------------------------------

class SampleAnalysisEval:
    """Evaluation harness for RESONATE's sample-analysis pipeline."""

    # ------------------------------------------------------------------
    # 1. Role classification accuracy
    # ------------------------------------------------------------------

    def evaluate_role_classification(
        self,
        items: list[tuple[str, str]],
        classifier: RoleClassifier,
    ) -> ClassificationReport:
        """Evaluate role classification accuracy.

        Parameters
        ----------
        items:
            List of ``(filepath, ground_truth_role)`` pairs.
        classifier:
            A :class:`RoleClassifier` instance to evaluate.

        Returns
        -------
        ClassificationReport
            Accuracy, per-class precision/recall/F1, and confusion pairs.
        """
        true_labels: list[str] = []
        pred_labels: list[str] = []
        confusion_pairs: list[tuple[str, str, str]] = []

        for filepath, ground_truth in items:
            predicted_role, _confidence = classifier.classify(filepath)
            true_labels.append(ground_truth)
            pred_labels.append(predicted_role)
            if predicted_role != ground_truth:
                confusion_pairs.append((ground_truth, predicted_role, filepath))

        # Overall accuracy
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        accuracy = correct / len(items) if items else 0.0

        # Per-class precision / recall / F1
        all_roles = sorted(set(true_labels) | set(pred_labels))
        per_class: dict[str, dict[str, float]] = {}

        for role in all_roles:
            tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == role and p == role)
            fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != role and p == role)
            fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == role and p != role)
            support = tp + fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class[role] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": float(support),
            }

        return ClassificationReport(
            accuracy=round(accuracy, 4),
            per_class=per_class,
            confusion_pairs=confusion_pairs,
        )

    # ------------------------------------------------------------------
    # 2. Pitch / key estimation accuracy
    # ------------------------------------------------------------------

    def evaluate_key_estimation(
        self,
        items: list[tuple[str, str]],
    ) -> KeyEstimationReport:
        """Evaluate pitch/key estimation accuracy.

        Each item is analyzed via the full ingestion pipeline to obtain the
        predicted key from harmonic descriptors.

        Parameters
        ----------
        items:
            List of ``(filepath, ground_truth_key)`` pairs.  Keys should be
            formatted as ``"C major"`` or ``"A minor"`` to match the output
            of :func:`backend.ml.analysis.mix_analyzer._detect_key`.

        Returns
        -------
        KeyEstimationReport
            Exact-match accuracy, close-match accuracy (circle-of-fifths
            distance <= 1), and per-item errors.
        """
        exact_matches = 0
        close_matches = 0
        errors: list[tuple[str, str, str]] = []

        for filepath, ground_truth_key in items:
            try:
                profile = analyze_sample(filepath, skip_embeddings=True)
                # The ingestion pipeline stores chroma but not a key label
                # directly on SampleProfile; fall back to mix-level detection.
                from ml.analysis.mix_analyzer import _detect_key
                import soundfile as sf

                audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
                mono = audio.mean(axis=1)
                predicted_key, _conf, _hz = _detect_key(mono, sr)
            except Exception:
                logger.warning("Key estimation failed for %s", filepath)
                predicted_key = ""

            gt_norm = ground_truth_key.strip().lower()
            pred_norm = predicted_key.strip().lower()

            if gt_norm == pred_norm:
                exact_matches += 1
                close_matches += 1
            else:
                dist = _cof_distance(ground_truth_key, predicted_key)
                if dist <= 1:
                    close_matches += 1
                errors.append((filepath, ground_truth_key, predicted_key))

        total = len(items) if items else 1
        return KeyEstimationReport(
            exact_accuracy=round(exact_matches / total, 4),
            close_accuracy=round(close_matches / total, 4),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # 3. Descriptor stability (determinism check)
    # ------------------------------------------------------------------

    def evaluate_stability(
        self,
        filepath: str,
        n_runs: int = 3,
    ) -> StabilityReport:
        """Run analysis N times and verify numeric descriptors are identical.

        Parameters
        ----------
        filepath:
            Path to the audio file to analyze repeatedly.
        n_runs:
            Number of analysis runs (default 3).

        Returns
        -------
        StabilityReport
            Stability score (fraction of fields stable across all runs)
            and the names of any unstable fields.
        """
        profiles: list[dict] = []
        for _ in range(n_runs):
            profile = analyze_sample(filepath, skip_embeddings=True)
            profiles.append(profile.to_dict())

        # Collect all leaf-level numeric fields
        numeric_fields: dict[str, list[float]] = {}
        self._collect_numeric_fields(profiles[0], prefix="", out=numeric_fields)

        # For each field, check if value is identical across all runs
        stable_count = 0
        total_count = 0
        unstable_fields: list[str] = []

        for field_name, _ in numeric_fields.items():
            values_across_runs: list[float] = []
            for prof_dict in profiles:
                field_vals: dict[str, list[float]] = {}
                self._collect_numeric_fields(prof_dict, prefix="", out=field_vals)
                if field_name in field_vals:
                    values_across_runs.extend(field_vals[field_name])

            total_count += 1
            # Check if all values are exactly equal
            if values_across_runs and all(v == values_across_runs[0] for v in values_across_runs):
                stable_count += 1
            else:
                unstable_fields.append(field_name)

        stability_score = stable_count / total_count if total_count > 0 else 1.0

        return StabilityReport(
            stability_score=round(stability_score, 4),
            unstable_fields=sorted(unstable_fields),
        )

    @staticmethod
    def _collect_numeric_fields(
        d: dict | list | float | int | str,
        prefix: str,
        out: dict[str, list[float]],
    ) -> None:
        """Recursively collect numeric leaf values from a nested dict.

        Parameters
        ----------
        d:
            The data structure to traverse.
        prefix:
            Dot-separated path prefix for field naming.
        out:
            Accumulator dict mapping field path to a list containing the value.
        """
        if isinstance(d, dict):
            for key, val in d.items():
                child_prefix = f"{prefix}.{key}" if prefix else key
                SampleAnalysisEval._collect_numeric_fields(val, child_prefix, out)
        elif isinstance(d, list):
            for i, val in enumerate(d):
                child_prefix = f"{prefix}[{i}]"
                SampleAnalysisEval._collect_numeric_fields(val, child_prefix, out)
        elif isinstance(d, (int, float)) and not isinstance(d, bool):
            out[prefix] = [d]

    # ------------------------------------------------------------------
    # 4. Embedding retrieval sanity (role coherence)
    # ------------------------------------------------------------------

    def evaluate_retrieval_coherence(
        self,
        profiles: list[SampleProfile],
        k: int = 5,
        embedding_field: str = "clap_general",
    ) -> RetrievalCoherenceReport:
        """For each sample, find k nearest neighbors and check role coherence.

        Parameters
        ----------
        profiles:
            Sample profiles with populated embeddings and role labels.
        k:
            Number of nearest neighbors to consider.
        embedding_field:
            Which embedding vector to use for similarity search.  Must be an
            attribute name on :class:`~backend.ml.models.sample_profile.Embeddings`
            (e.g. ``"clap_general"``, ``"panns_music"``, ``"ast_spectrogram"``).

        Returns
        -------
        RetrievalCoherenceReport
            Overall coherence score, per-role scores, total queries, and
            coherent hit count.
        """
        # Build embedding matrix and role list
        valid_profiles: list[tuple[np.ndarray, str]] = []
        for profile in profiles:
            embedding = getattr(profile.embeddings, embedding_field, [])
            if not embedding:
                continue
            vec = np.array(embedding, dtype=np.float64)
            if np.linalg.norm(vec) < 1e-12:
                continue
            valid_profiles.append((vec, profile.labels.role))

        if len(valid_profiles) < 2:
            return RetrievalCoherenceReport(
                coherence_score=0.0,
                per_role_coherence={},
                total_queries=0,
                coherent_hits=0,
            )

        embeddings_matrix = np.stack([vp[0] for vp in valid_profiles])
        roles = [vp[1] for vp in valid_profiles]

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normed = embeddings_matrix / norms

        # Cosine similarity matrix
        sim_matrix = normed @ normed.T

        coherent_hits = 0
        total_queries = len(valid_profiles)
        role_hits: dict[str, int] = defaultdict(int)
        role_totals: dict[str, int] = defaultdict(int)

        for i in range(len(valid_profiles)):
            query_role = roles[i]
            role_totals[query_role] += 1

            # Zero out self-similarity
            sims = sim_matrix[i].copy()
            sims[i] = -np.inf

            # Get top-k neighbor indices
            actual_k = min(k, len(valid_profiles) - 1)
            top_k_indices = np.argpartition(sims, -actual_k)[-actual_k:]

            # Check if the single nearest neighbor shares the role
            nearest_idx = top_k_indices[np.argmax(sims[top_k_indices])]
            if roles[nearest_idx] == query_role:
                coherent_hits += 1
                role_hits[query_role] += 1

        coherence_score = coherent_hits / total_queries if total_queries > 0 else 0.0

        per_role_coherence: dict[str, float] = {}
        for role in sorted(role_totals.keys()):
            total = role_totals[role]
            hits = role_hits.get(role, 0)
            per_role_coherence[role] = round(hits / total, 4) if total > 0 else 0.0

        return RetrievalCoherenceReport(
            coherence_score=round(coherence_score, 4),
            per_role_coherence=per_role_coherence,
            total_queries=total_queries,
            coherent_hits=coherent_hits,
        )
