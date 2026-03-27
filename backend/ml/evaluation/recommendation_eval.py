"""
Recommendation-quality evaluation metrics for RESONATE.

Measures precision@k, accepted recommendation rate, preference win rate,
and diversity of recommendation sets.  These metrics answer the question:
"Is the model actually useful — does it recommend things the producer keeps?"
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

from ml.models.preference import FeedbackEvent, PreferencePair
from ml.models.recommendation import Recommendation, RecommendationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PrecisionAtKReport:
    """Precision@k results averaged and per-query."""

    per_k: dict[int, float] = field(default_factory=dict)
    """Averaged precision for each k value (e.g. {1: 0.80, 3: 0.65, ...})."""

    per_query_detail: list[dict] = field(default_factory=list)
    """Per-query breakdown: [{mix_filepath, per_k: {k: precision}}, ...]."""


@dataclass
class AcceptanceReport:
    """Fraction of recommendations that the user actually kept/used."""

    acceptance_rate: float = 0.0
    """Overall rate: total_accepted / total_recommended."""

    total_recommended: int = 0
    total_accepted: int = 0

    per_role_rate: dict[str, float] = field(default_factory=dict)
    """Acceptance rate broken down by sample role."""


@dataclass
class WinRateReport:
    """How often the model's higher-scored item matches the user's preference."""

    win_rate: float = 0.0
    """Fraction of pairs where model agrees with user preference."""

    total_pairs: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0


@dataclass
class DiversityReport:
    """Diversity of a recommendation set — role entropy and spectral spread."""

    role_entropy: float = 0.0
    """Shannon entropy over role distribution in top-k recommendations."""

    spectral_spread: float = 0.0
    """Standard deviation of spectral centroids across top-k recommendations."""

    mean_diversity_score: float = 0.0
    """Combined diversity metric (average of normalized entropy and spread)."""


# ---------------------------------------------------------------------------
# RecommendationEval
# ---------------------------------------------------------------------------


class RecommendationEval:
    """Evaluation harness for RESONATE's recommendation pipeline."""

    # ------------------------------------------------------------------
    # 1. Precision@k
    # ------------------------------------------------------------------

    def evaluate_precision_at_k(
        self,
        results: list[RecommendationResult],
        ground_truth: dict[str, set[str]],
        k_values: tuple[int, ...] = (1, 3, 5, 10),
    ) -> PrecisionAtKReport:
        """Compute precision@k for a set of recommendation results.

        Parameters
        ----------
        results:
            List of :class:`RecommendationResult` objects, one per mix.
        ground_truth:
            Mapping from ``mix_filepath`` to the set of sample filepaths
            that were actually accepted/used in that mix.
        k_values:
            The k values at which to evaluate precision (default 1, 3, 5, 10).

        Returns
        -------
        PrecisionAtKReport
            Per-k averaged precision and per-query detail.
        """
        per_k_accum: dict[int, list[float]] = {k: [] for k in k_values}
        per_query_detail: list[dict] = []

        for result in results:
            accepted = ground_truth.get(result.mix_filepath, set())
            ranked_filepaths = [r.filepath for r in result.recommendations]

            query_scores: dict[int, float] = {}
            for k in k_values:
                top_k = ranked_filepaths[:k]
                if not top_k:
                    precision = 0.0
                else:
                    hits = sum(1 for fp in top_k if fp in accepted)
                    precision = hits / len(top_k)
                query_scores[k] = round(precision, 4)
                per_k_accum[k].append(precision)

            per_query_detail.append({
                "mix_filepath": result.mix_filepath,
                "per_k": query_scores,
            })

        averaged: dict[int, float] = {}
        for k in k_values:
            values = per_k_accum[k]
            averaged[k] = round(sum(values) / len(values), 4) if values else 0.0

        return PrecisionAtKReport(
            per_k=averaged,
            per_query_detail=per_query_detail,
        )

    # ------------------------------------------------------------------
    # 2. Accepted recommendation rate
    # ------------------------------------------------------------------

    def evaluate_acceptance_rate(
        self,
        results: list[RecommendationResult],
        feedback_events: list[FeedbackEvent],
        positive_actions: frozenset[str] = frozenset({"keep", "drag", "rate"}),
    ) -> AcceptanceReport:
        """Compute the fraction of recommendations the user accepted.

        A recommendation is considered "accepted" when a feedback event with
        a positive action (keep, drag, or a rating >= 3) exists for the same
        sample_filepath and mix_filepath.

        Parameters
        ----------
        results:
            Recommendation results to evaluate.
        feedback_events:
            Raw feedback events from :class:`PreferenceDataset`.
        positive_actions:
            Set of action strings considered positive.

        Returns
        -------
        AcceptanceReport
            Overall and per-role acceptance rates.
        """
        # Index feedback into a set of (mix_filepath, sample_filepath) that
        # are positive.
        accepted_pairs: set[tuple[str, str]] = set()
        for ev in feedback_events:
            if ev.action in positive_actions:
                if ev.action == "rate" and (ev.rating is None or ev.rating < 3):
                    continue
                accepted_pairs.add((ev.mix_filepath, ev.sample_filepath))

        total_recommended = 0
        total_accepted = 0
        role_recommended: dict[str, int] = defaultdict(int)
        role_accepted: dict[str, int] = defaultdict(int)

        for result in results:
            for rec in result.recommendations:
                total_recommended += 1
                role = rec.role or "unknown"
                role_recommended[role] += 1
                if (result.mix_filepath, rec.filepath) in accepted_pairs:
                    total_accepted += 1
                    role_accepted[role] += 1

        overall_rate = (
            total_accepted / total_recommended if total_recommended > 0 else 0.0
        )

        per_role_rate: dict[str, float] = {}
        for role in sorted(role_recommended.keys()):
            recommended = role_recommended[role]
            accepted = role_accepted.get(role, 0)
            per_role_rate[role] = (
                round(accepted / recommended, 4) if recommended > 0 else 0.0
            )

        return AcceptanceReport(
            acceptance_rate=round(overall_rate, 4),
            total_recommended=total_recommended,
            total_accepted=total_accepted,
            per_role_rate=per_role_rate,
        )

    # ------------------------------------------------------------------
    # 3. Preference win rate
    # ------------------------------------------------------------------

    def evaluate_preference_win_rate(
        self,
        pairs: list[PreferencePair],
        model_scores: dict[str, float],
        score_tie_threshold: float = 1e-6,
    ) -> WinRateReport:
        """Compute how often the model's ranking agrees with user preference.

        For each preference pair (preferred, rejected), the model "wins" if it
        assigned a strictly higher score to the preferred item, "ties" if the
        scores are within ``score_tie_threshold``, and "loses" otherwise.

        Parameters
        ----------
        pairs:
            Preference pairs from :class:`PreferenceDataset`.
        model_scores:
            Mapping from ``sample_filepath`` to the model's predicted score.
        score_tie_threshold:
            Absolute score difference below which scores are considered tied.

        Returns
        -------
        WinRateReport
            Win rate, total pairs evaluated, wins, ties, losses.
        """
        wins = 0
        ties = 0
        losses = 0
        evaluated = 0

        for pair in pairs:
            score_pref = model_scores.get(pair.preferred_filepath)
            score_rej = model_scores.get(pair.rejected_filepath)
            if score_pref is None or score_rej is None:
                continue

            evaluated += 1
            diff = score_pref - score_rej
            if abs(diff) < score_tie_threshold:
                ties += 1
            elif diff > 0:
                wins += 1
            else:
                losses += 1

        win_rate = wins / evaluated if evaluated > 0 else 0.0

        return WinRateReport(
            win_rate=round(win_rate, 4),
            total_pairs=evaluated,
            wins=wins,
            ties=ties,
            losses=losses,
        )

    # ------------------------------------------------------------------
    # 4. Diversity score
    # ------------------------------------------------------------------

    def evaluate_diversity(
        self,
        results: list[RecommendationResult],
        spectral_centroids: dict[str, float] | None = None,
        k: int = 10,
    ) -> DiversityReport:
        """Measure role diversity and spectral diversity across top-k recs.

        Role diversity uses Shannon entropy over the role distribution of the
        top-k recommendations (averaged across all queries).  Spectral
        diversity uses the standard deviation of spectral centroids of the
        top-k recommendations (averaged across queries), if centroids are
        provided.

        Parameters
        ----------
        results:
            Recommendation results to evaluate.
        spectral_centroids:
            Optional mapping from ``sample_filepath`` to spectral centroid
            (Hz).  Used to compute spectral spread.  If ``None``, spectral
            spread is reported as 0.
        k:
            Number of top recommendations to consider per query.

        Returns
        -------
        DiversityReport
            Role entropy, spectral spread, and combined mean diversity score.
        """
        entropy_scores: list[float] = []
        spread_scores: list[float] = []

        for result in results:
            top_k = result.recommendations[:k]
            if not top_k:
                continue

            # --- Role entropy ---
            role_counts: dict[str, int] = defaultdict(int)
            for rec in top_k:
                role_counts[rec.role or "unknown"] += 1

            total = len(top_k)
            entropy = 0.0
            for count in role_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
            entropy_scores.append(entropy)

            # --- Spectral spread ---
            if spectral_centroids is not None:
                centroids = [
                    spectral_centroids[rec.filepath]
                    for rec in top_k
                    if rec.filepath in spectral_centroids
                ]
                if len(centroids) >= 2:
                    mean_c = sum(centroids) / len(centroids)
                    variance = sum((c - mean_c) ** 2 for c in centroids) / len(centroids)
                    spread_scores.append(math.sqrt(variance))

        mean_entropy = (
            sum(entropy_scores) / len(entropy_scores) if entropy_scores else 0.0
        )
        mean_spread = (
            sum(spread_scores) / len(spread_scores) if spread_scores else 0.0
        )

        # Combined score: normalized entropy (assume max ~3.32 for 10 roles)
        # plus normalized spread (assume max ~5000 Hz centroid range).
        max_entropy = math.log2(10)  # ~3.32 bits for 10 possible roles
        max_spread = 5000.0  # approximate upper bound for centroid spread (Hz)
        norm_entropy = min(mean_entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0
        norm_spread = min(mean_spread / max_spread, 1.0) if max_spread > 0 else 0.0

        components = [norm_entropy]
        if spread_scores:
            components.append(norm_spread)
        mean_diversity = sum(components) / len(components) if components else 0.0

        return DiversityReport(
            role_entropy=round(mean_entropy, 4),
            spectral_spread=round(mean_spread, 4),
            mean_diversity_score=round(mean_diversity, 4),
        )
