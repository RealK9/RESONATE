"""
Ranker training pipeline — learns per-user taste models from preference pairs.

No neural networks: uses count-based biases and heuristic weight adjustment.
The output is a UserTasteModel that the PreferenceServer feeds into the reranker.
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict

from ml.db.sample_store import SampleStore
from ml.models.preference import PreferencePair, UserTasteModel
from ml.training.preference_dataset import PreferenceDataset


# Maximum absolute value for a weight delta.
_MAX_DELTA = 0.05


def _normalize_bias(counts: dict[str, float]) -> dict[str, float]:
    """Normalize a dict of raw counts to the [-1, 1] range.

    The key with the highest absolute count maps to +1 or -1; others scale
    linearly.  Returns an empty dict if *counts* is empty or all zero.
    """
    if not counts:
        return {}
    max_abs = max(abs(v) for v in counts.values())
    if max_abs == 0:
        return {k: 0.0 for k in counts}
    return {k: max(-1.0, min(1.0, v / max_abs)) for k, v in counts.items()}


class RankerTrainer:
    """Train a per-user taste model from preference pairs.

    Parameters
    ----------
    dataset : PreferenceDataset
        Source of preference pairs and destination for trained models.
    sample_store : SampleStore | None
        Optional sample store for resolving filepaths to roles.  If *None*,
        the trainer can still compute ``style_bias`` (from ``context_style``
        on pairs) but ``role_bias`` will be empty.
    """

    def __init__(
        self,
        dataset: PreferenceDataset,
        sample_store: SampleStore | None = None,
    ) -> None:
        self._dataset = dataset
        self._store = sample_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        user_id: str = "default",
        min_pairs: int = 10,
    ) -> UserTasteModel | None:
        """Train a taste model from preference pairs.

        Returns *None* if there are fewer than *min_pairs* preference pairs
        available in the dataset.

        Algorithm
        ---------
        1. Load all preference pairs from the dataset.
        2. Count role preferences: for each pair, increment the preferred
           sample's role and decrement the rejected sample's role to
           produce ``role_bias``.
        3. Count style preferences: same approach using ``context_style``
           to produce ``style_bias``.
        4. Compute ``quality_threshold``: median ``commercial_readiness``
           of all preferred samples.
        5. Compute ``weight_deltas``: analyse which scoring components
           predicted preferences correctly and boost/dampen accordingly.
        6. Normalise all biases to the [-1, 1] range.
        """
        pairs = self._dataset.get_training_data(min_pairs=min_pairs)
        if not pairs:
            return None

        # Attempt to load the previous model version for incrementing.
        existing = self._dataset.load_taste_model(user_id)
        prev_version = existing.model_version if existing else 0

        role_bias = self._compute_role_bias(pairs)
        style_bias = self._compute_style_bias(pairs)
        weight_deltas = self._compute_weight_deltas(pairs)
        quality_threshold = self._compute_quality_threshold(pairs)

        model = UserTasteModel(
            user_id=user_id,
            role_bias=role_bias,
            style_bias=style_bias,
            weight_deltas=weight_deltas,
            quality_threshold=quality_threshold,
            training_pairs=len(pairs),
            model_version=prev_version + 1,
            last_trained=time.time(),
        )

        self._dataset.save_taste_model(model)
        return model

    # ------------------------------------------------------------------
    # Bias computations
    # ------------------------------------------------------------------

    def _compute_role_bias(
        self, pairs: list[PreferencePair]
    ) -> dict[str, float]:
        """Count which roles the user prefers.

        For each pair, +strength for the preferred sample's role and
        -strength for the rejected sample's role.  Requires a
        ``SampleStore`` to resolve filepaths to roles.
        """
        if self._store is None:
            return {}

        counts: dict[str, float] = defaultdict(float)
        for pair in pairs:
            pref_profile = self._store.load(pair.preferred_filepath)
            rej_profile = self._store.load(pair.rejected_filepath)
            if pref_profile:
                counts[pref_profile.labels.role] += pair.strength
            if rej_profile:
                counts[rej_profile.labels.role] -= pair.strength
        return _normalize_bias(dict(counts))

    def _compute_style_bias(
        self, pairs: list[PreferencePair]
    ) -> dict[str, float]:
        """Count which styles the user prefers.

        Uses the ``context_style`` on each preference pair.  A preference
        for a sample in a given style increments that style; a rejection
        in a given style decrements it.
        """
        counts: dict[str, float] = defaultdict(float)
        for pair in pairs:
            style = pair.context_style
            if not style:
                continue
            # Increment for preferred sample's style, decrement for rejected
            pref_profile = self._store.load(pair.preferred_filepath) if self._store else None
            rej_profile = self._store.load(pair.rejected_filepath) if self._store else None
            pref_style = pref_profile.labels.style_tags[0] if pref_profile and pref_profile.labels.style_tags else style
            rej_style = rej_profile.labels.style_tags[0] if rej_profile and rej_profile.labels.style_tags else None
            counts[pref_style] += pair.strength
            if rej_style:
                counts[rej_style] -= pair.strength
        return _normalize_bias(dict(counts))

    def _compute_weight_deltas(
        self, pairs: list[PreferencePair]
    ) -> dict[str, float]:
        """Analyse which reranker weights should be adjusted.

        Simple heuristics based on observable sample properties:

        * If the user consistently prefers higher-quality samples, boost
          ``eta`` (quality weight).
        * If the user prefers samples that fill spectral gaps, boost
          ``gamma`` (spectral complement).
        * If the user prefers rhythmically compatible samples, boost
          ``epsilon`` (rhythmic compatibility).

        Returns small deltas in the range [-MAX_DELTA, +MAX_DELTA] per
        weight key.
        """
        if self._store is None:
            return {}

        quality_wins = 0
        quality_total = 0
        spectral_wins = 0
        spectral_total = 0
        rhythmic_wins = 0
        rhythmic_total = 0

        for pair in pairs:
            pref = self._store.load(pair.preferred_filepath)
            rej = self._store.load(pair.rejected_filepath)
            if not pref or not rej:
                continue

            # Quality comparison
            pq = pref.labels.commercial_readiness
            rq = rej.labels.commercial_readiness
            if pq != rq:
                quality_total += 1
                if pq > rq:
                    quality_wins += 1

            # Spectral: use centroid spread as a rough proxy for
            # "fills a different spectral niche".
            pc = pref.spectral.centroid
            rc = rej.spectral.centroid
            if pc != rc and pc > 0 and rc > 0:
                spectral_total += 1
                # Higher centroid diversity → user prefers spectral variety
                if abs(pc - rc) > 500:
                    spectral_wins += 1

            # Rhythmic: prefer lower onset_rate difference (more compatible)
            # is hard to judge without the mix, so use a simpler proxy:
            # if preferred sample has higher onset_rate, user likes busier
            # patterns.
            por = pref.transients.onset_rate
            ror = rej.transients.onset_rate
            if por != ror and por > 0 and ror > 0:
                rhythmic_total += 1
                if por > ror:
                    rhythmic_wins += 1

        deltas: dict[str, float] = {}

        # Quality delta
        if quality_total >= 3:
            ratio = quality_wins / quality_total
            # ratio > 0.5 means user prefers quality; < 0.5 means indifferent
            deltas["eta"] = _clamp_delta((ratio - 0.5) * 2 * _MAX_DELTA)

        # Spectral delta
        if spectral_total >= 3:
            ratio = spectral_wins / spectral_total
            deltas["gamma"] = _clamp_delta((ratio - 0.5) * 2 * _MAX_DELTA)

        # Rhythmic delta
        if rhythmic_total >= 3:
            ratio = rhythmic_wins / rhythmic_total
            deltas["epsilon"] = _clamp_delta((ratio - 0.5) * 2 * _MAX_DELTA)

        return deltas

    def _compute_quality_threshold(
        self, pairs: list[PreferencePair]
    ) -> float:
        """Compute the median commercial_readiness of preferred samples.

        Falls back to 0.3 (the UserTasteModel default) if no quality
        data is available.
        """
        if self._store is None:
            return 0.3

        qualities: list[float] = []
        for pair in pairs:
            prof = self._store.load(pair.preferred_filepath)
            if prof and prof.labels.commercial_readiness > 0:
                qualities.append(prof.labels.commercial_readiness)

        if not qualities:
            return 0.3

        return max(0.0, min(1.0, statistics.median(qualities)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_delta(value: float) -> float:
    """Clamp a weight delta to [-MAX_DELTA, +MAX_DELTA]."""
    return max(-_MAX_DELTA, min(_MAX_DELTA, value))
