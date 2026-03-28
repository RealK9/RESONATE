"""
Reranker -- second stage of the recommendation pipeline.

Scores each candidate sample using a weighted multi-factor formula and
returns a sorted list of Recommendation objects with full scoring
breakdowns.

Scoring formula:
  Score = alpha * NeedFit
        + beta  * RoleFit
        + gamma * SpectralComplement
        + delta * TonalCompatibility
        + epsilon * RhythmicCompatibility
        + zeta  * StylePriorFit
        + eta   * QualityPrior
        + theta * UserPreferencePrior
        - lambda_ * MaskingPenalty
        - mu    * RedundancyPenalty

Each component returns a value in [0, 1].
"""
from __future__ import annotations

import math

from ml.models.mix_profile import MixProfile, NeedOpportunity, SpectralOccupancy
from ml.models.sample_profile import SampleProfile
from ml.models.recommendation import (
    Recommendation,
    ScoringBreakdown,
    RecommendationResult,
)
from ml.models.reference_profile import ReferenceCorpus
from ml.training.preference_serving import PreferenceServer
from ml.recommendation.candidate_generator import (
    _POLICY_TO_ROLES,
    _normalize_key,
    _root_of,
    _is_minor,
    _cof_distance,
)


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "alpha": 0.10,    # need_fit — reduced: only matters when gaps are real
    "beta": 0.08,     # role_fit — reduced: role labels alone don't make a match
    "gamma": 0.22,    # spectral_complement — boosted: the sonic fit is king
    "delta": 0.15,    # tonal_compatibility — boosted: key match matters
    "epsilon": 0.12,  # rhythmic_compatibility — boosted: groove alignment
    "zeta": 0.08,     # style_prior_fit
    "eta": 0.10,      # quality_prior
    "theta": 0.05,    # user_preference
    "lambda_": 0.05,  # masking_penalty
    "mu": 0.05,       # redundancy_penalty
}

# Spectral band boundaries in Hz (10 bands, matching SpectralOccupancy).
_BAND_EDGES_HZ = [
    0, 60, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000, 22050,
]
_BAND_CENTERS_HZ = [
    (lo + hi) / 2 for lo, hi in zip(_BAND_EDGES_HZ[:-1], _BAND_EDGES_HZ[1:])
]


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """
    Scores and ranks candidate samples against a mix using a weighted
    multi-factor formula.  Returns a descending-sorted list of
    Recommendation objects.
    """

    def __init__(
        self,
        corpus: ReferenceCorpus | None = None,
        weights: dict[str, float] | None = None,
        preference_server: PreferenceServer | None = None,
    ):
        self._corpus = corpus
        self._preference_server = preference_server
        self._weights = dict(_DEFAULT_WEIGHTS)
        # Apply learned weight deltas from the preference server.
        if preference_server is not None and preference_server.is_loaded:
            for k, delta in preference_server.get_weight_adjustments().items():
                if k in self._weights:
                    self._weights[k] = max(0.0, self._weights[k] + delta)
        if weights is not None:
            self._weights.update(weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidates: list[SampleProfile],
        mix_profile: MixProfile,
        needs: list[NeedOpportunity],
        already_selected: list[SampleProfile] | None = None,
    ) -> list[Recommendation]:
        """
        Score every candidate, sort descending by composite score, and
        return a list of Recommendation objects with full breakdowns.
        """
        if not candidates:
            return []

        already_selected = already_selected or []
        w = self._weights

        recommendations: list[Recommendation] = []
        for sample in candidates:
            bd = self._score(sample, mix_profile, needs, already_selected)

            composite = (
                w["alpha"] * bd.need_fit
                + w["beta"] * bd.role_fit
                + w["gamma"] * bd.spectral_complement
                + w["delta"] * bd.tonal_compatibility
                + w["epsilon"] * bd.rhythmic_compatibility
                + w["zeta"] * bd.style_prior_fit
                + w["eta"] * bd.quality_prior
                + w["theta"] * bd.user_preference
                - w["lambda_"] * bd.masking_penalty
                - w["mu"] * bd.redundancy_penalty
            )
            # Clamp final score to [0, 1].
            composite = max(0.0, min(1.0, composite))

            # Build the best-match need for this sample.
            best_need = self._best_matching_need(sample, needs)

            recommendations.append(
                Recommendation(
                    filepath=sample.filepath,
                    filename=sample.filename,
                    score=composite,
                    breakdown=bd,
                    explanation=self._explain(sample, bd, best_need),
                    policy=best_need.recommendation_policy if best_need else "",
                    need_addressed=best_need.description if best_need else "",
                    role=sample.labels.role,
                )
            )

        # Sort descending by score, stable.
        recommendations.sort(key=lambda r: r.score, reverse=True)
        return recommendations

    # ------------------------------------------------------------------
    # Component scorers (each returns 0-1)
    # ------------------------------------------------------------------

    def _score(
        self,
        sample: SampleProfile,
        mix: MixProfile,
        needs: list[NeedOpportunity],
        already_selected: list[SampleProfile],
    ) -> ScoringBreakdown:
        return ScoringBreakdown(
            need_fit=self._need_fit(sample, needs),
            role_fit=self._role_fit(sample),
            spectral_complement=self._spectral_complement(sample, mix),
            tonal_compatibility=self._tonal_compatibility(sample, mix),
            rhythmic_compatibility=self._rhythmic_compatibility(sample, mix),
            style_prior_fit=self._style_prior_fit(sample, mix),
            quality_prior=self._quality_prior(sample),
            user_preference=self._user_preference(sample, mix),
            masking_penalty=self._masking_penalty(sample, mix),
            redundancy_penalty=self._redundancy_penalty(sample, already_selected),
        )

    # 1. Need fit -------------------------------------------------------

    @staticmethod
    def _need_fit(sample: SampleProfile, needs: list[NeedOpportunity]) -> float:
        """
        How well the sample's role matches unaddressed needs.
        Returns the maximum severity among needs whose policy maps to a
        role set that includes the sample's role.
        """
        role = sample.labels.role
        best = 0.0
        for need in needs:
            roles_for_policy = _POLICY_TO_ROLES.get(
                need.recommendation_policy, []
            )
            if role in roles_for_policy:
                best = max(best, need.severity)
        return min(best, 1.0)

    # 2. Role fit -------------------------------------------------------

    @staticmethod
    def _role_fit(sample: SampleProfile) -> float:
        """Confidence that the sample IS the role it claims to be."""
        return max(0.0, min(1.0, sample.labels.role_confidence))

    # 3. Spectral complement -------------------------------------------

    @staticmethod
    def _spectral_complement(sample: SampleProfile, mix: MixProfile) -> float:
        """
        How well the sample fills spectral gaps in the mix.

        Compare sample's spectral centroid to bands where the mix's
        spectral occupancy is low.  If sample centroid falls in an
        under-occupied band, the score is high.
        """
        occ = mix.spectral_occupancy
        if not occ.mean_by_band or not occ.bands:
            # No spectral data -- neutral score.
            return 0.5

        centroid = sample.spectral.centroid
        if centroid <= 0:
            return 0.5

        # Determine which band the sample centroid falls in.
        band_idx = _centroid_to_band_index(centroid, len(occ.mean_by_band))
        if band_idx < 0 or band_idx >= len(occ.mean_by_band):
            return 0.5

        # Score: 1 - occupancy (lower occupancy = higher complement value).
        occupancy = max(0.0, min(1.0, occ.mean_by_band[band_idx]))
        return 1.0 - occupancy

    # 4. Tonal compatibility -------------------------------------------

    @staticmethod
    def _tonal_compatibility(sample: SampleProfile, mix: MixProfile) -> float:
        """
        Circle-of-fifths distance for tonal samples.
        Non-tonal samples always score 1.0.
        """
        if not sample.labels.tonal:
            return 1.0

        mix_key = mix.analysis.key
        if not mix_key:
            return 1.0

        # Infer sample key from chroma.
        sample_key = _infer_key_from_chroma(sample)
        if not sample_key:
            return 1.0

        mk = _normalize_key(mix_key)
        sk = _normalize_key(sample_key)
        if not mk or not sk:
            return 1.0

        root_m = _root_of(mk)
        root_s = _root_of(sk)

        # If modes differ, convert to comparable roots via relative mapping.
        if _is_minor(mk) != _is_minor(sk):
            # Use the root directly -- cross-mode comparison is still
            # meaningful on the circle of fifths.
            pass

        dist = _cof_distance(root_m, root_s)
        # Scoring: 0 steps = 1.0, 1 step = 0.8, 2+ = linear decay to 0.
        if dist == 0:
            return 1.0
        if dist == 1:
            return 0.8
        # Linear decay from 0.6 at dist=2 down to 0 at dist=6.
        return max(0.0, 1.0 - dist * 0.2)

    # 5. Rhythmic compatibility ----------------------------------------

    @staticmethod
    def _rhythmic_compatibility(sample: SampleProfile, mix: MixProfile) -> float:
        """
        Compare sample onset_rate to mix BPM expectations.
        """
        bpm = mix.analysis.bpm
        if bpm <= 0:
            return 0.5

        onset_rate = sample.transients.onset_rate
        if onset_rate <= 0:
            # No transient data -- neutral score.
            return 0.5

        # Expected onset rate: quarter notes per second = bpm / 60.
        expected = bpm / 60.0
        if expected <= 0:
            return 0.5

        # Ratio of actual to expected; score based on how close to
        # a musically meaningful multiple (1x, 2x, 0.5x, etc.).
        ratio = onset_rate / expected

        # Find nearest power-of-2 multiple.
        if ratio <= 0:
            return 0.5
        log2_ratio = math.log2(ratio)
        nearest_int = round(log2_ratio)
        deviation = abs(log2_ratio - nearest_int)

        # Score: perfect alignment = 1.0, each semitone-ish deviation
        # reduces score.  deviation of 0.5 (sqrt(2) off) maps to ~0.
        return max(0.0, 1.0 - deviation * 2.0)

    # 6. Style prior fit -----------------------------------------------

    def _style_prior_fit(self, sample: SampleProfile, mix: MixProfile) -> float:
        """
        Check if the sample's role is expected for this style.
        """
        if self._corpus is None:
            return 0.5  # No corpus -- neutral.

        cluster = mix.style.primary_cluster
        if not cluster:
            return 0.5

        prior = self._corpus.get_prior(cluster)
        if prior is None:
            return 0.5

        role = sample.labels.role

        # Check typical_roles first.
        if role in prior.typical_roles:
            return max(0.0, min(1.0, prior.typical_roles[role]))

        # Check common_complements.
        if role in prior.common_complements:
            return 0.7

        # Role not mentioned at all -- mildly penalize.
        return 0.3

    # 7. Quality prior -------------------------------------------------

    @staticmethod
    def _quality_prior(sample: SampleProfile) -> float:
        """Simply the sample's commercial readiness score."""
        return max(0.0, min(1.0, sample.labels.commercial_readiness))

    # 8. User preference -----------------------------------------------

    def _user_preference(self, sample: SampleProfile, mix: MixProfile) -> float:
        """Score based on learned user taste model via PreferenceServer."""
        if self._preference_server is None or not self._preference_server.is_loaded:
            return 0.0
        return self._preference_server.score(
            sample_filepath=sample.filepath,
            sample_role=sample.labels.role,
            context_style=mix.style.primary_cluster,
        )

    # 9. Masking penalty -----------------------------------------------

    @staticmethod
    def _masking_penalty(sample: SampleProfile, mix: MixProfile) -> float:
        """
        Penalize if the sample's spectral centroid overlaps with
        already highly-occupied bands in the mix.
        """
        occ = mix.spectral_occupancy
        if not occ.mean_by_band or not occ.bands:
            return 0.0

        centroid = sample.spectral.centroid
        if centroid <= 0:
            return 0.0

        band_idx = _centroid_to_band_index(centroid, len(occ.mean_by_band))
        if band_idx < 0 or band_idx >= len(occ.mean_by_band):
            return 0.0

        occupancy = max(0.0, min(1.0, occ.mean_by_band[band_idx]))
        # Only penalize if the band is significantly occupied.
        if occupancy < 0.5:
            return 0.0
        # Scale from 0 at 0.5 occupancy to 1.0 at 1.0 occupancy.
        return (occupancy - 0.5) * 2.0

    # 10. Redundancy penalty -------------------------------------------

    @staticmethod
    def _redundancy_penalty(
        sample: SampleProfile,
        already_selected: list[SampleProfile],
    ) -> float:
        """
        Penalize if already_selected contains samples with the same role.
        0.3 per same-role sample, capped at 1.0.
        """
        if not already_selected:
            return 0.0
        role = sample.labels.role
        count = sum(1 for s in already_selected if s.labels.role == role)
        return min(1.0, count * 0.3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_matching_need(
        sample: SampleProfile,
        needs: list[NeedOpportunity],
    ) -> NeedOpportunity | None:
        """Return the highest-severity need that the sample's role addresses."""
        role = sample.labels.role
        best: NeedOpportunity | None = None
        best_sev = -1.0
        for need in needs:
            roles_for_policy = _POLICY_TO_ROLES.get(
                need.recommendation_policy, []
            )
            if role in roles_for_policy and need.severity > best_sev:
                best = need
                best_sev = need.severity
        return best

    @staticmethod
    def _explain(
        sample: SampleProfile,
        bd: ScoringBreakdown,
        need: NeedOpportunity | None,
    ) -> str:
        """Build a human-readable explanation string."""
        parts: list[str] = []
        if need:
            parts.append(f"Addresses: {need.description}")
        parts.append(f"Role: {sample.labels.role} (confidence {bd.role_fit:.0%})")
        if bd.spectral_complement > 0.7:
            parts.append("Fills a spectral gap")
        if bd.tonal_compatibility >= 0.8:
            parts.append("Tonally compatible")
        if bd.masking_penalty > 0.3:
            parts.append("Warning: masking risk")
        if bd.redundancy_penalty > 0:
            parts.append("Redundancy with already-selected samples")
        return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _centroid_to_band_index(centroid_hz: float, n_bands: int) -> int:
    """
    Map a spectral centroid (Hz) to a band index.

    Uses the standard 10-band edges if *n_bands* is 10, otherwise falls
    back to a linear division of 0-22050 Hz.
    """
    if n_bands == len(_BAND_EDGES_HZ) - 1:
        edges = _BAND_EDGES_HZ
    else:
        step = 22050.0 / n_bands
        edges = [step * i for i in range(n_bands + 1)]

    for i in range(len(edges) - 1):
        if centroid_hz < edges[i + 1]:
            return i
    return len(edges) - 2  # Last band.


def _infer_key_from_chroma(sample: SampleProfile) -> str:
    """Infer a key string from the sample's chroma profile."""
    chroma = sample.harmonic.chroma_profile
    if not chroma or len(chroma) != 12:
        return ""
    if sample.harmonic.pitch_confidence < 0.3:
        return ""
    note_names = [
        "C", "Db", "D", "Eb", "E", "F",
        "F#", "G", "Ab", "A", "Bb", "B",
    ]
    peak_idx = max(range(12), key=lambda i: chroma[i])
    return note_names[peak_idx]
