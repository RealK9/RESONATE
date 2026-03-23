"""
Explanation engine -- Phase 4 human-readable recommendation explanations.

Maps each recommendation to a natural-language explanation based on the
decision policy that triggered it, the scoring breakdown, and the mix
context.  The goal is output that reads like a music producer giving
quick, confident advice -- not a computer spitting out numbers.
"""
from __future__ import annotations

from backend.ml.models.mix_profile import MixProfile, NeedOpportunity
from backend.ml.models.recommendation import Recommendation, ScoringBreakdown
from backend.ml.recommendation.candidate_generator import _POLICY_TO_ROLES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BAND_NAMES = [
    "sub-bass", "bass", "low-mids", "mids", "upper-mids",
    "presence", "brilliance", "air", "ultra-highs", "beyond",
]

# Primary templates keyed by decision policy.
_POLICY_TEMPLATES: dict[str, str] = {
    "fill_missing_role": "Your mix is missing {role}. This {sample_role} would fill that gap.",
    "reinforce_existing": "Your {role} could use reinforcement. This sample adds {quality} to strengthen it.",
    "improve_polish": "This {role} adds commercial polish \u2014 {specific_quality}.",
    "increase_contrast": "This creates dynamic contrast against your existing {existing_element}.",
    "add_movement": "This adds rhythmic/melodic movement to keep the mix evolving.",
    "reduce_emptiness": "Your mix has spectral gaps in the {band} range. This fills that space.",
    "support_transition": "This FX/texture element supports smooth transitions between sections.",
    "enhance_groove": "This percussion element strengthens the rhythmic foundation.",
    "enhance_lift": "This adds emotional lift and energy to build sections.",
}


# ---------------------------------------------------------------------------
# Quality descriptor helpers
# ---------------------------------------------------------------------------

def _describe_quality(sample_role: str, breakdown: ScoringBreakdown) -> str:
    """
    Return a short perceptual quality phrase based on which scoring
    components dominate for this recommendation.
    """
    # Build a scored list of (component_value, descriptor) pairs.
    candidates: list[tuple[float, str]] = [
        (breakdown.spectral_complement, "spectral fullness"),
        (breakdown.tonal_compatibility, "harmonic richness"),
        (breakdown.rhythmic_compatibility, "groove tightness"),
        (breakdown.style_prior_fit, "stylistic cohesion"),
        (breakdown.quality_prior, "polished production quality"),
    ]

    # Role-specific overrides for more musical language.
    role_lower = sample_role.lower()
    if role_lower in ("kick", "snare", "clap"):
        candidates.append((breakdown.rhythmic_compatibility, "punchy attack"))
        candidates.append((breakdown.spectral_complement, "weight and body"))
    elif role_lower in ("bass",):
        candidates.append((breakdown.spectral_complement, "low-end warmth"))
        candidates.append((breakdown.tonal_compatibility, "deep tone"))
    elif role_lower in ("hat",):
        candidates.append((breakdown.spectral_complement, "crisp shimmer"))
        candidates.append((breakdown.rhythmic_compatibility, "tight groove feel"))
    elif role_lower in ("lead", "vocal"):
        candidates.append((breakdown.tonal_compatibility, "melodic character"))
        candidates.append((breakdown.spectral_complement, "bright presence"))
    elif role_lower in ("pad", "texture"):
        candidates.append((breakdown.spectral_complement, "warm texture"))
        candidates.append((breakdown.tonal_compatibility, "lush tone"))
    elif role_lower in ("fx",):
        candidates.append((breakdown.spectral_complement, "spatial depth"))
        candidates.append((breakdown.style_prior_fit, "atmospheric character"))

    # Pick the descriptor with the highest score.
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1] if candidates else "sonic character"


def _top_scoring_component(breakdown: ScoringBreakdown) -> tuple[str, float]:
    """Return the (name, value) of the highest positive scoring component."""
    components: list[tuple[str, float]] = [
        ("need_fit", breakdown.need_fit),
        ("role_fit", breakdown.role_fit),
        ("spectral_complement", breakdown.spectral_complement),
        ("tonal_compatibility", breakdown.tonal_compatibility),
        ("rhythmic_compatibility", breakdown.rhythmic_compatibility),
        ("style_prior_fit", breakdown.style_prior_fit),
        ("quality_prior", breakdown.quality_prior),
        ("user_preference", breakdown.user_preference),
    ]
    components.sort(key=lambda c: c[1], reverse=True)
    return components[0]


# ---------------------------------------------------------------------------
# Secondary detail generation
# ---------------------------------------------------------------------------

def _secondary_detail(
    breakdown: ScoringBreakdown,
    mix_profile: MixProfile,
    sample_role: str,
    primary_policy: str,
) -> str:
    """
    Generate an optional secondary sentence when the scoring breakdown
    reveals something interesting beyond the primary template.
    """
    top_name, top_val = _top_scoring_component(breakdown)

    # Only add a secondary detail if the top component is meaningfully strong
    # and is different from what the primary template already explains.
    if top_val < 0.4:
        return ""

    # Spectral complement detail -- reference the actual band gap.
    if top_name == "spectral_complement" and primary_policy != "reduce_emptiness":
        gap_band = _find_weakest_band(mix_profile)
        if gap_band:
            return f"It also brings warmth to your {gap_band} which are currently thin."

    # Tonal compatibility detail.
    if top_name == "tonal_compatibility" and primary_policy not in (
        "reinforce_existing", "improve_polish"
    ):
        return "It sits naturally in the harmonic space of your mix."

    # Rhythmic compatibility detail.
    if top_name == "rhythmic_compatibility" and primary_policy not in (
        "enhance_groove", "add_movement"
    ):
        return "Its rhythmic feel locks in with your existing groove."

    # Style prior detail.
    if top_name == "style_prior_fit" and primary_policy != "improve_polish":
        cluster = mix_profile.style.primary_cluster
        if cluster:
            return f"Stylistically, it fits right into the {cluster} palette you're building."

    return ""


def _find_weakest_band(mix_profile: MixProfile) -> str:
    """Find the band name with the lowest mean spectral occupancy."""
    means = mix_profile.spectral_occupancy.mean_by_band
    bands = mix_profile.spectral_occupancy.bands

    if not means or not bands:
        # Fall back to band names list if the profile has no data.
        return ""

    min_idx = 0
    min_val = means[0]
    for i, val in enumerate(means):
        if val < min_val:
            min_val = val
            min_idx = i

    if min_idx < len(bands):
        return bands[min_idx]
    if min_idx < len(_BAND_NAMES):
        return _BAND_NAMES[min_idx]
    return ""


def _find_gap_band(mix_profile: MixProfile) -> str:
    """Identify the spectral band with the largest gap for reduce_emptiness."""
    band = _find_weakest_band(mix_profile)
    return band if band else "mid"


def _find_existing_element(mix_profile: MixProfile, sample_role: str) -> str:
    """Find an existing role in the mix to contrast against."""
    roles = mix_profile.source_roles.roles
    if not roles:
        return "elements"

    # Pick the role with the highest presence that is different from the sample.
    best_role = ""
    best_val = -1.0
    for role_name, presence in roles.items():
        if role_name.lower() != sample_role.lower() and presence > best_val:
            best_val = presence
            best_role = role_name

    return best_role if best_role else "elements"


# ---------------------------------------------------------------------------
# Need matching
# ---------------------------------------------------------------------------

def _match_need(
    recommendation: Recommendation,
    needs: list[NeedOpportunity],
) -> NeedOpportunity | None:
    """
    Find the NeedOpportunity that best matches this recommendation.

    Strategy:
    1. Exact policy match -- if a need's recommendation_policy matches the
       recommendation's policy, prefer that.
    2. Role overlap in description -- if the recommendation's role appears
       in a need's description text, use that need.
    3. Return None if nothing matches.
    """
    if not needs:
        return None

    # Pass 1: exact policy match.
    policy_matches: list[NeedOpportunity] = []
    for need in needs:
        if need.recommendation_policy == recommendation.policy:
            policy_matches.append(need)

    if len(policy_matches) == 1:
        return policy_matches[0]

    # If multiple policy matches, prefer the one whose description mentions
    # the sample's role.
    role_lower = recommendation.role.lower()
    if policy_matches:
        for need in policy_matches:
            if role_lower and role_lower in need.description.lower():
                return need
        # Fall back to highest severity among policy matches.
        return max(policy_matches, key=lambda n: n.severity)

    # Pass 2: role overlap in description (across all needs).
    if role_lower:
        for need in needs:
            if role_lower in need.description.lower():
                return need

    return None


# ---------------------------------------------------------------------------
# Explanation engine
# ---------------------------------------------------------------------------

class ExplanationEngine:
    """
    Generates human-readable musical explanations for recommendations.

    Each explanation is 1-2 sentences that tell the producer *why* this
    sample was recommended, using language that sounds like advice from
    an experienced engineer rather than output from a scoring algorithm.
    """

    def explain(
        self,
        recommendation: Recommendation,
        mix_profile: MixProfile,
        need: NeedOpportunity | None = None,
    ) -> str:
        """
        Generate a 1-2 sentence explanation for a single recommendation.

        Parameters
        ----------
        recommendation : Recommendation
            The scored recommendation to explain.
        mix_profile : MixProfile
            The current mix context.
        need : NeedOpportunity | None
            The specific need being addressed, if known.

        Returns
        -------
        str
            A human-readable explanation string.
        """
        policy = recommendation.policy
        role = recommendation.role
        breakdown = recommendation.breakdown

        # --- Build the primary sentence from the policy template ----------
        template = _POLICY_TEMPLATES.get(policy, "")
        primary = self._fill_template(
            template, policy, role, breakdown, mix_profile, need
        )

        # If no template matched, craft a generic explanation.
        if not primary:
            quality = _describe_quality(role, breakdown)
            primary = f"This {role or 'sample'} brings {quality} to your mix."

        # --- Optionally add a secondary detail sentence -------------------
        secondary = _secondary_detail(breakdown, mix_profile, role, policy)

        # --- Incorporate the need description when available --------------
        need_clause = ""
        if need and need.description:
            # Weave the need description into the explanation when the
            # primary template hasn't already covered it fully.
            need_desc_lower = need.description.lower()
            primary_lower = primary.lower()
            # Avoid redundancy -- only add if the need description adds info.
            if not self._is_redundant(need_desc_lower, primary_lower):
                need_clause = f" Your mix analysis flagged: \"{need.description}.\""

        parts = [primary]
        if need_clause:
            parts.append(need_clause)
        if secondary:
            parts.append(secondary)

        return " ".join(parts).strip()

    def explain_batch(
        self,
        recommendations: list[Recommendation],
        mix_profile: MixProfile,
        needs: list[NeedOpportunity],
    ) -> list[Recommendation]:
        """
        Add explanations to every recommendation in the list, in-place.

        Each recommendation is matched to its best-fit need, then
        explained using :meth:`explain`.

        Returns the same list for convenience (mutations are in-place).
        """
        for rec in recommendations:
            matched_need = _match_need(rec, needs)
            rec.explanation = self.explain(rec, mix_profile, need=matched_need)
            if matched_need:
                rec.need_addressed = matched_need.description
        return recommendations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_template(
        self,
        template: str,
        policy: str,
        role: str,
        breakdown: ScoringBreakdown,
        mix_profile: MixProfile,
        need: NeedOpportunity | None,
    ) -> str:
        """Fill a policy template with context-specific values."""
        if not template:
            return ""

        quality = _describe_quality(role, breakdown)
        sample_role = role or "sample"

        # Determine which role label to use in the template.  For
        # fill_missing_role the "role" placeholder is the *missing* role
        # (from the need), while "sample_role" is what the sample is.
        missing_role = role
        if need and need.description:
            # Try to extract a role name from the need description.
            for candidate_role in (
                "kick", "snare", "clap", "hat", "bass", "lead",
                "pad", "vocal", "fx", "texture",
            ):
                if candidate_role in need.description.lower():
                    missing_role = candidate_role
                    break

        # Build substitution map.
        subs: dict[str, str] = {
            "role": missing_role or "element",
            "sample_role": sample_role,
            "quality": quality,
            "specific_quality": quality,
            "band": _find_gap_band(mix_profile),
            "existing_element": _find_existing_element(mix_profile, role),
        }

        try:
            return template.format(**subs)
        except KeyError:
            # Template had a placeholder we didn't expect -- fall back.
            return template

    @staticmethod
    def _is_redundant(need_text: str, explanation_text: str) -> bool:
        """
        Check whether the need description is already captured in the
        explanation text to avoid saying the same thing twice.
        """
        # Simple heuristic: if most significant words from the need appear
        # in the explanation, it's redundant.
        stop_words = {
            "the", "a", "an", "is", "are", "in", "to", "of", "and",
            "for", "with", "your", "this", "that", "has", "its", "it",
        }
        need_words = {
            w for w in need_text.split() if w not in stop_words and len(w) > 2
        }
        if not need_words:
            return True
        overlap = sum(1 for w in need_words if w in explanation_text)
        return overlap / len(need_words) > 0.6
