"""
Explanation engine -- Phase 4 human-readable recommendation explanations.

Maps each recommendation to a natural-language explanation based on the
decision policy that triggered it, the scoring breakdown, the mix
context, and gap analysis results.  The goal is output that reads like
a music producer giving quick, confident advice -- not a computer
spitting out numbers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ml.models.mix_profile import MixProfile, NeedOpportunity
from ml.models.recommendation import Recommendation, ScoringBreakdown
from ml.recommendation.candidate_generator import _POLICY_TO_ROLES

if TYPE_CHECKING:
    from ml.models.gap_analysis import GapAnalysisResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BAND_NAMES = [
    "sub-bass", "bass", "low-mids", "mids", "upper-mids",
    "presence", "brilliance", "air", "ultra-highs", "beyond",
]

# Human-friendly role labels (matches gap_analyzer._ROLE_LABELS)
_ROLE_DISPLAY: dict[str, str] = {
    "kick": "kick drum",
    "snare_clap": "snare / clap",
    "hats_tops": "hi-hats",
    "bass": "bass",
    "lead": "lead melody",
    "chord_support": "chords / harmony",
    "pad": "pads / atmosphere",
    "vocal_texture": "vocal texture",
    "fx_transitions": "FX / transitions",
    "ambience": "ambience / room",
    # Sample-level role names (shorter)
    "snare": "snare", "clap": "clap", "hat": "hi-hat",
    "vocal": "vocal", "fx": "FX", "texture": "texture",
}

# Genre display names
_GENRE_DISPLAY: dict[str, str] = {
    "modern_trap": "trap",
    "modern_drill": "drill",
    "2010s_edm_drop": "EDM",
    "2020s_melodic_house": "melodic house",
    "melodic_techno": "melodic techno",
    "dnb": "drum & bass",
    "afro_house": "afro house",
    "pop_production": "pop",
    "2000s_pop_chorus": "2000s pop",
    "r_and_b": "R&B",
    "1990s_boom_bap": "boom bap",
    "lo_fi_chill": "lo-fi",
    "cinematic": "cinematic",
    "ambient": "ambient",
}

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
# Gap-aware explanation helpers
# ---------------------------------------------------------------------------

def _gap_context_sentence(
    role: str,
    gap_result: "GapAnalysisResult | None",
    mix_profile: MixProfile,
) -> str:
    """Build a sentence referencing gap analysis data when available.

    Returns an empty string if gap analysis adds nothing beyond what the
    primary template already says.
    """
    if gap_result is None:
        return ""

    genre = _GENRE_DISPLAY.get(
        gap_result.blueprint_name, gap_result.blueprint_name
    )
    readiness = gap_result.production_readiness_score

    # Check if this sample's role fills a critical gap
    role_lower = role.lower() if role else ""
    # Map sample roles to gap analysis role names
    _role_map = {
        "kick": "kick", "snare": "snare_clap", "clap": "snare_clap",
        "hat": "hats_tops", "hihat": "hats_tops",
        "bass": "bass", "lead": "lead", "pad": "pad",
        "chord": "chord_support", "keys": "chord_support",
        "vocal": "vocal_texture", "fx": "fx_transitions",
        "texture": "ambience", "ambient": "ambience",
    }
    mapped_role = _role_map.get(role_lower, role_lower)

    # Check if this role is in the missing roles list
    if mapped_role in gap_result.missing_roles:
        role_label = _ROLE_DISPLAY.get(mapped_role, role)
        return (
            f"Your {genre} mix is currently at {readiness:.0f}/100 "
            f"production readiness — adding {role_label} is one of the "
            f"top priorities to get it chart-ready."
        )

    # Check for matching critical/moderate gaps
    for gap in gap_result.gaps[:5]:  # top 5 by severity
        if gap.category == "role" and gap.dimension == mapped_role:
            role_label = _ROLE_DISPLAY.get(mapped_role, role)
            if gap.severity > 0.7:
                return (
                    f"Gap analysis flagged weak {role_label} as a critical issue "
                    f"for your {genre} production."
                )
            elif gap.severity > 0.4:
                return (
                    f"Strengthening your {role_label} would bring your {genre} "
                    f"mix closer to chart-ready ({readiness:.0f}/100 currently)."
                )

        # Spectral gap that this role might address
        if gap.category == "spectral" and gap.severity > 0.5:
            band_role_map = {
                "sub": ["kick", "bass"],
                "bass": ["bass", "kick"],
                "low_mid": ["bass", "pad", "chord_support"],
                "mid": ["lead", "vocal_texture", "chord_support"],
                "upper_mid": ["lead", "hats_tops", "vocal_texture"],
                "presence": ["hats_tops", "lead"],
                "brilliance": ["hats_tops"],
                "air": ["hats_tops", "fx_transitions"],
            }
            filling_roles = band_role_map.get(gap.dimension, [])
            if mapped_role in filling_roles:
                return (
                    f"This helps fill the {gap.dimension.replace('_', '-')} "
                    f"energy gap identified in your mix."
                )

    # Chart potential uplift
    if gap_result.chart_potential_current < 50 and readiness < 70:
        return (
            f"Your {genre} mix has room to grow — currently at "
            f"{readiness:.0f}/100 readiness with "
            f"{gap_result.chart_potential_ceiling:.0f}/100 potential."
        )

    return ""


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

    When a ``gap_result`` is provided, explanations reference specific gap
    analysis data — production readiness scores, missing roles, and
    spectral gaps — so the producer understands exactly how each sample
    moves them closer to a chart-ready mix.
    """

    def __init__(self, gap_result: "GapAnalysisResult | None" = None):
        self._gap_result = gap_result

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

        # --- Gap-aware context sentence -----------------------------------
        gap_sentence = _gap_context_sentence(
            role, self._gap_result, mix_profile
        )

        # --- Optionally add a secondary detail sentence -------------------
        # Skip generic secondary if we already have a gap-specific sentence
        secondary = ""
        if not gap_sentence:
            secondary = _secondary_detail(breakdown, mix_profile, role, policy)

        # --- Incorporate the need description when available --------------
        need_clause = ""
        if need and need.description and not gap_sentence:
            # Weave the need description into the explanation when the
            # primary template hasn't already covered it fully.
            need_desc_lower = need.description.lower()
            primary_lower = primary.lower()
            # Avoid redundancy -- only add if the need description adds info.
            if not self._is_redundant(need_desc_lower, primary_lower):
                need_clause = f" Your mix analysis flagged: \"{need.description}.\""

        parts = [primary]
        if gap_sentence:
            parts.append(gap_sentence)
        elif need_clause:
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
