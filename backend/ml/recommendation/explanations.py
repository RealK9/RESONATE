"""
Explanation engine -- Phase 4 human-readable recommendation explanations.

Maps each recommendation to a natural-language explanation based on the
decision policy that triggered it, the scoring breakdown, and the mix
context.  The goal is output that reads like a music producer giving
quick, confident advice -- not a computer spitting out numbers.
"""
from __future__ import annotations

from typing import Any

from ml.models.mix_profile import MixProfile, NeedOpportunity
from ml.models.recommendation import Recommendation, ScoringBreakdown
from ml.recommendation.candidate_generator import _POLICY_TO_ROLES


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
# Genre family detection and genre-aware template overrides
# ---------------------------------------------------------------------------

_GENRE_FAMILIES: dict[str, list[str]] = {
    "trap": ["trap", "drill", "rage", "phonk"],
    "edm": ["edm", "house", "deep house", "tech house", "techno", "trance",
             "dubstep", "drum and bass", "dnb", "electro", "progressive house",
             "future bass"],
    "lofi": ["lo-fi", "lofi", "ambient", "chillhop", "downtempo", "chill"],
    "rnb": ["r&b", "rnb", "soul", "neo-soul", "neo soul", "funk"],
    "pop": ["pop", "synth-pop", "synthpop", "indie pop", "electropop",
            "dance pop", "k-pop"],
}


def _detect_genre_family(primary_cluster: str) -> str:
    """Map a primary_cluster string to one of the known genre families."""
    if not primary_cluster:
        return ""
    cluster_lower = primary_cluster.lower()
    for family, keywords in _GENRE_FAMILIES.items():
        for kw in keywords:
            if kw in cluster_lower:
                return family
    return ""


# Genre-specific template overrides keyed by (policy, genre_family).
# These replace the default template when the genre family is known,
# using vocabulary that resonates with producers in each world.
_GENRE_TEMPLATE_OVERRIDES: dict[tuple[str, str], str] = {
    # --- trap / drill ---
    ("fill_missing_role", "trap"):
        "Your beat is missing {role}. This {sample_role} knocks and would lock that gap down.",
    ("reinforce_existing", "trap"):
        "Your {role} needs more weight. This hits hard and adds {quality} to make it slap.",
    ("improve_polish", "trap"):
        "This {role} brings that bounce \u2014 {specific_quality} to make the 808 knock harder.",
    ("enhance_groove", "trap"):
        "This percussion slaps \u2014 gives your bounce that extra knock it needs.",
    ("reduce_emptiness", "trap"):
        "Your mix needs more weight in the {band} range. This hits that gap hard.",
    ("enhance_lift", "trap"):
        "This adds that build-up energy \u2014 makes the drop hit even harder.",

    # --- EDM / house ---
    ("fill_missing_role", "edm"):
        "Your mix is missing {role}. This {sample_role} drives the energy and fills that gap.",
    ("reinforce_existing", "edm"):
        "Your {role} could use reinforcement. This drives {quality} forward and strengthens the groove.",
    ("improve_polish", "edm"):
        "This {role} adds that main-stage energy \u2014 {specific_quality} to drive the drop.",
    ("enhance_groove", "edm"):
        "This drives the groove harder \u2014 gives your rhythm the energy it needs to build.",
    ("reduce_emptiness", "edm"):
        "Your mix has a gap in the {band} range. This builds out that frequency space with energy.",
    ("enhance_lift", "edm"):
        "This builds tension and energy \u2014 exactly what your drops need to land.",
    ("add_movement", "edm"):
        "This adds rhythmic drive to keep the energy building and evolving.",

    # --- lo-fi / ambient ---
    ("fill_missing_role", "lofi"):
        "Your mix is missing {role}. This {sample_role} adds warmth and texture to fill that space.",
    ("reinforce_existing", "lofi"):
        "Your {role} could use more texture. This brings {quality} and adds atmosphere.",
    ("improve_polish", "lofi"):
        "This {role} adds warmth and space \u2014 {specific_quality} for that lived-in vibe.",
    ("enhance_groove", "lofi"):
        "This adds subtle texture to the rhythm \u2014 gives your beat that dusty warmth.",
    ("reduce_emptiness", "lofi"):
        "Your mix has space in the {band} range. This fills it with warm atmosphere.",
    ("add_movement", "lofi"):
        "This adds gentle movement and texture to keep the vibe breathing.",

    # --- R&B / soul ---
    ("fill_missing_role", "rnb"):
        "Your mix is missing {role}. This {sample_role} brings that smooth feel to fill the pocket.",
    ("reinforce_existing", "rnb"):
        "Your {role} needs more pocket. This adds {quality} \u2014 smooth and locked in.",
    ("improve_polish", "rnb"):
        "This {role} brings warmth and groove \u2014 {specific_quality} for that polished feel.",
    ("enhance_groove", "rnb"):
        "This locks right into the pocket \u2014 gives your groove that smooth feel.",
    ("reduce_emptiness", "rnb"):
        "Your mix is missing warmth in the {band} range. This fills that space with smooth tone.",

    # --- pop ---
    ("fill_missing_role", "pop"):
        "Your mix is missing {role}. This {sample_role} adds the hook and fills that gap.",
    ("reinforce_existing", "pop"):
        "Your {role} needs more shine. This adds {quality} to make it radio-ready.",
    ("improve_polish", "pop"):
        "This {role} brings commercial polish \u2014 {specific_quality} for that radio-ready shine.",
    ("enhance_groove", "pop"):
        "This tightens the rhythm up \u2014 gives your hook that polished, radio-ready bounce.",
    ("reduce_emptiness", "pop"):
        "Your mix needs more presence in the {band} range. This adds shine to fill that space.",
}


# ---------------------------------------------------------------------------
# Alternative openings for batch de-duplication
# ---------------------------------------------------------------------------

# Keyed by policy, each list provides alternative opening patterns so that
# batches of recommendations don't all start the same way.
_ALTERNATIVE_OPENINGS: dict[str, list[str]] = {
    "fill_missing_role": [
        "There's a {role}-shaped hole in your mix. This {sample_role} fills it.",
        "Your track needs {role} \u2014 this {sample_role} is the missing piece.",
    ],
    "reinforce_existing": [
        "Your {role} is there but thin. This layers in {quality} to thicken it up.",
        "Think of this as doubling down on your {role} \u2014 adds {quality} where it counts.",
    ],
    "improve_polish": [
        "This {role} is that final coat of lacquer \u2014 {specific_quality}.",
        "For mix-ready sheen, this {role} delivers {specific_quality}.",
    ],
    "increase_contrast": [
        "Against your {existing_element}, this creates the contrast your mix needs.",
        "This plays off your {existing_element} \u2014 the tension makes both hit harder.",
    ],
    "add_movement": [
        "This keeps your arrangement from sitting still \u2014 rhythmic and melodic motion.",
        "Keeps the ear interested \u2014 this brings movement where the mix was static.",
    ],
    "reduce_emptiness": [
        "The {band} range is wide open \u2014 this fills that frequency real estate.",
        "You've got empty space in the {band}. This claims it.",
    ],
    "support_transition": [
        "Smooth out your transitions \u2014 this FX/texture bridges the gap between sections.",
        "This makes your sections flow into each other instead of jumping.",
    ],
    "enhance_groove": [
        "Your rhythm needs another layer \u2014 this percussion locks it in.",
        "This tightens up the groove and gives the rhythm more pocket.",
    ],
    "enhance_lift": [
        "Your builds need more payoff \u2014 this brings the emotional arc.",
        "This lifts the energy where your arrangement needs it most.",
    ],
}


# ---------------------------------------------------------------------------
# Confidence-calibrated language
# ---------------------------------------------------------------------------

def _confidence_phrase(breakdown: ScoringBreakdown) -> str:
    """
    Return a confidence qualifier based on overall breakdown strength.
    High scores get emphatic language, moderate gets measured,
    low gets exploratory.
    """
    # Use the average of all scoring components as a proxy for confidence.
    scores = [
        breakdown.need_fit, breakdown.role_fit,
        breakdown.spectral_complement, breakdown.tonal_compatibility,
        breakdown.rhythmic_compatibility, breakdown.style_prior_fit,
        breakdown.quality_prior,
    ]
    avg = sum(scores) / len(scores) if scores else 0.5

    if avg > 0.8:
        return "exactly what your mix needs"
    elif avg > 0.4:
        return "a solid addition"
    else:
        return "worth exploring"


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
        (breakdown.spectral_complement, "fills the frequency spectrum"),
        (breakdown.tonal_compatibility, "harmonic depth and color"),
        (breakdown.rhythmic_compatibility, "locked-in rhythm"),
        (breakdown.style_prior_fit, "stylistic cohesion"),
        (breakdown.quality_prior, "mix-ready polish"),
    ]

    # Role-specific overrides for more vivid, musical language.
    role_lower = sample_role.lower()
    if role_lower in ("kick",):
        candidates.append((breakdown.rhythmic_compatibility, "chest-thumping punch"))
        candidates.append((breakdown.spectral_complement, "sub-rattling weight"))
    elif role_lower in ("snare", "clap"):
        candidates.append((breakdown.rhythmic_compatibility, "crack that cuts through"))
        candidates.append((breakdown.spectral_complement, "weight and body"))
    elif role_lower in ("bass",):
        candidates.append((breakdown.spectral_complement, "ground-shaking low-end"))
        candidates.append((breakdown.tonal_compatibility, "deep, round tone"))
    elif role_lower in ("hat",):
        candidates.append((breakdown.spectral_complement, "crisp shimmer"))
        candidates.append((breakdown.rhythmic_compatibility, "tight groove feel"))
    elif role_lower in ("lead", "vocal"):
        candidates.append((breakdown.tonal_compatibility, "melodic character"))
        candidates.append((breakdown.spectral_complement, "bright, cutting presence"))
    elif role_lower in ("pad", "texture"):
        candidates.append((breakdown.spectral_complement, "warm, enveloping texture"))
        candidates.append((breakdown.tonal_compatibility, "lush harmonic tone"))
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

    Parameters
    ----------
    gap_result : dict | None
        Optional gap analysis result dict.  When provided, the engine
        references readiness scores and chart-potential ceilings in its
        explanations for fill_missing_role and improve_polish policies.
    """

    def __init__(self, gap_result: dict[str, Any] | None = None) -> None:
        self._gap_result = gap_result or {}
        # Track which opening patterns have been used during a batch
        # to avoid repetitive language.  Reset at the start of each batch.
        self._used_patterns: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Gap-context sentence generation
    # ------------------------------------------------------------------

    def _gap_context_sentence(self, policy: str, role: str, genre_family: str) -> str:
        """
        Build an optional sentence that references gap-analysis metrics
        (readiness score, chart potential) when available.
        """
        if not self._gap_result:
            return ""

        genre_label = genre_family or "current"

        if policy == "fill_missing_role":
            readiness = self._gap_result.get("readiness_score")
            if readiness is not None:
                return (
                    f"Your {genre_label} mix is at {readiness}/100 "
                    f"\u2014 adding {role or 'this element'} is a top priority."
                )

        if policy == "improve_polish":
            current_potential = self._gap_result.get("chart_potential")
            ceiling = self._gap_result.get("chart_potential_ceiling")
            if current_potential is not None and ceiling is not None:
                return (
                    f"This pushes your chart potential from {current_potential} "
                    f"toward {ceiling}."
                )

        return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # Detect genre family from the mix's style cluster.
        genre_family = _detect_genre_family(
            mix_profile.style.primary_cluster if mix_profile.style else ""
        )

        # --- Pick a template: genre-specific override or default ----------
        template = self._pick_template(policy, genre_family)
        primary = self._fill_template(
            template, policy, role, breakdown, mix_profile, need
        )

        # If no template matched, craft a generic explanation.
        if not primary:
            quality = _describe_quality(role, breakdown)
            primary = f"This {role or 'sample'} brings {quality} to your mix."

        # --- Optionally add a secondary detail sentence -------------------
        secondary = _secondary_detail(breakdown, mix_profile, role, policy)

        # --- Gap-context sentence when gap analysis is available ----------
        gap_sentence = self._gap_context_sentence(policy, role, genre_family)

        # --- Confidence calibration ---------------------------------------
        confidence = _confidence_phrase(breakdown)

        # --- Incorporate the need description when available --------------
        need_clause = ""
        if need and need.description:
            need_desc_lower = need.description.lower()
            primary_lower = primary.lower()
            if not self._is_redundant(need_desc_lower, primary_lower):
                need_clause = f" Your mix analysis flagged: \"{need.description}.\""

        parts = [primary]
        if gap_sentence:
            parts.append(gap_sentence)
        if need_clause:
            parts.append(need_clause)
        if secondary:
            parts.append(secondary)

        # Append a confidence-calibrated closer when not already wordy.
        total_len = sum(len(p) for p in parts)
        if total_len < 140:
            parts.append(f"Overall, {confidence}.")

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
        explained using :meth:`explain`.  The engine tracks which opening
        patterns have been used so that the batch reads naturally without
        repeating the same phrasing more than twice.

        Returns the same list for convenience (mutations are in-place).
        """
        # Reset the pattern tracker for this batch.
        self._used_patterns = {}

        for rec in recommendations:
            matched_need = _match_need(rec, needs)
            rec.explanation = self.explain(rec, mix_profile, need=matched_need)
            if matched_need:
                rec.need_addressed = matched_need.description
        return recommendations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_template(self, policy: str, genre_family: str) -> str:
        """
        Select the best template for a (policy, genre_family) pair.

        Checks genre-specific overrides first, falls back to the default
        policy template, and applies batch de-duplication to avoid
        repetitive openings.
        """
        # Track how many times we've used the primary template for this policy.
        pattern_key = f"{policy}:primary"
        use_count = self._used_patterns.get(pattern_key, 0)

        # Try genre-specific override first.
        if genre_family:
            genre_key = (policy, genre_family)
            genre_template = _GENRE_TEMPLATE_OVERRIDES.get(genre_key)
            if genre_template:
                genre_pattern_key = f"{policy}:{genre_family}"
                genre_use_count = self._used_patterns.get(genre_pattern_key, 0)
                if genre_use_count < 2:
                    self._used_patterns[genre_pattern_key] = genre_use_count + 1
                    return genre_template

        # If we've used the primary template twice already, try alternatives.
        if use_count >= 2:
            alternatives = _ALTERNATIVE_OPENINGS.get(policy, [])
            for i, alt in enumerate(alternatives):
                alt_key = f"{policy}:alt{i}"
                if self._used_patterns.get(alt_key, 0) < 2:
                    self._used_patterns[alt_key] = self._used_patterns.get(alt_key, 0) + 1
                    return alt

        # Default: use the primary template.
        self._used_patterns[pattern_key] = use_count + 1
        return _POLICY_TEMPLATES.get(policy, "")

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
