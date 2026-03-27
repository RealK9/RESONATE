"""
Needs engine -- the heart of RESONATE's mix intelligence.

Analyzes a MixProfile and diagnoses what the mix needs: spectral gaps,
missing sound roles, dynamic issues, spatial problems, and arrangement
weaknesses.  Each diagnosed need carries a severity (0-1) and a
recommendation policy that downstream modules use to select samples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ml.models.mix_profile import MixProfile, NeedOpportunity
from ml.models.reference_profile import ReferenceCorpus, StylePrior


# ---------------------------------------------------------------------------
# Valid categories & recommendation policies
# ---------------------------------------------------------------------------

VALID_CATEGORIES = frozenset({
    "spectral",
    "role",
    "dynamic",
    "spatial",
    "arrangement",
})

VALID_POLICIES = frozenset({
    "fill_missing_role",
    "reinforce_existing",
    "improve_polish",
    "increase_contrast",
    "add_movement",
    "reduce_emptiness",
    "support_transition",
    "enhance_groove",
    "enhance_lift",
})


# ---------------------------------------------------------------------------
# Style-aware spectral norms
# ---------------------------------------------------------------------------
# Per-band expected mean energy (0-1 scale, same ordering as BAND_NAMES in
# mix_analyzer: sub, bass, low_mid, mid, upper_mid, presence, brilliance,
# air, ultra_high, ceiling).
#
# These are idealized, but the *relative* shape is what matters -- the engine
# uses them to detect deviations that feel "wrong" for the style.

_STYLE_BAND_NORMS: dict[str, list[float]] = {
    # EDM / big-room: punchy sub, prominent kick, bright top
    "2010s_edm_drop":       [0.70, 0.75, 0.40, 0.45, 0.50, 0.55, 0.55, 0.50, 0.45, 0.30],
    # Melodic house: warmer, less aggressive top
    "2020s_melodic_house":  [0.55, 0.60, 0.45, 0.50, 0.50, 0.50, 0.45, 0.40, 0.35, 0.20],
    # Pop chorus: balanced full spectrum
    "2000s_pop_chorus":     [0.35, 0.45, 0.50, 0.55, 0.55, 0.55, 0.50, 0.45, 0.35, 0.20],
    # Boom bap: warm mids, moderate sub
    "1990s_boom_bap":       [0.45, 0.55, 0.50, 0.55, 0.45, 0.40, 0.35, 0.25, 0.15, 0.10],
    # Modern trap: heavy sub, sparse mids, crisp hats
    "modern_trap":          [0.80, 0.70, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.50, 0.35],
    # Drill: similar to trap, slightly denser mids
    "modern_drill":         [0.70, 0.65, 0.35, 0.40, 0.45, 0.50, 0.50, 0.50, 0.45, 0.30],
    # Melodic techno: driving bass, atmospheric highs
    "melodic_techno":       [0.50, 0.60, 0.45, 0.45, 0.50, 0.55, 0.50, 0.45, 0.35, 0.25],
    # Afro house: round low end, warm presence
    "afro_house":           [0.50, 0.55, 0.50, 0.50, 0.50, 0.50, 0.45, 0.40, 0.30, 0.20],
    # Cinematic: wide, deep low, atmospheric highs
    "cinematic":            [0.55, 0.50, 0.50, 0.50, 0.45, 0.45, 0.50, 0.55, 0.50, 0.40],
    # Lo-fi: rolled-off highs, warm mids, soft everything
    "lo_fi_chill":          [0.30, 0.40, 0.50, 0.55, 0.45, 0.35, 0.25, 0.15, 0.08, 0.05],
    # DnB: powerful sub, crisp highs, full mids
    "dnb":                  [0.65, 0.60, 0.45, 0.50, 0.50, 0.55, 0.55, 0.50, 0.45, 0.30],
    # Ambient: gentle, diffuse, no hard edges
    "ambient":              [0.25, 0.30, 0.40, 0.45, 0.40, 0.40, 0.45, 0.50, 0.40, 0.30],
    # R&B: warm low-mids, smooth presence
    "r_and_b":              [0.40, 0.50, 0.50, 0.55, 0.50, 0.50, 0.40, 0.35, 0.25, 0.15],
    # Pop production: balanced, polished
    "pop_production":       [0.35, 0.45, 0.50, 0.55, 0.55, 0.55, 0.50, 0.45, 0.35, 0.20],
}

# Fallback: average of all style norms (used when cluster is unknown)
_DEFAULT_NORM: list[float] = [
    sum(v[i] for v in _STYLE_BAND_NORMS.values()) / len(_STYLE_BAND_NORMS)
    for i in range(10)
]


# Band name → index mapping for convenience
_BAND_INDEX: dict[str, int] = {
    "sub": 0, "bass": 1, "low_mid": 2, "mid": 3, "upper_mid": 4,
    "presence": 5, "brilliance": 6, "air": 7, "ultra_high": 8, "ceiling": 9,
}

# Which styles expect strong rhythmic elements
_RHYTHMIC_STYLES = frozenset({
    "2010s_edm_drop", "2020s_melodic_house", "modern_trap", "modern_drill",
    "melodic_techno", "afro_house", "dnb", "1990s_boom_bap",
})

# Which styles expect harmonic content (chords, pads)
_HARMONIC_STYLES = frozenset({
    "2020s_melodic_house", "2000s_pop_chorus", "cinematic", "lo_fi_chill",
    "ambient", "r_and_b", "pop_production", "melodic_techno",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _get_norm(primary_cluster: str) -> list[float]:
    """Return the style-aware band norm for the given cluster."""
    return _STYLE_BAND_NORMS.get(primary_cluster, _DEFAULT_NORM)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _safe_mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ---------------------------------------------------------------------------
# NeedsEngine
# ---------------------------------------------------------------------------

class NeedsEngine:
    """
    Diagnose what a mix needs based on its MixProfile.

    Runs five diagnostic passes:
    1. Spectral gap analysis (band energy vs style norms)
    2. Missing role analysis (expected instruments not detected)
    3. Dynamic analysis (compression, cohesion, energy shape)
    4. Spatial analysis (stereo width issues)
    5. Arrangement analysis (density variation, breathing room)

    Returns a list of NeedOpportunity objects sorted by severity (highest first).

    Parameters
    ----------
    corpus:
        Optional :class:`ReferenceCorpus` to pull style-aware priors from.
        When provided, spectral norms are drawn from the corpus's
        ``StylePrior`` for the mix's primary cluster, making the engine
        upgrade automatically when better reference data is available.
        Falls back to the built-in ``_STYLE_BAND_NORMS`` when the corpus
        has no prior for a given cluster (or when *corpus* is ``None``).
    """

    def __init__(self, corpus: ReferenceCorpus | None = None) -> None:
        self._corpus = corpus

    def _get_spectral_norm(self, primary_cluster: str) -> list[float]:
        """Return the spectral band norm for *primary_cluster*.

        Prefers the corpus prior when available, falling back to the
        hardcoded ``_STYLE_BAND_NORMS`` dict.
        """
        if self._corpus is not None:
            prior = self._corpus.get_prior(primary_cluster)
            if prior is not None and len(prior.target_spectral_mean) == 10:
                return prior.target_spectral_mean
        return _get_norm(primary_cluster)

    def diagnose(self, mix_profile: MixProfile) -> list[NeedOpportunity]:
        """Analyze the profile and return a list of needs, sorted by severity."""
        needs: list[NeedOpportunity] = []

        primary = mix_profile.style.primary_cluster
        norm = self._get_spectral_norm(primary)

        needs.extend(self._spectral_analysis(mix_profile, norm, primary))
        needs.extend(self._role_analysis(mix_profile, primary))
        needs.extend(self._dynamic_analysis(mix_profile))
        needs.extend(self._spatial_analysis(mix_profile))
        needs.extend(self._arrangement_analysis(mix_profile))

        # Clamp all severities and sort descending
        for n in needs:
            n.severity = _clamp(n.severity)
        needs.sort(key=lambda n: n.severity, reverse=True)

        return needs

    # ------------------------------------------------------------------
    # 1. Spectral gap analysis
    # ------------------------------------------------------------------

    def _spectral_analysis(
        self,
        profile: MixProfile,
        norm: list[float],
        primary_cluster: str,
    ) -> list[NeedOpportunity]:
        needs: list[NeedOpportunity] = []
        mean_by_band = profile.spectral_occupancy.mean_by_band
        if not mean_by_band or len(mean_by_band) < 10:
            return needs

        # --- Top-end too sparse ---
        # Check air, brilliance, ultra_high against their style norms
        top_bands = ["air", "brilliance", "ultra_high"]
        top_deficits = []
        for name in top_bands:
            idx = _BAND_INDEX[name]
            deficit = norm[idx] - mean_by_band[idx]
            if deficit > 0:
                top_deficits.append(deficit)

        if top_deficits:
            avg_deficit = _safe_mean(top_deficits)
            # Trigger if average deficit exceeds threshold
            if avg_deficit > 0.12:
                severity = _clamp(avg_deficit * 2.0)
                needs.append(NeedOpportunity(
                    category="spectral",
                    description=(
                        "Top-end too sparse for this style -- the mix lacks "
                        "high-frequency energy in the air and brilliance bands, "
                        "making it sound dull or muffled."
                    ),
                    severity=severity,
                    recommendation_policy="reinforce_existing",
                ))

        # --- Upper mids overcrowded ---
        crowded_bands = ["upper_mid", "presence"]
        excesses = []
        for name in crowded_bands:
            idx = _BAND_INDEX[name]
            excess = mean_by_band[idx] - norm[idx]
            if excess > 0:
                excesses.append(excess)

        if excesses:
            avg_excess = _safe_mean(excesses)
            if avg_excess > 0.12:
                severity = _clamp(avg_excess * 2.0)
                needs.append(NeedOpportunity(
                    category="spectral",
                    description=(
                        "Upper mids overcrowded -- the presence and upper-mid "
                        "bands are significantly louder than expected for this "
                        "style, which can cause harshness and listener fatigue."
                    ),
                    severity=severity,
                    recommendation_policy="improve_polish",
                ))

        # --- Low-end too broad ---
        low_bands = ["sub", "bass"]
        low_excesses = []
        for name in low_bands:
            idx = _BAND_INDEX[name]
            excess = mean_by_band[idx] - norm[idx]
            if excess > 0:
                low_excesses.append(excess)

        if low_excesses:
            avg_excess = _safe_mean(low_excesses)
            if avg_excess > 0.12:
                severity = _clamp(avg_excess * 1.8)
                needs.append(NeedOpportunity(
                    category="spectral",
                    description=(
                        "Low-end too broad -- sub and bass bands carry "
                        "excessive energy for this style, which muddies the "
                        "mix and eats headroom."
                    ),
                    severity=severity,
                    recommendation_policy="improve_polish",
                ))

        # --- Harmonic layer too thin ---
        mid_bands = ["low_mid", "mid", "upper_mid"]
        mid_deficits = []
        for name in mid_bands:
            idx = _BAND_INDEX[name]
            deficit = norm[idx] - mean_by_band[idx]
            if deficit > 0:
                mid_deficits.append(deficit)

        if mid_deficits:
            avg_deficit = _safe_mean(mid_deficits)
            if avg_deficit > 0.15:
                severity = _clamp(avg_deficit * 1.8)
                needs.append(NeedOpportunity(
                    category="spectral",
                    description=(
                        "Harmonic layer too thin -- the mid-range lacks "
                        "body and warmth, leaving the mix feeling hollow "
                        "and unsupported."
                    ),
                    severity=severity,
                    recommendation_policy="fill_missing_role",
                ))

        return needs

    # ------------------------------------------------------------------
    # 2. Missing role analysis
    # ------------------------------------------------------------------

    def _role_analysis(
        self,
        profile: MixProfile,
        primary_cluster: str,
    ) -> list[NeedOpportunity]:
        needs: list[NeedOpportunity] = []
        roles = profile.source_roles.roles
        if not roles:
            return needs

        is_rhythmic = primary_cluster in _RHYTHMIC_STYLES
        is_harmonic = primary_cluster in _HARMONIC_STYLES

        # --- Weak attack support ---
        # Kick and snare expected in rhythmic styles
        kick_conf = roles.get("kick", 0.0)
        snare_conf = roles.get("snare_clap", 0.0)
        if is_rhythmic and (kick_conf + snare_conf) / 2.0 < 0.25:
            avg = (kick_conf + snare_conf) / 2.0
            severity = _clamp(1.0 - avg * 4.0)  # lower confidence → higher severity
            needs.append(NeedOpportunity(
                category="role",
                description=(
                    "Weak attack support -- kick and snare presence is "
                    "very low for a rhythmic style, leaving the groove "
                    "without a solid backbone."
                ),
                severity=severity,
                recommendation_policy="fill_missing_role",
            ))

        # --- No rhythmic sparkle ---
        hats_conf = roles.get("hats_tops", 0.0)
        if is_rhythmic and hats_conf < 0.20:
            severity = _clamp(1.0 - hats_conf * 5.0)
            needs.append(NeedOpportunity(
                category="role",
                description=(
                    "No rhythmic sparkle -- hi-hats and top percussion are "
                    "nearly absent, making the rhythm feel sluggish and "
                    "lacking energy."
                ),
                severity=severity,
                recommendation_policy="enhance_groove",
            ))

        # --- No glue texture ---
        pad_conf = roles.get("pad", 0.0)
        ambience_conf = roles.get("ambience", 0.0)
        if pad_conf < 0.15 and ambience_conf < 0.15:
            combined = (pad_conf + ambience_conf) / 2.0
            severity = _clamp(0.8 - combined * 4.0)
            needs.append(NeedOpportunity(
                category="role",
                description=(
                    "No glue texture -- pads and ambient layers are missing, "
                    "so the mix elements feel disconnected and sterile."
                ),
                severity=severity,
                recommendation_policy="fill_missing_role",
            ))

        # --- Needs chord support ---
        chord_conf = roles.get("chord_support", 0.0)
        if is_harmonic and chord_conf < 0.20:
            severity = _clamp(0.9 - chord_conf * 4.0)
            needs.append(NeedOpportunity(
                category="role",
                description=(
                    "Needs chord support -- the harmonic foundation is weak "
                    "for a style that relies on chordal movement and tonal "
                    "context."
                ),
                severity=severity,
                recommendation_policy="fill_missing_role",
            ))

        # --- Lacks emotional support ---
        vocal_conf = roles.get("vocal_texture", 0.0)
        if vocal_conf < 0.10 and pad_conf < 0.15:
            combined = (vocal_conf + pad_conf) / 2.0
            severity = _clamp(0.7 - combined * 4.0)
            needs.append(NeedOpportunity(
                category="role",
                description=(
                    "Lacks emotional support -- neither vocal texture nor "
                    "pad layers are present to provide warmth and emotional "
                    "connection."
                ),
                severity=severity,
                recommendation_policy="fill_missing_role",
            ))

        return needs

    # ------------------------------------------------------------------
    # 3. Dynamic analysis
    # ------------------------------------------------------------------

    def _dynamic_analysis(self, profile: MixProfile) -> list[NeedOpportunity]:
        needs: list[NeedOpportunity] = []
        analysis = profile.analysis

        # --- Section energy too flat (needs lift into transitions) ---
        section_energy = analysis.section_energy
        if len(section_energy) >= 4:
            energy_std = _safe_std(section_energy)
            # Very flat section energy → no builds or drops
            if energy_std < 0.05:
                severity = _clamp((0.05 - energy_std) * 15.0 + 0.3)
                needs.append(NeedOpportunity(
                    category="dynamic",
                    description=(
                        "Section energy is too flat -- the mix needs "
                        "lift into transitions, with builds and drops "
                        "to create forward momentum."
                    ),
                    severity=severity,
                    recommendation_policy="add_movement",
                ))

        # --- Dynamic range too low (over-compressed) ---
        dynamic_range = analysis.dynamic_range
        if dynamic_range < 4.0 and dynamic_range >= 0.0:
            severity = _clamp((4.0 - dynamic_range) / 4.0)
            needs.append(NeedOpportunity(
                category="dynamic",
                description=(
                    "Over-compressed -- dynamic range is very low, making "
                    "the mix sound squashed, fatiguing, and lifeless."
                ),
                severity=severity,
                recommendation_policy="improve_polish",
            ))

        # --- Dynamic range too high (lacks cohesion) ---
        if dynamic_range > 20.0:
            severity = _clamp((dynamic_range - 20.0) / 15.0)
            needs.append(NeedOpportunity(
                category="dynamic",
                description=(
                    "Lacks cohesion -- dynamic range is unusually high, "
                    "suggesting elements aren't sitting together properly "
                    "in the mix."
                ),
                severity=severity,
                recommendation_policy="improve_polish",
            ))

        return needs

    # ------------------------------------------------------------------
    # 4. Spatial analysis
    # ------------------------------------------------------------------

    def _spatial_analysis(self, profile: MixProfile) -> list[NeedOpportunity]:
        needs: list[NeedOpportunity] = []
        width = profile.stereo_width

        # --- Overall width too low ---
        if width.overall_width < 0.15:
            severity = _clamp((0.15 - width.overall_width) / 0.15 * 0.8 + 0.2)
            needs.append(NeedOpportunity(
                category="spatial",
                description=(
                    "Too narrow, needs width layer -- the stereo image is "
                    "essentially mono, which sounds flat and small on "
                    "speakers and headphones alike."
                ),
                severity=severity,
                recommendation_policy="reduce_emptiness",
            ))

        # --- Width imbalance between bands ---
        width_by_band = width.width_by_band
        if len(width_by_band) >= 4:
            w_std = _safe_std(width_by_band)
            w_mean = _safe_mean(width_by_band)
            if w_mean > 0.05 and w_std > 0.25:
                severity = _clamp((w_std - 0.25) * 2.0 + 0.2)
                needs.append(NeedOpportunity(
                    category="spatial",
                    description=(
                        "Width imbalance by band -- some frequency bands "
                        "are wide while others are narrow, creating an "
                        "uneven stereo image that shifts with the content."
                    ),
                    severity=severity,
                    recommendation_policy="improve_polish",
                ))

        # --- Low-end too wide (bass should be centered) ---
        if len(width_by_band) >= 2:
            low_width = _safe_mean(width_by_band[:2])  # sub + bass
            if low_width > 0.40:
                severity = _clamp((low_width - 0.40) / 0.60 * 0.7 + 0.2)
                needs.append(NeedOpportunity(
                    category="spatial",
                    description=(
                        "Bass should be more centered -- the low end is "
                        "spread wide in the stereo field, which reduces "
                        "punch and causes phase issues on mono playback."
                    ),
                    severity=severity,
                    recommendation_policy="improve_polish",
                ))

        return needs

    # ------------------------------------------------------------------
    # 5. Arrangement analysis
    # ------------------------------------------------------------------

    def _arrangement_analysis(self, profile: MixProfile) -> list[NeedOpportunity]:
        needs: list[NeedOpportunity] = []
        density_map = profile.density_map

        if len(density_map) < 4:
            return needs

        d_std = _safe_std(density_map)
        d_mean = _safe_mean(density_map)
        d_max = max(density_map)
        d_min = min(density_map)
        d_range = d_max - d_min

        # --- Density map too uniform (no movement/contrast) ---
        if d_range < 0.10 and d_mean > 0.1:
            severity = _clamp((0.10 - d_range) * 6.0 + 0.3)
            needs.append(NeedOpportunity(
                category="arrangement",
                description=(
                    "No movement or contrast -- the density is almost "
                    "constant throughout, with no builds, drops, or "
                    "structural variation to maintain listener interest."
                ),
                severity=severity,
                recommendation_policy="increase_contrast",
            ))

        # --- Density always high (no breathing room) ---
        if d_mean > 0.85 and d_min > 0.70:
            severity = _clamp((d_mean - 0.85) * 4.0 + 0.3)
            needs.append(NeedOpportunity(
                category="arrangement",
                description=(
                    "No breathing room -- the arrangement is dense "
                    "throughout with no space for the listener to "
                    "reset, causing fatigue."
                ),
                severity=severity,
                recommendation_policy="increase_contrast",
            ))

        # --- Low density throughout (feels empty) ---
        if d_mean < 0.20 and d_max < 0.35:
            severity = _clamp((0.20 - d_mean) * 4.0 + 0.3)
            needs.append(NeedOpportunity(
                category="arrangement",
                description=(
                    "Feels empty, needs layers -- the overall density is "
                    "very low throughout, suggesting the arrangement is "
                    "under-populated and needs supporting elements."
                ),
                severity=severity,
                recommendation_policy="reduce_emptiness",
            ))

        # --- No energy variation in sections (arrangement is static) ---
        section_energy = profile.analysis.section_energy
        if len(section_energy) >= 4:
            # Compare first half to second half for any arc
            half = len(section_energy) // 2
            first_half = _safe_mean(section_energy[:half])
            second_half = _safe_mean(section_energy[half:])
            energy_contrast = abs(first_half - second_half)

            # Also check overall std
            e_std = _safe_std(section_energy)

            if e_std < 0.04 and energy_contrast < 0.03:
                severity = _clamp(0.5 - e_std * 5.0)
                needs.append(NeedOpportunity(
                    category="arrangement",
                    description=(
                        "Arrangement is static -- section energy barely "
                        "changes, with no arc or development from start "
                        "to finish."
                    ),
                    severity=severity,
                    recommendation_policy="add_movement",
                ))

        return needs
