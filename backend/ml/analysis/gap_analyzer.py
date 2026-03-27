"""
RESONATE — Gap Analyzer

The core intelligence engine.  Compares a producer's MixProfile against
a genre-specific GenreBlueprint and identifies every gap standing between
the current mix and a chart-ready production.

Five analysis passes run in sequence:
  1. Spectral gaps   — frequency-band energy vs. target
  2. Role gaps       — missing or underrepresented sound roles
  3. Dynamic gaps    — loudness / dynamic range vs. genre norms
  4. Perceptual gaps — brightness, warmth, punch vs. expectations
  5. Arrangement gaps — layer count and density vs. genre norms

The results feed directly into the recommendation engine so RESONATE can
tell a producer *exactly* what to add, remove, or adjust.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ml.analysis.genre_blueprints import (
    BAND_NAMES,
    NUM_BANDS,
    GenreBlueprint,
    get_best_blueprint,
    get_blueprint,
)
from ml.models.gap_analysis import GapAnalysisResult, GapItem
from ml.models.mix_profile import MixProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROLE_THRESHOLD = 0.35  # below this confidence the role is considered missing

# Human-friendly band labels for messages
_BAND_LABELS: dict[str, str] = {
    "sub":        "sub-bass (20-60 Hz)",
    "bass":       "bass (60-250 Hz)",
    "low_mid":    "low-mids (250-500 Hz)",
    "mid":        "mids (500 Hz-2 kHz)",
    "upper_mid":  "upper-mids (2-4 kHz)",
    "presence":   "presence (4-6 kHz)",
    "brilliance": "brilliance (6-10 kHz)",
    "air":        "air (10-14 kHz)",
    "ultra_high":  "ultra-highs (14-18 kHz)",
    "ceiling":    "ceiling (18-20 kHz)",
}

# Human-friendly role labels for messages
_ROLE_LABELS: dict[str, str] = {
    "kick":           "kick drum",
    "snare_clap":     "snare / clap",
    "hats_tops":      "hi-hats / tops",
    "bass":           "bass",
    "lead":           "lead melody",
    "chord_support":  "chords / harmony",
    "pad":            "pads / atmosphere",
    "vocal_texture":  "vocal texture",
    "fx_transitions": "FX / transitions",
    "ambience":       "ambience / room",
}

_GENRE_DISPLAY: dict[str, str] = {
    "modern_trap":          "trap",
    "modern_drill":         "drill",
    "2010s_edm_drop":       "EDM",
    "2020s_melodic_house":  "melodic house",
    "melodic_techno":       "melodic techno",
    "dnb":                  "drum & bass",
    "afro_house":           "afro house",
    "pop_production":       "pop",
    "2000s_pop_chorus":     "2000s pop",
    "r_and_b":              "R&B",
    "1990s_boom_bap":       "boom bap",
    "lo_fi_chill":          "lo-fi",
    "cinematic":            "cinematic",
    "ambient":              "ambient",
}


# ---------------------------------------------------------------------------
# GapAnalyzer
# ---------------------------------------------------------------------------

class GapAnalyzer:
    """Analyze a MixProfile and identify gaps to chart readiness."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, mix_profile: MixProfile) -> GapAnalysisResult:
        """Run the full gap analysis pipeline.

        Returns a fully populated ``GapAnalysisResult``.  This method
        **never** raises — on unexpected errors it returns a minimal
        result with ``production_readiness_score = 0``.
        """
        try:
            return self._analyze_impl(mix_profile)
        except Exception:
            logger.exception("Gap analysis failed unexpectedly")
            return GapAnalysisResult(
                genre_detected=mix_profile.style.primary_cluster,
                era_detected=mix_profile.style.era_estimate,
                blueprint_name="pop_production",
                confidence=0.0,
                production_readiness_score=0.0,
            )

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _analyze_impl(self, mp: MixProfile) -> GapAnalysisResult:
        # 1. Resolve genre blueprint — try direct cluster match first,
        #    then fall back to probability-based selection
        blueprint = get_blueprint(mp.style.primary_cluster)
        if blueprint is None and mp.style.cluster_probabilities:
            blueprint, _ = get_best_blueprint(mp.style.cluster_probabilities)
        if blueprint is None:
            # Ultimate fallback — pop production is the most generic
            blueprint = get_blueprint("pop_production")

        genre_display = _GENRE_DISPLAY.get(blueprint.name, blueprint.name)

        # 2. Run five analysis passes
        all_gaps: list[GapItem] = []

        spectral_gaps = self._spectral_gaps(mp, blueprint)
        all_gaps.extend(spectral_gaps)

        role_gaps, missing_roles, present_roles = self._role_gaps(mp, blueprint)
        all_gaps.extend(role_gaps)

        dynamic_gaps = self._dynamic_gaps(mp, blueprint, genre_display)
        all_gaps.extend(dynamic_gaps)

        perceptual_gaps = self._perceptual_gaps(mp, blueprint, genre_display)
        all_gaps.extend(perceptual_gaps)

        arrangement_gaps = self._arrangement_gaps(mp, blueprint, genre_display)
        all_gaps.extend(arrangement_gaps)

        # 3. Sort by severity descending
        all_gaps.sort(key=lambda g: g.severity, reverse=True)

        # 4. Compute headline scores
        production_readiness = self._compute_readiness_score(all_gaps, blueprint)
        genre_coherence = self._compute_genre_coherence(mp)
        chart_current = self._compute_chart_potential(
            production_readiness, genre_coherence, blueprint,
        )
        chart_ceiling = self._compute_chart_potential(
            100.0, genre_coherence, blueprint,
        )

        # Detection confidence
        probs = mp.style.cluster_probabilities
        detection_confidence = probs.get(mp.style.primary_cluster, 0.0) if probs else 0.0

        # Gap severity counts
        critical = sum(1 for g in all_gaps if g.severity > 0.7)
        moderate = sum(1 for g in all_gaps if 0.4 <= g.severity <= 0.7)
        minor = sum(1 for g in all_gaps if g.severity < 0.4)

        result = GapAnalysisResult(
            genre_detected=mp.style.primary_cluster,
            era_detected=mp.style.era_estimate,
            blueprint_name=blueprint.name,
            confidence=detection_confidence,
            production_readiness_score=round(production_readiness, 1),
            chart_potential_current=round(chart_current, 1),
            chart_potential_ceiling=round(chart_ceiling, 1),
            genre_coherence_score=round(genre_coherence, 3),
            gaps=all_gaps,
            missing_roles=missing_roles,
            present_roles=present_roles,
            total_gaps=len(all_gaps),
            critical_gaps=critical,
            moderate_gaps=moderate,
            minor_gaps=minor,
        )

        logger.info(
            "Gap analysis complete — genre=%s  readiness=%.1f  gaps=%d "
            "(critical=%d, moderate=%d, minor=%d)",
            blueprint.name,
            production_readiness,
            len(all_gaps),
            critical,
            moderate,
            minor,
        )

        return result

    # ------------------------------------------------------------------
    # Pass 1: Spectral gaps
    # ------------------------------------------------------------------

    def _spectral_gaps(
        self,
        mp: MixProfile,
        bp: GenreBlueprint,
    ) -> list[GapItem]:
        """Compare spectral occupancy per band against the blueprint target."""
        gaps: list[GapItem] = []

        mean_by_band = mp.spectral_occupancy.mean_by_band
        if not mean_by_band or len(mean_by_band) < NUM_BANDS:
            return gaps

        target = bp.target_spectral
        tolerance = bp.spectral_tolerance

        if len(target) < NUM_BANDS or len(tolerance) < NUM_BANDS:
            return gaps

        # Derive band importance from inverse tolerance (tighter tolerance = more important)
        band_importance = [1.0 / max(t, 0.01) for t in tolerance]
        max_imp = max(band_importance) if band_importance else 1.0
        band_importance = [imp / max_imp for imp in band_importance]  # normalize 0-1

        for i in range(NUM_BANDS):
            actual = mean_by_band[i]
            tgt = target[i]
            tol = tolerance[i] if i < len(tolerance) else 0.1
            weight = band_importance[i] if i < len(band_importance) else 1.0

            diff = actual - tgt
            magnitude = abs(diff)

            if magnitude <= tol:
                continue

            band_name = BAND_NAMES[i] if i < len(BAND_NAMES) else f"band_{i}"
            band_label = _BAND_LABELS.get(band_name, band_name)

            direction = "increase" if diff < 0 else "decrease"
            severity = min(1.0, (magnitude - tol) * weight * 2.0)

            if direction == "increase":
                msg = (
                    f"Your {band_label} energy is lower than expected — "
                    f"boost this range to fill out the low end"
                    if i < 2
                    else f"Your {band_label} energy could use a lift — "
                    f"this frequency range helps your track cut through"
                )
            else:
                msg = (
                    f"Your {band_label} is a bit hot — "
                    f"tame this range to avoid muddiness"
                    if i < 4
                    else f"Your {band_label} is running heavy — "
                    f"pull it back for a cleaner, more balanced mix"
                )

            gaps.append(GapItem(
                category="spectral",
                dimension=band_name,
                current_value=round(actual, 3),
                target_value=round(tgt, 3),
                gap_magnitude=round(magnitude, 3),
                direction=direction,
                severity=round(severity, 3),
                message=msg,
            ))

        return gaps

    # ------------------------------------------------------------------
    # Pass 2: Role gaps
    # ------------------------------------------------------------------

    def _role_gaps(
        self,
        mp: MixProfile,
        bp: GenreBlueprint,
    ) -> tuple[list[GapItem], list[str], list[str]]:
        """Check for missing or weak sound roles."""
        gaps: list[GapItem] = []
        missing_roles: list[str] = []
        present_roles: list[str] = []

        detected = mp.source_roles.roles
        required = bp.required_roles
        # Derive role criticality from required confidence thresholds
        # Higher required confidence = more critical to the genre
        criticality = {role: min(1.0, conf * 1.3) for role, conf in required.items()}

        for role, target_conf in required.items():
            actual_conf = detected.get(role, 0.0)

            if actual_conf >= _ROLE_THRESHOLD:
                present_roles.append(role)

                # Check if it is significantly below the genre expectation
                deficit = target_conf - actual_conf
                if deficit > 0.15:
                    role_label = _ROLE_LABELS.get(role, role)
                    crit = criticality.get(role, 0.5)
                    severity = min(1.0, deficit * crit * 1.5)
                    gaps.append(GapItem(
                        category="role",
                        dimension=role,
                        current_value=round(actual_conf, 3),
                        target_value=round(target_conf, 3),
                        gap_magnitude=round(deficit, 3),
                        direction="increase",
                        severity=round(severity, 3),
                        message=(
                            f"Your {role_label} is present but could be stronger — "
                            f"it's sitting below where it should for this style"
                        ),
                    ))
            else:
                # Role is missing or very weak
                missing_roles.append(role)
                role_label = _ROLE_LABELS.get(role, role)
                crit = criticality.get(role, 0.5)
                deficit = target_conf - actual_conf
                severity = min(1.0, crit * (0.6 + deficit * 0.5))

                gaps.append(GapItem(
                    category="role",
                    dimension=role,
                    current_value=round(actual_conf, 3),
                    target_value=round(target_conf, 3),
                    gap_magnitude=round(deficit, 3),
                    direction="add",
                    severity=round(severity, 3),
                    message=(
                        f"Missing {role_label} — this is a core element "
                        f"in this genre and adding it will make a big difference"
                    ),
                ))

        return gaps, missing_roles, present_roles

    # ------------------------------------------------------------------
    # Pass 3: Dynamic gaps
    # ------------------------------------------------------------------

    def _dynamic_gaps(
        self,
        mp: MixProfile,
        bp: GenreBlueprint,
        genre_display: str,
    ) -> list[GapItem]:
        """Check loudness and dynamic range against genre norms."""
        gaps: list[GapItem] = []

        # --- Loudness (LUFS) ---
        lufs = mp.analysis.loudness_lufs
        if lufs > -100.0:  # sentinel check — -100 means uninitialized
            lufs_diff = lufs - bp.target_lufs
            lufs_mag = abs(lufs_diff)

            if lufs_mag > bp.lufs_tolerance:
                direction = "increase" if lufs_diff < 0 else "decrease"
                severity = min(1.0, (lufs_mag - bp.lufs_tolerance) / 6.0)

                if direction == "increase":
                    msg = (
                        f"Track is too quiet for {genre_display} — "
                        f"target around {bp.target_lufs:.0f} LUFS, "
                        f"currently {lufs:.1f} LUFS"
                    )
                else:
                    msg = (
                        f"Track is pushed too loud for {genre_display} — "
                        f"back off to around {bp.target_lufs:.0f} LUFS "
                        f"to preserve dynamics (currently {lufs:.1f} LUFS)"
                    )

                gaps.append(GapItem(
                    category="dynamics",
                    dimension="loudness_lufs",
                    current_value=round(lufs, 2),
                    target_value=round(bp.target_lufs, 2),
                    gap_magnitude=round(lufs_mag, 2),
                    direction=direction,
                    severity=round(severity, 3),
                    message=msg,
                ))

        # --- Dynamic range (crest factor) ---
        dr = mp.analysis.dynamic_range
        if dr > 0.0:
            # Blueprint provides (min_dr, max_dr) tuple
            dr_min, dr_max = bp.target_dynamic_range
            dr_target = (dr_min + dr_max) / 2.0
            dr_tolerance = (dr_max - dr_min) / 2.0

            dr_diff = dr - dr_target
            dr_mag = abs(dr_diff)

            if dr_mag > dr_tolerance:
                direction = "increase" if dr_diff < 0 else "decrease"
                severity = min(1.0, (dr_mag - dr_tolerance) / 8.0)

                if direction == "increase":
                    msg = (
                        f"Your mix is over-compressed — dynamic range is "
                        f"{dr:.1f} dB but {genre_display} tracks typically "
                        f"sit around {dr_target:.0f} dB. "
                        f"Ease up on the limiter to let it breathe"
                    )
                else:
                    msg = (
                        f"Your mix could be tighter — dynamic range is "
                        f"{dr:.1f} dB, which is wide for {genre_display}. "
                        f"Consider more bus compression to glue things together"
                    )

                gaps.append(GapItem(
                    category="dynamics",
                    dimension="dynamic_range",
                    current_value=round(dr, 2),
                    target_value=round(dr_target, 2),
                    gap_magnitude=round(dr_mag, 2),
                    direction=direction,
                    severity=round(severity, 3),
                    message=msg,
                ))

        return gaps

    # ------------------------------------------------------------------
    # Pass 4: Perceptual gaps
    # ------------------------------------------------------------------

    def _perceptual_gaps(
        self,
        mp: MixProfile,
        bp: GenreBlueprint,
        genre_display: str,
    ) -> list[GapItem]:
        """Evaluate brightness, warmth, and punch against genre expectations."""
        gaps: list[GapItem] = []

        mean_by_band = mp.spectral_occupancy.mean_by_band
        if not mean_by_band or len(mean_by_band) < NUM_BANDS:
            return gaps

        target = bp.target_perceptual        # dict[str, float]
        tol_map = bp.perceptual_tolerance     # dict[str, float]

        # Brightness: mean of bands 5-8 (presence, brilliance, air, ultra_high)
        brightness = _safe_mean(mean_by_band[5:9])
        target_brightness = target.get("brightness", 0.5)
        tol = tol_map.get("brightness", 0.1)
        bright_diff = brightness - target_brightness
        bright_mag = abs(bright_diff)

        if bright_mag > tol:
            direction = "increase" if bright_diff < 0 else "decrease"
            severity = min(1.0, (bright_mag - tol) * 3.0)

            if direction == "increase":
                msg = (
                    f"Mix sounds a bit dark for {genre_display} — "
                    f"add some high-frequency energy or air to open it up"
                )
            else:
                msg = (
                    f"Mix is brighter than typical {genre_display} — "
                    f"consider softening the highs for a smoother character"
                )

            gaps.append(GapItem(
                category="perceptual",
                dimension="brightness",
                current_value=round(brightness, 3),
                target_value=round(target_brightness, 3),
                gap_magnitude=round(bright_mag, 3),
                direction=direction,
                severity=round(severity, 3),
                message=msg,
            ))

        # Warmth: mean of bands 1-3 (bass, low_mid, mid)
        warmth = _safe_mean(mean_by_band[1:4])
        target_warmth = target.get("warmth", 0.5)
        tol = tol_map.get("warmth", 0.1)
        warm_diff = warmth - target_warmth
        warm_mag = abs(warm_diff)

        if warm_mag > tol:
            direction = "increase" if warm_diff < 0 else "decrease"
            severity = min(1.0, (warm_mag - tol) * 3.0)

            if direction == "increase":
                msg = (
                    f"Mix feels thin in the low-mids — "
                    f"typical {genre_display} tracks have more warmth and body"
                )
            else:
                msg = (
                    f"Low-mid region is heavier than expected for {genre_display} — "
                    f"clean up this range to reduce muddiness"
                )

            gaps.append(GapItem(
                category="perceptual",
                dimension="warmth",
                current_value=round(warmth, 3),
                target_value=round(target_warmth, 3),
                gap_magnitude=round(warm_mag, 3),
                direction=direction,
                severity=round(severity, 3),
                message=msg,
            ))

        # Punch: proxy from sub + bass energy (transient density estimate)
        punch = _safe_mean(mean_by_band[0:2])
        target_punch = target.get("punch", 0.5)
        tol = tol_map.get("punch", 0.1)
        punch_diff = punch - target_punch
        punch_mag = abs(punch_diff)

        if punch_mag > tol:
            direction = "increase" if punch_diff < 0 else "decrease"
            severity = min(1.0, (punch_mag - tol) * 2.5)

            if direction == "increase":
                msg = (
                    f"Track lacks punch in the low end — "
                    f"in {genre_display}, the kick and bass need to hit harder"
                )
            else:
                msg = (
                    f"Low-end punch is overshooting for {genre_display} — "
                    f"dial back the sub/bass transients for a cleaner hit"
                )

            gaps.append(GapItem(
                category="perceptual",
                dimension="punch",
                current_value=round(punch, 3),
                target_value=round(target_punch, 3),
                gap_magnitude=round(punch_mag, 3),
                direction=direction,
                severity=round(severity, 3),
                message=msg,
            ))

        return gaps

    # ------------------------------------------------------------------
    # Pass 5: Arrangement gaps
    # ------------------------------------------------------------------

    def _arrangement_gaps(
        self,
        mp: MixProfile,
        bp: GenreBlueprint,
        genre_display: str,
    ) -> list[GapItem]:
        """Evaluate arrangement density and layer count."""
        gaps: list[GapItem] = []

        # Count active roles (confidence > 0.3)
        active_count = sum(
            1 for conf in mp.source_roles.roles.values() if conf > 0.3
        )
        # Blueprint provides (min, max) tuple for instrument count
        count_min, count_max = bp.typical_instrument_count
        target_count = (count_min + count_max) // 2
        count_tolerance = (count_max - count_min) // 2

        diff = active_count - target_count

        if abs(diff) > count_tolerance:
            direction = "increase" if diff < 0 else "decrease"
            magnitude = abs(diff) / max(target_count, 1)
            severity = min(1.0, (abs(diff) - count_tolerance) / max(target_count, 1) * 2.0)

            if direction == "increase":
                msg = (
                    f"Only {active_count} active layers detected — "
                    f"{genre_display} hits typically have {count_min}"
                    f"-{count_max}. "
                    f"Consider adding more elements to fill out the arrangement"
                )
            else:
                msg = (
                    f"{active_count} active layers detected — "
                    f"that's a lot for {genre_display} "
                    f"(typical range is {count_min}"
                    f"-{count_max}). "
                    f"Some elements may be competing — try muting weaker layers"
                )

            gaps.append(GapItem(
                category="arrangement",
                dimension="layer_count",
                current_value=float(active_count),
                target_value=float(target_count),
                gap_magnitude=round(magnitude, 3),
                direction=direction,
                severity=round(severity, 3),
                message=msg,
            ))

        # Density check via density_map if available
        if mp.density_map:
            avg_density = sum(mp.density_map) / len(mp.density_map)
            target_density = (
                bp.scoring_weights.get("_density_mean", 0.0)
                or _estimate_density_target(bp)
            )

            density_diff = avg_density - target_density
            density_mag = abs(density_diff)

            if density_mag > 0.15:
                direction = "increase" if density_diff < 0 else "decrease"
                severity = min(1.0, (density_mag - 0.15) * 2.0)

                if direction == "increase":
                    msg = (
                        f"Arrangement feels sparse overall — "
                        f"adding layers or fills in the quieter sections "
                        f"would bring it closer to a full {genre_display} production"
                    )
                else:
                    msg = (
                        f"Arrangement is denser than typical for {genre_display} — "
                        f"creating more contrast between sections (drops, breaks) "
                        f"can make the big moments hit harder"
                    )

                gaps.append(GapItem(
                    category="arrangement",
                    dimension="density",
                    current_value=round(avg_density, 3),
                    target_value=round(target_density, 3),
                    gap_magnitude=round(density_mag, 3),
                    direction=direction,
                    severity=round(severity, 3),
                    message=msg,
                ))

        return gaps

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_readiness_score(
        self,
        gaps: Sequence[GapItem],
        bp: GenreBlueprint,
    ) -> float:
        """Compute the 0-100 production readiness score."""
        score = 100.0

        for gap in gaps:
            weight = bp.scoring_weights.get(gap.category, 0.2)
            penalty = gap.severity * gap.gap_magnitude * weight * 100.0
            score -= penalty

        return max(0.0, min(100.0, score))

    def _compute_genre_coherence(self, mp: MixProfile) -> float:
        """How internally consistent is the genre classification?

        Returns a 0-1 score.  Higher = more confident single-genre identity.
        """
        probs = mp.style.cluster_probabilities
        if not probs:
            return 0.5  # unknown — neutral

        top_prob = max(probs.values()) if probs else 0.0

        if top_prob > 0.6:
            # Strong single-genre identity
            return min(1.0, 0.7 + top_prob * 0.3)
        elif top_prob > 0.4:
            # Moderate coherence
            return 0.4 + (top_prob - 0.4) * 1.5
        else:
            # Very mixed signals
            return max(0.1, top_prob)

    def _compute_chart_potential(
        self,
        readiness: float,
        coherence: float,
        bp: GenreBlueprint,
    ) -> float:
        """Estimate chart potential as a 0-100 score.

        chart_potential = readiness * coherence * chart_hit_rate
        (scaled so a perfect production in a chart-friendly genre ≈ 85-95)
        """
        raw = (readiness / 100.0) * coherence * bp.chart_hit_rate
        # Scale to 0-100 with diminishing returns at the top
        scaled = raw * 130.0  # 1.0 * 1.0 * 0.7 * 130 ≈ 91
        return max(0.0, min(100.0, scaled))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for empty input."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _estimate_density_target(bp: GenreBlueprint) -> float:
    """Rough density target from the number of expected active roles."""
    # Count roles with target confidence > 0.5 as "expected active"
    active = sum(1 for v in bp.required_roles.values() if v > 0.5)
    total = max(len(bp.required_roles), 1)
    return active / total
