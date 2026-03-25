"""
Reference profiles — builds and stores style-aware reference priors.

Two classes:
  ReferenceProfileBuilder  — aggregates analyzed MixProfiles by cluster and
                              computes averaged StylePriors.
  DefaultPriors            — ships hand-tuned priors for all 14 style clusters
                              so the system works without any reference tracks.
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev

from ml.models.mix_profile import MixProfile
from ml.models.reference_profile import ReferenceCorpus, StylePrior


# ---------------------------------------------------------------------------
# Canonical cluster names (must match style_classifier.py)
# ---------------------------------------------------------------------------

ALL_CLUSTERS: list[str] = [
    "2010s_edm_drop",
    "2020s_melodic_house",
    "2000s_pop_chorus",
    "1990s_boom_bap",
    "modern_trap",
    "modern_drill",
    "melodic_techno",
    "afro_house",
    "cinematic",
    "lo_fi_chill",
    "dnb",
    "ambient",
    "r_and_b",
    "pop_production",
]

# The 10 canonical sound roles (same ordering as SourceRolePresence docs)
_ALL_ROLES: list[str] = [
    "kick", "snare_clap", "hats_tops", "bass", "lead",
    "chord_support", "pad", "vocal_texture", "fx_transitions", "ambience",
]

NUM_BANDS = 10
NUM_SECTIONS = 8  # energy-arc segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return mean(values)


def _safe_stdev(values: list[float], fallback: float = 0.05) -> float:
    if len(values) < 2:
        return fallback
    return stdev(values)


# ---------------------------------------------------------------------------
# ReferenceProfileBuilder
# ---------------------------------------------------------------------------

class ReferenceProfileBuilder:
    """Accumulates analyzed reference MixProfiles and builds a ReferenceCorpus."""

    def __init__(self) -> None:
        self._profiles_by_cluster: dict[str, list[MixProfile]] = defaultdict(list)

    def add_reference(
        self,
        mix_profile: MixProfile,
        cluster_override: str | None = None,
    ) -> None:
        """Add an analyzed reference track to the corpus.

        Uses *cluster_override* if given, otherwise the profile's own
        ``style.primary_cluster``.
        """
        cluster = cluster_override or mix_profile.style.primary_cluster
        if not cluster:
            return  # cannot file without a cluster label
        self._profiles_by_cluster[cluster].append(mix_profile)

    @property
    def cluster_names(self) -> list[str]:
        return list(self._profiles_by_cluster.keys())

    @property
    def total_references(self) -> int:
        return sum(len(v) for v in self._profiles_by_cluster.values())

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_corpus(self) -> ReferenceCorpus:
        """Compute a StylePrior for each cluster that has at least one reference."""
        priors: dict[str, StylePrior] = {}

        for cluster, profiles in self._profiles_by_cluster.items():
            priors[cluster] = self._build_prior(cluster, profiles)

        return ReferenceCorpus(
            priors=priors,
            total_references=self.total_references,
        )

    # ------------------------------------------------------------------
    # Internal: per-cluster aggregation
    # ------------------------------------------------------------------

    def _build_prior(self, cluster: str, profiles: list[MixProfile]) -> StylePrior:
        n = len(profiles)

        # --- Spectral means & stds ---
        band_columns: list[list[float]] = [[] for _ in range(NUM_BANDS)]
        for p in profiles:
            band_vals = p.spectral_occupancy.mean_by_band
            if len(band_vals) >= NUM_BANDS:
                for i in range(NUM_BANDS):
                    band_columns[i].append(band_vals[i])

        spectral_mean = [_safe_mean(col) for col in band_columns]
        spectral_std = [_safe_stdev(col) for col in band_columns]

        # --- Density ---
        density_means: list[float] = []
        for p in profiles:
            if p.density_map:
                density_means.append(_safe_mean(p.density_map))
        density_mean = _safe_mean(density_means)
        if len(density_means) >= 2:
            lo = min(density_means)
            hi = max(density_means)
        else:
            lo, hi = max(0.0, density_mean - 0.1), min(1.0, density_mean + 0.1)

        # --- Role co-occurrence ---
        role_accum: dict[str, list[float]] = defaultdict(list)
        for p in profiles:
            for role, conf in p.source_roles.roles.items():
                role_accum[role].append(conf)
        typical_roles = {role: _safe_mean(vals) for role, vals in role_accum.items()}

        # --- Width ---
        width_columns: list[list[float]] = [[] for _ in range(NUM_BANDS)]
        overall_widths: list[float] = []
        for p in profiles:
            wbb = p.stereo_width.width_by_band
            if len(wbb) >= NUM_BANDS:
                for i in range(NUM_BANDS):
                    width_columns[i].append(wbb[i])
            overall_widths.append(p.stereo_width.overall_width)

        width_by_band = [_safe_mean(col) for col in width_columns]
        overall_width = _safe_mean(overall_widths)

        # --- Section energy arc ---
        section_columns: list[list[float]] = [[] for _ in range(NUM_SECTIONS)]
        for p in profiles:
            sec = p.analysis.section_energy
            if len(sec) >= NUM_SECTIONS:
                for i in range(NUM_SECTIONS):
                    section_columns[i].append(sec[i])
        section_lift = [_safe_mean(col) for col in section_columns]

        # --- Tonal complexity / layering depth ---
        harmonic_densities = [p.analysis.harmonic_density for p in profiles]
        tonal_complexity = _safe_mean(harmonic_densities)

        active_role_counts: list[float] = []
        for p in profiles:
            count = sum(1.0 for c in p.source_roles.roles.values() if c > 0.3)
            active_role_counts.append(count / max(len(_ALL_ROLES), 1))
        layering_depth = _safe_mean(active_role_counts)

        # --- Complementary roles (roles with *low* average presence) ---
        complement_threshold = 0.25
        common_complements = [
            role for role in _ALL_ROLES
            if typical_roles.get(role, 0.0) < complement_threshold
        ]

        # --- Confidence (more references = higher confidence, cap at 1.0) ---
        confidence = min(1.0, n / 20.0)

        # --- Arrangement density range ---
        all_densities: list[float] = []
        for p in profiles:
            all_densities.extend(p.density_map)
        if all_densities:
            arr_lo = min(all_densities)
            arr_hi = max(all_densities)
        else:
            arr_lo, arr_hi = 0.0, 1.0

        return StylePrior(
            cluster_name=cluster,
            target_spectral_mean=spectral_mean,
            target_spectral_std=spectral_std,
            target_density_mean=density_mean,
            target_density_range=(lo, hi),
            typical_roles=typical_roles,
            target_width_by_band=width_by_band,
            target_overall_width=overall_width,
            section_lift_pattern=section_lift,
            arrangement_density_range=(arr_lo, arr_hi),
            tonal_complexity=tonal_complexity,
            layering_depth=layering_depth,
            common_complements=common_complements,
            reference_count=n,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# DefaultPriors  (hand-tuned, ships with the app)
# ---------------------------------------------------------------------------

class DefaultPriors:
    """Ships hand-tuned StylePriors for all 14 style clusters.

    These are built from production knowledge rather than analyzed references.
    They let the needs engine work out of the box, before users have added
    any reference tracks of their own.
    """

    @staticmethod
    def get_corpus() -> ReferenceCorpus:
        """Return hand-tuned priors for all 14 style clusters."""
        priors: dict[str, StylePrior] = {}

        for cluster, cfg in _DEFAULT_CONFIGS.items():
            priors[cluster] = StylePrior(
                cluster_name=cluster,
                target_spectral_mean=cfg["spectral_mean"],
                target_spectral_std=cfg["spectral_std"],
                target_density_mean=cfg["density_mean"],
                target_density_range=cfg["density_range"],
                typical_roles=cfg["typical_roles"],
                target_width_by_band=cfg["width_by_band"],
                target_overall_width=cfg["overall_width"],
                section_lift_pattern=cfg["section_lift"],
                arrangement_density_range=cfg["arrangement_density_range"],
                tonal_complexity=cfg["tonal_complexity"],
                layering_depth=cfg["layering_depth"],
                common_complements=cfg["common_complements"],
                reference_count=0,
                confidence=0.6,  # hand-tuned = moderate confidence
            )

        return ReferenceCorpus(
            priors=priors,
            total_references=0,
        )


# ---------------------------------------------------------------------------
# Hand-tuned default configurations
# ---------------------------------------------------------------------------
# Band order: sub, bass, low_mid, mid, upper_mid, presence, brilliance,
#             air, ultra_high, ceiling
#
# Width order matches the same 10 bands.
#
# Section-lift pattern: 8 segments representing the expected energy arc from
# start to end (normalized 0-1).  Most genres follow an intro-build-peak-outro
# shape with genre-specific variations.
#
# Roles: kick, snare_clap, hats_tops, bass, lead, chord_support, pad,
#        vocal_texture, fx_transitions, ambience
# ---------------------------------------------------------------------------

_DEFAULT_CONFIGS: dict[str, dict] = {

    # -----------------------------------------------------------------------
    # 2010s EDM Drop — punchy sub, aggressive transients, wide stereo,
    # big build-drop energy arc
    # -----------------------------------------------------------------------
    "2010s_edm_drop": {
        "spectral_mean": [0.70, 0.75, 0.40, 0.45, 0.50, 0.55, 0.55, 0.50, 0.45, 0.30],
        "spectral_std":  [0.10, 0.08, 0.08, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.10],
        "density_mean": 0.60,
        "density_range": (0.35, 0.85),
        "typical_roles": {
            "kick": 0.90, "snare_clap": 0.85, "hats_tops": 0.70,
            "bass": 0.90, "lead": 0.80, "chord_support": 0.50,
            "pad": 0.40, "vocal_texture": 0.35, "fx_transitions": 0.75,
            "ambience": 0.30,
        },
        "width_by_band": [0.15, 0.20, 0.35, 0.50, 0.60, 0.70, 0.75, 0.80, 0.75, 0.65],
        "overall_width": 0.65,
        "section_lift":  [0.30, 0.50, 0.70, 0.90, 1.00, 0.85, 0.95, 0.50],
        "arrangement_density_range": (0.25, 0.90),
        "tonal_complexity": 0.35,
        "layering_depth": 0.65,
        "common_complements": ["pad", "vocal_texture", "ambience"],
    },

    # -----------------------------------------------------------------------
    # 2020s Melodic House — warm low end, lush mids, wide stereo field,
    # progressive energy build
    # -----------------------------------------------------------------------
    "2020s_melodic_house": {
        "spectral_mean": [0.55, 0.60, 0.45, 0.50, 0.50, 0.50, 0.45, 0.40, 0.35, 0.20],
        "spectral_std":  [0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08],
        "density_mean": 0.55,
        "density_range": (0.35, 0.75),
        "typical_roles": {
            "kick": 0.90, "snare_clap": 0.60, "hats_tops": 0.70,
            "bass": 0.85, "lead": 0.65, "chord_support": 0.70,
            "pad": 0.75, "vocal_texture": 0.50, "fx_transitions": 0.60,
            "ambience": 0.55,
        },
        "width_by_band": [0.10, 0.15, 0.35, 0.50, 0.55, 0.65, 0.70, 0.75, 0.70, 0.60],
        "overall_width": 0.60,
        "section_lift":  [0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 0.80, 0.45],
        "arrangement_density_range": (0.25, 0.80),
        "tonal_complexity": 0.55,
        "layering_depth": 0.70,
        "common_complements": ["vocal_texture", "fx_transitions"],
    },

    # -----------------------------------------------------------------------
    # 2000s Pop Chorus — balanced full spectrum, clear vocal space,
    # classic verse-chorus energy
    # -----------------------------------------------------------------------
    "2000s_pop_chorus": {
        "spectral_mean": [0.35, 0.45, 0.50, 0.55, 0.55, 0.55, 0.50, 0.45, 0.35, 0.20],
        "spectral_std":  [0.07, 0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08],
        "density_mean": 0.55,
        "density_range": (0.30, 0.80),
        "typical_roles": {
            "kick": 0.80, "snare_clap": 0.80, "hats_tops": 0.65,
            "bass": 0.80, "lead": 0.50, "chord_support": 0.75,
            "pad": 0.55, "vocal_texture": 0.85, "fx_transitions": 0.40,
            "ambience": 0.35,
        },
        "width_by_band": [0.10, 0.15, 0.30, 0.45, 0.55, 0.60, 0.65, 0.60, 0.55, 0.45],
        "overall_width": 0.55,
        "section_lift":  [0.35, 0.50, 0.75, 1.00, 0.50, 0.75, 1.00, 0.40],
        "arrangement_density_range": (0.25, 0.85),
        "tonal_complexity": 0.50,
        "layering_depth": 0.65,
        "common_complements": ["fx_transitions", "ambience"],
    },

    # -----------------------------------------------------------------------
    # 1990s Boom Bap — warm mids, moderate sub, dusty tops, sample-driven
    # -----------------------------------------------------------------------
    "1990s_boom_bap": {
        "spectral_mean": [0.45, 0.55, 0.50, 0.55, 0.45, 0.40, 0.35, 0.25, 0.15, 0.10],
        "spectral_std":  [0.08, 0.07, 0.07, 0.06, 0.07, 0.07, 0.08, 0.08, 0.08, 0.08],
        "density_mean": 0.50,
        "density_range": (0.30, 0.70),
        "typical_roles": {
            "kick": 0.90, "snare_clap": 0.90, "hats_tops": 0.70,
            "bass": 0.80, "lead": 0.40, "chord_support": 0.55,
            "pad": 0.30, "vocal_texture": 0.70, "fx_transitions": 0.25,
            "ambience": 0.20,
        },
        "width_by_band": [0.10, 0.10, 0.20, 0.30, 0.35, 0.40, 0.35, 0.30, 0.25, 0.20],
        "overall_width": 0.30,
        "section_lift":  [0.40, 0.55, 0.65, 0.70, 0.70, 0.65, 0.70, 0.45],
        "arrangement_density_range": (0.30, 0.75),
        "tonal_complexity": 0.25,
        "layering_depth": 0.45,
        "common_complements": ["pad", "fx_transitions", "ambience"],
    },

    # -----------------------------------------------------------------------
    # Modern Trap — heavy sub (808), sparse mids, crisp hats, narrow bass,
    # wide hi-hats, strong kick/snare presence
    # -----------------------------------------------------------------------
    "modern_trap": {
        "spectral_mean": [0.80, 0.70, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.50, 0.35],
        "spectral_std":  [0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.08, 0.10],
        "density_mean": 0.45,
        "density_range": (0.20, 0.70),
        "typical_roles": {
            "kick": 0.90, "snare_clap": 0.85, "hats_tops": 0.90,
            "bass": 0.95, "lead": 0.55, "chord_support": 0.30,
            "pad": 0.35, "vocal_texture": 0.65, "fx_transitions": 0.50,
            "ambience": 0.25,
        },
        "width_by_band": [0.05, 0.08, 0.20, 0.35, 0.50, 0.60, 0.70, 0.75, 0.70, 0.55],
        "overall_width": 0.50,
        "section_lift":  [0.35, 0.50, 0.65, 0.80, 0.90, 1.00, 0.85, 0.45],
        "arrangement_density_range": (0.15, 0.75),
        "tonal_complexity": 0.30,
        "layering_depth": 0.50,
        "common_complements": ["chord_support", "pad", "ambience"],
    },

    # -----------------------------------------------------------------------
    # Modern Drill — similar to trap but denser mids, more aggressive
    # percussion patterns, sliding 808s
    # -----------------------------------------------------------------------
    "modern_drill": {
        "spectral_mean": [0.70, 0.65, 0.35, 0.40, 0.45, 0.50, 0.50, 0.50, 0.45, 0.30],
        "spectral_std":  [0.09, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.10],
        "density_mean": 0.50,
        "density_range": (0.25, 0.75),
        "typical_roles": {
            "kick": 0.85, "snare_clap": 0.80, "hats_tops": 0.85,
            "bass": 0.90, "lead": 0.50, "chord_support": 0.35,
            "pad": 0.30, "vocal_texture": 0.60, "fx_transitions": 0.45,
            "ambience": 0.25,
        },
        "width_by_band": [0.05, 0.10, 0.25, 0.35, 0.45, 0.55, 0.65, 0.70, 0.65, 0.50],
        "overall_width": 0.45,
        "section_lift":  [0.40, 0.55, 0.70, 0.85, 0.95, 1.00, 0.90, 0.50],
        "arrangement_density_range": (0.20, 0.80),
        "tonal_complexity": 0.25,
        "layering_depth": 0.50,
        "common_complements": ["chord_support", "pad", "ambience"],
    },

    # -----------------------------------------------------------------------
    # Melodic Techno — driving kick, moderate sub, atmospheric highs,
    # wide stereo, progressive build, moderate harmonic density
    # -----------------------------------------------------------------------
    "melodic_techno": {
        "spectral_mean": [0.50, 0.60, 0.45, 0.45, 0.50, 0.55, 0.50, 0.45, 0.35, 0.25],
        "spectral_std":  [0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08],
        "density_mean": 0.58,
        "density_range": (0.35, 0.80),
        "typical_roles": {
            "kick": 0.95, "snare_clap": 0.55, "hats_tops": 0.80,
            "bass": 0.85, "lead": 0.60, "chord_support": 0.65,
            "pad": 0.80, "vocal_texture": 0.40, "fx_transitions": 0.70,
            "ambience": 0.65,
        },
        "width_by_band": [0.10, 0.12, 0.30, 0.50, 0.60, 0.70, 0.75, 0.80, 0.75, 0.65],
        "overall_width": 0.60,
        "section_lift":  [0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 0.85, 0.40],
        "arrangement_density_range": (0.25, 0.85),
        "tonal_complexity": 0.50,
        "layering_depth": 0.70,
        "common_complements": ["vocal_texture"],
    },

    # -----------------------------------------------------------------------
    # Afro House — round low end, warm presence, percussive mids,
    # organic textures, moderate width
    # -----------------------------------------------------------------------
    "afro_house": {
        "spectral_mean": [0.50, 0.55, 0.50, 0.50, 0.50, 0.50, 0.45, 0.40, 0.30, 0.20],
        "spectral_std":  [0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08],
        "density_mean": 0.55,
        "density_range": (0.35, 0.75),
        "typical_roles": {
            "kick": 0.90, "snare_clap": 0.70, "hats_tops": 0.75,
            "bass": 0.85, "lead": 0.55, "chord_support": 0.60,
            "pad": 0.55, "vocal_texture": 0.65, "fx_transitions": 0.45,
            "ambience": 0.40,
        },
        "width_by_band": [0.10, 0.12, 0.30, 0.45, 0.50, 0.55, 0.60, 0.60, 0.55, 0.45],
        "overall_width": 0.50,
        "section_lift":  [0.30, 0.45, 0.60, 0.75, 0.85, 1.00, 0.80, 0.45],
        "arrangement_density_range": (0.30, 0.80),
        "tonal_complexity": 0.45,
        "layering_depth": 0.60,
        "common_complements": ["fx_transitions", "ambience"],
    },

    # -----------------------------------------------------------------------
    # Cinematic — wide, deep low end, atmospheric highs, dynamic range,
    # rich harmonic layering, strong energy arcs
    # -----------------------------------------------------------------------
    "cinematic": {
        "spectral_mean": [0.55, 0.50, 0.50, 0.50, 0.45, 0.45, 0.50, 0.55, 0.50, 0.40],
        "spectral_std":  [0.10, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.10],
        "density_mean": 0.45,
        "density_range": (0.15, 0.80),
        "typical_roles": {
            "kick": 0.45, "snare_clap": 0.35, "hats_tops": 0.25,
            "bass": 0.70, "lead": 0.65, "chord_support": 0.80,
            "pad": 0.90, "vocal_texture": 0.50, "fx_transitions": 0.70,
            "ambience": 0.85,
        },
        "width_by_band": [0.15, 0.20, 0.40, 0.55, 0.65, 0.75, 0.80, 0.85, 0.80, 0.70],
        "overall_width": 0.70,
        "section_lift":  [0.15, 0.25, 0.45, 0.70, 1.00, 0.80, 0.60, 0.25],
        "arrangement_density_range": (0.10, 0.85),
        "tonal_complexity": 0.60,
        "layering_depth": 0.75,
        "common_complements": ["hats_tops"],
    },

    # -----------------------------------------------------------------------
    # Lo-fi Chill — rolled-off highs, warm mids, low transient density,
    # narrow stereo, gentle energy, dusty character
    # -----------------------------------------------------------------------
    "lo_fi_chill": {
        "spectral_mean": [0.30, 0.40, 0.50, 0.55, 0.45, 0.35, 0.25, 0.15, 0.08, 0.05],
        "spectral_std":  [0.07, 0.06, 0.06, 0.05, 0.06, 0.06, 0.07, 0.07, 0.06, 0.05],
        "density_mean": 0.35,
        "density_range": (0.20, 0.50),
        "typical_roles": {
            "kick": 0.65, "snare_clap": 0.55, "hats_tops": 0.50,
            "bass": 0.70, "lead": 0.50, "chord_support": 0.75,
            "pad": 0.60, "vocal_texture": 0.45, "fx_transitions": 0.25,
            "ambience": 0.55,
        },
        "width_by_band": [0.08, 0.10, 0.20, 0.30, 0.35, 0.40, 0.40, 0.35, 0.30, 0.25],
        "overall_width": 0.35,
        "section_lift":  [0.40, 0.50, 0.55, 0.60, 0.60, 0.55, 0.55, 0.45],
        "arrangement_density_range": (0.15, 0.55),
        "tonal_complexity": 0.40,
        "layering_depth": 0.50,
        "common_complements": ["fx_transitions"],
    },

    # -----------------------------------------------------------------------
    # DnB — powerful sub, crisp highs, full mids, fast and dense,
    # wide stereo, high transient energy
    # -----------------------------------------------------------------------
    "dnb": {
        "spectral_mean": [0.65, 0.60, 0.45, 0.50, 0.50, 0.55, 0.55, 0.50, 0.45, 0.30],
        "spectral_std":  [0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.07, 0.07, 0.08, 0.09],
        "density_mean": 0.65,
        "density_range": (0.40, 0.90),
        "typical_roles": {
            "kick": 0.85, "snare_clap": 0.90, "hats_tops": 0.80,
            "bass": 0.90, "lead": 0.55, "chord_support": 0.45,
            "pad": 0.50, "vocal_texture": 0.40, "fx_transitions": 0.60,
            "ambience": 0.45,
        },
        "width_by_band": [0.10, 0.12, 0.30, 0.45, 0.55, 0.65, 0.70, 0.70, 0.65, 0.55],
        "overall_width": 0.55,
        "section_lift":  [0.30, 0.50, 0.70, 0.90, 1.00, 0.90, 0.95, 0.50],
        "arrangement_density_range": (0.30, 0.95),
        "tonal_complexity": 0.30,
        "layering_depth": 0.60,
        "common_complements": ["chord_support", "vocal_texture"],
    },

    # -----------------------------------------------------------------------
    # Ambient — gentle, diffuse, no hard edges, wide stereo, very low
    # density, rich harmonic texture, minimal percussion
    # -----------------------------------------------------------------------
    "ambient": {
        "spectral_mean": [0.25, 0.30, 0.40, 0.45, 0.40, 0.40, 0.45, 0.50, 0.40, 0.30],
        "spectral_std":  [0.08, 0.07, 0.07, 0.06, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08],
        "density_mean": 0.25,
        "density_range": (0.08, 0.45),
        "typical_roles": {
            "kick": 0.10, "snare_clap": 0.08, "hats_tops": 0.10,
            "bass": 0.35, "lead": 0.40, "chord_support": 0.55,
            "pad": 0.90, "vocal_texture": 0.35, "fx_transitions": 0.50,
            "ambience": 0.90,
        },
        "width_by_band": [0.15, 0.20, 0.40, 0.55, 0.65, 0.75, 0.80, 0.85, 0.80, 0.70],
        "overall_width": 0.65,
        "section_lift":  [0.30, 0.40, 0.50, 0.60, 0.65, 0.60, 0.50, 0.35],
        "arrangement_density_range": (0.05, 0.50),
        "tonal_complexity": 0.55,
        "layering_depth": 0.45,
        "common_complements": ["kick", "snare_clap", "hats_tops"],
    },

    # -----------------------------------------------------------------------
    # R&B — warm low-mids, smooth presence, vocal-centric,
    # moderate width, rich chords, groovy rhythm
    # -----------------------------------------------------------------------
    "r_and_b": {
        "spectral_mean": [0.40, 0.50, 0.50, 0.55, 0.50, 0.50, 0.40, 0.35, 0.25, 0.15],
        "spectral_std":  [0.07, 0.06, 0.06, 0.05, 0.06, 0.06, 0.07, 0.07, 0.07, 0.07],
        "density_mean": 0.48,
        "density_range": (0.25, 0.70),
        "typical_roles": {
            "kick": 0.75, "snare_clap": 0.70, "hats_tops": 0.65,
            "bass": 0.80, "lead": 0.45, "chord_support": 0.80,
            "pad": 0.65, "vocal_texture": 0.90, "fx_transitions": 0.35,
            "ambience": 0.40,
        },
        "width_by_band": [0.08, 0.10, 0.25, 0.40, 0.50, 0.55, 0.55, 0.50, 0.45, 0.35],
        "overall_width": 0.50,
        "section_lift":  [0.35, 0.50, 0.65, 0.80, 0.75, 0.85, 0.80, 0.45],
        "arrangement_density_range": (0.20, 0.75),
        "tonal_complexity": 0.50,
        "layering_depth": 0.60,
        "common_complements": ["fx_transitions", "ambience"],
    },

    # -----------------------------------------------------------------------
    # Pop Production — balanced, polished, full spectrum, clear vocal space,
    # moderate everything, well-structured energy arc
    # -----------------------------------------------------------------------
    "pop_production": {
        "spectral_mean": [0.35, 0.45, 0.50, 0.55, 0.55, 0.55, 0.50, 0.45, 0.35, 0.20],
        "spectral_std":  [0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07],
        "density_mean": 0.52,
        "density_range": (0.30, 0.75),
        "typical_roles": {
            "kick": 0.80, "snare_clap": 0.80, "hats_tops": 0.65,
            "bass": 0.80, "lead": 0.55, "chord_support": 0.70,
            "pad": 0.50, "vocal_texture": 0.85, "fx_transitions": 0.45,
            "ambience": 0.35,
        },
        "width_by_band": [0.10, 0.12, 0.30, 0.45, 0.55, 0.60, 0.65, 0.60, 0.55, 0.45],
        "overall_width": 0.55,
        "section_lift":  [0.30, 0.50, 0.70, 1.00, 0.50, 0.70, 1.00, 0.40],
        "arrangement_density_range": (0.25, 0.80),
        "tonal_complexity": 0.45,
        "layering_depth": 0.60,
        "common_complements": ["fx_transitions", "ambience"],
    },
}
