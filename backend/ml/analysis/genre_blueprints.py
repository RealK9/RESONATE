"""
Genre Blueprints — chart-readiness knowledge base for the Gap Analysis Engine.

Defines what a commercially viable, chart-ready track needs per genre cluster.
Each GenreBlueprint encodes spectral targets, required instrument roles, dynamics
expectations, perceptual profiles, and scoring weights derived from production
knowledge and the reference profiles in reference_profiles.py.

All 14 style clusters are covered.  Cluster names match reference_profiles.py
and style_classifier.py exactly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BAND_NAMES: list[str] = [
    "sub", "bass", "low_mid", "mid", "upper_mid",
    "presence", "brilliance", "air", "ultra_high", "ceiling",
]

ROLE_NAMES: list[str] = [
    "kick", "snare_clap", "hats_tops", "bass", "lead",
    "chord_support", "pad", "vocal_texture", "fx_transitions", "ambience",
]

PERCEPTUAL_KEYS: list[str] = [
    "brightness", "warmth", "air", "punch", "body",
    "bite", "smoothness", "width", "depth",
]

NUM_BANDS: int = len(BAND_NAMES)
NUM_ROLES: int = len(ROLE_NAMES)
NUM_PERCEPTUAL: int = len(PERCEPTUAL_KEYS)


# ---------------------------------------------------------------------------
# RPM genre label -> cluster name mapping
# ---------------------------------------------------------------------------

RPM_GENRE_TO_CLUSTER: dict[str, str] = {
    # EDM / Electronic
    "EDM": "2010s_edm_drop",
    "edm": "2010s_edm_drop",
    "Big Room": "2010s_edm_drop",
    "Festival EDM": "2010s_edm_drop",
    "Electro House": "2010s_edm_drop",
    "Future Bass": "2010s_edm_drop",
    # Melodic House
    "Melodic House": "2020s_melodic_house",
    "Progressive House": "2020s_melodic_house",
    "Organic House": "2020s_melodic_house",
    "Deep House": "2020s_melodic_house",
    # Pop (2000s chorus style)
    "Pop": "2000s_pop_chorus",
    "Pop Rock": "2000s_pop_chorus",
    "Dance Pop": "2000s_pop_chorus",
    "Electro Pop": "2000s_pop_chorus",
    # Boom Bap
    "Boom Bap": "1990s_boom_bap",
    "Classic Hip Hop": "1990s_boom_bap",
    "East Coast Hip Hop": "1990s_boom_bap",
    "90s Hip Hop": "1990s_boom_bap",
    # Trap
    "Trap": "modern_trap",
    "trap": "modern_trap",
    "Hip Hop": "modern_trap",
    "Southern Trap": "modern_trap",
    # Drill
    "Drill": "modern_drill",
    "UK Drill": "modern_drill",
    "NY Drill": "modern_drill",
    "Chicago Drill": "modern_drill",
    # Melodic Techno
    "Melodic Techno": "melodic_techno",
    "Techno": "melodic_techno",
    "Peak Time Techno": "melodic_techno",
    # Afro House
    "Afro House": "afro_house",
    "Afrobeats": "afro_house",
    "Amapiano": "afro_house",
    # Cinematic
    "Cinematic": "cinematic",
    "Film Score": "cinematic",
    "Orchestral": "cinematic",
    "Trailer": "cinematic",
    "Epic": "cinematic",
    # Lo-fi
    "Lo-fi": "lo_fi_chill",
    "Lo-Fi": "lo_fi_chill",
    "Lofi": "lo_fi_chill",
    "Chillhop": "lo_fi_chill",
    "Chill": "lo_fi_chill",
    # DnB
    "DnB": "dnb",
    "Drum and Bass": "dnb",
    "Drum & Bass": "dnb",
    "Liquid DnB": "dnb",
    "Jungle": "dnb",
    "Neurofunk": "dnb",
    # Ambient
    "Ambient": "ambient",
    "Downtempo": "ambient",
    "Drone": "ambient",
    "Atmospheric": "ambient",
    "New Age": "ambient",
    # R&B
    "R&B": "r_and_b",
    "RnB": "r_and_b",
    "Neo Soul": "r_and_b",
    "Soul": "r_and_b",
    "Contemporary R&B": "r_and_b",
    # Pop Production (modern)
    "Pop Production": "pop_production",
    "Modern Pop": "pop_production",
    "Indie Pop": "pop_production",
    "Synth Pop": "pop_production",
    "Alt Pop": "pop_production",
}


# ---------------------------------------------------------------------------
# GenreBlueprint dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenreBlueprint:
    """Complete chart-readiness specification for a single genre cluster.

    Every field is populated for all 14 clusters. The gap analysis engine
    compares an analyzed MixProfile against these targets to produce
    actionable, genre-aware feedback.
    """

    name: str                                     # cluster name (matches reference_profiles.py)
    display_name: str                             # human-friendly label

    # Required roles (role -> minimum confidence for "present")
    required_roles: dict[str, float]              # must have these to be chart-ready
    optional_roles: dict[str, float]              # nice-to-have

    # Spectral targets (10-band, same ordering as mix_analyzer)
    target_spectral: list[float]                  # from reference_profiles._DEFAULT_CONFIGS
    spectral_tolerance: list[float]               # per-band acceptable deviation

    # Tempo
    bpm_range: tuple[float, float]
    bpm_ideal: float

    # Dynamics
    target_lufs: float                            # target integrated loudness
    lufs_tolerance: float                         # acceptable deviation (+/-)
    target_dynamic_range: tuple[float, float]     # (min_dr, max_dr) in dB

    # Perceptual profile
    target_perceptual: dict[str, float]           # brightness, warmth, etc. all 0-1
    perceptual_tolerance: dict[str, float]        # per-attribute acceptable deviation

    # Chart intelligence
    chart_hit_rate: float                         # how often this genre hits charts (0-1)
    typical_instrument_count: tuple[int, int]     # (min, max) active roles

    # Scoring weights per category (must sum to 1.0)
    scoring_weights: dict[str, float]             # spectral, role, dynamics, perceptual, arrangement


# ---------------------------------------------------------------------------
# Blueprint definitions — all 14 clusters
# ---------------------------------------------------------------------------

_BLUEPRINTS: dict[str, GenreBlueprint] = {}


def _register(bp: GenreBlueprint) -> None:
    """Register a blueprint in the module-level registry."""
    _BLUEPRINTS[bp.name] = bp


# ---- Modern Trap -----------------------------------------------------------
# Heavy 808 sub, sparse mids, crisp hi-hats, narrow bass imaging.
# Roles matter most for gap scoring — you need the core rhythmic bones.

_register(GenreBlueprint(
    name="modern_trap",
    display_name="Modern Trap",
    required_roles={
        "kick": 0.70, "snare_clap": 0.70, "hats_tops": 0.70,
        "bass": 0.70,
    },
    optional_roles={
        "lead": 0.40, "vocal_texture": 0.40, "fx_transitions": 0.40,
    },
    target_spectral=[0.80, 0.70, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.50, 0.35],
    spectral_tolerance=[0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.09, 0.09, 0.10, 0.12],
    bpm_range=(130.0, 170.0),
    bpm_ideal=145.0,
    target_lufs=-8.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(4.0, 8.0),
    target_perceptual={
        "brightness": 0.55, "warmth": 0.45, "air": 0.55,
        "punch": 0.80, "body": 0.70, "bite": 0.60,
        "smoothness": 0.30, "width": 0.50, "depth": 0.45,
    },
    perceptual_tolerance={
        "brightness": 0.12, "warmth": 0.12, "air": 0.12,
        "punch": 0.10, "body": 0.10, "bite": 0.12,
        "smoothness": 0.15, "width": 0.12, "depth": 0.12,
    },
    chart_hit_rate=0.7,
    typical_instrument_count=(4, 7),
    scoring_weights={
        "spectral": 0.20, "role": 0.35, "dynamics": 0.15,
        "perceptual": 0.15, "arrangement": 0.15,
    },
))


# ---- Modern Drill ----------------------------------------------------------
# Similar to trap but denser mids, more aggressive percussion, sliding 808s.

_register(GenreBlueprint(
    name="modern_drill",
    display_name="Modern Drill",
    required_roles={
        "kick": 0.70, "snare_clap": 0.70, "hats_tops": 0.70,
        "bass": 0.70,
    },
    optional_roles={
        "lead": 0.40, "vocal_texture": 0.40, "fx_transitions": 0.40,
    },
    target_spectral=[0.70, 0.65, 0.35, 0.40, 0.45, 0.50, 0.50, 0.50, 0.45, 0.30],
    spectral_tolerance=[0.11, 0.10, 0.10, 0.09, 0.09, 0.09, 0.09, 0.10, 0.10, 0.12],
    bpm_range=(138.0, 148.0),
    bpm_ideal=142.0,
    target_lufs=-8.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(4.0, 8.0),
    target_perceptual={
        "brightness": 0.50, "warmth": 0.47, "air": 0.50,
        "punch": 0.75, "body": 0.65, "bite": 0.65,
        "smoothness": 0.25, "width": 0.45, "depth": 0.40,
    },
    perceptual_tolerance={
        "brightness": 0.12, "warmth": 0.12, "air": 0.12,
        "punch": 0.10, "body": 0.10, "bite": 0.12,
        "smoothness": 0.15, "width": 0.12, "depth": 0.12,
    },
    chart_hit_rate=0.5,
    typical_instrument_count=(4, 7),
    scoring_weights={
        "spectral": 0.20, "role": 0.35, "dynamics": 0.15,
        "perceptual": 0.15, "arrangement": 0.15,
    },
))


# ---- 2010s EDM Drop --------------------------------------------------------
# Punchy sub, aggressive transients, wide stereo, big build-drop arc.

_register(GenreBlueprint(
    name="2010s_edm_drop",
    display_name="2010s EDM Drop",
    required_roles={
        "kick": 0.70, "snare_clap": 0.70, "hats_tops": 0.70,
        "bass": 0.70, "lead": 0.70, "fx_transitions": 0.70,
    },
    optional_roles={
        "chord_support": 0.40, "pad": 0.40,
    },
    target_spectral=[0.70, 0.75, 0.40, 0.45, 0.50, 0.55, 0.55, 0.50, 0.45, 0.30],
    spectral_tolerance=[0.12, 0.10, 0.10, 0.09, 0.09, 0.09, 0.10, 0.10, 0.10, 0.12],
    bpm_range=(126.0, 132.0),
    bpm_ideal=128.0,
    target_lufs=-8.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(4.0, 9.0),
    target_perceptual={
        "brightness": 0.65, "warmth": 0.40, "air": 0.55,
        "punch": 0.85, "body": 0.70, "bite": 0.70,
        "smoothness": 0.25, "width": 0.65, "depth": 0.50,
    },
    perceptual_tolerance={
        "brightness": 0.10, "warmth": 0.12, "air": 0.12,
        "punch": 0.08, "body": 0.10, "bite": 0.10,
        "smoothness": 0.15, "width": 0.10, "depth": 0.12,
    },
    chart_hit_rate=0.6,
    typical_instrument_count=(5, 8),
    scoring_weights={
        "spectral": 0.25, "role": 0.25, "dynamics": 0.20,
        "perceptual": 0.15, "arrangement": 0.15,
    },
))


# ---- 2020s Melodic House ---------------------------------------------------
# Warm low end, lush mids, wide stereo field, progressive energy build.

_register(GenreBlueprint(
    name="2020s_melodic_house",
    display_name="2020s Melodic House",
    required_roles={
        "kick": 0.70, "hats_tops": 0.70, "bass": 0.70,
        "chord_support": 0.70, "pad": 0.70,
    },
    optional_roles={
        "lead": 0.40, "vocal_texture": 0.40, "fx_transitions": 0.40,
        "ambience": 0.40,
    },
    target_spectral=[0.55, 0.60, 0.45, 0.50, 0.50, 0.50, 0.45, 0.40, 0.35, 0.20],
    spectral_tolerance=[0.10, 0.09, 0.09, 0.08, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10],
    bpm_range=(120.0, 128.0),
    bpm_ideal=124.0,
    target_lufs=-10.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(6.0, 12.0),
    target_perceptual={
        "brightness": 0.45, "warmth": 0.65, "air": 0.50,
        "punch": 0.55, "body": 0.65, "bite": 0.35,
        "smoothness": 0.60, "width": 0.60, "depth": 0.65,
    },
    perceptual_tolerance={
        "brightness": 0.12, "warmth": 0.10, "air": 0.12,
        "punch": 0.12, "body": 0.10, "bite": 0.12,
        "smoothness": 0.10, "width": 0.10, "depth": 0.10,
    },
    chart_hit_rate=0.4,
    typical_instrument_count=(5, 9),
    scoring_weights={
        "spectral": 0.25, "role": 0.20, "dynamics": 0.15,
        "perceptual": 0.25, "arrangement": 0.15,
    },
))


# ---- 2000s Pop Chorus ------------------------------------------------------
# Balanced full spectrum, clear vocal space, classic verse-chorus energy.

_register(GenreBlueprint(
    name="2000s_pop_chorus",
    display_name="2000s Pop Chorus",
    required_roles={
        "kick": 0.70, "snare_clap": 0.70, "bass": 0.70,
        "chord_support": 0.70, "vocal_texture": 0.70,
    },
    optional_roles={
        "hats_tops": 0.40, "lead": 0.40, "pad": 0.40,
    },
    target_spectral=[0.35, 0.45, 0.50, 0.55, 0.55, 0.55, 0.50, 0.45, 0.35, 0.20],
    spectral_tolerance=[0.09, 0.08, 0.08, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09, 0.10],
    bpm_range=(110.0, 135.0),
    bpm_ideal=120.0,
    target_lufs=-10.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(5.0, 10.0),
    target_perceptual={
        "brightness": 0.55, "warmth": 0.55, "air": 0.50,
        "punch": 0.60, "body": 0.60, "bite": 0.45,
        "smoothness": 0.55, "width": 0.55, "depth": 0.50,
    },
    perceptual_tolerance={
        "brightness": 0.10, "warmth": 0.10, "air": 0.10,
        "punch": 0.10, "body": 0.10, "bite": 0.12,
        "smoothness": 0.10, "width": 0.10, "depth": 0.10,
    },
    chart_hit_rate=0.8,
    typical_instrument_count=(5, 8),
    scoring_weights={
        "spectral": 0.20, "role": 0.20, "dynamics": 0.20,
        "perceptual": 0.20, "arrangement": 0.20,
    },
))


# ---- 1990s Boom Bap --------------------------------------------------------
# Warm mids, moderate sub, dusty tops, sample-driven aesthetic.

_register(GenreBlueprint(
    name="1990s_boom_bap",
    display_name="1990s Boom Bap",
    required_roles={
        "kick": 0.70, "snare_clap": 0.70, "hats_tops": 0.70,
        "bass": 0.70, "vocal_texture": 0.70,
    },
    optional_roles={
        "chord_support": 0.40, "lead": 0.40,
    },
    target_spectral=[0.45, 0.55, 0.50, 0.55, 0.45, 0.40, 0.35, 0.25, 0.15, 0.10],
    spectral_tolerance=[0.10, 0.09, 0.09, 0.08, 0.09, 0.09, 0.10, 0.10, 0.10, 0.10],
    bpm_range=(85.0, 100.0),
    bpm_ideal=92.0,
    target_lufs=-12.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(8.0, 14.0),
    target_perceptual={
        "brightness": 0.30, "warmth": 0.70, "air": 0.20,
        "punch": 0.75, "body": 0.70, "bite": 0.50,
        "smoothness": 0.40, "width": 0.30, "depth": 0.35,
    },
    perceptual_tolerance={
        "brightness": 0.12, "warmth": 0.10, "air": 0.12,
        "punch": 0.10, "body": 0.10, "bite": 0.12,
        "smoothness": 0.12, "width": 0.12, "depth": 0.12,
    },
    chart_hit_rate=0.3,
    typical_instrument_count=(4, 6),
    scoring_weights={
        "spectral": 0.20, "role": 0.30, "dynamics": 0.20,
        "perceptual": 0.15, "arrangement": 0.15,
    },
))


# ---- Melodic Techno --------------------------------------------------------
# Driving kick, moderate sub, atmospheric highs, wide stereo,
# progressive build, moderate harmonic density.

_register(GenreBlueprint(
    name="melodic_techno",
    display_name="Melodic Techno",
    required_roles={
        "kick": 0.70, "hats_tops": 0.70, "bass": 0.70,
        "pad": 0.70, "fx_transitions": 0.70,
    },
    optional_roles={
        "lead": 0.40, "chord_support": 0.40, "ambience": 0.40,
        "snare_clap": 0.40,
    },
    target_spectral=[0.50, 0.60, 0.45, 0.45, 0.50, 0.55, 0.50, 0.45, 0.35, 0.25],
    spectral_tolerance=[0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10],
    bpm_range=(120.0, 130.0),
    bpm_ideal=125.0,
    target_lufs=-10.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(6.0, 12.0),
    target_perceptual={
        "brightness": 0.50, "warmth": 0.55, "air": 0.55,
        "punch": 0.65, "body": 0.60, "bite": 0.40,
        "smoothness": 0.55, "width": 0.60, "depth": 0.70,
    },
    perceptual_tolerance={
        "brightness": 0.10, "warmth": 0.10, "air": 0.10,
        "punch": 0.10, "body": 0.10, "bite": 0.12,
        "smoothness": 0.10, "width": 0.10, "depth": 0.10,
    },
    chart_hit_rate=0.3,
    typical_instrument_count=(5, 9),
    scoring_weights={
        "spectral": 0.25, "role": 0.20, "dynamics": 0.15,
        "perceptual": 0.25, "arrangement": 0.15,
    },
))


# ---- Afro House ------------------------------------------------------------
# Round low end, warm presence, percussive mids, organic textures.

_register(GenreBlueprint(
    name="afro_house",
    display_name="Afro House",
    required_roles={
        "kick": 0.70, "snare_clap": 0.70, "hats_tops": 0.70,
        "bass": 0.70,
    },
    optional_roles={
        "lead": 0.40, "chord_support": 0.40, "pad": 0.40,
        "vocal_texture": 0.40, "fx_transitions": 0.40,
    },
    target_spectral=[0.50, 0.55, 0.50, 0.50, 0.50, 0.50, 0.45, 0.40, 0.30, 0.20],
    spectral_tolerance=[0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10],
    bpm_range=(118.0, 126.0),
    bpm_ideal=122.0,
    target_lufs=-10.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(6.0, 11.0),
    target_perceptual={
        "brightness": 0.45, "warmth": 0.65, "air": 0.40,
        "punch": 0.65, "body": 0.65, "bite": 0.40,
        "smoothness": 0.55, "width": 0.50, "depth": 0.50,
    },
    perceptual_tolerance={
        "brightness": 0.12, "warmth": 0.10, "air": 0.12,
        "punch": 0.10, "body": 0.10, "bite": 0.12,
        "smoothness": 0.10, "width": 0.10, "depth": 0.10,
    },
    chart_hit_rate=0.3,
    typical_instrument_count=(5, 8),
    scoring_weights={
        "spectral": 0.20, "role": 0.25, "dynamics": 0.15,
        "perceptual": 0.20, "arrangement": 0.20,
    },
))


# ---- Cinematic -------------------------------------------------------------
# Wide, deep low end, atmospheric highs, dynamic range, rich harmonic
# layering, strong energy arcs.

_register(GenreBlueprint(
    name="cinematic",
    display_name="Cinematic",
    required_roles={
        "bass": 0.70, "chord_support": 0.70, "pad": 0.70,
        "fx_transitions": 0.70, "ambience": 0.70,
    },
    optional_roles={
        "lead": 0.40, "vocal_texture": 0.40, "kick": 0.40,
    },
    target_spectral=[0.55, 0.50, 0.50, 0.50, 0.45, 0.45, 0.50, 0.55, 0.50, 0.40],
    spectral_tolerance=[0.12, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09, 0.10, 0.10, 0.12],
    bpm_range=(80.0, 140.0),
    bpm_ideal=110.0,
    target_lufs=-14.0,
    lufs_tolerance=3.0,
    target_dynamic_range=(10.0, 20.0),
    target_perceptual={
        "brightness": 0.45, "warmth": 0.60, "air": 0.70,
        "punch": 0.40, "body": 0.65, "bite": 0.25,
        "smoothness": 0.65, "width": 0.70, "depth": 0.80,
    },
    perceptual_tolerance={
        "brightness": 0.12, "warmth": 0.10, "air": 0.10,
        "punch": 0.15, "body": 0.10, "bite": 0.15,
        "smoothness": 0.10, "width": 0.10, "depth": 0.08,
    },
    chart_hit_rate=0.2,
    typical_instrument_count=(4, 8),
    scoring_weights={
        "spectral": 0.25, "role": 0.15, "dynamics": 0.20,
        "perceptual": 0.25, "arrangement": 0.15,
    },
))


# ---- Lo-fi Chill -----------------------------------------------------------
# Rolled-off highs, warm mids, low transient density, narrow stereo,
# gentle energy, dusty character.

_register(GenreBlueprint(
    name="lo_fi_chill",
    display_name="Lo-fi Chill",
    required_roles={
        "bass": 0.70, "chord_support": 0.70,
    },
    optional_roles={
        "kick": 0.40, "snare_clap": 0.40, "hats_tops": 0.40,
        "lead": 0.40, "pad": 0.40, "vocal_texture": 0.40,
        "ambience": 0.40,
    },
    target_spectral=[0.30, 0.40, 0.50, 0.55, 0.45, 0.35, 0.25, 0.15, 0.08, 0.05],
    spectral_tolerance=[0.09, 0.08, 0.08, 0.07, 0.08, 0.08, 0.09, 0.09, 0.08, 0.07],
    bpm_range=(70.0, 90.0),
    bpm_ideal=80.0,
    target_lufs=-14.0,
    lufs_tolerance=3.0,
    target_dynamic_range=(8.0, 16.0),
    target_perceptual={
        "brightness": 0.20, "warmth": 0.75, "air": 0.15,
        "punch": 0.35, "body": 0.65, "bite": 0.15,
        "smoothness": 0.80, "width": 0.35, "depth": 0.50,
    },
    perceptual_tolerance={
        "brightness": 0.10, "warmth": 0.08, "air": 0.10,
        "punch": 0.12, "body": 0.10, "bite": 0.10,
        "smoothness": 0.08, "width": 0.12, "depth": 0.10,
    },
    chart_hit_rate=0.2,
    typical_instrument_count=(3, 7),
    scoring_weights={
        "spectral": 0.25, "role": 0.15, "dynamics": 0.15,
        "perceptual": 0.30, "arrangement": 0.15,
    },
))


# ---- DnB ------------------------------------------------------------------
# Powerful sub, crisp highs, full mids, fast and dense, wide stereo,
# high transient energy.

_register(GenreBlueprint(
    name="dnb",
    display_name="Drum & Bass",
    required_roles={
        "kick": 0.70, "snare_clap": 0.70, "hats_tops": 0.70,
        "bass": 0.70,
    },
    optional_roles={
        "lead": 0.40, "chord_support": 0.40, "pad": 0.40,
        "fx_transitions": 0.40, "vocal_texture": 0.40, "ambience": 0.40,
    },
    target_spectral=[0.65, 0.60, 0.45, 0.50, 0.50, 0.55, 0.55, 0.50, 0.45, 0.30],
    spectral_tolerance=[0.10, 0.09, 0.09, 0.08, 0.08, 0.08, 0.09, 0.09, 0.10, 0.11],
    bpm_range=(170.0, 180.0),
    bpm_ideal=174.0,
    target_lufs=-8.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(5.0, 10.0),
    target_perceptual={
        "brightness": 0.60, "warmth": 0.40, "air": 0.55,
        "punch": 0.80, "body": 0.65, "bite": 0.65,
        "smoothness": 0.25, "width": 0.55, "depth": 0.45,
    },
    perceptual_tolerance={
        "brightness": 0.10, "warmth": 0.12, "air": 0.10,
        "punch": 0.08, "body": 0.10, "bite": 0.10,
        "smoothness": 0.15, "width": 0.10, "depth": 0.12,
    },
    chart_hit_rate=0.2,
    typical_instrument_count=(4, 8),
    scoring_weights={
        "spectral": 0.20, "role": 0.30, "dynamics": 0.20,
        "perceptual": 0.15, "arrangement": 0.15,
    },
))


# ---- Ambient ---------------------------------------------------------------
# Gentle, diffuse, no hard edges, wide stereo, very low density,
# rich harmonic texture, minimal percussion.

_register(GenreBlueprint(
    name="ambient",
    display_name="Ambient",
    required_roles={
        "pad": 0.70, "ambience": 0.70,
    },
    optional_roles={
        "lead": 0.40, "chord_support": 0.40, "bass": 0.40,
        "vocal_texture": 0.40, "fx_transitions": 0.40,
    },
    target_spectral=[0.25, 0.30, 0.40, 0.45, 0.40, 0.40, 0.45, 0.50, 0.40, 0.30],
    spectral_tolerance=[0.10, 0.09, 0.09, 0.08, 0.09, 0.09, 0.09, 0.09, 0.10, 0.10],
    bpm_range=(60.0, 120.0),
    bpm_ideal=90.0,
    target_lufs=-18.0,
    lufs_tolerance=4.0,
    target_dynamic_range=(12.0, 24.0),
    target_perceptual={
        "brightness": 0.35, "warmth": 0.60, "air": 0.75,
        "punch": 0.10, "body": 0.50, "bite": 0.08,
        "smoothness": 0.85, "width": 0.65, "depth": 0.85,
    },
    perceptual_tolerance={
        "brightness": 0.12, "warmth": 0.10, "air": 0.08,
        "punch": 0.10, "body": 0.12, "bite": 0.10,
        "smoothness": 0.08, "width": 0.10, "depth": 0.08,
    },
    chart_hit_rate=0.1,
    typical_instrument_count=(2, 6),
    scoring_weights={
        "spectral": 0.35, "role": 0.10, "dynamics": 0.15,
        "perceptual": 0.30, "arrangement": 0.10,
    },
))


# ---- R&B -------------------------------------------------------------------
# Warm low-mids, smooth presence, vocal-centric, moderate width,
# rich chords, groovy rhythm.

_register(GenreBlueprint(
    name="r_and_b",
    display_name="R&B",
    required_roles={
        "kick": 0.70, "snare_clap": 0.70, "bass": 0.70,
        "chord_support": 0.70, "vocal_texture": 0.70,
    },
    optional_roles={
        "hats_tops": 0.40, "lead": 0.40, "pad": 0.40,
        "ambience": 0.40,
    },
    target_spectral=[0.40, 0.50, 0.50, 0.55, 0.50, 0.50, 0.40, 0.35, 0.25, 0.15],
    spectral_tolerance=[0.09, 0.08, 0.08, 0.07, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09],
    bpm_range=(65.0, 100.0),
    bpm_ideal=82.0,
    target_lufs=-10.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(6.0, 12.0),
    target_perceptual={
        "brightness": 0.40, "warmth": 0.70, "air": 0.35,
        "punch": 0.50, "body": 0.65, "bite": 0.30,
        "smoothness": 0.75, "width": 0.50, "depth": 0.55,
    },
    perceptual_tolerance={
        "brightness": 0.10, "warmth": 0.08, "air": 0.12,
        "punch": 0.12, "body": 0.10, "bite": 0.12,
        "smoothness": 0.08, "width": 0.10, "depth": 0.10,
    },
    chart_hit_rate=0.6,
    typical_instrument_count=(5, 8),
    scoring_weights={
        "spectral": 0.20, "role": 0.20, "dynamics": 0.15,
        "perceptual": 0.25, "arrangement": 0.20,
    },
))


# ---- Pop Production --------------------------------------------------------
# Balanced, polished, full spectrum, clear vocal space, moderate
# everything, well-structured energy arc.

_register(GenreBlueprint(
    name="pop_production",
    display_name="Pop Production",
    required_roles={
        "kick": 0.70, "snare_clap": 0.70, "bass": 0.70,
        "chord_support": 0.70, "vocal_texture": 0.70,
    },
    optional_roles={
        "hats_tops": 0.40, "lead": 0.40, "pad": 0.40,
        "fx_transitions": 0.40,
    },
    target_spectral=[0.35, 0.45, 0.50, 0.55, 0.55, 0.55, 0.50, 0.45, 0.35, 0.20],
    spectral_tolerance=[0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09],
    bpm_range=(100.0, 130.0),
    bpm_ideal=118.0,
    target_lufs=-10.0,
    lufs_tolerance=2.0,
    target_dynamic_range=(5.0, 10.0),
    target_perceptual={
        "brightness": 0.55, "warmth": 0.55, "air": 0.50,
        "punch": 0.60, "body": 0.60, "bite": 0.45,
        "smoothness": 0.55, "width": 0.55, "depth": 0.50,
    },
    perceptual_tolerance={
        "brightness": 0.10, "warmth": 0.10, "air": 0.10,
        "punch": 0.10, "body": 0.10, "bite": 0.10,
        "smoothness": 0.10, "width": 0.10, "depth": 0.10,
    },
    chart_hit_rate=0.7,
    typical_instrument_count=(5, 8),
    scoring_weights={
        "spectral": 0.20, "role": 0.20, "dynamics": 0.20,
        "perceptual": 0.20, "arrangement": 0.20,
    },
))


# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------

def get_blueprint(cluster_name: str) -> Optional[GenreBlueprint]:
    """Return the GenreBlueprint for a cluster name, or ``None`` if unknown.

    Parameters
    ----------
    cluster_name:
        Must match one of the 14 canonical cluster names exactly
        (e.g. ``"modern_trap"``, ``"2020s_melodic_house"``).
    """
    return _BLUEPRINTS.get(cluster_name)


def get_best_blueprint(
    cluster_probabilities: dict[str, float],
) -> tuple[GenreBlueprint, float]:
    """Select the best-matching blueprint given a probability distribution.

    Parameters
    ----------
    cluster_probabilities:
        Mapping of cluster name to probability / confidence (0-1).
        Typically comes from the style classifier output.

    Returns
    -------
    tuple[GenreBlueprint, float]
        The blueprint with the highest probability among those that have
        a registered blueprint, and the associated probability.

    Raises
    ------
    ValueError
        If no known cluster name is present in *cluster_probabilities*.
    """
    best_bp: Optional[GenreBlueprint] = None
    best_prob: float = -1.0

    for cluster, prob in cluster_probabilities.items():
        bp = _BLUEPRINTS.get(cluster)
        if bp is not None and prob > best_prob:
            best_bp = bp
            best_prob = prob

    if best_bp is None:
        known = sorted(_BLUEPRINTS.keys())
        raise ValueError(
            f"No known cluster found in probabilities. "
            f"Known clusters: {known}"
        )

    return best_bp, best_prob


def all_blueprints() -> dict[str, GenreBlueprint]:
    """Return a shallow copy of the full blueprint registry."""
    return dict(_BLUEPRINTS)
