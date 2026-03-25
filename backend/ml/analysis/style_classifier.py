"""
Style classifier — maps a MixProfile to style cluster probabilities.

Extracts a 6-dimensional feature vector from the MixProfile (BPM, spectral
centroid proxy, sub-bass ratio, transient density proxy, overall width,
harmonic density), computes weighted Euclidean distance to manually-tuned
cluster centroids, and converts distances to probabilities via softmax.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ml.models.mix_profile import MixProfile, StyleCluster


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class _FeatureVector:
    """Raw features extracted from a MixProfile."""
    bpm: float
    spectral_centroid: float  # Hz — weighted mean of band energies
    sub_bass_ratio: float     # (sub + bass) / total energy
    transient_density: float  # variance of density_map
    overall_width: float      # 0-1
    harmonic_density: float   # 0-1


# Band center frequencies for centroid calculation (geometric mean of edges)
_BAND_CENTERS_HZ: list[float] = [
    34.6,    # sub:        sqrt(20 * 60)
    94.9,    # bass:       sqrt(60 * 150)
    244.9,   # low_mid:    sqrt(150 * 400)
    632.5,   # mid:        sqrt(400 * 1000)
    1581.1,  # upper_mid:  sqrt(1000 * 2500)
    3535.5,  # presence:   sqrt(2500 * 5000)
    6324.6,  # brilliance: sqrt(5000 * 8000)
    9797.9,  # air:        sqrt(8000 * 12000)
    13856.4, # ultra_high: sqrt(12000 * 16000)
    17888.5, # ceiling:    sqrt(16000 * 20000)
]


def _extract_features(profile: MixProfile) -> _FeatureVector:
    """Build a feature vector from the MixProfile fields."""
    analysis = profile.analysis
    spec = profile.spectral_occupancy

    # --- Spectral centroid proxy ---
    mean_by_band = spec.mean_by_band if spec.mean_by_band else [0.0] * 10
    total_energy = sum(mean_by_band)
    if total_energy > 1e-12:
        centroid = sum(
            e * c for e, c in zip(mean_by_band, _BAND_CENTERS_HZ)
        ) / total_energy
    else:
        centroid = 1000.0  # neutral fallback

    # --- Sub-bass ratio ---
    if total_energy > 1e-12 and len(mean_by_band) >= 2:
        sub_bass_ratio = (mean_by_band[0] + mean_by_band[1]) / total_energy
    else:
        sub_bass_ratio = 0.2  # neutral fallback

    # --- Transient density proxy (variance of density map) ---
    dmap = profile.density_map if profile.density_map else [0.5]
    transient_density = float(np.var(dmap))

    # --- Overall width ---
    overall_width = profile.stereo_width.overall_width

    # --- Harmonic density ---
    harmonic_density = analysis.harmonic_density

    return _FeatureVector(
        bpm=analysis.bpm,
        spectral_centroid=centroid,
        sub_bass_ratio=sub_bass_ratio,
        transient_density=transient_density,
        overall_width=overall_width,
        harmonic_density=harmonic_density,
    )


# ---------------------------------------------------------------------------
# Cluster centroids  (manually tuned)
# ---------------------------------------------------------------------------

# Each centroid: (bpm, centroid_hz, sub_bass_ratio, transient_density, width, harmonic_density)
_CLUSTER_CENTROIDS: dict[str, tuple[float, float, float, float, float, float]] = {
    "2010s_edm_drop":       (128, 2500, 0.30, 0.55, 0.65, 0.35),
    "2020s_melodic_house":  (124, 3200, 0.20, 0.40, 0.60, 0.55),
    "2000s_pop_chorus":     (120, 3500, 0.15, 0.35, 0.55, 0.50),
    "1990s_boom_bap":       ( 90, 1800, 0.35, 0.50, 0.30, 0.25),
    "modern_trap":          (140, 2000, 0.35, 0.40, 0.50, 0.30),
    "modern_drill":         (140, 2200, 0.30, 0.55, 0.45, 0.25),
    "melodic_techno":       (125, 3000, 0.20, 0.60, 0.60, 0.50),
    "afro_house":           (118, 2800, 0.22, 0.45, 0.50, 0.45),
    "cinematic":            (100, 2600, 0.18, 0.30, 0.70, 0.60),
    "lo_fi_chill":          ( 80, 1500, 0.25, 0.15, 0.35, 0.40),
    "dnb":                  (174, 3200, 0.28, 0.65, 0.55, 0.30),
    "ambient":              ( 70, 2000, 0.15, 0.08, 0.65, 0.55),
    "r_and_b":              (100, 2400, 0.25, 0.30, 0.50, 0.50),
    "pop_production":       (115, 3000, 0.18, 0.35, 0.55, 0.45),
}

# Normalization ranges (min, max) for each feature dimension
_FEATURE_RANGES: list[tuple[float, float]] = [
    ( 60.0, 180.0),  # bpm
    (500.0, 6000.0), # spectral centroid Hz
    (0.0, 0.5),      # sub_bass_ratio
    (0.0, 0.8),      # transient_density
    (0.0, 1.0),      # overall_width
    (0.0, 1.0),      # harmonic_density
]

# Per-dimension importance weights
_FEATURE_WEIGHTS: list[float] = [
    1.5,  # bpm — strong genre signal
    1.0,  # centroid
    1.2,  # sub-bass ratio
    1.0,  # transient density
    0.7,  # width
    0.8,  # harmonic density
]

# Softmax temperature — lower = sharper; higher = flatter distribution
_TEMPERATURE: float = 0.6


# ---------------------------------------------------------------------------
# Era heuristic
# ---------------------------------------------------------------------------

_CLUSTER_ERAS: dict[str, str] = {
    "2010s_edm_drop":       "2010s",
    "2020s_melodic_house":  "2020s",
    "2000s_pop_chorus":     "2000s",
    "1990s_boom_bap":       "1990s",
    "modern_trap":          "2020s",
    "modern_drill":         "2020s",
    "melodic_techno":       "2020s",
    "afro_house":           "2020s",
    "cinematic":            "contemporary",
    "lo_fi_chill":          "2020s",
    "dnb":                  "contemporary",
    "ambient":              "contemporary",
    "r_and_b":              "contemporary",
    "pop_production":       "contemporary",
}


def _estimate_era(primary_cluster: str) -> str:
    """Return an era string for the primary cluster."""
    return _CLUSTER_ERAS.get(primary_cluster, "contemporary")


# ---------------------------------------------------------------------------
# Distance & softmax helpers
# ---------------------------------------------------------------------------

def _normalize(value: float, lo: float, hi: float) -> float:
    """Normalize value to 0-1 using min/max range."""
    span = hi - lo
    if span < 1e-12:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / span))


def _weighted_euclidean(a: list[float], b: list[float],
                        weights: list[float]) -> float:
    """Weighted Euclidean distance between two vectors."""
    return math.sqrt(sum(
        w * (ai - bi) ** 2 for ai, bi, w in zip(a, b, weights)
    ))


def _softmax(neg_distances: list[float], temperature: float) -> list[float]:
    """Softmax over negative distances → probabilities."""
    scaled = [d / temperature for d in neg_distances]
    max_val = max(scaled)
    exps = [math.exp(s - max_val) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StyleClassifier:
    """Classify a MixProfile into style clusters with probabilities."""

    def classify(self, mix_profile: MixProfile) -> StyleCluster:
        """
        Classify the given mix profile into style clusters.

        Returns a StyleCluster with:
        - cluster_probabilities: dict mapping each of 14 clusters to [0, 1]
        - primary_cluster: the highest-probability cluster name
        - era_estimate: decade/era string for the primary cluster
        """
        features = _extract_features(mix_profile)

        # Build normalized feature vector
        raw = [
            features.bpm,
            features.spectral_centroid,
            features.sub_bass_ratio,
            features.transient_density,
            features.overall_width,
            features.harmonic_density,
        ]
        normed = [
            _normalize(v, lo, hi)
            for v, (lo, hi) in zip(raw, _FEATURE_RANGES)
        ]

        # Compute distance to each centroid
        cluster_names: list[str] = []
        neg_distances: list[float] = []

        for name, centroid_raw in _CLUSTER_CENTROIDS.items():
            centroid_normed = [
                _normalize(v, lo, hi)
                for v, (lo, hi) in zip(centroid_raw, _FEATURE_RANGES)
            ]
            dist = _weighted_euclidean(normed, centroid_normed, _FEATURE_WEIGHTS)
            cluster_names.append(name)
            neg_distances.append(-dist)  # negate so closer = higher

        # Convert to probabilities
        probs = _softmax(neg_distances, _TEMPERATURE)

        cluster_probabilities = dict(zip(cluster_names, probs))

        # Primary cluster
        primary_cluster = max(cluster_probabilities, key=cluster_probabilities.get)  # type: ignore[arg-type]

        # Era estimate
        era_estimate = _estimate_era(primary_cluster)

        return StyleCluster(
            cluster_probabilities=cluster_probabilities,
            primary_cluster=primary_cluster,
            era_estimate=era_estimate,
        )
