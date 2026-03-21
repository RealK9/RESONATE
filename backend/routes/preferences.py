"""
RESONATE — Preference Learning Route.
Builds a taste vector from user ratings to subtly boost matching samples.
"""

import numpy as np
from fastapi import APIRouter

from db.database import get_top_rated_samples
from indexer import sample_cache

router = APIRouter()


@router.get("/preferences/taste-vector")
async def get_taste_vector():
    """
    Build a taste vector from highly-rated samples.
    Returns the average MFCC profile and spectral characteristics
    of the user's top-rated samples — used to boost similar samples in scoring.
    """
    top = get_top_rated_samples(limit=100)
    if not top:
        return {"has_taste": False, "sample_count": 0}

    mfcc_profiles = []
    band_profiles = []
    centroid_vals = []
    type_counts = {}

    for item in top:
        filepath = item["sample_filepath"]
        sa = sample_cache.get(filepath, {})
        if not sa:
            continue

        avg_rating = item.get("avg_rating", 3)
        weight = avg_rating / 5.0  # weight by rating strength

        if sa.get("mfcc_profile"):
            mfcc_profiles.append((sa["mfcc_profile"], weight))

        if sa.get("frequency_bands"):
            band_profiles.append((sa["frequency_bands"], weight))

        if sa.get("spectral_centroid"):
            centroid_vals.append((sa["spectral_centroid"], weight))

        stype = sa.get("sample_type", "unknown")
        type_counts[stype] = type_counts.get(stype, 0) + weight

    # Weighted average MFCC profile
    taste_mfcc = None
    if mfcc_profiles:
        total_weight = sum(w for _, w in mfcc_profiles)
        weighted = [np.array(m) * w for m, w in mfcc_profiles]
        taste_mfcc = (sum(weighted) / total_weight).tolist()

    # Weighted average frequency bands
    taste_bands = None
    if band_profiles:
        total_weight = sum(w for _, w in band_profiles)
        all_keys = set()
        for b, _ in band_profiles:
            all_keys.update(b.keys())
        taste_bands = {}
        for key in all_keys:
            weighted_sum = sum(b.get(key, 0) * w for b, w in band_profiles)
            taste_bands[key] = weighted_sum / total_weight

    # Weighted average centroid
    taste_centroid = None
    if centroid_vals:
        total_weight = sum(w for _, w in centroid_vals)
        taste_centroid = sum(c * w for c, w in centroid_vals) / total_weight

    # Preferred types
    preferred_types = sorted(type_counts.items(), key=lambda x: -x[1])

    return {
        "has_taste": True,
        "sample_count": len(top),
        "taste_mfcc": taste_mfcc,
        "taste_bands": taste_bands,
        "taste_centroid": taste_centroid,
        "preferred_types": [{"type": t, "weight": round(w, 2)} for t, w in preferred_types],
    }
