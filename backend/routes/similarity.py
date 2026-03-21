"""
RESONATE — Similarity Search Routes.
Find samples sonically similar to a given sample using MFCC + spectral data.
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from indexer import sample_cache
from utils import clean_name, TYPE_LABELS

router = APIRouter()


def _cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _band_similarity(bands_a, bands_b):
    """Compare two frequency band profiles (7-band)."""
    if not bands_a or not bands_b:
        return 0.5
    keys = ["sub_bass_20_80", "bass_80_250", "low_mid_250_500", "mid_500_2k",
            "upper_mid_2k_6k", "presence_6k_12k", "air_12k_20k"]
    a = [bands_a.get(k, 0) for k in keys]
    b = [bands_b.get(k, 0) for k in keys]
    diff = sum(abs(x - y) for x, y in zip(a, b))
    return max(0, 1 - diff / 2)


@router.get("/samples/similar/{sample_id:path}")
async def find_similar(sample_id: str, limit: int = 20):
    """Find samples most sonically similar to the given sample."""
    source = sample_cache.get(sample_id)
    if not source:
        raise HTTPException(status_code=404, detail="Sample not found in index")

    src_mfcc = source.get("mfcc_profile", [])
    src_bands = source.get("frequency_bands", {})
    src_centroid = source.get("spectral_centroid", 0)
    src_type = source.get("sample_type", "unknown")

    results = []
    for cache_key, sa in sample_cache.items():
        if cache_key == sample_id or not sa.get("_hash"):
            continue

        # MFCC similarity (timbral — 50% weight)
        mfcc_sim = 0.5
        sa_mfcc = sa.get("mfcc_profile", [])
        if src_mfcc and sa_mfcc:
            mfcc_sim = (_cosine_similarity(src_mfcc, sa_mfcc) + 1) / 2  # normalize to 0-1

        # Frequency band similarity (spectral shape — 30% weight)
        band_sim = _band_similarity(src_bands, sa.get("frequency_bands", {}))

        # Spectral centroid closeness (brightness — 10% weight)
        sa_centroid = sa.get("spectral_centroid", 0)
        if src_centroid > 0 and sa_centroid > 0:
            ratio = min(src_centroid, sa_centroid) / max(src_centroid, sa_centroid)
            centroid_sim = ratio
        else:
            centroid_sim = 0.5

        # Type match bonus (10% weight)
        type_sim = 1.0 if sa.get("sample_type") == src_type else 0.3

        similarity = (
            mfcc_sim * 0.50 +
            band_sim * 0.30 +
            centroid_sim * 0.10 +
            type_sim * 0.10
        )

        results.append({
            "id": cache_key,
            "similarity": round(similarity * 100, 1),
            "sample_type": sa.get("sample_type", "unknown"),
        })

    results.sort(key=lambda x: -x["similarity"])
    results = results[:limit]

    # Enrich with display data
    from pathlib import Path
    import config
    enriched = []
    for r in results:
        fp = Path(r["id"])
        if not fp.exists():
            continue
        try:
            rel = fp.relative_to(config.SAMPLE_DIR)
        except ValueError:
            rel = Path(fp.name)
        parts = rel.parts
        sa = sample_cache.get(r["id"], {})
        enriched.append({
            "id": r["id"],
            "name": fp.stem,
            "clean_name": clean_name(fp.stem),
            "filename": fp.name,
            "path": str(rel).replace("#", "%23"),
            "category": parts[0] if len(parts) > 1 else "Uncategorized",
            "similarity": r["similarity"],
            "duration": sa.get("duration", 0),
            "bpm": sa.get("bpm", 0),
            "key": sa.get("key", "N/A"),
            "sample_type": r["sample_type"],
            "type_label": TYPE_LABELS.get(r["sample_type"], "Sound"),
            "frequency_bands": sa.get("frequency_bands", {}),
        })

    return {"similar": enriched, "source_id": sample_id, "total": len(enriched)}
