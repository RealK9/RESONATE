"""
RESONATE — Auto-Layering Suggestions.
Given a sample, find complementary samples that layer well with it.
"""

import numpy as np
from pathlib import Path
from fastapi import APIRouter, HTTPException

import config
from indexer import sample_cache
from utils import clean_name, TYPE_LABELS

router = APIRouter()

# Layering rules: for each sample type, which types complement it
LAYER_RULES = {
    "kick": ["bass", "sub_bass", "kick", "fx"],
    "snare": ["clap", "percussion", "fx", "hihat"],
    "hihat": ["percussion", "hihat", "fx"],
    "bass": ["kick", "sub_bass", "pad"],
    "melody": ["pad", "strings", "vocals", "melody"],
    "pad": ["melody", "strings", "vocals"],
    "vocals": ["melody", "pad", "strings"],
    "strings": ["melody", "pad", "vocals"],
    "percussion": ["hihat", "snare", "fx"],
    "fx": ["melody", "pad", "vocals"],
}

# Frequency band complementarity — which bands should the layer fill
BAND_KEYS = [
    "sub_bass_20_80", "bass_80_250", "low_mid_250_500", "mid_500_2k",
    "upper_mid_2k_6k", "presence_6k_12k", "air_12k_20k",
]


def _frequency_complement_score(src_bands, layer_bands):
    """Score how well layer fills gaps in source's frequency spectrum."""
    if not src_bands or not layer_bands:
        return 50

    score = 50
    for band in BAND_KEYS:
        src_val = src_bands.get(band, 0)
        layer_val = layer_bands.get(band, 0)
        # Layer is strong where source is weak = good
        if src_val < 0.03 and layer_val > 0.05:
            score += 10
        # Layer overlaps heavily = stacking risk
        elif src_val > 0.15 and layer_val > 0.15:
            score -= 5
    return max(0, min(100, score))


@router.get("/samples/layers/{sample_id:path}")
async def find_layers(sample_id: str, limit: int = 15):
    """Find samples that would layer well with the given sample."""
    source = sample_cache.get(sample_id)
    if not source:
        raise HTTPException(status_code=404, detail="Sample not found")

    src_type = source.get("sample_type", "unknown")
    src_bands = source.get("frequency_bands", {})
    src_centroid = source.get("spectral_centroid", 0)
    src_key = source.get("key", "N/A")

    complement_types = LAYER_RULES.get(src_type, list(LAYER_RULES.keys()))

    results = []
    for cache_key, sa in sample_cache.items():
        if cache_key == sample_id or not sa.get("_hash"):
            continue

        sa_type = sa.get("sample_type", "unknown")

        # Type complementarity (40%)
        if sa_type in complement_types:
            type_score = 85
        elif sa_type == src_type:
            type_score = 40  # Same type can layer but less ideal
        else:
            type_score = 20

        # Frequency complementarity (35%)
        freq_score = _frequency_complement_score(src_bands, sa.get("frequency_bands", {}))

        # Spectral separation (15%) — different brightness = cleaner layer
        sa_centroid = sa.get("spectral_centroid", 0)
        if src_centroid > 0 and sa_centroid > 0:
            ratio = min(src_centroid, sa_centroid) / max(src_centroid, sa_centroid)
            # Lower ratio = more separation = better layering
            sep_score = (1 - ratio) * 100
        else:
            sep_score = 50

        # Key compatibility (10%)
        sa_key = sa.get("key", "N/A")
        if src_key in ("N/A", "—") or sa_key in ("N/A", "—"):
            key_score = 70  # Unknown = neutral
        elif src_key == sa_key:
            key_score = 100  # Same key = perfect
        elif src_key.replace("m", "") == sa_key.replace("m", ""):
            key_score = 80  # Same root, diff mode
        else:
            key_score = 40

        layer_score = (
            type_score * 0.40 +
            freq_score * 0.35 +
            sep_score * 0.15 +
            key_score * 0.10
        )

        results.append({
            "id": cache_key,
            "layer_score": round(layer_score, 1),
            "sample_type": sa_type,
            "layer_reason": _get_layer_reason(src_type, sa_type, src_centroid, sa_centroid, src_bands, sa.get("frequency_bands", {})),
        })

    results.sort(key=lambda x: -x["layer_score"])
    results = results[:limit]

    # Enrich with display data
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
            "layer_score": r["layer_score"],
            "layer_reason": r["layer_reason"],
            "duration": sa.get("duration", 0),
            "bpm": sa.get("bpm", 0),
            "key": sa.get("key", "N/A"),
            "sample_type": r["sample_type"],
            "type_label": TYPE_LABELS.get(r["sample_type"], "Sound"),
            "frequency_bands": sa.get("frequency_bands", {}),
        })

    return {"layers": enriched, "source_id": sample_id, "total": len(enriched)}


def _get_layer_reason(src_type, layer_type, src_c, layer_c, src_bands, layer_bands):
    """Generate a human-readable reason for the layering suggestion."""
    reasons = []
    if layer_type in LAYER_RULES.get(src_type, []):
        reasons.append(f"{TYPE_LABELS.get(layer_type, layer_type)} complements {TYPE_LABELS.get(src_type, src_type)}")

    if src_c > 0 and layer_c > 0:
        if layer_c > src_c * 1.5:
            reasons.append("adds high-end")
        elif layer_c < src_c * 0.6:
            reasons.append("adds low-end weight")

    if src_bands and layer_bands:
        for band in BAND_KEYS:
            if src_bands.get(band, 0) < 0.03 and layer_bands.get(band, 0) > 0.05:
                nice = band.split("_")[0]
                reasons.append(f"fills {nice} gap")
                break

    return " · ".join(reasons[:2]) if reasons else "frequency complement"
