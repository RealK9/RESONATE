"""
RESONATE — Smart Collection Curation Route.
Auto-generate themed sample kits from gap analysis and recommendations.
One-click ZIP export.
"""

import io
import json
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import state
from routes.samples import find_sample_file

router = APIRouter()

# ---------------------------------------------------------------------------
# Genre label helper — map cluster names to human-friendly display names
# ---------------------------------------------------------------------------

GENRE_LABELS = {
    "modern_trap": "Trap",
    "modern_drill": "Drill",
    "2010s_edm_drop": "EDM",
    "2020s_melodic_house": "Melodic House",
    "2000s_pop_chorus": "Pop",
    "1990s_boom_bap": "Boom Bap",
    "melodic_techno": "Techno",
    "afro_house": "Afro House",
    "cinematic": "Cinematic",
    "lo_fi_chill": "Lo-Fi",
    "dnb": "DnB",
    "ambient": "Ambient",
    "r_and_b": "R&B",
    "pop_production": "Pop Production",
}


def get_genre_label(cluster_name: str) -> str:
    """Return a human-friendly genre label for a cluster name."""
    if not cluster_name:
        return "Your"
    return GENRE_LABELS.get(cluster_name, cluster_name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Collection generation helpers
# ---------------------------------------------------------------------------

def _extract_genre(mix_profile) -> str:
    """Extract the primary genre/cluster from the mix profile."""
    if mix_profile is None:
        return ""
    if isinstance(mix_profile, dict):
        style = mix_profile.get("style", {})
        return style.get("primary_cluster", "") if isinstance(style, dict) else ""
    return getattr(getattr(mix_profile, "style", None), "primary_cluster", "")


def _extract_missing_roles(gap_result) -> list[str]:
    """Extract missing roles from the gap analysis result."""
    if gap_result is None:
        return []
    if isinstance(gap_result, dict):
        return gap_result.get("missing_roles", []) or []
    roles = getattr(gap_result, "missing_roles", None)
    return list(roles) if roles else []


def _get_recommendations_list() -> list[dict]:
    """Get the raw recommendation list from state, as dicts."""
    rec_result = state.latest_recommendations
    if rec_result is None:
        return []
    # RecommendationResult dataclass
    if hasattr(rec_result, "recommendations"):
        recs = rec_result.recommendations
        result = []
        for r in recs:
            if hasattr(r, "__dict__"):
                d = {}
                d["filepath"] = getattr(r, "filepath", "")
                d["filename"] = getattr(r, "filename", "")
                d["score"] = float(getattr(r, "score", 0))
                d["policy"] = getattr(r, "policy", "")
                d["role"] = getattr(r, "role", "")
                d["explanation"] = getattr(r, "explanation", "")
                d["need_addressed"] = getattr(r, "need_addressed", "")
                result.append(d)
            elif isinstance(r, dict):
                result.append(r)
        return result
    # Already a dict (from to_dict())
    if isinstance(rec_result, dict):
        return rec_result.get("recommendations", [])
    return []


def _sample_entry(rec: dict) -> dict:
    """Build a collection sample entry from a recommendation dict."""
    filepath = rec.get("filepath", "")
    filename = rec.get("filename", "")
    name = filename or Path(filepath).stem if filepath else "Unknown"
    # Clean display name
    name = name.replace("_", " ").replace("-", " ").strip()
    return {
        "filepath": filepath,
        "name": name,
        "role": rec.get("role", ""),
        "score": round(rec.get("score", 0), 3),
        "explanation": rec.get("explanation", ""),
    }


def _total_impact(samples: list[dict]) -> float:
    """Sum of scores, rounded."""
    return round(sum(s.get("score", 0) for s in samples), 2)


def generate_collections() -> list[dict]:
    """Generate themed collections from current recommendations and gap analysis."""
    recs = _get_recommendations_list()
    if not recs:
        return []

    genre_cluster = _extract_genre(state.latest_mix_profile)
    genre_label = get_genre_label(genre_cluster)
    missing_roles = _extract_missing_roles(state.latest_gap_result)

    collections = []

    # 1. "{Genre} Essentials" — fill_missing_role samples (top 10)
    fill_recs = [r for r in recs if r.get("policy") == "fill_missing_role"]
    if fill_recs:
        samples = [_sample_entry(r) for r in fill_recs[:10]]
        desc_parts = []
        if missing_roles:
            role_names = [r.replace("_", " ").title() for r in missing_roles[:3]]
            desc_parts.append(f"Fills gaps: {', '.join(role_names)}")
        else:
            desc_parts.append("Samples that fill the missing roles in your mix")
        collections.append({
            "id": "essentials",
            "name": f"{genre_label} Essentials",
            "description": desc_parts[0],
            "icon": "target",
            "samples": samples,
            "total_impact": _total_impact(samples),
        })

    # 2. "Polish Pack" — improve_polish + enhance_lift (top 8)
    polish_recs = [r for r in recs if r.get("policy") in ("improve_polish", "enhance_lift")]
    if polish_recs:
        samples = [_sample_entry(r) for r in polish_recs[:8]]
        collections.append({
            "id": "polish",
            "name": "Polish Pack",
            "description": "Commercial polish and emotional lift for your mix",
            "icon": "sparkle",
            "samples": samples,
            "total_impact": _total_impact(samples),
        })

    # 3. "Groove Kit" — enhance_groove + add_movement (top 8)
    groove_recs = [r for r in recs if r.get("policy") in ("enhance_groove", "add_movement")]
    if groove_recs:
        samples = [_sample_entry(r) for r in groove_recs[:8]]
        collections.append({
            "id": "groove",
            "name": "Groove Kit",
            "description": "Rhythmic elements and movement to keep things evolving",
            "icon": "rhythm",
            "samples": samples,
            "total_impact": _total_impact(samples),
        })

    # 4. "Top 10 Picks" — top 10 by score regardless of policy
    top_recs = sorted(recs, key=lambda r: r.get("score", 0), reverse=True)[:10]
    if top_recs:
        samples = [_sample_entry(r) for r in top_recs]
        collections.append({
            "id": "top-picks",
            "name": "Top 10 Picks",
            "description": "The highest-scoring samples across all categories",
            "icon": "crown",
            "samples": samples,
            "total_impact": _total_impact(samples),
        })

    return collections


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/collections/generate")
async def get_collections():
    """Generate themed sample collections from current analysis state."""
    if state.latest_recommendations is None:
        raise HTTPException(status_code=404, detail="No recommendations available. Analyze a track first.")
    if state.latest_mix_profile is None:
        raise HTTPException(status_code=404, detail="No mix profile available. Analyze a track first.")

    collections = generate_collections()
    return {"collections": collections}


@router.post("/collections/export/{collection_id}")
async def export_collection(collection_id: str):
    """Export a collection as a ZIP file with metadata."""
    if state.latest_recommendations is None:
        raise HTTPException(status_code=404, detail="No recommendations available")

    collections = generate_collections()
    target = None
    for c in collections:
        if c["id"] == collection_id:
            target = c
            break

    if target is None:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_id}' not found")

    buf = io.BytesIO()
    found = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sample in target["samples"]:
            fp = find_sample_file(sample["filepath"])
            if fp and fp.exists():
                zf.write(fp, arcname=fp.name)
                found += 1

        # Include metadata JSON
        metadata = {
            "collection": target["name"],
            "description": target["description"],
            "total_impact": target["total_impact"],
            "samples": [
                {
                    "filename": Path(s["filepath"]).name if s["filepath"] else s["name"],
                    "role": s["role"],
                    "score": s["score"],
                    "explanation": s["explanation"],
                }
                for s in target["samples"]
            ],
        }
        zf.writestr("_resonate_collection.json", json.dumps(metadata, indent=2))

    if found == 0:
        raise HTTPException(status_code=404, detail="No sample files found for this collection")

    buf.seek(0)
    safe_name = target["name"].replace(" ", "-").lower()
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="RESONATE-{safe_name}.zip"'},
    )
