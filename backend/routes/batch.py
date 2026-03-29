"""
RESONATE — Batch Analysis Route.
Analyze multiple tracks and find cross-track sample recommendations.
"""

from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from typing import List

from config import HAS_CLAUDE, UPLOAD_DIR, TRANSPOSED_DIR
from analysis.track_analyzer import analyze_track
from ai.claude_engine import claude_analyze_track
from db.database import save_session
import state

router = APIRouter()


@router.post("/analyze/batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    """Analyze multiple tracks and return combined profile."""
    if len(files) > 10:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Maximum 10 tracks per batch")

    results = []
    combined_bands = {}
    combined_gaps = set()
    combined_instruments = set()

    for f in files:
        safe_name = Path(f.filename).name  # strips directory components
        dest = UPLOAD_DIR / safe_name
        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)

        track = analyze_track(str(dest))

        ai = None
        if HAS_CLAUDE:
            ai = claude_analyze_track(track)
        if not ai:
            ai = {
                "source": "heuristic",
                "genre": track.get("detected_genre", "unknown"),
                "mood": "unknown",
                "energy": "Medium",
                "what_track_has": track.get("detected_instruments", []),
                "what_track_needs": track.get("frequency_gaps", []),
                "summary": f"{track.get('detected_genre', 'unknown')} track at {track.get('bpm')} BPM",
            }

        # Save session for each track
        save_session(
            track_filename=f.filename,
            track_profile=track,
            ai_analysis=ai,
            name=f"Batch: {f.filename}",
        )

        # Accumulate frequency bands (average across tracks)
        for band, val in track.get("frequency_bands", {}).items():
            combined_bands[band] = combined_bands.get(band, 0) + val

        combined_gaps.update(track.get("frequency_gaps", []))
        combined_instruments.update(track.get("detected_instruments", []))

        results.append({
            "filename": f.filename,
            "key": track["key"],
            "bpm": track["bpm"],
            "genre": (ai or {}).get("genre", track.get("detected_genre", "unknown")),
            "mood": (ai or {}).get("mood", "unknown"),
            "energy": (ai or {}).get("energy", "Medium"),
            "summary": (ai or {}).get("summary", ""),
        })

    # Average frequency bands
    n = len(results)
    if n > 0:
        combined_bands = {k: v / n for k, v in combined_bands.items()}

    # Set combined profile as the active one for sample matching
    if results:
        last_track = analyze_track(str(UPLOAD_DIR / results[-1]["filename"]))
        last_track["frequency_bands"] = combined_bands
        last_track["frequency_gaps"] = list(combined_gaps)
        last_track["detected_instruments"] = list(combined_instruments)
        state.latest_track_profile = last_track
        state.latest_track_file = UPLOAD_DIR / results[-1]["filename"]

        # Use combined AI analysis
        state.latest_ai_analysis = {
            "genre": results[0].get("genre", "unknown"),
            "mood": results[0].get("mood", "unknown"),
            "energy": results[0].get("energy", "Medium"),
            "what_track_needs": list(combined_gaps),
            "what_track_has": list(combined_instruments),
            "summary": f"Batch analysis of {n} tracks",
        }

    # Clear transposition cache
    for fp in TRANSPOSED_DIR.glob("*"):
        try:
            fp.unlink()
        except Exception:
            pass

    return {
        "tracks": results,
        "count": len(results),
        "combined_frequency_bands": combined_bands,
        "combined_gaps": list(combined_gaps),
        "combined_instruments": list(combined_instruments),
    }
