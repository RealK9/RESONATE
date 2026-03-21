"""
RESONATE — Session Save/Load Routes.
Persist analysis sessions and reload them later.
"""

from fastapi import APIRouter, HTTPException, Request

from db.database import (
    save_session, get_sessions, get_session, delete_session,
    rate_sample, get_sample_ratings, get_average_rating, get_top_rated_samples,
    log_usage, get_most_used_samples, get_recently_used_samples,
    get_collections, create_collection, add_to_collection,
    remove_from_collection, get_collection_samples, delete_collection,
)
import state

router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════
# SESSION ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/sessions")
async def save_current_session(request: Request):
    """Save the current analysis session."""
    body = await request.json()
    name = body.get("name")
    track_filename = body.get("track_filename", "Unknown Track")

    if not state.latest_track_profile:
        raise HTTPException(status_code=400, detail="No active analysis to save")

    session_id = save_session(
        track_filename=track_filename,
        track_profile=state.latest_track_profile,
        ai_analysis=state.latest_ai_analysis,
        name=name,
    )
    return {"id": session_id, "status": "saved"}


@router.get("/sessions")
async def list_sessions(limit: int = 50):
    """List recent analysis sessions."""
    sessions = get_sessions(limit)
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/sessions/{session_id}")
async def load_session(session_id: int):
    """Load a specific session, restoring track profile and AI analysis."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Restore state so /samples can re-score
    state.latest_track_profile = session["track_profile"]
    state.latest_ai_analysis = session.get("ai_analysis", {})

    return {
        "id": session["id"],
        "name": session["name"],
        "track_filename": session["track_filename"],
        "track_profile": session["track_profile"],
        "ai_analysis": session.get("ai_analysis"),
        "created_at": session["created_at"],
    }


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: int):
    """Delete a session."""
    delete_session(session_id)
    return {"status": "deleted"}


# ═══════════════════════════════════════════════════════════════════════════
# RATING ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/ratings")
async def rate(request: Request):
    """Rate a sample (1-5 stars)."""
    body = await request.json()
    filepath = body.get("sample_filepath")
    rating = body.get("rating")
    session_id = body.get("session_id")
    if not filepath or not rating or not (1 <= rating <= 5):
        raise HTTPException(status_code=400, detail="Need sample_filepath and rating (1-5)")
    rate_sample(filepath, rating, session_id)
    return {"status": "rated", "rating": rating}


@router.get("/ratings/{sample_path:path}")
async def get_ratings(sample_path: str):
    """Get ratings for a sample."""
    avg = get_average_rating(sample_path)
    ratings = get_sample_ratings(sample_path)
    return {"average": avg, "ratings": ratings}


@router.get("/ratings")
async def top_rated(limit: int = 50):
    """Get top rated samples."""
    return {"samples": get_top_rated_samples(limit)}


# ═══════════════════════════════════════════════════════════════════════════
# USAGE HISTORY ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/usage")
async def log(request: Request):
    """Log a usage event (play, drag, favorite, unfavorite)."""
    body = await request.json()
    filepath = body.get("sample_filepath")
    action = body.get("action")
    session_id = body.get("session_id")
    if not filepath or action not in ("play", "drag", "favorite", "unfavorite"):
        raise HTTPException(status_code=400, detail="Need sample_filepath and valid action")
    log_usage(filepath, action, session_id)
    return {"status": "logged"}


@router.get("/usage/most-used")
async def most_used(action: str = None, limit: int = 50):
    """Get most frequently used samples."""
    return {"samples": get_most_used_samples(action, limit)}


@router.get("/usage/recent")
async def recent(limit: int = 50):
    """Get recently used samples."""
    return {"samples": get_recently_used_samples(limit)}


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTION ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/collections")
async def create_coll(request: Request):
    """Create a new collection."""
    body = await request.json()
    name = body.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Collection name required")
    coll_id = create_collection(
        name=name,
        description=body.get("description"),
        is_smart=body.get("is_smart", False),
        smart_query=body.get("smart_query"),
    )
    return {"id": coll_id, "status": "created"}


@router.get("/collections")
async def list_collections():
    """List all collections."""
    return {"collections": get_collections()}


@router.post("/collections/{collection_id}/samples")
async def add_sample_to_coll(collection_id: int, request: Request):
    """Add a sample to a collection."""
    body = await request.json()
    filepath = body.get("sample_filepath")
    if not filepath:
        raise HTTPException(status_code=400, detail="sample_filepath required")
    add_to_collection(collection_id, filepath)
    return {"status": "added"}


@router.delete("/collections/{collection_id}/samples/{sample_path:path}")
async def remove_sample_from_coll(collection_id: int, sample_path: str):
    """Remove a sample from a collection."""
    remove_from_collection(collection_id, sample_path)
    return {"status": "removed"}


@router.get("/collections/{collection_id}/samples")
async def get_coll_samples(collection_id: int):
    """Get samples in a collection."""
    return {"samples": get_collection_samples(collection_id)}


@router.delete("/collections/{collection_id}")
async def remove_collection(collection_id: int):
    """Delete a collection."""
    delete_collection(collection_id)
    return {"status": "deleted"}
