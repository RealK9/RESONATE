"""
RESONATE — Track audio serving route.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException

from audio.serve import serve_audio
import state

router = APIRouter()


@router.get("/track/audio")
async def get_track_audio():
    """Serve the uploaded track audio for dual playback."""
    if not state.latest_track_file or not Path(state.latest_track_file).exists():
        raise HTTPException(status_code=404, detail="No track uploaded")
    return serve_audio(Path(state.latest_track_file))
