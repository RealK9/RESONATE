"""
RESONATE — Bridge API Routes.
Frontend queries for DAW transport state and sends key changes to plugin.
Detects BPM changes to trigger real-time re-scoring.
"""

import asyncio
from fastapi import APIRouter
from pydantic import BaseModel

from bridge import bridge_state, send_to_plugin
import state

router = APIRouter(prefix="/bridge")

# Minimum BPM change to trigger rescore (avoids noise from tiny fluctuations)
BPM_CHANGE_THRESHOLD = 2.0


@router.get("/status")
async def get_bridge_status():
    """Return current bridge connection state, DAW transport, and rescore flag."""
    current_bpm = bridge_state["bpm"]
    rescore_needed = False

    if bridge_state["connected"] and state.last_scored_bpm > 0:
        bpm_delta = abs(current_bpm - state.last_scored_bpm)
        rescore_needed = bpm_delta >= BPM_CHANGE_THRESHOLD

    return {
        "connected": bridge_state["connected"],
        "bpm": current_bpm,
        "timeSigNum": bridge_state["timeSigNum"],
        "timeSigDen": bridge_state["timeSigDen"],
        "playing": bridge_state["playing"],
        "position": bridge_state["position"],
        "rescoreNeeded": rescore_needed,
        "lastScoredBpm": state.last_scored_bpm,
    }


class KeyChangeRequest(BaseModel):
    key: str


@router.post("/key")
async def send_key_change(req: KeyChangeRequest):
    """Push a key change message to the DAW via bridge plugin."""
    await send_to_plugin({"type": "keyChange", "key": req.key})
    return {"sent": True, "key": req.key}
