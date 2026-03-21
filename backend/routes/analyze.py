"""
RESONATE — Track analysis route.
Supports file upload and bridge (DAW master) analysis.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException

from config import HAS_CLAUDE
from analysis.track_analyzer import analyze_track
from analysis.sample_analyzer import classify_mood
from ai.claude_engine import claude_analyze_track
from bridge import request_audio_capture, bridge_state
import state

router = APIRouter()


def _run_ai_analysis(track):
    """AI analysis with heuristic fallback. Returns analysis dict."""
    print("\n  [2/3] AI musical intelligence...")
    ai = None
    if HAS_CLAUDE:
        ai = claude_analyze_track(track)
    if not ai:
        print("  Using Essentia heuristic analysis only")
        mood_info = classify_mood(
            track.get("rms", 0.04),
            track.get("spectral_centroid", 2000),
            track.get("key", "C"),
            track.get("frequency_bands", {})
        )
        ai = {
            "source": "heuristic",
            "genre": track.get("detected_genre", "unknown"),
            "mood": mood_info["mood"],
            "energy": mood_info["energy"].capitalize(),
            "what_track_has": track.get("detected_instruments", []),
            "what_track_needs": track.get("frequency_gaps", []),
            "summary": f"{mood_info['mood']} {track.get('detected_genre', 'unknown')} track at {track.get('bpm')} BPM",
        }
    return ai


def _clear_transposition_cache():
    """Clear cached transposed files."""
    from config import TRANSPOSED_DIR
    for f in TRANSPOSED_DIR.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass


def _finalize(track, ai, label="Analysis"):
    """Store state, clear cache, log completion."""
    state.latest_track_profile = track
    state.latest_ai_analysis = ai
    _clear_transposition_cache()
    print(f"\n  [3/3] Ready for sample matching")
    print(f"\n  {'=' * 40}")
    print(f"  {label} complete!")
    print(f"  {'=' * 40}\n")


@router.post("/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    """Analyze an uploaded track."""
    from config import UPLOAD_DIR

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    track = analyze_track(str(dest))
    state.latest_track_file = dest
    ai = _run_ai_analysis(track)
    _finalize(track, ai)
    return _build_response(track, ai)


@router.post("/analyze/bridge")
async def analyze_from_daw():
    """Analyze audio captured directly from the DAW via the bridge plugin."""
    from config import UPLOAD_DIR

    if not bridge_state["connected"]:
        raise HTTPException(status_code=400, detail="No bridge plugin connected")

    print("\n  [Bridge] Requesting audio capture from DAW...")
    wav_data = await request_audio_capture()

    if not wav_data or len(wav_data) < 100:
        raise HTTPException(status_code=400, detail="No audio captured from DAW — make sure transport has played")

    dest = UPLOAD_DIR / "daw_capture.wav"
    with open(dest, "wb") as f:
        f.write(wav_data)
    print(f"  [Bridge] Captured {len(wav_data)} bytes → {dest}")

    track = analyze_track(str(dest))
    if bridge_state["bpm"] > 0:
        track["bpm"] = bridge_state["bpm"]

    state.latest_track_file = dest
    ai = _run_ai_analysis(track)
    _finalize(track, ai, "Bridge analysis")
    return _build_response(track, ai, filename="DAW Master")


def _build_response(track, ai, filename=None):
    """Build the standard analysis response."""
    resp = {
        "duration": track["duration"],
        "analysis": {
            "key": track["key"],
            "key_confidence": track["key_confidence"],
            "bpm": track["bpm"],
            "genre": (ai or {}).get("genre", None) or track.get("detected_genre", "unknown"),
            "mood": (ai or {}).get("mood", "unknown"),
            "energy_label": (ai or {}).get("energy", "Medium"),
            "what_track_has": (ai or {}).get("what_track_has", []),
            "what_track_needs": (ai or {}).get("what_track_needs", []),
            "summary": (ai or {}).get("summary", ""),
            "frequency_bands": track.get("frequency_bands", {}),
            "frequency_gaps": track.get("frequency_gaps", []),
            "detected_instruments": track.get("detected_instruments", []),
        },
    }
    if filename:
        resp["filename"] = filename
    return resp
