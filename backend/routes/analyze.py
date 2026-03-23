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

# v2 ML pipeline imports — wrapped so server starts even if ml modules are missing
try:
    from ml.analysis.mix_analyzer import analyze_mix
    from ml.analysis.style_classifier import StyleClassifier
    from ml.analysis.needs_engine import NeedsEngine
    from ml.training.style_priors import StylePriorsTrainer
    _HAS_V2_ML = True
except ImportError as _v2_err:
    _HAS_V2_ML = False
    print(f"  [v2] ML modules not available ({_v2_err}); v2 endpoints will 503")

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


# ---------------------------------------------------------------------------
# v2 endpoints — ML-powered mix analysis
# ---------------------------------------------------------------------------

def _require_v2():
    """Raise 503 if v2 ML modules failed to import."""
    if not _HAS_V2_ML:
        raise HTTPException(
            status_code=503,
            detail="v2 ML pipeline not available on this server",
        )


@router.post("/analyze/v2")
async def analyze_v2(file: UploadFile = File(...)):
    """Full v2 mix analysis — audio features, style classification, and needs diagnosis."""
    _require_v2()
    from config import UPLOAD_DIR, REFERENCE_CORPUS_PATH

    # Save uploaded file
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    # --- v2 pipeline ---
    print("\n  [v2] Running mix analysis...")
    mix_profile = analyze_mix(str(dest))

    print("  [v2] Classifying style...")
    StyleClassifier().classify(mix_profile)

    print("  [v2] Loading reference corpus...")
    corpus = StylePriorsTrainer(str(REFERENCE_CORPUS_PATH)).load_or_default()

    print("  [v2] Diagnosing needs...")
    NeedsEngine(corpus=corpus).diagnose(mix_profile)

    state.latest_mix_profile = mix_profile

    # --- backwards-compat: also run v1 pipeline ---
    print("  [v2] Running legacy v1 analysis for backwards compatibility...")
    track = analyze_track(str(dest))
    state.latest_track_file = dest
    ai = _run_ai_analysis(track)
    _finalize(track, ai, label="v2 Analysis")

    return mix_profile


@router.get("/analyze/v2/needs")
async def get_v2_needs():
    """Return just the needs vector from the latest v2 analysis."""
    if state.latest_mix_profile is None:
        raise HTTPException(status_code=404, detail="No mix analyzed yet")
    return {"needs": state.latest_mix_profile.get("needs", [])}


@router.get("/analyze/v2/profile")
async def get_v2_profile():
    """Return the full MixProfile from the latest v2 analysis."""
    if state.latest_mix_profile is None:
        raise HTTPException(status_code=404, detail="No mix analyzed yet")
    return state.latest_mix_profile


@router.post("/analyze/v2/reference")
async def upload_reference(file: UploadFile = File(...), genre: str | None = None):
    """Upload a reference track to improve style priors.

    The file is saved to the uploads directory and stored for future
    training.  If the full v2 ML pipeline is available the file is
    immediately analyzed and added to the on-disk reference corpus.
    """
    _require_v2()
    from config import UPLOAD_DIR, REFERENCE_CORPUS_PATH

    dest = UPLOAD_DIR / f"ref_{file.filename}"
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    print(f"\n  [v2] Reference track saved: {dest}")

    # Analyze and fold into the corpus
    trainer = StylePriorsTrainer(str(REFERENCE_CORPUS_PATH))

    # Load existing corpus references (if any) by starting fresh and
    # just training the new file.  The full retrain-from-all approach
    # would require storing raw profiles; for now we just save the file
    # and let a batch retrain pick it up later.
    try:
        trainer.add_reference_file(str(dest), genre=genre)
        corpus = trainer.train()
        print(f"  [v2] Reference corpus updated ({corpus.total_references} refs)")
        return {
            "status": "ok",
            "filepath": str(dest),
            "total_references": corpus.total_references,
        }
    except Exception as exc:
        # Even if analysis fails, the file is saved for later
        print(f"  [v2] Reference analysis failed ({exc}); file saved for batch retrain")
        return {
            "status": "saved",
            "filepath": str(dest),
            "error": str(exc),
        }
