"""
RESONATE — Track analysis route.
Supports file upload and bridge (DAW master) analysis.
"""

from typing import Optional
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
    from ml.analysis.gap_analyzer import GapAnalyzer
    from ml.training.style_priors import StylePriorsTrainer
    from ml.recommendation.candidate_generator import CandidateGenerator
    from ml.recommendation.reranker import Reranker
    from ml.recommendation.explanations import ExplanationEngine
    from ml.db.sample_store import SampleStore
    from ml.models.recommendation import RecommendationResult
    from ml.models.gap_analysis import GapAnalysisResult
    from ml.models.preference import FeedbackEvent
    from ml.training.preference_dataset import PreferenceDataset
    from ml.training.preference_serving import PreferenceServer
    from ml.training.train_ranker import RankerTrainer
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
    mix_profile.style = StyleClassifier().classify(mix_profile)

    print("  [v2] Loading reference corpus...")
    corpus = StylePriorsTrainer(str(REFERENCE_CORPUS_PATH)).load_or_default()

    print("  [v2] Diagnosing needs...")
    mix_profile.needs = NeedsEngine(corpus=corpus).diagnose(mix_profile)

    print("  [v2] Running gap analysis...")
    gap_result = GapAnalyzer().analyze(mix_profile)
    mix_profile.gap_analysis = gap_result.to_dict()
    state.latest_gap_result = gap_result

    state.latest_mix_profile = mix_profile

    # --- backwards-compat: also run v1 pipeline ---
    print("  [v2] Running legacy v1 analysis for backwards compatibility...")
    track = analyze_track(str(dest))
    state.latest_track_file = dest
    ai = _run_ai_analysis(track)
    _finalize(track, ai, label="v2 Analysis")

    return mix_profile.to_dict()


@router.get("/analyze/v2/needs")
async def get_v2_needs():
    """Return just the needs vector from the latest v2 analysis."""
    if state.latest_mix_profile is None:
        raise HTTPException(status_code=404, detail="No mix analyzed yet")
    profile_dict = state.latest_mix_profile.to_dict() if hasattr(state.latest_mix_profile, "to_dict") else state.latest_mix_profile
    return {"needs": profile_dict.get("needs", [])}


@router.get("/analyze/v2/gap")
async def get_v2_gap():
    """Return the gap analysis from the latest v2 analysis.

    Includes production readiness score (0-100), chart potential,
    genre coherence, and an actionable list of gaps sorted by severity.
    """
    gap = getattr(state, "latest_gap_result", None)
    if gap is None:
        raise HTTPException(status_code=404, detail="No gap analysis available — call /analyze/v2 first")
    return gap.to_dict()


@router.get("/analyze/v2/profile")
async def get_v2_profile():
    """Return the full MixProfile from the latest v2 analysis."""
    if state.latest_mix_profile is None:
        raise HTTPException(status_code=404, detail="No mix analyzed yet")
    return state.latest_mix_profile.to_dict() if hasattr(state.latest_mix_profile, "to_dict") else state.latest_mix_profile


@router.post("/analyze/v2/reference")
async def upload_reference(file: UploadFile = File(...), genre: Optional[str] = None):
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


# ---------------------------------------------------------------------------
# v2 recommendation endpoints
# ---------------------------------------------------------------------------

@router.post("/recommend/v2")
async def recommend_v2(max_results: int = 20):
    """Generate sample recommendations for the latest analyzed mix.

    Requires a prior call to ``/analyze/v2`` so that a MixProfile is available.
    Uses the full recommendation pipeline: candidate generation → reranking →
    explanation generation.
    """
    _require_v2()
    from config import PROFILE_DB_PATH, REFERENCE_CORPUS_PATH

    if state.latest_mix_profile is None:
        raise HTTPException(status_code=404, detail="No mix analyzed yet — call /analyze/v2 first")

    mix_profile = state.latest_mix_profile
    needs = mix_profile.needs
    if not needs:
        return RecommendationResult(mix_filepath=mix_profile.filepath).to_dict()

    # Load sample store and reference corpus
    store = SampleStore(str(PROFILE_DB_PATH))
    store.init()

    corpus = StylePriorsTrainer(str(REFERENCE_CORPUS_PATH)).load_or_default()

    # Stage 1: candidate generation (with vector index if available)
    from indexer import get_vector_index
    vector_index = get_vector_index()

    gap_result = getattr(state, "latest_gap_result", None)
    print(f"  [v2] Generating candidates... (vector index: {'yes' if vector_index else 'no'}, gap analysis: {'yes' if gap_result else 'no'})")
    generator = CandidateGenerator(sample_store=store, vector_index=vector_index, gap_result=gap_result)
    candidates = generator.generate(mix_profile, needs, max_candidates=max_results * 5)

    # Load preference server (Phase 5)
    pref_db_path = str(PROFILE_DB_PATH).replace(".db", "_prefs.db")
    pref_dataset = PreferenceDataset(pref_db_path)
    pref_dataset.init()
    pref_server = PreferenceServer(pref_dataset)
    pref_server.load()

    # Stage 2: reranking (with learned preferences + gap intelligence)
    print(f"  [v2] Reranking {len(candidates)} candidates... (gap analysis: {'yes' if gap_result else 'no'})")
    reranker = Reranker(corpus=corpus, preference_server=pref_server, gap_result=gap_result)
    recommendations = reranker.rerank(candidates, mix_profile, needs)

    # Stage 3: explanations (gap-aware)
    print("  [v2] Generating explanations...")
    engine = ExplanationEngine(gap_result=gap_result)
    engine.explain_batch(recommendations, mix_profile, needs)

    # Trim to requested count
    recommendations = recommendations[:max_results]

    result = RecommendationResult(
        mix_filepath=mix_profile.filepath,
        recommendations=recommendations,
        needs_addressed=list({r.need_addressed for r in recommendations if r.need_addressed}),
        total_candidates_considered=len(candidates),
    )
    state.latest_recommendations = result

    print(f"  [v2] Returning {len(recommendations)} recommendations")
    return result.to_dict()


@router.get("/recommend/v2/latest")
async def get_latest_recommendations():
    """Return the latest recommendation result."""
    if state.latest_recommendations is None:
        raise HTTPException(status_code=404, detail="No recommendations generated yet")
    return state.latest_recommendations.to_dict()


# ---------------------------------------------------------------------------
# v2 combined endpoint — the CORE workflow in a single call
# ---------------------------------------------------------------------------

@router.post("/analyze/v2/full")
async def analyze_and_recommend(
    file: UploadFile = File(...),
    max_results: int = 20,
):
    """Full RESONATE workflow in one call: upload → analyze → gap analysis → recommend.

    This is the primary endpoint for the core product experience.
    Returns the complete picture: mix analysis, gap analysis (what's missing),
    and scored sample recommendations (what to add).
    """
    _require_v2()
    from config import UPLOAD_DIR, REFERENCE_CORPUS_PATH, PROFILE_DB_PATH

    # Save uploaded file
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    # ── Stage 1: Mix Analysis ──
    print("\n  [v2/full] Running mix analysis...")
    mix_profile = analyze_mix(str(dest))

    print("  [v2/full] Classifying style...")
    mix_profile.style = StyleClassifier().classify(mix_profile)

    print("  [v2/full] Loading reference corpus...")
    corpus = StylePriorsTrainer(str(REFERENCE_CORPUS_PATH)).load_or_default()

    print("  [v2/full] Diagnosing needs...")
    mix_profile.needs = NeedsEngine(corpus=corpus).diagnose(mix_profile)

    # ── Stage 2: Gap Analysis ──
    print("  [v2/full] Running gap analysis...")
    gap_result = GapAnalyzer().analyze(mix_profile)
    mix_profile.gap_analysis = gap_result.to_dict()
    state.latest_gap_result = gap_result
    state.latest_mix_profile = mix_profile

    # ── Stage 3: Recommendation ──
    needs = mix_profile.needs
    recommendations_list = []
    total_candidates = 0

    if needs:
        store = SampleStore(str(PROFILE_DB_PATH))
        store.init()

        from indexer import get_vector_index
        vector_index = get_vector_index()

        print(f"  [v2/full] Generating candidates...")
        generator = CandidateGenerator(
            sample_store=store, vector_index=vector_index, gap_result=gap_result
        )
        candidates = generator.generate(mix_profile, needs, max_candidates=max_results * 5)
        total_candidates = len(candidates)

        # Load preference server
        pref_db_path = str(PROFILE_DB_PATH).replace(".db", "_prefs.db")
        pref_dataset = PreferenceDataset(pref_db_path)
        pref_dataset.init()
        pref_server = PreferenceServer(pref_dataset)
        pref_server.load()

        print(f"  [v2/full] Reranking {len(candidates)} candidates...")
        reranker = Reranker(corpus=corpus, preference_server=pref_server, gap_result=gap_result)
        recommendations_list = reranker.rerank(candidates, mix_profile, needs)

        print("  [v2/full] Generating explanations...")
        engine = ExplanationEngine(gap_result=gap_result)
        engine.explain_batch(recommendations_list, mix_profile, needs)

        recommendations_list = recommendations_list[:max_results]

    rec_result = RecommendationResult(
        mix_filepath=mix_profile.filepath,
        recommendations=recommendations_list,
        needs_addressed=list({r.need_addressed for r in recommendations_list if r.need_addressed}),
        total_candidates_considered=total_candidates,
    )
    state.latest_recommendations = rec_result

    # ── Also run v1 for backwards compat ──
    print("  [v2/full] Running legacy v1 analysis...")
    track = analyze_track(str(dest))
    state.latest_track_file = dest
    ai = _run_ai_analysis(track)
    _finalize(track, ai, label="v2/full Analysis")

    # ── Build combined response ──
    print(f"  [v2/full] Complete! Readiness: {gap_result.production_readiness_score:.0f}/100 | "
          f"Recommendations: {len(recommendations_list)}")

    return {
        "mix_profile": mix_profile.to_dict(),
        "gap_analysis": gap_result.to_dict(),
        "recommendations": rec_result.to_dict(),
        "summary": {
            "production_readiness": gap_result.production_readiness_score,
            "chart_potential_current": gap_result.chart_potential_current,
            "chart_potential_ceiling": gap_result.chart_potential_ceiling,
            "genre_detected": gap_result.genre_detected,
            "blueprint_used": gap_result.blueprint_name,
            "total_gaps": gap_result.total_gaps,
            "critical_gaps": gap_result.critical_gaps,
            "missing_roles": gap_result.missing_roles,
            "recommendations_count": len(recommendations_list),
            "gap_summary": gap_result.summary,
        },
    }


# ---------------------------------------------------------------------------
# v2 preference learning endpoints
# ---------------------------------------------------------------------------

def _get_pref_dataset() -> "PreferenceDataset":
    """Get or create the preference dataset."""
    from config import PROFILE_DB_PATH
    pref_db_path = str(PROFILE_DB_PATH).replace(".db", "_prefs.db")
    ds = PreferenceDataset(pref_db_path)
    ds.init()
    return ds


@router.post("/feedback/v2")
async def log_feedback(
    sample_filepath: str,
    action: str,
    mix_filepath: str = "",
    session_id: str = "",
    rating: Optional[int] = None,
    recommendation_rank: int = 0,
):
    """Log a user interaction with a recommended sample.

    Actions: click, audition, drag, keep, discard, rate, skip
    """
    _require_v2()
    valid_actions = {"click", "audition", "drag", "keep", "discard", "rate", "skip"}
    if action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {valid_actions}")

    if action == "rate" and (rating is None or not 1 <= rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be 1-5 for 'rate' action")

    import time
    event = FeedbackEvent(
        sample_filepath=sample_filepath,
        mix_filepath=mix_filepath or (state.latest_mix_profile.filepath if state.latest_mix_profile else ""),
        session_id=session_id,
        action=action,
        rating=rating,
        recommendation_rank=recommendation_rank,
        context_style=state.latest_mix_profile.style.primary_cluster if state.latest_mix_profile else "",
        timestamp=time.time(),
    )
    ds = _get_pref_dataset()
    ds.log_feedback(event)
    return {"status": "ok"}


@router.post("/feedback/v2/build-pairs")
async def build_preference_pairs(session_id: str = ""):
    """Construct preference pairs from logged feedback events."""
    _require_v2()
    ds = _get_pref_dataset()
    pairs = ds.build_pairs(session_id=session_id)
    return {"pairs_created": len(pairs)}


@router.post("/preference/v2/train")
async def train_preference_model(user_id: str = "default", min_pairs: int = 10):
    """Train a per-user taste model from accumulated preference pairs."""
    _require_v2()
    from config import PROFILE_DB_PATH

    ds = _get_pref_dataset()
    store = SampleStore(str(PROFILE_DB_PATH))
    store.init()

    trainer = RankerTrainer(dataset=ds, sample_store=store)
    model = trainer.train(user_id=user_id, min_pairs=min_pairs)

    if model is None:
        return {"status": "insufficient_data", "message": f"Need at least {min_pairs} preference pairs"}

    return {
        "status": "ok",
        "model_version": model.model_version,
        "training_pairs": model.training_pairs,
        "role_biases": len(model.role_bias),
        "style_biases": len(model.style_bias),
    }


@router.get("/preference/v2/model")
async def get_preference_model(user_id: str = "default"):
    """Return the current taste model for a user."""
    _require_v2()
    ds = _get_pref_dataset()
    model = ds.load_taste_model(user_id)
    if model is None:
        raise HTTPException(status_code=404, detail="No taste model trained yet")
    return model.to_dict()
