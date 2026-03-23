"""
RESONATE — Sample listing, audio serving, and waveform routes.
"""

import time as _time
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

import config
from indexer import indexing_status, sample_cache
from analysis.scoring import math_match, semitones_to_transpose
from analysis.genre_profiles import get_genre_profile
from audio.serve import serve_audio
from audio.transpose import transpose_sample
from audio.waveform import extract_waveform_peaks
from utils import classify_type, clean_name, TYPE_LABELS
import state

router = APIRouter()

# Waveform peak cache
_waveform_cache = {}


def find_sample_file(sample_path):
    """Find a sample file using multiple lookup strategies."""
    decoded = unquote(sample_path)
    SAMPLE_DIR = config.SAMPLE_DIR

    # Check if it's an absolute path (external library — Splice, Loopcloud)
    abs_path = Path(decoded).resolve()
    if abs_path.is_absolute() and abs_path.exists() and abs_path.is_file():
        # Only allow paths within known sample directories
        allowed = [SAMPLE_DIR] + list(getattr(config, 'SPLICE_DIRS', [])) + list(getattr(config, 'LOOPCLOUD_DIRS', []))
        if any(str(abs_path).startswith(str(d.resolve())) for d in allowed if d):
            return abs_path
        return None  # path outside allowed directories

    # Prevent path traversal for relative paths
    fp = (SAMPLE_DIR / decoded).resolve()
    if not str(fp).startswith(str(SAMPLE_DIR.resolve())):
        return None
    if fp.exists() and fp.is_file():
        return fp

    for part in Path(decoded).parts:
        test = SAMPLE_DIR / part / Path(decoded).name
        if test.exists() and test.is_file():
            return test

    parent = fp.parent
    if parent.exists():
        target = fp.name.lower()
        for f in parent.iterdir():
            if f.name.lower() == target and f.is_file():
                return f

    target_name = Path(decoded).name
    for f in SAMPLE_DIR.rglob("*"):
        if f.name == target_name and f.is_file():
            return f

    target_lower = target_name.lower()
    for f in SAMPLE_DIR.rglob("*"):
        if f.name.lower() == target_lower and f.is_file():
            return f

    return None


@router.get("/samples")
async def list_samples():
    """List all samples with match scores. Uses pre-indexed cache."""
    SAMPLE_DIR = config.SAMPLE_DIR

    if not indexing_status["done"]:
        import asyncio
        print("  Waiting for sample indexing to complete...")
        while not indexing_status["done"]:
            await asyncio.sleep(0.5)
        print(f"  Indexing done — {len(sample_cache)} samples ready")

    all_sample_data = []
    for cache_key, sa in sample_cache.items():
        if sa.get("_hash"):
            fp = Path(cache_key)
            if not fp.exists():
                continue

            source = sa.get("source", "local")

            try:
                rel = fp.relative_to(SAMPLE_DIR)
            except ValueError:
                rel = Path(fp.name)

            parts = rel.parts
            cat = parts[0] if len(parts) > 1 else "Uncategorized"
            sub = parts[1] if len(parts) > 2 else ""

            math_score = math_match(
                state.latest_track_profile, sa,
                sample_filepath=str(fp),
                ai_template=state.latest_ai_analysis if state.latest_ai_analysis else None
            ) if state.latest_track_profile else 50

            all_sample_data.append({
                "id": cache_key,
                "name": fp.stem,
                "filename": fp.name,
                "path": str(rel).replace("#", "%23"),
                "category": cat,
                "sub_category": sub,
                "duration": sa.get("duration", 0),
                "bpm": sa.get("bpm", 0),
                "key": sa.get("key", "N/A"),
                "sample_type": sa.get("sample_type", "unknown"),
                "math_score": math_score,
                "match": math_score,
                "frequency_bands": sa.get("frequency_bands", {}),
                "source": source,
            })

    # ── Deterministic scoring — apply duplicate hard cap ──
    if state.latest_track_profile and all_sample_data:
        t_start = _time.time()
        genre_str = state.latest_ai_analysis.get("genre", "") if state.latest_ai_analysis else ""
        if not genre_str:
            genre_str = state.latest_track_profile.get("detected_genre", "default")
        genre_ref = get_genre_profile(genre_str)
        print(f"\n  Scoring {len(all_sample_data)} samples (deterministic, measured audio)...")
        print(f"  Genre reference: {genre_str} → target centroid {genre_ref['spectral_centroid_target']}Hz")

        for s in all_sample_data:
            final = s["math_score"]
            stype = s.get("sample_type", "unknown")
            is_duplicate = False

            if state.latest_track_profile.get("detected_instruments"):
                detected = state.latest_track_profile["detected_instruments"]
                if stype == "bass" and any(x in detected for x in ["sub_bass_808", "bass"]):
                    is_duplicate = True
                if stype == "kick" and "kick" in detected:
                    is_duplicate = True
                if stype == "snare" and "snare_clap" in detected:
                    is_duplicate = True

            if state.latest_ai_analysis and not is_duplicate:
                type_priorities = state.latest_ai_analysis.get("type_priority_scores", {})
                type_pri = type_priorities.get(stype, 50)
                if isinstance(type_pri, (int, float)) and type_pri <= 15:
                    is_duplicate = True

            if not is_duplicate:
                dup_genre_str = state.latest_ai_analysis.get("genre", "") if state.latest_ai_analysis else ""
                if not dup_genre_str:
                    dup_genre_str = state.latest_track_profile.get("detected_genre", "default")
                dup_genre_ref = get_genre_profile(dup_genre_str)
                genre_type_need = dup_genre_ref["type_needs"].get(stype, 50)
                if genre_type_need <= 10:
                    is_duplicate = True

            if is_duplicate:
                final = min(final, 15)

            s["match"] = max(0, min(100, final))

        # ── Preference Learning Boost (5-8% weight) ──
        try:
            from db.database import get_top_rated_samples
            import numpy as np
            top_rated = get_top_rated_samples(limit=100)
            if top_rated:
                # Build taste vector from rated samples
                taste_mfcc_list = []
                taste_centroid_list = []
                for item in top_rated:
                    sa = sample_cache.get(item["sample_filepath"], {})
                    if not sa:
                        continue
                    w = item.get("avg_rating", 3) / 5.0
                    if sa.get("mfcc_profile"):
                        taste_mfcc_list.append((np.array(sa["mfcc_profile"]), w))
                    if sa.get("spectral_centroid"):
                        taste_centroid_list.append((sa["spectral_centroid"], w))

                if taste_mfcc_list:
                    total_w = sum(w for _, w in taste_mfcc_list)
                    taste_mfcc = sum(m * w for m, w in taste_mfcc_list) / total_w

                    for s in all_sample_data:
                        sa = sample_cache.get(s["id"], {})
                        if not sa or not sa.get("mfcc_profile"):
                            continue
                        sample_mfcc = np.array(sa["mfcc_profile"])
                        na = np.linalg.norm(taste_mfcc)
                        nb = np.linalg.norm(sample_mfcc)
                        if na > 0 and nb > 0:
                            cos = float(np.dot(taste_mfcc, sample_mfcc) / (na * nb))
                            # Subtle boost: max +6 points for perfect taste match
                            taste_boost = max(0, cos * 6)
                            s["match"] = max(0, min(100, s["match"] + taste_boost))
                    print(f"  ✓ Preference learning applied ({len(top_rated)} rated samples)")
        except Exception:
            pass  # Preference learning is optional

        # Track which BPM was used for scoring (for bridge change detection)
        state.last_scored_bpm = state.daw_bpm if state.daw_bpm > 0 else state.latest_track_profile.get("bpm", 0)

        elapsed = round(_time.time() - t_start, 3)
        target_bpm_src = "DAW" if state.daw_bpm > 0 else "track"
        print(f"  ✓ Scoring complete in {elapsed}s (target BPM: {state.last_scored_bpm:.1f} from {target_bpm_src})")

    all_sample_data.sort(key=lambda x: -x["match"])

    # Build response — always show original key/BPM, include synced values for bridge
    synced_key = state.latest_track_profile.get("key", "") if state.latest_track_profile else ""
    synced_bpm = (state.daw_bpm if state.daw_bpm > 0 else state.latest_track_profile.get("bpm", 0)) if state.latest_track_profile else 0

    samples_response = []
    for s in all_sample_data:

        def get_match_reason(s):
            stype = s.get("sample_type", "unknown")
            gaps = state.latest_track_profile.get("frequency_gaps", []) if state.latest_track_profile else []
            sc = s.get("spectral_centroid", 2000)
            reasons = []
            if sc < 1500:
                reasons.append("dark tone")
            elif sc < 2200:
                reasons.append("warm character")
            if stype in ("melody", "pad", "strings") and "midrange_melody" in gaps:
                reasons.append("fills midrange")
            elif stype == "vocals" and "upper_mid_presence" in gaps:
                reasons.append("adds presence")
            elif stype == "hihat" and "high_end_sparkle" in gaps:
                reasons.append("high-end sparkle")
            elif stype in ("melody", "pad") and "low_mid_warmth" in gaps:
                reasons.append("adds warmth")
            if s.get("duration", 0) > 4 and stype in ("melody", "vocals", "pad"):
                reasons.append("rich content")
            if not reasons:
                reasons.append("spectral fit")
            return " · ".join(reasons[:2])

        samples_response.append({
            "id": s["id"],
            "name": s["name"],
            "clean_name": clean_name(s["name"]),
            "filename": s["filename"],
            "path": s["path"],
            "category": s["category"],
            "sub_category": s["sub_category"],
            "duration": s["duration"],
            "bpm": s["bpm"],
            "original_bpm": s["bpm"],
            "key": s["key"],
            "original_key": s["key"],
            "synced_key": synced_key or s["key"],
            "synced_bpm": synced_bpm or s["bpm"],
            "match": s["match"],
            "sample_type": s.get("sample_type", "unknown"),
            "type_label": TYPE_LABELS.get(s.get("sample_type", "unknown"), "Sound"),
            "match_reason": get_match_reason(s),
            "frequency_bands": s.get("frequency_bands", {}),
            "source": s.get("source", "local"),
            "mood": sample_cache.get(s["id"], {}).get("mood", "neutral"),
            "energy": sample_cache.get(s["id"], {}).get("energy", "medium"),
        })

    return {
        "samples": samples_response,
        "total": len(samples_response),
        "daw_bpm": state.daw_bpm if state.daw_bpm > 0 else None,
        "scored_bpm": state.last_scored_bpm if state.last_scored_bpm > 0 else None,
    }


@router.get("/samples/abspath/{sample_path:path}")
async def get_sample_abspath(sample_path: str, sync: int = 0):
    """Return absolute file path for drag-to-DAW. When sync=1, returns transposed file."""
    fp = find_sample_file(sample_path)
    if not fp:
        raise HTTPException(status_code=404, detail="Not found")

    if sync and state.latest_track_profile and state.latest_track_profile.get("key"):
        track_key = state.latest_track_profile["key"]
        track_bpm = state.daw_bpm if state.daw_bpm > 0 else state.latest_track_profile.get("bpm", 0)
        sample_info = sample_cache.get(str(fp), {})
        sample_key = sample_info.get("key", "N/A")
        sample_bpm = sample_info.get("bpm", 0)
        semitones = semitones_to_transpose(sample_key, track_key) if sample_key not in ("N/A", "—") else 0
        tempo_ratio = 1.0
        if sample_bpm and sample_bpm > 0 and track_bpm and track_bpm > 0:
            ratio = track_bpm / sample_bpm
            candidates = [ratio, ratio * 2, ratio / 2]
            tempo_ratio = min(candidates, key=lambda r: abs(r - 1.0))
        if abs(semitones) > 0 or abs(tempo_ratio - 1.0) > 0.01:
            transposed = transpose_sample(fp, semitones, tempo_ratio)
            return {"path": str(transposed)}

    return {"path": str(fp)}


@router.get("/samples/waveform/{sample_path:path}")
async def get_waveform(sample_path: str, bars: int = 100):
    """Return waveform peaks for visualizing the actual audio shape."""
    cache_key = f"{sample_path}:{bars}"
    if cache_key in _waveform_cache:
        return JSONResponse(content=_waveform_cache[cache_key])

    fp = find_sample_file(sample_path)
    if not fp:
        raise HTTPException(status_code=404, detail="Not found")

    try:
        result = extract_waveform_peaks(fp, bars)
        _waveform_cache[cache_key] = result
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"peaks": [], "error": str(e)})


@router.get("/samples/audio/{sample_path:path}")
async def get_audio(sample_path: str, sync: int = 0):
    """Serve sample audio. Only transposes when sync=1 (bridge sync enabled)."""
    fp = find_sample_file(sample_path)

    if not fp:
        print(f"  404: {sample_path} (tried all methods)")
        raise HTTPException(status_code=404, detail=f"Not found: {sample_path}")

    # Only transpose when explicitly requested via bridge sync
    if sync and state.latest_track_profile and state.latest_track_profile.get("key"):
        track_key = state.latest_track_profile["key"]
        track_bpm = state.daw_bpm if state.daw_bpm > 0 else state.latest_track_profile.get("bpm", 0)

        sample_info = sample_cache.get(str(fp), {})
        sample_key = sample_info.get("key", "N/A")
        sample_bpm = sample_info.get("bpm", 0)

        semitones = 0
        if sample_key and sample_key not in ("N/A", "—"):
            semitones = semitones_to_transpose(sample_key, track_key)

        tempo_ratio = 1.0
        if sample_bpm and sample_bpm > 0 and track_bpm and track_bpm > 0:
            ratio = track_bpm / sample_bpm
            candidates = [ratio, ratio * 2, ratio / 2]
            tempo_ratio = min(candidates, key=lambda r: abs(r - 1.0))

        if abs(semitones) > 0 or abs(tempo_ratio - 1.0) > 0.01:
            transposed = transpose_sample(fp, semitones, tempo_ratio)
            return serve_audio(transposed)

    return serve_audio(fp)


# ── V2 Pipeline Endpoints ─────────────────────────────────────────────────

@router.get("/samples/v2/profile/{sample_path:path}")
def get_sample_profile(sample_path: str):
    """Get the full v2 analysis profile for a sample."""
    from config import PROFILE_DB_PATH
    from backend.ml.db.sample_store import SampleStore

    store = SampleStore(str(PROFILE_DB_PATH))
    store.init()

    profile = store.load(sample_path)
    if not profile:
        # Try finding by filename
        found = find_sample_file(sample_path)
        if found:
            profile = store.load(str(found))

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return profile.to_dict()


@router.get("/samples/v2/stats")
def get_indexing_stats():
    """Get v2 pipeline indexing statistics."""
    from config import PROFILE_DB_PATH
    from backend.ml.db.sample_store import SampleStore

    store = SampleStore(str(PROFILE_DB_PATH))
    store.init()

    return {
        "total_profiles": store.count(),
        "pipeline_version": "phase1",
    }
