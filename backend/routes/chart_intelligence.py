"""
RESONATE — Chart Intelligence 2.0 routes.
Trends, comparison, and insights from chart data.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

import state
from config import BACKEND_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/charts", tags=["charts"])

CHART_DB_PATH = BACKEND_DIR / "chart_features.db"

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _sanitize_profile(d: dict) -> dict:
    """Convert numpy arrays/types to JSON-safe Python types."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.float32, np.float64)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, np.int32, np.int64)):
            out[k] = int(v)
        elif isinstance(v, list):
            out[k] = [float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else x for x in v]
        else:
            out[k] = v
    return out


def _load_analyzer():
    """Load and run ChartAnalyzer. Returns analyzer or raises 404."""
    from ml.training.charts.chart_analysis import ChartAnalyzer

    if not CHART_DB_PATH.exists():
        raise HTTPException(status_code=404, detail="Chart database not found")

    analyzer = ChartAnalyzer(str(CHART_DB_PATH))
    analyzer.analyze()
    return analyzer


def _generate_insights(mix: dict, genre_profile) -> list[str]:
    """Generate text insights comparing a mix to its genre profile."""
    insights = []

    mix_bpm = mix.get("bpm") or mix.get("analysis", {}).get("bpm", 0)
    if mix_bpm and genre_profile.bpm_mean:
        diff = mix_bpm - genre_profile.bpm_mean
        if abs(diff) > genre_profile.bpm_std:
            direction = "faster" if diff > 0 else "slower"
            insights.append(
                f"Your BPM ({mix_bpm:.0f}) is {abs(diff):.0f} BPM {direction} than "
                f"the {genre_profile.genre} average ({genre_profile.bpm_mean:.0f})"
            )
        else:
            insights.append(
                f"Your BPM ({mix_bpm:.0f}) is right in the sweet spot for "
                f"{genre_profile.genre} ({genre_profile.bpm_mean:.0f} avg)"
            )

    mix_energy = mix.get("energy") or mix.get("analysis", {}).get("energy")
    if mix_energy is not None and genre_profile.energy_mean:
        diff = mix_energy - genre_profile.energy_mean
        if abs(diff) > 0.15:
            level = "higher" if diff > 0 else "lower"
            insights.append(
                f"Energy is {level} than typical {genre_profile.genre} tracks "
                f"({mix_energy:.2f} vs {genre_profile.energy_mean:.2f} avg)"
            )

    mix_valence = mix.get("valence") or mix.get("analysis", {}).get("valence")
    if mix_valence is not None and genre_profile.valence_mean:
        if mix_valence > genre_profile.valence_mean + 0.1:
            insights.append(
                f"More positive/upbeat mood than average {genre_profile.genre} charts"
            )
        elif mix_valence < genre_profile.valence_mean - 0.1:
            insights.append(
                f"Darker mood than typical charting {genre_profile.genre} tracks"
            )

    mix_dance = mix.get("danceability") or mix.get("analysis", {}).get("danceability")
    if mix_dance is not None and genre_profile.danceability_mean:
        diff = mix_dance - genre_profile.danceability_mean
        if abs(diff) > 0.15:
            level = "more" if diff > 0 else "less"
            insights.append(
                f"Your track is {level} danceable than the {genre_profile.genre} chart average"
            )

    if not insights:
        insights.append(
            f"Your mix aligns well with current {genre_profile.genre} chart trends"
        )

    return insights


@router.get("/trends")
async def get_chart_trends(
    genre: Optional[str] = Query(None, description="Filter by genre"),
    decade: Optional[int] = Query(None, description="Filter by decade (e.g. 2020)"),
):
    """Get chart trend data — decade profiles and genre profiles."""
    analyzer = _load_analyzer()

    decade_profiles = analyzer.get_decade_profiles()
    genre_profiles = analyzer.get_genre_profiles()

    # Serialize
    decades_out = {}
    for dec, profile in sorted(decade_profiles.items()):
        if decade and dec != decade:
            continue
        decades_out[str(dec)] = _sanitize_profile(asdict(profile))

    genres_out = {}
    for g, profile in genre_profiles.items():
        if genre and g.lower() != genre.lower():
            continue
        genres_out[g] = _sanitize_profile(asdict(profile))

    return {
        "decades": decades_out,
        "genres": genres_out,
    }


@router.get("/compare")
async def get_chart_comparison():
    """Compare latest mix profile against chart data."""
    mix = state.latest_mix_profile
    if not mix:
        raise HTTPException(status_code=404, detail="No mix analyzed yet — upload a track first")

    analyzer = _load_analyzer()
    genre_profiles = analyzer.get_genre_profiles()
    decade_profiles = analyzer.get_decade_profiles()

    # Extract genre from mix profile
    mix_genre = (
        mix.get("style", {}).get("primary_cluster")
        or mix.get("analysis", {}).get("genre")
        or ""
    ).lower().strip()

    # Find matching genre profile (fuzzy match)
    matched_profile = None
    for g, profile in genre_profiles.items():
        if g.lower() == mix_genre or mix_genre in g.lower() or g.lower() in mix_genre:
            matched_profile = profile
            break

    # Fallback to pop if no match
    if not matched_profile:
        matched_profile = genre_profiles.get("pop")

    # Build comparison
    mix_bpm = mix.get("bpm") or mix.get("analysis", {}).get("bpm", 0)
    mix_energy = mix.get("energy") or mix.get("analysis", {}).get("energy")
    mix_valence = mix.get("valence") or mix.get("analysis", {}).get("valence")
    mix_dance = mix.get("danceability") or mix.get("analysis", {}).get("danceability")
    mix_key = mix.get("key") or mix.get("analysis", {}).get("key", "")

    your_mix = {
        "bpm": mix_bpm,
        "energy": mix_energy,
        "valence": mix_valence,
        "danceability": mix_dance,
        "key": mix_key,
        "genre": mix_genre,
    }

    chart_average = {}
    insights = []

    if matched_profile:
        chart_average = {
            "genre": matched_profile.genre,
            "bpm_mean": matched_profile.bpm_mean,
            "bpm_std": matched_profile.bpm_std,
            "energy_mean": matched_profile.energy_mean,
            "valence_mean": matched_profile.valence_mean,
            "danceability_mean": matched_profile.danceability_mean,
            "major_ratio": matched_profile.major_ratio,
            "key_distribution": matched_profile.key_distribution,
            "avg_peak_position": matched_profile.avg_peak_position,
            "avg_weeks_on_chart": matched_profile.avg_weeks_on_chart,
            "count": matched_profile.count,
        }
        insights = _generate_insights(mix, matched_profile)

    # Decade trends (last 3 decades)
    sorted_decades = sorted(decade_profiles.keys(), reverse=True)
    decade_trends = []
    for dec in sorted_decades[:3]:
        dp = decade_profiles[dec]
        decade_trends.append({
            "decade": dec,
            "bpm_mean": dp.bpm_mean,
            "energy_mean": dp.energy_mean,
            "valence_mean": dp.valence_mean,
            "danceability_mean": dp.danceability_mean,
            "major_ratio": dp.major_ratio,
            "count": dp.count,
        })

    return {
        "your_mix": your_mix,
        "chart_average": chart_average,
        "insights": insights,
        "decade_trends": decade_trends,
    }
