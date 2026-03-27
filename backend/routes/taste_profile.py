"""
RESONATE — Taste Profile Route.
Producer DNA endpoints: fetch the user's sonic identity and trigger retraining.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from fastapi import APIRouter

from config import PROFILE_DB_PATH
from ml.models.preference import UserTasteModel
from ml.training.preference_dataset import PreferenceDataset
from ml.training.train_ranker import RankerTrainer

router = APIRouter()

# Preference DB uses the "_prefs.db" suffix convention
_PREFS_DB = str(PROFILE_DB_PATH).replace(".db", "_prefs.db")


def _get_dataset() -> PreferenceDataset:
    ds = PreferenceDataset(_PREFS_DB)
    ds.init()
    return ds


def _action_breakdown(db_path: str) -> dict[str, int]:
    """Query feedback_events table directly for action counts."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        rows = conn.execute(
            "SELECT action, COUNT(*) FROM feedback_events GROUP BY action"
        ).fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}


@router.get("/taste/profile")
async def get_taste_profile(user_id: str = "default"):
    """Return the user's taste profile — role affinities, style preferences, stats."""
    ds = _get_dataset()
    model = ds.load_taste_model(user_id)

    if model is None:
        return {
            "status": "no_data",
            "message": "Use RESONATE more to build your sonic identity. "
            "Audition, keep, and rate samples to train your taste profile.",
        }

    # Format role_bias as sorted list of {role, affinity}
    role_affinities = sorted(
        [{"role": k, "affinity": round(v, 4)} for k, v in model.role_bias.items()],
        key=lambda x: abs(x["affinity"]),
        reverse=True,
    )

    # Format style_bias as sorted list of {style, preference}
    style_preferences = sorted(
        [{"style": k, "preference": round(v, 4)} for k, v in model.style_bias.items()],
        key=lambda x: abs(x["preference"]),
        reverse=True,
    )

    # Action breakdown stats from the raw feedback_events table
    breakdown = _action_breakdown(_PREFS_DB)
    total_interactions = sum(breakdown.values())

    return {
        "status": "ok",
        "user_id": model.user_id,
        "model_version": model.model_version,
        "training_pairs": model.training_pairs,
        "last_trained": model.last_trained,
        "role_affinities": role_affinities,
        "style_preferences": style_preferences,
        "quality_threshold": round(model.quality_threshold, 4),
        "weight_profile": {k: round(v, 4) for k, v in model.weight_deltas.items()},
        "total_interactions": total_interactions,
        "action_breakdown": breakdown,
    }


@router.post("/taste/train")
async def train_taste(user_id: str = "default"):
    """Build pairs from recent feedback and train the taste model."""
    ds = _get_dataset()

    # Build new pairs from any recent feedback
    pairs = ds.build_pairs()

    # Train with min_pairs=5 for faster cold-start
    trainer = RankerTrainer(dataset=ds, sample_store=None)
    model = trainer.train(user_id=user_id, min_pairs=5)

    if model is None:
        return {
            "status": "insufficient_data",
            "message": "Not enough preference data yet. Keep using RESONATE!",
            "training_pairs": 0,
            "model_version": 0,
        }

    return {
        "status": "ok",
        "training_pairs": model.training_pairs,
        "model_version": model.model_version,
    }
