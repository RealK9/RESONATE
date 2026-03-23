"""Preference learning data structures for Phase 5."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np


def _json_serializer(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


@dataclass
class FeedbackEvent:
    """A single user interaction with a recommended sample."""

    sample_filepath: str = ""
    mix_filepath: str = ""
    session_id: str = ""
    action: str = ""  # click, audition, drag, keep, discard, rate, skip
    rating: int | None = None  # 1-5, only for "rate" action
    recommendation_rank: int = 0  # position in the recommendation list (0-indexed)
    context_style: str = ""  # primary style cluster at time of interaction
    timestamp: float = 0.0


@dataclass
class PreferencePair:
    """A pairwise preference: sample A was preferred over sample B in context."""

    preferred_filepath: str = ""
    rejected_filepath: str = ""
    mix_filepath: str = ""
    context_style: str = ""
    strength: float = 1.0  # how strong the preference signal is (0-1)
    timestamp: float = 0.0


@dataclass
class UserTasteModel:
    """Learned per-user taste model -- adjusts reranker weights."""

    user_id: str = "default"
    # Per-role bias: how much the user likes each role
    # (0 = neutral, positive = likes, negative = dislikes)
    role_bias: dict[str, float] = field(default_factory=dict)
    # Per-style bias: user's affinity for style clusters
    style_bias: dict[str, float] = field(default_factory=dict)
    # Feature weight adjustments: deltas applied to reranker weights
    weight_deltas: dict[str, float] = field(default_factory=dict)
    # Quality threshold: learned minimum quality the user accepts
    quality_threshold: float = 0.3
    # Training metadata
    training_pairs: int = 0
    model_version: int = 0
    last_trained: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer)

    @classmethod
    def from_dict(cls, d: dict) -> UserTasteModel:
        model = cls()
        for k, v in d.items():
            if hasattr(model, k):
                setattr(model, k, v)
        return model
