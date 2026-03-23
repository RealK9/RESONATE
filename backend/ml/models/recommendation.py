"""
Recommendation data structures for Phase 4 -- Complement Recommendation.

These dataclasses represent the output of the recommendation pipeline:
a scored, explained list of samples that would IMPROVE a given mix.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def _json_serializer(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


@dataclass
class ScoringBreakdown:
    """Detailed breakdown of how a candidate was scored."""
    need_fit: float = 0.0          # alpha -- how well it fills a diagnosed need
    role_fit: float = 0.0          # beta -- role match to what's needed
    spectral_complement: float = 0.0  # gamma -- fills spectral gaps
    tonal_compatibility: float = 0.0  # delta -- key/pitch compatibility
    rhythmic_compatibility: float = 0.0  # epsilon -- tempo/groove compatibility
    style_prior_fit: float = 0.0   # zeta -- matches style cluster expectations
    quality_prior: float = 0.0     # eta -- commercial readiness
    user_preference: float = 0.0   # theta -- learned preference boost (Phase 5)
    masking_penalty: float = 0.0   # lambda -- frequency masking risk
    redundancy_penalty: float = 0.0  # mu -- too similar to existing elements


@dataclass
class Recommendation:
    """A single sample recommendation with explanation."""
    filepath: str = ""
    filename: str = ""
    score: float = 0.0
    breakdown: ScoringBreakdown = field(default_factory=ScoringBreakdown)
    explanation: str = ""  # human-readable "why"
    policy: str = ""  # which decision policy triggered this
    need_addressed: str = ""  # which NeedOpportunity this addresses
    role: str = ""  # the sample's role


@dataclass
class RecommendationResult:
    """Full recommendation output for a mix."""
    mix_filepath: str = ""
    recommendations: list[Recommendation] = field(default_factory=list)
    needs_addressed: list[str] = field(default_factory=list)
    total_candidates_considered: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer)
