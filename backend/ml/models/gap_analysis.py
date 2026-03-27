"""
RESONATE — Gap Analysis Data Models

Defines the output structures for the Gap Analysis Engine.
The GapAnalyzer compares a mix against a genre-specific blueprint
and identifies what's missing to make it chart-ready.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class GapItem:
    """A single identified gap between the current mix and chart-ready target."""

    category: str          # "spectral" | "role" | "production" | "dynamics" | "arrangement"
    dimension: str         # human-readable: "808 bass", "hi-hat density", "sub energy"
    current_value: float   # what the mix has now (normalized 0-1 or actual measurement)
    target_value: float    # what a chart-ready track in this genre has
    gap_magnitude: float   # abs(target - current), 0-1
    direction: str         # "increase" | "decrease" | "add" | "remove"
    severity: float        # 0-1, how much this gap hurts chart potential
    message: str           # human-readable explanation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GapItem:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GapAnalysisResult:
    """
    Complete gap analysis output for a mix.

    Scores are 0-100 scale unless noted otherwise.
    """

    # Genre / style detection
    genre_detected: str = ""                    # primary style cluster
    era_detected: str = ""                      # era estimate
    blueprint_name: str = ""                    # which blueprint was matched
    confidence: float = 0.0                     # 0-1, confidence in genre detection

    # Headline scores (0-100)
    production_readiness_score: float = 0.0     # the main number
    chart_potential_current: float = 0.0        # estimated chart potential as-is
    chart_potential_ceiling: float = 0.0        # potential if all gaps filled

    # Genre coherence (0-1)
    genre_coherence_score: float = 0.0          # how internally consistent the genre is

    # Gap details
    gaps: list[GapItem] = field(default_factory=list)  # sorted by severity desc
    missing_roles: list[str] = field(default_factory=list)
    present_roles: list[str] = field(default_factory=list)

    # Summary stats
    total_gaps: int = 0
    critical_gaps: int = 0                      # severity > 0.7
    moderate_gaps: int = 0                      # severity 0.4-0.7
    minor_gaps: int = 0                         # severity < 0.4

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gaps"] = [g.to_dict() if isinstance(g, GapItem) else g for g in self.gaps]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> GapAnalysisResult:
        if not d:
            return cls()
        gaps = [GapItem.from_dict(g) if isinstance(g, dict) else g for g in d.get("gaps", [])]
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "gaps"}
        return cls(gaps=gaps, **filtered)

    @property
    def top_priorities(self) -> list[GapItem]:
        """Return the top 5 most severe gaps."""
        return sorted(self.gaps, key=lambda g: g.severity, reverse=True)[:5]

    @property
    def summary(self) -> str:
        """Human-readable one-liner."""
        if self.production_readiness_score >= 85:
            return f"Nearly chart-ready ({self.production_readiness_score:.0f}/100). Minor tweaks needed."
        elif self.production_readiness_score >= 60:
            return (
                f"Good foundation ({self.production_readiness_score:.0f}/100). "
                f"{self.critical_gaps} critical gaps to address."
            )
        elif self.production_readiness_score >= 35:
            return (
                f"Work in progress ({self.production_readiness_score:.0f}/100). "
                f"Missing {len(self.missing_roles)} key elements."
            )
        else:
            return (
                f"Early stage ({self.production_readiness_score:.0f}/100). "
                f"Needs core elements: {', '.join(self.missing_roles[:3])}."
            )
