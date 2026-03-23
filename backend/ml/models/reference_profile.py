"""
Reference profile — style-aware priors learned from commercial reference tracks.

Contains StylePrior (production norms for a single style cluster) and
ReferenceCorpus (collection of priors across all style clusters).  These are
the foundation for mix comparison: Phase 3's reference model tells the needs
engine what a *good* mix in a given style should look like.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


# ---------------------------------------------------------------------------
# StylePrior
# ---------------------------------------------------------------------------

@dataclass
class StylePrior:
    """Learned production norms for a style cluster."""

    cluster_name: str = ""

    # Spectral norms (10-band means and acceptable ranges)
    target_spectral_mean: list[float] = field(default_factory=list)  # 10 bands, expected mean energy
    target_spectral_std: list[float] = field(default_factory=list)   # acceptable deviation per band

    # Density norms
    target_density_mean: float = 0.0
    target_density_range: tuple[float, float] = (0.0, 1.0)

    # Role co-occurrence (which roles typically appear together)
    typical_roles: dict[str, float] = field(default_factory=dict)  # role -> expected confidence (0-1)

    # Width norms
    target_width_by_band: list[float] = field(default_factory=list)  # 10 bands
    target_overall_width: float = 0.0

    # Arrangement patterns
    section_lift_pattern: list[float] = field(default_factory=list)  # expected energy arc (8 segments, normalized)
    arrangement_density_range: tuple[float, float] = (0.0, 1.0)

    # Tonal/harmonic
    tonal_complexity: float = 0.0  # 0=simple, 1=complex
    layering_depth: float = 0.0    # expected number of active layers, normalized

    # Complementary elements (what samples typically fill gaps)
    common_complements: list[str] = field(default_factory=list)  # role names that commonly complement this style

    # Metadata
    reference_count: int = 0  # how many reference tracks contributed
    confidence: float = 0.0   # 0-1, how confident the prior is

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuples to lists for JSON compatibility
        d["target_density_range"] = list(self.target_density_range)
        d["arrangement_density_range"] = list(self.arrangement_density_range)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StylePrior:
        prior = cls()
        for k, v in d.items():
            if k in ("target_density_range", "arrangement_density_range"):
                setattr(prior, k, tuple(v) if isinstance(v, list) else v)
            elif hasattr(prior, k):
                setattr(prior, k, v)
        return prior


# ---------------------------------------------------------------------------
# ReferenceCorpus
# ---------------------------------------------------------------------------

@dataclass
class ReferenceCorpus:
    """Collection of style priors from reference tracks."""

    priors: dict[str, StylePrior] = field(default_factory=dict)  # cluster_name -> StylePrior
    version: str = "1.0"
    total_references: int = 0

    def get_prior(self, cluster: str) -> StylePrior | None:
        """Return the StylePrior for *cluster*, or None if not present."""
        return self.priors.get(cluster)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "total_references": self.total_references,
            "priors": {name: prior.to_dict() for name, prior in self.priors.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> ReferenceCorpus:
        priors: dict[str, StylePrior] = {}
        for name, prior_dict in d.get("priors", {}).items():
            priors[name] = StylePrior.from_dict(prior_dict)
        return cls(
            priors=priors,
            version=d.get("version", "1.0"),
            total_references=d.get("total_references", 0),
        )
