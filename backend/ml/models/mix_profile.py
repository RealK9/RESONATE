"""
Mix profile — the canonical data structure for an analyzed mix.

Contains mix-level analysis (BPM, key, loudness), spectral occupancy,
stereo width, source-role presence, style cluster classification,
need/opportunity diagnosis, and density map.
"""
from __future__ import annotations

import dataclasses
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
class MixLevelAnalysis:
    """Core mix-level descriptors."""
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    key: str = ""
    key_confidence: float = 0.0
    tonal_center: float = 0.0  # Hz
    harmonic_density: float = 0.0  # 0-1
    duration: float = 0.0
    loudness_lufs: float = -100.0
    loudness_range: float = 0.0  # LRA in dB
    peak: float = 0.0
    dynamic_range: float = 0.0  # crest factor dB
    section_energy: list[float] = field(default_factory=list)


@dataclass
class SpectralOccupancy:
    """Spectral energy by band over time."""
    bands: list[str] = field(default_factory=list)
    time_frames: int = 0
    occupancy_matrix: list[list[float]] = field(default_factory=list)  # [band][time]
    mean_by_band: list[float] = field(default_factory=list)


@dataclass
class StereoWidth:
    """Stereo width analysis by frequency band."""
    bands: list[str] = field(default_factory=list)
    width_by_band: list[float] = field(default_factory=list)
    overall_width: float = 0.0
    correlation: float = 0.0


@dataclass
class SourceRolePresence:
    """Estimated presence and confidence of each sound role in the mix."""
    roles: dict[str, float] = field(default_factory=dict)
    # Expected roles: kick, snare_clap, hats_tops, bass, lead,
    # chord_support, pad, vocal_texture, fx_transitions, ambience


@dataclass
class StyleCluster:
    """Style cluster classification."""
    cluster_probabilities: dict[str, float] = field(default_factory=dict)
    primary_cluster: str = ""
    era_estimate: str = ""


@dataclass
class NeedOpportunity:
    """A single diagnosed need or opportunity."""
    category: str = ""  # spectral/role/dynamic/spatial/arrangement
    description: str = ""
    severity: float = 0.0  # 0-1
    recommendation_policy: str = ""  # fill_missing_role/reinforce_existing/etc


@dataclass
class MixProfile:
    """Complete profile for an analyzed mix."""
    filepath: str = ""
    filename: str = ""
    analysis: MixLevelAnalysis = field(default_factory=MixLevelAnalysis)
    spectral_occupancy: SpectralOccupancy = field(default_factory=SpectralOccupancy)
    stereo_width: StereoWidth = field(default_factory=StereoWidth)
    source_roles: SourceRolePresence = field(default_factory=SourceRolePresence)
    style: StyleCluster = field(default_factory=StyleCluster)
    needs: list[NeedOpportunity] = field(default_factory=list)
    density_map: list[float] = field(default_factory=list)
    gap_analysis: dict = field(default_factory=dict)  # serialized GapAnalysisResult
    rpm_embedding: list[float] = field(default_factory=list)  # 768-d RPM embedding for FAISS query

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer)

    @classmethod
    def from_dict(cls, d: dict) -> MixProfile:
        profile = cls()
        for top_key in ["filepath", "filename"]:
            if top_key in d:
                setattr(profile, top_key, d[top_key])

        # Reconstruct nested dataclass fields
        sub_map = {
            "analysis": MixLevelAnalysis,
            "spectral_occupancy": SpectralOccupancy,
            "stereo_width": StereoWidth,
            "source_roles": SourceRolePresence,
            "style": StyleCluster,
        }
        for key, klass in sub_map.items():
            if key in d and isinstance(d[key], dict):
                valid_fields = {f.name for f in dataclasses.fields(klass)}
                filtered = {k: v for k, v in d[key].items() if k in valid_fields}
                setattr(profile, key, klass(**filtered))

        # Reconstruct needs list
        if "needs" in d and isinstance(d["needs"], list):
            need_fields = {f.name for f in dataclasses.fields(NeedOpportunity)}
            profile.needs = [
                NeedOpportunity(**{k: v for k, v in item.items() if k in need_fields})
                for item in d["needs"]
                if isinstance(item, dict)
            ]

        # Reconstruct density_map
        if "density_map" in d and isinstance(d["density_map"], list):
            profile.density_map = d["density_map"]

        return profile
