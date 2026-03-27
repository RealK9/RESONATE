"""
Complete sample profile — the canonical data structure for every analyzed sample.
Every field is optional except filepath, so profiles can be built incrementally.
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json
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
class CoreDescriptors:
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0  # 1=mono, 2=stereo
    rms: float = 0.0
    lufs: float = -100.0
    peak: float = 0.0
    crest_factor: float = 0.0
    attack_time: float = 0.0   # seconds to reach peak
    decay_time: float = 0.0    # seconds from peak to sustain
    sustain_level: float = 0.0 # relative to peak (0-1)


@dataclass
class SpectralDescriptors:
    centroid: float = 0.0
    rolloff: float = 0.0
    flatness: float = 0.0
    contrast: list[float] = field(default_factory=list)  # per-band contrast
    bandwidth: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    harshness_zones: list[float] = field(default_factory=list)  # energy in 2-5kHz bands
    low_energy_distribution: list[float] = field(default_factory=list)  # sub-bands
    sub_to_bass_ratio: float = 0.0  # sub (<60Hz) / bass (60-250Hz)
    resonant_peaks: list[float] = field(default_factory=list)  # Hz of resonant peaks


@dataclass
class HarmonicDescriptors:
    f0: float = 0.0  # fundamental frequency Hz
    pitch_confidence: float = 0.0  # 0-1
    chroma_profile: list[float] = field(default_factory=list)  # 12-bin
    harmonic_to_noise_ratio: float = 0.0  # dB
    inharmonicity: float = 0.0  # 0-1
    overtone_slope: float = 0.0  # dB/octave
    tonalness: float = 0.0  # 0-1 (1=pure tone)
    noisiness: float = 0.0  # 0-1 (1=pure noise)
    dissonance: float = 0.0  # 0-1
    roughness: float = 0.0  # 0-1


@dataclass
class TransientDescriptors:
    onset_count: int = 0
    onset_rate: float = 0.0  # onsets per second
    onset_strength_mean: float = 0.0
    onset_strength_std: float = 0.0
    transient_positions: list[float] = field(default_factory=list)  # seconds
    attack_sharpness: float = 0.0  # how steep the transient is (0-1)
    transient_density: float = 0.0  # transients per second


@dataclass
class PerceptualDescriptors:
    brightness: float = 0.0   # 0-1
    warmth: float = 0.0       # 0-1
    air: float = 0.0          # 0-1
    punch: float = 0.0        # 0-1
    body: float = 0.0         # 0-1
    bite: float = 0.0         # 0-1
    smoothness: float = 0.0   # 0-1
    width: float = 0.0        # 0-1 (0=mono, 1=full stereo)
    depth_impression: float = 0.0  # 0-1


@dataclass
class Embeddings:
    clap_general: list[float] = field(default_factory=list)     # 512-dim
    panns_music: list[float] = field(default_factory=list)      # 2048-dim
    ast_spectrogram: list[float] = field(default_factory=list)  # 768-dim
    panns_tags: dict[str, float] = field(default_factory=dict)  # tag -> confidence
    rpm: list[float] = field(default_factory=list)              # 768-dim RPM unified embedding


@dataclass
class PredictedLabels:
    role: str = "unknown"              # kick/snare/clap/hat/bass/lead/pad/fx/texture/vocal
    role_confidence: float = 0.0
    tonal: bool = False
    is_loop: bool = False
    loop_confidence: float = 0.0
    genre_affinity: dict[str, float] = field(default_factory=dict)   # genre -> 0-1
    era_affinity: dict[str, float] = field(default_factory=dict)     # decade -> 0-1
    commercial_readiness: float = 0.0  # 0-1
    style_tags: dict[str, float] = field(default_factory=dict)       # tag -> confidence

    # RPM model predictions (replaces above when RPM is active)
    rpm_genre_top: str = ""                                          # top-level genre
    rpm_genre_sub: str = ""                                          # sub-genre
    rpm_instruments: list[tuple[str, float]] = field(default_factory=list)  # detected instruments
    rpm_key: str = ""                                                # detected key
    rpm_chord_quality: str = ""                                      # dominant chord quality
    rpm_mode: str = ""                                               # detected mode
    rpm_era: str = ""                                                # decade estimate
    rpm_chart_potential: float = 0.0                                 # 0-1 chart potential


@dataclass
class SampleProfile:
    """Complete profile for a single audio sample."""
    filepath: str = ""
    filename: str = ""
    file_hash: str = ""
    source: str = "local"  # local / splice / loopcloud

    core: CoreDescriptors = field(default_factory=CoreDescriptors)
    spectral: SpectralDescriptors = field(default_factory=SpectralDescriptors)
    harmonic: HarmonicDescriptors = field(default_factory=HarmonicDescriptors)
    transients: TransientDescriptors = field(default_factory=TransientDescriptors)
    perceptual: PerceptualDescriptors = field(default_factory=PerceptualDescriptors)
    embeddings: Embeddings = field(default_factory=Embeddings)
    labels: PredictedLabels = field(default_factory=PredictedLabels)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_serializer)

    @classmethod
    def from_dict(cls, d: dict) -> SampleProfile:
        profile = cls()
        for top_key in ["filepath", "filename", "file_hash", "source"]:
            if top_key in d:
                setattr(profile, top_key, d[top_key])
        sub_map = {
            "core": CoreDescriptors,
            "spectral": SpectralDescriptors,
            "harmonic": HarmonicDescriptors,
            "transients": TransientDescriptors,
            "perceptual": PerceptualDescriptors,
            "embeddings": Embeddings,
            "labels": PredictedLabels,
        }
        for key, klass in sub_map.items():
            if key in d and isinstance(d[key], dict):
                valid_fields = {f.name for f in dataclasses.fields(klass)}
                filtered = {k: v for k, v in d[key].items() if k in valid_fields}
                setattr(profile, key, klass(**filtered))
        return profile
