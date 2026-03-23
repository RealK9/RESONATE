# Phase 1 — Sample Analyzer Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rich profile database for every sample in the user's library — extracting DSP features, spectral/harmonic/perceptual descriptors, learned embeddings, and predicted labels — then index them for fast retrieval.

**Architecture:** New `backend/ml/` package houses all analysis, embedding, classification, and retrieval modules — coexisting cleanly with the existing `backend/analysis/` pipeline. Each module is a pure-function library with no framework coupling. A new `backend/ml/retrieval/` module builds vector indexes. The existing SQLite database gets a new `sample_profiles` table with JSONB-style columns for rich feature storage. An ingestion pipeline orchestrates extraction → embedding → classification → storage. Pretrained models (CLAP, PANNs, AST) run locally via PyTorch.

**Tech Stack:** Python 3.11+, NumPy, SciPy, librosa, pyloudnorm, librosa (pyin for pitch, CREPE optional), laion-clap, panns-inference, transformers (AST), scikit-learn, FAISS, SQLite (existing), pytest

---

## File Structure

```
backend/ml/
├── __init__.py
├── analysis/
│   ├── __init__.py
│   ├── core_descriptors.py      # duration, SR, channels, RMS, LUFS, peak, crest, envelope
│   ├── spectral_descriptors.py  # centroid, rolloff, flatness, contrast, bandwidth, skew/kurt, harshness, sub-bass
│   ├── harmonic_descriptors.py  # F0, pitch confidence, chroma, HNR, inharmonicity, overtones, tonalness, dissonance
│   ├── transient_descriptors.py # onset detection, transient positions, onset strength, attack/decay/sustain
│   ├── perceptual_descriptors.py# brightness, warmth, air, punch, body, bite, smoothness, width, depth
│   └── loop_detection.py        # one-shot vs loop classification
├── embeddings/
│   ├── __init__.py
│   ├── clap_embeddings.py       # CLAP general audio embeddings
│   ├── panns_embeddings.py      # PANNs audio tagging + embeddings
│   ├── ast_embeddings.py        # AST spectrogram transformer embeddings
│   └── embedding_manager.py     # orchestrates all embedding extractors
├── classifiers/
│   ├── __init__.py
│   ├── role_classifier.py       # kick/snare/clap/hat/bass/lead/pad/fx/texture/vocal
│   ├── genre_era_classifier.py  # genre affinity, era affinity
│   ├── style_tagger.py          # bright/dark/wide/punchy/analog/digital/gritty/clean
│   └── quality_scorer.py        # commercial readiness score
├── retrieval/
│   ├── __init__.py
│   └── vector_index.py          # FAISS index build + search
├── pipeline/
│   ├── __init__.py
│   ├── ingestion.py             # orchestrates full analysis pipeline per sample
│   └── batch_processor.py       # parallel batch processing with progress
├── db/
│   ├── __init__.py
│   └── sample_store.py          # sample_profiles table CRUD
└── models/
    └── sample_profile.py        # dataclass for the full sample profile

tests/
├── conftest.py                  # shared fixtures, test audio generation
├── test_core_descriptors.py
├── test_spectral_descriptors.py
├── test_harmonic_descriptors.py
├── test_transient_descriptors.py
├── test_perceptual_descriptors.py
├── test_loop_detection.py
├── test_clap_embeddings.py
├── test_panns_embeddings.py
├── test_ast_embeddings.py
├── test_role_classifier.py
├── test_genre_era_classifier.py
├── test_style_tagger.py
├── test_quality_scorer.py
├── test_vector_index.py
├── test_sample_store.py
├── test_ingestion.py
└── test_batch_processor.py
```

---

## Task 1: Test Infrastructure + Sample Profile Dataclass

**Files:**
- Create: `backend/ml/__init__.py`
- Create: `backend/ml/models/__init__.py`
- Create: `backend/ml/models/sample_profile.py`
- Create: `backend/ml/analysis/__init__.py`
- Create: `backend/ml/embeddings/__init__.py`
- Create: `backend/ml/classifiers/__init__.py`
- Create: `backend/ml/retrieval/__init__.py`
- Create: `backend/ml/pipeline/__init__.py`
- Create: `backend/ml/db/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_sample_profile.py`
- Create: `requirements-ml.txt`
- Create: `pyproject.toml` (pytest config)

- [ ] **Step 1: Create requirements file**

```
# requirements-ml.txt — Phase 1 ML/DSP dependencies
numpy>=1.24
scipy>=1.11
librosa>=0.10.1
pyloudnorm>=0.1.1
scikit-learn>=1.3
faiss-cpu>=1.7.4
torch>=2.1
torchaudio>=2.1
transformers>=4.35
laion-clap>=1.1.4
panns-inference>=0.1.1
pytest>=7.4
pytest-asyncio>=0.21
soundfile>=0.12
# Optional: crepe for neural pitch detection (requires tensorflow)
# crepe>=0.0.16
# tensorflow>=2.14
```

- [ ] **Step 2: Create pyproject.toml with pytest path config**

```toml
# pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements-ml.txt`
Expected: All packages install successfully.

- [ ] **Step 4: Create ALL __init__.py files upfront (prevents race conditions in parallel tasks)**

```python
# backend/ml/__init__.py
# (empty)
```

Create empty `__init__.py` in each of:
- `backend/ml/analysis/__init__.py`
- `backend/ml/embeddings/__init__.py`
- `backend/ml/classifiers/__init__.py`
- `backend/ml/retrieval/__init__.py`
- `backend/ml/pipeline/__init__.py`
- `backend/ml/db/__init__.py`

```python
# backend/ml/models/__init__.py
from backend.ml.models.sample_profile import SampleProfile
```

- [ ] **Step 5: Write the SampleProfile dataclass**

```python
# backend/ml/models/sample_profile.py
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
```

- [ ] **Step 4: Write test for SampleProfile**

```python
# tests/test_sample_profile.py
import json
import numpy as np
from src.models.sample_profile import (
    SampleProfile, CoreDescriptors, SpectralDescriptors,
    HarmonicDescriptors, TransientDescriptors, PerceptualDescriptors,
    Embeddings, PredictedLabels,
)


def test_default_construction():
    p = SampleProfile()
    assert p.filepath == ""
    assert p.core.duration == 0.0
    assert p.labels.role == "unknown"


def test_to_dict_roundtrip():
    p = SampleProfile(filepath="/test.wav", filename="test.wav")
    p.core.duration = 2.5
    p.core.lufs = -14.0
    p.spectral.centroid = 3000.0
    p.labels.role = "kick"
    p.labels.genre_affinity = {"trap": 0.8, "house": 0.3}
    d = p.to_dict()
    restored = SampleProfile.from_dict(d)
    assert restored.filepath == "/test.wav"
    assert restored.core.duration == 2.5
    assert restored.labels.role == "kick"
    assert restored.labels.genre_affinity["trap"] == 0.8


def test_json_serialization_with_numpy():
    p = SampleProfile(filepath="/test.wav")
    p.spectral.contrast = np.array([1.0, 2.0, 3.0])
    p.harmonic.chroma_profile = np.zeros(12, dtype=np.float32).tolist()
    j = p.to_json()
    parsed = json.loads(j)
    assert parsed["spectral"]["contrast"] == [1.0, 2.0, 3.0]


def test_from_dict_ignores_unknown_keys():
    d = {"filepath": "/x.wav", "core": {"duration": 1.0, "unknown_field": 999}}
    p = SampleProfile.from_dict(d)
    assert p.core.duration == 1.0
    assert p.filepath == "/x.wav"
```

- [ ] **Step 7: Create conftest with test audio fixtures**

All audio fixtures use `session` scope with `tmp_path_factory` so they can be shared
with session/module-scoped embedding model fixtures without scope mismatch.

```python
# tests/conftest.py
"""Shared fixtures for all tests. Generates synthetic audio for deterministic testing."""
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

SR = 44100


@pytest.fixture(scope="session")
def sample_rate():
    return SR


@pytest.fixture(scope="session")
def sine_440hz(tmp_path_factory):
    """1-second 440Hz sine wave, mono, 44.1kHz."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path_factory.mktemp("audio") / "sine_440.wav"
    sf.write(str(path), audio, SR)
    return path


@pytest.fixture(scope="session")
def stereo_noise(tmp_path_factory):
    """0.5-second stereo white noise."""
    rng = np.random.default_rng(42)
    audio = (0.3 * rng.standard_normal((SR // 2, 2))).astype(np.float32)
    path = tmp_path_factory.mktemp("audio") / "stereo_noise.wav"
    sf.write(str(path), audio, SR)
    return path


@pytest.fixture(scope="session")
def kick_like(tmp_path_factory):
    """Synthetic kick drum: short sine sweep with fast decay."""
    duration = 0.15
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    freq = 150 * np.exp(-30 * t) + 40  # pitch sweep down
    phase = 2 * np.pi * np.cumsum(freq) / SR
    envelope = np.exp(-20 * t)
    audio = (0.9 * envelope * np.sin(phase)).astype(np.float32)
    path = tmp_path_factory.mktemp("audio") / "kick.wav"
    sf.write(str(path), audio, SR)
    return path


@pytest.fixture(scope="session")
def silence(tmp_path_factory):
    """1-second silence."""
    audio = np.zeros(SR, dtype=np.float32)
    path = tmp_path_factory.mktemp("audio") / "silence.wav"
    sf.write(str(path), audio, SR)
    return path


@pytest.fixture(scope="session")
def pad_like(tmp_path_factory):
    """2-second rich pad: stacked detuned sines with slow attack."""
    duration = 2.0
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    envelope = 1.0 - np.exp(-2 * t)
    freqs = [220, 220.5, 330, 440.3, 554, 660.2]
    audio = np.zeros(n, dtype=np.float64)
    for f in freqs:
        audio += np.sin(2 * np.pi * f * t) / len(freqs)
    audio = (0.4 * envelope * audio).astype(np.float32)
    left = audio
    right = np.roll(audio, int(0.003 * SR))
    stereo = np.stack([left, right], axis=-1)
    path = tmp_path_factory.mktemp("audio") / "pad.wav"
    sf.write(str(path), stereo, SR)
    return path


@pytest.fixture(scope="session")
def hihat_like(tmp_path_factory):
    """Synthetic hi-hat: filtered noise burst."""
    duration = 0.08
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = np.random.default_rng(123)
    noise = rng.standard_normal(n)
    envelope = np.exp(-60 * t)
    audio = (0.5 * envelope * noise).astype(np.float32)
    path = tmp_path_factory.mktemp("audio") / "hihat.wav"
    sf.write(str(path), audio, SR)
    return path
```

- [ ] **Step 8: Run tests to verify**

Run: `cd /path/to/RESONATE && python -m pytest tests/test_sample_profile.py -v`
Expected: 4 tests PASS

- [ ] **Step 9: Commit**

```bash
git add backend/ml/ tests/ requirements-ml.txt pyproject.toml
git commit -m "feat(phase1): add SampleProfile dataclass and test infrastructure"
```

---

## Task 2: Core Descriptors Extraction

**Files:**
- Create: `backend/ml/analysis/__init__.py`
- Create: `backend/ml/analysis/core_descriptors.py`
- Create: `tests/test_core_descriptors.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core_descriptors.py
import numpy as np
from src.analysis.core_descriptors import extract_core_descriptors
from src.models.sample_profile import CoreDescriptors


def test_sine_duration(sine_440hz, sample_rate):
    result = extract_core_descriptors(str(sine_440hz))
    assert isinstance(result, CoreDescriptors)
    assert abs(result.duration - 1.0) < 0.01


def test_sine_sample_rate(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    assert result.sample_rate == 44100


def test_mono_detection(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    assert result.channels == 1


def test_stereo_detection(stereo_noise):
    result = extract_core_descriptors(str(stereo_noise))
    assert result.channels == 2


def test_rms_nonzero_for_signal(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    assert result.rms > 0.1


def test_rms_near_zero_for_silence(silence):
    result = extract_core_descriptors(str(silence))
    assert result.rms < 0.001


def test_lufs_reasonable_range(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    # 440Hz sine at 0.5 amplitude should be roughly -10 to -6 LUFS
    assert -20.0 < result.lufs < 0.0


def test_peak_amplitude(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    assert 0.4 < result.peak < 0.6  # sine at 0.5 amplitude


def test_crest_factor_sine(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    # Sine wave crest factor = sqrt(2) ≈ 3.01 dB
    assert 2.0 < result.crest_factor < 5.0


def test_kick_fast_attack(kick_like):
    result = extract_core_descriptors(str(kick_like))
    assert result.attack_time < 0.01  # kick has near-instant attack


def test_pad_slow_attack(pad_like):
    result = extract_core_descriptors(str(pad_like))
    assert result.attack_time > 0.05  # pad has slow attack


def test_kick_fast_decay(kick_like):
    result = extract_core_descriptors(str(kick_like))
    assert result.decay_time < 0.2


def test_sustain_level_range(sine_440hz):
    result = extract_core_descriptors(str(sine_440hz))
    assert 0.0 <= result.sustain_level <= 1.0
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `python -m pytest tests/test_core_descriptors.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement core_descriptors.py**

```python
# backend/ml/analysis/__init__.py
# (empty)
```

```python
# backend/ml/analysis/core_descriptors.py
"""
Core audio descriptors: duration, sample rate, channels, loudness metrics, envelope.
All computed via direct DSP — no ML models needed.
"""
from __future__ import annotations
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from src.models.sample_profile import CoreDescriptors


def extract_core_descriptors(filepath: str) -> CoreDescriptors:
    """Extract all core descriptors from an audio file."""
    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
    n_channels = audio.shape[1]

    # Mix to mono for analysis
    mono = audio.mean(axis=1)

    duration = len(mono) / sr
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))

    # LUFS (ITU-R BS.1770)
    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(audio))
    except Exception:
        lufs = -100.0
    if np.isinf(lufs) or np.isnan(lufs):
        lufs = -100.0

    # Crest factor (peak-to-RMS ratio in dB)
    if rms > 1e-10:
        crest_factor = float(20 * np.log10(peak / rms))
    else:
        crest_factor = 0.0

    # Envelope analysis for attack/decay/sustain
    attack_time, decay_time, sustain_level = _envelope_profile(mono, sr)

    return CoreDescriptors(
        duration=round(duration, 4),
        sample_rate=sr,
        channels=n_channels,
        rms=round(rms, 6),
        lufs=round(lufs, 2),
        peak=round(peak, 6),
        crest_factor=round(crest_factor, 2),
        attack_time=round(attack_time, 4),
        decay_time=round(decay_time, 4),
        sustain_level=round(sustain_level, 4),
    )


def _envelope_profile(mono: np.ndarray, sr: int) -> tuple[float, float, float]:
    """
    Compute attack/decay/sustain from the amplitude envelope.
    Uses a smoothed RMS envelope with ~10ms windows.
    """
    hop = max(1, sr // 100)  # ~10ms
    frame_len = hop * 2
    n_frames = len(mono) // hop

    if n_frames < 3:
        return 0.0, 0.0, 0.0

    # Compute RMS envelope
    envelope = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(mono))
        frame = mono[start:end]
        envelope[i] = np.sqrt(np.mean(frame ** 2))

    if envelope.max() < 1e-10:
        return 0.0, 0.0, 0.0

    # Normalize envelope
    env_norm = envelope / envelope.max()

    # Attack: time from start to first frame reaching 90% of peak
    peak_idx = np.argmax(envelope)
    threshold_90 = 0.9
    attack_frames = 0
    for i in range(peak_idx + 1):
        if env_norm[i] >= threshold_90:
            attack_frames = i
            break
    attack_time = float(attack_frames * hop / sr)

    # Sustain level: median of the last third of the sound
    last_third_start = max(peak_idx, n_frames * 2 // 3)
    if last_third_start < n_frames:
        sustain_level = float(np.median(env_norm[last_third_start:]))
    else:
        sustain_level = float(env_norm[-1])

    # Decay: time from peak to first crossing below sustain_level + 10%
    decay_threshold = min(sustain_level + 0.1, 0.95)
    decay_frames = 0
    for i in range(peak_idx, n_frames):
        if env_norm[i] <= decay_threshold:
            decay_frames = i - peak_idx
            break
    decay_time = float(decay_frames * hop / sr)

    return attack_time, decay_time, sustain_level
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_core_descriptors.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/analysis/ tests/test_core_descriptors.py
git commit -m "feat(phase1): core descriptors — duration, RMS, LUFS, peak, crest, envelope"
```

---

## Task 3: Spectral Descriptors Extraction

**Files:**
- Create: `backend/ml/analysis/spectral_descriptors.py`
- Create: `tests/test_spectral_descriptors.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_spectral_descriptors.py
import numpy as np
from src.analysis.spectral_descriptors import extract_spectral_descriptors
from src.models.sample_profile import SpectralDescriptors


def test_returns_spectral_descriptors(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert isinstance(result, SpectralDescriptors)


def test_centroid_for_sine(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    # 440Hz sine should have centroid near 440Hz
    assert 400 < result.centroid < 500


def test_centroid_for_noise(stereo_noise):
    result = extract_spectral_descriptors(str(stereo_noise))
    # White noise has centroid roughly at sr/4
    assert result.centroid > 2000


def test_flatness_high_for_noise(stereo_noise):
    result = extract_spectral_descriptors(str(stereo_noise))
    # Noise is spectrally flat
    assert result.flatness > 0.1


def test_flatness_low_for_sine(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    # Pure sine is spectrally concentrated
    assert result.flatness < 0.05


def test_rolloff_reasonable(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert result.rolloff > 0


def test_bandwidth_positive(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert result.bandwidth >= 0


def test_contrast_is_list(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert isinstance(result.contrast, list)
    assert len(result.contrast) >= 6  # librosa default 7 bands


def test_sub_to_bass_ratio_for_kick(kick_like):
    result = extract_spectral_descriptors(str(kick_like))
    # Kick has sub/bass energy
    assert result.sub_to_bass_ratio > 0


def test_harshness_zones_list(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert isinstance(result.harshness_zones, list)
    assert len(result.harshness_zones) >= 3  # multiple harsh bands


def test_resonant_peaks_list(sine_440hz):
    result = extract_spectral_descriptors(str(sine_440hz))
    assert isinstance(result.resonant_peaks, list)
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `python -m pytest tests/test_spectral_descriptors.py -v`
Expected: FAIL

- [ ] **Step 3: Implement spectral_descriptors.py**

```python
# backend/ml/analysis/spectral_descriptors.py
"""
Spectral descriptors: centroid, rolloff, flatness, contrast, bandwidth,
skewness/kurtosis, harshness zones, low-end distribution, sub-to-bass ratio,
resonant peak analysis.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf
from scipy import stats as scipy_stats
from scipy.signal import find_peaks
from src.models.sample_profile import SpectralDescriptors


def extract_spectral_descriptors(filepath: str) -> SpectralDescriptors:
    """Extract all spectral descriptors from an audio file."""
    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    # Compute STFT
    n_fft = 2048
    hop_length = 512
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Spectral centroid (mean across frames)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=n_fft)[0]
    centroid_mean = float(np.mean(centroid))

    # Spectral rolloff (85% energy threshold)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    rolloff_mean = float(np.mean(rolloff))

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    flatness_mean = float(np.mean(flatness))

    # Spectral contrast (7 bands by default)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=n_fft)
    contrast_mean = [float(np.mean(contrast[i])) for i in range(contrast.shape[0])]

    # Spectral bandwidth
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=n_fft)[0]
    bandwidth_mean = float(np.mean(bw))

    # Spectral skewness and kurtosis from the mean power spectrum
    mean_spectrum = np.mean(S ** 2, axis=1)
    if mean_spectrum.sum() > 1e-10:
        # Normalize to probability distribution
        p = mean_spectrum / mean_spectrum.sum()
        mu = np.sum(freqs * p)
        variance = np.sum(((freqs - mu) ** 2) * p)
        std = np.sqrt(variance) if variance > 0 else 1e-10
        skewness = float(np.sum(((freqs - mu) / std) ** 3 * p))
        kurtosis = float(np.sum(((freqs - mu) / std) ** 4 * p))
    else:
        skewness = 0.0
        kurtosis = 0.0

    # Harshness zones (energy in 2-5kHz sub-bands)
    harshness_zones = _band_energy(mean_spectrum, freqs, [
        (2000, 2500), (2500, 3000), (3000, 3500), (3500, 4000), (4000, 5000)
    ])

    # Low-end energy distribution
    low_energy_distribution = _band_energy(mean_spectrum, freqs, [
        (20, 40), (40, 60), (60, 100), (100, 150), (150, 250)
    ])

    # Sub-to-bass ratio
    sub_energy = np.sum(mean_spectrum[(freqs >= 20) & (freqs < 60)])
    bass_energy = np.sum(mean_spectrum[(freqs >= 60) & (freqs < 250)])
    sub_to_bass = float(sub_energy / bass_energy) if bass_energy > 1e-10 else 0.0

    # Resonant peak analysis
    resonant_peaks = _find_resonant_peaks(mean_spectrum, freqs)

    return SpectralDescriptors(
        centroid=round(centroid_mean, 2),
        rolloff=round(rolloff_mean, 2),
        flatness=round(flatness_mean, 6),
        contrast=contrast_mean,
        bandwidth=round(bandwidth_mean, 2),
        skewness=round(skewness, 4),
        kurtosis=round(kurtosis, 4),
        harshness_zones=harshness_zones,
        low_energy_distribution=low_energy_distribution,
        sub_to_bass_ratio=round(sub_to_bass, 4),
        resonant_peaks=resonant_peaks,
    )


def _band_energy(spectrum: np.ndarray, freqs: np.ndarray,
                 bands: list[tuple[float, float]]) -> list[float]:
    """Compute normalized energy in each frequency band."""
    total = spectrum.sum()
    if total < 1e-10:
        return [0.0] * len(bands)
    energies = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        energies.append(round(float(spectrum[mask].sum() / total), 6))
    return energies


def _find_resonant_peaks(spectrum: np.ndarray, freqs: np.ndarray,
                         min_prominence: float = 0.1) -> list[float]:
    """Find prominent spectral peaks indicating resonances."""
    if spectrum.max() < 1e-10:
        return []
    norm = spectrum / spectrum.max()
    # Smooth slightly to avoid noise peaks
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(norm, size=5)
    peaks, properties = find_peaks(smoothed, prominence=min_prominence, distance=5)
    # Return frequencies of top peaks (max 10)
    if len(peaks) == 0:
        return []
    prominences = properties["prominences"]
    top_idx = np.argsort(prominences)[::-1][:10]
    return [round(float(freqs[peaks[i]]), 1) for i in top_idx]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_spectral_descriptors.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/analysis/spectral_descriptors.py tests/test_spectral_descriptors.py
git commit -m "feat(phase1): spectral descriptors — centroid, rolloff, flatness, contrast, harshness, sub-bass"
```

---

## Task 4: Harmonic / Pitch Descriptors

**Files:**
- Create: `backend/ml/analysis/harmonic_descriptors.py`
- Create: `tests/test_harmonic_descriptors.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_harmonic_descriptors.py
import numpy as np
from src.analysis.harmonic_descriptors import extract_harmonic_descriptors
from src.models.sample_profile import HarmonicDescriptors


def test_returns_harmonic_descriptors(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert isinstance(result, HarmonicDescriptors)


def test_f0_for_sine(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    # Should detect ~440Hz
    assert 420 < result.f0 < 460


def test_pitch_confidence_high_for_sine(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert result.pitch_confidence > 0.7


def test_pitch_confidence_low_for_noise(stereo_noise):
    result = extract_harmonic_descriptors(str(stereo_noise))
    assert result.pitch_confidence < 0.5


def test_chroma_profile_length(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert len(result.chroma_profile) == 12


def test_chroma_a_dominant_for_440(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    # A=440Hz, chroma index 9 (A)
    chroma = result.chroma_profile
    a_idx = 9
    assert chroma[a_idx] == max(chroma) or chroma[a_idx] > 0.5


def test_hnr_high_for_sine(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert result.harmonic_to_noise_ratio > 10  # dB


def test_hnr_low_for_noise(stereo_noise):
    result = extract_harmonic_descriptors(str(stereo_noise))
    assert result.harmonic_to_noise_ratio < 10


def test_tonalness_high_for_sine(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert result.tonalness > 0.5


def test_noisiness_high_for_noise(stereo_noise):
    result = extract_harmonic_descriptors(str(stereo_noise))
    assert result.noisiness > 0.3


def test_dissonance_range(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert 0.0 <= result.dissonance <= 1.0


def test_roughness_range(sine_440hz):
    result = extract_harmonic_descriptors(str(sine_440hz))
    assert 0.0 <= result.roughness <= 1.0
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_harmonic_descriptors.py -v`
Expected: FAIL

- [ ] **Step 3: Implement harmonic_descriptors.py**

```python
# backend/ml/analysis/harmonic_descriptors.py
"""
Harmonic/pitch descriptors: F0, pitch confidence, chroma, HNR, inharmonicity,
overtone slope, tonalness/noisiness, dissonance/roughness.
Uses librosa pyin for pitch by default (CREPE optional if tensorflow installed).
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf
from src.models.sample_profile import HarmonicDescriptors


def extract_harmonic_descriptors(filepath: str) -> HarmonicDescriptors:
    """Extract all harmonic/pitch descriptors from an audio file."""
    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    # Resample to 16kHz for CREPE (required)
    mono_16k = librosa.resample(mono, orig_sr=sr, target_sr=16000)

    # F0 estimation via CREPE
    f0, confidence = _estimate_pitch(mono_16k)

    # Chroma profile
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr, n_fft=2048, hop_length=512)
    chroma_profile = [round(float(np.mean(chroma[i])), 4) for i in range(12)]

    # Harmonic-to-noise ratio
    hnr = _compute_hnr(mono, sr)

    # Harmonic/percussive separation for tonalness
    H, P = librosa.decompose.hpss(librosa.stft(mono))
    h_energy = float(np.sum(np.abs(H) ** 2))
    p_energy = float(np.sum(np.abs(P) ** 2))
    total_energy = h_energy + p_energy
    tonalness = h_energy / total_energy if total_energy > 1e-10 else 0.0
    noisiness = p_energy / total_energy if total_energy > 1e-10 else 0.0

    # Inharmonicity
    inharmonicity = _compute_inharmonicity(mono, sr, f0)

    # Overtone slope
    overtone_slope = _compute_overtone_slope(mono, sr, f0)

    # Dissonance and roughness
    dissonance = _compute_dissonance(mono, sr)
    roughness = _compute_roughness(mono, sr)

    return HarmonicDescriptors(
        f0=round(f0, 2),
        pitch_confidence=round(confidence, 4),
        chroma_profile=chroma_profile,
        harmonic_to_noise_ratio=round(hnr, 2),
        inharmonicity=round(inharmonicity, 4),
        overtone_slope=round(overtone_slope, 2),
        tonalness=round(tonalness, 4),
        noisiness=round(noisiness, 4),
        dissonance=round(dissonance, 4),
        roughness=round(roughness, 4),
    )


def _estimate_pitch(mono_16k: np.ndarray) -> tuple[float, float]:
    """Estimate F0 using librosa pyin (default) or CREPE (if available)."""
    # Try CREPE first if installed (higher quality, requires tensorflow)
    try:
        import crepe
        time_arr, frequency, confidence, _ = crepe.predict(
            mono_16k, 16000, model_capacity="full", viterbi=True, step_size=10
        )
        mask = confidence > 0.3
        if mask.sum() > 0:
            return float(np.median(frequency[mask])), float(np.mean(confidence[mask]))
        return 0.0, 0.0
    except Exception:
        pass

    # Default: librosa pyin (no extra dependencies)
    f0_arr, voiced_flag, voiced_prob = librosa.pyin(
        mono_16k, fmin=30, fmax=4000, sr=16000
    )
    valid = ~np.isnan(f0_arr)
    if valid.sum() > 0:
        return float(np.median(f0_arr[valid])), float(np.mean(voiced_prob[valid]))
    return 0.0, 0.0


def _compute_hnr(mono: np.ndarray, sr: int) -> float:
    """Compute harmonic-to-noise ratio in dB using autocorrelation."""
    # Use librosa's harmonic/percussive decomposition as proxy
    S = librosa.stft(mono)
    H, P = librosa.decompose.hpss(S)
    h_power = np.sum(np.abs(H) ** 2)
    p_power = np.sum(np.abs(P) ** 2)
    if p_power < 1e-10:
        return 40.0  # Essentially pure harmonic
    return float(10 * np.log10(h_power / p_power))


def _compute_inharmonicity(mono: np.ndarray, sr: int, f0: float) -> float:
    """Measure deviation of partials from perfect harmonic series."""
    if f0 < 20:
        return 0.0

    S = np.abs(librosa.stft(mono, n_fft=4096))
    mean_spec = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    deviations = []
    for n in range(1, 9):  # Check first 8 harmonics
        expected = f0 * n
        if expected > sr / 2:
            break
        # Find nearest peak to expected harmonic
        window = (freqs > expected * 0.9) & (freqs < expected * 1.1)
        if window.sum() == 0:
            continue
        local_spec = mean_spec[window]
        local_freqs = freqs[window]
        peak_idx = np.argmax(local_spec)
        actual = local_freqs[peak_idx]
        if local_spec[peak_idx] > mean_spec.max() * 0.01:  # Above noise floor
            deviations.append(abs(actual - expected) / expected)

    if len(deviations) < 2:
        return 0.0
    return float(np.clip(np.mean(deviations), 0, 1))


def _compute_overtone_slope(mono: np.ndarray, sr: int, f0: float) -> float:
    """Compute how quickly overtone energy falls off (dB/octave)."""
    if f0 < 20:
        return 0.0

    S = np.abs(librosa.stft(mono, n_fft=4096))
    mean_spec = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    harmonics_db = []
    for n in range(1, 9):
        freq = f0 * n
        if freq > sr / 2:
            break
        idx = np.argmin(np.abs(freqs - freq))
        power = mean_spec[idx]
        if power > 1e-10:
            harmonics_db.append((np.log2(n), 20 * np.log10(power)))

    if len(harmonics_db) < 2:
        return 0.0

    octaves, dbs = zip(*harmonics_db)
    # Linear regression: dB vs octaves
    coeffs = np.polyfit(octaves, dbs, 1)
    return float(coeffs[0])  # slope in dB/octave


def _compute_dissonance(mono: np.ndarray, sr: int) -> float:
    """Estimate perceptual dissonance from spectral peaks using Plomp-Levelt model."""
    S = np.abs(librosa.stft(mono, n_fft=4096))
    mean_spec = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    # Find spectral peaks
    from scipy.signal import find_peaks
    peaks, props = find_peaks(mean_spec, prominence=mean_spec.max() * 0.02, distance=3)
    if len(peaks) < 2:
        return 0.0

    # Take top 20 peaks by amplitude
    top_idx = np.argsort(mean_spec[peaks])[::-1][:20]
    peak_freqs = freqs[peaks[top_idx]]
    peak_amps = mean_spec[peaks[top_idx]]
    peak_amps = peak_amps / peak_amps.max()  # normalize

    # Plomp-Levelt pairwise dissonance
    total_dissonance = 0.0
    count = 0
    for i in range(len(peak_freqs)):
        for j in range(i + 1, len(peak_freqs)):
            f1, f2 = min(peak_freqs[i], peak_freqs[j]), max(peak_freqs[i], peak_freqs[j])
            if f1 < 20:
                continue
            s = 0.24 / (0.021 * f1 + 19)  # critical bandwidth scaling
            diff = (f2 - f1) * s
            d = np.exp(-3.5 * diff) - np.exp(-5.75 * diff)
            d *= peak_amps[i] * peak_amps[j]
            total_dissonance += max(0, d)
            count += 1

    if count == 0:
        return 0.0
    return float(np.clip(total_dissonance / count * 10, 0, 1))


def _compute_roughness(mono: np.ndarray, sr: int) -> float:
    """Estimate roughness from amplitude modulation in critical bands."""
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=256))

    # Measure frame-to-frame fluctuation in each band
    fluctuation = np.diff(S, axis=1)
    # Weight by frequency (roughness most perceived 20-300Hz modulation)
    roughness_per_band = np.mean(np.abs(fluctuation), axis=1)
    total = np.mean(roughness_per_band)

    # Normalize to 0-1 range (empirical scaling)
    return float(np.clip(total * 5, 0, 1))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_harmonic_descriptors.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/analysis/harmonic_descriptors.py tests/test_harmonic_descriptors.py
git commit -m "feat(phase1): harmonic descriptors — F0, chroma, HNR, tonalness, dissonance, roughness"
```

---

## Task 5: Transient Descriptors

**Files:**
- Create: `backend/ml/analysis/transient_descriptors.py`
- Create: `tests/test_transient_descriptors.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transient_descriptors.py
import numpy as np
from src.analysis.transient_descriptors import extract_transient_descriptors
from src.models.sample_profile import TransientDescriptors


def test_returns_transient_descriptors(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    assert isinstance(result, TransientDescriptors)


def test_kick_has_onset(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    assert result.onset_count >= 1


def test_hihat_has_onset(hihat_like):
    result = extract_transient_descriptors(str(hihat_like))
    assert result.onset_count >= 1


def test_onset_positions_in_range(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    for pos in result.transient_positions:
        assert 0.0 <= pos <= 0.2  # kick is 150ms


def test_onset_strength_positive(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    assert result.onset_strength_mean > 0


def test_kick_sharp_attack(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    assert result.attack_sharpness > 0.3


def test_pad_soft_attack(pad_like):
    result = extract_transient_descriptors(str(pad_like))
    assert result.attack_sharpness < 0.5


def test_silence_no_onsets(silence):
    result = extract_transient_descriptors(str(silence))
    assert result.onset_count == 0


def test_transient_density_reasonable(kick_like):
    result = extract_transient_descriptors(str(kick_like))
    assert result.transient_density >= 0
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_transient_descriptors.py -v`
Expected: FAIL

- [ ] **Step 3: Implement transient_descriptors.py**

```python
# backend/ml/analysis/transient_descriptors.py
"""
Transient descriptors: onset detection, transient positions, onset strength,
attack sharpness, transient density.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf
from src.models.sample_profile import TransientDescriptors


def extract_transient_descriptors(filepath: str) -> TransientDescriptors:
    """Extract all transient-related descriptors."""
    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    duration = len(mono) / sr

    # Onset detection
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        backtrack=True, units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    onset_count = len(onset_times)
    onset_rate = onset_count / duration if duration > 0 else 0.0

    # Onset strength statistics
    if len(onset_env) > 0 and onset_env.max() > 0:
        onset_strength_mean = float(np.mean(onset_env))
        onset_strength_std = float(np.std(onset_env))
    else:
        onset_strength_mean = 0.0
        onset_strength_std = 0.0

    # Attack sharpness: how fast the amplitude rises at the first onset
    attack_sharpness = _compute_attack_sharpness(mono, sr)

    # Transient density (onsets per second)
    transient_density = onset_rate

    return TransientDescriptors(
        onset_count=onset_count,
        onset_rate=round(onset_rate, 2),
        onset_strength_mean=round(onset_strength_mean, 4),
        onset_strength_std=round(onset_strength_std, 4),
        transient_positions=[round(float(t), 4) for t in onset_times[:50]],  # cap at 50
        attack_sharpness=round(attack_sharpness, 4),
        transient_density=round(transient_density, 2),
    )


def _compute_attack_sharpness(mono: np.ndarray, sr: int) -> float:
    """
    Measure how sharp/percussive the initial transient is.
    0 = very soft onset, 1 = extremely sharp click.
    Uses the slope of the amplitude envelope at onset.
    """
    # Compute amplitude envelope with short window
    hop = max(1, sr // 1000)  # 1ms resolution
    frame_len = hop * 2
    n_frames = min(len(mono) // hop, sr // hop)  # analyze first second max

    if n_frames < 3:
        return 0.0

    envelope = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(mono))
        frame = mono[start:end]
        envelope[i] = np.sqrt(np.mean(frame ** 2))

    if envelope.max() < 1e-10:
        return 0.0

    env_norm = envelope / envelope.max()

    # Find the peak in the first half
    half = len(env_norm) // 2
    if half < 1:
        return 0.0
    first_half = env_norm[:max(half, 2)]
    peak_idx = np.argmax(first_half)

    if peak_idx == 0:
        # Instantaneous peak
        return 1.0

    # Sharpness = normalized rise rate to peak
    rise = env_norm[peak_idx] / (peak_idx * hop / sr)  # amplitude per second
    # Empirical normalization: very sharp ~= 500 amp/s, soft ~= 5 amp/s
    sharpness = np.clip(rise / 500.0, 0.0, 1.0)
    return float(sharpness)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_transient_descriptors.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/analysis/transient_descriptors.py tests/test_transient_descriptors.py
git commit -m "feat(phase1): transient descriptors — onsets, attack sharpness, transient density"
```

---

## Task 6: Perceptual Descriptors

**Files:**
- Create: `backend/ml/analysis/perceptual_descriptors.py`
- Create: `tests/test_perceptual_descriptors.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_perceptual_descriptors.py
import numpy as np
from src.analysis.perceptual_descriptors import extract_perceptual_descriptors
from src.models.sample_profile import PerceptualDescriptors


def test_returns_perceptual_descriptors(sine_440hz):
    result = extract_perceptual_descriptors(str(sine_440hz))
    assert isinstance(result, PerceptualDescriptors)


def test_all_values_in_range(sine_440hz):
    result = extract_perceptual_descriptors(str(sine_440hz))
    for field_name in ["brightness", "warmth", "air", "punch", "body",
                       "bite", "smoothness", "width", "depth_impression"]:
        val = getattr(result, field_name)
        assert 0.0 <= val <= 1.0, f"{field_name}={val} out of range"


def test_noise_has_brightness(stereo_noise):
    result = extract_perceptual_descriptors(str(stereo_noise))
    # White noise has lots of high-frequency content
    assert result.brightness > 0.3


def test_kick_has_punch(kick_like):
    result = extract_perceptual_descriptors(str(kick_like))
    assert result.punch > 0.2


def test_kick_has_body(kick_like):
    result = extract_perceptual_descriptors(str(kick_like))
    assert result.body > 0.1


def test_pad_has_warmth(pad_like):
    result = extract_perceptual_descriptors(str(pad_like))
    assert result.warmth > 0.1


def test_stereo_has_width(pad_like):
    result = extract_perceptual_descriptors(str(pad_like))
    assert result.width > 0.1


def test_mono_low_width(sine_440hz):
    result = extract_perceptual_descriptors(str(sine_440hz))
    assert result.width < 0.1


def test_hihat_has_bite(hihat_like):
    result = extract_perceptual_descriptors(str(hihat_like))
    assert result.bite > 0.1
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_perceptual_descriptors.py -v`
Expected: FAIL

- [ ] **Step 3: Implement perceptual_descriptors.py**

```python
# backend/ml/analysis/perceptual_descriptors.py
"""
Perceptual descriptors: brightness, warmth, air, punch, body, bite,
smoothness, width, depth impression.

These are derived from measurable spectral/temporal features mapped to
perceptual dimensions. Some are direct measurements, others are composite
scores that approximate how a trained ear would describe the sound.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf
from src.models.sample_profile import PerceptualDescriptors


def extract_perceptual_descriptors(filepath: str) -> PerceptualDescriptors:
    """Extract all perceptual descriptors."""
    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
    n_channels = audio.shape[1]
    mono = audio.mean(axis=1)

    # Pre-compute shared features
    n_fft = 2048
    hop_length = 512
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_spec = np.mean(S ** 2, axis=1)
    total_energy = mean_spec.sum()

    if total_energy < 1e-10:
        return PerceptualDescriptors()

    # Band energy helper
    def band_ratio(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(mean_spec[mask].sum() / total_energy)

    # --- Brightness: ratio of energy above 4kHz ---
    brightness = np.clip(band_ratio(4000, sr / 2) * 3.0, 0, 1)

    # --- Warmth: energy in 200-800Hz relative to total ---
    warmth = np.clip(band_ratio(200, 800) * 4.0, 0, 1)

    # --- Air: energy above 10kHz ---
    air = np.clip(band_ratio(10000, sr / 2) * 8.0, 0, 1)

    # --- Body: energy in 80-300Hz ---
    body = np.clip(band_ratio(80, 300) * 4.0, 0, 1)

    # --- Bite: energy in 2-5kHz (presence/aggression range) ---
    bite = np.clip(band_ratio(2000, 5000) * 5.0, 0, 1)

    # --- Punch: combination of transient sharpness + low-mid energy ---
    # Compute onset strength as proxy for transient impact
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop_length)
    if onset_env.max() > 0:
        onset_peak_ratio = float(onset_env.max() / np.mean(onset_env)) if np.mean(onset_env) > 0 else 0
        low_mid = band_ratio(60, 500)
        punch = np.clip((onset_peak_ratio / 10.0) * 0.6 + low_mid * 4.0 * 0.4, 0, 1)
    else:
        punch = 0.0

    # --- Smoothness: inverse of spectral flux (frame-to-frame change) ---
    flux = np.mean(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    max_flux = np.mean(np.sum(S ** 2, axis=0))
    if max_flux > 1e-10:
        smoothness = np.clip(1.0 - (flux / max_flux), 0, 1)
    else:
        smoothness = 0.5

    # --- Width: stereo correlation analysis ---
    if n_channels >= 2:
        left = audio[:, 0]
        right = audio[:, 1]
        # Cross-correlation at zero lag
        correlation = np.corrcoef(left, right)[0, 1]
        if np.isnan(correlation):
            correlation = 1.0
        # Width: 0 = identical (mono), 1 = completely uncorrelated
        width = np.clip(float(1.0 - abs(correlation)), 0, 1)
    else:
        width = 0.0

    # --- Depth impression: combination of reverb tail detection + spectral decay ---
    # Use RT60-like estimation from energy decay
    depth_impression = _estimate_depth(mono, sr)

    return PerceptualDescriptors(
        brightness=round(float(brightness), 4),
        warmth=round(float(warmth), 4),
        air=round(float(air), 4),
        punch=round(float(punch), 4),
        body=round(float(body), 4),
        bite=round(float(bite), 4),
        smoothness=round(float(smoothness), 4),
        width=round(float(width), 4),
        depth_impression=round(float(depth_impression), 4),
    )


def _estimate_depth(mono: np.ndarray, sr: int) -> float:
    """
    Estimate perceived depth/space from the energy decay curve.
    Longer, smoother tails suggest more reverb/space.
    """
    # Compute energy envelope in 50ms frames
    hop = sr // 20
    n_frames = len(mono) // hop
    if n_frames < 4:
        return 0.0

    env = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(mono))
        env[i] = np.mean(mono[start:end] ** 2)

    if env.max() < 1e-10:
        return 0.0

    env_norm = env / env.max()

    # Find how long the tail persists above noise floor
    peak_idx = np.argmax(env_norm)
    tail = env_norm[peak_idx:]

    if len(tail) < 2:
        return 0.0

    # Count frames above -30dB (0.001 in linear)
    above_threshold = np.sum(tail > 0.001)
    tail_duration = above_threshold * (hop / sr)

    # Normalize: 0s = 0 depth, >1s tail = 1.0 depth
    return float(np.clip(tail_duration / 1.0, 0, 1))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_perceptual_descriptors.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/analysis/perceptual_descriptors.py tests/test_perceptual_descriptors.py
git commit -m "feat(phase1): perceptual descriptors — brightness, warmth, air, punch, body, bite, width, depth"
```

---

## Task 7: Loop Detection

**Files:**
- Create: `backend/ml/analysis/loop_detection.py`
- Create: `tests/test_loop_detection.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_loop_detection.py
import numpy as np
import soundfile as sf
from src.analysis.loop_detection import detect_loop


def test_oneshot_kick(kick_like):
    is_loop, confidence = detect_loop(str(kick_like))
    assert is_loop is False
    assert confidence > 0.5


def test_oneshot_hihat(hihat_like):
    is_loop, confidence = detect_loop(str(hihat_like))
    assert is_loop is False


def test_loop_detection_on_repeated_pattern(tmp_path, sample_rate):
    """A repeated pattern should be detected as a loop."""
    # Create a 2-bar repeating pattern
    bar_len = sample_rate // 2  # 0.5s per bar
    t = np.linspace(0, 0.5, bar_len, endpoint=False)
    pattern = (0.3 * np.sin(2 * np.pi * 200 * t) *
               np.exp(-4 * (t % 0.125) / 0.125)).astype(np.float32)
    # Repeat 4 times
    audio = np.tile(pattern, 4)
    path = tmp_path / "loop_pattern.wav"
    sf.write(str(path), audio, sample_rate)
    is_loop, confidence = detect_loop(str(path))
    assert is_loop is True


def test_confidence_range(sine_440hz):
    _, confidence = detect_loop(str(sine_440hz))
    assert 0.0 <= confidence <= 1.0
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_loop_detection.py -v`
Expected: FAIL

- [ ] **Step 3: Implement loop_detection.py**

```python
# backend/ml/analysis/loop_detection.py
"""
Loop vs one-shot detection.
Uses multiple signals: duration, self-similarity, onset regularity, energy tail.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf


def detect_loop(filepath: str) -> tuple[bool, float]:
    """
    Detect if an audio file is a loop or a one-shot.
    Returns (is_loop, confidence) where confidence is 0-1.
    """
    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    duration = len(mono) / sr

    scores = []

    # Signal 1: Duration. One-shots are typically < 1s, loops > 1s
    if duration < 0.3:
        scores.append(0.0)  # Almost certainly one-shot
    elif duration < 1.0:
        scores.append(0.2)
    elif duration < 2.0:
        scores.append(0.5)
    else:
        scores.append(0.7)

    # Signal 2: Self-similarity via autocorrelation
    if duration > 0.5:
        autocorr_score = _autocorrelation_periodicity(mono, sr)
        scores.append(autocorr_score)

    # Signal 3: Onset regularity
    if duration > 0.5:
        regularity = _onset_regularity(mono, sr)
        scores.append(regularity)

    # Signal 4: Energy at boundaries (loops maintain energy, one-shots decay)
    boundary_score = _boundary_energy_score(mono)
    scores.append(boundary_score)

    # Signal 5: Start/end similarity (loops should be similar at boundaries)
    if duration > 0.3:
        boundary_sim = _boundary_similarity(mono, sr)
        scores.append(boundary_sim)

    # Combine scores
    combined = float(np.mean(scores))
    is_loop = combined > 0.5
    confidence = abs(combined - 0.5) * 2  # Distance from decision boundary

    return is_loop, round(confidence, 4)


def _autocorrelation_periodicity(mono: np.ndarray, sr: int) -> float:
    """Check for repeating patterns via autocorrelation."""
    # Downsample for efficiency
    hop = max(1, sr // 100)
    envelope = np.array([
        np.sqrt(np.mean(mono[i:i+hop] ** 2))
        for i in range(0, len(mono) - hop, hop)
    ])
    if len(envelope) < 10:
        return 0.0

    envelope = envelope - envelope.mean()
    if np.std(envelope) < 1e-10:
        return 0.0
    envelope = envelope / np.std(envelope)

    # Autocorrelation
    corr = np.correlate(envelope, envelope, mode="full")
    corr = corr[len(corr) // 2:]  # Keep positive lags
    corr = corr / corr[0] if corr[0] > 0 else corr

    # Look for strong secondary peaks (indicating repetition)
    min_lag = max(5, len(corr) // 8)  # Minimum half-bar length
    if min_lag >= len(corr):
        return 0.0
    secondary = corr[min_lag:]
    if len(secondary) == 0:
        return 0.0
    max_secondary = float(np.max(secondary))
    return float(np.clip(max_secondary, 0, 1))


def _onset_regularity(mono: np.ndarray, sr: int) -> float:
    """Measure how regular/periodic the onsets are."""
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
    if len(onsets) < 3:
        return 0.0

    intervals = np.diff(onsets)
    if len(intervals) < 2:
        return 0.0

    # Regularity = 1 - coefficient of variation
    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1.0
    return float(np.clip(1.0 - cv, 0, 1))


def _boundary_energy_score(mono: np.ndarray) -> float:
    """Loops maintain energy at the end; one-shots decay to silence."""
    n = len(mono)
    if n < 100:
        return 0.0

    start_rms = np.sqrt(np.mean(mono[:n // 20] ** 2))
    end_rms = np.sqrt(np.mean(mono[-n // 20:] ** 2))

    if start_rms < 1e-10:
        return 0.0

    ratio = end_rms / start_rms
    # If end energy is close to start, likely a loop
    return float(np.clip(ratio, 0, 1))


def _boundary_similarity(mono: np.ndarray, sr: int) -> float:
    """Compare spectral content at start and end of the file."""
    chunk_len = min(len(mono) // 4, sr // 4)  # 250ms or quarter of file
    if chunk_len < 256:
        return 0.0

    start_chunk = mono[:chunk_len]
    end_chunk = mono[-chunk_len:]

    # Compare MFCCs
    mfcc_start = librosa.feature.mfcc(y=start_chunk, sr=sr, n_mfcc=13)
    mfcc_end = librosa.feature.mfcc(y=end_chunk, sr=sr, n_mfcc=13)

    start_mean = np.mean(mfcc_start, axis=1)
    end_mean = np.mean(mfcc_end, axis=1)

    # Cosine similarity
    dot = np.dot(start_mean, end_mean)
    norm = np.linalg.norm(start_mean) * np.linalg.norm(end_mean)
    if norm < 1e-10:
        return 0.0
    similarity = dot / norm
    return float(np.clip(similarity, 0, 1))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_loop_detection.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/analysis/loop_detection.py tests/test_loop_detection.py
git commit -m "feat(phase1): loop detection — autocorrelation, onset regularity, boundary analysis"
```

---

## Task 8: Embedding Extraction — CLAP

**Files:**
- Create: `backend/ml/embeddings/__init__.py`
- Create: `backend/ml/embeddings/clap_embeddings.py`
- Create: `tests/test_clap_embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_clap_embeddings.py
import numpy as np
import pytest
from src.embeddings.clap_embeddings import CLAPExtractor


@pytest.fixture(scope="module")
def clap():
    """Load CLAP model once for all tests in this module."""
    try:
        return CLAPExtractor()
    except Exception:
        pytest.skip("CLAP model not available")


def test_embedding_shape(clap, sine_440hz):
    emb = clap.extract(str(sine_440hz))
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert len(emb) == 512  # CLAP embedding dim


def test_embedding_normalized(clap, sine_440hz):
    emb = clap.extract(str(sine_440hz))
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 0.01  # Should be L2-normalized


def test_different_sounds_different_embeddings(clap, sine_440hz, kick_like):
    emb1 = clap.extract(str(sine_440hz))
    emb2 = clap.extract(str(kick_like))
    cosine_sim = np.dot(emb1, emb2)
    # Different sounds should not be identical
    assert cosine_sim < 0.99


def test_same_sound_consistent(clap, sine_440hz):
    emb1 = clap.extract(str(sine_440hz))
    emb2 = clap.extract(str(sine_440hz))
    cosine_sim = np.dot(emb1, emb2)
    assert cosine_sim > 0.99
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_clap_embeddings.py -v`
Expected: FAIL

- [ ] **Step 3: Implement clap_embeddings.py**

```python
# backend/ml/embeddings/__init__.py
# (empty)
```

```python
# backend/ml/embeddings/clap_embeddings.py
"""
CLAP (Contrastive Language-Audio Pretraining) embedding extractor.
Produces 512-dim embeddings in a joint audio-text space.
Useful for general audio similarity and text-based retrieval.
"""
from __future__ import annotations
import numpy as np
import librosa
import torch


class CLAPExtractor:
    """Extracts CLAP embeddings from audio files."""

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load the CLAP model."""
        import laion_clap
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.model.load_ckpt()  # Downloads checkpoint if needed
        self.model.eval()

    def extract(self, filepath: str) -> np.ndarray:
        """
        Extract a 512-dim L2-normalized embedding from an audio file.
        Audio is resampled to 48kHz as required by CLAP.
        """
        # Load and prepare audio
        audio, sr = librosa.load(filepath, sr=48000, mono=True)

        # Pad or truncate to 10 seconds (CLAP default)
        target_len = 48000 * 10
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))

        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()

        with torch.no_grad():
            embedding = self.model.get_audio_embedding_from_data(
                x=audio_tensor, use_tensor=True
            )

        emb = embedding.cpu().numpy().flatten()
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        return emb.astype(np.float32)

    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract embedding for a text query (for text-to-audio search)."""
        with torch.no_grad():
            embedding = self.model.get_text_embedding([text])
        emb = embedding.cpu().numpy().flatten()
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        return emb.astype(np.float32)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_clap_embeddings.py -v`
Expected: All 4 tests PASS (or SKIP if no GPU/model)

- [ ] **Step 5: Commit**

```bash
git add backend/ml/embeddings/ tests/test_clap_embeddings.py
git commit -m "feat(phase1): CLAP embedding extractor — 512-dim audio-text space"
```

---

## Task 9: Embedding Extraction — PANNs

**Files:**
- Create: `backend/ml/embeddings/panns_embeddings.py`
- Create: `tests/test_panns_embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_panns_embeddings.py
import numpy as np
import pytest
from src.embeddings.panns_embeddings import PANNsExtractor


@pytest.fixture(scope="module")
def panns():
    try:
        return PANNsExtractor()
    except Exception:
        pytest.skip("PANNs model not available")


def test_embedding_shape(panns, sine_440hz):
    emb = panns.extract_embedding(str(sine_440hz))
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert len(emb) == 2048  # PANNs Cnn14 embedding dim


def test_tags_returned(panns, kick_like):
    tags = panns.extract_tags(str(kick_like))
    assert isinstance(tags, dict)
    assert len(tags) > 0
    # All confidences should be 0-1
    for tag, conf in tags.items():
        assert 0.0 <= conf <= 1.0


def test_embedding_deterministic(panns, sine_440hz):
    emb1 = panns.extract_embedding(str(sine_440hz))
    emb2 = panns.extract_embedding(str(sine_440hz))
    np.testing.assert_array_almost_equal(emb1, emb2, decimal=4)
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_panns_embeddings.py -v`
Expected: FAIL

- [ ] **Step 3: Implement panns_embeddings.py**

```python
# backend/ml/embeddings/panns_embeddings.py
"""
PANNs (Pretrained Audio Neural Networks) embedding and tagging extractor.
Uses Cnn14 for robust audio tagging and 2048-dim embeddings.
"""
from __future__ import annotations
import numpy as np
import librosa


class PANNsExtractor:
    """Extract PANNs embeddings and audio tags."""

    def __init__(self, device: str | None = None):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load PANNs Cnn14 model."""
        from panns_inference import AudioTagging
        self.model = AudioTagging(checkpoint_path=None, device=self.device)

    def extract_embedding(self, filepath: str) -> np.ndarray:
        """Extract 2048-dim embedding."""
        audio = self._load_audio(filepath)
        _, embedding = self.model.inference(audio[np.newaxis, :])
        return embedding.flatten().astype(np.float32)

    def extract_tags(self, filepath: str, top_k: int = 20) -> dict[str, float]:
        """Extract AudioSet tags with confidence scores."""
        audio = self._load_audio(filepath)
        clipwise_output, _ = self.model.inference(audio[np.newaxis, :])
        probs = clipwise_output.flatten()

        # Get AudioSet labels
        import panns_inference
        labels = panns_inference.labels
        if hasattr(labels, 'labels'):
            label_list = labels.labels
        else:
            # Fallback: load from CSV
            label_list = [f"class_{i}" for i in range(len(probs))]

        # Top-k tags
        top_indices = np.argsort(probs)[::-1][:top_k]
        return {
            label_list[i]: round(float(probs[i]), 4)
            for i in top_indices
            if probs[i] > 0.01
        }

    def _load_audio(self, filepath: str) -> np.ndarray:
        """Load audio at 32kHz mono (PANNs requirement)."""
        audio, _ = librosa.load(filepath, sr=32000, mono=True)
        return audio.astype(np.float32)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_panns_embeddings.py -v`
Expected: All 3 tests PASS (or SKIP if model unavailable)

- [ ] **Step 5: Commit**

```bash
git add backend/ml/embeddings/panns_embeddings.py tests/test_panns_embeddings.py
git commit -m "feat(phase1): PANNs embedding + tagging — 2048-dim + AudioSet labels"
```

---

## Task 10: Embedding Extraction — AST

**Files:**
- Create: `backend/ml/embeddings/ast_embeddings.py`
- Create: `tests/test_ast_embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ast_embeddings.py
import numpy as np
import pytest
from src.embeddings.ast_embeddings import ASTExtractor


@pytest.fixture(scope="module")
def ast_model():
    try:
        return ASTExtractor()
    except Exception:
        pytest.skip("AST model not available")


def test_embedding_shape(ast_model, sine_440hz):
    emb = ast_model.extract(str(sine_440hz))
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert len(emb) == 768  # AST hidden dim


def test_different_sounds_differ(ast_model, sine_440hz, kick_like):
    emb1 = ast_model.extract(str(sine_440hz))
    emb2 = ast_model.extract(str(kick_like))
    cosine = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    assert cosine < 0.99


def test_deterministic(ast_model, sine_440hz):
    emb1 = ast_model.extract(str(sine_440hz))
    emb2 = ast_model.extract(str(sine_440hz))
    np.testing.assert_array_almost_equal(emb1, emb2, decimal=4)
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_ast_embeddings.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ast_embeddings.py**

```python
# backend/ml/embeddings/ast_embeddings.py
"""
AST (Audio Spectrogram Transformer) embedding extractor.
Produces 768-dim embeddings from spectrogram representations.
Good for capturing timbral and structural features.
"""
from __future__ import annotations
import numpy as np
import librosa
import torch
from transformers import ASTModel, ASTFeatureExtractor


class ASTExtractor:
    """Extract AST embeddings from audio files."""

    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                 device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def extract(self, filepath: str) -> np.ndarray:
        """
        Extract 768-dim embedding from an audio file.
        Uses the [CLS] token output from AST.
        """
        # Load at 16kHz (AST requirement)
        audio, _ = librosa.load(filepath, sr=16000, mono=True)

        # Truncate to 10 seconds max
        max_samples = 16000 * 10
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Extract features (mel spectrogram)
        inputs = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            # Use the [CLS] token (first token) as the embedding
            embedding = outputs.last_hidden_state[:, 0, :]

        emb = embedding.cpu().numpy().flatten().astype(np.float32)
        return emb
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_ast_embeddings.py -v`
Expected: All 3 tests PASS (or SKIP)

- [ ] **Step 5: Commit**

```bash
git add backend/ml/embeddings/ast_embeddings.py tests/test_ast_embeddings.py
git commit -m "feat(phase1): AST embedding extractor — 768-dim spectrogram transformer"
```

---

## Task 11: Embedding Manager (Orchestrator)

**Files:**
- Create: `backend/ml/embeddings/embedding_manager.py`
- Create: `tests/test_embedding_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_embedding_manager.py
import pytest
from src.embeddings.embedding_manager import EmbeddingManager
from src.models.sample_profile import Embeddings


@pytest.fixture(scope="module")
def manager():
    try:
        return EmbeddingManager()
    except Exception:
        pytest.skip("Embedding models not available")


def test_returns_embeddings_dataclass(manager, sine_440hz):
    result = manager.extract_all(str(sine_440hz))
    assert isinstance(result, Embeddings)


def test_clap_populated(manager, sine_440hz):
    result = manager.extract_all(str(sine_440hz))
    assert len(result.clap_general) == 512


def test_panns_populated(manager, sine_440hz):
    result = manager.extract_all(str(sine_440hz))
    assert len(result.panns_music) == 2048


def test_ast_populated(manager, sine_440hz):
    result = manager.extract_all(str(sine_440hz))
    assert len(result.ast_spectrogram) == 768


def test_panns_tags_populated(manager, sine_440hz):
    result = manager.extract_all(str(sine_440hz))
    assert isinstance(result.panns_tags, dict)
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_embedding_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Implement embedding_manager.py**

```python
# backend/ml/embeddings/embedding_manager.py
"""
Orchestrates all embedding extractors. Loads models once, extracts all embeddings
for a given audio file in one call.
"""
from __future__ import annotations
import logging
from src.models.sample_profile import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages all embedding extractors with lazy loading."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._clap = None
        self._panns = None
        self._ast = None

    @property
    def clap(self):
        if self._clap is None:
            from src.embeddings.clap_embeddings import CLAPExtractor
            self._clap = CLAPExtractor(device=self.device)
            logger.info("CLAP model loaded")
        return self._clap

    @property
    def panns(self):
        if self._panns is None:
            from src.embeddings.panns_embeddings import PANNsExtractor
            self._panns = PANNsExtractor(device=self.device)
            logger.info("PANNs model loaded")
        return self._panns

    @property
    def ast(self):
        if self._ast is None:
            from src.embeddings.ast_embeddings import ASTExtractor
            self._ast = ASTExtractor(device=self.device)
            logger.info("AST model loaded")
        return self._ast

    def extract_all(self, filepath: str) -> Embeddings:
        """Extract all embeddings for a single audio file."""
        result = Embeddings()

        # CLAP
        try:
            emb = self.clap.extract(filepath)
            result.clap_general = emb.tolist()
        except Exception as e:
            logger.warning(f"CLAP extraction failed for {filepath}: {e}")

        # PANNs
        try:
            emb = self.panns.extract_embedding(filepath)
            result.panns_music = emb.tolist()
            result.panns_tags = self.panns.extract_tags(filepath)
        except Exception as e:
            logger.warning(f"PANNs extraction failed for {filepath}: {e}")

        # AST
        try:
            emb = self.ast.extract(filepath)
            result.ast_spectrogram = emb.tolist()
        except Exception as e:
            logger.warning(f"AST extraction failed for {filepath}: {e}")

        return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_embedding_manager.py -v`
Expected: All 5 tests PASS (or SKIP)

- [ ] **Step 5: Commit**

```bash
git add backend/ml/embeddings/embedding_manager.py tests/test_embedding_manager.py
git commit -m "feat(phase1): embedding manager — orchestrates CLAP, PANNs, AST extraction"
```

---

## Task 12: Role Classifier

**Files:**
- Create: `backend/ml/classifiers/__init__.py`
- Create: `backend/ml/classifiers/role_classifier.py`
- Create: `tests/test_role_classifier.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_role_classifier.py
import pytest
from src.classifiers.role_classifier import RoleClassifier


@pytest.fixture(scope="module")
def classifier():
    return RoleClassifier()


VALID_ROLES = {"kick", "snare", "clap", "hat", "bass", "lead", "pad",
               "fx", "texture", "vocal", "percussion", "unknown"}


def test_returns_role_and_confidence(classifier, kick_like):
    role, confidence = classifier.classify(str(kick_like))
    assert role in VALID_ROLES
    assert 0.0 <= confidence <= 1.0


def test_kick_classified(classifier, kick_like):
    role, _ = classifier.classify(str(kick_like))
    assert role in {"kick", "percussion", "bass"}  # Allow reasonable confusion


def test_hihat_classified(classifier, hihat_like):
    role, _ = classifier.classify(str(hihat_like))
    assert role in {"hat", "percussion"}


def test_pad_classified(classifier, pad_like):
    role, _ = classifier.classify(str(pad_like))
    assert role in {"pad", "texture", "lead"}


def test_classify_with_filename_hint(classifier, kick_like):
    role, conf = classifier.classify(str(kick_like), filename_hint="808_kick_hard.wav")
    assert role == "kick"
    assert conf > 0.5
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_role_classifier.py -v`
Expected: FAIL

- [ ] **Step 3: Implement role_classifier.py**

```python
# backend/ml/classifiers/__init__.py
# (empty)
```

```python
# backend/ml/classifiers/role_classifier.py
"""
Sound role classifier: kick/snare/clap/hat/bass/lead/pad/fx/texture/vocal/percussion.
Uses a combination of:
1. Filename-based heuristics (high confidence when keywords match)
2. Audio feature rules (spectral centroid, duration, transients, pitch)
3. PANNs tag mapping (when available)

Later phases will add a trained neural classifier on top.
"""
from __future__ import annotations
import re
import numpy as np
from pathlib import Path

# Keyword → role mapping (prioritized)
ROLE_KEYWORDS: dict[str, list[str]] = {
    "kick": ["kick", "kik", "808", "bd"],
    "snare": ["snare", "snr", "sd", "rim"],
    "clap": ["clap", "clp", "handclap"],
    "hat": ["hat", "hh", "hihat", "hi-hat", "openhat", "closedhat", "oh", "ch"],
    "bass": ["bass", "sub", "808bass", "reese", "bassline"],
    "lead": ["lead", "ld", "synth", "pluck", "stab"],
    "pad": ["pad", "atmosphere", "atmo", "ambient", "drone"],
    "fx": ["fx", "effect", "riser", "downlifter", "sweep", "impact", "whoosh", "transition"],
    "texture": ["texture", "foley", "noise", "grain", "field"],
    "vocal": ["vocal", "vox", "voice", "acapella", "adlib", "chant", "choir"],
    "percussion": ["perc", "percussion", "conga", "bongo", "shaker", "tambourine", "tom", "cymbal", "crash", "ride"],
}

# PANNs AudioSet tag → role mapping
PANNS_TAG_MAP: dict[str, str] = {
    "Bass drum": "kick", "Kick drum": "kick",
    "Snare drum": "snare",
    "Clapping": "clap",
    "Hi-hat": "hat", "Cymbal": "hat",
    "Bass guitar": "bass", "Bass": "bass",
    "Synthesizer": "lead",
    "Singing": "vocal", "Speech": "vocal", "Voice": "vocal",
    "Drum": "percussion", "Drum kit": "percussion",
}


class RoleClassifier:
    """Classify the role/function of an audio sample."""

    def classify(self, filepath: str, filename_hint: str | None = None,
                 panns_tags: dict[str, float] | None = None) -> tuple[str, float]:
        """
        Classify sample role.
        Returns (role, confidence).
        """
        filename = filename_hint or Path(filepath).name
        scores: dict[str, float] = {role: 0.0 for role in ROLE_KEYWORDS}
        scores["unknown"] = 0.1  # small prior

        # Stage 1: Filename heuristics
        filename_lower = re.sub(r"[_\-\.]", " ", filename.lower())
        tokens = set(filename_lower.split())
        for role, keywords in ROLE_KEYWORDS.items():
            for kw in keywords:
                if kw in tokens or kw in filename_lower:
                    scores[role] += 0.7
                    break

        # Stage 2: Audio features (lightweight analysis)
        try:
            feature_scores = self._feature_based_scores(filepath)
            for role, score in feature_scores.items():
                scores[role] = scores.get(role, 0) + score * 0.5
        except Exception:
            pass

        # Stage 3: PANNs tags (if provided)
        if panns_tags:
            for tag, conf in panns_tags.items():
                if tag in PANNS_TAG_MAP and conf > 0.1:
                    role = PANNS_TAG_MAP[tag]
                    if role in scores:
                        scores[role] += conf * 0.4

        # Pick winner
        best_role = max(scores, key=scores.get)
        best_score = scores[best_role]
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.0

        return best_role, round(confidence, 4)

    def _feature_based_scores(self, filepath: str) -> dict[str, float]:
        """Score roles based on audio features."""
        import soundfile as sf
        import librosa

        audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        duration = len(mono) / sr

        # Quick spectral analysis
        S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mean_spec = np.mean(S ** 2, axis=1)
        total_energy = mean_spec.sum()

        if total_energy < 1e-10:
            return {}

        def band_ratio(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(mean_spec[mask].sum() / total_energy)

        sub_bass = band_ratio(20, 100)
        low_mid = band_ratio(100, 500)
        mid = band_ratio(500, 2000)
        high_mid = band_ratio(2000, 6000)
        high = band_ratio(6000, sr / 2)

        # Onset detection
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        onset_peak = onset_env.max() if len(onset_env) > 0 else 0
        onset_mean = onset_env.mean() if len(onset_env) > 0 else 0
        transient_ratio = onset_peak / onset_mean if onset_mean > 0 else 0

        scores = {}

        # Kick: short, sub-heavy, strong transient
        if duration < 1.0 and sub_bass > 0.3:
            scores["kick"] = 0.5 + min(transient_ratio / 10, 0.3)

        # Snare: short, mid-focused, strong transient
        if duration < 0.8 and mid > 0.2 and transient_ratio > 3:
            scores["snare"] = 0.4

        # Hat: short, high-frequency dominant
        if duration < 0.5 and high > 0.3:
            scores["hat"] = 0.5

        # Bass: longer, sub/low dominant, tonal
        if sub_bass + low_mid > 0.5 and duration > 0.3:
            scores["bass"] = 0.4

        # Pad: long, spread spectrum, slow
        if duration > 1.0 and transient_ratio < 3:
            scores["pad"] = 0.4

        # Lead: mid-focused, tonal
        if mid + high_mid > 0.4 and 0.1 < duration < 3.0:
            scores["lead"] = 0.3

        # FX: anything weird or long with lots of spectral change
        spectral_flux = np.mean(np.diff(S, axis=1) ** 2)
        if spectral_flux > mean_spec.mean() * 0.5:
            scores["fx"] = 0.3

        return scores
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_role_classifier.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/classifiers/ tests/test_role_classifier.py
git commit -m "feat(phase1): role classifier — filename heuristics + audio features + PANNs tag mapping"
```

---

## Task 13: Genre/Era Classifier + Style Tagger + Quality Scorer

**Files:**
- Create: `backend/ml/classifiers/genre_era_classifier.py`
- Create: `backend/ml/classifiers/style_tagger.py`
- Create: `backend/ml/classifiers/quality_scorer.py`
- Create: `tests/test_genre_era_classifier.py`
- Create: `tests/test_style_tagger.py`
- Create: `tests/test_quality_scorer.py`

- [ ] **Step 1: Write failing tests for all three**

```python
# tests/test_genre_era_classifier.py
from src.classifiers.genre_era_classifier import GenreEraClassifier


def test_genre_affinity_returns_dict(kick_like):
    clf = GenreEraClassifier()
    genres = clf.classify_genre(str(kick_like))
    assert isinstance(genres, dict)
    for genre, score in genres.items():
        assert 0.0 <= score <= 1.0


def test_era_affinity_returns_dict(kick_like):
    clf = GenreEraClassifier()
    eras = clf.classify_era(str(kick_like))
    assert isinstance(eras, dict)


def test_genre_keys_are_strings(sine_440hz):
    clf = GenreEraClassifier()
    genres = clf.classify_genre(str(sine_440hz))
    for key in genres:
        assert isinstance(key, str)
```

```python
# tests/test_style_tagger.py
from src.classifiers.style_tagger import StyleTagger

EXPECTED_TAGS = {"bright", "dark", "wide", "punchy", "analog", "digital",
                 "gritty", "clean", "warm", "airy", "tight", "loose"}


def test_returns_dict(kick_like):
    tagger = StyleTagger()
    tags = tagger.tag(str(kick_like))
    assert isinstance(tags, dict)
    for tag, score in tags.items():
        assert isinstance(tag, str)
        assert 0.0 <= score <= 1.0


def test_kick_is_punchy(kick_like):
    tagger = StyleTagger()
    tags = tagger.tag(str(kick_like))
    # A kick should have some "punchy" quality
    assert "punchy" in tags
    assert tags["punchy"] > 0.1


def test_noise_is_not_clean(stereo_noise):
    tagger = StyleTagger()
    tags = tagger.tag(str(stereo_noise))
    if "clean" in tags:
        assert tags["clean"] < 0.5
```

```python
# tests/test_quality_scorer.py
from src.classifiers.quality_scorer import QualityScorer


def test_returns_float(kick_like):
    scorer = QualityScorer()
    score = scorer.score(str(kick_like))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_silence_low_quality(silence):
    scorer = QualityScorer()
    score = scorer.score(str(silence))
    assert score < 0.3


def test_real_sound_higher_quality(kick_like, silence):
    scorer = QualityScorer()
    kick_score = scorer.score(str(kick_like))
    silence_score = scorer.score(str(silence))
    assert kick_score > silence_score
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_genre_era_classifier.py tests/test_style_tagger.py tests/test_quality_scorer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement genre_era_classifier.py**

```python
# backend/ml/classifiers/genre_era_classifier.py
"""
Genre and era affinity classifier.
Phase 1: rule-based using spectral/rhythmic features + PANNs tags.
Phase 3 will replace with learned style-cluster models.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf

GENRES = [
    "trap", "hip-hop", "drill", "edm", "house", "techno",
    "pop", "r&b", "lo-fi", "cinematic", "dnb", "afro",
    "melodic-techno", "bass-music", "ambient"
]

ERAS = ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]

# Genre spectral signatures (centroid range, sub_bass_ratio, transient_density)
GENRE_SIGNATURES = {
    "trap": {"centroid": (800, 3000), "sub_bass": (0.2, 0.6), "transient": (2, 8)},
    "house": {"centroid": (1500, 4000), "sub_bass": (0.1, 0.3), "transient": (4, 10)},
    "techno": {"centroid": (1000, 3500), "sub_bass": (0.1, 0.4), "transient": (5, 12)},
    "drill": {"centroid": (800, 2500), "sub_bass": (0.2, 0.5), "transient": (3, 8)},
    "lo-fi": {"centroid": (500, 2000), "sub_bass": (0.05, 0.2), "transient": (1, 4)},
    "cinematic": {"centroid": (1000, 5000), "sub_bass": (0.05, 0.3), "transient": (0, 3)},
    "ambient": {"centroid": (500, 3000), "sub_bass": (0.02, 0.2), "transient": (0, 2)},
}


class GenreEraClassifier:
    """Classify genre and era affinity from audio features."""

    def classify_genre(self, filepath: str,
                       panns_tags: dict[str, float] | None = None) -> dict[str, float]:
        """Return genre affinity scores (0-1) for each genre."""
        features = self._extract_features(filepath)
        scores = {}

        for genre in GENRES:
            score = 0.0
            if genre in GENRE_SIGNATURES:
                sig = GENRE_SIGNATURES[genre]
                # Centroid match
                c_lo, c_hi = sig["centroid"]
                if c_lo <= features["centroid"] <= c_hi:
                    score += 0.3
                # Sub-bass match
                sb_lo, sb_hi = sig["sub_bass"]
                if sb_lo <= features["sub_bass_ratio"] <= sb_hi:
                    score += 0.3
                # Transient density match
                t_lo, t_hi = sig["transient"]
                if t_lo <= features["transient_density"] <= t_hi:
                    score += 0.2
            else:
                score = 0.1  # Uniform low prior for genres without signatures

            # Boost from PANNs tags
            if panns_tags:
                for tag, conf in panns_tags.items():
                    tag_lower = tag.lower()
                    if genre in tag_lower or tag_lower in genre:
                        score += conf * 0.3

            scores[genre] = round(min(score, 1.0), 4)

        return scores

    def classify_era(self, filepath: str) -> dict[str, float]:
        """Return era affinity scores. Rule-based on spectral characteristics."""
        features = self._extract_features(filepath)
        scores = {}

        centroid = features["centroid"]
        bandwidth = features["bandwidth"]

        # Older eras: narrower bandwidth, lower centroid
        # Newer eras: wider bandwidth, higher centroid (generally)
        for era in ERAS:
            decade = int(era[:4])
            # Simple model: more modern = brighter, wider bandwidth
            modernity = (decade - 1970) / 50.0  # 0 to 1
            centroid_score = 1.0 - abs(centroid / 6000.0 - modernity)
            bw_score = 1.0 - abs(bandwidth / 5000.0 - modernity)
            scores[era] = round(float(np.clip((centroid_score + bw_score) / 2, 0, 1)), 4)

        return scores

    def _extract_features(self, filepath: str) -> dict:
        """Quick feature extraction for classification."""
        audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        duration = len(mono) / sr

        S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mean_spec = np.mean(S ** 2, axis=1)
        total = mean_spec.sum()

        centroid = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
        bandwidth = float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean())

        sub_mask = (freqs >= 20) & (freqs < 100)
        sub_bass_ratio = float(mean_spec[sub_mask].sum() / total) if total > 0 else 0

        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        transient_density = len(onsets) / duration if duration > 0 else 0

        return {
            "centroid": centroid,
            "bandwidth": bandwidth,
            "sub_bass_ratio": sub_bass_ratio,
            "transient_density": transient_density,
            "duration": duration,
        }
```

- [ ] **Step 4: Implement style_tagger.py**

```python
# backend/ml/classifiers/style_tagger.py
"""
Style tag predictor: bright/dark/wide/punchy/analog/digital/gritty/clean/warm/airy/tight/loose.
Maps from perceptual and spectral features to subjective style descriptors.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf


class StyleTagger:
    """Predict subjective style tags from audio."""

    def tag(self, filepath: str) -> dict[str, float]:
        """Return style tags with confidence scores (0-1)."""
        audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
        n_channels = audio.shape[1]
        mono = audio.mean(axis=1)

        S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mean_spec = np.mean(S ** 2, axis=1)
        total = mean_spec.sum()

        if total < 1e-10:
            return {}

        def band_ratio(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(mean_spec[mask].sum() / total)

        high_energy = band_ratio(4000, sr / 2)
        low_energy = band_ratio(20, 300)
        mid_energy = band_ratio(300, 4000)
        presence = band_ratio(2000, 5000)

        # Onset analysis
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        transient_sharpness = 0.0
        if len(onset_env) > 0 and onset_env.mean() > 0:
            transient_sharpness = onset_env.max() / onset_env.mean()

        # Spectral flatness
        flatness = float(librosa.feature.spectral_flatness(S=S).mean())

        # Width
        if n_channels >= 2:
            corr = np.corrcoef(audio[:, 0], audio[:, 1])[0, 1]
            width_score = 1.0 - abs(corr) if not np.isnan(corr) else 0.0
        else:
            width_score = 0.0

        tags = {}

        # Bright vs Dark
        tags["bright"] = round(float(np.clip(high_energy * 5, 0, 1)), 4)
        tags["dark"] = round(float(np.clip(1.0 - high_energy * 3, 0, 1)), 4)

        # Wide
        tags["wide"] = round(float(np.clip(width_score, 0, 1)), 4)

        # Punchy
        tags["punchy"] = round(float(np.clip(transient_sharpness / 8.0, 0, 1)), 4)

        # Warm
        warm_band = band_ratio(200, 800)
        tags["warm"] = round(float(np.clip(warm_band * 4, 0, 1)), 4)

        # Airy
        air_band = band_ratio(10000, sr / 2)
        tags["airy"] = round(float(np.clip(air_band * 10, 0, 1)), 4)

        # Gritty vs Clean
        tags["gritty"] = round(float(np.clip(flatness * 3 + presence * 2, 0, 1)), 4)
        tags["clean"] = round(float(np.clip(1.0 - flatness * 2, 0, 1)), 4)

        # Analog vs Digital (approximation: analog = less perfect, warmer)
        spectral_regularity = float(np.std(np.diff(mean_spec)) / (mean_spec.mean() + 1e-10))
        tags["analog"] = round(float(np.clip(warm_band * 2 + (1 - flatness), 0, 1)), 4)
        tags["digital"] = round(float(np.clip(high_energy * 3 + flatness, 0, 1)), 4)

        # Tight vs Loose
        tags["tight"] = round(float(np.clip(transient_sharpness / 6.0, 0, 1)), 4)
        tags["loose"] = round(float(np.clip(1.0 - transient_sharpness / 8.0, 0, 1)), 4)

        return tags
```

- [ ] **Step 5: Implement quality_scorer.py**

```python
# backend/ml/classifiers/quality_scorer.py
"""
Commercial readiness / quality scorer.
Evaluates: signal integrity, spectral balance, dynamic range, noise floor,
transient clarity, tonal quality.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf


class QualityScorer:
    """Score the commercial readiness / quality of an audio sample."""

    def score(self, filepath: str) -> float:
        """
        Return a quality score from 0.0 (unusable) to 1.0 (professional grade).
        """
        audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)

        if len(mono) < 100:
            return 0.0

        scores = []

        # 1. Signal presence (not silence)
        rms = np.sqrt(np.mean(mono ** 2))
        peak = np.max(np.abs(mono))
        if rms < 1e-6:
            return 0.0
        scores.append(min(rms * 10, 1.0))

        # 2. Dynamic range (crest factor in reasonable range)
        crest = peak / rms if rms > 0 else 0
        # Good dynamic range: crest between 3-20 dB
        crest_db = 20 * np.log10(crest) if crest > 0 else 0
        if 3 < crest_db < 20:
            scores.append(0.8)
        elif crest_db <= 3:
            scores.append(0.4)  # Too compressed
        else:
            scores.append(0.5)  # Too dynamic / sparse

        # 3. Spectral balance (not all energy in one band)
        S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mean_spec = np.mean(S ** 2, axis=1)
        total = mean_spec.sum()
        if total > 0:
            # Split into 4 bands and check balance
            bands = [
                (20, 250), (250, 2000), (2000, 8000), (8000, sr / 2)
            ]
            ratios = []
            for lo, hi in bands:
                mask = (freqs >= lo) & (freqs < hi)
                ratios.append(mean_spec[mask].sum() / total)
            # Entropy-based balance score
            ratios = np.array(ratios)
            ratios = ratios[ratios > 0]
            entropy = -np.sum(ratios * np.log2(ratios + 1e-10))
            max_entropy = np.log2(len(bands))
            scores.append(entropy / max_entropy)
        else:
            scores.append(0.0)

        # 4. No DC offset
        dc_offset = abs(np.mean(mono))
        if dc_offset < 0.01:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - dc_offset * 10))

        # 5. No clipping
        clip_ratio = np.mean(np.abs(mono) > 0.99)
        scores.append(max(0, 1.0 - clip_ratio * 20))

        # 6. Noise floor quality (SNR proxy)
        # Sort RMS frames, compare bottom 10% to top 10%
        hop = max(1, sr // 100)
        n_frames = len(mono) // hop
        if n_frames > 10:
            frame_rms = np.array([
                np.sqrt(np.mean(mono[i*hop:(i+1)*hop] ** 2))
                for i in range(n_frames)
            ])
            sorted_rms = np.sort(frame_rms)
            noise_floor = np.mean(sorted_rms[:max(1, n_frames // 10)])
            signal_level = np.mean(sorted_rms[-max(1, n_frames // 10):])
            if noise_floor > 0:
                snr = signal_level / noise_floor
                scores.append(min(snr / 100, 1.0))
            else:
                scores.append(1.0)
        else:
            scores.append(0.5)

        return round(float(np.mean(scores)), 4)
```

- [ ] **Step 6: Run all classifier tests**

Run: `python -m pytest tests/test_genre_era_classifier.py tests/test_style_tagger.py tests/test_quality_scorer.py -v`
Expected: All 9 tests PASS

- [ ] **Step 7: Commit**

```bash
git add backend/ml/classifiers/ tests/test_genre_era_classifier.py tests/test_style_tagger.py tests/test_quality_scorer.py
git commit -m "feat(phase1): genre/era classifier, style tagger, quality scorer"
```

---

## Task 14: Sample Store (Database Layer)

**Files:**
- Create: `backend/ml/db/__init__.py`
- Create: `backend/ml/db/sample_store.py`
- Create: `tests/test_sample_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sample_store.py
import pytest
import tempfile
from pathlib import Path
from src.db.sample_store import SampleStore
from src.models.sample_profile import SampleProfile, CoreDescriptors, PredictedLabels


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = SampleStore(str(db_path))
    s.init()
    return s


@pytest.fixture
def sample_profile():
    p = SampleProfile(filepath="/samples/kick.wav", filename="kick.wav", file_hash="abc123")
    p.core = CoreDescriptors(duration=0.15, sample_rate=44100, channels=1,
                              rms=0.3, lufs=-14.0, peak=0.9, crest_factor=9.5)
    p.labels = PredictedLabels(role="kick", role_confidence=0.85,
                                tonal=False, genre_affinity={"trap": 0.8})
    return p


def test_save_and_load(store, sample_profile):
    store.save(sample_profile)
    loaded = store.load("/samples/kick.wav")
    assert loaded is not None
    assert loaded.filepath == "/samples/kick.wav"
    assert loaded.core.duration == 0.15
    assert loaded.labels.role == "kick"


def test_load_nonexistent(store):
    result = store.load("/nonexistent.wav")
    assert result is None


def test_update_existing(store, sample_profile):
    store.save(sample_profile)
    sample_profile.core.rms = 0.5
    sample_profile.labels.role = "bass"
    store.save(sample_profile)
    loaded = store.load("/samples/kick.wav")
    assert loaded.core.rms == 0.5
    assert loaded.labels.role == "bass"


def test_list_all(store, sample_profile):
    store.save(sample_profile)
    p2 = SampleProfile(filepath="/samples/snare.wav", filename="snare.wav")
    store.save(p2)
    all_profiles = store.list_all()
    assert len(all_profiles) == 2


def test_count(store, sample_profile):
    assert store.count() == 0
    store.save(sample_profile)
    assert store.count() == 1


def test_delete(store, sample_profile):
    store.save(sample_profile)
    store.delete("/samples/kick.wav")
    assert store.load("/samples/kick.wav") is None


def test_search_by_role(store, sample_profile):
    store.save(sample_profile)
    results = store.search_by_role("kick")
    assert len(results) == 1
    assert results[0].filepath == "/samples/kick.wav"


def test_needs_reanalysis(store, sample_profile):
    store.save(sample_profile)
    assert not store.needs_reanalysis("/samples/kick.wav", "abc123")
    assert store.needs_reanalysis("/samples/kick.wav", "different_hash")
    assert store.needs_reanalysis("/nonexistent.wav", "any")
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_sample_store.py -v`
Expected: FAIL

- [ ] **Step 3: Implement sample_store.py**

```python
# backend/ml/db/__init__.py
# (empty)
```

```python
# backend/ml/db/sample_store.py
"""
Persistent storage for SampleProfile objects.
Uses SQLite with JSON columns for rich nested data.
Designed to coexist with the existing backend/db/database.py.
"""
from __future__ import annotations
import sqlite3
import json
from contextlib import contextmanager
from src.models.sample_profile import SampleProfile


class SampleStore:
    """SQLite-backed storage for sample profiles."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _db(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self):
        """Create the sample_profiles table."""
        with self._db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sample_profiles (
                    filepath TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT,
                    source TEXT DEFAULT 'local',
                    core TEXT DEFAULT '{}',
                    spectral TEXT DEFAULT '{}',
                    harmonic TEXT DEFAULT '{}',
                    transients TEXT DEFAULT '{}',
                    perceptual TEXT DEFAULT '{}',
                    embeddings TEXT DEFAULT '{}',
                    labels TEXT DEFAULT '{}',
                    created_at REAL DEFAULT (strftime('%s','now')),
                    updated_at REAL DEFAULT (strftime('%s','now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_profiles_hash
                ON sample_profiles(file_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_profiles_role
                ON sample_profiles(json_extract(labels, '$.role'))
            """)

    def save(self, profile: SampleProfile):
        """Insert or update a sample profile."""
        d = profile.to_dict()
        with self._db() as conn:
            conn.execute("""
                INSERT INTO sample_profiles
                    (filepath, filename, file_hash, source, core, spectral,
                     harmonic, transients, perceptual, embeddings, labels, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
                ON CONFLICT(filepath) DO UPDATE SET
                    filename=excluded.filename,
                    file_hash=excluded.file_hash,
                    source=excluded.source,
                    core=excluded.core,
                    spectral=excluded.spectral,
                    harmonic=excluded.harmonic,
                    transients=excluded.transients,
                    perceptual=excluded.perceptual,
                    embeddings=excluded.embeddings,
                    labels=excluded.labels,
                    updated_at=strftime('%s','now')
            """, (
                d["filepath"], d["filename"], d["file_hash"], d["source"],
                json.dumps(d["core"]), json.dumps(d["spectral"]),
                json.dumps(d["harmonic"]), json.dumps(d["transients"]),
                json.dumps(d["perceptual"]), json.dumps(d["embeddings"]),
                json.dumps(d["labels"]),
            ))

    def load(self, filepath: str) -> SampleProfile | None:
        """Load a sample profile by filepath."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM sample_profiles WHERE filepath = ?", (filepath,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_profile(row)

    def delete(self, filepath: str):
        """Delete a sample profile."""
        with self._db() as conn:
            conn.execute("DELETE FROM sample_profiles WHERE filepath = ?", (filepath,))

    def list_all(self, limit: int = 0) -> list[SampleProfile]:
        """List all sample profiles."""
        with self._db() as conn:
            query = "SELECT * FROM sample_profiles ORDER BY updated_at DESC"
            if limit > 0:
                query += f" LIMIT {limit}"
            rows = conn.execute(query).fetchall()
            return [self._row_to_profile(r) for r in rows]

    def count(self) -> int:
        """Count total profiles."""
        with self._db() as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM sample_profiles").fetchone()
            return row["c"]

    def search_by_role(self, role: str) -> list[SampleProfile]:
        """Find samples by their classified role."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM sample_profiles WHERE json_extract(labels, '$.role') = ?",
                (role,)
            ).fetchall()
            return [self._row_to_profile(r) for r in rows]

    def needs_reanalysis(self, filepath: str, current_hash: str) -> bool:
        """Check if a file needs (re)analysis based on its hash."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT file_hash FROM sample_profiles WHERE filepath = ?", (filepath,)
            ).fetchone()
            if not row:
                return True
            return row["file_hash"] != current_hash

    def _row_to_profile(self, row: sqlite3.Row) -> SampleProfile:
        """Convert a database row to a SampleProfile."""
        d = {
            "filepath": row["filepath"],
            "filename": row["filename"],
            "file_hash": row["file_hash"],
            "source": row["source"],
            "core": json.loads(row["core"]),
            "spectral": json.loads(row["spectral"]),
            "harmonic": json.loads(row["harmonic"]),
            "transients": json.loads(row["transients"]),
            "perceptual": json.loads(row["perceptual"]),
            "embeddings": json.loads(row["embeddings"]),
            "labels": json.loads(row["labels"]),
        }
        return SampleProfile.from_dict(d)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_sample_store.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/db/ tests/test_sample_store.py
git commit -m "feat(phase1): sample store — SQLite persistence for SampleProfile with JSON columns"
```

---

## Task 15: Vector Index (FAISS)

**Files:**
- Create: `backend/ml/retrieval/__init__.py`
- Create: `backend/ml/retrieval/vector_index.py`
- Create: `tests/test_vector_index.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vector_index.py
import numpy as np
import pytest
from src.retrieval.vector_index import VectorIndex


@pytest.fixture
def index():
    return VectorIndex(dim=512)


@pytest.fixture
def populated_index():
    idx = VectorIndex(dim=8)
    rng = np.random.default_rng(42)
    for i in range(100):
        vec = rng.standard_normal(8).astype(np.float32)
        vec /= np.linalg.norm(vec)
        idx.add(f"sample_{i}.wav", vec)
    return idx


def test_empty_index(index):
    assert index.size() == 0


def test_add_and_size(index):
    vec = np.random.randn(512).astype(np.float32)
    index.add("test.wav", vec)
    assert index.size() == 1


def test_search_returns_results(populated_index):
    query = np.random.randn(8).astype(np.float32)
    query /= np.linalg.norm(query)
    results = populated_index.search(query, k=5)
    assert len(results) == 5


def test_search_result_format(populated_index):
    query = np.random.randn(8).astype(np.float32)
    results = populated_index.search(query, k=3)
    for filepath, score in results:
        assert isinstance(filepath, str)
        assert isinstance(score, float)


def test_search_self_is_top_result(populated_index):
    # The vector for sample_0 should be most similar to itself
    vec = populated_index.get_vector("sample_0.wav")
    assert vec is not None
    results = populated_index.search(vec, k=1)
    assert results[0][0] == "sample_0.wav"


def test_save_and_load(populated_index, tmp_path):
    save_path = tmp_path / "test_index"
    populated_index.save(str(save_path))
    loaded = VectorIndex.load(str(save_path))
    assert loaded.size() == 100
    # Search should work on loaded index
    query = np.random.randn(8).astype(np.float32)
    results = loaded.search(query, k=5)
    assert len(results) == 5


def test_remove(populated_index):
    populated_index.remove("sample_0.wav")
    assert populated_index.size() == 99
    assert populated_index.get_vector("sample_0.wav") is None
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_vector_index.py -v`
Expected: FAIL

- [ ] **Step 3: Implement vector_index.py**

```python
# backend/ml/retrieval/__init__.py
# (empty)
```

```python
# backend/ml/retrieval/vector_index.py
"""
FAISS-based vector index for fast nearest-neighbor sample retrieval.
Supports add, remove, search, save/load.
"""
from __future__ import annotations
import json
import numpy as np
import faiss
from pathlib import Path


class VectorIndex:
    """FAISS vector index mapping filepaths to embedding vectors."""

    def __init__(self, dim: int):
        self.dim = dim
        # Use IndexFlatIP (inner product) for cosine similarity on normalized vectors
        self.index = faiss.IndexFlatIP(dim)
        self._id_to_filepath: list[str] = []
        self._filepath_to_id: dict[str, int] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, filepath: str, vector: np.ndarray):
        """Add a vector for a filepath. Overwrites if exists."""
        if filepath in self._filepath_to_id:
            self.remove(filepath)

        vec = vector.astype(np.float32).reshape(1, -1)
        # L2 normalize for cosine similarity via inner product
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        idx = len(self._id_to_filepath)
        self.index.add(vec)
        self._id_to_filepath.append(filepath)
        self._filepath_to_id[filepath] = idx
        self._vectors[filepath] = vec.flatten()

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Search for k nearest neighbors. Returns [(filepath, score), ...]."""
        if self.index.ntotal == 0:
            return []

        q = query.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            q = q / norm

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._id_to_filepath):
                results.append((self._id_to_filepath[idx], float(score)))
        return results

    def get_vector(self, filepath: str) -> np.ndarray | None:
        """Get the stored vector for a filepath."""
        return self._vectors.get(filepath)

    def remove(self, filepath: str):
        """Remove a filepath from the index. Rebuilds index."""
        if filepath not in self._filepath_to_id:
            return
        del self._vectors[filepath]
        self._rebuild()

    def size(self) -> int:
        return self.index.ntotal

    def _rebuild(self):
        """Rebuild the FAISS index from stored vectors."""
        self.index = faiss.IndexFlatIP(self.dim)
        self._id_to_filepath = []
        self._filepath_to_id = {}

        for fp, vec in self._vectors.items():
            idx = len(self._id_to_filepath)
            self.index.add(vec.reshape(1, -1))
            self._id_to_filepath.append(fp)
            self._filepath_to_id[fp] = idx

    def save(self, path: str):
        """Save index to disk (FAISS index + metadata)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        metadata = {
            "dim": self.dim,
            "filepaths": self._id_to_filepath,
        }
        with open(p / "metadata.json", "w") as f:
            json.dump(metadata, f)
        # Save vectors for rebuild capability
        np.savez(str(p / "vectors.npz"), **{
            fp.replace("/", "__SLASH__"): vec
            for fp, vec in self._vectors.items()
        })

    @classmethod
    def load(cls, path: str) -> VectorIndex:
        """Load index from disk."""
        p = Path(path)
        with open(p / "metadata.json") as f:
            metadata = json.load(f)

        idx = cls(dim=metadata["dim"])
        idx.index = faiss.read_index(str(p / "index.faiss"))
        idx._id_to_filepath = metadata["filepaths"]
        idx._filepath_to_id = {fp: i for i, fp in enumerate(idx._id_to_filepath)}

        # Load vectors
        data = np.load(str(p / "vectors.npz"))
        for key in data:
            fp = key.replace("__SLASH__", "/")
            idx._vectors[fp] = data[key]

        return idx
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_vector_index.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ml/retrieval/ tests/test_vector_index.py
git commit -m "feat(phase1): FAISS vector index — add, search, save/load, remove"
```

---

## Task 16: Ingestion Pipeline

**Files:**
- Create: `backend/ml/pipeline/__init__.py`
- Create: `backend/ml/pipeline/ingestion.py`
- Create: `backend/ml/pipeline/batch_processor.py`
- Create: `tests/test_ingestion.py`
- Create: `tests/test_batch_processor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ingestion.py
import pytest
from src.pipeline.ingestion import analyze_sample
from src.models.sample_profile import SampleProfile


def test_analyze_sample_returns_profile(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert isinstance(profile, SampleProfile)


def test_core_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert profile.core.duration > 0
    assert profile.core.sample_rate > 0


def test_spectral_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert profile.spectral.centroid > 0


def test_harmonic_populated(sine_440hz):
    profile = analyze_sample(str(sine_440hz), skip_embeddings=True)
    assert profile.harmonic.f0 > 0


def test_transients_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert profile.transients.onset_count >= 1


def test_perceptual_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert profile.perceptual.punch > 0


def test_labels_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert profile.labels.role != "unknown" or profile.labels.role_confidence == 0


def test_loop_detection_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    # One-shot kick should not be a loop
    assert profile.labels.is_loop is False
```

```python
# tests/test_batch_processor.py
import pytest
import soundfile as sf
import numpy as np
from src.pipeline.batch_processor import BatchProcessor


@pytest.fixture
def sample_dir(tmp_path, sample_rate):
    """Create a directory with multiple test audio files."""
    for name in ["kick.wav", "snare.wav", "hat.wav"]:
        t = np.linspace(0, 0.1, sample_rate // 10, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        sf.write(str(tmp_path / name), audio, sample_rate)
    return tmp_path


def test_batch_discovers_files(sample_dir):
    processor = BatchProcessor(skip_embeddings=True)
    files = processor.discover_audio_files(str(sample_dir))
    assert len(files) == 3


def test_batch_processes_all(sample_dir, tmp_path):
    db_path = tmp_path / "test_batch.db"
    processor = BatchProcessor(skip_embeddings=True, db_path=str(db_path))
    results = processor.process_directory(str(sample_dir))
    assert results["processed"] == 3
    assert results["failed"] == 0


def test_batch_reports_progress(sample_dir, tmp_path):
    db_path = tmp_path / "test_batch.db"
    processor = BatchProcessor(skip_embeddings=True, db_path=str(db_path))
    progress_updates = []
    processor.process_directory(
        str(sample_dir),
        on_progress=lambda done, total: progress_updates.append((done, total))
    )
    assert len(progress_updates) == 3
    assert progress_updates[-1] == (3, 3)
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_ingestion.py tests/test_batch_processor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ingestion.py**

```python
# backend/ml/pipeline/__init__.py
# (empty)
```

```python
# backend/ml/pipeline/ingestion.py
"""
Single-sample analysis pipeline. Orchestrates all extractors into a complete SampleProfile.
"""
from __future__ import annotations
import logging
from pathlib import Path
from src.models.sample_profile import SampleProfile
from src.analysis.core_descriptors import extract_core_descriptors
from src.analysis.spectral_descriptors import extract_spectral_descriptors
from src.analysis.harmonic_descriptors import extract_harmonic_descriptors
from src.analysis.transient_descriptors import extract_transient_descriptors
from src.analysis.perceptual_descriptors import extract_perceptual_descriptors
from src.analysis.loop_detection import detect_loop
from src.classifiers.role_classifier import RoleClassifier
from src.classifiers.genre_era_classifier import GenreEraClassifier
from src.classifiers.style_tagger import StyleTagger
from src.classifiers.quality_scorer import QualityScorer

logger = logging.getLogger(__name__)

# Singletons for classifiers (stateless, safe to share)
_role_clf = RoleClassifier()
_genre_clf = GenreEraClassifier()
_style_tagger = StyleTagger()
_quality_scorer = QualityScorer()


def analyze_sample(filepath: str, skip_embeddings: bool = False,
                   embedding_manager=None, file_hash: str = "",
                   source: str = "local") -> SampleProfile:
    """
    Run the complete analysis pipeline on a single audio file.

    Args:
        filepath: Path to the audio file.
        skip_embeddings: If True, skip ML embedding extraction (faster).
        embedding_manager: Optional EmbeddingManager instance (reuse across calls).
        file_hash: Pre-computed file hash for cache invalidation.
        source: Sample source label (local/splice/loopcloud).

    Returns:
        Complete SampleProfile with all descriptors populated.
    """
    path = Path(filepath)
    profile = SampleProfile(
        filepath=str(path),
        filename=path.name,
        file_hash=file_hash,
        source=source,
    )

    # Stage 1: DSP features (all local, no ML)
    try:
        profile.core = extract_core_descriptors(filepath)
    except Exception as e:
        logger.error(f"Core extraction failed for {filepath}: {e}")

    try:
        profile.spectral = extract_spectral_descriptors(filepath)
    except Exception as e:
        logger.error(f"Spectral extraction failed for {filepath}: {e}")

    try:
        profile.harmonic = extract_harmonic_descriptors(filepath)
    except Exception as e:
        logger.error(f"Harmonic extraction failed for {filepath}: {e}")

    try:
        profile.transients = extract_transient_descriptors(filepath)
    except Exception as e:
        logger.error(f"Transient extraction failed for {filepath}: {e}")

    try:
        profile.perceptual = extract_perceptual_descriptors(filepath)
    except Exception as e:
        logger.error(f"Perceptual extraction failed for {filepath}: {e}")

    # Stage 2: Loop detection
    try:
        is_loop, loop_conf = detect_loop(filepath)
        profile.labels.is_loop = is_loop
        profile.labels.loop_confidence = loop_conf
    except Exception as e:
        logger.error(f"Loop detection failed for {filepath}: {e}")

    # Stage 3: Embeddings (optional, requires ML models)
    if not skip_embeddings and embedding_manager is not None:
        try:
            profile.embeddings = embedding_manager.extract_all(filepath)
        except Exception as e:
            logger.error(f"Embedding extraction failed for {filepath}: {e}")

    # Stage 4: Classification
    try:
        panns_tags = profile.embeddings.panns_tags if profile.embeddings.panns_tags else None
        role, role_conf = _role_clf.classify(filepath, filename_hint=path.name,
                                              panns_tags=panns_tags)
        profile.labels.role = role
        profile.labels.role_confidence = role_conf
    except Exception as e:
        logger.error(f"Role classification failed for {filepath}: {e}")

    try:
        profile.labels.tonal = profile.harmonic.tonalness > 0.5
    except Exception:
        pass

    try:
        profile.labels.genre_affinity = _genre_clf.classify_genre(filepath)
        profile.labels.era_affinity = _genre_clf.classify_era(filepath)
    except Exception as e:
        logger.error(f"Genre/era classification failed for {filepath}: {e}")

    try:
        profile.labels.style_tags = _style_tagger.tag(filepath)
    except Exception as e:
        logger.error(f"Style tagging failed for {filepath}: {e}")

    try:
        profile.labels.commercial_readiness = _quality_scorer.score(filepath)
    except Exception as e:
        logger.error(f"Quality scoring failed for {filepath}: {e}")

    return profile
```

- [ ] **Step 4: Implement batch_processor.py**

```python
# backend/ml/pipeline/batch_processor.py
"""
Batch processing pipeline. Discovers audio files, analyzes in parallel,
persists to database with progress reporting.
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from src.pipeline.ingestion import analyze_sample
from src.db.sample_store import SampleStore

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}


class BatchProcessor:
    """Process directories of audio files in parallel."""

    def __init__(self, skip_embeddings: bool = False,
                 embedding_manager=None,
                 db_path: str | None = None,
                 max_workers: int = 4):
        self.skip_embeddings = skip_embeddings
        self.embedding_manager = embedding_manager
        self.max_workers = max_workers
        self.store = None
        if db_path:
            self.store = SampleStore(db_path)
            self.store.init()

    def discover_audio_files(self, directory: str) -> list[str]:
        """Find all audio files in a directory tree."""
        files = []
        for root, _, filenames in os.walk(directory):
            for fname in filenames:
                if Path(fname).suffix.lower() in AUDIO_EXTENSIONS:
                    files.append(os.path.join(root, fname))
        return sorted(files)

    def process_directory(self, directory: str,
                          on_progress: Callable[[int, int], None] | None = None,
                          source: str = "local") -> dict:
        """
        Process all audio files in a directory.
        Returns {"processed": int, "failed": int, "skipped": int}.
        """
        files = self.discover_audio_files(directory)
        total = len(files)
        processed = 0
        failed = 0
        skipped = 0

        def _process_one(filepath: str):
            file_hash = _quick_hash(filepath)
            if self.store and not self.store.needs_reanalysis(filepath, file_hash):
                return "skipped"
            try:
                profile = analyze_sample(
                    filepath,
                    skip_embeddings=self.skip_embeddings,
                    embedding_manager=self.embedding_manager,
                    file_hash=file_hash,
                    source=source,
                )
                if self.store:
                    self.store.save(profile)
                return "ok"
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                return "failed"

        # Process with thread pool (IO-bound file reading benefits from threads)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_process_one, f): f for f in files}
            done_count = 0
            for future in as_completed(futures):
                result = future.result()
                if result == "ok":
                    processed += 1
                elif result == "failed":
                    failed += 1
                else:
                    skipped += 1
                done_count += 1
                if on_progress:
                    on_progress(done_count, total)

        return {"processed": processed, "failed": failed, "skipped": skipped, "total": total}


def _quick_hash(filepath: str) -> str:
    """Quick hash based on path + size + mtime (no content reading)."""
    stat = os.stat(filepath)
    return f"{filepath}:{stat.st_size}:{stat.st_mtime}"
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_ingestion.py tests/test_batch_processor.py -v`
Expected: All 11 tests PASS

- [ ] **Step 6: Commit**

```bash
git add backend/ml/pipeline/ tests/test_ingestion.py tests/test_batch_processor.py
git commit -m "feat(phase1): ingestion pipeline + batch processor with parallel execution"
```

---

## Task 17: Integration — Wire Into Existing Backend

**Files:**
- Modify: `backend/config.py`
- Modify: `backend/indexer.py`
- Modify: `backend/server.py`
- Modify: `backend/routes/samples.py`

This task bridges the new `backend/ml/` analysis pipeline with the existing FastAPI backend. The new pipeline runs alongside the existing one — the old `sample_analyzer.py` stays until the new system is validated.

- [ ] **Step 1: Add src to Python path in config.py**

Add to `backend/config.py`:

```python
# ── New analysis pipeline ─────────────────────────────────────────────────
import sys
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROFILE_DB_PATH = BACKEND_DIR / "sample_profiles.db"
VECTOR_INDEX_DIR = BACKEND_DIR / "vector_indexes"
VECTOR_INDEX_DIR.mkdir(exist_ok=True)
```

- [ ] **Step 2: Update indexer.py to use new pipeline**

Add a parallel indexing path in `backend/indexer.py` that calls the new pipeline:

```python
# Add at top of indexer.py
from config import PROFILE_DB_PATH

def background_index_v2():
    """Background indexing using the new Phase 1 analysis pipeline."""
    from src.pipeline.batch_processor import BatchProcessor
    from src.db.sample_store import SampleStore

    store = SampleStore(str(PROFILE_DB_PATH))
    store.init()

    processor = BatchProcessor(
        skip_embeddings=True,  # Start without embeddings for speed
        db_path=str(PROFILE_DB_PATH),
        max_workers=4,
    )

    # Index local samples
    from config import SAMPLE_DIR
    if SAMPLE_DIR.exists():
        result = processor.process_directory(str(SAMPLE_DIR), source="local")
        print(f"  ✓ Indexed {result['processed']} local samples (v2 pipeline)")

    # Index external libraries
    from config import SPLICE_DIRS, LOOPCLOUD_DIRS
    for d in SPLICE_DIRS:
        if d.exists():
            result = processor.process_directory(str(d), source="splice")
            print(f"  ✓ Indexed {result['processed']} Splice samples (v2 pipeline)")
    for d in LOOPCLOUD_DIRS:
        if d.exists():
            result = processor.process_directory(str(d), source="loopcloud")
            print(f"  ✓ Indexed {result['processed']} Loopcloud samples (v2 pipeline)")
```

- [ ] **Step 3: Add v2 indexing to server startup**

Add to `backend/server.py` after the existing `background_index()` call:

```python
# Start v2 indexing in background
import threading
from indexer import background_index_v2
t2 = threading.Thread(target=background_index_v2, daemon=True)
t2.start()
print("  ⟳ V2 sample analysis running in background...")
```

- [ ] **Step 4: Add a /samples/v2/profile endpoint**

Create a new route that exposes the rich profile data:

Add to `backend/routes/samples.py`:

```python
@router.get("/samples/v2/profile/{sample_path:path}")
def get_sample_profile(sample_path: str):
    """Get the full v2 analysis profile for a sample."""
    from config import PROFILE_DB_PATH
    from src.db.sample_store import SampleStore

    store = SampleStore(str(PROFILE_DB_PATH))
    store.init()

    profile = store.load(sample_path)
    if not profile:
        # Try finding by filename
        from routes.samples import find_sample_file
        found = find_sample_file(sample_path)
        if found:
            profile = store.load(str(found))

    if not profile:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Profile not found")

    return profile.to_dict()
```

- [ ] **Step 5: Add a /samples/v2/stats endpoint**

```python
@router.get("/samples/v2/stats")
def get_indexing_stats():
    """Get v2 pipeline indexing statistics."""
    from config import PROFILE_DB_PATH
    from src.db.sample_store import SampleStore

    store = SampleStore(str(PROFILE_DB_PATH))
    store.init()

    return {
        "total_profiles": store.count(),
        "pipeline_version": "phase1",
    }
```

- [ ] **Step 6: Test the integration manually**

Run: `cd /path/to/RESONATE/backend && python server.py`
Expected: Server starts, v2 indexing begins in background, no errors.

Run: `curl http://localhost:8000/samples/v2/stats`
Expected: `{"total_profiles": <N>, "pipeline_version": "phase1"}`

- [ ] **Step 7: Commit**

```bash
git add backend/config.py backend/indexer.py backend/server.py backend/routes/samples.py
git commit -m "feat(phase1): integrate new analysis pipeline with existing backend"
```

---

## Task 18: Full Test Suite Pass

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (embedding tests may SKIP if models not downloaded)

- [ ] **Step 2: Run with coverage**

Run: `python -m pytest tests/ --cov=src --cov-report=term-missing`
Expected: >80% coverage on `backend/ml/` modules

- [ ] **Step 3: Fix any failures**

Address any test failures discovered in the full run.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(phase1): complete Phase 1 — Sample Analyzer Foundation

All modules implemented and tested:
- Core/spectral/harmonic/transient/perceptual descriptors
- Loop detection
- CLAP/PANNs/AST embedding extractors
- Role/genre/era classifiers, style tagger, quality scorer
- SQLite sample store with JSON columns
- FAISS vector index
- Ingestion pipeline + batch processor
- Backend integration with v2 endpoints"
```

---

## Dependency Graph

```
Task 1 (Profile + Tests) ──┬── Task 2 (Core) ──────────────┐
                            ├── Task 3 (Spectral) ──────────┤
                            ├── Task 4 (Harmonic) ──────────┤
                            ├── Task 5 (Transient) ─────────┤
                            ├── Task 6 (Perceptual) ────────┤
                            └── Task 7 (Loop) ──────────────┤
                                                            │
Task 1 ─── Task 8 (CLAP) ──────┐                           │
           Task 9 (PANNs) ─────┼── Task 11 (Emb Manager) ──┤
           Task 10 (AST) ──────┘                            │
                                                            │
Task 1 ─── Task 12 (Role Clf) ─────────────────────────────┤
           Task 13 (Genre/Style/Quality) ───────────────────┤
                                                            │
Task 1 ─── Task 14 (Sample Store) ─────────────────────────┤
           Task 15 (Vector Index) ─────────────────────────┤
                                                            │
                            ┌───────────────────────────────┘
                            ▼
                   Task 16 (Ingestion Pipeline)
                            │
                            ▼
                   Task 17 (Backend Integration)
                            │
                            ▼
                   Task 18 (Full Test Suite)
```

**Parallelizable pairs:**
- Tasks 2+3, 4+5, 6+7 (all DSP, independent)
- Tasks 8+9, 9+10 (all embeddings, independent)
- Tasks 12+13 (classifiers, independent)
- Tasks 14+15 (storage, independent)
