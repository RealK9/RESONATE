# Phase 2 — Mix Intelligence

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given a 30-second uploaded mix, create a comprehensive mix context profile: BPM, key, style cluster probabilities, sound-role presence, spectral occupancy map, density map, width map, and a deficiency/opportunity vector that diagnoses what the mix needs.

**Architecture:** Three new modules in `backend/ml/analysis/`: `mix_profile.py` (full mix analysis), `style_classifier.py` (style cluster detection), and `needs_engine.py` (deficiency diagnosis). A `MixProfile` dataclass in `backend/ml/models/` holds all outputs. Integration adds `/analyze/v2` endpoint alongside the existing one.

**Tech Stack:** Python 3.11+, librosa, numpy, scipy, pyloudnorm, existing Phase 1 modules, pytest

---

## File Structure

```
backend/ml/
├── models/
│   └── mix_profile.py          # MixProfile dataclass
├── analysis/
│   ├── mix_analyzer.py         # Full mix-level analysis
│   ├── style_classifier.py     # Style cluster classification
│   └── needs_engine.py         # Deficiency/opportunity diagnosis
tests/
├── test_mix_profile.py
├── test_mix_analyzer.py
├── test_style_classifier.py
└── test_needs_engine.py
```

---

## Dependency Graph

```
Task 1 (MixProfile dataclass) ──┬── Task 2 (Mix Analyzer)
                                 ├── Task 3 (Style Classifier)
                                 └── Task 4 (Needs Engine) ── depends on Task 2+3
Task 5 (Backend Integration) ── depends on all above
Task 6 (Full test suite)
```

Parallelizable: Tasks 2+3 (independent analysis modules)

---

## Task 1: MixProfile Dataclass

**Files:**
- Create: `backend/ml/models/mix_profile.py`
- Create: `tests/test_mix_profile.py`

**MixProfile contains:**

```python
@dataclass
class MixLevelAnalysis:
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
    section_energy: list[float]  # energy per section (normalized)

@dataclass
class SpectralOccupancy:
    """Spectral energy by band over time."""
    bands: list[str]  # band names
    time_frames: int = 0
    occupancy_matrix: list[list[float]]  # [band][time] -> 0-1
    mean_by_band: list[float]  # average per band

@dataclass
class StereoWidth:
    """Stereo width analysis by frequency band."""
    bands: list[str]
    width_by_band: list[float]  # 0=mono, 1=full stereo per band
    overall_width: float = 0.0
    correlation: float = 0.0  # L/R correlation

@dataclass
class SourceRolePresence:
    """Estimated presence and confidence of each sound role in the mix."""
    roles: dict[str, float]  # role -> confidence 0-1
    # kick, snare_clap, hats_tops, bass, lead, chord_support,
    # pad, vocal_texture, fx_transitions, ambience

@dataclass
class StyleCluster:
    """Style cluster classification."""
    cluster_probabilities: dict[str, float]  # cluster -> 0-1
    primary_cluster: str = ""
    era_estimate: str = ""

@dataclass
class NeedOpportunity:
    """A single diagnosed need or opportunity."""
    category: str  # spectral/role/dynamic/spatial/arrangement
    description: str  # human-readable
    severity: float  # 0-1 (1 = critical)
    recommendation_policy: str  # fill_role/reinforce/polish/contrast/movement/etc

@dataclass
class MixProfile:
    filepath: str = ""
    filename: str = ""
    analysis: MixLevelAnalysis
    spectral_occupancy: SpectralOccupancy
    stereo_width: StereoWidth
    source_roles: SourceRolePresence
    style: StyleCluster
    needs: list[NeedOpportunity]
    density_map: list[float]  # density per time segment
```

---

## Task 2: Mix Analyzer

Core mix-level analysis: BPM, key, loudness, spectral occupancy, stereo width, source-role inference, density map, section energy.

**Key algorithms:**
- BPM: librosa.beat.beat_track with tempo confidence
- Key: chroma-based key detection (Krumhansl-Kessler)
- Spectral occupancy: 10-band STFT energy over time
- Stereo width: per-band L/R correlation
- Source-role inference: band energy + transient patterns heuristics
- Density: spectral flux + onset density over time
- Section energy: RMS energy in equal-length segments

---

## Task 3: Style Classifier

Classify the mix into style clusters. Uses spectral/rhythmic features to estimate probabilities across defined clusters.

**Style clusters:**
- 2010s_edm_drop, 2020s_melodic_house, 2000s_pop_chorus, 1990s_boom_bap,
- modern_trap, modern_drill, melodic_techno, afro_house, cinematic,
- lo_fi_chill, dnb, ambient, r_and_b, pop_production

**Method:** Feature vector (BPM, centroid, sub-bass ratio, transient density, width, harmonic density) → distance to cluster centroids → softmax probabilities

---

## Task 4: Needs Engine

The heart of the system. Diagnoses what the mix needs based on:
1. Spectral gaps (comparing band energies to style priors)
2. Missing roles (what instruments aren't present that should be)
3. Dynamic issues (too compressed, too dynamic)
4. Spatial issues (too narrow, width imbalance)
5. Arrangement gaps (no movement, no transitions)

Each need maps to a **recommendation policy** (from the decision policy layer):
- fill_missing_role, reinforce_existing, improve_polish, increase_contrast,
- add_movement, reduce_emptiness, support_transition, enhance_groove, enhance_lift

---

## Task 5: Backend Integration

- Add `/analyze/v2` endpoint
- Store MixProfile in state alongside existing track_profile
- Add `/analyze/v2/needs` endpoint for just the needs vector

---

## Task 6: Full Test Suite

Run all Phase 1 + Phase 2 tests together.
