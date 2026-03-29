"""
Mix-level analyzer — full profile extraction for uploaded mixes.

Extracts BPM, key, loudness, spectral occupancy, stereo width,
source-role presence heuristics, density map, and section energy.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt

from ml.models.mix_profile import (
    MixLevelAnalysis,
    MixProfile,
    SourceRolePresence,
    SpectralOccupancy,
    StereoWidth,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BAND_EDGES: list[tuple[str, float, float]] = [
    ("sub", 20, 60),
    ("bass", 60, 150),
    ("low_mid", 150, 400),
    ("mid", 400, 1000),
    ("upper_mid", 1000, 2500),
    ("presence", 2500, 5000),
    ("brilliance", 5000, 8000),
    ("air", 8000, 12000),
    ("ultra_high", 12000, 16000),
    ("ceiling", 16000, 20000),
]

BAND_NAMES = [b[0] for b in BAND_EDGES]

# Krumhansl-Kessler key profiles
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Frequency of each pitch class in octave 4 (A4 = 440 Hz)
_TONAL_CENTER_HZ = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
    "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_audio(filepath: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (stereo_2d, mono, sr). Stereo is always (samples, 2)."""
    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        stereo = np.column_stack([audio[:, 0], audio[:, 0]])
    else:
        stereo = audio[:, :2]
    mono = stereo.mean(axis=1)
    return stereo, mono, sr


def _bandpass(signal: np.ndarray, low: float, high: float, sr: int,
              order: int = 4) -> np.ndarray:
    """Apply Butterworth bandpass filter. Clamps to valid Nyquist range."""
    nyq = sr / 2.0
    lo = max(low / nyq, 1e-5)
    hi = min(high / nyq, 0.9999)
    if lo >= hi:
        return np.zeros_like(signal)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfilt(sos, signal)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


# ---------------------------------------------------------------------------
# Key detection (Krumhansl-Kessler)
# ---------------------------------------------------------------------------

def _detect_key(mono: np.ndarray, sr: int) -> tuple[str, float, float]:
    """Return (key_label, confidence, tonal_center_hz)."""
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    chroma_mean = chroma.mean(axis=1)  # (12,)

    best_corr = -2.0
    best_key = "C"
    best_mode = "major"

    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        corr_maj = float(np.corrcoef(rolled, _MAJOR_PROFILE)[0, 1])
        corr_min = float(np.corrcoef(rolled, _MINOR_PROFILE)[0, 1])
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = _KEY_NAMES[shift]
            best_mode = "major"
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = _KEY_NAMES[shift]
            best_mode = "minor"

    confidence = max(0.0, min(1.0, (best_corr + 1.0) / 2.0))  # map -1..1 → 0..1
    label = f"{best_key} {best_mode}"
    tonal_hz = _TONAL_CENTER_HZ.get(best_key, 261.63)
    return label, confidence, tonal_hz


# ---------------------------------------------------------------------------
# BPM
# ---------------------------------------------------------------------------

def _detect_bpm(mono: np.ndarray, sr: int) -> tuple[float, float]:
    """Return (bpm, confidence) with robust octave-aware detection.

    librosa.beat.beat_track frequently returns double or half the true
    tempo (e.g. 262 for a 131 BPM track).  We run multiple estimation
    methods and pick the candidate that falls in the most common music
    range (70-170 BPM), preferring the one closest to the median of all
    candidates.
    """
    # --- Method 1: librosa default beat tracker ---
    tempo1, beat_frames = librosa.beat.beat_track(y=mono, sr=sr)
    bpm1 = float(np.atleast_1d(tempo1)[0])

    # --- Method 2: librosa with prior centered at 120 ---
    tempo2, _ = librosa.beat.beat_track(y=mono, sr=sr, start_bpm=120)
    bpm2 = float(np.atleast_1d(tempo2)[0])

    # --- Method 3: onset-based autocorrelation (plp) ---
    try:
        oenv = librosa.onset.onset_strength(y=mono, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=oenv, sr=sr)
        bpm3 = float(np.atleast_1d(
            librosa.beat.tempo(onset_envelope=oenv, sr=sr)
        )[0])
    except Exception:
        bpm3 = bpm1

    # Collect all raw candidates
    raw_candidates = [bpm1, bpm2, bpm3]

    # Generate octave variants for each candidate and pick the one
    # closest to the sweet spot (70-170 BPM range for most music)
    def _octave_normalize(bpm: float) -> float:
        if bpm <= 0:
            return 120.0
        while bpm > 180:
            bpm /= 2.0
        while bpm < 60:
            bpm *= 2.0
        return bpm

    candidates = [_octave_normalize(b) for b in raw_candidates]

    # Pick the candidate closest to the median (most agreement)
    median = float(np.median(candidates))
    bpm = min(candidates, key=lambda b: abs(b - median))
    bpm = round(bpm, 1)

    # Confidence: how well do methods agree + beat count check
    spread = float(np.std(candidates))
    agreement = max(0.0, 1.0 - spread / 30.0)  # within 30 BPM spread → high conf
    n_beats = len(beat_frames)
    duration = len(mono) / sr
    expected_beats = (bpm / 60.0) * duration if bpm > 0 else 1.0
    beat_ratio = min(1.0, n_beats / max(expected_beats, 1.0))
    confidence = (agreement + beat_ratio) / 2.0

    return bpm, confidence


# ---------------------------------------------------------------------------
# Loudness
# ---------------------------------------------------------------------------

def _compute_loudness(stereo: np.ndarray, mono: np.ndarray, sr: int) -> dict:
    """Return dict with lufs, peak, dynamic_range, loudness_range."""
    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(stereo))
    except Exception:
        lufs = -100.0
    if not np.isfinite(lufs):
        lufs = -100.0

    peak = float(np.max(np.abs(mono)))
    rms = _rms(mono)
    if rms > 1e-10:
        dynamic_range = float(20 * np.log10(peak / rms))
    else:
        dynamic_range = 0.0

    # Loudness Range (LRA): difference between top 5% and bottom 10%
    # of short-term RMS levels
    frame_len = int(0.4 * sr)  # 400ms frames
    hop = int(0.1 * sr)        # 100ms hop
    n_frames = max(1, (len(mono) - frame_len) // hop + 1)
    frame_rms = np.array([
        _rms(mono[i * hop: i * hop + frame_len])
        for i in range(n_frames)
    ])
    # Filter out silence
    active = frame_rms[frame_rms > 1e-8]
    if len(active) >= 2:
        db_levels = 20 * np.log10(active + 1e-12)
        top_5 = float(np.percentile(db_levels, 95))
        bottom_10 = float(np.percentile(db_levels, 10))
        loudness_range = max(0.0, top_5 - bottom_10)
    else:
        loudness_range = 0.0

    return {
        "lufs": lufs,
        "peak": peak,
        "dynamic_range": dynamic_range,
        "loudness_range": loudness_range,
    }


# ---------------------------------------------------------------------------
# Harmonic density
# ---------------------------------------------------------------------------

def _harmonic_density(mono: np.ndarray, sr: int,
                      harmonic: np.ndarray | None = None,
                      percussive: np.ndarray | None = None) -> float:
    """Ratio of harmonic energy to total energy via HPSS."""
    if harmonic is None or percussive is None:
        harmonic, percussive = librosa.effects.hpss(mono)
    h_energy = float(np.sum(harmonic ** 2))
    total = float(np.sum(mono ** 2))
    if total < 1e-12:
        return 0.0
    return min(1.0, h_energy / total)


# ---------------------------------------------------------------------------
# Section energy
# ---------------------------------------------------------------------------

def _section_energy(mono: np.ndarray, n_sections: int = 8) -> list[float]:
    """RMS energy per equal-length section, normalized 0-1."""
    sections = np.array_split(mono, n_sections)
    energies = [_rms(s) for s in sections]
    mx = max(energies) if energies else 1.0
    if mx < 1e-12:
        return [0.0] * n_sections
    return [e / mx for e in energies]


# ---------------------------------------------------------------------------
# Spectral occupancy
# ---------------------------------------------------------------------------

def _spectral_occupancy(mono: np.ndarray, sr: int) -> SpectralOccupancy:
    """10-band spectral energy over time."""
    hop = int(0.1 * sr)  # ~100ms hop
    frame_len = int(0.2 * sr)  # 200ms frames
    n_frames = max(1, (len(mono) - frame_len) // hop + 1)

    matrix: list[list[float]] = []
    for name, lo, hi in BAND_EDGES:
        filtered = _bandpass(mono, lo, hi, sr)
        band_energies = []
        for i in range(n_frames):
            start = i * hop
            end = start + frame_len
            frame = filtered[start:end]
            band_energies.append(float(np.mean(frame ** 2)))
        matrix.append(band_energies)

    # Normalize entire matrix 0-1
    mat = np.array(matrix)
    mx = mat.max()
    if mx > 1e-12:
        mat = mat / mx
    else:
        mat = np.zeros_like(mat)

    occupancy = mat.tolist()
    mean_by_band = [float(np.mean(row)) for row in occupancy]

    return SpectralOccupancy(
        bands=list(BAND_NAMES),
        time_frames=n_frames,
        occupancy_matrix=occupancy,
        mean_by_band=mean_by_band,
    )


# ---------------------------------------------------------------------------
# Stereo width
# ---------------------------------------------------------------------------

def _stereo_width(stereo: np.ndarray, sr: int) -> StereoWidth:
    """Per-band stereo width via L/R correlation."""
    left = stereo[:, 0]
    right = stereo[:, 1]

    width_by_band: list[float] = []
    for name, lo, hi in BAND_EDGES:
        l_filt = _bandpass(left, lo, hi, sr)
        r_filt = _bandpass(right, lo, hi, sr)
        l_std = np.std(l_filt)
        r_std = np.std(r_filt)
        if l_std < 1e-10 or r_std < 1e-10:
            corr = 1.0  # mono / silence → no width
        else:
            corr = float(np.corrcoef(l_filt, r_filt)[0, 1])
            if not np.isfinite(corr):
                corr = 1.0
        w = 1.0 - abs(corr)
        width_by_band.append(max(0.0, min(1.0, w)))

    overall = float(np.mean(width_by_band))

    # Full-band correlation
    l_std = np.std(left)
    r_std = np.std(right)
    if l_std < 1e-10 or r_std < 1e-10:
        full_corr = 1.0
    else:
        full_corr = float(np.corrcoef(left, right)[0, 1])
        if not np.isfinite(full_corr):
            full_corr = 1.0

    return StereoWidth(
        bands=list(BAND_NAMES),
        width_by_band=width_by_band,
        overall_width=overall,
        correlation=full_corr,
    )


# ---------------------------------------------------------------------------
# Source-role presence heuristics
# ---------------------------------------------------------------------------

def _source_role_presence(mono: np.ndarray, sr: int,
                          harmonic: np.ndarray | None = None,
                          percussive: np.ndarray | None = None) -> SourceRolePresence:
    """Estimate presence of each sound role using band energy + transient heuristics."""
    # Pre-compute band energies
    band_energy: dict[str, float] = {}
    for name, lo, hi in BAND_EDGES:
        filtered = _bandpass(mono, lo, hi, sr)
        band_energy[name] = _rms(filtered)

    # Pre-compute onset envelope for transient detection
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
    onset_mean = float(np.mean(onset_env))
    onset_max = float(np.max(onset_env)) if len(onset_env) > 0 else 1e-10
    transient_density = onset_mean / max(onset_max, 1e-10)

    # Normalize band energies for comparisons
    all_energies = list(band_energy.values())
    max_energy = max(all_energies) if all_energies else 1e-10
    if max_energy < 1e-10:
        max_energy = 1e-10
    norm = {k: v / max_energy for k, v in band_energy.items()}

    # Harmonic / percussive split (reuse pre-computed if available)
    if harmonic is None or percussive is None:
        harmonic, percussive = librosa.effects.hpss(mono)
    perc_energy = _rms(percussive)
    harm_energy = _rms(harmonic)
    total_energy = _rms(mono)
    perc_ratio = perc_energy / max(total_energy, 1e-10)
    harm_ratio = harm_energy / max(total_energy, 1e-10)

    # Onset strength in specific bands for transient detection
    perc_onset = librosa.onset.onset_strength(y=percussive, sr=sr)
    perc_onset_mean = float(np.mean(perc_onset))
    perc_onset_max = float(np.max(perc_onset)) if len(perc_onset) > 0 else 1e-10

    roles: dict[str, float] = {}

    # Percussive onset sharpness — better indicator than overall perc_ratio
    # for detecting individual hits within a dense mix
    perc_sharpness = perc_onset_mean / max(perc_onset_max, 1e-10)

    # kick: sub + bass energy with percussive transients
    kick_energy = (norm["sub"] + norm["bass"]) / 2.0
    kick_perc = max(perc_ratio, perc_sharpness)
    roles["kick"] = min(1.0, kick_energy * kick_perc * 2.5)

    # snare_clap: mid + upper_mid transients
    # Normalize percussive band energy against PERCUSSIVE energy (not total),
    # so dense loud mixes don't crush the score when snare is clearly there.
    perc_mid = _bandpass(percussive, 400, 2500, sr)
    perc_mid_energy = _rms(perc_mid) / max(perc_energy, 1e-10)
    snare_energy = (norm["mid"] + norm["upper_mid"]) / 2.0
    # Onset peak detection in snare range — periodic strong mid transients = snare
    # Reuse cached bandpass result instead of computing again
    snare_onset = librosa.onset.onset_strength(
        y=perc_mid, sr=sr,
    )
    snare_onset_peak = float(np.mean(np.sort(snare_onset)[-max(1, len(snare_onset) // 8):])) if len(snare_onset) > 0 else 0.0
    snare_onset_score = min(1.0, snare_onset_peak / max(perc_onset_max, 1e-10) * 1.5)
    snare_combined = (perc_mid_energy * 2.0 + snare_energy + snare_onset_score) / 3.0
    roles["snare_clap"] = min(1.0, snare_combined * max(perc_ratio, 0.35) * 2.5)

    # hats_tops: high-frequency transient energy (8k+)
    # Same fix: normalize against percussive energy, not total energy.
    perc_hf = _bandpass(percussive, 8000, min(sr // 2 - 1, 20000), sr)
    perc_hf_energy = _rms(perc_hf) / max(perc_energy, 1e-10)
    hf_energy = (norm["air"] + norm["ultra_high"] + norm["ceiling"]) / 3.0
    # Hi-hats produce frequent sharp HF transients — reuse cached bandpass result
    hf_onset = librosa.onset.onset_strength(
        y=perc_hf, sr=sr,
    )
    hf_onset_density = float(np.mean(hf_onset > np.percentile(hf_onset, 60))) if len(hf_onset) > 0 else 0.0
    hf_onset_score = min(1.0, hf_onset_density * 2.0)
    hats_combined = (perc_hf_energy * 2.0 + hf_energy + hf_onset_score) / 3.0
    roles["hats_tops"] = min(1.0, hats_combined * max(perc_ratio, 0.35) * 2.5)

    # bass: sustained low-frequency energy (60-250Hz range)
    bass_sustained = (norm["bass"] + norm["low_mid"]) / 2.0
    roles["bass"] = min(1.0, bass_sustained * harm_ratio * 2.0)

    # lead: prominent mid-range sustained energy
    lead_energy = (norm["mid"] + norm["upper_mid"] + norm["presence"]) / 3.0
    roles["lead"] = min(1.0, lead_energy * harm_ratio * 1.5)

    # chord_support: wide mid-range harmonic energy
    chord_energy = (norm["low_mid"] + norm["mid"] + norm["upper_mid"]) / 3.0
    roles["chord_support"] = min(1.0, chord_energy * harm_ratio * 1.5)

    # pad: sustained, spectrally spread energy with low transient density
    spread = float(np.std(list(norm.values())))
    smoothness = 1.0 - transient_density
    roles["pad"] = min(1.0, (1.0 - spread) * smoothness * harm_ratio * 2.0)

    # vocal_texture: formant-range energy (300Hz-3kHz) with modulation
    vocal_energy = (norm["low_mid"] + norm["mid"] + norm["upper_mid"]) / 3.0
    # Check for modulation via spectral flux in the vocal range
    vocal_filtered = _bandpass(mono, 300, 3000, sr)
    vocal_spec = np.abs(librosa.stft(vocal_filtered, n_fft=1024, hop_length=512))
    if vocal_spec.shape[1] > 1:
        flux = np.mean(np.diff(vocal_spec, axis=1) ** 2)
        flux_norm = min(1.0, float(flux) / max(float(np.mean(vocal_spec ** 2)), 1e-10))
    else:
        flux_norm = 0.0
    roles["vocal_texture"] = min(1.0, vocal_energy * flux_norm * 2.0)

    # fx_transitions: spectral flux spikes / sweeping energy
    full_spec = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
    if full_spec.shape[1] > 1:
        full_flux = np.mean(np.diff(full_spec, axis=1) ** 2, axis=0)
        flux_std = float(np.std(full_flux))
        flux_mean = float(np.mean(full_flux))
        fx_score = flux_std / max(flux_mean, 1e-10)
    else:
        fx_score = 0.0
    roles["fx_transitions"] = min(1.0, fx_score * 0.3)

    # ambience: sustained high-frequency diffuse energy
    hf_sustained = (norm["brilliance"] + norm["air"]) / 2.0
    roles["ambience"] = min(1.0, hf_sustained * smoothness * 1.5)

    return SourceRolePresence(roles=roles)


# ---------------------------------------------------------------------------
# Density map
# ---------------------------------------------------------------------------

def _density_map(mono: np.ndarray, sr: int, n_segments: int = 16) -> list[float]:
    """Density per time segment: spectral flux + onset density, normalized 0-1."""
    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=512)
    # Spectral flux
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
    if S.shape[1] > 1:
        flux = np.concatenate([[0], np.mean(np.diff(S, axis=1) ** 2, axis=0)])
    else:
        flux = np.zeros(S.shape[1])

    # Align lengths
    min_len = min(len(onset_env), len(flux))
    onset_env = onset_env[:min_len]
    flux = flux[:min_len]

    # Combine (equal weight)
    combined = onset_env / max(float(np.max(onset_env)), 1e-10) + \
               flux / max(float(np.max(flux)), 1e-10)

    # Split into segments
    segments = np.array_split(combined, n_segments)
    densities = [float(np.mean(s)) for s in segments]

    # Normalize 0-1
    mx = max(densities) if densities else 1.0
    if mx < 1e-12:
        return [0.0] * n_segments
    return [d / mx for d in densities]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_mix(filepath: str) -> MixProfile:
    """Analyze a full uploaded mix and return a MixProfile."""
    stereo, mono, sr = _load_audio(filepath)
    duration = len(mono) / sr

    # BPM
    bpm, bpm_conf = _detect_bpm(mono, sr)

    # Key
    key_label, key_conf, tonal_hz = _detect_key(mono, sr)

    # Loudness
    loud = _compute_loudness(stereo, mono, sr)

    # Compute HPSS once and share across functions
    harmonic, percussive = librosa.effects.hpss(mono)

    # Harmonic density
    h_density = _harmonic_density(mono, sr, harmonic=harmonic, percussive=percussive)

    # Section energy
    sec_energy = _section_energy(mono, n_sections=8)

    # Spectral occupancy
    spec_occ = _spectral_occupancy(mono, sr)

    # Stereo width
    sw = _stereo_width(stereo, sr)

    # Source-role presence
    roles = _source_role_presence(mono, sr, harmonic=harmonic, percussive=percussive)

    # Density map
    dmap = _density_map(mono, sr, n_segments=16)

    analysis = MixLevelAnalysis(
        bpm=bpm,
        bpm_confidence=bpm_conf,
        key=key_label,
        key_confidence=key_conf,
        tonal_center=tonal_hz,
        harmonic_density=h_density,
        duration=duration,
        loudness_lufs=loud["lufs"],
        loudness_range=loud["loudness_range"],
        peak=loud["peak"],
        dynamic_range=loud["dynamic_range"],
        section_energy=sec_energy,
    )

    return MixProfile(
        filepath=filepath,
        filename=filepath.split("/")[-1] if "/" in filepath else filepath,
        analysis=analysis,
        spectral_occupancy=spec_occ,
        stereo_width=sw,
        source_roles=roles,
        density_map=dmap,
    )
