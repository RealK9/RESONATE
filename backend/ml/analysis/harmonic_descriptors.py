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
from ml.models.sample_profile import HarmonicDescriptors


def extract_harmonic_descriptors(filepath: str) -> HarmonicDescriptors:
    """Extract all harmonic/pitch descriptors from an audio file."""
    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    # Resample to 16kHz for pitch estimation
    mono_16k = librosa.resample(mono, orig_sr=sr, target_sr=16000)

    # F0 estimation
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
    """Compute harmonic-to-noise ratio in dB using harmonic/percussive decomposition."""
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
