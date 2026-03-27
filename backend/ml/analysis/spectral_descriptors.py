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
from ml.models.sample_profile import SpectralDescriptors


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
