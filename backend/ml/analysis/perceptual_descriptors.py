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
from ml.models.sample_profile import PerceptualDescriptors


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
