"""
Transient descriptors: onset detection, transient positions, onset strength,
attack sharpness, transient density.
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf
from backend.ml.models.sample_profile import TransientDescriptors


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
