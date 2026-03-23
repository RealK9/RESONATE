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
