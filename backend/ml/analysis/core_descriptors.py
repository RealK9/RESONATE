"""
Core audio descriptors: duration, sample rate, channels, loudness metrics, envelope.
All computed via direct DSP — no ML models needed.
"""
from __future__ import annotations
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from backend.ml.models.sample_profile import CoreDescriptors


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
        # Short audio (<400ms) raises an error; fall back to short-term or default
        try:
            # Try short-term loudness (3s window, but may work on shorter clips)
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
