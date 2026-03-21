"""
RESONATE — Transposition Engine.
Pitch-shifting and time-stretching with caching.
"""

import librosa
import soundfile as sf

from config import TRANSPOSED_DIR


def transpose_sample(filepath, semitones, tempo_ratio):
    """Pitch-shift and time-stretch a sample, return cached file path."""
    cache_name = f"{filepath.stem}_t{semitones}_r{int(tempo_ratio * 1000)}.wav"
    cache_path = TRANSPOSED_DIR / cache_name

    if cache_path.exists():
        return cache_path

    try:
        y, sr = librosa.load(str(filepath), sr=44100, mono=True)

        # Time-stretch first (preserves pitch)
        if abs(tempo_ratio - 1.0) > 0.01:
            y = librosa.effects.time_stretch(y, rate=tempo_ratio)

        # Then pitch-shift
        if abs(semitones) > 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

        sf.write(str(cache_path), y, sr)
        return cache_path
    except Exception as e:
        print(f"  Transpose error for {filepath.name}: {e}")
        return filepath  # fall back to original
