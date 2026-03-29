"""
RESONATE — Silence Trimmer.
Strips leading and trailing silence from audio samples with caching.
"""

from collections import OrderedDict

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

from config import TRANSPOSED_DIR

# Threshold in dB below peak — anything below this is considered silence
SILENCE_THRESHOLD_DB = -25
# Minimum pad to keep around content (seconds)
PAD_SEC = 0.005

# In-memory cache of already-trimmed paths — bounded to prevent memory leaks
_TRIM_CACHE_MAX = 200
_trim_cache: OrderedDict = OrderedDict()


def _cache_put(key: str, value: Path):
    """Insert into the trim cache with LRU eviction."""
    _trim_cache[key] = value
    if len(_trim_cache) > _TRIM_CACHE_MAX:
        _trim_cache.popitem(last=False)


def _load_audio(filepath: Path):
    """Load audio robustly — always mono float32 at native SR."""
    try:
        y, sr = librosa.load(str(filepath), sr=None, mono=True)
        return y, sr
    except Exception:
        return None, None


def trim_silence(filepath: Path) -> Path:
    """Strip leading and trailing silence from an audio file.

    Returns the path to a trimmed copy (cached in transposed_cache/),
    or the original if no significant silence was found.
    """
    key = str(filepath)
    if key in _trim_cache:
        _trim_cache.move_to_end(key)
        cached = _trim_cache[key]
        if cached.exists():
            return cached
        else:
            del _trim_cache[key]

    try:
        y, sr = _load_audio(filepath)
        if y is None or len(y) == 0:
            _cache_put(key, filepath)
            return filepath

        # Find non-silent intervals
        ref = np.max(np.abs(y))
        if ref < 1e-10:
            _cache_put(key, filepath)
            return filepath

        threshold = ref * (10 ** (SILENCE_THRESHOLD_DB / 20))
        above = np.abs(y) > threshold
        nonzero = np.nonzero(above)[0]

        if len(nonzero) == 0:
            _cache_put(key, filepath)
            return filepath

        pad_samples = int(PAD_SEC * sr)
        start = max(0, nonzero[0] - pad_samples)
        end = min(len(y), nonzero[-1] + pad_samples + 1)

        leading_sec = nonzero[0] / sr
        trailing_sec = (len(y) - nonzero[-1]) / sr
        total_removed = leading_sec + trailing_sec

        # Only trim if there's meaningful silence (> 100ms total)
        if total_removed < 0.05:
            _cache_put(key, filepath)
            return filepath

        y_trimmed = y[start:end]

        # Write trimmed version to cache dir
        cache_name = f"{filepath.stem}_trimmed.wav"
        cache_path = TRANSPOSED_DIR / cache_name

        # Handle name collisions from different directories
        if cache_path.exists():
            import hashlib
            h = hashlib.md5(str(filepath).encode()).hexdigest()[:8]
            cache_name = f"{filepath.stem}_{h}_trimmed.wav"
            cache_path = TRANSPOSED_DIR / cache_name

        sf.write(str(cache_path), y_trimmed, sr)
        _cache_put(key, cache_path)
        print(f"  ✂ Trimmed {filepath.name}: removed {leading_sec:.2f}s lead + {trailing_sec:.2f}s trail")
        return cache_path

    except Exception as e:
        print(f"  Trim error for {filepath.name}: {e}")
        _cache_put(key, filepath)
        return filepath


def get_trim_info(filepath: Path) -> dict:
    """Analyze silence in a file and return timing info without trimming."""
    try:
        y, sr = _load_audio(filepath)
        if y is None or len(y) == 0:
            return _empty_info()

        ref = np.max(np.abs(y))
        if ref < 1e-10:
            dur = len(y) / sr
            return {"leading_silence": dur, "trailing_silence": 0,
                    "content_start": dur, "content_end": dur,
                    "total_duration": dur, "needs_trim": False}

        threshold = ref * (10 ** (SILENCE_THRESHOLD_DB / 20))
        above = np.abs(y) > threshold
        nonzero = np.nonzero(above)[0]

        if len(nonzero) == 0:
            dur = len(y) / sr
            return {"leading_silence": dur, "trailing_silence": 0,
                    "content_start": dur, "content_end": dur,
                    "total_duration": dur, "needs_trim": False}

        total_dur = len(y) / sr
        content_start = nonzero[0] / sr
        content_end = nonzero[-1] / sr
        leading = content_start
        trailing = total_dur - content_end

        return {
            "leading_silence": round(leading, 3),
            "trailing_silence": round(trailing, 3),
            "content_start": round(content_start, 3),
            "content_end": round(content_end, 3),
            "total_duration": round(total_dur, 3),
            "needs_trim": (leading + trailing) > 0.1,
        }
    except Exception:
        return _empty_info()


def _empty_info():
    return {"leading_silence": 0, "trailing_silence": 0,
            "content_start": 0, "content_end": 0,
            "total_duration": 0, "needs_trim": False}
