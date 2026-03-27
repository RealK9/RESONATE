#!/usr/bin/env python3
"""
RESONATE — Batch Silence Trimmer.
Scans ALL samples and trims leading/trailing silence IN PLACE.
Run once: python batch_trim.py
"""

import sys
import time
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Config
SILENCE_THRESHOLD_DB = -25
PAD_SEC = 0.005
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}
SAMPLE_DIR = Path(__file__).parent / "samples"
SPLICE_DIRS = [
    Path.home() / "Splice" / "sounds",
    Path.home() / "Splice" / "INSTRUMENT",
]


def trim_file(filepath: str) -> dict:
    """Trim a single file in place. Returns stats."""
    fp = Path(filepath)
    try:
        y, sr = librosa.load(str(fp), sr=None, mono=True)
        if len(y) == 0:
            return {"path": str(fp), "status": "empty"}

        ref = np.max(np.abs(y))
        if ref < 1e-10:
            return {"path": str(fp), "status": "silent"}

        threshold = ref * (10 ** (SILENCE_THRESHOLD_DB / 20))
        above = np.nonzero(np.abs(y) > threshold)[0]

        if len(above) == 0:
            return {"path": str(fp), "status": "silent"}

        pad = int(PAD_SEC * sr)
        start = max(0, above[0] - pad)
        end = min(len(y), above[-1] + pad + 1)

        leading = above[0] / sr
        trailing = (len(y) - above[-1]) / sr
        total_removed = leading + trailing

        if total_removed < 0.05:
            return {"path": str(fp), "status": "ok", "removed": 0}

        y_trimmed = y[start:end]

        # Overwrite in place — always write as WAV for consistency
        # If original was .aif/.mp3/etc, write .wav next to it and delete original
        if fp.suffix.lower() in (".wav",):
            sf.write(str(fp), y_trimmed, sr)
        else:
            wav_path = fp.with_suffix(".wav")
            sf.write(str(wav_path), y_trimmed, sr)
            try:
                fp.unlink()  # remove original .aif/.mp3
            except Exception:
                pass
            fp = wav_path

        return {
            "path": str(fp),
            "status": "trimmed",
            "removed": round(total_removed, 2),
            "leading": round(leading, 2),
            "trailing": round(trailing, 2),
            "old_dur": round(len(y) / sr, 2),
            "new_dur": round(len(y_trimmed) / sr, 2),
        }
    except Exception as e:
        return {"path": str(fp), "status": "error", "error": str(e)}


def main():
    # Collect all sample files
    files = []

    if SAMPLE_DIR.exists():
        files.extend(
            f for f in SAMPLE_DIR.rglob("*")
            if f.suffix.lower() in AUDIO_EXT and f.is_file()
        )

    for sd in SPLICE_DIRS:
        if sd.exists():
            files.extend(
                f for f in sd.rglob("*")
                if f.suffix.lower() in AUDIO_EXT and f.is_file()
            )

    print(f"\n✂  RESONATE Batch Trimmer")
    print(f"   Scanning {len(files)} audio files...\n")

    if not files:
        print("   No audio files found.")
        return

    start = time.time()
    trimmed = 0
    errors = 0
    skipped = 0

    # Use multiprocessing for speed
    workers = min(8, len(files))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(trim_file, str(f)): f for f in files}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result["status"] == "trimmed":
                trimmed += 1
                print(f"   ✂ {Path(result['path']).name}: -{result['removed']}s ({result['old_dur']}s → {result['new_dur']}s)")
            elif result["status"] == "error":
                errors += 1
            else:
                skipped += 1

            # Progress every 500 files
            done = i + 1
            if done % 500 == 0 or done == len(files):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(files) - done) / rate if rate > 0 else 0
                print(f"   [{done}/{len(files)}] {rate:.0f} files/sec, ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\n   Done in {elapsed:.1f}s")
    print(f"   ✂ Trimmed: {trimmed}")
    print(f"   ✓ Already clean: {skipped}")
    if errors:
        print(f"   ✗ Errors: {errors}")
    print()


if __name__ == "__main__":
    main()
