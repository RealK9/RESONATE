"""
RESONATE — Waveform peak extraction for visualization.
"""

import numpy as np


def extract_waveform_peaks(filepath, bars=100):
    """Extract waveform peaks from an audio file for visualization."""
    import essentia.standard as es
    audio = es.MonoLoader(filename=str(filepath), sampleRate=22050)()
    seg_size = max(1, len(audio) // bars)
    peaks = []
    for i in range(bars):
        start = i * seg_size
        end = min(start + seg_size, len(audio))
        if start >= len(audio):
            peaks.append(0)
        else:
            segment = np.abs(audio[start:end])
            peaks.append(float(np.max(segment)) if len(segment) > 0 else 0)

    max_peak = max(peaks) if peaks else 1
    if max_peak > 0:
        peaks = [round(p / max_peak, 3) for p in peaks]

    return {"peaks": peaks, "bars": bars, "duration": len(audio) / 22050}
