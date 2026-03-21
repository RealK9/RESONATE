"""
RESONATE — Track Analysis Engine.
Deep sonic fingerprint extraction for uploaded tracks.
"""

import numpy as np
import essentia.standard as es

from utils import normalize_key


def analyze_track(filepath):
    """Deep sonic fingerprint of uploaded track."""
    print("\n  [1/3] Deep sonic fingerprint...")
    audio = es.MonoLoader(filename=str(filepath), sampleRate=44100)()
    duration = float(len(audio)) / 44100

    # ── Key detection (3-profile voting) ──
    profiles = ["edma", "krumhansl", "temperley"]
    votes = {}
    for prof in profiles:
        try:
            k, scale, strength = es.KeyExtractor(profileType=prof)(audio)
            key_str = k if scale == "major" else f"{k}m"
            key_str = normalize_key(key_str)
            votes[key_str] = votes.get(key_str, 0) + strength
        except Exception:
            pass

    if votes:
        best_key = max(votes, key=votes.get)
        confidence = round(votes[best_key] / sum(votes.values()) * 100, 1)
    else:
        best_key = "C"
        confidence = 0.0
    print(f"  Key: {best_key} ({confidence}%)")

    # ── BPM ──
    try:
        bpm, *_ = es.RhythmExtractor2013(method="multifeature")(audio)
        bpm = float(bpm)
        while bpm > 180: bpm /= 2
        while bpm < 60: bpm *= 2
        bpm = round(bpm, 1)
    except Exception:
        bpm = 120.0
    print(f"  BPM: {bpm}")

    # ── Loudness ──
    try:
        loudness = float(es.Loudness()(audio))
    except Exception:
        loudness = 0.0
    print(f"  Loudness: {loudness:.2f}dB")

    # ── RMS and Spectral Centroid (for mood/energy detection) ──
    rms = 0.04
    spectral_centroid = 2000.0
    try:
        rms = float(es.RMS()(audio))
        chunk = audio[:44100 * min(int(duration), 10)]
        if len(chunk) > 4096:
            spec = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / 44100)
            total = np.sum(spec) + 1e-10
            spectral_centroid = float(np.sum(freqs * spec) / total)
    except Exception:
        pass

    # ── 7-band frequency analysis ──
    freq_bands = {}
    try:
        chunk = audio[:44100 * min(int(duration), 10)]
        if len(chunk) > 4096:
            spec = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / 44100)
            total_e = np.sum(spec ** 2) + 1e-10
            bands_def = {
                "sub_bass_20_80": (20, 80), "bass_80_250": (80, 250),
                "low_mid_250_500": (250, 500), "mid_500_2k": (500, 2000),
                "upper_mid_2k_6k": (2000, 6000), "presence_6k_12k": (6000, 12000),
                "air_12k_20k": (12000, 20000),
            }
            for bname, (lo, hi) in bands_def.items():
                mask = (freqs >= lo) & (freqs < hi)
                freq_bands[bname] = round(float(np.sum(spec[mask] ** 2) / total_e), 4)
    except Exception:
        pass

    # ── MFCCs ──
    mfcc_profile = [0] * 13
    if len(audio) > 8192:
        try:
            w = es.Windowing(type='hann')
            spec_algo = es.Spectrum()
            mfcc_algo = es.MFCC(numberCoefficients=13)
            frames = []
            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
                sp = spec_algo(w(frame))
                _, c = mfcc_algo(sp)
                frames.append(c)
            if frames:
                mfcc_profile = np.mean(frames, axis=0).tolist()
        except Exception:
            pass

    # ── Spectral contrast ──
    spectral_contrast = []
    if len(audio) > 8192:
        try:
            sc_algo = es.SpectralContrast(frameSize=2048, hopSize=1024)
            sc_frames = []
            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
                spec_fr = es.Spectrum()(es.Windowing(type='hann')(frame))
                sc, _ = sc_algo(spec_fr)
                sc_frames.append(sc)
            if sc_frames:
                spectral_contrast = np.mean(sc_frames, axis=0).tolist()
        except Exception:
            pass

    # ── Instrument detection ──
    detected_instruments = []
    if freq_bands:
        sb = freq_bands.get("sub_bass_20_80", 0)
        bass = freq_bands.get("bass_80_250", 0)
        if sb > 0.15:
            detected_instruments.append("sub_bass_808")
        if bass > 0.15:
            detected_instruments.append("bass")
        try:
            rms = float(es.RMS()(audio))
            if rms > 0.05 and sb > 0.08:
                detected_instruments.append("kick")
        except Exception:
            pass
        mid = freq_bands.get("mid_500_2k", 0)
        upper = freq_bands.get("upper_mid_2k_6k", 0)
        if mid > 0.02 and upper > 0.01:
            detected_instruments.append("snare_clap")

    print(f"  Instruments detected: {detected_instruments}")

    # ── Frequency gaps ──
    frequency_gaps = []
    if freq_bands:
        thresholds = {
            "low_mid_warmth": ("low_mid_250_500", 0.03),
            "midrange_melody": ("mid_500_2k", 0.03),
            "upper_mid_presence": ("upper_mid_2k_6k", 0.02),
            "high_end_sparkle": ("presence_6k_12k", 0.01),
            "air": ("air_12k_20k", 0.005),
        }
        for gap_name, (band, thresh) in thresholds.items():
            if freq_bands.get(band, 0) < thresh:
                frequency_gaps.append(gap_name)
    print(f"  Frequency gaps: {frequency_gaps}")

    # ── Heuristic genre detection ──
    detected_genre = "default"
    sb_energy = freq_bands.get("sub_bass_20_80", 0) if freq_bands else 0
    bass_energy = freq_bands.get("bass_80_250", 0) if freq_bands else 0
    mid_energy = freq_bands.get("mid_500_2k", 0) if freq_bands else 0
    hi_energy = freq_bands.get("presence_6k_12k", 0) if freq_bands else 0
    total_low = sb_energy + bass_energy

    if sb_energy > 0.15 and total_low > 0.30:
        if 135 <= bpm <= 148:
            detected_genre = "drill"
        else:
            detected_genre = "trap/hip-hop"
    elif sb_energy > 0.10 and 70 <= bpm <= 170:
        detected_genre = "trap/hip-hop"
    elif 80 <= bpm <= 115 and mid_energy > 0.10 and sb_energy < 0.10:
        detected_genre = "r&b"
    elif mid_energy > 0.15 and hi_energy > 0.08 and sb_energy < 0.08:
        detected_genre = "pop"
    elif 120 <= bpm <= 150 and sb_energy > 0.08 and mid_energy > 0.08:
        detected_genre = "edm/electronic"
    elif 70 <= bpm <= 100 and sb_energy < 0.08:
        detected_genre = "lo-fi/chill"
    print(f"  Heuristic genre: {detected_genre}")

    return {
        "key": best_key,
        "key_confidence": confidence,
        "bpm": bpm,
        "duration": round(duration, 2),
        "loudness": round(loudness, 2),
        "rms": round(rms, 4),
        "spectral_centroid": round(spectral_centroid, 2),
        "frequency_bands": freq_bands,
        "mfcc_profile": [round(x, 2) for x in mfcc_profile],
        "spectral_contrast": [round(x, 2) for x in spectral_contrast],
        "detected_instruments": detected_instruments,
        "frequency_gaps": frequency_gaps,
        "detected_genre": detected_genre,
    }
