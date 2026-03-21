"""
RESONATE — Sample Analysis Engine.
Analyzes individual sample files for indexing.
"""

import numpy as np
import essentia.standard as es

from utils import parse_key, parse_bpm, is_percussion, classify_type, normalize_key


def classify_mood(rms, spectral_centroid, key_string, frequency_bands):
    """Classify mood and energy level from audio features."""
    sub_bass = frequency_bands.get("sub_bass_20_80", 0)
    low_mid = frequency_bands.get("low_mid_250_500", 0)
    presence = frequency_bands.get("presence_6k_12k", 0)
    air = frequency_bands.get("air_12k_20k", 0)
    is_minor = key_string.endswith("m") and key_string != "N/A"

    # Mood classification
    if spectral_centroid < 1500 and (is_minor or sub_bass > 0.12):
        mood = "dark"
    elif 1500 <= spectral_centroid < 2500 and low_mid > 0.08:
        mood = "warm"
    elif spectral_centroid >= 3000 or (presence + air > 0.15):
        mood = "bright"
    elif rms > 0.08 and spectral_centroid > 2000:
        mood = "aggressive"
    elif rms < 0.03 and spectral_centroid < 2500:
        mood = "chill"
    else:
        mood = "neutral"

    # Energy classification
    if rms > 0.06:
        energy = "high"
    elif rms >= 0.02:
        energy = "medium"
    else:
        energy = "low"

    return {"mood": mood, "energy": energy}


def analyze_sample(filepath):
    """Analyze a single sample file."""
    fname = filepath.name
    pk = parse_key(fname)
    pb = parse_bpm(fname)
    perc = is_percussion(filepath)
    sample_type = classify_type(filepath)

    # Try loading audio
    try:
        audio = es.MonoLoader(filename=str(filepath), sampleRate=44100)()
        duration = float(len(audio)) / 44100
    except Exception:
        key_string = "N/A" if perc else (pk or "N/A")
        bpm_val = pb or 0
        print(f"  {fname}: key={key_string} (tiny-file), bpm={bpm_val}, type={sample_type}")
        return {
            "duration": 0, "bpm": bpm_val, "key": normalize_key(key_string),
            "rms": 0, "spectral_centroid": 0,
            "mfcc_profile": [0] * 13, "frequency_bands": {},
            "sample_type": sample_type,
        }

    # ── Key ──
    if perc:
        key_string = "N/A"; key_src = "percussion"
    elif pk:
        key_string = pk; key_src = "filename"
    elif duration > 1.5 and len(audio) > 8192:
        try:
            k, sc, _ = es.KeyExtractor(profileType='edma')(audio)
            key_string = k if sc == "major" else f"{k}m"
            key_string = normalize_key(key_string)
            key_src = "essentia"
        except Exception:
            key_string = "N/A"; key_src = "failed"
    else:
        key_string = "N/A"; key_src = "too-short"

    # ── BPM ──
    if pb:
        bpm_val = pb; bpm_src = "filename"
    elif duration > 3 and len(audio) > 44100:
        try:
            b, *_ = es.RhythmExtractor2013(method="multifeature")(audio)
            b = float(b)
            while b > 180: b /= 2
            while b < 60: b *= 2
            bpm_val = round(b, 1); bpm_src = "essentia"
        except Exception:
            bpm_val = 0; bpm_src = "failed"
    else:
        bpm_val = 0; bpm_src = "too-short"

    # ── Features (only if audio is long enough) ──
    rms = 0; sc_val = 0; mfcc_profile = [0] * 13; freq_bands = {}

    if len(audio) > 8192:
        try:
            rms = float(es.RMS()(audio))
            sc_val = float(es.SpectralCentroidTime()(audio))
        except Exception:
            pass

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

        try:
            chunk = audio[:44100 * min(int(duration), 10)]
            if len(chunk) > 4096:
                spec = np.abs(np.fft.rfft(chunk))
                freqs = np.fft.rfftfreq(len(chunk), 1.0 / 44100)
                total_e = np.sum(spec ** 2) + 1e-10
                for bname, (lo, hi) in [
                    ("sub_bass_20_80", (20, 80)), ("bass_80_250", (80, 250)),
                    ("low_mid_250_500", (250, 500)), ("mid_500_2k", (500, 2000)),
                    ("upper_mid_2k_6k", (2000, 6000)), ("presence_6k_12k", (6000, 12000)),
                    ("air_12k_20k", (12000, 20000)),
                ]:
                    mask = (freqs >= lo) & (freqs < hi)
                    freq_bands[bname] = round(float(np.sum(spec[mask] ** 2) / total_e), 4)
        except Exception:
            pass

    # ── Mood / Energy ──
    mood_energy = classify_mood(rms, sc_val, key_string, freq_bands)

    print(f"  {fname}: key={key_string} ({key_src}), bpm={bpm_val} ({bpm_src}), type={sample_type}")

    return {
        "duration": round(duration, 2), "bpm": bpm_val,
        "key": normalize_key(key_string),
        "rms": round(rms, 4), "spectral_centroid": round(sc_val, 4),
        "mfcc_profile": [round(x, 2) for x in mfcc_profile],
        "frequency_bands": freq_bands,
        "sample_type": sample_type,
        "mood": mood_energy["mood"],
        "energy": mood_energy["energy"],
    }
