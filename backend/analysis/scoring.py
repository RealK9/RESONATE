"""
RESONATE — Sample Scoring Engine.
Deterministic scoring based on actual audio measurements + genre science.
"""

import numpy as np
from pathlib import Path

from utils import normalize_key, classify_type, NOTE_NAMES
from analysis.genre_profiles import get_genre_profile
import state


def semitones_to_transpose(from_key, to_key):
    """Calculate semitones needed to shift from_key to to_key."""
    if not from_key or from_key in ("—", "N/A") or not to_key:
        return 0
    fk = normalize_key(from_key)
    tk = normalize_key(to_key)
    fn = fk.replace("m", "")
    tn = tk.replace("m", "")
    if fn not in NOTE_NAMES or tn not in NOTE_NAMES:
        return 0
    fi = NOTE_NAMES.index(fn)
    ti = NOTE_NAMES.index(tn)
    diff = (ti - fi) % 12
    if diff > 6:
        diff -= 12
    return diff


def transpose_quality(semitones):
    """Score 0-100: how clean will the pitch shift sound?"""
    s = abs(semitones)
    if s == 0: return 100
    if s <= 2: return 97
    if s <= 3: return 93
    if s <= 4: return 88
    if s <= 5: return 80
    if s <= 6: return 65
    return 50


def timestretch_quality(from_bpm, to_bpm):
    """Score 0-100: how clean will the time stretch sound?"""
    if not from_bpm or from_bpm == 0 or not to_bpm:
        return 75
    ratio = to_bpm / from_bpm
    candidates = [ratio, ratio * 2, ratio / 2]
    best = min(candidates, key=lambda r: abs(r - 1.0))
    dev = abs(best - 1.0)
    if dev < 0.02: return 100
    if dev < 0.05: return 97
    if dev < 0.10: return 92
    if dev < 0.15: return 85
    if dev < 0.20: return 75
    if dev < 0.30: return 55
    if dev < 0.40: return 35
    return 15


def bpm_proximity_score(sample_bpm, target_bpm):
    """Score 0-100: how close is the sample's native BPM to the target?
    Rewards exact matches and harmonic multiples (half/double time)."""
    if not sample_bpm or sample_bpm == 0 or not target_bpm or target_bpm == 0:
        return 50  # Neutral when unknown
    candidates = [
        abs(sample_bpm - target_bpm),
        abs(sample_bpm * 2 - target_bpm),
        abs(sample_bpm / 2 - target_bpm),
    ]
    best_diff = min(candidates)
    if best_diff < 1:
        return 100
    if best_diff < 3:
        return 95
    if best_diff < 6:
        return 88
    if best_diff < 10:
        return 78
    if best_diff < 15:
        return 65
    if best_diff < 25:
        return 45
    if best_diff < 40:
        return 30
    return 15


def timbre_compat(track_mfcc, sample_mfcc):
    """Compare timbral similarity via MFCC cosine distance."""
    if not track_mfcc or not sample_mfcc:
        return 50
    a = np.array(track_mfcc)
    b = np.array(sample_mfcc)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 50
    cos = float(np.dot(a, b) / (na * nb))
    if cos > 0.85:
        return 75
    elif cos > 0.6:
        return 95
    elif cos > 0.4:
        return 80
    elif cos > 0.2:
        return 55
    elif cos > 0.0:
        return 30
    else:
        return 15


def sonic_genre_compat(track, sample, sample_filepath, genre_str):
    """
    Score how well a sample's SONIC CHARACTER fits the track's genre.
    Uses measured audio features + filename genre clues.
    """
    score = 50
    genre_lower = (genre_str or "").lower()

    track_centroid = track.get("spectral_centroid", 1500)
    sample_centroid = sample.get("spectral_centroid", 1500)
    track_bands = track.get("frequency_bands", {})
    sample_bands = sample.get("frequency_bands", {})

    track_is_hiphop = any(g in genre_lower for g in ["trap", "hip", "drill", "rap", "r&b"])
    track_is_edm = any(g in genre_lower for g in ["edm", "house", "techno", "electronic", "dance"])
    track_is_lofi = any(g in genre_lower for g in ["lo-fi", "lofi", "chill", "ambient"])

    # ── 1. Filename/folder genre detection ──
    sample_genre = "neutral"
    if sample_filepath:
        all_text = str(sample_filepath).lower() + " " + Path(sample_filepath).stem.lower()

        HIPHOP_WORDS = {"trap", "hiphop", "hip-hop", "hip_hop", "drill", "808", "southside",
                        "metro", "dark", "hard", "grimey", "dirty", "hood", "street", "murda",
                        "plugg", "rage", "phonk", "atlanta", "zaytoven", "pierre", "carti",
                        "travis", "bouncy", "migos", "thug", "gunna", "lil", "juice"}
        EDM_WORDS = {"house", "techno", "trance", "edm", "dance", "electro", "rave",
                     "progressive", "dubstep", "dnb", "drum_and_bass", "jungle", "garage",
                     "bigroom", "festival", "tropical", "future_house", "deep_house",
                     "melodic_techno", "club", "pluck_house", "vocal_house", "breakbeat"}
        ROCK_WORDS = {"rock", "metal", "punk", "grunge", "indie_rock", "alternative",
                      "blues_rock", "classic_rock", "heavy_metal", "power_chord"}

        if any(w in all_text for w in HIPHOP_WORDS):
            sample_genre = "hiphop"
        elif any(w in all_text for w in EDM_WORDS):
            sample_genre = "edm"
        elif any(w in all_text for w in ROCK_WORDS):
            sample_genre = "rock"

    # ── 2. AUDIO-BASED genre inference for "neutral" samples (spectral only, no BPM) ──
    if sample_genre == "neutral":
        s_air = sample_bands.get("air_12k_20k", 0)
        s_presence = sample_bands.get("presence_6k_12k", 0)
        s_sub = sample_bands.get("sub_bass_20_80", 0)
        s_bass = sample_bands.get("bass_80_250", 0)
        s_brightness = s_air + s_presence
        s_darkness = s_sub + s_bass

        if sample_centroid > 3800 and s_brightness > 0.15:
            sample_genre = "edm_inferred"
        elif sample_centroid > 3200 and s_brightness > s_darkness * 1.5:
            sample_genre = "edm_inferred"
        elif sample_centroid < 1800 and s_darkness > 0.12:
            sample_genre = "hiphop_inferred"
        elif sample_centroid < 2200 and s_darkness > s_brightness * 1.3:
            sample_genre = "hiphop_inferred"

    # ── 3. Apply genre match/mismatch ──
    if track_is_hiphop:
        if sample_genre in ("hiphop", "hiphop_inferred"):
            score += 30
        elif sample_genre == "edm":
            score -= 40
        elif sample_genre == "edm_inferred":
            score -= 32
        elif sample_genre == "rock":
            score -= 38
    elif track_is_edm:
        if sample_genre in ("edm", "edm_inferred"): score += 25
        elif sample_genre in ("hiphop", "hiphop_inferred"): score -= 20
        elif sample_genre == "rock": score -= 25
    elif track_is_lofi:
        if sample_genre in ("hiphop", "hiphop_inferred"): score += 10
        elif sample_genre in ("edm", "edm_inferred"): score -= 25
        elif sample_genre == "rock": score -= 25

    # ── 4. Spectral centroid vs genre expectations ──
    if track_is_hiphop:
        if sample_centroid < 1200:
            score += 20
        elif sample_centroid < 1800:
            score += 14
        elif sample_centroid < 2500:
            score += 5
        elif sample_centroid < 3500:
            score -= 10
        elif sample_centroid < 4500:
            score -= 20
        else:
            score -= 28
    elif track_is_edm:
        if sample_centroid > 3500: score += 15
        elif sample_centroid > 2500: score += 8
        elif sample_centroid < 1500: score -= 12
    elif track_is_lofi:
        if sample_centroid < 1800: score += 15
        elif sample_centroid < 2500: score += 5
        else: score -= 10

    # ── 5. Frequency band profile ──
    if track_bands and sample_bands:
        s_low = sample_bands.get("bass_80_250", 0) + sample_bands.get("low_mid_250_500", 0)
        s_high = sample_bands.get("presence_6k_12k", 0) + sample_bands.get("air_12k_20k", 0)

        if track_is_hiphop:
            if s_low > s_high * 1.5:
                score += 10
            elif s_high > s_low * 2.0:
                score -= 18
        elif track_is_edm:
            if s_high > s_low: score += 8

    return round(max(0, min(100, score)), 1)


def frequency_complement(track_bands, sample_bands):
    """Score how well sample fills frequency gaps in track."""
    if not track_bands or not sample_bands:
        return 50
    score = 50
    for band, track_level in track_bands.items():
        sample_level = sample_bands.get(band, 0)
        if track_level < 0.03 and sample_level > 0.05:
            score += 12
        elif track_level > 0.20 and sample_level > 0.20:
            score -= 8
    return max(0, min(100, score))


def math_match(track, sample, sample_filepath=None, ai_template=None):
    """
    Deterministic scoring based on actual audio measurements + genre science.
    Key/BPM are NOT scoring factors — the bridge handles transposition.

    Scoring components:
    1. Type Priority (12%) — genre-calibrated need for this sample type
    2. Sonic Genre Compatibility (30%) — does sample SOUND like it belongs in this genre?
    3. Frequency Deficit Filling (18%) — does sample fill measured gaps vs ideal?
    4. Spectral Placement (14%) — is sample's energy where the track needs it?
    5. Timbre Match (18%) — timbral compatibility via MFCC
    6. Musical Content Value (8%) — longer harmonic content > short transients
    """
    stype = classify_type(Path(sample_filepath)) if sample_filepath else "unknown"
    duration = sample.get("duration", 0)
    track_bands = track.get("frequency_bands", {})
    sample_bands = sample.get("frequency_bands", {})

    genre_str = ""
    if ai_template:
        genre_str = ai_template.get("genre", "")
    if not genre_str and track:
        genre_str = track.get("detected_genre", "default")
    genre_ref = get_genre_profile(genre_str)

    # ── 1. TYPE PRIORITY (18%) ──
    ref_type_needs = genre_ref["type_needs"]
    type_score = ref_type_needs.get(stype, 25)

    if ai_template:
        ai_type = ai_template.get("type_priority_scores", {})
        if stype in ai_type:
            ai_val = float(ai_type[stype])
            type_score = ref_type_needs.get(stype, 25) * 0.6 + ai_val * 0.4

    # ── 2. FREQUENCY DEFICIT FILLING (22%) ──
    deficit_score = 50
    ideal_balance = genre_ref["freq_balance"]
    if ai_template and ai_template.get("ideal_frequency_balance"):
        ai_ideal = ai_template["ideal_frequency_balance"]
        ideal_balance = {}
        for band in genre_ref["freq_balance"]:
            ref_val = genre_ref["freq_balance"].get(band, 0.1)
            ai_val = ai_ideal.get(band, ref_val)
            ideal_balance[band] = ref_val * 0.7 + ai_val * 0.3

    if track_bands and sample_bands:
        deficit_fill = 0
        total_deficit_weight = 0

        for band, ideal_val in ideal_balance.items():
            track_val = track_bands.get(band, 0)
            sample_val = sample_bands.get(band, 0)

            deficit = max(0, ideal_val - track_val)
            contribution = min(sample_val, deficit) if deficit > 0 else 0

            deficit_weight = deficit * 10 + 0.1
            total_deficit_weight += deficit_weight

            if deficit > 0.03 and sample_val > 0.03:
                fill_pct = min(contribution / max(deficit, 0.01), 1.5)
                deficit_fill += fill_pct * 100 * deficit_weight
            elif deficit < 0.01 and sample_val > 0.12:
                deficit_fill += 15 * deficit_weight
            elif sample_val > 0.01:
                deficit_fill += 30 * deficit_weight
            else:
                deficit_fill += 20 * deficit_weight

        deficit_score = deficit_fill / max(total_deficit_weight, 1)
        deficit_score = min(100, max(0, deficit_score))

    # ── 3. SPECTRAL PLACEMENT (15%) ──
    spectral_score = 50
    sample_centroid = sample.get("spectral_centroid", 0)
    if sample_centroid > 0 and track_bands:
        max_deficit_band = None
        max_deficit = 0
        band_centers = {
            "sub_bass_20_80": 50, "bass_80_250": 165,
            "low_mid_250_500": 375, "mid_500_2k": 1250,
            "upper_mid_2k_6k": 4000, "presence_6k_12k": 9000,
            "air_12k_20k": 16000,
        }
        for band, center in band_centers.items():
            deficit = max(0, ideal_balance.get(band, 0.1) - track_bands.get(band, 0))
            if deficit > max_deficit:
                max_deficit = deficit
                max_deficit_band = band

        if max_deficit_band:
            target_center = band_centers[max_deficit_band]
            if target_center > 0:
                ratio = sample_centroid / target_center
                if 0.5 <= ratio <= 2.0:
                    spectral_score = 90
                elif 0.3 <= ratio <= 3.0:
                    spectral_score = 65
                else:
                    spectral_score = 30

    # ── 4. MUSICAL CONTENT VALUE (13%) ──
    content_score = 35
    if duration > 5 and stype in ("melody", "vocals", "pad", "strings"):
        content_score = 98
    elif duration > 2 and stype in ("melody", "vocals", "pad", "strings"):
        content_score = 88
    elif duration > 1 and stype in ("melody", "vocals"):
        content_score = 72
    elif duration > 2 and stype in ("hihat", "percussion"):
        content_score = 62
    elif duration > 0.5 and stype in ("hihat", "percussion", "fx"):
        content_score = 45
    elif duration < 0.3 and stype in ("snare", "kick", "hihat"):
        content_score = 18
    elif duration < 0.15:
        content_score = 10

    # ── 5. TIMBRE MATCH (12%) ──
    timbre_score = timbre_compat(track.get("mfcc_profile"), sample.get("mfcc_profile"))

    # ── 5b. SONIC GENRE COMPATIBILITY (22%) ──
    genre_compat_score = sonic_genre_compat(track, sample, sample_filepath, genre_str)

    # ── COMBINE (no key/BPM — bridge handles transposition) ──
    final = (
        type_score * 0.12 +
        genre_compat_score * 0.30 +
        deficit_score * 0.18 +
        spectral_score * 0.14 +
        timbre_score * 0.18 +
        content_score * 0.08
    )

    return round(max(0, min(100, final)), 1)
