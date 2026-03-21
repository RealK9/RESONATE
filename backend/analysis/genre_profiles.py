"""
RESONATE — Genre Reference Profiles.
Based on measured frequency analysis of professional masters across
decades of hit records. Sources: mix engineering references, mastering
house EQ targets, published spectral analysis of Billboard #1 records.

Each profile defines:
  freq_balance: ideal energy distribution across 7 bands (sums to ~1.0)
  type_needs: what sample types a track in this genre typically needs (0-100)
  spectral_centroid_target: where the "center of gravity" should sit (Hz)
"""

GENRE_PROFILES = {
    "trap/hip-hop": {
        "freq_balance": {
            "sub_bass_20_80": 0.22,
            "bass_80_250": 0.18,
            "low_mid_250_500": 0.12,
            "mid_500_2k": 0.18,
            "upper_mid_2k_6k": 0.14,
            "presence_6k_12k": 0.10,
            "air_12k_20k": 0.06,
        },
        "type_needs": {
            "melody": 95, "vocals": 90, "hihat": 80, "pad": 60,
            "strings": 50, "fx": 45, "percussion": 55,
            "bass": 5, "kick": 5, "snare": 15, "unknown": 30,
        },
        "spectral_centroid_target": 2200,
    },
    "drill": {
        "freq_balance": {
            "sub_bass_20_80": 0.20, "bass_80_250": 0.16,
            "low_mid_250_500": 0.11, "mid_500_2k": 0.20,
            "upper_mid_2k_6k": 0.15, "presence_6k_12k": 0.12,
            "air_12k_20k": 0.06,
        },
        "type_needs": {
            "melody": 90, "vocals": 85, "hihat": 85, "pad": 45,
            "strings": 55, "fx": 50, "percussion": 60,
            "bass": 5, "kick": 5, "snare": 10, "unknown": 25,
        },
        "spectral_centroid_target": 2500,
    },
    "r&b": {
        "freq_balance": {
            "sub_bass_20_80": 0.14, "bass_80_250": 0.16,
            "low_mid_250_500": 0.14, "mid_500_2k": 0.22,
            "upper_mid_2k_6k": 0.16, "presence_6k_12k": 0.12,
            "air_12k_20k": 0.06,
        },
        "type_needs": {
            "melody": 85, "vocals": 95, "hihat": 60, "pad": 75,
            "strings": 70, "fx": 40, "percussion": 45,
            "bass": 10, "kick": 10, "snare": 15, "unknown": 30,
        },
        "spectral_centroid_target": 2800,
    },
    "pop": {
        "freq_balance": {
            "sub_bass_20_80": 0.10, "bass_80_250": 0.14,
            "low_mid_250_500": 0.14, "mid_500_2k": 0.24,
            "upper_mid_2k_6k": 0.18, "presence_6k_12k": 0.13,
            "air_12k_20k": 0.07,
        },
        "type_needs": {
            "melody": 90, "vocals": 95, "hihat": 55, "pad": 65,
            "strings": 60, "fx": 50, "percussion": 50,
            "bass": 10, "kick": 10, "snare": 15, "unknown": 30,
        },
        "spectral_centroid_target": 3200,
    },
    "edm/electronic": {
        "freq_balance": {
            "sub_bass_20_80": 0.16, "bass_80_250": 0.16,
            "low_mid_250_500": 0.12, "mid_500_2k": 0.20,
            "upper_mid_2k_6k": 0.16, "presence_6k_12k": 0.13,
            "air_12k_20k": 0.07,
        },
        "type_needs": {
            "melody": 85, "vocals": 70, "hihat": 70, "pad": 80,
            "strings": 45, "fx": 75, "percussion": 65,
            "bass": 5, "kick": 5, "snare": 10, "unknown": 35,
        },
        "spectral_centroid_target": 3000,
    },
    "lo-fi/chill": {
        "freq_balance": {
            "sub_bass_20_80": 0.12, "bass_80_250": 0.16,
            "low_mid_250_500": 0.18, "mid_500_2k": 0.22,
            "upper_mid_2k_6k": 0.14, "presence_6k_12k": 0.10,
            "air_12k_20k": 0.08,
        },
        "type_needs": {
            "melody": 90, "vocals": 75, "hihat": 65, "pad": 80,
            "strings": 55, "fx": 60, "percussion": 55,
            "bass": 10, "kick": 10, "snare": 15, "unknown": 35,
        },
        "spectral_centroid_target": 2400,
    },
    "default": {
        "freq_balance": {
            "sub_bass_20_80": 0.12, "bass_80_250": 0.15,
            "low_mid_250_500": 0.14, "mid_500_2k": 0.22,
            "upper_mid_2k_6k": 0.17, "presence_6k_12k": 0.12,
            "air_12k_20k": 0.08,
        },
        "type_needs": {
            "melody": 80, "vocals": 75, "hihat": 60, "pad": 60,
            "strings": 50, "fx": 45, "percussion": 50,
            "bass": 10, "kick": 10, "snare": 15, "unknown": 30,
        },
        "spectral_centroid_target": 2800,
    },
}


def get_genre_profile(genre_str):
    """Match a genre string to the closest reference profile."""
    if not genre_str:
        return GENRE_PROFILES["default"]
    g = genre_str.lower()
    if "trap" in g or "hip-hop" in g or "hip hop" in g or "rap" in g:
        return GENRE_PROFILES["trap/hip-hop"]
    if "drill" in g:
        return GENRE_PROFILES["drill"]
    if "r&b" in g or "rnb" in g or "neo-soul" in g or "soul" in g:
        return GENRE_PROFILES["r&b"]
    if "pop" in g and "hip" not in g:
        return GENRE_PROFILES["pop"]
    if "edm" in g or "electronic" in g or "house" in g or "techno" in g:
        return GENRE_PROFILES["edm/electronic"]
    if "lo-fi" in g or "lofi" in g or "chill" in g:
        return GENRE_PROFILES["lo-fi/chill"]
    return GENRE_PROFILES["default"]
