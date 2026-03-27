"""
RESONATE Production Model — Label Alignment.

Maps labels from external datasets to our unified RPM taxonomy:
  - FMA genre labels → RPM 500+ genre hierarchy
  - NSynth instrument families → RPM 200+ instrument classes
  - MTG-Jamendo tags → RPM genre/instrument/mood labels
  - Spotify key/mode → RPM 24-key system

This is critical for multi-dataset training — every source needs to
speak the same label language.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Top-level genre mapping (12 categories)
# ──────────────────────────────────────────────────────────────────────

TOP_GENRE_MAP = {
    0: "electronic",
    1: "hip_hop",
    2: "rnb_soul",
    3: "pop",
    4: "rock",
    5: "jazz",
    6: "country",
    7: "latin",
    8: "classical",
    9: "folk_world",
    10: "metal",
    11: "other",
}

TOP_GENRE_REVERSE = {v: k for k, v in TOP_GENRE_MAP.items()}


# ──────────────────────────────────────────────────────────────────────
# FMA → RPM genre alignment
# ──────────────────────────────────────────────────────────────────────

FMA_TO_RPM_GENRE = {
    # FMA top-level genres → RPM top-level genre IDs
    "Electronic": 0,
    "Hip-Hop": 1,
    "R&B": 2,
    "Soul-RnB": 2,
    "Pop": 3,
    "Rock": 4,
    "Jazz": 5,
    "Country": 6,
    "Latin": 7,
    "Classical": 8,
    "Folk": 9,
    "International": 9,
    "Metal": 10,
    "Experimental": 11,
    "Instrumental": 11,
    "Spoken": 11,
    "Blues": 4,  # closest to rock in our taxonomy
    "Easy Listening": 3,  # closest to pop
    "Old-Time / Historic": 9,
}

# FMA sub-genre → RPM sub-genre IDs (partial mapping — full mapping
# would be generated from genre_taxonomy.py at runtime)
FMA_SUB_GENRE_MAP = {
    "Techno": 10, "House": 11, "Ambient": 12, "Drum & Bass": 13,
    "Trance": 14, "Dubstep": 15, "IDM": 16, "Downtempo": 17,
    "Trip-Hop": 18, "Breakbeat": 19,
    "Boom Bap": 50, "Trap": 51, "Lo-Fi": 52, "Cloud Rap": 53,
    "Conscious": 54, "Gangsta": 55,
    "Neo-Soul": 80, "Funk": 81, "Gospel": 82, "Motown": 83,
    "Indie Pop": 100, "Synth-Pop": 101, "Electropop": 102,
    "Classic Rock": 120, "Punk": 121, "Grunge": 122, "Alternative": 123,
    "Indie Rock": 124, "Post-Rock": 125, "Psychedelic": 126,
    "Bebop": 150, "Smooth Jazz": 151, "Fusion": 152, "Swing": 153,
    "Bluegrass": 170, "Honky-Tonk": 171, "Americana": 172,
    "Reggaeton": 200, "Salsa": 201, "Bossa Nova": 202, "Samba": 203,
    "Baroque": 230, "Romantic": 231, "Minimalist": 232,
    "Flamenco": 260, "Celtic": 261, "Afrobeat": 262,
    "Heavy Metal": 290, "Death Metal": 291, "Black Metal": 292,
}


def align_fma_genre(fma_genre_top: str, fma_genre_sub: str = "") -> tuple[int, int]:
    """
    Map FMA genre labels to RPM genre IDs.

    Returns:
        (top_genre_id, sub_genre_id) — sub_genre_id may be -1 if no mapping
    """
    top_id = FMA_TO_RPM_GENRE.get(fma_genre_top, 11)  # default: other
    sub_id = FMA_SUB_GENRE_MAP.get(fma_genre_sub, -1)
    return top_id, sub_id


# ──────────────────────────────────────────────────────────────────────
# NSynth → RPM instrument alignment
# ──────────────────────────────────────────────────────────────────────

NSYNTH_FAMILY_TO_RPM = {
    # NSynth instrument_family_str → list of RPM instrument IDs
    "bass": [30, 31, 32],       # bass guitar (finger, pick, slap)
    "brass": [40, 41, 42, 43],  # trumpet, trombone, french horn, tuba
    "flute": [50, 51],          # flute, piccolo
    "guitar": [15, 16, 17],     # acoustic, electric clean, electric distorted
    "keyboard": [60, 61, 62],   # grand piano, electric piano, clavinet
    "mallet": [70, 71, 72],     # marimba, vibraphone, xylophone
    "organ": [65, 66, 67],      # Hammond, pipe organ, Farfisa
    "reed": [45, 46, 47, 48],   # clarinet, sax alto, sax tenor, sax soprano
    "string": [0, 1, 2, 3],     # violin, viola, cello, double bass
    "synth_lead": [80, 81, 82], # synth lead, synth pad, synth bass
    "vocal": [90, 91],          # singing voice, vocal texture
}

# NSynth source → additional context
NSYNTH_SOURCE_MAP = {
    "acoustic": "acoustic",
    "electronic": "electronic",
    "synthetic": "synthetic",
}


def align_nsynth_instrument(family: str, source: str = "",
                            instrument_str: str = "") -> list[int]:
    """
    Map NSynth instrument labels to RPM instrument IDs.

    Returns:
        list of RPM instrument IDs (can be multiple for ambiguous families)
    """
    base_ids = NSYNTH_FAMILY_TO_RPM.get(family, [])

    if not base_ids:
        return []

    # Refine by source if possible
    if source == "acoustic" and len(base_ids) > 1:
        return [base_ids[0]]  # first ID is usually the acoustic variant
    elif source == "electronic" and len(base_ids) > 1:
        return [base_ids[-1]]  # last ID is usually the electronic variant

    return base_ids


# ──────────────────────────────────────────────────────────────────────
# MTG-Jamendo → RPM alignment
# ──────────────────────────────────────────────────────────────────────

JAMENDO_GENRE_TO_RPM = {
    "electronic": 0, "techno": 0, "trance": 0, "house": 0,
    "ambient": 0, "dnb": 0, "dubstep": 0, "chillout": 0,
    "hiphop": 1, "rap": 1, "triphop": 1,
    "rnb": 2, "soul": 2, "funk": 2,
    "pop": 3, "disco": 3, "easylistening": 3,
    "rock": 4, "punk": 4, "hardrock": 4, "alternative": 4,
    "indie": 4, "postrock": 4, "grunge": 4,
    "jazz": 5, "swing": 5, "bigband": 5, "bossanova": 5,
    "country": 6, "bluegrass": 6, "folk": 9,
    "latin": 7, "reggae": 7, "ska": 7, "salsa": 7,
    "classical": 8, "orchestral": 8, "opera": 8,
    "world": 9, "celtic": 9, "african": 9, "asian": 9,
    "metal": 10, "heavymetal": 10, "deathmetal": 10,
    "experimental": 11, "noise": 11, "avantgarde": 11,
}

JAMENDO_INSTRUMENT_TO_RPM = {
    "guitar": [15, 16, 17],
    "electricguitar": [16, 17],
    "acousticguitar": [15],
    "bass": [30, 31],
    "piano": [60, 61],
    "keyboards": [60, 61, 62],
    "strings": [0, 1, 2, 3],
    "violin": [0],
    "cello": [2],
    "drums": [100, 101, 102, 103],
    "percussion": [104, 105, 106],
    "synthesizer": [80, 81, 82],
    "voice": [90, 91],
    "choir": [92],
    "trumpet": [40],
    "saxophone": [46],
    "flute": [50],
    "harmonica": [55],
    "organ": [65],
    "harp": [20],
}


def align_jamendo_genre(tags: list[str]) -> tuple[int, int]:
    """Map Jamendo genre tags to RPM genre IDs."""
    for tag in tags:
        tag_lower = tag.lower().replace(" ", "").replace("-", "")
        if tag_lower in JAMENDO_GENRE_TO_RPM:
            return JAMENDO_GENRE_TO_RPM[tag_lower], -1
    return 11, -1  # default: other


def align_jamendo_instruments(tags: list[str]) -> list[int]:
    """Map Jamendo instrument tags to RPM instrument IDs."""
    result = set()
    for tag in tags:
        tag_lower = tag.lower().replace(" ", "").replace("-", "")
        if tag_lower in JAMENDO_INSTRUMENT_TO_RPM:
            result.update(JAMENDO_INSTRUMENT_TO_RPM[tag_lower])
    return sorted(result)


# ──────────────────────────────────────────────────────────────────────
# Spotify → RPM alignment
# ──────────────────────────────────────────────────────────────────────

def align_spotify_key(spotify_key: int, spotify_mode: int) -> int:
    """
    Map Spotify key/mode to RPM 24-key system.
    Spotify: key 0-11 (C=0, C#=1, ... B=11), mode 0=minor 1=major
    RPM: 0-11 = major keys, 12-23 = minor keys
    """
    if spotify_key < 0 or spotify_key > 11:
        return -1  # unknown
    if spotify_mode == 1:
        return spotify_key          # major: 0-11
    else:
        return spotify_key + 12     # minor: 12-23


def align_spotify_era(year: int) -> int:
    """Map year to RPM era index (0=1950s, 7=2020s)."""
    decade_starts = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
    for i in range(len(decade_starts) - 1, -1, -1):
        if year >= decade_starts[i]:
            return i
    return 0  # before 1950 → map to 1950s


def align_spotify_chart_potential(peak_position: int, weeks_on_chart: int) -> float:
    """
    Convert chart performance to 0-1 score.
    Combines peak position and longevity.
    """
    # Peak position: #1 = 1.0, #100 = ~0.0
    position_score = max(0.0, 1.0 - (peak_position - 1) / 99.0)

    # Weeks bonus: more weeks = more successful
    # Cap at 52 weeks (1 year)
    weeks_score = min(1.0, weeks_on_chart / 52.0)

    # Weighted combination: position matters more
    return 0.7 * position_score + 0.3 * weeks_score


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────

def validate_alignment():
    """Quick validation of all alignment functions."""
    print("Label Alignment Validation")
    print("=" * 50)

    # FMA
    top, sub = align_fma_genre("Electronic", "Techno")
    assert top == 0, f"Expected 0, got {top}"
    print(f"  FMA Electronic/Techno → top={top}, sub={sub} ✓")

    top, sub = align_fma_genre("Hip-Hop", "Boom Bap")
    assert top == 1
    print(f"  FMA Hip-Hop/Boom Bap → top={top}, sub={sub} ✓")

    # NSynth
    ids = align_nsynth_instrument("guitar", "acoustic")
    assert len(ids) > 0
    print(f"  NSynth guitar/acoustic → {ids} ✓")

    ids = align_nsynth_instrument("synth_lead", "synthetic")
    assert len(ids) > 0
    print(f"  NSynth synth_lead → {ids} ✓")

    # Jamendo
    top, sub = align_jamendo_genre(["electronic", "techno"])
    assert top == 0
    print(f"  Jamendo [electronic, techno] → top={top} ✓")

    insts = align_jamendo_instruments(["guitar", "drums", "bass"])
    assert len(insts) > 0
    print(f"  Jamendo [guitar, drums, bass] → {insts} ✓")

    # Spotify
    key = align_spotify_key(0, 1)  # C major
    assert key == 0
    print(f"  Spotify C major → key={key} ✓")

    key = align_spotify_key(9, 0)  # A minor
    assert key == 21
    print(f"  Spotify A minor → key={key} ✓")

    era = align_spotify_era(1985)
    assert era == 3  # 1980s
    print(f"  Spotify year 1985 → era={era} (1980s) ✓")

    chart = align_spotify_chart_potential(1, 30)
    assert chart > 0.8
    print(f"  Spotify #1, 30 weeks → potential={chart:.2f} ✓")

    chart = align_spotify_chart_potential(100, 1)
    assert chart < 0.1
    print(f"  Spotify #100, 1 week → potential={chart:.2f} ✓")

    print("\n✓ All alignment functions validated.")


if __name__ == "__main__":
    validate_alignment()
