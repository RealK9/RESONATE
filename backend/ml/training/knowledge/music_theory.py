"""
RESONATE Production Model (RPM) — Music Theory Knowledge Corpus

Structured music theory data used for:
1. CLAP text description generation (text-audio alignment during training)
2. Ground truth labels for the theory classification head
3. Encoding musical relationships not learnable from audio alone

This is THE canonical music theory brain for the entire RPM pipeline.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
}
FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb"}
SHARP_KEYS = {"G", "D", "A", "E", "B", "F#"}

FLAT_NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _note_index(name: str) -> int:
    """Return 0-11 semitone index for a note name (sharp or flat)."""
    if name in NOTE_NAMES:
        return NOTE_NAMES.index(name)
    if name in FLAT_NOTE_NAMES:
        return FLAT_NOTE_NAMES.index(name)
    raise ValueError(f"Unknown note: {name}")


def _transpose(root: str, semitones: int) -> str:
    """Transpose *root* up by *semitones*, choosing sharps/flats contextually."""
    idx = (_note_index(root) + semitones) % 12
    if root in FLAT_KEYS or root in ENHARMONIC and ENHARMONIC[root] in FLAT_KEYS:
        return FLAT_NOTE_NAMES[idx]
    return NOTE_NAMES[idx]


def _build_scale(root: str, intervals: List[int]) -> List[str]:
    """Build note names from a root and list of semitone intervals from root."""
    return [_transpose(root, iv) for iv in intervals]


# ---------------------------------------------------------------------------
# Scale Interval Templates (semitones from root)
# ---------------------------------------------------------------------------

SCALE_INTERVALS: Dict[str, List[int]] = {
    # Diatonic modes
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "lydian":           [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "aeolian":          [0, 2, 3, 5, 7, 8, 10],
    "locrian":          [0, 1, 3, 5, 6, 8, 10],
    # Minor variants
    "natural_minor":    [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":    [0, 2, 3, 5, 7, 9, 11],
    # Pentatonic & blues
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    # Symmetric
    "whole_tone":       [0, 2, 4, 6, 8, 10],
    "diminished_hw":    [0, 1, 3, 4, 6, 7, 9, 10],
    "diminished_wh":    [0, 2, 3, 5, 6, 8, 9, 11],
    "chromatic":        list(range(12)),
    # Bebop
    "bebop_dominant":   [0, 2, 4, 5, 7, 9, 10, 11],
    "bebop_major":      [0, 2, 4, 5, 7, 8, 9, 11],
    "bebop_minor":      [0, 2, 3, 5, 7, 8, 9, 10],
    "bebop_dorian":     [0, 2, 3, 4, 5, 7, 9, 10],
}

# Aliases for convenience
SCALE_INTERVALS["ionian"] = SCALE_INTERVALS["major"]

MODE_NAMES = [
    "ionian", "dorian", "phrygian", "lydian",
    "mixolydian", "aeolian", "locrian",
]


def get_all_scales() -> Dict[str, Dict[str, List[str]]]:
    """Return {scale_name: {root: [notes]}} for every scale template x 12 keys."""
    result: Dict[str, Dict[str, List[str]]] = {}
    for scale_name, intervals in SCALE_INTERVALS.items():
        by_key: Dict[str, List[str]] = {}
        for root in NOTE_NAMES:
            by_key[root] = _build_scale(root, intervals)
        result[scale_name] = by_key
    return result


# ---------------------------------------------------------------------------
# Chord Types
# ---------------------------------------------------------------------------

@dataclass
class ChordType:
    """Single chord quality with theory metadata."""
    name: str
    symbol: str
    intervals: Tuple[int, ...]
    category: str  # triad | seventh | extended | altered | suspended | added | power | polychord
    common_functions: List[str]  # tonic | dominant | subdominant | predominant | passing | chromatic
    tension_level: float  # 0.0 (fully resolved) .. 1.0 (maximum tension)
    genre_associations: List[str]
    description: str = ""


# Master chord registry - built procedurally then frozen
_CHORD_DEFS: List[dict] = [
    # ---- Triads ----
    {"name": "major", "symbol": "", "intervals": (0, 4, 7),
     "category": "triad", "common_functions": ["tonic", "subdominant", "dominant"],
     "tension_level": 0.05, "genre_associations": ["pop", "rock", "classical", "country", "folk"],
     "description": "Stable, bright, resolved"},
    {"name": "minor", "symbol": "m", "intervals": (0, 3, 7),
     "category": "triad", "common_functions": ["tonic", "predominant"],
     "tension_level": 0.10, "genre_associations": ["pop", "rock", "classical", "r&b", "metal"],
     "description": "Dark, melancholic, introspective"},
    {"name": "diminished", "symbol": "dim", "intervals": (0, 3, 6),
     "category": "triad", "common_functions": ["dominant", "passing"],
     "tension_level": 0.70, "genre_associations": ["classical", "jazz", "film"],
     "description": "Tense, unstable, wants resolution"},
    {"name": "augmented", "symbol": "aug", "intervals": (0, 4, 8),
     "category": "triad", "common_functions": ["dominant", "chromatic"],
     "tension_level": 0.65, "genre_associations": ["jazz", "classical", "psychedelic", "film"],
     "description": "Bright tension, dreamlike, unresolved"},
    {"name": "suspended 2nd", "symbol": "sus2", "intervals": (0, 2, 7),
     "category": "suspended", "common_functions": ["tonic", "subdominant"],
     "tension_level": 0.15, "genre_associations": ["pop", "rock", "ambient", "post-rock"],
     "description": "Open, airy, ambiguous"},
    {"name": "suspended 4th", "symbol": "sus4", "intervals": (0, 5, 7),
     "category": "suspended", "common_functions": ["dominant", "subdominant"],
     "tension_level": 0.20, "genre_associations": ["pop", "rock", "folk", "hymnal"],
     "description": "Yearning, wants to resolve to major"},

    # ---- Sevenths ----
    {"name": "major 7th", "symbol": "maj7", "intervals": (0, 4, 7, 11),
     "category": "seventh", "common_functions": ["tonic", "subdominant"],
     "tension_level": 0.15, "genre_associations": ["jazz", "neo-soul", "r&b", "bossa nova", "city pop"],
     "description": "Lush, sophisticated, warm"},
    {"name": "minor 7th", "symbol": "m7", "intervals": (0, 3, 7, 10),
     "category": "seventh", "common_functions": ["tonic", "predominant"],
     "tension_level": 0.20, "genre_associations": ["jazz", "r&b", "neo-soul", "funk"],
     "description": "Smooth, mellow, relaxed"},
    {"name": "dominant 7th", "symbol": "7", "intervals": (0, 4, 7, 10),
     "category": "seventh", "common_functions": ["dominant"],
     "tension_level": 0.40, "genre_associations": ["blues", "jazz", "funk", "rock", "gospel"],
     "description": "Driving, bluesy, wants to resolve"},
    {"name": "minor 7th flat 5", "symbol": "m7b5", "intervals": (0, 3, 6, 10),
     "category": "seventh", "common_functions": ["predominant"],
     "tension_level": 0.55, "genre_associations": ["jazz", "bossa nova", "classical"],
     "description": "Dark predominant, ii chord in minor keys"},
    {"name": "diminished 7th", "symbol": "dim7", "intervals": (0, 3, 6, 9),
     "category": "seventh", "common_functions": ["dominant", "passing", "chromatic"],
     "tension_level": 0.75, "genre_associations": ["classical", "jazz", "film", "romantic era"],
     "description": "Symmetrical tension, highly chromatic"},
    {"name": "minor-major 7th", "symbol": "mMaj7", "intervals": (0, 3, 7, 11),
     "category": "seventh", "common_functions": ["tonic", "chromatic"],
     "tension_level": 0.45, "genre_associations": ["jazz", "film", "spy", "noir"],
     "description": "Mysterious, cinematic, James Bond chord"},
    {"name": "augmented 7th", "symbol": "aug7", "intervals": (0, 4, 8, 10),
     "category": "seventh", "common_functions": ["dominant", "chromatic"],
     "tension_level": 0.60, "genre_associations": ["jazz", "blues", "gospel"],
     "description": "Dominant with raised color, heightened tension"},
    {"name": "augmented major 7th", "symbol": "augMaj7", "intervals": (0, 4, 8, 11),
     "category": "seventh", "common_functions": ["tonic", "chromatic"],
     "tension_level": 0.50, "genre_associations": ["jazz", "film", "progressive"],
     "description": "Bright, floating, otherworldly"},
    {"name": "dominant 7th sus4", "symbol": "7sus4", "intervals": (0, 5, 7, 10),
     "category": "seventh", "common_functions": ["dominant", "subdominant"],
     "tension_level": 0.35, "genre_associations": ["jazz", "funk", "gospel", "fusion"],
     "description": "Suspended dominant, gospel flavor"},

    # ---- Extended ----
    {"name": "dominant 9th", "symbol": "9", "intervals": (0, 4, 7, 10, 14),
     "category": "extended", "common_functions": ["dominant"],
     "tension_level": 0.40, "genre_associations": ["jazz", "funk", "r&b", "blues"],
     "description": "Rich dominant, funky"},
    {"name": "major 9th", "symbol": "maj9", "intervals": (0, 4, 7, 11, 14),
     "category": "extended", "common_functions": ["tonic", "subdominant"],
     "tension_level": 0.15, "genre_associations": ["jazz", "neo-soul", "r&b", "city pop"],
     "description": "Dreamy, lush, sophisticated"},
    {"name": "minor 9th", "symbol": "m9", "intervals": (0, 3, 7, 10, 14),
     "category": "extended", "common_functions": ["tonic", "predominant"],
     "tension_level": 0.20, "genre_associations": ["jazz", "neo-soul", "r&b", "lo-fi"],
     "description": "Smooth, contemplative, lo-fi chill"},
    {"name": "dominant 11th", "symbol": "11", "intervals": (0, 4, 7, 10, 14, 17),
     "category": "extended", "common_functions": ["dominant", "subdominant"],
     "tension_level": 0.45, "genre_associations": ["jazz", "funk", "fusion"],
     "description": "Stacked, ambiguous, modal"},
    {"name": "major 11th", "symbol": "maj11", "intervals": (0, 4, 7, 11, 14, 17),
     "category": "extended", "common_functions": ["tonic", "subdominant"],
     "tension_level": 0.25, "genre_associations": ["jazz", "ambient", "progressive"],
     "description": "Wide voicing, ethereal"},
    {"name": "minor 11th", "symbol": "m11", "intervals": (0, 3, 7, 10, 14, 17),
     "category": "extended", "common_functions": ["tonic", "predominant"],
     "tension_level": 0.25, "genre_associations": ["jazz", "neo-soul", "lo-fi", "ambient"],
     "description": "Spacious, meditative, chill"},
    {"name": "dominant 13th", "symbol": "13", "intervals": (0, 4, 7, 10, 14, 17, 21),
     "category": "extended", "common_functions": ["dominant"],
     "tension_level": 0.45, "genre_associations": ["jazz", "funk", "big band"],
     "description": "Full, swinging, big band color"},
    {"name": "major 13th", "symbol": "maj13", "intervals": (0, 4, 7, 11, 14, 17, 21),
     "category": "extended", "common_functions": ["tonic", "subdominant"],
     "tension_level": 0.20, "genre_associations": ["jazz", "bossa nova", "neo-soul"],
     "description": "Ultimate lush tonic, Jobim vibes"},
    {"name": "minor 13th", "symbol": "m13", "intervals": (0, 3, 7, 10, 14, 17, 21),
     "category": "extended", "common_functions": ["tonic", "predominant"],
     "tension_level": 0.25, "genre_associations": ["jazz", "neo-soul", "fusion"],
     "description": "Rich minor color, dorian character"},

    # ---- Altered ----
    {"name": "7 sharp 9", "symbol": "7#9", "intervals": (0, 4, 7, 10, 15),
     "category": "altered", "common_functions": ["dominant"],
     "tension_level": 0.70, "genre_associations": ["rock", "funk", "blues", "psychedelic"],
     "description": "The Hendrix chord, aggressive dominant"},
    {"name": "7 flat 9", "symbol": "7b9", "intervals": (0, 4, 7, 10, 13),
     "category": "altered", "common_functions": ["dominant"],
     "tension_level": 0.65, "genre_associations": ["jazz", "bossa nova", "classical"],
     "description": "Dark dominant, strong pull to resolution"},
    {"name": "7 sharp 11", "symbol": "7#11", "intervals": (0, 4, 7, 10, 14, 18),
     "category": "altered", "common_functions": ["dominant", "subdominant"],
     "tension_level": 0.55, "genre_associations": ["jazz", "fusion", "progressive"],
     "description": "Lydian dominant, bright tension"},
    {"name": "7 flat 13", "symbol": "7b13", "intervals": (0, 4, 7, 10, 14, 20),
     "category": "altered", "common_functions": ["dominant"],
     "tension_level": 0.60, "genre_associations": ["jazz", "latin", "classical"],
     "description": "Mixolydian b6 color, bittersweet"},
    {"name": "7 altered", "symbol": "7alt", "intervals": (0, 4, 6, 10, 13, 15),
     "category": "altered", "common_functions": ["dominant"],
     "tension_level": 0.80, "genre_associations": ["jazz", "fusion", "avant-garde"],
     "description": "Maximum dominant tension, all extensions altered"},
    {"name": "7 sharp 5", "symbol": "7#5", "intervals": (0, 4, 8, 10),
     "category": "altered", "common_functions": ["dominant", "chromatic"],
     "tension_level": 0.60, "genre_associations": ["jazz", "blues", "gospel"],
     "description": "Augmented dominant, heightened pull"},
    {"name": "7 flat 5", "symbol": "7b5", "intervals": (0, 4, 6, 10),
     "category": "altered", "common_functions": ["dominant", "chromatic"],
     "tension_level": 0.65, "genre_associations": ["jazz", "bebop", "film"],
     "description": "Tritone-heavy, strong chromatic motion"},
    {"name": "9 sharp 11", "symbol": "9#11", "intervals": (0, 4, 7, 10, 14, 18),
     "category": "altered", "common_functions": ["dominant", "subdominant"],
     "tension_level": 0.55, "genre_associations": ["jazz", "fusion"],
     "description": "Lydian dominant 9th, sophisticated"},
    {"name": "13 flat 9", "symbol": "13b9", "intervals": (0, 4, 7, 10, 13, 17, 21),
     "category": "altered", "common_functions": ["dominant"],
     "tension_level": 0.70, "genre_associations": ["jazz", "big band"],
     "description": "Full altered dominant with 13"},
    {"name": "7 sharp 9 sharp 5", "symbol": "7#9#5", "intervals": (0, 4, 8, 10, 15),
     "category": "altered", "common_functions": ["dominant"],
     "tension_level": 0.75, "genre_associations": ["jazz", "funk", "psychedelic"],
     "description": "Super Hendrix, maximum altered crunch"},
    {"name": "7 flat 9 flat 5", "symbol": "7b9b5", "intervals": (0, 4, 6, 10, 13),
     "category": "altered", "common_functions": ["dominant"],
     "tension_level": 0.75, "genre_associations": ["jazz", "bebop"],
     "description": "Tritone sub territory, dark and angular"},
    {"name": "7 sharp 9 flat 5", "symbol": "7#9b5", "intervals": (0, 4, 6, 10, 15),
     "category": "altered", "common_functions": ["dominant"],
     "tension_level": 0.78, "genre_associations": ["jazz", "avant-garde"],
     "description": "Extreme altered tension"},
    {"name": "7 flat 9 sharp 5", "symbol": "7b9#5", "intervals": (0, 4, 8, 10, 13),
     "category": "altered", "common_functions": ["dominant"],
     "tension_level": 0.72, "genre_associations": ["jazz", "classical"],
     "description": "Augmented dominant with flat 9"},
    {"name": "7 flat 9 sharp 11", "symbol": "7b9#11", "intervals": (0, 4, 7, 10, 13, 18),
     "category": "altered", "common_functions": ["dominant"],
     "tension_level": 0.72, "genre_associations": ["jazz", "fusion"],
     "description": "Complex altered dominant"},

    # ---- Added tone ----
    {"name": "add 9", "symbol": "add9", "intervals": (0, 4, 7, 14),
     "category": "added", "common_functions": ["tonic", "subdominant"],
     "tension_level": 0.10, "genre_associations": ["pop", "rock", "indie", "alternative"],
     "description": "Bright, shimmering, Radiohead-esque"},
    {"name": "add 11", "symbol": "add11", "intervals": (0, 4, 7, 17),
     "category": "added", "common_functions": ["tonic", "subdominant"],
     "tension_level": 0.15, "genre_associations": ["indie", "folk", "ambient"],
     "description": "Open, modal, atmospheric"},
    {"name": "minor add 9", "symbol": "madd9", "intervals": (0, 3, 7, 14),
     "category": "added", "common_functions": ["tonic", "predominant"],
     "tension_level": 0.15, "genre_associations": ["indie", "alternative", "shoegaze"],
     "description": "Dark shimmer, bittersweet"},
    {"name": "sixth", "symbol": "6", "intervals": (0, 4, 7, 9),
     "category": "added", "common_functions": ["tonic", "subdominant"],
     "tension_level": 0.10, "genre_associations": ["jazz", "swing", "country", "hawaiian"],
     "description": "Classic, nostalgic, swing era"},
    {"name": "minor sixth", "symbol": "m6", "intervals": (0, 3, 7, 9),
     "category": "added", "common_functions": ["tonic", "predominant"],
     "tension_level": 0.20, "genre_associations": ["jazz", "bossa nova", "film noir"],
     "description": "Dorian tonic, sophisticated minor"},
    {"name": "six-nine", "symbol": "6/9", "intervals": (0, 4, 7, 9, 14),
     "category": "added", "common_functions": ["tonic"],
     "tension_level": 0.10, "genre_associations": ["jazz", "funk", "neo-soul", "r&b"],
     "description": "The ultimate tonic chord, complete resolution"},
    {"name": "minor six-nine", "symbol": "m6/9", "intervals": (0, 3, 7, 9, 14),
     "category": "added", "common_functions": ["tonic"],
     "tension_level": 0.15, "genre_associations": ["jazz", "bossa nova", "neo-soul"],
     "description": "Rich minor tonic, dorian flavor"},
    {"name": "add sharp 11", "symbol": "add#11", "intervals": (0, 4, 7, 18),
     "category": "added", "common_functions": ["tonic", "subdominant"],
     "tension_level": 0.20, "genre_associations": ["jazz", "progressive", "ambient"],
     "description": "Lydian color, floating"},

    # ---- Power ----
    {"name": "power chord", "symbol": "5", "intervals": (0, 7),
     "category": "power", "common_functions": ["tonic", "subdominant", "dominant"],
     "tension_level": 0.05, "genre_associations": ["rock", "metal", "punk", "grunge", "hard rock"],
     "description": "Raw, ambiguous, distortion-friendly"},
    {"name": "power chord octave", "symbol": "5(8)", "intervals": (0, 7, 12),
     "category": "power", "common_functions": ["tonic", "subdominant", "dominant"],
     "tension_level": 0.05, "genre_associations": ["rock", "metal", "punk"],
     "description": "Thicker power chord with octave doubling"},

    # ---- Cluster / Quartal / Quintal ----
    {"name": "quartal triad", "symbol": "q", "intervals": (0, 5, 10),
     "category": "quartal", "common_functions": ["tonic", "passing", "chromatic"],
     "tension_level": 0.30, "genre_associations": ["jazz", "modal jazz", "post-bop", "ambient"],
     "description": "Open, McCoy Tyner, modal jazz staple"},
    {"name": "quintal triad", "symbol": "Q", "intervals": (0, 7, 14),
     "category": "quintal", "common_functions": ["tonic", "passing"],
     "tension_level": 0.25, "genre_associations": ["ambient", "film", "orchestral"],
     "description": "Wide, spacious, cinematic"},

    # ---- Slash / Inversion concepts ----
    {"name": "slash chord (concept)", "symbol": "/bass", "intervals": (),
     "category": "slash", "common_functions": ["tonic", "dominant", "passing", "chromatic"],
     "tension_level": 0.30, "genre_associations": ["pop", "rock", "jazz", "r&b"],
     "description": "Any chord over a non-root bass note; creates voice leading or pedal effects"},
    {"name": "polychord (concept)", "symbol": "poly", "intervals": (),
     "category": "polychord", "common_functions": ["chromatic"],
     "tension_level": 0.70, "genre_associations": ["jazz", "avant-garde", "film", "contemporary classical"],
     "description": "Two triads stacked; e.g. Eb/C creates Cmaj9#11"},
]

_CHORD_TYPE_CACHE: Optional[List[ChordType]] = None


def get_all_chord_types() -> List[ChordType]:
    """Return all ChordType instances."""
    global _CHORD_TYPE_CACHE
    if _CHORD_TYPE_CACHE is None:
        _CHORD_TYPE_CACHE = [ChordType(**d) for d in _CHORD_DEFS]
    return list(_CHORD_TYPE_CACHE)


# ---------------------------------------------------------------------------
# Chord Progressions
# ---------------------------------------------------------------------------

@dataclass
class Progression:
    name: str
    numerals: List[str]
    genre_associations: List[str]
    emotional_quality: str
    tension_profile: str  # narrative of how tension builds/resolves
    famous_examples: List[str]
    mode: str = "major"  # major | minor | modal | chromatic


_PROGRESSION_DEFS: List[dict] = [
    {
        "name": "Blues/Rock Backbone",
        "numerals": ["I", "IV", "V", "I"],
        "genre_associations": ["blues", "rock", "country", "folk", "gospel"],
        "emotional_quality": "Strong, resolved, foundational",
        "tension_profile": "Stable start, mild lift on IV, strong tension on V, full resolution on I",
        "famous_examples": ["Twist and Shout", "La Bamba", "Wild Thing"],
    },
    {
        "name": "Pop Anthem",
        "numerals": ["I", "V", "vi", "IV"],
        "genre_associations": ["pop", "rock", "indie", "anthemic"],
        "emotional_quality": "Uplifting, triumphant, singable",
        "tension_profile": "Bright start, lift on V, emotional dip on vi, warm resolve on IV",
        "famous_examples": ["Let It Be", "No Woman No Cry", "With or Without You"],
    },
    {
        "name": "Jazz ii-V-I",
        "numerals": ["ii7", "V7", "Imaj7"],
        "genre_associations": ["jazz", "bossa nova", "big band", "smooth jazz"],
        "emotional_quality": "Sophisticated, resolved, warm",
        "tension_profile": "Gentle predominant, strong dominant tension, satisfying tonic resolution",
        "famous_examples": ["Autumn Leaves", "Fly Me to the Moon", "All The Things You Are"],
    },
    {
        "name": "50s Doo-Wop",
        "numerals": ["I", "vi", "IV", "V"],
        "genre_associations": ["doo-wop", "50s pop", "rock and roll", "soul"],
        "emotional_quality": "Nostalgic, romantic, innocent",
        "tension_profile": "Stable, down to relative minor, warm subdominant, tension on V loops back",
        "famous_examples": ["Stand By Me", "Earth Angel", "Last Kiss"],
    },
    {
        "name": "Emo / Pop-Punk Minor",
        "numerals": ["i", "bVI", "bIII", "bVII"],
        "genre_associations": ["emo", "pop-punk", "alternative", "post-hardcore"],
        "emotional_quality": "Angsty, driving, cathartic",
        "tension_profile": "Dark tonic, bright lift on bVI, unstable bIII, suspended tension on bVII",
        "famous_examples": ["Welcome to the Black Parade", "The Kill", "Misery Business"],
        "mode": "minor",
    },
    {
        "name": "Minor Blues",
        "numerals": ["i", "iv", "v"],
        "genre_associations": ["blues", "minor blues", "r&b", "soul"],
        "emotional_quality": "Dark, heavy, soulful",
        "tension_profile": "Dark tonic, deeper on iv, minor dominant pulls back to i",
        "famous_examples": ["The Thrill Is Gone", "Black Magic Woman"],
        "mode": "minor",
    },
    {
        "name": "Modern Pop Standard",
        "numerals": ["I", "IV", "vi", "V"],
        "genre_associations": ["pop", "dance pop", "synth-pop"],
        "emotional_quality": "Feel-good, energetic, radio-friendly",
        "tension_profile": "Bright start, warm IV, emotional dip vi, dominant lift V",
        "famous_examples": ["Shut Up and Dance", "Shake It Off"],
    },
    {
        "name": "Modern Pop Variant",
        "numerals": ["vi", "IV", "I", "V"],
        "genre_associations": ["pop", "indie pop", "singer-songwriter"],
        "emotional_quality": "Bittersweet, reflective, anthemic build",
        "tension_profile": "Starts in emotional minor territory, warms through IV, bright peak on I, lift on V",
        "famous_examples": ["Despacito", "Grenade", "Complicated"],
    },
    {
        "name": "Rock Modal",
        "numerals": ["I", "bVII", "IV"],
        "genre_associations": ["rock", "classic rock", "grunge", "hard rock"],
        "emotional_quality": "Gritty, earthy, modal",
        "tension_profile": "Stable tonic, bluesy drop to bVII, subdominant warmth, mixolydian color",
        "famous_examples": ["Sweet Child O' Mine (verse)", "Hey Jude (coda)", "Sympathy for the Devil"],
    },
    {
        "name": "Jazz Turnaround",
        "numerals": ["Imaj7", "vi7", "ii7", "V7"],
        "genre_associations": ["jazz", "swing", "bebop"],
        "emotional_quality": "Circular, flowing, sophisticated",
        "tension_profile": "Resolved tonic, step down through circle of fifths, rising tension to V7",
        "famous_examples": ["Rhythm changes bridge", "I Got Rhythm"],
    },
    {
        "name": "Beatles / Creative Pop",
        "numerals": ["I", "III", "IV", "iv"],
        "genre_associations": ["classic pop", "art rock", "indie"],
        "emotional_quality": "Surprising, colorful, wistful",
        "tension_profile": "Bright I, chromatic lift on III, warm IV, bittersweet minor iv",
        "famous_examples": ["In My Life", "Creep (variant)", "Space Oddity"],
    },
    {
        "name": "Andalusian Cadence",
        "numerals": ["i", "bVII", "bVI", "V"],
        "genre_associations": ["flamenco", "classical", "metal", "surf rock", "spanish"],
        "emotional_quality": "Dramatic, descending, passionate",
        "tension_profile": "Dark tonic, stepwise descent builds anticipation, V creates strong pull back",
        "famous_examples": ["Hit the Road Jack", "Sultans of Swing", "Stairway to Heaven"],
        "mode": "minor",
    },
    {
        "name": "Canon Progression",
        "numerals": ["I", "V", "vi", "iii", "IV", "I", "IV", "V"],
        "genre_associations": ["classical", "pop", "wedding", "baroque pop"],
        "emotional_quality": "Grand, emotional, timeless",
        "tension_profile": "Descending bass line creates gentle wave of tension and release over 8 bars",
        "famous_examples": ["Canon in D", "Basket Case", "Graduation"],
    },
    {
        "name": "12-Bar Blues",
        "numerals": ["I7", "I7", "I7", "I7", "IV7", "IV7", "I7", "I7", "V7", "IV7", "I7", "V7"],
        "genre_associations": ["blues", "rock and roll", "jazz blues", "boogie"],
        "emotional_quality": "Grounded, expressive, soulful, call-and-response",
        "tension_profile": "4 bars stable tonic, lift to IV, return, V-IV turnaround creates cycle",
        "famous_examples": ["Sweet Home Chicago", "Johnny B. Goode", "Hound Dog"],
    },
    {
        "name": "Rhythm Changes",
        "numerals": ["Imaj7", "vi7", "ii7", "V7", "iii7", "VI7", "ii7", "V7"],
        "genre_associations": ["jazz", "bebop", "swing"],
        "emotional_quality": "Fast, virtuosic, bouncy",
        "tension_profile": "Rapid circle-of-fifths motion keeps harmonic rhythm active and propulsive",
        "famous_examples": ["I Got Rhythm", "Oleo", "Anthropology"],
    },
    {
        "name": "Coltrane Changes",
        "numerals": ["Imaj7", "bIIImaj7", "Vmaj7", "Imaj7"],
        "genre_associations": ["jazz", "post-bop", "avant-garde jazz"],
        "emotional_quality": "Kaleidoscopic, adventurous, searching",
        "tension_profile": "Tonal centers shift by major thirds creating three distinct key areas",
        "famous_examples": ["Giant Steps", "Countdown"],
        "mode": "chromatic",
    },
    {
        "name": "Neo-Soul Maj7 Voice Leading",
        "numerals": ["Imaj9", "IVmaj9", "iii7", "vi9", "ii9", "V13"],
        "genre_associations": ["neo-soul", "r&b", "gospel", "lo-fi"],
        "emotional_quality": "Warm, intimate, floating",
        "tension_profile": "Gentle voice leading between maj9 voicings, minimal tension, maximum smoothness",
        "famous_examples": ["Erykah Badu style", "D'Angelo style", "Robert Glasper style"],
    },
    {
        "name": "Trap / Dark Minor",
        "numerals": ["i", "bVI", "bVII"],
        "genre_associations": ["trap", "hip-hop", "drill", "dark pop"],
        "emotional_quality": "Dark, menacing, sparse, hypnotic",
        "tension_profile": "Minimal harmonic motion, stays in dark minor territory, bVII adds slight lift",
        "famous_examples": ["XO Tour Llif3", "Mask Off", "HUMBLE."],
        "mode": "minor",
    },
    {
        "name": "Plagal Gospel",
        "numerals": ["I", "IV", "I", "IV"],
        "genre_associations": ["gospel", "soul", "worship", "r&b"],
        "emotional_quality": "Devotional, warm, peaceful, resolved",
        "tension_profile": "Gentle rocking between tonic and subdominant, no strong dominant tension",
        "famous_examples": ["Amazing Grace (ending)", "Lean on Me"],
    },
    {
        "name": "Mixolydian Vamp",
        "numerals": ["I7", "bVII", "IV"],
        "genre_associations": ["funk", "jam band", "classic rock", "psychedelic"],
        "emotional_quality": "Groovy, earthy, hypnotic",
        "tension_profile": "Dominant tonic gives blues color, bVII is modal, IV is warm and familiar",
        "famous_examples": ["Grateful Dead jams", "Allman Brothers"],
        "mode": "modal",
    },
    {
        "name": "Dorian Vamp",
        "numerals": ["i7", "IV7"],
        "genre_associations": ["funk", "jazz-funk", "soul", "lo-fi"],
        "emotional_quality": "Groovy, cool, sophisticated minor",
        "tension_profile": "Minor tonic with raised 6th gives brightness, IV7 adds funk",
        "famous_examples": ["So What", "Oye Como Va", "Evil Ways"],
        "mode": "modal",
    },
    {
        "name": "Chromatic Mediant",
        "numerals": ["I", "bIII", "I", "bVI"],
        "genre_associations": ["film", "orchestral", "cinematic", "video game"],
        "emotional_quality": "Epic, magical, otherworldly",
        "tension_profile": "Chromatic shifts create harmonic surprise while maintaining tonal gravity",
        "famous_examples": ["John Williams scores", "Howard Shore scores"],
        "mode": "chromatic",
    },
]

_PROGRESSION_CACHE: Optional[List[Progression]] = None


def get_progressions() -> List[Progression]:
    """Return all Progression instances."""
    global _PROGRESSION_CACHE
    if _PROGRESSION_CACHE is None:
        _PROGRESSION_CACHE = [Progression(**d) for d in _PROGRESSION_DEFS]
    return list(_PROGRESSION_CACHE)


# ---------------------------------------------------------------------------
# Cadences
# ---------------------------------------------------------------------------

@dataclass
class Cadence:
    name: str
    numerals: List[str]
    resolution_strength: float  # 0.0 (no resolution) .. 1.0 (strongest)
    description: str
    genre_associations: List[str]


_CADENCE_DEFS: List[dict] = [
    {"name": "Perfect Authentic", "numerals": ["V", "I"],
     "resolution_strength": 1.0,
     "description": "Root position V to root position I, soprano on tonic. Strongest resolution in tonal music.",
     "genre_associations": ["classical", "pop", "rock", "all"]},
    {"name": "Imperfect Authentic", "numerals": ["V", "I"],
     "resolution_strength": 0.80,
     "description": "V to I but soprano not on tonic, or one chord inverted. Resolved but not fully closed.",
     "genre_associations": ["classical", "pop"]},
    {"name": "Plagal", "numerals": ["IV", "I"],
     "resolution_strength": 0.70,
     "description": "The amen cadence. Subdominant to tonic, gentle confirmation rather than strong resolution.",
     "genre_associations": ["gospel", "hymnal", "classical", "rock"]},
    {"name": "Half Cadence", "numerals": ["?", "V"],
     "resolution_strength": 0.20,
     "description": "Any chord to V. Creates expectation, phrase ends on dominant without resolving.",
     "genre_associations": ["classical", "pop", "all"]},
    {"name": "Deceptive", "numerals": ["V", "vi"],
     "resolution_strength": 0.30,
     "description": "Expected I replaced by vi. Surprise, subverted expectation, emotional twist.",
     "genre_associations": ["classical", "pop", "film", "musical theatre"]},
    {"name": "Backdoor", "numerals": ["bVII", "I"],
     "resolution_strength": 0.65,
     "description": "Flat-VII resolving to I. Jazz and pop shortcut that avoids dominant tension.",
     "genre_associations": ["jazz", "pop", "Beatles", "neo-soul"]},
    {"name": "Tritone Substitution", "numerals": ["bII7", "I"],
     "resolution_strength": 0.85,
     "description": "Dominant replaced by chord a tritone away. Chromatic bass descent, smooth jazz staple.",
     "genre_associations": ["jazz", "bebop", "neo-soul", "gospel"]},
    {"name": "Phrygian Half", "numerals": ["iv6", "V"],
     "resolution_strength": 0.25,
     "description": "Half cadence with b6-5 bass motion. Dramatic, Spanish-tinged.",
     "genre_associations": ["classical", "flamenco", "baroque"]},
    {"name": "Picardy Third", "numerals": ["v", "I"],
     "resolution_strength": 0.90,
     "description": "Minor piece ending on major tonic. Surprise brightness, baroque tradition.",
     "genre_associations": ["classical", "baroque", "choral"]},
]

_CADENCE_CACHE: Optional[List[Cadence]] = None


def get_cadences() -> List[Cadence]:
    global _CADENCE_CACHE
    if _CADENCE_CACHE is None:
        _CADENCE_CACHE = [Cadence(**d) for d in _CADENCE_DEFS]
    return list(_CADENCE_CACHE)


# ---------------------------------------------------------------------------
# Voice Leading Rules
# ---------------------------------------------------------------------------

@dataclass
class VoiceLeadingRule:
    name: str
    description: str
    priority: int  # 1 = must follow, 5 = stylistic preference
    applicable_genres: List[str]


VOICE_LEADING_RULES: List[VoiceLeadingRule] = [
    VoiceLeadingRule(
        "Common tone retention",
        "When two consecutive chords share a note, keep it in the same voice.",
        priority=1, applicable_genres=["classical", "jazz", "choral", "pop"]),
    VoiceLeadingRule(
        "Contrary motion",
        "When outer voices move in opposite directions, the part writing sounds more independent.",
        priority=2, applicable_genres=["classical", "jazz", "choral"]),
    VoiceLeadingRule(
        "Stepwise motion preference",
        "Inner voices should move by step when possible. Leaps create energy but reduce smoothness.",
        priority=2, applicable_genres=["classical", "jazz", "neo-soul"]),
    VoiceLeadingRule(
        "Leading tone resolution",
        "Scale degree 7 resolves up to 1. In dominant-tonic motion this is the strongest pull.",
        priority=1, applicable_genres=["classical", "pop", "all"]),
    VoiceLeadingRule(
        "Chordal 7th resolution",
        "The 7th of a chord resolves down by step (b7 -> 6 in V7-I, 7 -> 8 context-dependent).",
        priority=1, applicable_genres=["classical", "jazz", "all"]),
    VoiceLeadingRule(
        "4 resolves to 3",
        "Scale degree 4 tends to resolve down to 3, especially in sus4 to major resolution.",
        priority=2, applicable_genres=["pop", "classical", "rock"]),
    VoiceLeadingRule(
        "#4 resolves to 5",
        "Raised 4th (tritone of the key) resolves up to 5, especially in V7 contexts.",
        priority=2, applicable_genres=["classical", "jazz"]),
    VoiceLeadingRule(
        "Avoid parallel 5ths",
        "Two voices moving in parallel perfect 5ths lose independence. Critical in classical, less so in pop.",
        priority=1, applicable_genres=["classical", "choral"]),
    VoiceLeadingRule(
        "Avoid parallel octaves",
        "Two voices moving in parallel octaves reduce the effective voice count.",
        priority=1, applicable_genres=["classical", "choral"]),
    VoiceLeadingRule(
        "Guide tone lines",
        "3rds and 7ths of chords form smooth chromatic or stepwise lines across a progression.",
        priority=2, applicable_genres=["jazz", "neo-soul", "bossa nova"]),
    VoiceLeadingRule(
        "Bass motion by 4th/5th",
        "Root motion by perfect 4th or 5th is the strongest harmonic motion (circle of fifths).",
        priority=3, applicable_genres=["jazz", "classical", "pop"]),
    VoiceLeadingRule(
        "Chromatic approach",
        "Any chord tone can be approached by half step from above or below for smooth voice leading.",
        priority=3, applicable_genres=["jazz", "neo-soul", "gospel"]),
    VoiceLeadingRule(
        "Drop-2 voicing",
        "Second voice from top drops an octave. Standard jazz guitar/piano voicing for smooth inner movement.",
        priority=4, applicable_genres=["jazz", "r&b"]),
    VoiceLeadingRule(
        "Cluster avoidance (classical)",
        "Avoid minor 2nd intervals between adjacent voices in non-jazz contexts.",
        priority=3, applicable_genres=["classical", "choral"]),
    VoiceLeadingRule(
        "Open vs closed voicing",
        "Closed voicings (all notes within an octave) are dense; open voicings are spacious and clear.",
        priority=5, applicable_genres=["all"]),
]


# ---------------------------------------------------------------------------
# Rhythm Patterns
# ---------------------------------------------------------------------------

@dataclass
class RhythmPattern:
    name: str
    time_signature: str
    feel: str  # straight | swing | shuffle | half-time | double-time
    subdivision: str  # 8th | 16th | triplet
    genre_associations: List[str]
    description: str
    bpm_range: Tuple[int, int]
    # Pattern as list of (beat_position, velocity_weight) where beat_position
    # is float (1.0 = beat 1, 1.5 = 8th note after beat 1, etc.)
    kick_pattern: List[float] = field(default_factory=list)
    snare_pattern: List[float] = field(default_factory=list)
    hihat_pattern: List[float] = field(default_factory=list)


_RHYTHM_DEFS: List[dict] = [
    # Straight feels
    {"name": "Four on the Floor", "time_signature": "4/4", "feel": "straight", "subdivision": "8th",
     "genre_associations": ["disco", "house", "edm", "dance pop", "techno"],
     "description": "Kick on every quarter note. Foundation of dance music.",
     "bpm_range": (115, 135),
     "kick_pattern": [1.0, 2.0, 3.0, 4.0],
     "snare_pattern": [2.0, 4.0],
     "hihat_pattern": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]},
    {"name": "Standard Rock", "time_signature": "4/4", "feel": "straight", "subdivision": "8th",
     "genre_associations": ["rock", "pop", "alternative"],
     "description": "Kick on 1 and 3, snare on 2 and 4. The universal backbeat.",
     "bpm_range": (100, 140),
     "kick_pattern": [1.0, 3.0],
     "snare_pattern": [2.0, 4.0],
     "hihat_pattern": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]},
    {"name": "Boom-Bap", "time_signature": "4/4", "feel": "straight", "subdivision": "16th",
     "genre_associations": ["hip-hop", "boom-bap", "east coast rap"],
     "description": "Syncopated kick with snare on 2 and 4. Classic hip-hop groove.",
     "bpm_range": (80, 100),
     "kick_pattern": [1.0, 1.75, 3.5],
     "snare_pattern": [2.0, 4.0],
     "hihat_pattern": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75]},
    {"name": "Trap", "time_signature": "4/4", "feel": "straight", "subdivision": "16th",
     "genre_associations": ["trap", "southern hip-hop", "drill", "modern rap"],
     "description": "808 kick, sparse snare/clap, rapid hi-hat rolls with varying subdivisions.",
     "bpm_range": (130, 170),
     "kick_pattern": [1.0, 3.0],
     "snare_pattern": [2.0, 4.0],
     "hihat_pattern": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75]},
    {"name": "Reggaeton / Dembow", "time_signature": "4/4", "feel": "straight", "subdivision": "16th",
     "genre_associations": ["reggaeton", "latin trap", "dancehall"],
     "description": "Characteristic tresillo-based kick pattern with snare on the and-of-2 and 4.",
     "bpm_range": (88, 100),
     "kick_pattern": [1.0, 2.5, 3.5],
     "snare_pattern": [1.75, 2.5, 3.75, 4.5],
     "hihat_pattern": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]},
    {"name": "Waltz", "time_signature": "3/4", "feel": "straight", "subdivision": "8th",
     "genre_associations": ["classical", "folk", "country waltz", "jazz waltz"],
     "description": "Strong downbeat, lighter beats 2 and 3. OOM-pah-pah.",
     "bpm_range": (80, 150),
     "kick_pattern": [1.0],
     "snare_pattern": [2.0, 3.0],
     "hihat_pattern": [1.0, 2.0, 3.0]},
    {"name": "6/8 Compound", "time_signature": "6/8", "feel": "straight", "subdivision": "8th",
     "genre_associations": ["folk", "celtic", "rock ballad", "metal ballad"],
     "description": "Two groups of three. Strong pulse on 1 and 4 (of 6).",
     "bpm_range": (50, 100),
     "kick_pattern": [1.0, 4.0],
     "snare_pattern": [4.0],
     "hihat_pattern": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},

    # Swing and shuffle
    {"name": "Jazz Swing", "time_signature": "4/4", "feel": "swing", "subdivision": "triplet",
     "genre_associations": ["jazz", "big band", "bebop", "swing"],
     "description": "Ride cymbal plays swing pattern. Triplet feel, accents on 2 and 4.",
     "bpm_range": (100, 280),
     "kick_pattern": [1.0],
     "snare_pattern": [2.0, 4.0],
     "hihat_pattern": [1.0, 1.67, 2.0, 2.67, 3.0, 3.67, 4.0, 4.67]},
    {"name": "Shuffle", "time_signature": "4/4", "feel": "shuffle", "subdivision": "triplet",
     "genre_associations": ["blues", "blues rock", "boogie", "southern rock"],
     "description": "Triplet-based feel with accent on first and third triplet partial. Bouncy, bluesy.",
     "bpm_range": (80, 140),
     "kick_pattern": [1.0, 3.0],
     "snare_pattern": [2.0, 4.0],
     "hihat_pattern": [1.0, 1.67, 2.0, 2.67, 3.0, 3.67, 4.0, 4.67]},
    {"name": "Half-Time Shuffle", "time_signature": "4/4", "feel": "shuffle", "subdivision": "triplet",
     "genre_associations": ["funk", "fusion", "progressive rock"],
     "description": "Snare on 3 only with ghost notes. Bernard Purdie / Jeff Porcaro signature.",
     "bpm_range": (85, 115),
     "kick_pattern": [1.0, 2.67, 3.0],
     "snare_pattern": [3.0],
     "hihat_pattern": [1.0, 1.67, 2.0, 2.67, 3.0, 3.67, 4.0, 4.67]},

    # Half-time and double-time
    {"name": "Half-Time", "time_signature": "4/4", "feel": "half-time", "subdivision": "8th",
     "genre_associations": ["metal", "post-rock", "ambient", "trap", "pop ballad"],
     "description": "Snare on 3 instead of 2 and 4. Makes tempo feel half speed.",
     "bpm_range": (60, 140),
     "kick_pattern": [1.0, 2.5],
     "snare_pattern": [3.0],
     "hihat_pattern": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]},
    {"name": "Double-Time", "time_signature": "4/4", "feel": "double-time", "subdivision": "16th",
     "genre_associations": ["punk", "drum and bass", "speed metal", "hardcore"],
     "description": "Snare on every beat, kick rapid. Feels twice the indicated tempo.",
     "bpm_range": (120, 200),
     "kick_pattern": [1.0, 1.5, 2.5, 3.0, 3.5, 4.5],
     "snare_pattern": [1.0, 2.0, 3.0, 4.0],
     "hihat_pattern": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75]},

    # Syncopation
    {"name": "Bossa Nova", "time_signature": "4/4", "feel": "straight", "subdivision": "16th",
     "genre_associations": ["bossa nova", "latin jazz", "mpb"],
     "description": "Syncopated bass with cross-stick on rim. Gentle Latin feel.",
     "bpm_range": (100, 140),
     "kick_pattern": [1.0, 2.5, 3.0, 4.5],
     "snare_pattern": [2.0, 4.0],
     "hihat_pattern": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]},
    {"name": "Funk 16th", "time_signature": "4/4", "feel": "straight", "subdivision": "16th",
     "genre_associations": ["funk", "r&b", "disco", "neo-soul"],
     "description": "Heavily syncopated kick, tight hi-hats on 16ths, ghost notes on snare.",
     "bpm_range": (90, 120),
     "kick_pattern": [1.0, 1.75, 2.5, 3.0, 3.75, 4.5],
     "snare_pattern": [2.0, 4.0],
     "hihat_pattern": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75]},
    {"name": "Reggae One Drop", "time_signature": "4/4", "feel": "straight", "subdivision": "8th",
     "genre_associations": ["reggae", "dub", "ska"],
     "description": "Kick and snare together on 3, nothing on 1. Offbeat guitar.",
     "bpm_range": (70, 100),
     "kick_pattern": [3.0],
     "snare_pattern": [3.0],
     "hihat_pattern": [1.5, 2.5, 3.5, 4.5]},
    {"name": "Second Line", "time_signature": "4/4", "feel": "swing", "subdivision": "16th",
     "genre_associations": ["new orleans", "brass band", "funk"],
     "description": "Syncopated New Orleans street beat. Layered interlocking patterns.",
     "bpm_range": (100, 140),
     "kick_pattern": [1.0, 2.0, 3.0, 4.0],
     "snare_pattern": [1.5, 2.5, 3.75, 4.5],
     "hihat_pattern": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]},

    # Polyrhythm templates
    {"name": "3:2 Polyrhythm", "time_signature": "4/4", "feel": "straight", "subdivision": "8th",
     "genre_associations": ["afrobeat", "latin", "world", "progressive"],
     "description": "Three evenly spaced hits over two beats. Foundation of Afro-Cuban clave.",
     "bpm_range": (80, 140),
     "kick_pattern": [1.0, 1.67, 2.33],
     "snare_pattern": [1.0, 2.0],
     "hihat_pattern": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]},
    {"name": "4:3 Polyrhythm", "time_signature": "4/4", "feel": "straight", "subdivision": "16th",
     "genre_associations": ["progressive", "math rock", "afrobeat"],
     "description": "Four against three. Creates fascinating rhythmic tension.",
     "bpm_range": (80, 140),
     "kick_pattern": [1.0, 1.75, 2.5, 3.25],
     "snare_pattern": [1.0, 2.33, 3.67],
     "hihat_pattern": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]},
    {"name": "5:4 Polyrhythm", "time_signature": "4/4", "feel": "straight", "subdivision": "16th",
     "genre_associations": ["progressive", "math rock", "experimental"],
     "description": "Five against four. Complex but hypnotic.",
     "bpm_range": (80, 130),
     "kick_pattern": [1.0, 1.8, 2.6, 3.4, 4.2],
     "snare_pattern": [1.0, 2.0, 3.0, 4.0],
     "hihat_pattern": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]},

    # Odd meters
    {"name": "7/8 Progressive", "time_signature": "7/8", "feel": "straight", "subdivision": "8th",
     "genre_associations": ["progressive rock", "math rock", "balkan", "fusion"],
     "description": "Commonly grouped 2+2+3 or 3+2+2. Asymmetric drive.",
     "bpm_range": (100, 160),
     "kick_pattern": [1.0, 3.0, 5.0],
     "snare_pattern": [3.0, 7.0],
     "hihat_pattern": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]},
    {"name": "5/4 Take Five", "time_signature": "5/4", "feel": "swing", "subdivision": "8th",
     "genre_associations": ["jazz", "progressive rock", "film"],
     "description": "Grouped 3+2. Dave Brubeck made it famous.",
     "bpm_range": (130, 180),
     "kick_pattern": [1.0, 4.0],
     "snare_pattern": [4.0],
     "hihat_pattern": [1.0, 2.0, 3.0, 4.0, 5.0]},
]

_RHYTHM_CACHE: Optional[List[RhythmPattern]] = None


def get_rhythm_patterns() -> List[RhythmPattern]:
    global _RHYTHM_CACHE
    if _RHYTHM_CACHE is None:
        _RHYTHM_CACHE = [RhythmPattern(**d) for d in _RHYTHM_DEFS]
    return list(_RHYTHM_CACHE)


# ---------------------------------------------------------------------------
# Key Relationships
# ---------------------------------------------------------------------------

def get_key_relationships(key: str, mode: str = "major") -> Dict[str, str]:
    """Return related keys for a given key and mode.

    Keys returned:
        relative_major / relative_minor
        parallel_major / parallel_minor
        dominant
        subdominant
        secondary_dominant (V/V)
        tritone_sub (for the dominant)
    """
    idx = _note_index(key)
    result: Dict[str, str] = {}

    if mode == "major":
        result["relative_minor"] = _transpose(key, 9)   # 9 semitones up = minor 6th
        result["parallel_minor"] = key                    # same root, minor mode
        result["dominant"] = _transpose(key, 7)
        result["subdominant"] = _transpose(key, 5)
        result["secondary_dominant"] = _transpose(key, 2)  # V/V = a whole step up
        result["tritone_sub_of_dominant"] = _transpose(key, 1)  # bII
    elif mode in ("minor", "aeolian", "natural_minor"):
        result["relative_major"] = _transpose(key, 3)
        result["parallel_major"] = key
        result["dominant"] = _transpose(key, 7)
        result["subdominant"] = _transpose(key, 5)
        result["secondary_dominant"] = _transpose(key, 2)
        result["tritone_sub_of_dominant"] = _transpose(key, 1)
    else:
        # For any mode, still provide dominant/subdominant
        result["dominant"] = _transpose(key, 7)
        result["subdominant"] = _transpose(key, 5)

    return result


# ---------------------------------------------------------------------------
# CLAP Text Description Generator
# ---------------------------------------------------------------------------

# Emotional/quality adjectives keyed by tension ranges
_TENSION_ADJECTIVES = {
    (0.0, 0.15): ["stable", "resolved", "restful", "calm", "peaceful"],
    (0.15, 0.30): ["warm", "smooth", "gentle", "mellow", "relaxed"],
    (0.30, 0.50): ["driving", "moving", "moderate tension", "expectant", "flowing"],
    (0.50, 0.70): ["tense", "searching", "unsettled", "edgy", "restless"],
    (0.70, 1.01): ["highly tense", "dissonant", "angular", "aggressive", "unresolved"],
}

_GENRE_CONTEXT = {
    "jazz": ["in a jazz context", "with jazz voicings", "in a jazz arrangement"],
    "blues": ["in a blues setting", "with a blues feel", "over a blues groove"],
    "pop": ["in a pop arrangement", "in a pop song", "with pop production"],
    "rock": ["in a rock song", "with rock energy", "in a rock arrangement"],
    "classical": ["in a classical context", "in an orchestral setting", "in a classical piece"],
    "neo-soul": ["in a neo-soul groove", "with neo-soul voicings", "in a neo-soul track"],
    "funk": ["in a funk groove", "with a funky feel", "over a funk rhythm"],
    "r&b": ["in an R&B track", "with R&B production", "in an R&B arrangement"],
    "metal": ["in a metal song", "with heavy distortion", "in a metal arrangement"],
    "hip-hop": ["in a hip-hop beat", "over a hip-hop groove", "in a rap production"],
    "trap": ["in a trap beat", "over 808s", "with trap production"],
    "latin": ["in a Latin arrangement", "with Latin rhythms", "in a Latin style"],
    "gospel": ["in a gospel arrangement", "with gospel harmonies", "in a worship setting"],
    "ambient": ["in an ambient texture", "with ambient production", "in an atmospheric piece"],
    "film": ["in a film score", "in a cinematic context", "in a soundtrack"],
}


def _tension_adj(level: float) -> str:
    for (lo, hi), adjs in _TENSION_ADJECTIVES.items():
        if lo <= level < hi:
            return random.choice(adjs)
    return "neutral"


def generate_clap_descriptions(
    n_chord_descs: int = 500,
    n_progression_descs: int = 200,
    n_scale_descs: int = 200,
    n_rhythm_descs: int = 100,
    n_cadence_descs: int = 80,
    seed: int = 42,
) -> List[Tuple[str, Dict]]:
    """Generate (text_description, labels_dict) pairs for CLAP text-audio alignment.

    These natural language descriptions become training targets for the text encoder.
    The labels_dict contains structured ground-truth for the theory classification head.

    Args:
        n_chord_descs: number of chord descriptions to generate
        n_progression_descs: number of progression descriptions to generate
        n_scale_descs: number of scale descriptions to generate
        n_rhythm_descs: number of rhythm descriptions to generate
        n_cadence_descs: number of cadence descriptions to generate
        seed: random seed for reproducibility

    Returns:
        list of (text_description, labels_dict) tuples
    """
    rng = random.Random(seed)
    results: List[Tuple[str, Dict]] = []
    chord_types = get_all_chord_types()
    progressions = get_progressions()
    cadences = get_cadences()
    rhythms = get_rhythm_patterns()

    # --- Chord descriptions ---
    roots = NOTE_NAMES + ["Db", "Eb", "Gb", "Ab", "Bb"]
    playable_chords = [c for c in chord_types if c.intervals]  # skip concept chords

    for _ in range(n_chord_descs):
        chord = rng.choice(playable_chords)
        root = rng.choice(roots)
        genre = rng.choice(chord.genre_associations) if chord.genre_associations else "pop"
        context = rng.choice(_GENRE_CONTEXT.get(genre, [f"in a {genre} style"]))
        adj = _tension_adj(chord.tension_level)

        templates = [
            f"a {root} {chord.name} chord {context}",
            f"a {adj} {root}{chord.symbol} chord",
            f"a {root} {chord.name} chord with a {adj} quality",
            f"{root}{chord.symbol} voicing {context}, sounding {adj}",
            f"a {chord.category} chord: {root}{chord.symbol}, functioning as {rng.choice(chord.common_functions)}",
        ]
        text = rng.choice(templates)
        labels = {
            "chord_root": root,
            "chord_type": chord.name,
            "chord_symbol": chord.symbol,
            "category": chord.category,
            "intervals": list(chord.intervals),
            "tension_level": chord.tension_level,
            "genre": genre,
        }
        results.append((text, labels))

    # --- Progression descriptions ---
    for _ in range(n_progression_descs):
        prog = rng.choice(progressions)
        key = rng.choice(roots)
        genre = rng.choice(prog.genre_associations)
        context = rng.choice(_GENRE_CONTEXT.get(genre, [f"in a {genre} style"]))
        numerals_str = " - ".join(prog.numerals)

        templates = [
            f"a {prog.name} progression ({numerals_str}) in the key of {key} {prog.mode} {context}",
            f"the chord progression {numerals_str} in {key}, sounding {prog.emotional_quality.lower().split(',')[0]}",
            f"a {genre} progression in {key}: {numerals_str}, with {prog.emotional_quality.lower()} character",
            f"{prog.name} in {key} {prog.mode} — {numerals_str} — {context}",
        ]
        text = rng.choice(templates)
        labels = {
            "progression_name": prog.name,
            "numerals": prog.numerals,
            "key": key,
            "mode": prog.mode,
            "genre": genre,
            "emotional_quality": prog.emotional_quality,
        }
        results.append((text, labels))

    # --- Scale descriptions ---
    all_scales = get_all_scales()
    scale_names = list(all_scales.keys())

    for _ in range(n_scale_descs):
        scale_name = rng.choice(scale_names)
        root = rng.choice(roots[:12])  # keep to sharps for scales
        notes = all_scales[scale_name].get(root, all_scales[scale_name].get("C"))

        # Genre context for scale types
        scale_genre_map = {
            "blues": "blues", "pentatonic_minor": "rock", "pentatonic_major": "country",
            "dorian": "jazz", "mixolydian": "rock", "lydian": "film",
            "phrygian": "metal", "harmonic_minor": "classical",
            "melodic_minor": "jazz", "bebop_dominant": "bebop",
            "whole_tone": "impressionist", "diminished_hw": "jazz",
        }
        genre = scale_genre_map.get(scale_name, rng.choice(["pop", "jazz", "rock"]))
        context = rng.choice(_GENRE_CONTEXT.get(genre, [f"in a {genre} context"]))

        templates = [
            f"a {root} {scale_name.replace('_', ' ')} scale melody {context}",
            f"melody using the {scale_name.replace('_', ' ')} scale in {root}",
            f"a {root} {scale_name.replace('_', ' ')} run {context}",
            f"notes from the {root} {scale_name.replace('_', ' ')} scale played melodically",
        ]
        text = rng.choice(templates)
        labels = {
            "scale_name": scale_name,
            "root": root,
            "notes": notes,
            "genre": genre,
        }
        results.append((text, labels))

    # --- Rhythm descriptions ---
    for _ in range(n_rhythm_descs):
        rhythm = rng.choice(rhythms)
        bpm = rng.randint(rhythm.bpm_range[0], rhythm.bpm_range[1])
        genre = rng.choice(rhythm.genre_associations)

        templates = [
            f"a {rhythm.name} drum pattern at {bpm} BPM in a {genre} track",
            f"{rhythm.feel} {rhythm.time_signature} groove at {bpm} BPM, {genre} style",
            f"a {genre} beat using a {rhythm.name} pattern, {rhythm.feel} feel, {bpm} BPM",
            f"{rhythm.name} rhythm in {rhythm.time_signature} time at {bpm} BPM",
        ]
        text = rng.choice(templates)
        labels = {
            "rhythm_name": rhythm.name,
            "time_signature": rhythm.time_signature,
            "feel": rhythm.feel,
            "bpm": bpm,
            "genre": genre,
            "subdivision": rhythm.subdivision,
        }
        results.append((text, labels))

    # --- Cadence descriptions ---
    for _ in range(n_cadence_descs):
        cadence = rng.choice(cadences)
        key = rng.choice(roots)
        genre = rng.choice(cadence.genre_associations) if cadence.genre_associations else "classical"
        numerals_str = " to ".join(cadence.numerals)

        templates = [
            f"a {cadence.name.lower()} cadence ({numerals_str}) in {key} major",
            f"{numerals_str} cadential motion in {key}, {cadence.name.lower()} resolution",
            f"a phrase ending with a {cadence.name.lower()} cadence in the key of {key}",
        ]
        text = rng.choice(templates)
        labels = {
            "cadence_name": cadence.name,
            "numerals": cadence.numerals,
            "key": key,
            "resolution_strength": cadence.resolution_strength,
            "genre": genre,
        }
        results.append((text, labels))

    # --- Combined / complex descriptions ---
    for _ in range(50):
        prog = rng.choice(progressions)
        rhythm = rng.choice(rhythms)
        key = rng.choice(roots)
        bpm = rng.randint(rhythm.bpm_range[0], rhythm.bpm_range[1])
        shared_genres = set(prog.genre_associations) & set(rhythm.genre_associations)
        genre = rng.choice(list(shared_genres)) if shared_genres else rng.choice(prog.genre_associations)

        numerals_str = " - ".join(prog.numerals)
        text = (
            f"a {genre} track in {key} {prog.mode} at {bpm} BPM "
            f"using a {rhythm.name} groove with a {prog.name} progression ({numerals_str}), "
            f"sounding {prog.emotional_quality.lower().split(',')[0]}"
        )
        labels = {
            "progression_name": prog.name,
            "numerals": prog.numerals,
            "key": key,
            "mode": prog.mode,
            "rhythm_name": rhythm.name,
            "time_signature": rhythm.time_signature,
            "feel": rhythm.feel,
            "bpm": bpm,
            "genre": genre,
        }
        results.append((text, labels))

    rng.shuffle(results)
    return results


# ---------------------------------------------------------------------------
# Utility: interval name lookup
# ---------------------------------------------------------------------------

INTERVAL_NAMES = {
    0: "unison", 1: "minor 2nd", 2: "major 2nd", 3: "minor 3rd",
    4: "major 3rd", 5: "perfect 4th", 6: "tritone", 7: "perfect 5th",
    8: "minor 6th", 9: "major 6th", 10: "minor 7th", 11: "major 7th",
    12: "octave", 13: "minor 9th", 14: "major 9th", 15: "minor 10th",
    16: "major 10th", 17: "perfect 11th", 18: "augmented 11th",
    19: "perfect 12th", 20: "minor 13th", 21: "major 13th",
}


def interval_name(semitones: int) -> str:
    """Human-readable name for a semitone interval."""
    return INTERVAL_NAMES.get(semitones % 24, f"{semitones} semitones")


# ---------------------------------------------------------------------------
# Module self-test / summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scales = get_all_scales()
    chords = get_all_chord_types()
    progs = get_progressions()
    cads = get_cadences()
    rhythms = get_rhythm_patterns()

    print(f"Scales:       {len(scales)} types x 12 keys = {len(scales) * 12} total")
    print(f"Chord types:  {len(chords)}")
    print(f"Progressions: {len(progs)}")
    print(f"Cadences:     {len(cads)}")
    print(f"Rhythms:      {len(rhythms)}")
    print(f"Voice leading rules: {len(VOICE_LEADING_RULES)}")

    descs = generate_clap_descriptions()
    print(f"\nCLAP descriptions generated: {len(descs)}")
    print("\nSample descriptions:")
    for text, labels in descs[:10]:
        print(f"  TEXT:   {text}")
        print(f"  LABELS: { {k: v for k, v in labels.items() if k != 'notes'} }")
        print()
