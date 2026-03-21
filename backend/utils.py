"""
RESONATE — Shared utility functions.
Music theory constants, key normalization, filename parsing, classification.
"""

import re
from pathlib import Path

# ── Music theory constants ──────────────────────────────────────────────────
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Enharmonic normalization
ENHARMONIC = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#",
    "Bb": "A#", "Cb": "B", "E#": "F", "B#": "C",
    "Dbm": "C#m", "Ebm": "D#m", "Fbm": "Em", "Gbm": "F#m", "Abm": "G#m",
    "Bbm": "A#m", "Cbm": "Bm", "E#m": "Fm", "B#m": "Cm",
}


def normalize_key(k):
    if not k:
        return k
    k = k.strip()
    return ENHARMONIC.get(k, k)


# ═══════════════════════════════════════════════════════════════════════════
# FILENAME PARSING
# ═══════════════════════════════════════════════════════════════════════════

KEY_PATTERN = re.compile(
    r'(?:^|[_\-\s])'
    r'([A-Ga-g][#b]?)'
    r'(min|maj|minor|major|m(?=[_\-\s.\d]|$))?'
    r'(?=[_\-\s.\d]|$)',
    re.IGNORECASE
)


def parse_key(filename):
    """Extract musical key from filename."""
    stem = Path(filename).stem
    matches = list(KEY_PATTERN.finditer(stem))
    if not matches:
        return None
    m = matches[-1]
    note = m.group(1)
    note = note[0].upper() + note[1:] if len(note) > 1 else note.upper()
    if len(note) == 2 and note[1] == 'b':
        note = note[0] + 'b'
    quality = (m.group(2) or "").lower()
    is_minor = quality in ("min", "minor", "m")
    key_str = f"{note}m" if is_minor else note
    return normalize_key(key_str)


BPM_PATTERN = re.compile(r'(?:^|[_\-\s])(\d{2,3})(?=[_\-\s.]|$)')
BPM_PATTERN_SUFFIX = re.compile(r'(\d{2,3})\s*[Bb][Pp][Mm]')


def parse_bpm(filename):
    """Extract BPM from filename."""
    stem = Path(filename).stem
    for m in BPM_PATTERN_SUFFIX.finditer(stem):
        val = int(m.group(1))
        if 60 <= val <= 200:
            return float(val)
    for m in BPM_PATTERN.finditer(stem):
        val = int(m.group(1))
        if 60 <= val <= 200:
            return float(val)
    return None


def file_hash(filepath):
    """Quick hash based on filename + size + mtime (no content read)."""
    st = filepath.stat()
    return f"{filepath.name}:{st.st_size}:{int(st.st_mtime)}"


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

PERC_FOLDER_WORDS = {
    "drums", "drum", "percussion", "percs", "perc", "one-shots", "oneshots",
    "kicks", "kick", "snares", "snare", "claps", "clap", "hihats", "hihat",
    "hi-hats", "cymbals", "shakers",
}

PERC_TOKENS_STRONG = {
    "kick", "snare", "clap", "hihat", "shaker", "rim",
    "cymbal", "ride", "cowbell", "tambourine",
    "conga", "bongo", "percussion", "hh",
}

PERC_TOKENS_WEAK = {"crash", "drum", "hat", "tom", "perc", "snap", "stomp", "click"}


def tokenize_name(name):
    """Split filename into tokens, handling camelCase and delimiters."""
    parts = re.split(r'[_\-\s]+', name)
    expanded = []
    for part in parts:
        sub = re.sub(r'([a-z])([A-Z])', r'\1_\2', part).lower().split('_')
        expanded.extend(sub)
    return expanded


def is_percussion(filepath):
    """Determine if a file is a percussion sample."""
    folder_parts = [p.lower() for p in Path(filepath).parts[:-1]]
    for p in folder_parts:
        if p in PERC_FOLDER_WORDS:
            return True

    tokens = tokenize_name(Path(filepath).stem)

    for t in tokens:
        if t in PERC_TOKENS_STRONG:
            return True

    for t in tokens:
        if t in PERC_TOKENS_WEAK:
            for p in folder_parts:
                if p in PERC_FOLDER_WORDS:
                    return True

    return False


TYPE_MAP = {
    "808": "bass", "bass": "bass", "sub": "bass", "reese": "bass",
    "kick": "kick",
    "snare": "snare", "clap": "snare", "rimshot": "snare",
    "hihat": "hihat", "hat": "hihat", "ride": "hihat", "cymbal": "hihat", "shaker": "hihat",
    "vocal": "vocals", "vox": "vocals", "voice": "vocals",
    "guitar": "melody", "piano": "melody", "keys": "melody", "synth": "melody",
    "melody": "melody", "lead": "melody", "arp": "melody", "pluck": "melody",
    "pad": "pad", "strings": "strings", "string": "strings",
    "fx": "fx", "riser": "fx", "sweep": "fx", "impact": "fx", "transition": "fx",
    "perc": "percussion", "bongo": "percussion", "conga": "percussion", "tambourine": "percussion",
    "chord": "melody", "chords": "melody", "stab": "melody",
    "flute": "melody", "sax": "melody", "brass": "melody", "horn": "melody",
    "bansuri": "melody", "kora": "melody",
}


def classify_type(filepath):
    """Classify sample type from filename and path."""
    name = Path(filepath).stem.lower()
    tokens = tokenize_name(name)
    folder_tokens = [p.lower() for p in Path(filepath).parts[:-1]]
    all_tokens = tokens + folder_tokens

    for token in all_tokens:
        if token in TYPE_MAP:
            return TYPE_MAP[token]

    for kw, t in TYPE_MAP.items():
        if len(kw) > 3 and kw in name:
            return t

    return "unknown"


def clean_name(raw_name):
    """Strip common prefixes and make readable."""
    name = raw_name
    name = re.sub(r'^[A-Z0-9_]{2,20}_[A-Z0-9_]{2,20}_\d+_', '', name)
    name = re.sub(r'^[A-Z]{1,4}_[A-Z0-9]{1,8}_\d+_', '', name)
    name = re.sub(r'^[A-Z]_[A-Z]{2,6}_[A-Z]?_?\d+_', '', name)
    name = re.sub(r'^[A-Z]{2,4}_[A-Za-z0-9]{2,12}_\d+_', '', name)
    name = re.sub(r'^\d+[A-Z]?_[A-Z]+_', '', name)
    name = name.replace('_', ' ')
    name = re.sub(r'\s+', ' ', name).strip()
    if name:
        name = ' '.join(w.capitalize() if len(w) > 2 else w for w in name.split())
    return name or raw_name.replace('_', ' ')


# Type display labels
TYPE_LABELS = {
    "melody": "Melody", "vocals": "Vocal", "hihat": "Hi-Hat",
    "pad": "Pad", "strings": "Strings", "fx": "FX",
    "percussion": "Perc", "bass": "Bass", "kick": "Kick",
    "snare": "Snare", "unknown": "Sound",
}
