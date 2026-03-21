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


_BRAND_PREFIXES = re.compile(
    r'^(?:Cymatics|Ultrasonic|Splice|Loopcloud|Output|Native Instruments'
    r'|KSHMR|DPTE2?|DS|JUA|LSLMP|PREM|HT|PR|US|iah|hhvt|ds)'
    r'\s*[-–—x_ ]\s*',
    re.IGNORECASE,
)

_BRAND_USCORE = re.compile(
    r'^(?:KSHMR|DPTE2?|DS[\s_]MT\d|JUA|LSLMP|PREM|US[\s_]\w+|PR[\s_]OF|HT|hhvt)'
    r'[\s_]',
    re.IGNORECASE,
)


def clean_name(raw_name):
    """
    Strip pack prefixes, brand names, codes, BPM/key suffixes,
    and produce a clean, minimal display name.
    """
    name = raw_name

    # If path separators exist (nested pack paths), take the last segment
    if '\\' in name:
        name = name.rsplit('\\', 1)[-1]
    if '/' in name:
        name = name.rsplit('/', 1)[-1]

    # Strip file extension if still present
    name = re.sub(r'\.(wav|mp3|flac|aiff|aif|ogg|m4a)$', '', name, flags=re.IGNORECASE)

    # Replace underscores with spaces early
    name = name.replace('_', ' ')

    # ── Remove BPM + key FIRST (before any dash splitting) ─────────────
    # Remove BPM: "136 BPM", "100bpm", "140 BPM"
    name = re.sub(r'\s*[-–]?\s*\d{2,3}\s*[Bb][Pp][Mm]\b', '', name)
    # Remove key + scale inline: "E Maj", "F# Min", "D# Minor", "A# Maj", "F Sharp"
    name = re.sub(
        r'\s+[A-G][#b]?\s+(?:sharp|flat|min(?:or)?|maj(?:or)?)\b',
        '', name, flags=re.IGNORECASE
    )
    # Remove trailing key: "- G", "- F#", "G#m" (must be preceded by space/dash)
    name = re.sub(
        r'(?<=\s)[-–]?\s*[A-G][#b]?\s*(?:min(?:or)?|maj(?:or)?|m(?=[\s\d]|$))?\s*$',
        '', name, flags=re.IGNORECASE
    )
    # Remove inline key in parens: "(F#)", "(G)"
    name = re.sub(r'\s*\([A-G][#b]?\)', '', name)

    # Remove "One Shot(s)", "Loop Stem(s)", "Drum One Shots", "Sample Pack"
    name = re.sub(r'\b(?:One\s*Shots?|Loop\s*Stems?|Drum\s*One\s*Shots?)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(?:Free\s+)?Sample\s+Pack\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\bVol\.?\s*\d+\b', '', name, flags=re.IGNORECASE)

    # ── Strip brand prefixes ────────────────────────────────────────────
    name = _BRAND_PREFIXES.sub('', name)
    name = _BRAND_USCORE.sub('', name)

    # ── Handle dash-separated multi-part names ──────────────────────────
    parts = re.split(r'\s*[-–—]\s*', name)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 3:
        # Keep meaningful segments (not pure numbers unless that's all we have)
        good = [p for p in parts if len(p) > 2 and not re.match(r'^\d+$', p)]
        if not good:
            good = [p for p in parts if p.strip()]
        name = ' '.join(good[-2:]) if len(good) >= 2 else good[-1] if good else name
    elif len(parts) == 2:
        # If second part is just a number, keep both; otherwise use second
        if re.match(r'^\d+$', parts[1]):
            name = ' '.join(parts)
        elif len(parts[1]) >= 3:
            name = parts[1]
        else:
            name = ' '.join(parts)
    elif len(parts) == 1:
        name = parts[0]

    # Remove short code prefixes at start: "x S1 "
    name = re.sub(r'^[xX]\s+[A-Z]\d\s+', '', name)

    # Remove leading numbers with dash/dot: "37 dt ", "12 hit "
    name = re.sub(r'^\d{1,3}[\-.\s]+', '', name)

    # Strip leading zeros from trailing numbers: " 003" → " 3", " 02" → " 2"
    name = re.sub(r'\b0+(\d+)\b', r'\1', name)

    # Clean up dashes, spaces, trailing punctuation
    name = re.sub(r'\s*[-–—]\s*$', '', name)
    name = re.sub(r'^\s*[-–—]\s*', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    # Title case — capitalize meaningful words
    if name:
        words = name.split()
        cleaned = []
        for w in words:
            # Keep acronyms like "FX", "HH", "808"
            if re.match(r'^[A-Z]{2,3}$', w) or re.match(r'^\d+$', w):
                cleaned.append(w)
            elif len(w) > 2:
                cleaned.append(w[0].upper() + w[1:].lower())
            else:
                cleaned.append(w.lower())
        name = ' '.join(cleaned)

    return name if name and len(name) > 1 else raw_name.replace('_', ' ')


# Type display labels
TYPE_LABELS = {
    "melody": "Melody", "vocals": "Vocal", "hihat": "Hi-Hat",
    "pad": "Pad", "strings": "Strings", "fx": "FX",
    "percussion": "Perc", "bass": "Bass", "kick": "Kick",
    "snare": "Snare", "unknown": "Sound",
}
