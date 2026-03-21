"""
RESONATE v7.0 — AI-Powered Sample Matching Engine
Cyanite.ai for real audio analysis, Essentia for local measurement,
genre reference profiles for deterministic scoring.
"""

import os
import re
import json
import hashlib
import threading
import time as _time
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import essentia.standard as es
import librosa
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

# ── Optional: Cyanite.ai client (real audio AI) ─────────────────────────────
CYANITE_TOKEN = os.environ.get("CYANITE_API_KEY", "")
HAS_CYANITE = bool(CYANITE_TOKEN)
if HAS_CYANITE:
    print("✓ Cyanite.ai connected (real audio AI)")
else:
    print("✗ Cyanite.ai unavailable — set CYANITE_API_KEY")

# ── Optional: Anthropic client (text AI fallback) ───────────────────────────
try:
    import anthropic
    claude_client = anthropic.Anthropic()
    HAS_CLAUDE = True
    print("✓ Claude AI connected (text analysis fallback)")
except Exception:
    claude_client = None
    HAS_CLAUDE = False
    print("  Claude AI unavailable — Essentia + genre profiles only")

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="RESONATE")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_DIR = Path(__file__).parent / "samples"
SAMPLE_DIR.mkdir(exist_ok=True)
TRANSPOSED_DIR = Path(__file__).parent / "transposed_cache"
TRANSPOSED_DIR.mkdir(exist_ok=True)

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}
SAMPLE_INDEX_FILE = Path(__file__).parent / "sample_index.json"

# ── Global state ────────────────────────────────────────────────────────────
latest_track_profile = {}
latest_ai_analysis = {}
latest_track_file = None  # path to uploaded track for dual playback
sample_cache = {}          # in-memory cache (filepath -> analysis)
indexing_status = {"done": False, "total": 0, "processed": 0}

# ═════════════════════════════════════════════════════════════════════════════
# GENRE REFERENCE PROFILES — based on measured frequency analysis of
# professional masters across decades of hit records.
# Sources: mix engineering references, mastering house EQ targets,
# published spectral analysis of Billboard #1 records.
#
# Each profile defines:
#   freq_balance: ideal energy distribution across 7 bands (sums to ~1.0)
#   type_needs: what sample types a track in this genre typically needs (0-100)
#   spectral_centroid_target: where the "center of gravity" should sit (Hz)
#   dynamic_range: expected loudness variance
# ═════════════════════════════════════════════════════════════════════════════

GENRE_PROFILES = {
    "trap/hip-hop": {
        "freq_balance": {
            "sub_bass_20_80": 0.22,      # heavy 808s
            "bass_80_250": 0.18,          # bass body
            "low_mid_250_500": 0.12,      # warmth, needs room for 808 harmonics
            "mid_500_2k": 0.18,           # melody, vocals — often the gap
            "upper_mid_2k_6k": 0.14,      # vocal presence, lead bite
            "presence_6k_12k": 0.10,      # hi-hats, crisp percussion
            "air_12k_20k": 0.06,          # air, shimmer
        },
        "type_needs": {
            "melody": 95, "vocals": 90, "hihat": 80, "pad": 60,
            "strings": 50, "fx": 45, "percussion": 55,
            "bass": 5, "kick": 5, "snare": 15, "unknown": 30,
        },
        "spectral_centroid_target": 2200,  # Hz — darker than pop
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
    # Default fallback
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


def load_disk_cache():
    """Load persistent sample index from disk."""
    global sample_cache
    if SAMPLE_INDEX_FILE.exists():
        try:
            data = json.loads(SAMPLE_INDEX_FILE.read_text())
            sample_cache = data
            print(f"  ✓ Loaded {len(sample_cache)} samples from disk cache")
        except Exception as e:
            print(f"  ✗ Cache load error: {e}")
            sample_cache = {}


def save_disk_cache():
    """Save sample index to disk."""
    try:
        SAMPLE_INDEX_FILE.write_text(json.dumps(sample_cache))
    except Exception as e:
        print(f"  ✗ Cache save error: {e}")


def file_hash(filepath):
    """Quick hash based on filename + size + mtime (no content read)."""
    st = filepath.stat()
    return f"{filepath.name}:{st.st_size}:{int(st.st_mtime)}"


def auto_organize_samples():
    """
    Auto-categorize all samples into proper subfolders based on filename analysis.
    Runs once on startup. Only moves files that are in the root or a generic folder.
    """
    # Map internal types to folder names
    TYPE_FOLDERS = {
        "melody": "Melody", "vocals": "Vocals", "hihat": "Hi-Hats",
        "pad": "Pads", "strings": "Strings", "fx": "FX",
        "percussion": "Percussion", "bass": "Bass", "kick": "Kick",
        "snare": "Snare", "unknown": "Other",
    }

    # Known generic folder names that should be reorganized
    GENERIC_FOLDERS = {"sounds", "packs", "samples", "audio", "all", "misc", "uncategorized", "other"}

    # Folders that are already properly organized — skip these
    ORGANIZED_FOLDERS = {v.lower() for v in TYPE_FOLDERS.values()}
    ORGANIZED_FOLDERS.update({"guitar", "synth", "piano", "vocals", "hi-hats",
                              "hihats", "clap", "drums", "leads", "fillers"})

    all_files = [f for f in SAMPLE_DIR.rglob("*") if f.suffix.lower() in AUDIO_EXT and f.is_file()]
    if not all_files:
        return

    # Check if samples are already organized
    needs_organizing = False
    for fp in all_files[:50]:  # Check first 50
        try:
            rel = fp.relative_to(SAMPLE_DIR)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) <= 1:
            # File is in root of samples/
            needs_organizing = True
            break
        parent_folder = parts[0].lower()
        if parent_folder in GENERIC_FOLDERS:
            needs_organizing = True
            break

    if not needs_organizing:
        print(f"  Samples already organized ({len(all_files)} files)")
        return

    print(f"  Auto-organizing {len(all_files)} samples into category folders...")
    moved = 0
    for fp in all_files:
        try:
            # Classify using the existing engine
            stype = classify_type(fp)

            # Refine: distinguish guitar/synth/piano within "melody"
            name_lower = fp.stem.lower()
            if stype == "melody":
                if "guitar" in name_lower:
                    folder = "Guitar"
                elif "piano" in name_lower or "keys" in name_lower or "rhodes" in name_lower:
                    folder = "Piano"
                elif "synth" in name_lower:
                    folder = "Synth"
                elif "lead" in name_lower:
                    folder = "Leads"
                elif "flute" in name_lower or "sax" in name_lower or "brass" in name_lower or "horn" in name_lower:
                    folder = "Brass-Wind"
                else:
                    folder = "Melody"
            elif stype == "snare" and "clap" in name_lower:
                folder = "Clap"
            elif stype == "snare":
                folder = "Snare"
            else:
                folder = TYPE_FOLDERS.get(stype, "Other")

            # Check if file is already in the right folder
            try:
                rel = fp.relative_to(SAMPLE_DIR)
            except ValueError:
                continue
            current_parent = rel.parts[0] if len(rel.parts) > 1 else None

            if current_parent and current_parent == folder:
                continue  # Already in correct folder

            # Don't move if already in a proper organized folder
            if current_parent and current_parent.lower() in ORGANIZED_FOLDERS:
                continue

            # Create target folder and move
            target_dir = SAMPLE_DIR / folder
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / fp.name

            # Handle name conflicts
            if target_path.exists():
                stem = fp.stem
                suffix = fp.suffix
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            import shutil
            shutil.move(str(fp), str(target_path))
            moved += 1
        except Exception as e:
            print(f"    Error moving {fp.name}: {e}")

    # Clean up empty directories
    for d in sorted(SAMPLE_DIR.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()  # Only removes if empty
            except OSError:
                pass

    if moved > 0:
        print(f"  ✓ Organized {moved} samples into category folders")
        # Clear disk cache since paths changed
        if SAMPLE_INDEX_FILE.exists():
            SAMPLE_INDEX_FILE.unlink()
            print(f"  Cache cleared (paths changed)")
    else:
        print(f"  No samples needed moving")


def background_index():
    """Index all samples in background thread on startup."""
    global sample_cache, indexing_status

    all_files = sorted(
        f for f in SAMPLE_DIR.rglob("*")
        if f.suffix.lower() in AUDIO_EXT and f.is_file()
    )
    indexing_status["total"] = len(all_files)

    # Check which files need (re-)analysis
    new_count = 0
    cached_count = 0
    for i, fp in enumerate(all_files):
        cache_key = str(fp)
        fh = file_hash(fp)

        # Skip if already cached with same hash
        existing = sample_cache.get(cache_key)
        if existing and existing.get("_hash") == fh:
            cached_count += 1
            indexing_status["processed"] = i + 1
            continue

        # Analyze and cache
        try:
            sa = analyze_sample(fp)
            if sa:
                sa["_hash"] = fh
                sample_cache[cache_key] = sa
                new_count += 1
        except Exception as e:
            print(f"  Index error {fp.name}: {e}")

        indexing_status["processed"] = i + 1

        # Save to disk every 50 samples
        if new_count > 0 and new_count % 50 == 0:
            save_disk_cache()

    # Final save
    if new_count > 0:
        save_disk_cache()

    indexing_status["done"] = True
    print(f"\n  ✓ Indexing complete: {cached_count} cached, {new_count} new, {len(all_files)} total\n")

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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: FILENAME PARSING
# ═════════════════════════════════════════════════════════════════════════════

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
    # Try explicit patterns first
    matches = list(KEY_PATTERN.finditer(stem))
    if not matches:
        return None
    # Take the last match (usually most specific)
    m = matches[-1]
    note = m.group(1)
    # Uppercase first letter only, preserve 'b' flat indicator
    note = note[0].upper() + note[1:] if len(note) > 1 else note.upper()
    # Normalize flats
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
    # Try suffix pattern first (e.g. "100BPM", "100 bpm")
    for m in BPM_PATTERN_SUFFIX.finditer(stem):
        val = int(m.group(1))
        if 60 <= val <= 200:
            return float(val)
    # Then try delimiter pattern (e.g. "_100_")
    for m in BPM_PATTERN.finditer(stem):
        val = int(m.group(1))
        if 60 <= val <= 200:
            return float(val)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

PERC_FOLDER_WORDS = {
    "drums", "drum", "percussion", "percs", "perc", "one-shots", "oneshots",
    "kicks", "kick", "snares", "snare", "claps", "clap", "hihats", "hihat",
    "hi-hats", "cymbals", "shakers",
}

# Tokens that are unambiguously percussion
PERC_TOKENS_STRONG = {
    "kick", "snare", "clap", "hihat", "shaker", "rim",
    "cymbal", "ride", "cowbell", "tambourine",
    "conga", "bongo", "percussion", "hh",
}

# Tokens that are ambiguous (e.g. "crash" in "CrashYourCar")
PERC_TOKENS_WEAK = {"crash", "drum", "hat", "tom", "perc", "snap", "stomp", "click"}


def tokenize_name(name):
    """Split filename into tokens, handling camelCase and delimiters."""
    # Split on delimiters
    parts = re.split(r'[_\-\s]+', name)
    # Split camelCase within each part
    expanded = []
    for part in parts:
        sub = re.sub(r'([a-z])([A-Z])', r'\1_\2', part).lower().split('_')
        expanded.extend(sub)
    return expanded


def is_percussion(filepath):
    """Determine if a file is a percussion sample."""
    folder_parts = [p.lower() for p in Path(filepath).parts[:-1]]
    # Check folder names
    for p in folder_parts:
        if p in PERC_FOLDER_WORDS:
            return True

    tokens = tokenize_name(Path(filepath).stem)

    # Strong percussion tokens: always percussion
    for t in tokens:
        if t in PERC_TOKENS_STRONG:
            return True

    # Weak tokens: only if also in a drums folder
    for t in tokens:
        if t in PERC_TOKENS_WEAK:
            for p in folder_parts:
                if p in PERC_FOLDER_WORDS:
                    return True

    return False


TYPE_MAP = {
    # Exact token matches (highest priority)
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

    # Exact token match first
    for token in all_tokens:
        if token in TYPE_MAP:
            return TYPE_MAP[token]

    # Longer keyword substring match
    for kw, t in TYPE_MAP.items():
        if len(kw) > 3 and kw in name:
            return t

    return "unknown"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: AUDIO ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

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
        # Look for kick presence
        try:
            rms = float(es.RMS()(audio))
            if rms > 0.05 and sb > 0.08:
                detected_instruments.append("kick")
        except Exception:
            pass
        # Snare presence via mid energy transients
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

    # ── Heuristic genre detection (no Claude needed) ──
    # Sub-bass energy is the strongest indicator for trap/hip-hop
    # Trap can be 70-170 BPM (half-time bounce to fast trap)
    detected_genre = "default"
    sb_energy = freq_bands.get("sub_bass_20_80", 0) if freq_bands else 0
    bass_energy = freq_bands.get("bass_80_250", 0) if freq_bands else 0
    mid_energy = freq_bands.get("mid_500_2k", 0) if freq_bands else 0
    hi_energy = freq_bands.get("presence_6k_12k", 0) if freq_bands else 0
    total_low = sb_energy + bass_energy

    # Sub-bass dominant = trap/hip-hop (any BPM — half-time trap is 70-100)
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
        "frequency_bands": freq_bands,
        "mfcc_profile": [round(x, 2) for x in mfcc_profile],
        "spectral_contrast": [round(x, 2) for x in spectral_contrast],
        "detected_instruments": detected_instruments,
        "frequency_gaps": frequency_gaps,
        "detected_genre": detected_genre,
    }


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

    print(f"  {fname}: key={key_string} ({key_src}), bpm={bpm_val} ({bpm_src}), type={sample_type}")

    return {
        "duration": round(duration, 2), "bpm": bpm_val,
        "key": normalize_key(key_string),
        "rms": round(rms, 4), "spectral_centroid": round(sc_val, 4),
        "mfcc_profile": [round(x, 2) for x in mfcc_profile],
        "frequency_bands": freq_bands,
        "sample_type": sample_type,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: SCORING — MUSIC THEORY + TRANSPOSITION AWARE
# ═════════════════════════════════════════════════════════════════════════════

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
        return 75  # one-shots: usually fine
    ratio = to_bpm / from_bpm
    # Check half-time and double-time
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


def timbre_compat(track_mfcc, sample_mfcc):
    """Compare timbral similarity via MFCC cosine distance.
    Higher = sample sounds like it belongs with the track."""
    if not track_mfcc or not sample_mfcc:
        return 50
    a = np.array(track_mfcc)
    b = np.array(sample_mfcc)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 50
    cos = float(np.dot(a, b) / (na * nb))
    # We want samples that sound COMPATIBLE with the track
    # High cosine similarity = similar timbral character = good
    # But not identical (that's boring) — sweet spot is 0.4-0.85
    if cos > 0.85:
        return 75   # very similar — slightly boring but compatible
    elif cos > 0.6:
        return 95   # sweet spot — related but adds something
    elif cos > 0.4:
        return 80   # complementary
    elif cos > 0.2:
        return 55   # getting foreign
    elif cos > 0.0:
        return 30   # quite different sonic world
    else:
        return 15   # completely different character


def sonic_genre_compat(track, sample, sample_filepath, genre_str):
    """
    Score how well a sample's SONIC CHARACTER fits the track's genre.
    Uses measured audio features + filename genre clues.
    
    KEY FIX: For "neutral" samples (no genre keywords in filename),
    we INFER genre from audio characteristics. A bright 128bpm synth
    is almost certainly EDM even if the filename doesn't say "house".
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

    # ── 2. AUDIO-BASED genre inference for "neutral" samples ──
    # This is the key fix — detect EDM/house-sounding samples even without keywords
    if sample_genre == "neutral":
        sample_bpm = sample.get("bpm", 0)
        s_air = sample_bands.get("air_12k_20k", 0)
        s_presence = sample_bands.get("presence_6k_12k", 0)
        s_sub = sample_bands.get("sub_bass_20_80", 0)
        s_bass = sample_bands.get("bass_80_250", 0)
        s_brightness = s_air + s_presence
        s_darkness = s_sub + s_bass
        
        # Bright + 125-132 BPM = almost certainly EDM/house
        if sample_centroid > 3200 and 122 <= sample_bpm <= 135:
            sample_genre = "edm_inferred"
        # Very bright with lots of air = EDM-like
        elif sample_centroid > 3800 and s_brightness > 0.15:
            sample_genre = "edm_inferred"
        # Bright + 4-on-the-floor tempo range
        elif sample_centroid > 2800 and 118 <= sample_bpm <= 138 and s_brightness > s_darkness * 1.5:
            sample_genre = "edm_inferred"
        # Dark + heavy sub/bass + trap tempo = hip-hop-like
        elif sample_centroid < 1800 and s_darkness > 0.12:
            sample_genre = "hiphop_inferred"
        # Warm mids, moderate darkness = hip-hop compatible
        elif sample_centroid < 2200 and s_darkness > s_brightness * 1.3:
            sample_genre = "hiphop_inferred"
    
    # ── 3. Apply genre match/mismatch ──
    if track_is_hiphop:
        if sample_genre in ("hiphop", "hiphop_inferred"):
            score += 30
        elif sample_genre == "edm":
            score -= 40   # explicit EDM label — hard penalty
        elif sample_genre == "edm_inferred":
            score -= 32   # sounds EDM — strong penalty
        elif sample_genre == "rock":
            score -= 38
        # "neutral" samples that didn't trigger audio inference = mild trust
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
            score += 20  # very dark — trap perfect
        elif sample_centroid < 1800:
            score += 14  # dark/warm — great
        elif sample_centroid < 2500:
            score += 5   # neutral warmth — ok
        elif sample_centroid < 3500:
            score -= 10  # getting bright — not ideal
        elif sample_centroid < 4500:
            score -= 20  # bright — EDM territory
        else:
            score -= 28  # extremely bright — definitely wrong genre
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
                score += 10  # warm-leaning — perfect for hip-hop
            elif s_high > s_low * 2.0:
                score -= 18  # way too bright/airy
        elif track_is_edm:
            if s_high > s_low: score += 8
    
    # ── 6. BPM origin (strong genre signal) ──
    sample_bpm = sample.get("bpm", 0)
    if sample_bpm > 0:
        if track_is_hiphop:
            if 122 <= sample_bpm <= 132:
                score -= 15  # house BPM range
            if 65 <= sample_bpm <= 100:
                score += 8   # trap half-time range
        elif track_is_edm:
            if 120 <= sample_bpm <= 135: score += 5
    
    return round(max(0, min(100, score)), 1)


def frequency_complement(track_bands, sample_bands):
    """Score how well sample fills frequency gaps in track."""
    if not track_bands or not sample_bands:
        return 50
    score = 50
    for band, track_level in track_bands.items():
        sample_level = sample_bands.get(band, 0)
        if track_level < 0.03 and sample_level > 0.05:
            score += 12  # sample fills a gap
        elif track_level > 0.20 and sample_level > 0.20:
            score -= 8   # both heavy in same band = clash
    return max(0, min(100, score))


def math_match(track, sample, sample_filepath=None, ai_template=None):
    """
    Deterministic scoring based on actual audio measurements + genre science.

    Scoring components (rebalanced for genre accuracy):
    1. Type Priority (18%) — genre-calibrated need for this sample type
    2. Sonic Genre Compatibility (22%) — does sample SOUND like it belongs in this genre?
    3. Frequency Deficit Filling (18%) — does sample fill measured gaps vs ideal?
    4. Spectral Placement (12%) — is sample's energy where the track needs it?
    5. Timbre Match (12%) — timbral compatibility (NOT contrast)
    6. Musical Content Value (8%) — longer harmonic content > short transients
    7. Transpose Quality (5%) — how clean will the pitch shift sound?
    8. Time-Stretch Quality (5%) — how clean will the tempo change sound?
    """
    tk = track.get("key", "C")
    sk = sample.get("key", "—")
    stype = classify_type(Path(sample_filepath)) if sample_filepath else "unknown"
    duration = sample.get("duration", 0)
    track_bands = track.get("frequency_bands", {})
    sample_bands = sample.get("frequency_bands", {})

    # Get the genre reference profile
    genre_str = ""
    if ai_template:
        genre_str = ai_template.get("genre", "")
    if not genre_str and track:
        genre_str = track.get("detected_genre", "default")
    genre_ref = get_genre_profile(genre_str)

    # ── 1. TYPE PRIORITY (18%) ──
    # Start with genre reference (what types do hit records in this genre need?)
    # Then refine with Claude's template if available
    ref_type_needs = genre_ref["type_needs"]
    type_score = ref_type_needs.get(stype, 25)

    # If Claude's template has more specific priorities, blend them in
    if ai_template:
        ai_type = ai_template.get("type_priority_scores", {})
        if stype in ai_type:
            ai_val = float(ai_type[stype])
            # 60% genre reference, 40% Claude refinement
            type_score = ref_type_needs.get(stype, 25) * 0.6 + ai_val * 0.4

    # ── 2. FREQUENCY DEFICIT FILLING (22%) ──
    # Compare track's current frequency balance to genre ideal
    # Score samples that fill the biggest deficits highest
    deficit_score = 50
    ideal_balance = genre_ref["freq_balance"]
    if ai_template and ai_template.get("ideal_frequency_balance"):
        # Blend genre reference with Claude's ideal (70/30)
        ai_ideal = ai_template["ideal_frequency_balance"]
        ideal_balance = {}
        for band in genre_ref["freq_balance"]:
            ref_val = genre_ref["freq_balance"].get(band, 0.1)
            ai_val = ai_ideal.get(band, ref_val)
            ideal_balance[band] = ref_val * 0.7 + ai_val * 0.3

    if track_bands and sample_bands:
        # Calculate deficit for each band
        deficit_fill = 0
        total_deficit_weight = 0

        for band, ideal_val in ideal_balance.items():
            track_val = track_bands.get(band, 0)
            sample_val = sample_bands.get(band, 0)

            # How much is the track missing vs the ideal?
            deficit = max(0, ideal_val - track_val)
            # How much would this sample help?
            contribution = min(sample_val, deficit) if deficit > 0 else 0

            # Weight by deficit size — bigger gaps matter more
            deficit_weight = deficit * 10 + 0.1  # avoid zero weights
            total_deficit_weight += deficit_weight

            if deficit > 0.03 and sample_val > 0.03:
                # Sample has energy right where the track needs it
                fill_pct = min(contribution / max(deficit, 0.01), 1.5)
                deficit_fill += fill_pct * 100 * deficit_weight
            elif deficit < 0.01 and sample_val > 0.12:
                # Track doesn't need this band, sample is heavy here — clash
                deficit_fill += 15 * deficit_weight
            elif sample_val > 0.01:
                deficit_fill += 30 * deficit_weight
            else:
                deficit_fill += 20 * deficit_weight

        deficit_score = deficit_fill / max(total_deficit_weight, 1)
        deficit_score = min(100, max(0, deficit_score))

    # ── 3. SPECTRAL PLACEMENT (15%) ──
    # Is the sample's spectral centroid in the range the track needs?
    # A sample centered at 1.5kHz fills the midrange gap; one at 60Hz doesn't
    spectral_score = 50
    sample_centroid = sample.get("spectral_centroid", 0)
    if sample_centroid > 0 and track_bands:
        # Find which frequency bands have the biggest deficits
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
            # How close is the sample's spectral centroid to the biggest gap?
            if target_center > 0:
                ratio = sample_centroid / target_center
                if 0.5 <= ratio <= 2.0:
                    spectral_score = 90  # right in the target zone
                elif 0.3 <= ratio <= 3.0:
                    spectral_score = 65  # nearby
                else:
                    spectral_score = 30  # far from where it's needed

    # ── 4. MUSICAL CONTENT VALUE (13%) ──
    # Longer melodic/harmonic content adds more production value
    content_score = 35
    if duration > 5 and stype in ("melody", "vocals", "pad", "strings"):
        content_score = 98  # long melodic loop — maximum value
    elif duration > 2 and stype in ("melody", "vocals", "pad", "strings"):
        content_score = 88  # medium melodic — excellent
    elif duration > 1 and stype in ("melody", "vocals"):
        content_score = 72  # short melodic — still great
    elif duration > 2 and stype in ("hihat", "percussion"):
        content_score = 62  # percussion loop — useful
    elif duration > 0.5 and stype in ("hihat", "percussion", "fx"):
        content_score = 45  # short perc/fx
    elif duration < 0.3 and stype in ("snare", "kick", "hihat"):
        content_score = 18  # tiny one-shot
    elif duration < 0.15:
        content_score = 10  # micro transient

    # ── 5. TIMBRE MATCH (12%) ──
    timbre_score = timbre_compat(track.get("mfcc_profile"), sample.get("mfcc_profile"))

    # ── 5b. SONIC GENRE COMPATIBILITY (22%) ──
    genre_compat_score = sonic_genre_compat(track, sample, sample_filepath, genre_str)

    # ── 6. TRANSPOSE QUALITY (5%) ──
    if sk in ("N/A", "—"):
        transpose_score = 88
    else:
        semis = semitones_to_transpose(sk, tk)
        transpose_score = transpose_quality(semis)

    # ── 7. TIME-STRETCH QUALITY (5%) ──
    stretch_score = timestretch_quality(sample.get("bpm", 0), track.get("bpm", 120))

    # ── COMBINE ──
    final = (
        type_score * 0.10 +
        genre_compat_score * 0.36 +
        deficit_score * 0.14 +
        spectral_score * 0.10 +
        timbre_score * 0.14 +
        content_score * 0.06 +
        transpose_score * 0.05 +
        stretch_score * 0.05
    )

    return round(max(0, min(100, final)), 1)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: AI INTELLIGENCE (Cyanite.ai primary, Claude fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _cyanite_graphql(query, variables=None):
    """Send a GraphQL request to Cyanite.ai API."""
    import urllib.request
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.cyanite.ai/graphql",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CYANITE_TOKEN}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.request.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else "no body"
        print(f"  Cyanite API error {e.code}: {body[:500]}")
        raise


def _convert_to_mp3(wav_path):
    """Convert WAV to MP3 for Cyanite (which only accepts MP3)."""
    mp3_path = str(wav_path).rsplit(".", 1)[0] + "_cyanite.mp3"
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(wav_path))
        audio.export(mp3_path, format="mp3", bitrate="320k")
        return mp3_path
    except Exception:
        pass
    # Fallback: try ffmpeg directly
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "320k", mp3_path],
            capture_output=True, timeout=60,
        )
        if Path(mp3_path).exists():
            return mp3_path
    except Exception:
        pass
    return None


def cyanite_analyze_track(filepath):
    """Upload track to Cyanite.ai for real AI audio analysis."""
    if not HAS_CYANITE:
        return None

    import urllib.request

    print("  Uploading to Cyanite.ai for real audio analysis...")

    # Clean up old library tracks first (free tier has a limit)
    print("  Checking Cyanite library space...")
    try:
        total_deleted = 0
        for attempt in range(5):
            # Use cursor-based pagination correctly
            old_tracks = _cyanite_graphql("""
                query LibraryCleanup {
                    libraryTracks {
                        edges { node { id } }
                        pageInfo { hasNextPage }
                    }
                }
            """)
            data = old_tracks.get("data")
            if not data or not data.get("libraryTracks"):
                print(f"  Library query result: {json.dumps(old_tracks)[:300]}")
                break
            edges = data["libraryTracks"].get("edges", [])
            if not edges:
                break
            old_ids = [e["node"]["id"] for e in edges[:100]]
            print(f"  Deleting {len(old_ids)} old Cyanite tracks (round {attempt + 1})...")
            del_result = _cyanite_graphql("""
                mutation Delete($input: LibraryTracksDeleteInput!) {
                    libraryTracksDelete(input: $input) {
                        __typename
                        ... on LibraryTracksDeleteSuccess { __typename }
                        ... on Error { message }
                    }
                }
            """, {"input": {"libraryTrackIds": old_ids}})
            print(f"  Delete result: {del_result.get('data', {})}")
            total_deleted += len(old_ids)
            _time.sleep(1)
        if total_deleted > 0:
            print(f"  ✓ Cleaned up {total_deleted} old tracks")
            _time.sleep(2)  # Give Cyanite a moment to process deletions
    except Exception as e:
        print(f"  Cleanup warning: {e}")

    # Step 1: Convert to MP3 if needed
    fpath = str(filepath)
    if not fpath.lower().endswith(".mp3"):
        print("  Converting to MP3 for Cyanite...")
        mp3_path = _convert_to_mp3(fpath)
        if not mp3_path:
            print("  ✗ MP3 conversion failed")
            return None
        fpath = mp3_path
    else:
        mp3_path = None

    track_id = None
    try:
        # Step 2: Request file upload URL
        print("  Requesting upload URL...")
        result = _cyanite_graphql("""
            mutation { fileUploadRequest { id uploadUrl } }
        """)
        upload_info = result["data"]["fileUploadRequest"]
        file_id = upload_info["id"]
        upload_url = upload_info["uploadUrl"]

        # Step 3: Upload MP3 to S3
        print("  Uploading audio to Cyanite...")
        with open(fpath, "rb") as f:
            audio_data = f.read()
        req = urllib.request.Request(
            upload_url,
            data=audio_data,
            headers={"Content-Type": "audio/mpeg"},
            method="PUT",
        )
        urllib.request.urlopen(req, timeout=120)

        # Step 4: Create library track
        print("  Creating library track...")
        result = _cyanite_graphql("""
            mutation CreateTrack($input: LibraryTrackCreateInput!) {
                libraryTrackCreate(input: $input) {
                    __typename
                    ... on LibraryTrackCreateSuccess {
                        createdLibraryTrack { id }
                    }
                    ... on Error { message }
                }
            }
        """, {"input": {"uploadId": file_id, "title": Path(filepath).stem}})

        create_data = result["data"]["libraryTrackCreate"]
        if create_data["__typename"] != "LibraryTrackCreateSuccess":
            print(f"  ✗ Cyanite create error: {create_data.get('message', 'unknown')}")
            return None

        track_id = create_data["createdLibraryTrack"]["id"]
        print(f"  Track created: {track_id}")

        # Step 5: Enqueue analysis (may auto-enqueue on creation)
        print("  Enqueuing analysis...")
        try:
            enqueue_result = _cyanite_graphql("""
                mutation Enqueue($input: LibraryTrackEnqueueInput!) {
                    libraryTrackEnqueue(input: $input) {
                        __typename
                        ... on LibraryTrackEnqueueSuccess { success }
                        ... on Error { message }
                    }
                }
            """, {"input": {"libraryTrackId": track_id}})
            enqueue_data = enqueue_result.get("data", {}).get("libraryTrackEnqueue", {})
            print(f"  Enqueue result: {enqueue_data.get('__typename', 'unknown')}")
            if enqueue_data.get("message"):
                print(f"  Enqueue message: {enqueue_data['message']}")
        except Exception as e:
            print(f"  Enqueue error (may already be queued): {e}")

        # Step 6: Poll for results (typically ~10 seconds)
        print("  Waiting for Cyanite analysis", end="", flush=True)
        analysis_query = """
            query Track($id: ID!) {
                libraryTrack(id: $id) {
                    __typename
                    ... on LibraryTrack {
                        id
                        audioAnalysisV6 {
                            __typename
                            ... on AudioAnalysisV6Finished {
                                result {
                                    bpmPrediction { value }
                                    keyPrediction { value }
                                    energyLevel
                                    moodTags
                                    genreTags
                                    voice { female male instrumental }
                                }
                            }
                            ... on AudioAnalysisV6Enqueued { __typename }
                            ... on AudioAnalysisV6Processing { __typename }
                            ... on AudioAnalysisV6Failed { error { message } }
                        }
                        audioAnalysisV7 {
                            __typename
                            ... on AudioAnalysisV7Finished {
                                result {
                                    advancedGenreTags
                                    advancedSubgenreTags
                                    advancedInstrumentTags
                                    freeGenreTags
                                }
                            }
                            ... on AudioAnalysisV7Enqueued { __typename }
                            ... on AudioAnalysisV7Processing { __typename }
                            ... on AudioAnalysisV7Failed { error { message } }
                        }
                    }
                    ... on Error { message }
                }
            }
        """

        max_polls = 30  # 30 * 3s = 90 seconds max
        for i in range(max_polls):
            _time.sleep(3)
            print(".", end="", flush=True)
            result = _cyanite_graphql(analysis_query, {"id": track_id})

            track_data = result.get("data", {}).get("libraryTrack", {})
            if track_data.get("__typename") != "LibraryTrack":
                continue

            v6 = track_data.get("audioAnalysisV6", {})
            v7 = track_data.get("audioAnalysisV7", {})

            # Check if V6 is done (V7 may still be processing)
            v6_done = v6.get("__typename") == "AudioAnalysisV6Finished"
            v7_done = v7.get("__typename") == "AudioAnalysisV7Finished"

            if v6_done:
                print(" ✓")
                r6 = v6["result"]
                r7 = v7.get("result", {}) if v7_done else {}

                # V6 basics
                cyanite_bpm = r6.get("bpmPrediction", {}).get("value")
                cyanite_key = r6.get("keyPrediction", {}).get("value")
                energy_level = r6.get("energyLevel")
                mood_tags = r6.get("moodTags", [])
                v6_genre_tags = r6.get("genreTags", [])
                voice = r6.get("voice", {})

                # V7 advanced (if available)
                genre_tags = r7.get("advancedGenreTags", v6_genre_tags)
                subgenre_tags = r7.get("advancedSubgenreTags", [])
                instrument_tags = r7.get("advancedInstrumentTags", [])
                free_genres = r7.get("freeGenreTags", [])

                genre_str = genre_tags[0] if genre_tags else "unknown"
                subgenre_str = subgenre_tags[0] if subgenre_tags else ""
                mood_str = ", ".join(mood_tags[:3]) if mood_tags else "unknown"

                print(f"  Cyanite results:")
                print(f"    Key: {cyanite_key}")
                print(f"    BPM: {cyanite_bpm}")
                print(f"    Genre: {genre_str} / {subgenre_str}")
                print(f"    Free genres: {free_genres}")
                print(f"    Mood: {mood_str}")
                print(f"    Energy: {energy_level}")
                print(f"    Instruments: {instrument_tags}")
                print(f"    Voice: {voice}")

                # Build what_track_has from instrument tags
                what_has = instrument_tags[:] if instrument_tags else []

                # Build what_track_needs
                all_instruments = ["percussion", "synth", "piano", "acousticGuitar",
                                   "electricGuitar", "strings", "bass", "bassGuitar",
                                   "brass", "woodwinds"]
                what_needs = [i for i in all_instruments if i not in instrument_tags]

                # Map Cyanite genre to our genre profile key
                cyanite_to_profile = {
                    "rapHipHop": "trap/hip-hop", "trap": "trap/hip-hop",
                    "pop": "pop", "popRap": "trap/hip-hop",
                    "rnb": "r&b", "contemporaryRnB": "r&b", "neoSoul": "r&b",
                    "electronicDance": "edm/electronic", "house": "edm/electronic",
                    "techno": "edm/electronic", "techHouse": "edm/electronic",
                    "ambient": "lo-fi/chill",
                }
                mapped_genre = genre_str
                for tag in genre_tags + subgenre_tags:
                    if tag in cyanite_to_profile:
                        mapped_genre = cyanite_to_profile[tag]
                        break

                # Cross-check: if measured audio features strongly disagree
                # with Cyanite's genre, trust the measurements
                # (Cyanite struggles with instrumental beats lacking vocals)
                heuristic_genre = latest_track_profile.get("detected_genre", "")
                if heuristic_genre and heuristic_genre != "default":
                    # Heavy sub-bass = trap, regardless of what Cyanite says
                    track_sb = latest_track_profile.get("frequency_bands", {}).get("sub_bass_20_80", 0)
                    if track_sb > 0.12 and mapped_genre in ("pop", "lo-fi/chill", "default", "unknown"):
                        print(f"  Genre override: Cyanite said '{mapped_genre}' but measured sub-bass={track_sb:.2f} → {heuristic_genre}")
                        mapped_genre = heuristic_genre

                return {
                    "source": "cyanite",
                    "genre": mapped_genre,
                    "genre_raw": genre_tags,
                    "subgenre": subgenre_tags,
                    "free_genres": free_genres,
                    "mood": mood_str,
                    "mood_tags": mood_tags,
                    "energy": energy_level or "medium",
                    "what_track_has": what_has,
                    "what_track_needs": what_needs,
                    "instruments": instrument_tags,
                    "voice": voice,
                    "cyanite_key": cyanite_key,
                    "cyanite_bpm": cyanite_bpm,
                    "summary": f"{mapped_genre} track — {mood_str}. Instruments: {', '.join(what_has) or 'minimal'}. Needs: {', '.join(what_needs[:4]) or 'refinement'}.",
                }

            elif v6.get("__typename") == "AudioAnalysisV6Failed":
                err = v6.get("error", {}).get("message", "unknown")
                print(f" ✗ Failed: {err}")
                return None

        print(" ✗ Timeout waiting for Cyanite analysis")
        return None

    except Exception as e:
        print(f"\n  Cyanite error: {e}")
        return None
    finally:
        # Clean up temp MP3
        if mp3_path and Path(mp3_path).exists():
            try:
                Path(mp3_path).unlink()
            except Exception:
                pass
        # Delete the track from Cyanite library (free up space for next analysis)
        try:
            if track_id:
                _cyanite_graphql("""
                    mutation Delete($input: LibraryTracksDeleteInput!) {
                        libraryTracksDelete(input: $input) {
                            __typename
                            ... on LibraryTracksDeleteSuccess { __typename }
                            ... on Error { message }
                        }
                    }
                """, {"input": {"libraryTrackIds": [track_id]}})
                print(f"  Cleaned up Cyanite track {track_id}")
        except Exception:
            pass


def claude_analyze_track(track_profile):
    """Ask Claude to interpret the sonic fingerprint and create a scoring template."""
    if not HAS_CLAUDE:
        return None

    print("  Consulting Claude AI...")

    fp = track_profile
    detected_genre = fp.get("detected_genre", "unknown")
    genre_ref = get_genre_profile(detected_genre)

    prompt = f"""You are an expert mix engineer and music producer. Analyze this track's measured audio data and create a scoring template.

MEASURED DATA (from Essentia audio analysis — these are real measurements, not guesses):
- Key: {fp.get('key')} ({fp.get('key_confidence')}% confidence)
- BPM: {fp.get('bpm')}
- Duration: {fp.get('duration')}s
- Loudness: {fp.get('loudness')}dB
- Frequency band energy distribution (% of total):
{json.dumps(fp.get('frequency_bands', {}), indent=2)}
- Detected instruments: {fp.get('detected_instruments', [])}
- Frequency gaps (bands below threshold): {fp.get('frequency_gaps', [])}
- Heuristic genre from audio: {detected_genre}

REFERENCE: For a professional {detected_genre} track, the ideal frequency balance is:
{json.dumps(genre_ref['freq_balance'], indent=2)}

Respond with ONLY a JSON object. No explanation, no markdown fences:
{{
  "genre": "specific genre/subgenre",
  "mood": "2-3 word mood description",
  "energy": "Low/Medium/High",
  "summary": "One sentence describing what this track sounds like and what it needs",
  "what_track_has": ["list", "of", "elements", "present"],
  "what_track_needs": ["specific", "elements", "missing"],
  "avoid": ["elements", "to", "NOT", "add"],
  "type_priority_scores": {{
    "melody": 0-100,
    "vocals": 0-100,
    "hihat": 0-100,
    "pad": 0-100,
    "strings": 0-100,
    "fx": 0-100,
    "percussion": 0-100,
    "bass": 0-100,
    "kick": 0-100,
    "snare": 0-100,
    "unknown": 0-100
  }},
  "ideal_frequency_balance": {{
    "sub_bass_20_80": 0.XX,
    "bass_80_250": 0.XX,
    "low_mid_250_500": 0.XX,
    "mid_500_2k": 0.XX,
    "upper_mid_2k_6k": 0.XX,
    "presence_6k_12k": 0.XX,
    "air_12k_20k": 0.XX
  }}
}}

RULES for type_priority_scores:
- If the track ALREADY HAS an instrument type (detected_instruments), score it 5-15
- If the track DESPERATELY NEEDS a type (in frequency_gaps), score it 85-100
- Be extreme — big gaps between needed (90+) and not-needed (5-15) types
- The scores directly control sample ranking, so precision matters

RULES for ideal_frequency_balance:
- Values must sum to approximately 1.0
- Base it on what a FINISHED professional {detected_genre} track should measure like
- This track is clearly missing midrange — the ideal should reflect what needs to be added"""

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        result = json.loads(text)
        result["source"] = "claude"

        print(f"  Claude: {result.get('summary', '')[:80]}")
        print(f"  Genre: {result.get('genre', '?')}")
        print(f"  Mood: {result.get('mood', '?')}")
        print(f"  Type priorities: {result.get('type_priority_scores', {})}")
        print(f"  Track has: {result.get('what_track_has', [])}")
        print(f"  Track needs: {result.get('what_track_needs', [])}")
        return result
    except Exception as e:
        print(f"  Claude error: {e}")
        return None



# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: TRANSPOSITION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def find_sample_file(sample_path):
    """Find a sample file using multiple lookup strategies."""
    decoded = unquote(sample_path)

    # Try 1: exact path
    fp = SAMPLE_DIR / decoded
    if fp.exists() and fp.is_file():
        return fp

    # Try 2: subfolder/filename combo
    for part in Path(decoded).parts:
        test = SAMPLE_DIR / part / Path(decoded).name
        if test.exists() and test.is_file():
            return test

    # Try 3: case-insensitive in parent dir
    parent = fp.parent
    if parent.exists():
        target = fp.name.lower()
        for f in parent.iterdir():
            if f.name.lower() == target and f.is_file():
                return f

    # Try 4: exact filename anywhere
    target_name = Path(decoded).name
    for f in SAMPLE_DIR.rglob("*"):
        if f.name == target_name and f.is_file():
            return f

    # Try 5: case-insensitive filename anywhere
    target_lower = target_name.lower()
    for f in SAMPLE_DIR.rglob("*"):
        if f.name.lower() == target_lower and f.is_file():
            return f

    return None


def transpose_sample(filepath, semitones, tempo_ratio):
    """Pitch-shift and time-stretch a sample, return cached file path."""
    cache_name = f"{filepath.stem}_t{semitones}_r{int(tempo_ratio * 1000)}.wav"
    cache_path = TRANSPOSED_DIR / cache_name

    if cache_path.exists():
        return cache_path

    try:
        y, sr = librosa.load(str(filepath), sr=44100, mono=True)

        # Time-stretch first (preserves pitch)
        if abs(tempo_ratio - 1.0) > 0.01:
            y = librosa.effects.time_stretch(y, rate=tempo_ratio)

        # Then pitch-shift
        if abs(semitones) > 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

        sf.write(str(cache_path), y, sr)
        return cache_path
    except Exception as e:
        print(f"  Transpose error for {filepath.name}: {e}")
        return filepath  # fall back to original


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: AUDIO SERVING
# ═════════════════════════════════════════════════════════════════════════════

def serve_audio(filepath):
    """Serve an audio file with range request support."""
    fsize = filepath.stat().st_size
    media = "audio/wav" if filepath.suffix.lower() == ".wav" else "audio/mpeg"

    return FileResponse(
        path=str(filepath),
        media_type=media,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(fsize),
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/samples/waveform/{sample_path:path}")
async def get_waveform(sample_path: str, bars: int = 80):
    """Return waveform peak data for a sample (for visualization)."""
    fp = find_sample_file(sample_path)
    if not fp:
        raise HTTPException(status_code=404, detail="Not found")
    try:
        import essentia.standard as es
        audio = es.MonoLoader(filename=str(fp), sampleRate=22050)()
        # Downsample to requested number of bars
        chunk_size = max(1, len(audio) // bars)
        peaks = []
        for i in range(bars):
            start = i * chunk_size
            end = min(start + chunk_size, len(audio))
            if start >= len(audio):
                peaks.append(0)
            else:
                chunk = audio[start:end]
                peaks.append(float(max(abs(chunk.min()), abs(chunk.max()))))
        # Normalize to 0-1
        max_peak = max(peaks) if peaks else 1
        if max_peak > 0:
            peaks = [round(p / max_peak, 3) for p in peaks]
        return {"peaks": peaks, "bars": len(peaks)}
    except Exception as e:
        return {"peaks": [0.5] * bars, "bars": bars}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/settings")
async def get_settings():
    """Get current settings."""
    return {
        "sample_dir": str(SAMPLE_DIR),
        "sample_count": len(sample_cache),
    }


@app.post("/settings/sample-dir")
async def set_sample_dir(request: Request):
    """Update sample library directory."""
    global SAMPLE_DIR, sample_cache, indexing_status
    body = await request.json()
    new_dir = Path(body.get("path", ""))
    if not new_dir.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {new_dir}")
    SAMPLE_DIR = new_dir
    # Clear cache and re-index
    sample_cache = {}
    indexing_status = {"done": False, "processed": 0, "total": 0}
    if SAMPLE_INDEX_FILE.exists():
        SAMPLE_INDEX_FILE.unlink()
    auto_organize_samples()
    idx_thread = threading.Thread(target=background_index, daemon=True)
    idx_thread.start()
    return {"status": "ok", "sample_dir": str(SAMPLE_DIR)}


@app.get("/samples/abspath/{sample_path:path}")
async def get_sample_abspath(sample_path: str):
    """Return absolute file path for drag-to-DAW."""
    fp = find_sample_file(sample_path)
    if not fp:
        raise HTTPException(status_code=404, detail="Not found")
    return {"path": str(fp)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ai": HAS_CLAUDE,
        "indexed": indexing_status["done"],
        "index_progress": indexing_status["processed"],
        "index_total": indexing_status["total"],
    }


@app.get("/index-status")
async def index_status():
    """Check sample indexing progress."""
    return {
        "done": indexing_status["done"],
        "processed": indexing_status["processed"],
        "total": indexing_status["total"],
        "cached": len(sample_cache),
    }


@app.post("/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    """Analyze an uploaded track."""
    global latest_track_profile, latest_ai_analysis, latest_track_file

    # Save uploaded file
    upload_dir = Path(__file__).parent / "uploads"
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    # Step 1: Deep sonic fingerprint
    track = analyze_track(str(dest))
    latest_track_profile = track
    latest_track_file = dest

    # Step 2: AI analysis — Claude primary, heuristic fallback
    print("\n  [2/3] AI musical intelligence...")
    ai = None
    if HAS_CLAUDE:
        ai = claude_analyze_track(track)
    if not ai:
        print("  Using Essentia heuristic analysis only")
        ai = {
            "source": "heuristic",
            "genre": track.get("detected_genre", "unknown"),
            "mood": "unknown",
            "energy": "Medium",
            "what_track_has": track.get("detected_instruments", []),
            "what_track_needs": track.get("frequency_gaps", []),
            "summary": f"{track.get('detected_genre', 'unknown')} track at {track.get('bpm')} BPM",
        }
    latest_ai_analysis = ai

    # Step 3: Clear transposition cache (not sample analysis cache!)
    for f in TRANSPOSED_DIR.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass
    print("  Transposition cache cleared")

    print(f"\n  [3/3] Ready for sample matching")
    print(f"\n  {'=' * 40}")
    print(f"  Analysis complete!")
    print(f"  {'=' * 40}\n")

    return {
        "duration": track["duration"],
        "analysis": {
            "key": track["key"],
            "key_confidence": track["key_confidence"],
            "bpm": track["bpm"],
            "genre": (ai or {}).get("genre", None) or track.get("detected_genre", "unknown"),
            "mood": (ai or {}).get("mood", "unknown"),
            "energy_label": (ai or {}).get("energy", "Medium"),
            "what_track_has": (ai or {}).get("what_track_has", []),
            "what_track_needs": (ai or {}).get("what_track_needs", []),
            "summary": (ai or {}).get("summary", ""),
            "frequency_bands": track.get("frequency_bands", {}),
            "frequency_gaps": track.get("frequency_gaps", []),
            "detected_instruments": track.get("detected_instruments", []),
        },
    }


@app.get("/samples")
async def list_samples():
    """List all samples with match scores. Uses pre-indexed cache."""

    # Wait for background indexing to complete (first run only)
    if not indexing_status["done"]:
        import asyncio
        print("  Waiting for sample indexing to complete...")
        while not indexing_status["done"]:
            await asyncio.sleep(0.5)
        print(f"  Indexing done — {len(sample_cache)} samples ready")

    # Build sample list from cache (already indexed in background)
    all_sample_data = []
    for cache_key, sa in sample_cache.items():
        if sa.get("_hash"):  # skip non-sample entries
            fp = Path(cache_key)
            if not fp.exists():
                continue

            try:
                rel = fp.relative_to(SAMPLE_DIR)
            except ValueError:
                rel = Path(fp.name)

            parts = rel.parts
            cat = parts[0] if len(parts) > 1 else "Uncategorized"
            sub = parts[1] if len(parts) > 2 else ""

            math_score = math_match(
                latest_track_profile, sa,
                sample_filepath=str(fp),
                ai_template=latest_ai_analysis if latest_ai_analysis else None
            ) if latest_track_profile else 50

            all_sample_data.append({
                "id": cache_key,
                "name": fp.stem,
                "filename": fp.name,
                "path": str(rel).replace("#", "%23"),
                "category": cat,
                "sub_category": sub,
                "duration": sa.get("duration", 0),
                "bpm": sa.get("bpm", 0),
                "key": sa.get("key", "N/A"),
                "sample_type": sa.get("sample_type", "unknown"),
                "math_score": math_score,
                "match": math_score,
                "frequency_bands": sa.get("frequency_bands", {}),
            })

    # ── Deterministic scoring — apply duplicate hard cap ──
    if latest_track_profile and all_sample_data:
        t_start = _time.time()
        genre_str = latest_ai_analysis.get("genre", "") if latest_ai_analysis else ""
        if not genre_str:
            genre_str = latest_track_profile.get("detected_genre", "default")
        genre_ref = get_genre_profile(genre_str)
        print(f"\n  Scoring {len(all_sample_data)} samples (deterministic, measured audio)...")
        print(f"  Genre reference: {genre_str} → target centroid {genre_ref['spectral_centroid_target']}Hz")

        for s in all_sample_data:
            final = s["math_score"]

            # ── DUPLICATE HARD CAP ──
            stype = s.get("sample_type", "unknown")
            is_duplicate = False

            if latest_track_profile.get("detected_instruments"):
                detected = latest_track_profile["detected_instruments"]

                if stype == "bass" and any(
                    x in detected for x in ["sub_bass_808", "bass"]
                ):
                    is_duplicate = True
                if stype == "kick" and "kick" in detected:
                    is_duplicate = True
                if stype == "snare" and "snare_clap" in detected:
                    is_duplicate = True

            # Also use AI template: if Claude scored a type below 15
            if latest_ai_analysis and not is_duplicate:
                type_priorities = latest_ai_analysis.get("type_priority_scores", {})
                type_pri = type_priorities.get(stype, 50)
                if isinstance(type_pri, (int, float)) and type_pri <= 15:
                    is_duplicate = True

            # Also use genre profile: if genre says this type scores ≤10
            if not is_duplicate:
                dup_genre_str = latest_ai_analysis.get("genre", "") if latest_ai_analysis else ""
                if not dup_genre_str:
                    dup_genre_str = latest_track_profile.get("detected_genre", "default")
                genre_ref = get_genre_profile(dup_genre_str)
                genre_type_need = genre_ref["type_needs"].get(stype, 50)
                if genre_type_need <= 10:
                    is_duplicate = True

            if is_duplicate:
                final = min(final, 15)

            s["match"] = max(0, min(100, final))

        elapsed = round(_time.time() - t_start, 3)
        print(f"  ✓ Scoring complete in {elapsed}s (deterministic — same results every time)")

    # Sort by match score (highest first)
    all_sample_data.sort(key=lambda x: -x["match"])

    # Build response with transposed display values
    samples_response = []
    for s in all_sample_data:
        display_key = s["key"]
        display_bpm = s["bpm"]

        if latest_track_profile.get("key") and s["key"] not in ("N/A", "—"):
            display_key = latest_track_profile["key"]
        if latest_track_profile.get("bpm") and s["bpm"] and s["bpm"] > 0:
            display_bpm = latest_track_profile["bpm"]

        # Generate match reason
        def get_match_reason(s):
            stype = s.get("sample_type", "unknown")
            gaps = latest_track_profile.get("frequency_gaps", []) if latest_track_profile else []
            sc = s.get("spectral_centroid", 2000)
            sb = s.get("frequency_bands", {})
            reasons = []
            # Sonic character
            if sc < 1500:
                reasons.append("dark tone")
            elif sc < 2200:
                reasons.append("warm character")
            # Gap filling
            if stype in ("melody", "pad", "strings") and "midrange_melody" in gaps:
                reasons.append("fills midrange")
            elif stype == "vocals" and "upper_mid_presence" in gaps:
                reasons.append("adds presence")
            elif stype == "hihat" and "high_end_sparkle" in gaps:
                reasons.append("high-end sparkle")
            elif stype in ("melody", "pad") and "low_mid_warmth" in gaps:
                reasons.append("adds warmth")
            # Content
            if s.get("duration", 0) > 4 and stype in ("melody", "vocals", "pad"):
                reasons.append("rich content")
            if not reasons:
                reasons.append("spectral fit")
            return " · ".join(reasons[:2])

        def clean_name(raw_name):
            """Strip common prefixes and make readable."""
            import re
            name = raw_name
            # Remove common pack prefixes like TRKTRN_FDGSPBS_130_
            name = re.sub(r'^[A-Z0-9_]{2,20}_[A-Z0-9_]{2,20}_\d+_', '', name)
            # Remove other prefixes like AU_SHP_135_ or DS_MT3_123_
            name = re.sub(r'^[A-Z]{1,4}_[A-Z0-9]{1,8}_\d+_', '', name)
            # Remove prefixes like SS_SS_100_ or T_TTS_K_126_
            name = re.sub(r'^[A-Z]_[A-Z]{2,6}_[A-Z]?_?\d+_', '', name)
            # Remove prefixes like OS_FLINT2_100_ or VOX_YULY_128_
            name = re.sub(r'^[A-Z]{2,4}_[A-Za-z0-9]{2,12}_\d+_', '', name)
            # Remove prefixes like 01_RAW_ or 91V_
            name = re.sub(r'^\d+[A-Z]?_[A-Z]+_', '', name)
            # Replace underscores with spaces
            name = name.replace('_', ' ')
            # Clean up multiple spaces
            name = re.sub(r'\s+', ' ', name).strip()
            # Capitalize words
            if name:
                name = ' '.join(w.capitalize() if len(w) > 2 else w for w in name.split())
            return name or raw_name.replace('_', ' ')

        # Type display labels
        type_labels = {
            "melody": "Melody", "vocals": "Vocal", "hihat": "Hi-Hat",
            "pad": "Pad", "strings": "Strings", "fx": "FX",
            "percussion": "Perc", "bass": "Bass", "kick": "Kick",
            "snare": "Snare", "unknown": "Sound",
        }

        samples_response.append({
            "id": s["id"],
            "name": s["name"],
            "clean_name": clean_name(s["name"]),
            "filename": s["filename"],
            "path": s["path"],
            "category": s["category"],
            "sub_category": s["sub_category"],
            "duration": s["duration"],
            "bpm": display_bpm,
            "original_bpm": s["bpm"],
            "key": display_key,
            "original_key": s["key"],
            "match": s["match"],
            "sample_type": s.get("sample_type", "unknown"),
            "type_label": type_labels.get(s.get("sample_type", "unknown"), "Sound"),
            "match_reason": get_match_reason(s),
            "frequency_bands": s.get("frequency_bands", {}),
        })

    return {"samples": samples_response, "total": len(samples_response)}


@app.get("/track/audio")
async def get_track_audio(request: Request):
    """Serve the uploaded track audio for dual playback."""
    if not latest_track_file or not Path(latest_track_file).exists():
        raise HTTPException(status_code=404, detail="No track uploaded")
    return serve_audio(Path(latest_track_file))


# Waveform peak cache
_waveform_cache = {}

@app.get("/samples/waveform/{sample_path:path}")
async def get_waveform(sample_path: str, bars: int = 100):
    """Return waveform peaks for visualizing the actual audio shape."""
    cache_key = f"{sample_path}:{bars}"
    if cache_key in _waveform_cache:
        return JSONResponse(content=_waveform_cache[cache_key])

    fp = find_sample_file(sample_path)
    if not fp:
        raise HTTPException(status_code=404, detail="Not found")

    try:
        import essentia.standard as es
        audio = es.MonoLoader(filename=str(fp), sampleRate=22050)()
        # Compute peaks by dividing audio into N segments
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

        # Normalize to 0-1
        max_peak = max(peaks) if peaks else 1
        if max_peak > 0:
            peaks = [round(p / max_peak, 3) for p in peaks]

        result = {"peaks": peaks, "bars": bars, "duration": len(audio) / 22050}
        _waveform_cache[cache_key] = result
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"peaks": [], "error": str(e)})


@app.get("/samples/audio/{sample_path:path}")
async def get_audio(sample_path: str):
    """Serve sample audio with real-time transposition."""
    fp = find_sample_file(sample_path)

    if not fp:
        print(f"  404: {sample_path} (tried all methods)")
        raise HTTPException(status_code=404, detail=f"Not found: {sample_path}")

    # If we have an active track analysis, transpose the sample
    if latest_track_profile and latest_track_profile.get("key"):
        track_key = latest_track_profile["key"]
        track_bpm = latest_track_profile.get("bpm", 0)

        # Get sample info from cache
        sample_info = sample_cache.get(str(fp), {})
        sample_key = sample_info.get("key", "N/A")
        sample_bpm = sample_info.get("bpm", 0)

        # Calculate transposition
        semitones = 0
        if sample_key and sample_key not in ("N/A", "—"):
            semitones = semitones_to_transpose(sample_key, track_key)

        # Calculate time-stretch ratio
        tempo_ratio = 1.0
        if sample_bpm and sample_bpm > 0 and track_bpm and track_bpm > 0:
            ratio = track_bpm / sample_bpm
            candidates = [ratio, ratio * 2, ratio / 2]
            tempo_ratio = min(candidates, key=lambda r: abs(r - 1.0))

        # Only transpose if needed
        if abs(semitones) > 0 or abs(tempo_ratio - 1.0) > 0.01:
            transposed = transpose_sample(fp, semitones, tempo_ratio)
            return serve_audio(transposed)

    return serve_audio(fp)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    n_samples = sum(
        1 for f in SAMPLE_DIR.rglob("*")
        if f.suffix.lower() in AUDIO_EXT
    )
    print(f"\n{'=' * 50}")
    print(f"  RESONATE v7.0 — AI Sample Matching Engine")
    ai_status = "Claude AI" if HAS_CLAUDE else "Essentia only"
    print(f"  Samples: {n_samples} | AI: {ai_status}")
    print(f"  Transposition: ✓ (librosa)")
    print(f"{'=' * 50}\n")

    # Auto-organize samples into category folders (runs once)
    auto_organize_samples()

    # Load persistent cache from disk (instant)
    load_disk_cache()

    # Start background indexing for any new/changed samples
    idx_thread = threading.Thread(target=background_index, daemon=True)
    idx_thread.start()

    if sample_cache:
        print(f"  → Server ready immediately with {len(sample_cache)} cached samples")
        print(f"  → New samples indexing in background...\n")
    else:
        print(f"  → First run: indexing {n_samples} samples in background...")
        print(f"  → Samples will appear as they're indexed\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
