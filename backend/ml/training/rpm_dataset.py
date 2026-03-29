"""
RESONATE Production Model — Unified Training Dataset.

Combines all data sources into a single PyTorch Dataset:
  - Local samples (33k) with DSP profiles + teacher embeddings
  - FMA tracks (106k) with genre labels
  - NSynth notes (300k) with instrument labels
  - MTG-Jamendo tracks (55k) with mood/genre/instrument tags
  - Chart songs with Spotify features + preview audio

Handles:
  - On-the-fly audio loading + mel spectrogram extraction
  - Label alignment across different datasets
  - Caching of pre-computed features
  - Data augmentation (time stretch, pitch shift, noise injection)
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)

# Deterministic genre-to-ID mapping (replaces non-deterministic hash())
_GENRE_TO_ID = {
    "electronic": 0, "hip_hop": 1, "rnb_soul": 2, "pop": 3,
    "rock": 4, "jazz": 5, "country": 6, "latin": 7,
    "classical": 8, "folk_acoustic": 9, "metal": 10, "other": 11,
}


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Configuration for the unified RPM dataset."""

    # Data roots
    local_samples_dir: str = ""                    # Path to our 33k samples
    local_profiles_dir: str = ""                   # Path to pre-computed profiles (JSON)
    fma_dir: str = ""                              # Path to FMA dataset
    nsynth_dir: str = ""                            # Path to NSynth dataset
    musdb_dir: str = ""                            # Path to MUSDB18 dataset
    jamendo_dir: str = ""                          # Path to MTG-Jamendo dataset
    chart_dir: str = ""                            # Path to chart preview audio
    cache_dir: str = "~/.resonate/feature_cache"   # Pre-computed features

    # Audio parameters
    sample_rate: int = 16000         # AST requirement
    max_duration: float = 10.0       # seconds
    target_length: int = 160000      # 10s at 16kHz

    # Augmentation
    augment: bool = True
    aug_time_stretch_range: tuple = (0.9, 1.1)
    aug_pitch_shift_range: tuple = (-2, 2)   # semitones
    aug_noise_snr_range: tuple = (20, 40)    # dB
    aug_probability: float = 0.5

    # Training
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True


# ──────────────────────────────────────────────────────────────────────
# Data source wrappers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainingSample:
    """Unified representation for a single training sample across all sources."""

    filepath: str
    source: str  # "local", "fma", "nsynth", "musdb", "jamendo", "chart"

    # Labels (all optional — different sources provide different labels)
    role: Optional[int] = None
    genre_top: Optional[int] = None
    genre_sub: Optional[int] = None
    instruments: Optional[list[int]] = None       # multi-hot indices
    key: Optional[int] = None                     # 0-23
    chord: Optional[int] = None                   # 0-11
    mode: Optional[int] = None                    # 0-6
    perceptual: Optional[list[float]] = None      # 9 values
    era: Optional[int] = None                     # 0-7
    chart_potential: Optional[float] = None       # 0-1

    # Teacher embeddings (for distillation)
    clap_embedding: Optional[np.ndarray] = None   # 512d
    panns_embedding: Optional[np.ndarray] = None  # 2048d
    ast_embedding: Optional[np.ndarray] = None    # 768d


class LocalSampleLoader:
    """Load our 33k local samples with pre-computed profiles."""

    def __init__(self, samples_dir: str, profiles_dir: str):
        self.samples_dir = Path(samples_dir)
        self.profiles_dir = Path(profiles_dir)
        self._samples: list[TrainingSample] = []

    def load(self) -> list[TrainingSample]:
        """Scan profiles directory and build training samples."""
        if self._samples:
            return self._samples

        profile_files = sorted(self.profiles_dir.glob("**/*.json"))
        logger.info(f"Loading {len(profile_files)} local sample profiles...")

        # Pre-build set of existing audio files to avoid per-file stat() calls
        existing_audio: set[str] = set()
        if self.samples_dir.exists():
            for root, _dirs, files in os.walk(str(self.samples_dir)):
                for fname in files:
                    existing_audio.add(os.path.join(root, fname))
            logger.info(f"  Found {len(existing_audio)} audio files in {self.samples_dir}")

        role_map = {
            "kick": 0, "snare": 1, "clap": 2, "hat": 3, "perc": 4,
            "bass": 5, "lead": 6, "pad": 7, "fx": 8, "texture": 9, "vocal": 10,
        }

        for pf in profile_files:
            if pf.name.startswith("._"):
                continue  # skip ExFAT resource fork files
            try:
                with open(pf) as f:
                    profile = json.load(f)

                filepath = profile.get("filepath", "")
                if not filepath:
                    continue
                # Resolve relative paths against samples_dir parent
                if not os.path.isabs(filepath):
                    filepath = str(self.samples_dir.parent / filepath)
                if filepath not in existing_audio:
                    continue

                sample = TrainingSample(
                    filepath=filepath,
                    source="local",
                )

                # Role label — check both precomputed format (top-level) and
                # full profile format (inside "labels" sub-dict)
                role_str = profile.get("role")  # precomputed format
                if role_str is None:
                    labels = profile.get("labels", {})
                    role_str = labels.get("role", "unknown")
                else:
                    labels = profile.get("labels", {})
                if role_str in role_map:
                    sample.role = role_map[role_str]

                # Also use role_id directly if available (precomputed format)
                if sample.role is None and "role_id" in profile:
                    sample.role = profile["role_id"]

                # Genre affinity → top genre
                genre_affinity = labels.get("genre_affinity", {})
                if genre_affinity:
                    top_genre = max(genre_affinity, key=genre_affinity.get)
                    sample.genre_top = _GENRE_TO_ID.get(
                        top_genre.lower().replace(" ", "_").replace("-", "_"), 11
                    )

                # Perceptual descriptors
                perc = profile.get("perceptual", {})
                if perc:
                    sample.perceptual = [
                        perc.get("brightness", 0.5),
                        perc.get("warmth", 0.5),
                        perc.get("air", 0.5),
                        perc.get("punch", 0.5),
                        perc.get("body", 0.5),
                        perc.get("bite", 0.5),
                        perc.get("smoothness", 0.5),
                        perc.get("width", 0.5),
                        perc.get("depth_impression", 0.5),
                    ]

                # Teacher embeddings
                embeddings = profile.get("embeddings", {})
                if "clap_general" in embeddings and embeddings["clap_general"]:
                    sample.clap_embedding = np.array(embeddings["clap_general"], dtype=np.float32)
                if "panns_music" in embeddings and embeddings["panns_music"]:
                    sample.panns_embedding = np.array(embeddings["panns_music"], dtype=np.float32)
                if "ast_spectrogram" in embeddings and embeddings["ast_spectrogram"]:
                    sample.ast_embedding = np.array(embeddings["ast_spectrogram"], dtype=np.float32)

                self._samples.append(sample)

            except Exception as e:
                logger.debug(f"Failed to load profile {pf}: {e}")

        logger.info(f"Loaded {len(self._samples)} local samples")
        return self._samples


class FMALoader:
    """Load FMA dataset tracks with genre labels."""

    def __init__(self, fma_dir: str):
        self.fma_dir = Path(fma_dir)
        self._samples: list[TrainingSample] = []

    def load(self) -> list[TrainingSample]:
        if self._samples:
            return self._samples

        tracks_csv = self.fma_dir / "fma_metadata" / "tracks.csv"
        if not tracks_csv.exists():
            logger.warning(f"FMA tracks.csv not found at {tracks_csv}")
            return []

        try:
            import pandas as pd
            # FMA tracks.csv has multi-level headers — skip first 2 rows, use 3rd as header
            tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

            genre_col = ("track", "genre_top")
            if genre_col not in tracks.columns:
                # Try alternative column names
                logger.warning("FMA genre column not found, trying alternatives...")
                return []

            # Map top-level genres to IDs
            genre_map = {
                "Electronic": 0, "Hip-Hop": 1, "R&B": 2, "Pop": 3,
                "Rock": 4, "Jazz": 5, "Country": 6, "Latin": 7,
                "Classical": 8, "Folk": 9, "Metal": 10, "Experimental": 11,
            }

            # Pre-build set of existing files per subdirectory (avoid per-file stat calls)
            fma_audio_root = self.fma_dir / "fma_audio"
            existing_by_subdir: dict[str, set[str]] = {}
            if fma_audio_root.exists():
                for subdir_name in os.listdir(str(fma_audio_root)):
                    subdir_path = fma_audio_root / subdir_name
                    if subdir_path.is_dir():
                        existing_by_subdir[subdir_name] = set(os.listdir(str(subdir_path)))
                logger.info(f"  FMA: {len(existing_by_subdir)} subdirs on disk")

            for track_id, row in tracks.iterrows():
                try:
                    # FMA file structure: {id:06d}.mp3 in subdirectories
                    tid = int(track_id)
                    subdir = f"{tid:06d}"[:3]
                    fname = f"{tid:06d}.mp3"

                    if fname not in existing_by_subdir.get(subdir, set()):
                        continue
                    filepath = self.fma_dir / "fma_audio" / subdir / fname

                    genre_str = str(row.get(genre_col, ""))
                    genre_id = genre_map.get(genre_str, 11)  # default to "other"

                    sample = TrainingSample(
                        filepath=str(filepath),
                        source="fma",
                        genre_top=genre_id,
                    )
                    self._samples.append(sample)
                except Exception:
                    continue

        except ImportError:
            logger.warning("pandas required for FMA loading: pip install pandas")
        except Exception as e:
            logger.error(f"Failed to load FMA: {e}")

        logger.info(f"Loaded {len(self._samples)} FMA tracks")
        return self._samples


class NSynthLoader:
    """Load NSynth dataset with instrument labels."""

    # NSynth instrument families → our instrument IDs
    FAMILY_MAP = {
        "bass": [5],           # bass guitar
        "brass": [40, 41],     # trumpet, trombone
        "flute": [50],         # flute
        "guitar": [15, 16],    # acoustic, electric guitar
        "keyboard": [60, 61],  # piano, electric piano
        "mallet": [70, 71],    # marimba, vibraphone
        "organ": [65, 66],     # Hammond, pipe organ
        "reed": [45, 46],      # clarinet, saxophone
        "string": [0, 1],      # violin, cello
        "synth_lead": [80, 81],  # synth lead, synth pad
        "vocal": [90],         # vocal
    }

    def __init__(self, nsynth_dir: str):
        self.nsynth_dir = Path(nsynth_dir)
        self._samples: list[TrainingSample] = []

    def load(self) -> list[TrainingSample]:
        if self._samples:
            return self._samples

        for split in ["nsynth-train", "nsynth-valid", "nsynth-test"]:
            examples_json = self.nsynth_dir / split / "examples.json"
            audio_dir = self.nsynth_dir / split / "audio"

            if not examples_json.exists():
                continue

            try:
                with open(examples_json) as f:
                    examples = json.load(f)

                # Pre-build set of existing filenames (one readdir vs 374k stat calls)
                if audio_dir.exists():
                    existing_files = set(os.listdir(str(audio_dir)))
                    logger.info(f"  NSynth {split}: {len(existing_files)} audio files on disk, {len(examples)} in metadata")
                else:
                    existing_files = set()
                    logger.warning(f"  NSynth {split}: audio dir not found at {audio_dir}")

                for note_id, meta in examples.items():
                    fname = f"{note_id}.wav"
                    if fname not in existing_files:
                        continue
                    filepath = audio_dir / fname

                    family = meta.get("instrument_family_str", "")
                    instrument_ids = self.FAMILY_MAP.get(family, [])

                    # Create multi-hot instrument vector
                    instruments = instrument_ids if instrument_ids else None

                    sample = TrainingSample(
                        filepath=str(filepath),
                        source="nsynth",
                        instruments=instruments,
                    )

                    # NSynth provides pitch info — convert to key
                    pitch = meta.get("pitch", -1)
                    if 0 <= pitch <= 127:
                        # MIDI note to key (0-11 for major, 12-23 for minor)
                        sample.key = pitch % 12  # simplified — just pitch class

                    self._samples.append(sample)

            except Exception as e:
                logger.error(f"Failed to load NSynth split {split}: {e}")

        logger.info(f"Loaded {len(self._samples)} NSynth notes")
        return self._samples


class JamendoLoader:
    """Load MTG-Jamendo dataset with mood/genre/instrument tags."""

    def __init__(self, jamendo_dir: str):
        self.jamendo_dir = Path(jamendo_dir)
        self._samples: list[TrainingSample] = []

    def load(self) -> list[TrainingSample]:
        if self._samples:
            return self._samples

        # Collect tags per track_id first, then create one sample per track
        track_tags: dict[str, dict] = {}  # track_id -> {filepath, genre_top, ...}

        # MTG-Jamendo has TSV files mapping track IDs to tags
        for tag_file in ["autotagging_genre.tsv", "autotagging_instrument.tsv",
                         "autotagging_moodtheme.tsv"]:
            tsv_path = self.jamendo_dir / tag_file
            if tsv_path.exists():
                tag_type = tag_file.split("_")[1].split(".")[0]
                self._collect_tags(tsv_path, tag_type, track_tags)

        # Build one TrainingSample per unique track
        for track_id, info in track_tags.items():
            sample = TrainingSample(
                filepath=info["filepath"],
                source="jamendo",
            )
            if info.get("genre_top") is not None:
                sample.genre_top = info["genre_top"]
            self._samples.append(sample)

        logger.info(f"Loaded {len(self._samples)} Jamendo tracks")
        return self._samples

    def _collect_tags(self, tsv_path: Path, tag_type: str,
                      track_tags: dict[str, dict]):
        """Parse a Jamendo TSV tag file and merge into track_tags."""
        try:
            with open(tsv_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue

                    track_id = parts[0]
                    tags = parts[1:]

                    # Audio file path
                    filepath = self.jamendo_dir / "audio" / f"{track_id}.mp3"
                    if not filepath.exists():
                        # Try alternative path structures
                        subdir = track_id[:2]
                        filepath = self.jamendo_dir / "audio" / subdir / f"{track_id}.mp3"
                        if not filepath.exists():
                            continue

                    if track_id not in track_tags:
                        track_tags[track_id] = {"filepath": str(filepath)}

                    if tag_type == "genre":
                        # Map Jamendo genres to our taxonomy
                        genre_map = {
                            "electronic": 0, "hiphop": 1, "rnb": 2, "pop": 3,
                            "rock": 4, "jazz": 5, "country": 6, "latin": 7,
                            "classical": 8, "folk": 9, "metal": 10,
                        }
                        for tag in tags:
                            tag_lower = tag.lower().strip()
                            if tag_lower in genre_map:
                                track_tags[track_id]["genre_top"] = genre_map[tag_lower]
                                break

        except Exception as e:
            logger.error(f"Failed to load Jamendo tags from {tsv_path}: {e}")


class ChartPreviewLoader:
    """Load chart song previews with Spotify audio features."""

    def __init__(self, chart_dir: str):
        self.chart_dir = Path(chart_dir)
        self._samples: list[TrainingSample] = []

    def load(self) -> list[TrainingSample]:
        if self._samples:
            return self._samples

        features_db = self.chart_dir / "chart_features.db"
        if not features_db.exists():
            logger.warning(f"Chart features DB not found at {features_db}")
            return []

        try:
            import sqlite3
            conn = sqlite3.connect(str(features_db))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM chart_entries
                WHERE (preview_path IS NOT NULL AND preview_path != '')
                   OR (deezer_preview_path IS NOT NULL AND deezer_preview_path != '')
            """)

            # Map decades to era indices
            decade_map = {
                1950: 0, 1960: 1, 1970: 2, 1980: 3,
                1990: 4, 2000: 5, 2010: 6, 2020: 7,
            }

            for row in cursor:
                preview_path = row["preview_path"] or ""
                # Fallback to Deezer preview
                try:
                    deezer_path = row["deezer_preview_path"] or ""
                except (IndexError, KeyError):
                    deezer_path = ""
                if not preview_path and deezer_path:
                    preview_path = deezer_path
                if not preview_path or not os.path.exists(preview_path):
                    continue

                year = row["year"]
                decade = (year // 10) * 10
                era = decade_map.get(decade, 7)  # default to 2020s

                # Spotify key → our key encoding
                # Spotify: 0=C, 1=C#, ..., 11=B, mode: 0=minor, 1=major
                spotify_key = row["key"] if row["key"] is not None else -1
                spotify_mode = row["mode"] if row["mode"] is not None else 1
                if 0 <= spotify_key <= 11:
                    key = spotify_key + (0 if spotify_mode == 1 else 12)
                else:
                    key = None

                # Chart potential from peak position
                peak = row["peak_position"]
                chart_potential = max(0.0, 1.0 - (peak - 1) / 99.0)  # #1 → 1.0, #100 → ~0.0

                sample = TrainingSample(
                    filepath=preview_path,
                    source="chart",
                    key=key,
                    era=era,
                    chart_potential=chart_potential,
                )

                self._samples.append(sample)

            conn.close()

        except Exception as e:
            logger.error(f"Failed to load chart previews: {e}")

        logger.info(f"Loaded {len(self._samples)} chart preview samples")
        return self._samples


# ──────────────────────────────────────────────────────────────────────
# Audio loading + augmentation
# ──────────────────────────────────────────────────────────────────────

def load_audio(filepath: str, sr: int = 16000, max_duration: float = 10.0) -> np.ndarray:
    """
    Load and normalize audio to mono, target sample rate, and max duration.
    Returns numpy array of shape (num_samples,).
    """
    import librosa

    audio, _ = librosa.load(filepath, sr=sr, mono=True, duration=max_duration)

    # Pad to target length if shorter
    target_length = int(sr * max_duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    elif len(audio) > target_length:
        audio = audio[:target_length]

    return audio


def augment_audio(audio: np.ndarray, sr: int = 16000, cfg: DatasetConfig = None) -> np.ndarray:
    """
    Apply random audio augmentation.
    - Time stretching
    - Pitch shifting
    - Background noise injection
    """
    if not np.isfinite(audio).all():
        return np.zeros_like(audio)

    if cfg is None:
        cfg = DatasetConfig()

    if random.random() > cfg.aug_probability:
        return audio

    import librosa

    augmented = audio.copy()

    # Time stretch (50% chance)
    if random.random() < 0.5:
        rate = random.uniform(*cfg.aug_time_stretch_range)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)

    # Pitch shift (50% chance)
    if random.random() < 0.5:
        semitones = random.uniform(*cfg.aug_pitch_shift_range)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=semitones)

    # Add noise (30% chance)
    if random.random() < 0.3:
        snr_db = random.uniform(*cfg.aug_noise_snr_range)
        signal_power = np.mean(augmented ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(augmented))
        augmented = augmented + noise.astype(np.float32)

    # Ensure same length as original
    target_len = len(audio)
    if len(augmented) < target_len:
        augmented = np.pad(augmented, (0, target_len - len(augmented)), mode="constant")
    elif len(augmented) > target_len:
        augmented = augmented[:target_len]

    return augmented.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────

class RPMDataset(Dataset):
    """
    Unified PyTorch Dataset for RPM training.
    Loads audio on-the-fly, applies augmentation, returns features + labels.
    """

    def __init__(self, samples: list[TrainingSample], cfg: DatasetConfig = None,
                 feature_extractor=None, augment: bool = True):
        """
        Args:
            samples: list of TrainingSample from all data sources
            cfg: dataset configuration
            feature_extractor: ASTFeatureExtractor instance
            augment: whether to apply data augmentation
        """
        self.samples = samples
        self.cfg = cfg or DatasetConfig()
        self.feature_extractor = feature_extractor
        self.augment = augment

        logger.info(f"RPMDataset initialized with {len(samples)} samples")

        # Count per source
        source_counts = {}
        for s in samples:
            source_counts[s.source] = source_counts.get(s.source, 0) + 1
        for source, count in sorted(source_counts.items()):
            logger.info(f"  {source}: {count:,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        try:
            audio = load_audio(
                sample.filepath,
                sr=self.cfg.sample_rate,
                max_duration=self.cfg.max_duration,
            )
        except Exception as e:
            logger.debug(f"Failed to load {sample.filepath}: {e}")
            # Return a silent sample on failure
            audio = np.zeros(self.cfg.target_length, dtype=np.float32)

        # Augment
        if self.augment:
            audio = augment_audio(audio, sr=self.cfg.sample_rate, cfg=self.cfg)

        # Extract mel spectrogram features for AST
        if self.feature_extractor is not None:
            inputs = self.feature_extractor(
                audio, sampling_rate=self.cfg.sample_rate, return_tensors="pt"
            )
            input_values = inputs.input_values.squeeze(0)  # [T, F]
        else:
            # Fallback: raw audio tensor
            input_values = torch.from_numpy(audio)

        # Build output dict
        result = {"input_values": input_values}

        # Labels — only include what's available
        if sample.role is not None:
            result["role"] = torch.tensor(sample.role, dtype=torch.long)

        if sample.genre_top is not None:
            result["genre_top"] = torch.tensor(sample.genre_top, dtype=torch.long)

        if sample.genre_sub is not None:
            result["genre_sub"] = torch.tensor(sample.genre_sub, dtype=torch.long)

        if sample.instruments is not None:
            # Multi-hot encoding
            inst_tensor = torch.zeros(200, dtype=torch.float32)
            for idx_inst in sample.instruments:
                if 0 <= idx_inst < 200:
                    inst_tensor[idx_inst] = 1.0
            result["instruments"] = inst_tensor

        if sample.key is not None:
            result["key"] = torch.tensor(sample.key, dtype=torch.long)

        if sample.chord is not None:
            result["chord"] = torch.tensor(sample.chord, dtype=torch.long)

        if sample.mode is not None:
            result["mode"] = torch.tensor(sample.mode, dtype=torch.long)

        if sample.perceptual is not None:
            result["perceptual"] = torch.tensor(sample.perceptual, dtype=torch.float32)

        if sample.era is not None:
            result["era"] = torch.tensor(sample.era, dtype=torch.long)

        if sample.chart_potential is not None:
            result["chart_potential"] = torch.tensor(sample.chart_potential, dtype=torch.float32)

        # Teacher embeddings for distillation
        if sample.clap_embedding is not None:
            result["clap_embedding"] = torch.from_numpy(sample.clap_embedding)

        if sample.panns_embedding is not None:
            result["panns_embedding"] = torch.from_numpy(sample.panns_embedding)

        if sample.ast_embedding is not None:
            result["ast_embedding"] = torch.from_numpy(sample.ast_embedding)

        return result


def rpm_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom collate function that handles variable-presence labels.
    Only batches labels that are present in ALL samples of the batch.
    Labels present in some but not all samples get masked.
    """
    result = {}

    # input_values is always present
    result["input_values"] = torch.stack([b["input_values"] for b in batch])

    # For each possible label key, check which samples have it
    label_keys = set()
    for b in batch:
        label_keys.update(k for k in b.keys() if k != "input_values")

    for key in label_keys:
        present = [b for b in batch if key in b]
        if len(present) == len(batch):
            # All samples have this label — stack normally
            result[key] = torch.stack([b[key] for b in batch])
        elif len(present) > len(batch) // 2:
            # More than half have it — stack present ones, mask missing
            # Create a mask tensor
            values = []
            mask = []
            for b in batch:
                if key in b:
                    values.append(b[key])
                    mask.append(1.0)
                else:
                    values.append(torch.zeros_like(present[0][key]))
                    mask.append(0.0)
            result[key] = torch.stack(values)
            result[f"{key}_mask"] = torch.tensor(mask, dtype=torch.float32)

    return result


# ──────────────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────────────

def build_dataset(cfg: DatasetConfig, feature_extractor=None,
                  split: str = "train", augment: bool = True) -> RPMDataset:
    """
    Build the unified RPM dataset from all available sources.

    Args:
        cfg: DatasetConfig with paths to all data sources
        feature_extractor: ASTFeatureExtractor for mel spectrogram extraction
        split: "train", "val", or "test"
        augment: whether to apply augmentation (only for train)
    """
    all_samples: list[TrainingSample] = []

    # 1. Local samples
    if cfg.local_samples_dir and os.path.isdir(cfg.local_samples_dir):
        loader = LocalSampleLoader(cfg.local_samples_dir, cfg.local_profiles_dir)
        all_samples.extend(loader.load())

    # 2. FMA
    if cfg.fma_dir and os.path.isdir(cfg.fma_dir):
        loader = FMALoader(cfg.fma_dir)
        all_samples.extend(loader.load())

    # 3. NSynth
    if cfg.nsynth_dir and os.path.isdir(cfg.nsynth_dir):
        loader = NSynthLoader(cfg.nsynth_dir)
        all_samples.extend(loader.load())

    # 4. MTG-Jamendo
    if cfg.jamendo_dir and os.path.isdir(cfg.jamendo_dir):
        loader = JamendoLoader(cfg.jamendo_dir)
        all_samples.extend(loader.load())

    # 5. Chart previews
    if cfg.chart_dir and os.path.isdir(cfg.chart_dir):
        loader = ChartPreviewLoader(cfg.chart_dir)
        all_samples.extend(loader.load())

    if not all_samples:
        logger.warning("No training samples found! Check your data paths.")
        return RPMDataset([], cfg, feature_extractor, augment=False)

    # Train/val/test split (90/5/5 by default)
    all_samples.sort(key=lambda s: s.filepath)
    rng = random.Random(42)
    rng.shuffle(all_samples)

    n = len(all_samples)
    if split == "train":
        samples = all_samples[:int(n * 0.90)]
        do_augment = augment
    elif split == "val":
        samples = all_samples[int(n * 0.90):int(n * 0.95)]
        do_augment = False
    else:  # test
        samples = all_samples[int(n * 0.95):]
        do_augment = False

    logger.info(f"Built {split} dataset: {len(samples):,} samples from {n:,} total")
    return RPMDataset(samples, cfg, feature_extractor, augment=do_augment)


def build_dataloader(dataset: RPMDataset, cfg: DatasetConfig,
                     shuffle: bool = True) -> DataLoader:
    """Build a DataLoader with proper collation and sampling."""
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=rpm_collate_fn,
        drop_last=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RPM Dataset — Structure Test")
    print("=" * 50)

    # Test with dummy samples
    dummy_samples = [
        TrainingSample(
            filepath="/tmp/test.wav",
            source="local",
            role=0,
            genre_top=3,
            perceptual=[0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.5, 0.3, 0.2],
            clap_embedding=np.random.randn(512).astype(np.float32),
        )
        for _ in range(8)
    ]

    cfg = DatasetConfig()
    dataset = RPMDataset(dummy_samples, cfg, feature_extractor=None, augment=False)

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample keys: {list(dataset[0].keys()) if len(dataset) > 0 else 'empty'}")
    print("\n✓ RPM Dataset structure validated.")
