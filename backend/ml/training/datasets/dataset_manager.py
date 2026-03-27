"""Public audio dataset manager for RESONATE Production Model training.

Downloads, organizes, and provides access to four public datasets:

    FMA (Free Music Archive)   — 106k tracks, 161 genres
    NSynth                     — 300k instrument notes (10 families)
    MUSDB18                    — 150 multi-track stems (mixture + 4 stems)
    MTG-Jamendo                — 55k tracks with mood/instrument/genre tags

Each dataset teaches the RPM something different:
    FMA         → genre classification, broad musical vocabulary
    NSynth      → timbral understanding, instrument recognition at note level
    MUSDB18     → production understanding, how stems combine into mixes
    MTG-Jamendo → mood/theme tagging, multi-label classification

Download state is persisted to disk so interrupted downloads can resume.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DATA_ROOT = Path.home() / ".resonate" / "datasets"

_FMA_ARCHIVE_BASE = "https://os.unil.cloud.switch.ch/fma"
_FMA_METADATA_URL = f"{_FMA_ARCHIVE_BASE}/fma_metadata.zip"
_FMA_SIZES: dict[str, dict[str, Any]] = {
    "small": {
        "url": f"{_FMA_ARCHIVE_BASE}/fma_small.zip",
        "tracks": 8_000,
        "size_gb": 7.2,
    },
    "medium": {
        "url": f"{_FMA_ARCHIVE_BASE}/fma_medium.zip",
        "tracks": 25_000,
        "size_gb": 22,
    },
    "large": {
        "url": f"{_FMA_ARCHIVE_BASE}/fma_large.zip",
        "tracks": 106_574,
        "size_gb": 93,
    },
}

_NSYNTH_BASE = "http://download.magenta.tensorflow.org/datasets/nsynth"
_NSYNTH_SPLITS: dict[str, dict[str, Any]] = {
    "train": {
        "url": f"{_NSYNTH_BASE}/nsynth-train.jsonwav.tar.gz",
        "notes": 289_205,
        "size_gb": 24,
    },
    "valid": {
        "url": f"{_NSYNTH_BASE}/nsynth-valid.jsonwav.tar.gz",
        "notes": 12_678,
        "size_gb": 1.0,
    },
    "test": {
        "url": f"{_NSYNTH_BASE}/nsynth-test.jsonwav.tar.gz",
        "notes": 4_096,
        "size_gb": 0.3,
    },
}

_NSYNTH_INSTRUMENT_FAMILIES = [
    "bass", "brass", "flute", "guitar", "keyboard",
    "mallet", "organ", "reed", "string", "synth_lead", "vocal",
]

# MUSDB18 is best accessed via the `musdb` pip package which handles
# download + stems decoding.  The raw archive URL is provided as fallback.
_MUSDB18_URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip"
_MUSDB18_TRACK_COUNT = 150

# MTG-Jamendo metadata lives on GitHub; audio on MTG servers.
_JAMENDO_META_REPO = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master"
_JAMENDO_SPLITS = ["autotagging.tsv", "autotagging_genre.tsv",
                   "autotagging_instrument.tsv", "autotagging_moodtheme.tsv"]
_JAMENDO_AUDIO_URL = "https://cdn.freesound.org/mtg-jamendo"
_JAMENDO_TRACK_COUNT = 55_000  # approximate

_DOWNLOAD_CHUNK_SIZE = 8192
_STATE_FILENAME = "download_state.json"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for which datasets to download and where to store them."""

    data_root: Path = field(default_factory=lambda: _DEFAULT_DATA_ROOT)
    fma_size: str = "small"  # "small" | "medium" | "large"
    download_nsynth: bool = True
    download_musdb: bool = True
    download_jamendo: bool = True
    num_workers: int = 4

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        if self.fma_size not in _FMA_SIZES:
            raise ValueError(
                f"fma_size must be one of {list(_FMA_SIZES)}, got {self.fma_size!r}"
            )


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(
    url: str,
    dest: Path,
    description: str = "",
    state: dict | None = None,
) -> bool:
    """Download a file with resume support and a tqdm progress bar.

    Returns True on success, False on failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    # Resume support — check bytes already downloaded
    existing_bytes = tmp.stat().st_size if tmp.exists() else 0
    headers: dict[str, str] = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        logger.info("Resuming %s from byte %d", dest.name, existing_bytes)

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        if resp.status_code == 416:
            # Range not satisfiable — file already complete
            if tmp.exists():
                tmp.rename(dest)
            return True
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        if existing_bytes and resp.status_code == 206:
            total += existing_bytes
        elif resp.status_code == 200:
            existing_bytes = 0  # server does not support range; restart
            if tmp.exists():
                tmp.unlink()

        mode = "ab" if existing_bytes else "wb"
        desc = description or dest.name

        with (
            open(tmp, mode) as fh,
            tqdm(
                total=total or None,
                initial=existing_bytes,
                unit="B",
                unit_scale=True,
                desc=desc,
            ) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                fh.write(chunk)
                pbar.update(len(chunk))

        tmp.rename(dest)
        logger.info("Downloaded %s (%s)", dest.name, _human_size(dest.stat().st_size))
        return True

    except (requests.RequestException, OSError) as exc:
        logger.error("Failed to download %s: %s", url, exc)
        return False


def _extract_zip(archive: Path, dest: Path) -> bool:
    """Extract a zip archive, showing progress."""
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            members = zf.namelist()
            for member in tqdm(members, desc=f"Extracting {archive.name}", unit="file"):
                zf.extract(member, dest)
        logger.info("Extracted %s → %s (%d entries)", archive.name, dest, len(members))
        return True
    except (zipfile.BadZipFile, OSError) as exc:
        logger.error("Extraction failed for %s: %s", archive.name, exc)
        return False


def _extract_tar(archive: Path, dest: Path) -> bool:
    """Extract a tar.gz archive, showing progress."""
    try:
        with tarfile.open(archive, "r:gz") as tf:
            members = tf.getmembers()
            for member in tqdm(members, desc=f"Extracting {archive.name}", unit="file"):
                tf.extract(member, dest, filter="data")
        logger.info("Extracted %s → %s (%d entries)", archive.name, dest, len(members))
        return True
    except (tarfile.TarError, OSError) as exc:
        logger.error("Extraction failed for %s: %s", archive.name, exc)
        return False


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} PB"


def _dir_size(path: Path) -> int:
    """Recursive directory size in bytes."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ---------------------------------------------------------------------------
# Download state persistence
# ---------------------------------------------------------------------------

class _DownloadState:
    """Tracks which datasets/components have been downloaded successfully."""

    def __init__(self, data_root: Path) -> None:
        self._path = data_root / _STATE_FILENAME
        self._state: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._state = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt download state file, starting fresh")
                self._state = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2))

    def is_done(self, key: str) -> bool:
        return self._state.get(key, {}).get("done", False)

    def mark_done(self, key: str, **metadata: Any) -> None:
        self._state[key] = {"done": True, **metadata}
        self._save()

    def as_dict(self) -> dict[str, Any]:
        return dict(self._state)


# ---------------------------------------------------------------------------
# FMA Manager
# ---------------------------------------------------------------------------

class FMAManager:
    """Manages the Free Music Archive dataset."""

    def __init__(self, data_root: Path, size: str = "small") -> None:
        self.root = data_root / "fma"
        self.audio_root = self.root / "audio"
        self.meta_root = self.root / "metadata"
        self.size = size
        self._size_info = _FMA_SIZES[size]

    # -- download ----------------------------------------------------------

    def download(self, state: _DownloadState) -> bool:
        """Download FMA audio and metadata. Returns True if all succeeded."""
        self.root.mkdir(parents=True, exist_ok=True)
        ok = True

        # Metadata
        if not state.is_done("fma_metadata"):
            archive = self.root / "fma_metadata.zip"
            if _download_file(_FMA_METADATA_URL, archive, "FMA metadata"):
                if _extract_zip(archive, self.meta_root):
                    state.mark_done("fma_metadata")
                    archive.unlink(missing_ok=True)
                else:
                    ok = False
            else:
                ok = False

        # Audio
        state_key = f"fma_audio_{self.size}"
        if not state.is_done(state_key):
            archive = self.root / f"fma_{self.size}.zip"
            desc = f"FMA {self.size} ({self._size_info['size_gb']} GB)"
            if _download_file(self._size_info["url"], archive, desc):
                if _extract_zip(archive, self.audio_root):
                    state.mark_done(state_key, tracks=self._size_info["tracks"])
                    archive.unlink(missing_ok=True)
                else:
                    ok = False
            else:
                ok = False

        return ok

    # -- access ------------------------------------------------------------

    def get_tracks(self) -> list[dict[str, Any]]:
        """Return metadata dicts for every available FMA track."""
        tracks_csv = self._find_tracks_csv()
        if tracks_csv is None:
            logger.warning("FMA tracks.csv not found — is the dataset downloaded?")
            return []

        import csv

        tracks: list[dict[str, Any]] = []
        with open(tracks_csv, newline="", encoding="utf-8") as fh:
            # FMA tracks.csv has a multi-row header; row 0 is category, row 1
            # is sub-field, row 2 is the column we care about.  We skip the
            # first three rows and parse manually.
            reader = csv.reader(fh)
            header_rows = [next(reader) for _ in range(3)]
            # Build a simplified column index from the third header row
            col_names = header_rows[2]

            for row in reader:
                if not row:
                    continue
                track_id = row[0].strip()
                if not track_id.isdigit():
                    continue
                tid = track_id.zfill(6)
                filepath = self.audio_root / tid[:3] / f"{tid}.mp3"
                genre_top = row[col_names.index("genre_top")] if "genre_top" in col_names else ""
                tracks.append({
                    "filepath": str(filepath),
                    "genre_top": genre_top.strip(),
                    "genre_sub": "",
                    "track_id": int(track_id),
                })
        logger.info("FMA: loaded %d track entries", len(tracks))
        return tracks

    def get_genre_mapping(self) -> dict[int, str]:
        """Map FMA genre IDs to genre names from genres.csv."""
        genres_csv = self.meta_root / "fma_metadata" / "genres.csv"
        if not genres_csv.exists():
            # Try flat structure
            genres_csv = self.meta_root / "genres.csv"
        if not genres_csv.exists():
            logger.warning("FMA genres.csv not found")
            return {}

        import csv

        mapping: dict[int, str] = {}
        with open(genres_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    mapping[int(row["genre_id"])] = row["title"]
                except (KeyError, ValueError):
                    continue
        return mapping

    # -- internal ----------------------------------------------------------

    def _find_tracks_csv(self) -> Optional[Path]:
        candidates = [
            self.meta_root / "fma_metadata" / "tracks.csv",
            self.meta_root / "tracks.csv",
            self.root / "tracks.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None


# ---------------------------------------------------------------------------
# NSynth Manager
# ---------------------------------------------------------------------------

class NSynthManager:
    """Manages the NSynth dataset (individual instrument notes)."""

    def __init__(self, data_root: Path) -> None:
        self.root = data_root / "nsynth"

    # -- download ----------------------------------------------------------

    def download(self, state: _DownloadState) -> bool:
        """Download all NSynth splits. Returns True if all succeeded."""
        self.root.mkdir(parents=True, exist_ok=True)
        ok = True

        for split, info in _NSYNTH_SPLITS.items():
            state_key = f"nsynth_{split}"
            if state.is_done(state_key):
                continue

            archive = self.root / f"nsynth-{split}.jsonwav.tar.gz"
            desc = f"NSynth {split} ({info['size_gb']} GB)"
            if _download_file(info["url"], archive, desc):
                if _extract_tar(archive, self.root):
                    state.mark_done(state_key, notes=info["notes"])
                    archive.unlink(missing_ok=True)
                else:
                    ok = False
            else:
                ok = False

        return ok

    # -- access ------------------------------------------------------------

    def get_notes(self) -> list[dict[str, Any]]:
        """Return metadata for every NSynth note across all downloaded splits."""
        notes: list[dict[str, Any]] = []

        for split in ("train", "valid", "test"):
            split_dir = self.root / f"nsynth-{split}"
            meta_file = split_dir / "examples.json"
            if not meta_file.exists():
                continue

            meta = json.loads(meta_file.read_text())
            audio_dir = split_dir / "audio"

            for note_id, attrs in meta.items():
                filepath = audio_dir / f"{note_id}.wav"
                notes.append({
                    "filepath": str(filepath),
                    "instrument": attrs.get("instrument_str", ""),
                    "instrument_family": _NSYNTH_INSTRUMENT_FAMILIES[
                        attrs.get("instrument_family", 0)
                    ] if attrs.get("instrument_family", -1) < len(_NSYNTH_INSTRUMENT_FAMILIES) else "unknown",
                    "pitch": attrs.get("pitch", 0),
                    "velocity": attrs.get("velocity", 0),
                    "qualities": attrs.get("qualities", []),
                })

        logger.info("NSynth: loaded %d note entries", len(notes))
        return notes


# ---------------------------------------------------------------------------
# MUSDB18 Manager
# ---------------------------------------------------------------------------

class MUSDB18Manager:
    """Manages the MUSDB18 multi-track stem dataset."""

    STEM_NAMES = ("mixture", "vocals", "drums", "bass", "other")

    def __init__(self, data_root: Path) -> None:
        self.root = data_root / "musdb18"

    # -- download ----------------------------------------------------------

    def download(self, state: _DownloadState) -> bool:
        """Download MUSDB18.

        Prefers the ``musdb`` Python package (``pip install musdb``) which
        handles download and decoding automatically.  Falls back to direct
        zip download from Zenodo.
        """
        self.root.mkdir(parents=True, exist_ok=True)

        if state.is_done("musdb18"):
            return True

        # Strategy 1: use the musdb package
        try:
            import musdb  # type: ignore[import-untyped]
            logger.info("Using musdb package to fetch MUSDB18-HQ")
            _ = musdb.DB(root=str(self.root), download=True)
            state.mark_done("musdb18", method="musdb_package", tracks=_MUSDB18_TRACK_COUNT)
            return True
        except ImportError:
            logger.info(
                "musdb package not installed — falling back to direct download. "
                "Install with: pip install musdb"
            )
        except Exception as exc:
            logger.warning("musdb download failed (%s), trying direct download", exc)

        # Strategy 2: direct zip from Zenodo
        archive = self.root / "musdb18hq.zip"
        desc = "MUSDB18-HQ (~3.3 GB)"
        if _download_file(_MUSDB18_URL, archive, desc):
            if _extract_zip(archive, self.root):
                state.mark_done("musdb18", method="direct_zip", tracks=_MUSDB18_TRACK_COUNT)
                archive.unlink(missing_ok=True)
                return True

        logger.error(
            "MUSDB18 download failed. You can download manually from:\n"
            "  %s\n"
            "and extract into %s",
            _MUSDB18_URL, self.root,
        )
        return False

    # -- access ------------------------------------------------------------

    def get_tracks(self) -> list[dict[str, Any]]:
        """Return track dicts with mixture path and stem paths."""
        tracks: list[dict[str, Any]] = []

        # MUSDB18 organises as: root/{train,test}/{track_name}/{stem}.wav
        for subset_dir in sorted(self.root.iterdir()):
            if not subset_dir.is_dir() or subset_dir.name.startswith("."):
                continue
            for track_dir in sorted(subset_dir.iterdir()):
                if not track_dir.is_dir():
                    continue
                mixture = track_dir / "mixture.wav"
                if not mixture.exists():
                    continue
                stems: dict[str, str] = {}
                for stem in self.STEM_NAMES:
                    stem_path = track_dir / f"{stem}.wav"
                    if stem_path.exists():
                        stems[stem] = str(stem_path)
                tracks.append({
                    "track_name": track_dir.name,
                    "mixture_path": str(mixture),
                    "stems": stems,
                })

        logger.info("MUSDB18: loaded %d tracks", len(tracks))
        return tracks


# ---------------------------------------------------------------------------
# MTG-Jamendo Manager
# ---------------------------------------------------------------------------

class MTGJamendoManager:
    """Manages the MTG-Jamendo dataset (55k tracks with auto-tags)."""

    def __init__(self, data_root: Path) -> None:
        self.root = data_root / "mtg-jamendo"
        self.meta_root = self.root / "metadata"
        self.audio_root = self.root / "audio"

    # -- download ----------------------------------------------------------

    def download(self, state: _DownloadState) -> bool:
        """Download MTG-Jamendo metadata and audio.

        Metadata TSVs are fetched from the GitHub repo.  Audio download
        requires the ``mtg-jamendo-dataset`` tools or manual download from
        the MTG servers, since the full 500 GB audio set is too large for
        a simple HTTP grab.  We download metadata unconditionally and
        provide instructions for audio.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_root.mkdir(parents=True, exist_ok=True)
        ok = True

        # Metadata TSVs
        if not state.is_done("jamendo_metadata"):
            all_ok = True
            for tsv_name in _JAMENDO_SPLITS:
                dest = self.meta_root / tsv_name
                if dest.exists():
                    continue
                url = f"{_JAMENDO_META_REPO}/data/splits/{tsv_name}"
                if not _download_file(url, dest, f"Jamendo {tsv_name}"):
                    all_ok = False
            if all_ok:
                state.mark_done("jamendo_metadata")
            else:
                ok = False

        # Audio — provide instructions since bulk download needs special tooling
        if not state.is_done("jamendo_audio"):
            audio_count = sum(1 for _ in self.audio_root.rglob("*.mp3")) if self.audio_root.exists() else 0
            if audio_count >= _JAMENDO_TRACK_COUNT * 0.9:
                state.mark_done("jamendo_audio", tracks=audio_count)
            else:
                logger.info(
                    "MTG-Jamendo audio requires bulk download (~500 GB).\n"
                    "Options:\n"
                    "  1. Use mtg-jamendo-dataset tools:\n"
                    "     pip install mtg-jamendo-dataset\n"
                    "     mtg-jamendo download --type audio --output %s\n"
                    "  2. Download a subset via the API at:\n"
                    "     https://mtg.github.io/mtg-jamendo-dataset/\n"
                    "  3. Use the low-quality (22kHz) version for faster download.\n"
                    "Place MP3 files in: %s",
                    self.audio_root, self.audio_root,
                )
                # We don't fail here — metadata alone is usable for tag training
                if not self.audio_root.exists():
                    self.audio_root.mkdir(parents=True, exist_ok=True)

        return ok

    # -- access ------------------------------------------------------------

    def get_tracks(self) -> list[dict[str, Any]]:
        """Return track dicts with mood, genre, and instrument tags."""
        # Parse the TSV label files
        mood_tags = self._parse_tsv("autotagging_moodtheme.tsv")
        genre_tags = self._parse_tsv("autotagging_genre.tsv")
        instrument_tags = self._parse_tsv("autotagging_instrument.tsv")

        # Union of all track IDs
        all_ids = set(mood_tags) | set(genre_tags) | set(instrument_tags)
        tracks: list[dict[str, Any]] = []

        for track_id in sorted(all_ids):
            # Jamendo audio is stored as {id}.mp3, possibly in subdirectories
            filepath = self._find_audio(track_id)
            tracks.append({
                "filepath": str(filepath) if filepath else "",
                "track_id": track_id,
                "mood_tags": mood_tags.get(track_id, []),
                "genre_tags": genre_tags.get(track_id, []),
                "instrument_tags": instrument_tags.get(track_id, []),
            })

        logger.info("MTG-Jamendo: loaded %d track entries", len(tracks))
        return tracks

    # -- internal ----------------------------------------------------------

    def _parse_tsv(self, filename: str) -> dict[str, list[str]]:
        """Parse an MTG-Jamendo autotagging TSV into {track_id: [tags]}."""
        tsv_path = self.meta_root / filename
        if not tsv_path.exists():
            return {}

        result: dict[str, list[str]] = {}
        with open(tsv_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                # Format: track_id\ttag1\ttag2\t...
                # Some files have path\ttags format
                track_id = parts[0].split("/")[-1].replace(".mp3", "")
                tags = [t.strip() for t in parts[1:] if t.strip()]
                result[track_id] = tags

        return result

    def _find_audio(self, track_id: str) -> Optional[Path]:
        """Locate the MP3 for a given track ID."""
        # Direct path
        direct = self.audio_root / f"{track_id}.mp3"
        if direct.exists():
            return direct

        # Subdirectory structure: {id[:2]}/{track_id}.mp3
        subdir = self.audio_root / track_id[:2] / f"{track_id}.mp3"
        if subdir.exists():
            return subdir

        return None


# ---------------------------------------------------------------------------
# Dataset Manager (orchestrator)
# ---------------------------------------------------------------------------

class DatasetManager:
    """Top-level orchestrator for all RESONATE training datasets."""

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.config = config or DatasetConfig()
        self.config.data_root.mkdir(parents=True, exist_ok=True)
        self._state = _DownloadState(self.config.data_root)

        self.fma = FMAManager(self.config.data_root, self.config.fma_size)
        self.nsynth = NSynthManager(self.config.data_root)
        self.musdb = MUSDB18Manager(self.config.data_root)
        self.jamendo = MTGJamendoManager(self.config.data_root)

    # -- download ----------------------------------------------------------

    def download_all(self) -> dict[str, bool]:
        """Download all enabled datasets. Returns {name: success} mapping."""
        results: dict[str, bool] = {}

        # FMA is always downloaded (it is the core genre dataset)
        logger.info("=" * 60)
        logger.info("Downloading FMA (%s) ...", self.config.fma_size)
        logger.info("=" * 60)
        try:
            results["fma"] = self.fma.download(self._state)
        except Exception as exc:
            logger.error("FMA download failed: %s", exc)
            results["fma"] = False

        if self.config.download_nsynth:
            logger.info("=" * 60)
            logger.info("Downloading NSynth ...")
            logger.info("=" * 60)
            try:
                results["nsynth"] = self.nsynth.download(self._state)
            except Exception as exc:
                logger.error("NSynth download failed: %s", exc)
                results["nsynth"] = False

        if self.config.download_musdb:
            logger.info("=" * 60)
            logger.info("Downloading MUSDB18 ...")
            logger.info("=" * 60)
            try:
                results["musdb18"] = self.musdb.download(self._state)
            except Exception as exc:
                logger.error("MUSDB18 download failed: %s", exc)
                results["musdb18"] = False

        if self.config.download_jamendo:
            logger.info("=" * 60)
            logger.info("Downloading MTG-Jamendo ...")
            logger.info("=" * 60)
            try:
                results["jamendo"] = self.jamendo.download(self._state)
            except Exception as exc:
                logger.error("MTG-Jamendo download failed: %s", exc)
                results["jamendo"] = False

        # Summary
        logger.info("=" * 60)
        logger.info("Download summary:")
        for name, success in results.items():
            status = "OK" if success else "FAILED"
            logger.info("  %-12s %s", name, status)
        logger.info("=" * 60)

        return results

    # -- status ------------------------------------------------------------

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Return download/size status for each dataset."""
        status: dict[str, dict[str, Any]] = {}

        status["fma"] = {
            "downloaded": self._state.is_done(f"fma_audio_{self.config.fma_size}"),
            "metadata": self._state.is_done("fma_metadata"),
            "size": self.config.fma_size,
            "expected_tracks": _FMA_SIZES[self.config.fma_size]["tracks"],
            "disk_usage": _human_size(_dir_size(self.fma.root)),
        }

        status["nsynth"] = {
            "downloaded": all(
                self._state.is_done(f"nsynth_{s}") for s in _NSYNTH_SPLITS
            ),
            "splits": {
                s: self._state.is_done(f"nsynth_{s}") for s in _NSYNTH_SPLITS
            },
            "expected_notes": sum(i["notes"] for i in _NSYNTH_SPLITS.values()),
            "disk_usage": _human_size(_dir_size(self.nsynth.root)),
        }

        status["musdb18"] = {
            "downloaded": self._state.is_done("musdb18"),
            "expected_tracks": _MUSDB18_TRACK_COUNT,
            "disk_usage": _human_size(_dir_size(self.musdb.root)),
        }

        status["jamendo"] = {
            "metadata_downloaded": self._state.is_done("jamendo_metadata"),
            "audio_downloaded": self._state.is_done("jamendo_audio"),
            "expected_tracks": _JAMENDO_TRACK_COUNT,
            "disk_usage": _human_size(_dir_size(self.jamendo.root)),
        }

        return status

    def get_combined_track_count(self) -> int:
        """Return total number of audio files across all downloaded datasets."""
        count = 0

        if self.fma.audio_root.exists():
            count += sum(1 for _ in self.fma.audio_root.rglob("*.mp3"))

        if self.nsynth.root.exists():
            count += sum(1 for _ in self.nsynth.root.rglob("*.wav"))

        if self.musdb.root.exists():
            count += sum(
                1 for d in self.musdb.root.rglob("mixture.wav")
            )

        if self.jamendo.audio_root.exists():
            count += sum(1 for _ in self.jamendo.audio_root.rglob("*.mp3"))

        return count

    def verify_integrity(self) -> dict[str, dict[str, Any]]:
        """Verify file counts against expected values for each dataset."""
        results: dict[str, dict[str, Any]] = {}

        # FMA
        expected_fma = _FMA_SIZES[self.config.fma_size]["tracks"]
        actual_fma = sum(1 for _ in self.fma.audio_root.rglob("*.mp3")) if self.fma.audio_root.exists() else 0
        results["fma"] = {
            "expected": expected_fma,
            "actual": actual_fma,
            "ok": actual_fma >= expected_fma * 0.95,  # 5% tolerance for corrupt/missing
        }

        # NSynth
        expected_nsynth = sum(i["notes"] for i in _NSYNTH_SPLITS.values())
        actual_nsynth = sum(1 for _ in self.nsynth.root.rglob("*.wav")) if self.nsynth.root.exists() else 0
        results["nsynth"] = {
            "expected": expected_nsynth,
            "actual": actual_nsynth,
            "ok": actual_nsynth >= expected_nsynth * 0.95,
        }

        # MUSDB18
        actual_musdb = sum(1 for _ in self.musdb.root.rglob("mixture.wav")) if self.musdb.root.exists() else 0
        results["musdb18"] = {
            "expected": _MUSDB18_TRACK_COUNT,
            "actual": actual_musdb,
            "ok": actual_musdb >= _MUSDB18_TRACK_COUNT * 0.95,
        }

        # Jamendo
        actual_jamendo = sum(1 for _ in self.jamendo.audio_root.rglob("*.mp3")) if self.jamendo.audio_root.exists() else 0
        results["jamendo"] = {
            "expected": _JAMENDO_TRACK_COUNT,
            "actual": actual_jamendo,
            "ok": actual_jamendo >= _JAMENDO_TRACK_COUNT * 0.90,  # looser for partial downloads
        }

        for name, info in results.items():
            status = "PASS" if info["ok"] else "FAIL"
            logger.info(
                "Integrity %-12s %s  (expected=%d, actual=%d)",
                name, status, info["expected"], info["actual"],
            )

        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_status(manager: DatasetManager) -> None:
    """Pretty-print dataset status to stdout."""
    status = manager.get_status()
    total = manager.get_combined_track_count()

    print("\n" + "=" * 60)
    print("  RESONATE Dataset Status")
    print("=" * 60)

    for name, info in status.items():
        downloaded = info.get("downloaded", False)
        disk = info.get("disk_usage", "0 B")
        marker = "[x]" if downloaded else "[ ]"
        print(f"  {marker} {name:12s}  {disk:>10s}")
        for k, v in info.items():
            if k in ("downloaded", "disk_usage"):
                continue
            print(f"      {k}: {v}")

    print("-" * 60)
    print(f"  Total audio files on disk: {total:,}")
    print(f"  Data root: {manager.config.data_root}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="RESONATE dataset manager — download and manage training data"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help=f"Root directory for datasets (default: {_DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--fma-size",
        choices=["small", "medium", "large"],
        default="small",
        help="FMA dataset variant (default: small)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all enabled datasets",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset integrity",
    )
    parser.add_argument(
        "--no-nsynth", action="store_true", help="Skip NSynth download"
    )
    parser.add_argument(
        "--no-musdb", action="store_true", help="Skip MUSDB18 download"
    )
    parser.add_argument(
        "--no-jamendo", action="store_true", help="Skip MTG-Jamendo download"
    )

    args = parser.parse_args()

    config = DatasetConfig(
        data_root=args.data_root,
        fma_size=args.fma_size,
        download_nsynth=not args.no_nsynth,
        download_musdb=not args.no_musdb,
        download_jamendo=not args.no_jamendo,
    )
    manager = DatasetManager(config)

    _print_status(manager)

    if args.download:
        print("Starting downloads...\n")
        results = manager.download_all()
        _print_status(manager)
        if not all(results.values()):
            print("Some downloads failed. Run again to retry.", file=sys.stderr)
            sys.exit(1)

    if args.verify:
        print("Verifying integrity...\n")
        integrity = manager.verify_integrity()
        failures = [n for n, i in integrity.items() if not i["ok"]]
        if failures:
            print(f"Integrity check FAILED for: {', '.join(failures)}", file=sys.stderr)
            sys.exit(1)
        else:
            print("All integrity checks passed.")
