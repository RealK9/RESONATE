"""
RESONATE Production Model — Phase F: Multi-Track Stem Understanding

Teaches the model how music is constructed from individual stems.
This is what separates a music AI from a sound classifier — understanding
how drums + bass + vocals + melodics combine into a mix.

Data sources (all free):
  - MUSDB18-HQ:  150 songs, 4 stems each (drums/bass/vocals/other), uncompressed WAV
  - MedleyDB:    122 multitracks, up to 32 individual stems per song
  - Slakh2100:   2,100 songs, MIDI-synthesized, every instrument separated
  - DSD100:      100 songs, 4 stems (predecessor to MUSDB18)
  - CocoChorales: 1,400 Bach chorales, 4 voice parts separated

Total: ~3,870 multitrack songs with separated stems.

Training objectives:
  - Stem identification: which instrument is this stem?
  - Mix understanding: how do stems combine?
  - Frequency masking: what happens when two sounds fight for space?
  - Mix balance: what's a "good" vs "bad" mix?
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path.home() / ".resonate" / "datasets" / "phase_f"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StemTrack:
    """A single multitrack song with separated stems."""
    track_name: str
    mix_path: str  # Full mix audio path
    stems: dict[str, str] = field(default_factory=dict)  # stem_name → audio_path
    source: str = ""
    duration: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class StemPair:
    """A training pair: mix audio + individual stem + stem label."""
    mix_path: str
    stem_path: str
    stem_name: str  # drums, bass, vocals, other, piano, guitar, etc.
    track_name: str
    source: str


# ---------------------------------------------------------------------------
# MUSDB18-HQ (150 songs, 4 stems, uncompressed)
# ---------------------------------------------------------------------------

MUSDB18_HQ_URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip"
MUSDB18_URL = "https://zenodo.org/records/1117372/files/musdb18.zip"

class MUSDB18Downloader:
    """
    Downloads MUSDB18/MUSDB18-HQ: the gold standard for source separation research.
    150 songs, each with: drums, bass, vocals, other (accompaniment).
    """

    STEM_NAMES = ["drums", "bass", "vocals", "other"]

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT, hq: bool = True):
        self.hq = hq
        name = "musdb18hq" if hq else "musdb18"
        self.data_dir = data_root / name
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> Path:
        """Download and extract MUSDB18(-HQ)."""
        url = MUSDB18_HQ_URL if self.hq else MUSDB18_URL
        name = "musdb18hq" if self.hq else "musdb18"
        zip_path = self.data_dir / f"{name}.zip"
        extract_dir = self.data_dir / name

        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"  {name} already extracted at {extract_dir}")
            return extract_dir

        if not zip_path.exists():
            size = "30GB" if self.hq else "4.4GB"
            logger.info(f"  Downloading {name} ({size})...")
            subprocess.run([
                "wget", "-q", "--show-progress", url, "-O", str(zip_path),
            ], check=True)

        logger.info(f"  Extracting {name}...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(self.data_dir)], check=True)
        zip_path.unlink(missing_ok=True)

        logger.info(f"  {name} ready at {extract_dir}")
        return extract_dir

    def load_tracks(self) -> list[StemTrack]:
        """Load all tracks with stem paths."""
        tracks = []
        name = "musdb18hq" if self.hq else "musdb18"
        base = self.data_dir / name

        for split in ["train", "test"]:
            split_dir = base / split
            if not split_dir.exists():
                # Try flat structure
                split_dir = base
                if not split_dir.exists():
                    continue

            for track_dir in sorted(split_dir.iterdir()):
                if not track_dir.is_dir():
                    continue

                stems = {}
                mix_path = ""

                # MUSDB18-HQ has wav files per stem
                for stem_name in self.STEM_NAMES + ["mixture"]:
                    for ext in [".wav", ".mp4", ".stem.mp4"]:
                        stem_file = track_dir / f"{stem_name}{ext}"
                        if stem_file.exists():
                            if stem_name == "mixture":
                                mix_path = str(stem_file)
                            else:
                                stems[stem_name] = str(stem_file)
                            break

                if stems:
                    tracks.append(StemTrack(
                        track_name=track_dir.name,
                        mix_path=mix_path,
                        stems=stems,
                        source=f"musdb18{'hq' if self.hq else ''}",
                    ))

        logger.info(f"  MUSDB18{'HQ' if self.hq else ''}: {len(tracks)} tracks loaded")
        return tracks


# ---------------------------------------------------------------------------
# MedleyDB (122 multitracks, up to 32 stems)
# ---------------------------------------------------------------------------

MEDLEYDB_METADATA_URL = "https://raw.githubusercontent.com/marl/medleydb/master/medleydb/resources/metadata"

class MedleyDBDownloader:
    """
    Downloads MedleyDB: 122 multitracks with up to 32 individual stems.
    Audio must be requested from the MedleyDB website (academic use).
    This downloader handles metadata and pairs with locally available audio.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "medleydb"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_metadata(self) -> list[dict]:
        """Download track listing and instrument annotations."""
        meta_dir = self.data_dir / "metadata"
        meta_dir.mkdir(exist_ok=True)

        # Download the track listing
        listing_url = f"{MEDLEYDB_METADATA_URL}/medleydb_tracklist.txt"
        listing_path = meta_dir / "tracklist.txt"

        if not listing_path.exists():
            logger.info("  Downloading MedleyDB tracklist...")
            try:
                resp = requests.get(listing_url, timeout=30)
                if resp.status_code == 200:
                    listing_path.write_bytes(resp.content)
            except Exception as e:
                logger.warning(f"  Failed to download MedleyDB tracklist: {e}")

        # Parse tracklist
        entries = []
        if listing_path.exists():
            for line in listing_path.read_text().strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    entries.append({"track_name": line})

        logger.info(f"  MedleyDB: {len(entries)} tracks in metadata")
        return entries

    def load_tracks(self, audio_root: Optional[Path] = None) -> list[StemTrack]:
        """Load tracks from local audio directory."""
        if audio_root is None:
            audio_root = self.data_dir / "Audio"

        tracks = []
        if not audio_root.exists():
            logger.info("  MedleyDB audio not found — download from medleydb.weebly.com")
            return tracks

        for track_dir in sorted(audio_root.iterdir()):
            if not track_dir.is_dir():
                continue

            stems = {}
            mix_path = ""

            # MedleyDB structure: TrackName/TrackName_MIX.wav + TrackName_STEMS/
            mix_file = track_dir / f"{track_dir.name}_MIX.wav"
            if mix_file.exists():
                mix_path = str(mix_file)

            stems_dir = track_dir / f"{track_dir.name}_STEMS"
            if stems_dir.exists():
                for stem_file in stems_dir.iterdir():
                    if stem_file.suffix in (".wav", ".flac"):
                        stem_name = stem_file.stem.replace(f"{track_dir.name}_STEM_", "")
                        stems[stem_name] = str(stem_file)

            # Also check RAW stems
            raw_dir = track_dir / f"{track_dir.name}_RAW"
            if raw_dir.exists():
                for stem_file in raw_dir.iterdir():
                    if stem_file.suffix in (".wav", ".flac"):
                        stem_name = f"raw_{stem_file.stem}"
                        stems[stem_name] = str(stem_file)

            if stems:
                tracks.append(StemTrack(
                    track_name=track_dir.name,
                    mix_path=mix_path,
                    stems=stems,
                    source="medleydb",
                ))

        logger.info(f"  MedleyDB: {len(tracks)} tracks loaded, {sum(len(t.stems) for t in tracks)} total stems")
        return tracks


# ---------------------------------------------------------------------------
# Slakh2100 (2,100 MIDI-synthesized multitracks)
# ---------------------------------------------------------------------------

SLAKH_URLS = {
    "slakh2100": "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz",
    "metadata": "https://zenodo.org/records/4599666/files/slakh2100_metadata.tar.gz",
}

class Slakh2100Downloader:
    """
    Downloads Slakh2100: 2,100 automatically mixed songs from MIDI.
    Each song has separated stems for every instrument track.
    FLAC audio, ~150GB uncompressed.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "slakh2100"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self, audio: bool = True) -> Path:
        """Download Slakh2100 dataset."""
        # Download metadata first (smaller)
        meta_tar = self.data_dir / "slakh2100_metadata.tar.gz"
        if not meta_tar.exists() and not (self.data_dir / "slakh2100").exists():
            logger.info("  Downloading Slakh2100 metadata...")
            subprocess.run([
                "wget", "-q", "--show-progress",
                SLAKH_URLS["metadata"], "-O", str(meta_tar),
            ], check=False)

            if meta_tar.exists():
                subprocess.run(["tar", "xzf", str(meta_tar), "-C", str(self.data_dir)], check=False)
                meta_tar.unlink(missing_ok=True)

        if audio:
            audio_tar = self.data_dir / "slakh2100_flac_redux.tar.gz"
            if not audio_tar.exists() and not (self.data_dir / "slakh2100" / "train").exists():
                logger.info("  Downloading Slakh2100 audio (~150GB FLAC)...")
                logger.info("  This is a LARGE download. Go get coffee.")
                subprocess.run([
                    "wget", "-q", "--show-progress",
                    SLAKH_URLS["slakh2100"], "-O", str(audio_tar),
                ], check=True)

                logger.info("  Extracting Slakh2100...")
                subprocess.run(["tar", "xzf", str(audio_tar), "-C", str(self.data_dir)], check=True)
                audio_tar.unlink(missing_ok=True)

        return self.data_dir / "slakh2100"

    def load_tracks(self) -> list[StemTrack]:
        """Load all Slakh2100 tracks with stems."""
        tracks = []
        base = self.data_dir / "slakh2100"

        for split in ["train", "validation", "test"]:
            split_dir = base / split
            if not split_dir.exists():
                continue

            for track_dir in sorted(split_dir.iterdir()):
                if not track_dir.is_dir():
                    continue

                stems = {}
                mix_path = ""

                # Mix file
                mix_file = track_dir / "mix.flac"
                if mix_file.exists():
                    mix_path = str(mix_file)

                # Stems directory
                stems_dir = track_dir / "stems"
                if stems_dir.exists():
                    for stem_file in stems_dir.iterdir():
                        if stem_file.suffix in (".flac", ".wav"):
                            stems[stem_file.stem] = str(stem_file)

                # Load metadata for instrument labels
                meta_file = track_dir / "metadata.yaml"
                metadata = {}
                if meta_file.exists():
                    try:
                        import yaml
                        with open(meta_file) as f:
                            metadata = yaml.safe_load(f) or {}
                    except ImportError:
                        pass

                if stems:
                    tracks.append(StemTrack(
                        track_name=track_dir.name,
                        mix_path=mix_path,
                        stems=stems,
                        source="slakh2100",
                        metadata=metadata,
                    ))

        logger.info(f"  Slakh2100: {len(tracks)} tracks, {sum(len(t.stems) for t in tracks)} stems")
        return tracks


# ---------------------------------------------------------------------------
# DSD100 (100 songs, 4 stems)
# ---------------------------------------------------------------------------

DSD100_URL = "https://zenodo.org/records/2747462/files/DSD100.zip"

class DSD100Downloader:
    """
    Downloads DSD100: 100 professionally mixed songs with 4 stems.
    Predecessor to MUSDB18.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "dsd100"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> Path:
        zip_path = self.data_dir / "DSD100.zip"
        extract_dir = self.data_dir / "DSD100"

        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"  DSD100 already extracted")
            return extract_dir

        if not zip_path.exists():
            logger.info("  Downloading DSD100 (~14GB)...")
            subprocess.run([
                "wget", "-q", "--show-progress", DSD100_URL, "-O", str(zip_path),
            ], check=True)

        logger.info("  Extracting DSD100...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(self.data_dir)], check=True)
        zip_path.unlink(missing_ok=True)
        return extract_dir

    def load_tracks(self) -> list[StemTrack]:
        tracks = []
        base = self.data_dir / "DSD100"

        for split in ["Sources", "Mixtures"]:
            pass  # DSD100 structure varies

        # DSD100 typical structure: DSD100/Sources/Dev/TrackName/stem.wav
        for subset in ["Dev", "Test"]:
            sources_dir = base / "Sources" / subset
            mix_dir = base / "Mixtures" / subset

            if not sources_dir.exists():
                continue

            for track_dir in sorted(sources_dir.iterdir()):
                if not track_dir.is_dir():
                    continue

                stems = {}
                for stem_file in track_dir.iterdir():
                    if stem_file.suffix == ".wav":
                        stems[stem_file.stem.lower()] = str(stem_file)

                mix_path = ""
                mix_track_dir = mix_dir / track_dir.name if mix_dir.exists() else None
                if mix_track_dir and mix_track_dir.exists():
                    mix_file = mix_track_dir / "mixture.wav"
                    if mix_file.exists():
                        mix_path = str(mix_file)

                if stems:
                    tracks.append(StemTrack(
                        track_name=track_dir.name,
                        mix_path=mix_path,
                        stems=stems,
                        source="dsd100",
                    ))

        logger.info(f"  DSD100: {len(tracks)} tracks")
        return tracks


# ---------------------------------------------------------------------------
# Stem Training Pair Generator
# ---------------------------------------------------------------------------

def generate_stem_pairs(tracks: list[StemTrack]) -> list[StemPair]:
    """
    Generate all (mix, stem) training pairs from loaded tracks.
    Each pair teaches the model about one instrument's role in a mix.
    """
    pairs = []
    for track in tracks:
        if not track.mix_path:
            continue
        for stem_name, stem_path in track.stems.items():
            pairs.append(StemPair(
                mix_path=track.mix_path,
                stem_path=stem_path,
                stem_name=stem_name,
                track_name=track.track_name,
                source=track.source,
            ))

    logger.info(f"  Generated {len(pairs):,} stem training pairs from {len(tracks)} tracks")
    return pairs


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------

class PhaseFPipeline:
    """
    Master pipeline for Phase F multi-track stem data.
    Downloads all stem datasets and builds unified training pairs.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT, hq: bool = True):
        self.data_root = data_root
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.hq = hq

    def download_all(self, include_slakh: bool = True) -> dict[str, int]:
        """Download all stem datasets."""
        stats = {}

        # MUSDB18-HQ (always download — gold standard)
        logger.info("Downloading MUSDB18-HQ...")
        musdb = MUSDB18Downloader(self.data_root, hq=self.hq)
        musdb.download()
        tracks = musdb.load_tracks()
        stats["musdb18"] = len(tracks)

        # DSD100
        logger.info("Downloading DSD100...")
        dsd = DSD100Downloader(self.data_root)
        try:
            dsd.download()
            tracks = dsd.load_tracks()
            stats["dsd100"] = len(tracks)
        except Exception as e:
            logger.warning(f"  DSD100 download failed: {e}")
            stats["dsd100"] = 0

        # MedleyDB (metadata only — audio requires manual download)
        logger.info("Loading MedleyDB metadata...")
        mdb = MedleyDBDownloader(self.data_root)
        mdb.download_metadata()
        tracks = mdb.load_tracks()
        stats["medleydb"] = len(tracks)

        # Slakh2100 (large — 150GB, optional)
        if include_slakh:
            logger.info("Downloading Slakh2100...")
            slakh = Slakh2100Downloader(self.data_root)
            try:
                slakh.download(audio=True)
                tracks = slakh.load_tracks()
                stats["slakh2100"] = len(tracks)
            except Exception as e:
                logger.warning(f"  Slakh2100 download failed: {e}")
                stats["slakh2100"] = 0

        logger.info(f"\nPhase F stem dataset summary:")
        for source, count in stats.items():
            logger.info(f"  {source}: {count} tracks")
        logger.info(f"  TOTAL: {sum(stats.values())} tracks")

        return stats

    def load_all_pairs(self) -> list[StemPair]:
        """Load all available stem pairs from downloaded data."""
        all_tracks = []

        # MUSDB18
        musdb = MUSDB18Downloader(self.data_root, hq=self.hq)
        all_tracks.extend(musdb.load_tracks())

        # DSD100
        dsd = DSD100Downloader(self.data_root)
        all_tracks.extend(dsd.load_tracks())

        # MedleyDB
        mdb = MedleyDBDownloader(self.data_root)
        all_tracks.extend(mdb.load_tracks())

        # Slakh2100
        slakh = Slakh2100Downloader(self.data_root)
        all_tracks.extend(slakh.load_tracks())

        return generate_stem_pairs(all_tracks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="RESONATE Phase F: Stem Data Pipeline")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--no-slakh", action="store_true", help="Skip Slakh2100 (150GB)")
    parser.add_argument("--no-hq", action="store_true", help="Use MUSDB18 instead of HQ")
    args = parser.parse_args()

    pipeline = PhaseFPipeline(
        data_root=Path(args.data_root),
        hq=not args.no_hq,
    )
    pipeline.download_all(include_slakh=not args.no_slakh)
