"""
RESONATE Production Model — Phase G: Song Structure & Temporal Awareness

Teaches the model to understand song-level architecture:
  - intro, verse, chorus, bridge, drop, outro detection
  - arrangement intelligence: which instruments enter when
  - build-ups, breakdowns, and transition analysis
  - beat/downbeat/bar/phrase boundaries

Data sources (all free):
  - SALAMI:        1,400+ songs with structural section annotations
  - Harmonix Set:  912 songs with beat, downbeat, section, and phrase labels
  - Billboard:     Structural annotations for ~1,000 songs
  - RWC:           315 songs with detailed structure + chord annotations
  - Beatles:       180 songs with beat-level structure annotations
  - Spotify 1M:    1M playlists (song sequence/flow patterns)

Total: ~3,800 structurally annotated songs + 1M playlist sequences.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path.home() / ".resonate" / "datasets" / "phase_g"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StructureAnnotation:
    """A time-stamped structural annotation for a song."""
    start_time: float  # seconds
    end_time: float
    label: str  # verse, chorus, bridge, intro, outro, etc.


@dataclass
class BeatAnnotation:
    """Beat/downbeat position in a song."""
    time: float
    beat_type: str  # beat, downbeat, phrase


@dataclass
class AnnotatedSong:
    """A song with full structural and beat annotations."""
    track_id: str
    title: str
    artist: str = ""
    audio_path: str = ""
    source: str = ""
    duration: float = 0.0
    sections: list[StructureAnnotation] = field(default_factory=list)
    beats: list[BeatAnnotation] = field(default_factory=list)
    chords: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SALAMI (1,400+ songs with structural annotations)
# ---------------------------------------------------------------------------

SALAMI_ANNOTATIONS_URL = "https://github.com/DDMAL/salami-data-public/archive/refs/heads/master.zip"

class SALAMIDownloader:
    """
    Downloads SALAMI (Structural Analysis of Large Amounts of Music Information).
    1,400+ songs with human-annotated section boundaries and labels.
    Annotations are free; audio must be matched with user's own collection.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "salami"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_annotations(self) -> Path:
        """Download SALAMI annotation data from GitHub."""
        zip_path = self.data_dir / "salami-data-public.zip"
        extract_dir = self.data_dir / "salami-data-public-master"

        if extract_dir.exists():
            logger.info("  SALAMI annotations already downloaded")
            return extract_dir

        if not zip_path.exists():
            logger.info("  Downloading SALAMI annotations...")
            subprocess.run([
                "wget", "-q", SALAMI_ANNOTATIONS_URL, "-O", str(zip_path),
            ], check=True)

        logger.info("  Extracting SALAMI...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(self.data_dir)], check=True)
        zip_path.unlink(missing_ok=True)
        return extract_dir

    def load_annotations(self) -> list[AnnotatedSong]:
        """Parse SALAMI annotation files."""
        base = self.data_dir / "salami-data-public-master" / "annotations"
        songs = []

        if not base.exists():
            logger.warning("  SALAMI annotations not found — download first")
            return songs

        for track_dir in sorted(base.iterdir()):
            if not track_dir.is_dir():
                continue

            track_id = track_dir.name
            sections = []

            # SALAMI has two annotator files per track
            for ann_file in ["textfile1_uppercase.txt", "textfile2_uppercase.txt",
                             "textfile1_functions.txt", "textfile2_functions.txt"]:
                ann_path = track_dir / ann_file
                if ann_path.exists():
                    sections = self._parse_annotation_file(ann_path)
                    if sections:
                        break  # Use first valid annotation

            if sections:
                songs.append(AnnotatedSong(
                    track_id=track_id,
                    title=track_id,
                    source="salami",
                    sections=sections,
                ))

        logger.info(f"  SALAMI: {len(songs)} annotated songs loaded")
        return songs

    def _parse_annotation_file(self, path: Path) -> list[StructureAnnotation]:
        """Parse a SALAMI annotation text file."""
        sections = []
        lines = path.read_text().strip().split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    start_time = float(parts[0])
                    label = parts[1].strip()

                    # Determine end time from next annotation
                    end_time = start_time + 30  # default
                    if i + 1 < len(lines):
                        next_parts = lines[i + 1].strip().split("\t")
                        if next_parts and next_parts[0]:
                            try:
                                end_time = float(next_parts[0])
                            except ValueError:
                                pass

                    sections.append(StructureAnnotation(
                        start_time=start_time,
                        end_time=end_time,
                        label=label,
                    ))
                except ValueError:
                    continue

        return sections


# ---------------------------------------------------------------------------
# Harmonix Set (912 songs with beats + sections + phrases)
# ---------------------------------------------------------------------------

HARMONIX_URL = "https://github.com/urinieto/harmonixset/archive/refs/heads/master.zip"

class HarmonixDownloader:
    """
    Downloads Harmonix Set: 912 songs with beat, downbeat, section,
    and phrase-level annotations. Created from the Rock Band video game.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "harmonix"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_annotations(self) -> Path:
        zip_path = self.data_dir / "harmonixset.zip"
        extract_dir = self.data_dir / "harmonixset-master"

        if extract_dir.exists():
            logger.info("  Harmonix Set already downloaded")
            return extract_dir

        if not zip_path.exists():
            logger.info("  Downloading Harmonix Set annotations...")
            subprocess.run([
                "wget", "-q", HARMONIX_URL, "-O", str(zip_path),
            ], check=True)

        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(self.data_dir)], check=True)
        zip_path.unlink(missing_ok=True)
        return extract_dir

    def load_annotations(self) -> list[AnnotatedSong]:
        """Load Harmonix Set annotations."""
        base = self.data_dir / "harmonixset-master" / "dataset"
        songs = []

        if not base.exists():
            # Try alternative paths
            for candidate in [
                self.data_dir / "harmonixset-master",
                self.data_dir,
            ]:
                if (candidate / "beats_and_downbeats").exists():
                    base = candidate
                    break

        beats_dir = base / "beats_and_downbeats"
        sections_dir = base / "segments"

        if not beats_dir.exists() and not sections_dir.exists():
            logger.warning("  Harmonix annotations not found")
            return songs

        # Load sections
        if sections_dir and sections_dir.exists():
            for ann_file in sorted(sections_dir.glob("*.txt")):
                track_id = ann_file.stem
                sections = []

                for line in ann_file.read_text().strip().split("\n"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        try:
                            start = float(parts[0])
                            label = parts[1] if len(parts) > 1 else "unknown"
                            sections.append(StructureAnnotation(
                                start_time=start,
                                end_time=start + 10,  # Updated when next section found
                                label=label,
                            ))
                        except ValueError:
                            continue

                # Fix end times
                for i in range(len(sections) - 1):
                    sections[i].end_time = sections[i + 1].start_time

                # Load beats for this track
                beats = []
                beat_file = beats_dir / f"{track_id}.txt" if beats_dir else None
                if beat_file and beat_file.exists():
                    for line in beat_file.read_text().strip().split("\n"):
                        parts = line.strip().split("\t")
                        if parts:
                            try:
                                t = float(parts[0])
                                beat_type = "downbeat" if len(parts) > 1 and parts[1] == "1" else "beat"
                                beats.append(BeatAnnotation(time=t, beat_type=beat_type))
                            except ValueError:
                                continue

                if sections:
                    songs.append(AnnotatedSong(
                        track_id=track_id,
                        title=track_id,
                        source="harmonix",
                        sections=sections,
                        beats=beats,
                    ))

        logger.info(f"  Harmonix: {len(songs)} annotated songs loaded")
        return songs


# ---------------------------------------------------------------------------
# Billboard Structure Annotations (~1,000 songs)
# ---------------------------------------------------------------------------

BILLBOARD_SALAMI_URL = "https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1"
BILLBOARD_INDEX_URL = "https://www.dropbox.com/s/o0olz0uwl9z9stb/billboard-2.0-index.csv?dl=1"
BILLBOARD_GITHUB_MIRROR = "https://github.com/boomerr1/The-McGill-Billboard-Project/archive/refs/heads/main.zip"

class BillboardStructureDownloader:
    """
    Downloads McGill Billboard chord/structure annotations.
    890 Billboard chart songs (1950s-1990s) with:
      - Section boundaries (intro, verse, chorus, bridge, etc.)
      - Full chord progressions with timing
      - Metadata: title, artist, metre, tonic key
    Source: DDMAL at McGill University (Burgoyne, Wild & Fujinaga, ISMIR 2011)
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "billboard_structure"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_annotations(self) -> Path:
        """Download McGill Billboard from Dropbox (official) or GitHub mirror."""
        extract_dir = self.data_dir / "McGill-Billboard"

        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info("  Billboard annotations already downloaded")
            return extract_dir

        tar_path = self.data_dir / "billboard-2.0-salami_chords.tar.gz"

        if not tar_path.exists():
            logger.info("  Downloading McGill Billboard annotations (890 songs)...")
            # Try Dropbox (official source)
            result = subprocess.run([
                "curl", "-sL", BILLBOARD_SALAMI_URL, "-o", str(tar_path),
            ], check=False)

            if result.returncode != 0 or not tar_path.exists() or tar_path.stat().st_size < 1000:
                # Fallback: GitHub mirror
                logger.info("  Dropbox failed, trying GitHub mirror...")
                zip_path = self.data_dir / "billboard-github.zip"
                subprocess.run([
                    "curl", "-sL", BILLBOARD_GITHUB_MIRROR, "-o", str(zip_path),
                ], check=False)
                if zip_path.exists():
                    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(self.data_dir)], check=False)
                    zip_path.unlink(missing_ok=True)
                    return extract_dir

        if tar_path.exists() and tar_path.stat().st_size > 1000:
            logger.info("  Extracting Billboard annotations...")
            subprocess.run(["tar", "xzf", str(tar_path), "-C", str(self.data_dir)], check=True)
            tar_path.unlink(missing_ok=True)

        # Download index CSV (title + artist metadata)
        index_path = self.data_dir / "billboard-2.0-index.csv"
        if not index_path.exists():
            subprocess.run([
                "curl", "-sL", BILLBOARD_INDEX_URL, "-o", str(index_path),
            ], check=False)

        logger.info(f"  Billboard annotations ready at {extract_dir}")
        return extract_dir

    def _load_index(self) -> dict[str, dict]:
        """Load the Billboard index CSV for title/artist metadata."""
        index_path = self.data_dir / "billboard-2.0-index.csv"
        index = {}
        if index_path.exists():
            for line in index_path.read_text().strip().split("\n")[1:]:  # skip header
                parts = line.split(",")
                if len(parts) >= 3:
                    track_id = parts[0].strip()
                    title = parts[1].strip().strip('"')
                    artist = parts[2].strip().strip('"')
                    index[track_id] = {"title": title, "artist": artist}
        return index

    def load_annotations(self) -> list[AnnotatedSong]:
        """Load Billboard structure annotations with rich metadata."""
        songs = []
        index = self._load_index()

        # Search for annotation directories containing salami_chords.txt
        for base in self.data_dir.iterdir():
            if not base.is_dir():
                continue

            for ann_dir in sorted(base.rglob("*")):
                if not ann_dir.is_dir():
                    continue
                chord_file = ann_dir / "salami_chords.txt"
                if not chord_file.exists():
                    continue

                track_id = ann_dir.name
                sections = []
                chords = []
                metadata = {}

                # Parse the salami_chords.txt file
                for line in chord_file.read_text().strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    # Extract metadata from comment headers
                    if line.startswith("#"):
                        if ":" in line:
                            key, val = line[1:].split(":", 1)
                            key = key.strip().lower()
                            val = val.strip()
                            metadata[key] = val
                        continue

                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            t = float(parts[0])
                            label_raw = parts[1]

                            # Parse section markers: "A, intro, | C:maj | ..."
                            # Section labels contain a comma-separated letter + name
                            if "," in label_raw:
                                label_parts = label_raw.split(",")
                                section_letter = label_parts[0].strip()
                                section_name = label_parts[1].strip().split("|")[0].strip().rstrip(",")

                                section_label = f"{section_name}" if section_name else section_letter
                                sections.append(StructureAnnotation(
                                    start_time=t,
                                    end_time=t + 10,  # Fixed below
                                    label=section_label,
                                ))

                            # Extract chord progression
                            chord_matches = [c.strip() for c in label_raw.split("|") if ":" in c]
                            for chord in chord_matches:
                                chords.append({"time": t, "chord": chord.strip()})

                        except ValueError:
                            continue

                # Fix section end times
                for i in range(len(sections) - 1):
                    sections[i].end_time = sections[i + 1].start_time

                if sections:
                    # Merge with index metadata
                    idx = index.get(track_id, {})
                    title = metadata.get("title", idx.get("title", track_id))
                    artist = metadata.get("artist", idx.get("artist", ""))

                    songs.append(AnnotatedSong(
                        track_id=track_id,
                        title=title,
                        artist=artist,
                        source="billboard_structure",
                        sections=sections,
                        chords=chords,
                        metadata={
                            "tonic": metadata.get("tonic", ""),
                            "metre": metadata.get("metre", ""),
                        },
                    ))

        logger.info(f"  Billboard: {len(songs)} annotated songs (with chords + structure + key)")
        return songs


# ---------------------------------------------------------------------------
# Spotify Million Playlist (1M playlists — song flow patterns)
# ---------------------------------------------------------------------------

class SpotifyMPDLoader:
    """
    Loads the Spotify Million Playlist Dataset.
    1M playlists → learn song-to-song flow and sequencing patterns.
    Must be downloaded manually from AIcrowd (requires registration).
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "spotify_mpd"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_playlists(self, max_playlists: int = 0) -> list[dict]:
        """Load playlists from MPD JSON slices."""
        playlists = []
        json_dir = self.data_dir / "data"

        if not json_dir.exists():
            logger.info("  Spotify MPD not found. Download from: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge")
            return playlists

        for json_file in sorted(json_dir.glob("mpd.slice.*.json")):
            with open(json_file) as f:
                data = json.load(f)
                for pl in data.get("playlists", []):
                    playlists.append({
                        "name": pl.get("name", ""),
                        "tracks": [
                            {
                                "track_uri": t.get("track_uri", ""),
                                "track_name": t.get("track_name", ""),
                                "artist_name": t.get("artist_name", ""),
                                "album_name": t.get("album_name", ""),
                                "duration_ms": t.get("duration_ms", 0),
                                "pos": t.get("pos", 0),
                            }
                            for t in pl.get("tracks", [])
                        ],
                        "num_tracks": pl.get("num_tracks", 0),
                        "num_followers": pl.get("num_followers", 0),
                    })

                    if max_playlists and len(playlists) >= max_playlists:
                        break

            if max_playlists and len(playlists) >= max_playlists:
                break

        logger.info(f"  Spotify MPD: {len(playlists):,} playlists loaded")
        return playlists


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------

class PhaseGPipeline:
    """
    Master pipeline for Phase G song structure data.
    Downloads all annotation datasets and builds the unified training set.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_root = data_root
        self.data_root.mkdir(parents=True, exist_ok=True)

    def download_all(self) -> dict[str, int]:
        """Download all structure annotation datasets."""
        stats = {}

        # SALAMI
        logger.info("Downloading SALAMI annotations...")
        salami = SALAMIDownloader(self.data_root)
        try:
            salami.download_annotations()
            songs = salami.load_annotations()
            stats["salami"] = len(songs)
        except Exception as e:
            logger.warning(f"  SALAMI failed: {e}")
            stats["salami"] = 0

        # Harmonix
        logger.info("Downloading Harmonix Set...")
        harmonix = HarmonixDownloader(self.data_root)
        try:
            harmonix.download_annotations()
            songs = harmonix.load_annotations()
            stats["harmonix"] = len(songs)
        except Exception as e:
            logger.warning(f"  Harmonix failed: {e}")
            stats["harmonix"] = 0

        # Billboard Structure
        logger.info("Downloading Billboard structure annotations...")
        bb = BillboardStructureDownloader(self.data_root)
        try:
            bb.download_annotations()
            songs = bb.load_annotations()
            stats["billboard"] = len(songs)
        except Exception as e:
            logger.warning(f"  Billboard structure failed: {e}")
            stats["billboard"] = 0

        logger.info(f"\nPhase G structure annotation summary:")
        for source, count in stats.items():
            logger.info(f"  {source}: {count} annotated songs")
        logger.info(f"  TOTAL: {sum(stats.values())} songs")

        return stats

    def load_all_songs(self) -> list[AnnotatedSong]:
        """Load all available structural annotations."""
        all_songs = []

        salami = SALAMIDownloader(self.data_root)
        all_songs.extend(salami.load_annotations())

        harmonix = HarmonixDownloader(self.data_root)
        all_songs.extend(harmonix.load_annotations())

        bb = BillboardStructureDownloader(self.data_root)
        all_songs.extend(bb.load_annotations())

        logger.info(f"  Total: {len(all_songs)} structurally annotated songs")
        return all_songs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="RESONATE Phase G: Structure Data Pipeline")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    args = parser.parse_args()

    pipeline = PhaseGPipeline(data_root=Path(args.data_root))
    pipeline.download_all()
