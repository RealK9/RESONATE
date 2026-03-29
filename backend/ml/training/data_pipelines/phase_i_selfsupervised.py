"""
RESONATE Production Model — Phase I: Self-Supervised Massive Audio Pre-Training

The scale play. Train on vast amounts of unlabeled audio using self-supervised
objectives — no labels needed. The model learns deep audio representations
purely from the structure of sound itself.

Training objectives:
  - Masked spectrogram modeling: mask patches, predict what's missing (like BERT)
  - Audio contrastive: different segments of same song should be nearby
  - Next-segment prediction: predict what comes next in a song
  - Time-frequency jigsaw: predict the arrangement of shuffled patches

Data sources (all free, no labels needed):
  - FMA-full:          106k tracks, 343 days of CC-licensed audio
  - MTG-Jamendo:       55k tracks (already downloading)
  - Internet Archive:  Millions of public domain recordings
  - LibriVox:          100k+ hours of public domain audiobooks
  - Common Voice:      19,000+ hours across 100+ languages
  - AudioSet (full):   5,800 hours across every sound category

Total: ~25,000+ hours of audio, all free.
"""
from __future__ import annotations

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

DEFAULT_DATA_ROOT = Path.home() / ".resonate" / "datasets" / "phase_i"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AudioSegment:
    """An unlabeled audio segment for self-supervised training."""
    audio_path: str
    source: str
    duration: float = 0.0
    sample_rate: int = 16000
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FMA Full (106k tracks, 343 days)
# ---------------------------------------------------------------------------

FMA_URLS = {
    "small": ("https://os.unil.cloud.switch.ch/fma/fma_small.zip", 7.2),
    "medium": ("https://os.unil.cloud.switch.ch/fma/fma_medium.zip", 22.0),
    "large": ("https://os.unil.cloud.switch.ch/fma/fma_large.zip", 93.0),
    "full": ("https://os.unil.cloud.switch.ch/fma/fma_full.zip", 879.0),
}

class FMAFullDownloader:
    """
    Downloads FMA (Free Music Archive) at various scales.
    Full version: 106,574 tracks, 161 unbalanced genres, 343 days of audio.
    All tracks are Creative Commons licensed.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "fma"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self, size: str = "large") -> Path:
        """
        Download FMA dataset at specified scale.
        Sizes: small (8k/7.2GB), medium (25k/22GB), large (106k/93GB), full (106k/879GB)
        Recommendation: 'large' is the sweet spot for self-supervised training.
        """
        if size not in FMA_URLS:
            raise ValueError(f"Unknown FMA size: {size}. Choose: {list(FMA_URLS.keys())}")

        url, gb = FMA_URLS[size]
        extract_dir = self.data_dir / f"fma_{size}"

        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"  FMA-{size} already exists at {extract_dir}")
            return extract_dir

        zip_path = self.data_dir / f"fma_{size}.zip"
        if not zip_path.exists():
            logger.info(f"  Downloading FMA-{size} ({gb:.0f}GB)...")
            subprocess.run([
                "wget", "-q", "--show-progress", url, "-O", str(zip_path),
            ], check=True)

        logger.info(f"  Extracting FMA-{size}...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(self.data_dir)], check=True)
        zip_path.unlink(missing_ok=True)

        return extract_dir

    def download_metadata(self) -> Path:
        """Download FMA metadata (track info, genres, etc.)."""
        meta_url = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
        meta_dir = self.data_dir / "fma_metadata"

        if meta_dir.exists():
            return meta_dir

        zip_path = self.data_dir / "fma_metadata.zip"
        if not zip_path.exists():
            subprocess.run(["wget", "-q", meta_url, "-O", str(zip_path)], check=True)
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(self.data_dir)], check=True)
        zip_path.unlink(missing_ok=True)
        return meta_dir

    def load_segments(self, size: str = "large") -> list[AudioSegment]:
        """Load all FMA tracks as audio segments."""
        extract_dir = self.data_dir / f"fma_{size}"
        segments = []

        if not extract_dir.exists():
            logger.warning(f"  FMA-{size} not downloaded yet")
            return segments

        for mp3_file in extract_dir.rglob("*.mp3"):
            segments.append(AudioSegment(
                audio_path=str(mp3_file),
                source=f"fma_{size}",
                duration=30.0,  # FMA clips are 30 seconds
            ))

        logger.info(f"  FMA-{size}: {len(segments):,} audio segments")
        return segments


# ---------------------------------------------------------------------------
# Internet Archive (millions of public domain recordings)
# ---------------------------------------------------------------------------

IA_API = "https://archive.org"

class InternetArchiveDownloader:
    """
    Downloads audio from the Internet Archive.
    Millions of public domain recordings across all genres and eras.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "internet_archive"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    def search_collections(
        self,
        query: str = "music",
        mediatype: str = "audio",
        max_items: int = 1000,
    ) -> list[dict]:
        """Search Internet Archive for audio collections."""
        items = []
        page = 1
        rows = 100

        while len(items) < max_items:
            params = {
                "q": f"{query} AND mediatype:{mediatype}",
                "fl[]": ["identifier", "title", "creator", "date", "downloads", "format"],
                "sort[]": "downloads desc",
                "rows": rows,
                "page": page,
                "output": "json",
            }

            try:
                resp = self._session.get(
                    f"{IA_API}/advancedsearch.php",
                    params=params,
                    timeout=30,
                )
                if resp.status_code != 200:
                    break

                data = resp.json()
                docs = data.get("response", {}).get("docs", [])
                if not docs:
                    break

                items.extend(docs)
                page += 1
                import time
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"  IA search error: {e}")
                break

        logger.info(f"  Internet Archive: {len(items):,} items found for '{query}'")
        return items[:max_items]

    def download_item_audio(self, identifier: str, max_files: int = 10) -> list[AudioSegment]:
        """Download audio files from a single IA item."""
        segments = []
        item_dir = self.data_dir / "audio" / identifier
        item_dir.mkdir(parents=True, exist_ok=True)

        try:
            resp = self._session.get(
                f"{IA_API}/metadata/{identifier}/files",
                timeout=30,
            )
            if resp.status_code != 200:
                return segments

            files = resp.json().get("result", [])
            audio_files = [
                f for f in files
                if f.get("format", "").lower() in ("mp3", "ogg vorbis", "flac", "wav", "vbr mp3", "128kbps mp3")
                   or f.get("name", "").lower().endswith((".mp3", ".ogg", ".flac", ".wav"))
            ]

            for f in audio_files[:max_files]:
                fname = f.get("name", "")
                if not fname:
                    continue

                local_path = item_dir / fname
                if not local_path.exists():
                    url = f"https://archive.org/download/{identifier}/{fname}"
                    try:
                        result = subprocess.run([
                            "wget", "-q", url, "-O", str(local_path),
                        ], timeout=120, check=False)
                        if result.returncode != 0:
                            local_path.unlink(missing_ok=True)
                            continue
                    except Exception:
                        local_path.unlink(missing_ok=True)
                        continue

                if local_path.exists() and local_path.stat().st_size > 0:
                    segments.append(AudioSegment(
                        audio_path=str(local_path),
                        source="internet_archive",
                        metadata={"identifier": identifier, "filename": fname},
                    ))

        except Exception as e:
            logger.warning(f"  IA download error for {identifier}: {e}")

        return segments

    def download_music_collections(
        self,
        queries: Optional[list[str]] = None,
        max_items_per_query: int = 200,
        max_files_per_item: int = 5,
    ) -> list[AudioSegment]:
        """Download diverse music from Internet Archive."""
        if queries is None:
            queries = [
                "jazz recordings", "blues music", "classical music recordings",
                "rock music", "folk music", "world music", "electronic music",
                "soul music", "funk music", "reggae music",
                "hip hop music", "country music", "punk music",
                "ambient music", "experimental music",
            ]

        all_segments = []

        for query in queries:
            items = self.search_collections(query, max_items=max_items_per_query)
            for item in tqdm(items[:50], desc=f"IA/{query}", unit="item"):
                identifier = item.get("identifier", "")
                if identifier:
                    segments = self.download_item_audio(identifier, max_files=max_files_per_item)
                    all_segments.extend(segments)

        logger.info(f"  Internet Archive total: {len(all_segments):,} audio segments")
        return all_segments


# ---------------------------------------------------------------------------
# Common Voice (19,000+ hours, 100+ languages)
# ---------------------------------------------------------------------------

class CommonVoiceDownloader:
    """
    Downloads Mozilla Common Voice — 19,000+ hours of validated speech.
    Useful for vocal understanding, timbre diversity, and multilingual audio.
    Requires Mozilla account for download.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "common_voice"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def check_availability(self) -> bool:
        """Check if Common Voice data is available locally."""
        clips_dir = self.data_dir / "clips"
        if clips_dir.exists():
            count = sum(1 for _ in clips_dir.rglob("*.mp3"))
            if count > 0:
                logger.info(f"  Common Voice: {count:,} clips available")
                return True

        logger.info("  Common Voice not found locally.")
        logger.info("  Download from: https://commonvoice.mozilla.org/en/datasets")
        logger.info("  Extract to: {self.data_dir}/clips/")
        return False

    def load_segments(self, max_segments: int = 0) -> list[AudioSegment]:
        """Load Common Voice audio clips."""
        segments = []
        clips_dir = self.data_dir / "clips"

        if not clips_dir.exists():
            return segments

        for audio_file in clips_dir.rglob("*.mp3"):
            segments.append(AudioSegment(
                audio_path=str(audio_file),
                source="common_voice",
            ))
            if max_segments and len(segments) >= max_segments:
                break

        logger.info(f"  Common Voice: {len(segments):,} segments loaded")
        return segments


# ---------------------------------------------------------------------------
# LibriVox / LibriSpeech (1,000+ hours)
# ---------------------------------------------------------------------------

LIBRISPEECH_URLS = {
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}

class LibriSpeechDownloader:
    """
    Downloads LibriSpeech — 1,000 hours of English read speech.
    Useful for vocal timbre understanding and voice separation.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "librispeech"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self, splits: Optional[list[str]] = None) -> Path:
        """Download LibriSpeech splits."""
        if splits is None:
            splits = ["train-clean-100"]  # Start small, 100 hours

        for split in splits:
            if split not in LIBRISPEECH_URLS:
                continue

            split_dir = self.data_dir / "LibriSpeech" / split
            if split_dir.exists():
                logger.info(f"  LibriSpeech {split} already downloaded")
                continue

            url = LIBRISPEECH_URLS[split]
            tar_path = self.data_dir / f"{split}.tar.gz"

            if not tar_path.exists():
                logger.info(f"  Downloading LibriSpeech {split}...")
                subprocess.run([
                    "wget", "-q", "--show-progress", url, "-O", str(tar_path),
                ], check=True)

            logger.info(f"  Extracting LibriSpeech {split}...")
            subprocess.run(["tar", "xzf", str(tar_path), "-C", str(self.data_dir)], check=True)
            tar_path.unlink(missing_ok=True)

        return self.data_dir / "LibriSpeech"

    def load_segments(self) -> list[AudioSegment]:
        """Load all LibriSpeech audio segments."""
        segments = []
        base = self.data_dir / "LibriSpeech"

        if not base.exists():
            return segments

        for flac_file in base.rglob("*.flac"):
            segments.append(AudioSegment(
                audio_path=str(flac_file),
                source="librispeech",
            ))

        logger.info(f"  LibriSpeech: {len(segments):,} audio segments")
        return segments


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------

class PhaseIPipeline:
    """
    Master pipeline for Phase I self-supervised data.
    Collects massive amounts of unlabeled audio for pre-training.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_root = data_root
        self.data_root.mkdir(parents=True, exist_ok=True)

    def download_all(
        self,
        fma_size: str = "large",
        include_ia: bool = True,
        include_librispeech: bool = True,
        librispeech_splits: Optional[list[str]] = None,
    ) -> dict[str, int]:
        """Download all Phase I audio sources."""
        stats = {}

        # FMA (large = 106k tracks, 93GB)
        logger.info(f"Downloading FMA-{fma_size}...")
        fma = FMAFullDownloader(self.data_root)
        try:
            fma.download(size=fma_size)
            fma.download_metadata()
            segments = fma.load_segments(size=fma_size)
            stats["fma"] = len(segments)
        except Exception as e:
            logger.warning(f"  FMA download failed: {e}")
            stats["fma"] = 0

        # Internet Archive
        if include_ia:
            logger.info("Downloading from Internet Archive...")
            ia = InternetArchiveDownloader(self.data_root)
            try:
                segments = ia.download_music_collections(
                    max_items_per_query=100,
                    max_files_per_item=3,
                )
                stats["internet_archive"] = len(segments)
            except Exception as e:
                logger.warning(f"  IA download failed: {e}")
                stats["internet_archive"] = 0

        # LibriSpeech
        if include_librispeech:
            logger.info("Downloading LibriSpeech...")
            ls = LibriSpeechDownloader(self.data_root)
            try:
                ls.download(splits=librispeech_splits or ["train-clean-100"])
                segments = ls.load_segments()
                stats["librispeech"] = len(segments)
            except Exception as e:
                logger.warning(f"  LibriSpeech download failed: {e}")
                stats["librispeech"] = 0

        # Common Voice (check only — requires manual download)
        cv = CommonVoiceDownloader(self.data_root)
        if cv.check_availability():
            segments = cv.load_segments()
            stats["common_voice"] = len(segments)

        logger.info(f"\nPhase I self-supervised audio summary:")
        for source, count in stats.items():
            logger.info(f"  {source}: {count:,} segments")
        logger.info(f"  TOTAL: {sum(stats.values()):,} segments")

        return stats

    def load_all_segments(self) -> list[AudioSegment]:
        """Load all available audio segments from downloaded data."""
        all_segments = []

        # FMA
        fma = FMAFullDownloader(self.data_root)
        for size in ["large", "medium", "small"]:
            segments = fma.load_segments(size=size)
            if segments:
                all_segments.extend(segments)
                break  # Use largest available

        # LibriSpeech
        ls = LibriSpeechDownloader(self.data_root)
        all_segments.extend(ls.load_segments())

        # Common Voice
        cv = CommonVoiceDownloader(self.data_root)
        all_segments.extend(cv.load_segments())

        logger.info(f"  Total Phase I segments: {len(all_segments):,}")
        return all_segments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="RESONATE Phase I: Self-Supervised Data Pipeline")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--fma-size", default="large", choices=["small", "medium", "large", "full"])
    parser.add_argument("--no-ia", action="store_true", help="Skip Internet Archive")
    parser.add_argument("--no-librispeech", action="store_true")
    args = parser.parse_args()

    pipeline = PhaseIPipeline(data_root=Path(args.data_root))
    pipeline.download_all(
        fma_size=args.fma_size,
        include_ia=not args.no_ia,
        include_librispeech=not args.no_librispeech,
    )
