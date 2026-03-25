"""
Batch processing pipeline. Discovers audio files, analyzes in parallel,
persists to database with progress reporting.
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from ml.pipeline.ingestion import analyze_sample
from ml.db.sample_store import SampleStore

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}


class BatchProcessor:
    """Process directories of audio files in parallel."""

    def __init__(self, skip_embeddings: bool = False,
                 embedding_manager=None,
                 db_path: str | None = None,
                 max_workers: int = 4):
        self.skip_embeddings = skip_embeddings
        self.embedding_manager = embedding_manager
        self.max_workers = max_workers
        self.store = None
        if db_path:
            self.store = SampleStore(db_path)
            self.store.init()

    def discover_audio_files(self, directory: str) -> list[str]:
        """Find all audio files in a directory tree."""
        files = []
        for root, _, filenames in os.walk(directory):
            for fname in filenames:
                if Path(fname).suffix.lower() in AUDIO_EXTENSIONS:
                    files.append(os.path.join(root, fname))
        return sorted(files)

    def process_directory(self, directory: str,
                          on_progress: Callable[[int, int], None] | None = None,
                          source: str = "local") -> dict:
        """
        Process all audio files in a directory.
        Returns {"processed": int, "failed": int, "skipped": int, "total": int}.
        """
        files = self.discover_audio_files(directory)
        total = len(files)
        processed = 0
        failed = 0
        skipped = 0

        def _process_one(filepath: str):
            file_hash = _quick_hash(filepath)
            if self.store and not self.store.needs_reanalysis(filepath, file_hash):
                return "skipped"
            try:
                profile = analyze_sample(
                    filepath,
                    skip_embeddings=self.skip_embeddings,
                    embedding_manager=self.embedding_manager,
                    file_hash=file_hash,
                    source=source,
                )
                if self.store:
                    self.store.save(profile)
                return "ok"
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                return "failed"

        # Process with thread pool (IO-bound file reading benefits from threads)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_process_one, f): f for f in files}
            done_count = 0
            for future in as_completed(futures):
                result = future.result()
                if result == "ok":
                    processed += 1
                elif result == "failed":
                    failed += 1
                else:
                    skipped += 1
                done_count += 1
                if on_progress:
                    on_progress(done_count, total)

        return {"processed": processed, "failed": failed, "skipped": skipped, "total": total}


def _quick_hash(filepath: str) -> str:
    """Quick hash based on path + size + mtime (no content reading)."""
    stat = os.stat(filepath)
    return f"{filepath}:{stat.st_size}:{stat.st_mtime}"
