#!/usr/bin/env python3
"""
RESONATE Batch Indexer — Analyze all samples with real ML models and build FAISS index.

Processes samples in phases:
  1. DSP features (fast, ~20ms/sample)
  2. ML embeddings (CLAP/PANNs/AST, ~1-3s/sample on MPS)
  3. FAISS index build from CLAP embeddings

Saves profiles to SQLite DB (sample_profiles.db) and FAISS index to disk.
Supports incremental indexing — skips already-processed samples by file hash.
"""
import sys
import os
import time
import json
import hashlib
import sqlite3
import logging
import signal
import faulthandler
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Enable faulthandler for crash debugging
faulthandler.enable()

# Limit thread usage to avoid memory pressure
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.pipeline.ingestion import analyze_sample
from ml.embeddings.embedding_manager import EmbeddingManager
from ml.retrieval.vector_index import VectorIndex

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("batch_index")

AUDIO_EXT = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg"}
DB_PATH = Path(__file__).parent / "sample_profiles.db"
INDEX_DIR = Path(__file__).parent / "faiss_index"
SAMPLES_DIR = Path(__file__).parent / "samples"

# ── Database ──

def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sample_profiles (
            filepath TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            indexed_at REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON sample_profiles(file_hash)")
    conn.commit()
    return conn


def get_indexed_hashes(conn: sqlite3.Connection) -> dict:
    """Return {filepath: file_hash} for all indexed samples."""
    rows = conn.execute("SELECT filepath, file_hash FROM sample_profiles").fetchall()
    return {r[0]: r[1] for r in rows}


def save_profile(conn: sqlite3.Connection, filepath: str, file_hash: str, profile):
    """Save a SampleProfile to the database."""
    profile_dict = profile.to_dict() if hasattr(profile, "to_dict") else {}
    conn.execute(
        "INSERT OR REPLACE INTO sample_profiles (filepath, file_hash, profile_json, indexed_at) "
        "VALUES (?, ?, ?, ?)",
        (filepath, file_hash, json.dumps(profile_dict), time.time()),
    )


# ── File hashing ──

def file_hash(path: Path) -> str:
    """Fast hash: first 64KB + file size."""
    h = hashlib.md5()
    h.update(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


# ── Main ──

def discover_samples(samples_dir: Path) -> list:
    """Find all audio files recursively."""
    files = []
    for f in samples_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in AUDIO_EXT:
            files.append(f)
    return sorted(files)


def batch_hash(files: list) -> dict:
    """Hash all files using thread pool (I/O bound)."""
    hashes = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(file_hash, f): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                hashes[str(f)] = fut.result()
            except Exception:
                pass
    return hashes


def main():
    print("🔬 RESONATE Batch Indexer")
    print(f"   Samples dir: {SAMPLES_DIR}")
    print(f"   Database: {DB_PATH}")
    print(f"   FAISS index: {INDEX_DIR}")
    print()

    # Discover samples
    files = discover_samples(SAMPLES_DIR)
    print(f"   Found {len(files)} audio files")

    # Hash all files
    print("   Hashing files...")
    t0 = time.time()
    hashes = batch_hash(files)
    print(f"   Hashed in {time.time() - t0:.1f}s")

    # Check what's already indexed
    conn = init_db(DB_PATH)
    indexed = get_indexed_hashes(conn)

    to_process = []
    for f in files:
        fp = str(f)
        h = hashes.get(fp)
        if h and indexed.get(fp) == h:
            continue  # Already indexed with same hash
        to_process.append((f, h))

    print(f"   Already indexed: {len(files) - len(to_process)}")
    print(f"   Need processing: {len(to_process)}")
    print()

    if not to_process:
        print("   Nothing new to index. Building FAISS index from existing profiles...")
        build_faiss_index(conn)
        conn.close()
        return

    # Check mode: --dsp-only skips ML embeddings (fast), --ml-only adds embeddings to existing profiles
    dsp_only = "--dsp-only" in sys.argv
    ml_only = "--ml-only" in sys.argv

    emb_mgr = None
    if not dsp_only:
        print("   Loading ML models (CPU)...")
        t0 = time.time()
        emb_mgr = EmbeddingManager(device="cpu")
        _ = emb_mgr.clap
        _ = emb_mgr.panns
        _ = emb_mgr.ast
        print(f"   Models loaded in {time.time() - t0:.1f}s (device: {emb_mgr.device})")
    else:
        print("   DSP-only mode — skipping ML embeddings (fast)")
    print()

    # Process samples
    import gc
    errors = 0
    processed = 0
    batch_t0 = time.time()

    for i, (f, h) in enumerate(to_process):
        try:
            # Skip files that are too large (>50MB) or too small (<100 bytes)
            fsize = f.stat().st_size
            if fsize > 50_000_000 or fsize < 100:
                errors += 1
                continue

            profile = analyze_sample(
                str(f),
                skip_embeddings=(emb_mgr is None),
                embedding_manager=emb_mgr,
                file_hash=h or "",
                source="local",
            )
            save_profile(conn, str(f), h or "", profile)
            processed += 1

            # Commit every 50 samples and free memory
            if processed % 50 == 0:
                conn.commit()
            if processed % 200 == 0:
                gc.collect()

        except KeyboardInterrupt:
            print("\n   Interrupted! Saving progress...")
            conn.commit()
            break
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"\n   ERR: {f.name}: {e}", flush=True)

        # Progress
        elapsed = time.time() - batch_t0
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(to_process) - i - 1) / rate if rate > 0 else 0
        print(
            f"\r   [{i+1}/{len(to_process)}] "
            f"{rate:.1f} samples/sec, "
            f"ETA {int(eta)}s, "
            f"{errors} errors",
            end="",
            flush=True,
        )

    conn.commit()
    total_time = time.time() - batch_t0
    print(f"\n\n   Processed {processed} samples in {total_time:.0f}s ({processed/total_time:.1f}/sec)")
    print(f"   Errors: {errors}")

    # Build FAISS index
    print()
    build_faiss_index(conn)
    conn.close()


def build_faiss_index(conn: sqlite3.Connection):
    """Build FAISS index from CLAP embeddings in the database."""
    print("   Building FAISS index from CLAP embeddings...")
    t0 = time.time()

    rows = conn.execute("SELECT filepath, profile_json FROM sample_profiles").fetchall()

    # CLAP = 512 dim
    index = VectorIndex(dim=512)
    count = 0

    for filepath, profile_json in rows:
        try:
            profile = json.loads(profile_json)
            clap = profile.get("embeddings", {}).get("clap_general")
            if clap and len(clap) == 512:
                import numpy as np
                index.add(filepath, np.array(clap, dtype=np.float32))
                count += 1
        except Exception:
            pass

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save(str(INDEX_DIR))
    print(f"   FAISS index: {count} vectors, saved in {time.time() - t0:.1f}s")
    print(f"   Index location: {INDEX_DIR}")


if __name__ == "__main__":
    main()
