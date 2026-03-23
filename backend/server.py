"""
RESONATE v8.0 — AI-Powered Sample Matching Engine
Modular architecture: config, analysis, AI, audio, routes.
"""

import sys
import threading
from pathlib import Path

# Ensure backend directory is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import SAMPLE_DIR, AUDIO_EXT, HAS_CLAUDE
from db.database import init_db
from indexer import (
    auto_organize_samples,
    load_disk_cache,
    background_index,
    background_index_v2,
    sample_cache,
)
from app import create_app

# Create the FastAPI app
app = create_app()

if __name__ == "__main__":
    import uvicorn

    n_samples = sum(
        1 for f in SAMPLE_DIR.rglob("*")
        if f.suffix.lower() in AUDIO_EXT
    )
    print(f"\n{'=' * 50}")
    print(f"  RESONATE v8.0 — AI Sample Matching Engine")
    ai_status = "Claude AI" if HAS_CLAUDE else "Essentia only"
    print(f"  Samples: {n_samples} | AI: {ai_status}")
    print(f"  Transposition: ✓ (librosa)")
    print(f"{'=' * 50}\n")

    # Initialize database
    init_db()

    # Auto-organize samples into category folders (runs once)
    auto_organize_samples()

    # Load persistent cache from disk (instant)
    load_disk_cache()

    # Start background indexing for any new/changed samples
    idx_thread = threading.Thread(target=background_index, daemon=True)
    idx_thread.start()

    # Start v2 indexing in background
    t2 = threading.Thread(target=background_index_v2, daemon=True)
    t2.start()
    print("  ⟳ V2 sample analysis running in background...")

    if sample_cache:
        print(f"  → Server ready immediately with {len(sample_cache)} cached samples")
        print(f"  → New samples indexing in background...\n")
    else:
        print(f"  → First run: indexing {n_samples} samples in background...")
        print(f"  → Samples will appear as they're indexed\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
