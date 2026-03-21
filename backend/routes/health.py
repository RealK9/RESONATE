"""
RESONATE — Health, settings, and status routes.
"""

import threading
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request

from config import HAS_CLAUDE, SAMPLE_INDEX_FILE
from indexer import indexing_status, sample_cache, auto_organize_samples, background_index
import config

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "ai": HAS_CLAUDE,
        "indexed": indexing_status["done"],
        "index_progress": indexing_status["processed"],
        "index_total": indexing_status["total"],
    }


@router.get("/index-status")
async def index_status():
    """Check sample indexing progress."""
    return {
        "done": indexing_status["done"],
        "processed": indexing_status["processed"],
        "total": indexing_status["total"],
        "cached": len(sample_cache),
    }


@router.get("/settings")
async def get_settings():
    """Get current settings."""
    return {
        "sample_dir": str(config.SAMPLE_DIR),
        "sample_count": len(sample_cache),
    }


@router.post("/settings/sample-dir")
async def set_sample_dir(request: Request):
    """Update sample library directory."""
    body = await request.json()
    new_dir = Path(body.get("path", ""))
    if not new_dir.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {new_dir}")
    config.SAMPLE_DIR = new_dir
    # Clear cache and re-index
    sample_cache.clear()
    indexing_status["done"] = False
    indexing_status["processed"] = 0
    indexing_status["total"] = 0
    if SAMPLE_INDEX_FILE.exists():
        SAMPLE_INDEX_FILE.unlink()
    auto_organize_samples()
    idx_thread = threading.Thread(target=background_index, daemon=True)
    idx_thread.start()
    return {"status": "ok", "sample_dir": str(config.SAMPLE_DIR)}
