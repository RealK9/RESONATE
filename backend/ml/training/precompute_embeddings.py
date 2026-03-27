"""
Pre-compute teacher embeddings (CLAP + PANNs + AST) for local samples.

This is a one-time operation before RPM training begins.
Saves embeddings + labels to disk so the training loop reads from cache
instead of running 3 models per sample every epoch.

Usage:
    python3 -m ml.training.precompute_embeddings \
        --samples-dir /path/to/samples \
        --output-dir ~/.resonate/precomputed
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Role mapping from directory names
DIR_TO_ROLE = {
    "kick": "kick",
    "snare": "snare",
    "clap": "clap",
    "hi-hats": "hat",
    "hats": "hat",
    "hi_hats": "hat",
    "hihat": "hat",
    "percussion": "perc",
    "perc": "perc",
    "bass": "bass",
    "leads": "lead",
    "lead": "lead",
    "melody": "lead",
    "pads": "pad",
    "pad": "pad",
    "fx": "fx",
    "sfx": "fx",
    "texture": "texture",
    "other": "texture",
    "vocals": "vocal",
    "vocal": "vocal",
    "guitar": "lead",
    "piano": "lead",
    "keys": "pad",
    "synth": "lead",
    "strings": "pad",
    "brass-wind": "lead",
    "brass": "lead",
}

ROLE_TO_ID = {
    "kick": 0, "snare": 1, "clap": 2, "hat": 3, "perc": 4,
    "bass": 5, "lead": 6, "pad": 7, "fx": 8, "texture": 9, "vocal": 10,
}


def infer_role_from_path(filepath: str) -> tuple[str, int]:
    """Infer sample role from parent directory name."""
    parts = Path(filepath).parts
    for part in reversed(parts):
        part_lower = part.lower().replace(" ", "").replace("-", "_")
        for dir_name, role in DIR_TO_ROLE.items():
            if dir_name.replace("-", "_") in part_lower:
                return role, ROLE_TO_ID[role]
    return "texture", ROLE_TO_ID["texture"]


def precompute(samples_dir: str, output_dir: str, batch_size: int = 50,
               skip_existing: bool = True, device: str = "auto"):
    """
    Pre-compute teacher embeddings for all WAV files in samples_dir.

    Saves one JSON profile per sample with:
    - filepath, filename, role, role_id
    - clap_embedding (512d)
    - panns_embedding (2048d)
    - ast_embedding (768d)
    - panns_tags (top-20 AudioSet tags)
    """
    samples_dir = Path(samples_dir)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device
    import torch
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Collect all WAV files
    wav_files = sorted(samples_dir.rglob("*.wav"))
    logger.info(f"Found {len(wav_files):,} WAV files in {samples_dir}")

    if not wav_files:
        logger.error("No WAV files found!")
        return

    # Check how many already done
    if skip_existing:
        existing = set()
        for f in output_dir.glob("*.json"):
            existing.add(f.stem)
        remaining = [f for f in wav_files if f.stem not in existing]
        logger.info(f"Already processed: {len(wav_files) - len(remaining):,}, remaining: {len(remaining):,}")
        wav_files = remaining

    if not wav_files:
        logger.info("All samples already pre-computed!")
        return

    # Load teacher models
    logger.info(f"Loading teacher models on {device}...")
    t0 = time.time()

    from ml.embeddings.embedding_manager import EmbeddingManager
    emb_manager = EmbeddingManager(device=device)

    # Force-load all models upfront
    logger.info("  Loading CLAP...")
    _ = emb_manager.clap
    logger.info("  Loading PANNs...")
    _ = emb_manager.panns
    logger.info("  Loading AST...")
    _ = emb_manager.ast
    logger.info(f"  All teachers loaded in {time.time() - t0:.1f}s")

    # Process
    success = 0
    failed = 0
    t_start = time.time()

    for i, wav_path in enumerate(wav_files):
        try:
            filepath = str(wav_path)
            role_name, role_id = infer_role_from_path(filepath)

            # Extract all embeddings
            embeddings = emb_manager.extract_all(filepath)

            # Build profile
            profile = {
                "filepath": filepath,
                "filename": wav_path.name,
                "role": role_name,
                "role_id": role_id,
                "embeddings": {
                    "clap_general": embeddings.clap_general,
                    "panns_music": embeddings.panns_music,
                    "ast_spectrogram": embeddings.ast_spectrogram,
                    "panns_tags": embeddings.panns_tags,
                },
            }

            # Save
            out_path = output_dir / f"{wav_path.stem}.json"
            with open(out_path, "w") as f:
                json.dump(profile, f)

            success += 1

        except Exception as e:
            failed += 1
            if failed <= 10:
                logger.warning(f"Failed on {wav_path.name}: {e}")

        # Progress
        if (i + 1) % batch_size == 0 or (i + 1) == len(wav_files):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(wav_files) - i - 1) / rate if rate > 0 else 0
            pct = (i + 1) / len(wav_files) * 100
            logger.info(
                f"  [{pct:5.1f}%] {i+1:,}/{len(wav_files):,} — "
                f"{rate:.1f} samples/s — ETA {eta/60:.0f}m — "
                f"✓{success:,} ✗{failed:,}"
            )

    elapsed = time.time() - t_start
    logger.info(f"\nPre-computation complete!")
    logger.info(f"  Success: {success:,}")
    logger.info(f"  Failed:  {failed:,}")
    logger.info(f"  Time:    {elapsed/60:.1f} minutes")
    logger.info(f"  Output:  {output_dir}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Pre-compute teacher embeddings for RPM training")
    parser.add_argument("--samples-dir", required=True, help="Directory containing WAV samples")
    parser.add_argument("--output-dir", default="~/.resonate/precomputed", help="Output directory for profiles")
    parser.add_argument("--device", default="auto", help="Device: auto/cuda/mps/cpu")
    parser.add_argument("--batch-size", type=int, default=50, help="Progress log interval")
    parser.add_argument("--no-skip", action="store_true", help="Re-process existing samples")

    args = parser.parse_args()
    precompute(
        args.samples_dir,
        args.output_dir,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip,
        device=args.device,
    )
