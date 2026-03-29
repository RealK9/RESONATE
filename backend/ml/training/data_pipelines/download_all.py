#!/usr/bin/env python3
"""
RESONATE Production Model — Master Data Pipeline Orchestrator

One script to download and prepare ALL data for Phases E through I.
Run this to build the god-tier training dataset.

Usage:
    # Download everything (metadata first, audio second):
    python3 -m ml.training.data_pipelines.download_all --all

    # Just Phase E (contrastive):
    python3 -m ml.training.data_pipelines.download_all --phase e

    # Metadata only (no audio downloads — fast):
    python3 -m ml.training.data_pipelines.download_all --metadata-only

    # Status check:
    python3 -m ml.training.data_pipelines.download_all --status

SoniqLabs — No shortcuts, highest quality, all out.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("RESONATE-DataPipeline")


DEFAULT_DATA_ROOT = Path.home() / ".resonate" / "datasets"


def run_phase_e(data_root: Path, metadata_only: bool = False, max_per_source: int = 0):
    """Phase E: Contrastive text-audio pairs."""
    from ml.training.data_pipelines.phase_e_contrastive import PhaseEPipeline

    logger.info("=" * 70)
    logger.info("PHASE E: Contrastive Text-Audio Pairs")
    logger.info("  WavCaps + MusicCaps + LP-MusicCaps + AudioSet + Freesound")
    logger.info("=" * 70)

    pipeline = PhaseEPipeline(
        data_root=data_root / "phase_e",
        freesound_api_key=os.environ.get("FREESOUND_API_KEY", ""),
    )

    stats = pipeline.download_all_metadata()
    if not metadata_only:
        pipeline.download_audio(max_per_source=max_per_source)

    return stats


def run_phase_f(data_root: Path, include_slakh: bool = True):
    """Phase F: Multi-track stem understanding."""
    from ml.training.data_pipelines.phase_f_stems import PhaseFPipeline

    logger.info("=" * 70)
    logger.info("PHASE F: Multi-Track Stem Understanding")
    logger.info("  MUSDB18-HQ + MedleyDB + Slakh2100 + DSD100")
    logger.info("=" * 70)

    pipeline = PhaseFPipeline(data_root=data_root / "phase_f")
    return pipeline.download_all(include_slakh=include_slakh)


def run_phase_g(data_root: Path):
    """Phase G: Song structure annotations."""
    from ml.training.data_pipelines.phase_g_structure import PhaseGPipeline

    logger.info("=" * 70)
    logger.info("PHASE G: Song Structure & Temporal Awareness")
    logger.info("  SALAMI + Harmonix + Billboard Structure")
    logger.info("=" * 70)

    pipeline = PhaseGPipeline(data_root=data_root / "phase_g")
    return pipeline.download_all()


def run_phase_h(data_root: Path, seed_artists: list[str] = None):
    """Phase H: Music knowledge graph."""
    from ml.training.data_pipelines.phase_h_knowledge_graph import PhaseHPipeline

    logger.info("=" * 70)
    logger.info("PHASE H: Music Knowledge Graph")
    logger.info("  MusicBrainz + Wikidata + Discogs + AcousticBrainz")
    logger.info("=" * 70)

    pipeline = PhaseHPipeline(data_root=data_root / "phase_h")
    try:
        return pipeline.build_graph(seed_artists=seed_artists)
    finally:
        pipeline.close()


def run_phase_i(data_root: Path, fma_size: str = "large"):
    """Phase I: Self-supervised massive audio."""
    from ml.training.data_pipelines.phase_i_selfsupervised import PhaseIPipeline

    logger.info("=" * 70)
    logger.info("PHASE I: Self-Supervised Massive Audio Pre-Training")
    logger.info("  FMA-full + Internet Archive + LibriSpeech + Common Voice")
    logger.info("=" * 70)

    pipeline = PhaseIPipeline(data_root=data_root / "phase_i")
    return pipeline.download_all(fma_size=fma_size)


def check_status(data_root: Path):
    """Check what data is already available."""
    logger.info("=" * 70)
    logger.info("RESONATE Data Pipeline — Status Check")
    logger.info("=" * 70)

    total_size = 0

    for phase_dir in sorted(data_root.iterdir()):
        if not phase_dir.is_dir():
            continue

        # Count files and size
        file_count = 0
        dir_size = 0
        audio_count = 0

        for f in phase_dir.rglob("*"):
            if f.is_file():
                file_count += 1
                dir_size += f.stat().st_size
                if f.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg", ".mp4"):
                    audio_count += 1

        size_gb = dir_size / (1024 ** 3)
        total_size += dir_size
        logger.info(f"  {phase_dir.name}: {file_count:,} files ({audio_count:,} audio), {size_gb:.1f} GB")

    total_gb = total_size / (1024 ** 3)
    logger.info(f"\n  TOTAL: {total_gb:.1f} GB across all phases")
    logger.info("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RESONATE Master Data Pipeline")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--phase", default="all", help="Phase to run: e, f, g, h, i, or all")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--metadata-only", action="store_true", help="Only download metadata")
    parser.add_argument("--max-per-source", type=int, default=0, help="Max items per source (0=all)")
    parser.add_argument("--fma-size", default="large", choices=["small", "medium", "large", "full"])
    parser.add_argument("--no-slakh", action="store_true")
    parser.add_argument("--status", action="store_true", help="Check data status")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.status:
        check_status(data_root)
        return

    phases = args.phase.lower()
    if args.all:
        phases = "efghi"

    all_stats = {}

    if "e" in phases:
        all_stats["phase_e"] = run_phase_e(data_root, args.metadata_only, args.max_per_source)

    if "f" in phases:
        all_stats["phase_f"] = run_phase_f(data_root, include_slakh=not args.no_slakh)

    if "g" in phases:
        all_stats["phase_g"] = run_phase_g(data_root)

    if "h" in phases:
        all_stats["phase_h"] = run_phase_h(data_root)

    if "i" in phases:
        all_stats["phase_i"] = run_phase_i(data_root, fma_size=args.fma_size)

    # Save stats
    stats_path = data_root / "pipeline_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("ALL DATA PIPELINES COMPLETE")
    logger.info(f"Stats saved: {stats_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
