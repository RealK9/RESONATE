#!/usr/bin/env python3
"""
RESONATE Production Model — Master Training Script.

Step 1: Pre-compute teacher embeddings (CLAP + PANNs + AST) for local 33k samples
Step 2: Phase A — Knowledge Distillation (backbone frozen, 5 epochs)
Step 3: Phase B — Multi-Task Training (backbone frozen, 8 epochs)

Phases C & D require external datasets (FMA, NSynth, chart previews).

Usage:
    python3 train_rpm.py
"""
import logging
import os
import sys
import time
from pathlib import Path

# Ensure we can import from the backend directory
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            str(Path("~/.resonate/rpm_training.log").expanduser()),
            mode="a",
        ),
    ],
)
logger = logging.getLogger("RPM")


def main():
    logger.info("=" * 70)
    logger.info("RESONATE Production Model — Training Pipeline")
    logger.info("=" * 70)

    # ── Paths ──
    SAMPLES_DIR = Path(__file__).parent / "samples"
    PRECOMPUTED_DIR = Path("~/.resonate/precomputed").expanduser()
    TRAINING_DIR = Path("~/.resonate/rpm_training").expanduser()
    CHECKPOINT_DIR = Path("~/.resonate/rpm_checkpoints").expanduser()

    for d in [PRECOMPUTED_DIR, TRAINING_DIR, CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Count samples
    wav_count = len(list(SAMPLES_DIR.rglob("*.wav")))
    logger.info(f"Samples directory: {SAMPLES_DIR}")
    logger.info(f"Sample count: {wav_count:,}")

    if wav_count == 0:
        logger.error(f"No WAV files found in {SAMPLES_DIR}!")
        sys.exit(1)

    # ── Step 1: Pre-compute teacher embeddings ──
    precomputed_count = len(list(PRECOMPUTED_DIR.glob("*.json")))
    if precomputed_count < wav_count * 0.9:  # allow 10% failure tolerance
        logger.info(f"\n{'='*70}")
        logger.info(f"STEP 1: Pre-computing teacher embeddings")
        logger.info(f"  Have {precomputed_count:,}, need ~{wav_count:,}")
        logger.info(f"{'='*70}")

        from ml.training.precompute_embeddings import precompute
        precompute(
            samples_dir=str(SAMPLES_DIR),
            output_dir=str(PRECOMPUTED_DIR),
            device="auto",
            batch_size=100,
        )
        precomputed_count = len(list(PRECOMPUTED_DIR.glob("*.json")))
    else:
        logger.info(f"Teacher embeddings already computed: {precomputed_count:,}")

    # ── Step 2: Build model and run Phase A + B ──
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 2: Training RPM (Phases A + B)")
    logger.info(f"{'='*70}")

    import torch
    from ml.training.rpm_model import RPMModel, RPMConfig, RPMLoss, count_parameters
    from ml.training.rpm_dataset import (
        DatasetConfig, RPMDataset, TrainingSample, build_dataloader, rpm_collate_fn,
        LocalSampleLoader,
    )
    from ml.training.rpm_trainer import RPMTrainer, TrainingConfig

    # Resolve device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Training device: {device}")

    # Build model
    logger.info("Building RPM model...")
    cfg = RPMConfig()
    model = RPMModel(cfg)
    model = model.to(device)

    # Force-load backbone
    logger.info("Loading AST backbone...")
    _ = model.backbone
    params = count_parameters(model)
    logger.info(f"Total params: {params['total']:,} (trainable: {params['trainable']:,})")

    # Get feature extractor
    feature_extractor = model.feature_extractor

    # Load local samples
    logger.info("Loading local sample profiles...")
    loader = LocalSampleLoader(str(SAMPLES_DIR), str(PRECOMPUTED_DIR))
    samples = loader.load()
    logger.info(f"Loaded {len(samples):,} samples with embeddings")

    if len(samples) == 0:
        logger.error("No samples loaded! Check pre-computation step.")
        sys.exit(1)

    # Split into train/val
    import random
    random.seed(42)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    logger.info(f"Train: {len(train_samples):,}, Val: {len(val_samples):,}")

    # Build datasets
    data_cfg = DatasetConfig(
        local_samples_dir=str(SAMPLES_DIR),
        local_profiles_dir=str(PRECOMPUTED_DIR),
        batch_size=16,  # conservative for MPS/CPU
        num_workers=0,  # safer for MPS
    )

    train_ds = RPMDataset(train_samples, data_cfg, feature_extractor, augment=True)
    val_ds = RPMDataset(val_samples, data_cfg, feature_extractor, augment=False)

    train_loader = build_dataloader(train_ds, data_cfg, shuffle=True)
    val_loader = build_dataloader(val_ds, data_cfg, shuffle=False)

    logger.info(f"Train batches: {len(train_loader):,}")
    logger.info(f"Val batches: {len(val_loader):,}")

    # Configure trainer
    train_cfg = TrainingConfig(
        output_dir=str(TRAINING_DIR),
        checkpoint_dir=str(CHECKPOINT_DIR),
        log_dir=str(TRAINING_DIR / "logs"),
        device=device,
        phase_a_epochs=5,
        phase_a_batch_size=16,
        phase_b_epochs=8,
        phase_b_batch_size=16,
    )

    trainer = RPMTrainer(train_cfg)

    # ── Phase A: Distillation ──
    logger.info(f"\n{'='*70}")
    logger.info("PHASE A: Knowledge Distillation")
    logger.info("Training RPM to match CLAP + PANNs + AST embeddings")
    logger.info(f"{'='*70}")
    t0 = time.time()
    trainer.train_phase_a(model, train_loader, val_loader)
    logger.info(f"Phase A completed in {(time.time() - t0)/60:.1f} minutes")

    # ── Phase B: Multi-Task ──
    logger.info(f"\n{'='*70}")
    logger.info("PHASE B: Multi-Task Training")
    logger.info("Training all 8 heads on local samples")
    logger.info(f"{'='*70}")
    t0 = time.time()
    trainer.train_phase_b(model, train_loader, val_loader)
    logger.info(f"Phase B completed in {(time.time() - t0)/60:.1f} minutes")

    # Save final Phase A+B model
    phase_ab_path = TRAINING_DIR / "rpm_phase_ab.pt"
    torch.save(model.state_dict(), phase_ab_path)
    logger.info(f"\nPhase A+B model saved: {phase_ab_path}")

    logger.info(f"\n{'='*70}")
    logger.info("PHASES A + B COMPLETE!")
    logger.info(f"Model saved to: {phase_ab_path}")
    logger.info(f"")
    logger.info(f"Next steps:")
    logger.info(f"  1. Download FMA dataset:     python3 -m ml.training.datasets.dataset_manager --download")
    logger.info(f"  2. Download NSynth dataset:   python3 -m ml.training.datasets.dataset_manager --download")
    logger.info(f"  3. Scrape Billboard charts:   python3 -m ml.training.charts.billboard_scraper")
    logger.info(f"  4. Enrich with Spotify:       python3 -m ml.training.charts.spotify_enrichment")
    logger.info(f"  5. Run Phase C (large-scale):  python3 -m ml.training.rpm_trainer --resume {CHECKPOINT_DIR}/rpm_phaseB_epoch7.pt")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
