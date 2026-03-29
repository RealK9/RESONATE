#!/usr/bin/env python3
"""
RESONATE Production Model — Phase C: Large-Scale Training.

Unfreezes the backbone and trains on ALL available datasets:
  - Local samples (33k) with teacher embeddings
  - NSynth (374k instrument notes)
  - FMA Small (16k tracks with genre labels)

This is the transformational training phase where the model learns to
actually HEAR music, not just parrot teacher embeddings.

Usage:
    python3 train_rpm_phase_c.py [--epochs N] [--batch-size N] [--resume PATH]
"""
import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            str(Path("~/.resonate/rpm_phase_c.log").expanduser()),
            mode="a",
        ),
    ],
)
logger = logging.getLogger("RPM-PhaseC")


def parse_args():
    parser = argparse.ArgumentParser(description="RPM Phase C Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Backbone learning rate")
    parser.add_argument("--heads-lr", type=float, default=5e-4, help="Head learning rate")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("RESONATE Production Model — PHASE C: Large-Scale Training")
    logger.info("  Backbone UNFROZEN — the model learns to hear music")
    logger.info("=" * 70)

    import torch
    from ml.training.rpm_model import RPMModel, RPMConfig, count_parameters
    from ml.training.rpm_dataset import (
        DatasetConfig, RPMDataset, TrainingSample, build_dataloader,
        LocalSampleLoader, NSynthLoader, FMALoader,
    )
    from ml.training.rpm_trainer import RPMTrainer, TrainingConfig

    # ── Device ──
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # ── Paths ──
    SAMPLES_DIR = Path(__file__).parent / "samples"
    PRECOMPUTED_DIR = Path("~/.resonate/precomputed").expanduser()
    CHECKPOINT_DIR = Path("~/.resonate/rpm_checkpoints").expanduser()
    TRAINING_DIR = Path("~/.resonate/rpm_training").expanduser()
    DATASETS_DIR = Path("~/.resonate/datasets").expanduser()

    for d in [CHECKPOINT_DIR, TRAINING_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Build model ──
    logger.info("Building RPM model...")
    cfg = RPMConfig()
    model = RPMModel(cfg)
    model = model.to(device)

    # Load checkpoint (Phase B done, or resume from Phase C epoch)
    resume_path = args.resume
    if resume_path is None:
        # Auto-find best starting point
        for candidate in [
            CHECKPOINT_DIR / "rpm_phaseC_done.pt",
            CHECKPOINT_DIR / "rpm_phaseB_done.pt",
            TRAINING_DIR / "rpm_final.pt",
            Path("/Users/krsn/Desktop/rpm_final.pt"),
        ]:
            if candidate.exists():
                resume_path = str(candidate)
                break

    if resume_path:
        logger.info(f"Loading checkpoint: {resume_path}")
        state_dict = torch.load(resume_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        logger.info("  ✓ Checkpoint loaded")
    else:
        logger.warning("No checkpoint found — training from scratch!")

    # Force-load backbone
    _ = model.backbone
    params = count_parameters(model)
    logger.info(f"Total params: {params['total']:,} (trainable: {params['trainable']:,})")

    feature_extractor = model.feature_extractor

    # ── Load ALL datasets ──
    all_samples: list[TrainingSample] = []

    # 1. Local samples (33k)
    logger.info("\n--- Loading local samples ---")
    local_loader = LocalSampleLoader(str(SAMPLES_DIR), str(PRECOMPUTED_DIR))
    local_samples = local_loader.load()
    logger.info(f"  Local: {len(local_samples):,} samples")
    all_samples.extend(local_samples)

    # 2. NSynth (374k instrument notes)
    nsynth_dir = DATASETS_DIR / "nsynth"
    if nsynth_dir.exists():
        logger.info("\n--- Loading NSynth ---")
        nsynth_loader = NSynthLoader(str(nsynth_dir))
        nsynth_samples = nsynth_loader.load()
        logger.info(f"  NSynth: {len(nsynth_samples):,} samples")
        all_samples.extend(nsynth_samples)
    else:
        logger.warning(f"NSynth not found at {nsynth_dir} — skipping")

    # 3. FMA (16k tracks)
    fma_dir = DATASETS_DIR / "fma"
    fma_small_dir = DATASETS_DIR / "fma_small"
    fma_audio_dir = None

    # Check for extracted FMA audio
    if (fma_dir / "fma_audio").exists():
        fma_audio_dir = fma_dir
    elif (fma_dir / "fma_small").exists():
        fma_audio_dir = fma_dir / "fma_small"
    elif fma_small_dir.exists():
        # Need to extract fma_small.zip
        zip_path = fma_small_dir / "fma_small.zip"
        if zip_path.exists():
            extracted_dir = fma_small_dir / "fma_audio"
            if not extracted_dir.exists():
                logger.info(f"Extracting FMA Small from {zip_path}...")
                import zipfile
                with zipfile.ZipFile(str(zip_path), 'r') as zf:
                    zf.extractall(str(fma_small_dir))
                logger.info("  ✓ FMA extracted")
            fma_audio_dir = fma_small_dir

    if fma_audio_dir and (fma_dir / "fma_metadata").exists():
        logger.info("\n--- Loading FMA ---")
        # FMALoader expects fma_dir with fma_metadata/ and fma_audio/ subdirs
        # Create symlink for fma_audio if needed
        combined_dir = DATASETS_DIR / "fma_combined"
        combined_dir.mkdir(exist_ok=True)
        meta_link = combined_dir / "fma_metadata"
        audio_link = combined_dir / "fma_audio"
        if not meta_link.exists():
            os.symlink(str(fma_dir / "fma_metadata"), str(meta_link))
        if not audio_link.exists():
            # Find the actual audio directory
            audio_candidates = list(fma_small_dir.glob("fma_small/???"))
            if audio_candidates:
                os.symlink(str(fma_small_dir / "fma_small"), str(audio_link))
            elif (fma_dir / "fma_audio").exists():
                os.symlink(str(fma_dir / "fma_audio"), str(audio_link))

        fma_loader = FMALoader(str(combined_dir))
        fma_samples = fma_loader.load()
        logger.info(f"  FMA: {len(fma_samples):,} samples")
        all_samples.extend(fma_samples)
    else:
        logger.warning("FMA audio not found or metadata missing — skipping")

    logger.info(f"\n{'='*70}")
    logger.info(f"TOTAL TRAINING SAMPLES: {len(all_samples):,}")
    logger.info(f"{'='*70}")

    if len(all_samples) == 0:
        logger.error("No samples loaded! Cannot train.")
        sys.exit(1)

    # ── Split train/val ──
    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - args.val_split))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    logger.info(f"Train: {len(train_samples):,} | Val: {len(val_samples):,}")

    # ── Build datasets ──
    data_cfg = DatasetConfig(
        local_samples_dir=str(SAMPLES_DIR),
        local_profiles_dir=str(PRECOMPUTED_DIR),
        batch_size=args.batch_size,
        num_workers=0,     # safer for MPS
        pin_memory=False,  # not supported on MPS
    )

    train_ds = RPMDataset(train_samples, data_cfg, feature_extractor, augment=True)
    val_ds = RPMDataset(val_samples, data_cfg, feature_extractor, augment=False)

    train_loader = build_dataloader(train_ds, data_cfg, shuffle=True)
    val_loader = build_dataloader(val_ds, data_cfg, shuffle=False)

    logger.info(f"Train batches: {len(train_loader):,}")
    logger.info(f"Val batches: {len(val_loader):,}")
    effective_batch = args.batch_size * args.grad_accum
    logger.info(f"Effective batch size: {effective_batch}")
    est_steps = len(train_loader) * args.epochs // args.grad_accum
    logger.info(f"Estimated total steps: {est_steps:,}")

    # ── Configure trainer ──
    train_cfg = TrainingConfig(
        output_dir=str(TRAINING_DIR),
        checkpoint_dir=str(CHECKPOINT_DIR),
        log_dir=str(TRAINING_DIR / "logs"),
        device=device,
        phase_c_epochs=args.epochs,
        phase_c_backbone_lr=args.backbone_lr,
        phase_c_heads_lr=args.heads_lr,
        phase_c_batch_size=args.batch_size,
        phase_c_gradient_accumulation=args.grad_accum,
        phase_c_use_amp=(device == "cuda"),  # AMP only on CUDA
    )

    trainer = RPMTrainer(train_cfg)

    # ── Phase C: Train! ──
    logger.info(f"\n{'='*70}")
    logger.info("PHASE C: BACKBONE UNFROZEN — LEARNING TO HEAR MUSIC")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Backbone LR: {args.backbone_lr}")
    logger.info(f"  Heads LR: {args.heads_lr}")
    logger.info(f"  Batch: {args.batch_size} x {args.grad_accum} accum = {effective_batch}")
    logger.info(f"{'='*70}\n")

    t0 = time.time()
    trainer.train_phase_c(model, train_loader, val_loader)
    elapsed = time.time() - t0

    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE C COMPLETE!")
    logger.info(f"  Duration: {elapsed/3600:.1f} hours ({elapsed/60:.0f} minutes)")
    logger.info(f"{'='*70}")

    # Save final model
    final_path = TRAINING_DIR / "rpm_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Model saved: {final_path}")

    # Also copy to Desktop for convenience
    import shutil
    desktop_path = Path("/Users/krsn/Desktop/rpm_final.pt")
    shutil.copy2(str(final_path), str(desktop_path))
    logger.info(f"Also saved to: {desktop_path}")


if __name__ == "__main__":
    main()
