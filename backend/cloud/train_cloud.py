#!/usr/bin/env python3
"""
RESONATE Production Model — Cloud GPU Training Script.

Runs all 4 training phases on a CUDA GPU with full optimizations:
  - Mixed precision (AMP) for 2x speed
  - Multi-worker data loading
  - Gradient accumulation for large effective batch sizes
  - Automatic dataset downloading (FMA, NSynth on cloud's fast network)
  - Resume from any checkpoint

Usage:
    # Full training from scratch:
    python3 train_cloud.py --phases all

    # Resume from Phase A+B checkpoint, run Phase C+D:
    python3 train_cloud.py --phases cd --checkpoint /workspace/resonate/data/checkpoints/rpm_phase_ab.pt

    # Just Phase C (large-scale):
    python3 train_cloud.py --phases c --checkpoint /workspace/resonate/data/checkpoints/rpm_phaseB_best.pt

    # Download datasets only (no training):
    python3 train_cloud.py --download-only
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rpm_cloud_training.log", mode="a"),
    ],
)
logger = logging.getLogger("RPM-Cloud")


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

WORK_DIR = Path(os.environ.get("RESONATE_WORK_DIR", "/workspace/resonate"))
DATA_DIR = WORK_DIR / "data"
PRECOMPUTED_DIR = DATA_DIR / "precomputed"
DATASETS_DIR = DATA_DIR / "datasets"
CHARTS_DIR = DATA_DIR / "charts"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
OUTPUT_DIR = WORK_DIR / "output"
LOG_DIR = WORK_DIR / "logs"

# Ensure code is importable
BACKEND_DIR = WORK_DIR / "backend"
if BACKEND_DIR.exists():
    sys.path.insert(0, str(BACKEND_DIR))
else:
    # Maybe running from within backend/
    sys.path.insert(0, str(Path(__file__).parent.parent))


def ensure_dirs():
    for d in [DATA_DIR, PRECOMPUTED_DIR, DATASETS_DIR, CHARTS_DIR,
              CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Dataset Download (fast on cloud network)
# ══════════════════════════════════════════════════════════════════════

def download_datasets():
    """Download FMA and NSynth directly on the cloud instance."""
    import subprocess

    # FMA-small (8k tracks, 7.2GB)
    fma_dir = DATASETS_DIR / "fma" / "fma_small"
    if not fma_dir.exists():
        logger.info("Downloading FMA-small (7.2GB)...")
        fma_zip = DATASETS_DIR / "fma" / "fma_small.zip"
        (DATASETS_DIR / "fma").mkdir(parents=True, exist_ok=True)

        subprocess.run([
            "wget", "-q", "--show-progress",
            "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
            "-O", str(fma_zip),
        ], check=True)

        logger.info("Extracting FMA-small...")
        subprocess.run(["unzip", "-q", str(fma_zip), "-d", str(DATASETS_DIR / "fma")], check=True)
        fma_zip.unlink()
        logger.info(f"FMA-small ready: {fma_dir}")

    # FMA metadata
    fma_meta = DATASETS_DIR / "fma" / "fma_metadata"
    if not fma_meta.exists():
        logger.info("Downloading FMA metadata...")
        meta_zip = DATASETS_DIR / "fma" / "fma_metadata.zip"
        subprocess.run([
            "wget", "-q", "--show-progress",
            "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
            "-O", str(meta_zip),
        ], check=True)
        subprocess.run(["unzip", "-q", str(meta_zip), "-d", str(DATASETS_DIR / "fma")], check=True)
        meta_zip.unlink()

    # NSynth-train (289k notes, 25GB)
    nsynth_dir = DATASETS_DIR / "nsynth" / "nsynth-train"
    if not nsynth_dir.exists():
        logger.info("Downloading NSynth-train (25GB)...")
        nsynth_tar = DATASETS_DIR / "nsynth" / "nsynth-train.jsonwav.tar.gz"
        (DATASETS_DIR / "nsynth").mkdir(parents=True, exist_ok=True)

        subprocess.run([
            "wget", "-q", "--show-progress",
            "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
            "-O", str(nsynth_tar),
        ], check=True)

        logger.info("Extracting NSynth-train...")
        subprocess.run(["tar", "xzf", str(nsynth_tar), "-C", str(DATASETS_DIR / "nsynth")], check=True)
        nsynth_tar.unlink()
        logger.info(f"NSynth-train ready: {nsynth_dir}")

    # NSynth validation + test (smaller, useful)
    for split in ["valid", "test"]:
        nsynth_split = DATASETS_DIR / "nsynth" / f"nsynth-{split}"
        if not nsynth_split.exists():
            logger.info(f"Downloading NSynth-{split}...")
            tar_path = DATASETS_DIR / "nsynth" / f"nsynth-{split}.jsonwav.tar.gz"
            subprocess.run([
                "wget", "-q", "--show-progress",
                f"http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{split}.jsonwav.tar.gz",
                "-O", str(tar_path),
            ], check=True)
            subprocess.run(["tar", "xzf", str(tar_path), "-C", str(DATASETS_DIR / "nsynth")], check=True)
            tar_path.unlink()


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def run_training(phases: str, checkpoint_path: str = None):
    """Run specified training phases."""
    import torch
    from ml.training.rpm_model import RPMModel, RPMConfig, RPMLoss, count_parameters
    from ml.training.rpm_dataset import (
        DatasetConfig, RPMDataset, LocalSampleLoader, build_dataloader,
    )
    from ml.training.rpm_trainer import RPMTrainer, TrainingConfig

    # ── Device ──
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Falling back to CPU (will be very slow).")
        device = "cpu"
    else:
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")

    # ── Batch sizes based on VRAM ──
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        if vram >= 40:  # A100 40GB+
            bs_frozen = 64
            bs_unfrozen = 32
            num_workers = 8
            grad_accum = 2  # effective = 64
        elif vram >= 20:  # RTX 4090, A5000
            bs_frozen = 48
            bs_unfrozen = 16
            num_workers = 6
            grad_accum = 4  # effective = 64
        elif vram >= 10:  # RTX 3080
            bs_frozen = 32
            bs_unfrozen = 8
            num_workers = 4
            grad_accum = 8  # effective = 64
        else:
            bs_frozen = 16
            bs_unfrozen = 4
            num_workers = 2
            grad_accum = 16
    else:
        bs_frozen = 8
        bs_unfrozen = 4
        num_workers = 0
        grad_accum = 8

    logger.info(f"Batch sizes: frozen={bs_frozen}, unfrozen={bs_unfrozen}, "
                f"grad_accum={grad_accum}, workers={num_workers}")

    # ── Build model ──
    logger.info("Building RPM model...")
    cfg = RPMConfig()
    model = RPMModel(cfg).to(device)
    _ = model.backbone  # force load AST

    params = count_parameters(model)
    logger.info(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # ── Load checkpoint ──
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint loaded successfully")

    feature_extractor = model.feature_extractor

    # ── Load all available data sources ──
    all_samples = []

    # Local pre-computed samples
    if PRECOMPUTED_DIR.exists() and any(PRECOMPUTED_DIR.glob("*.json")):
        logger.info("Loading pre-computed local samples...")
        loader = LocalSampleLoader(
            # Local samples aren't on cloud — use precomputed only
            str(PRECOMPUTED_DIR),  # samples_dir (won't be used for audio)
            str(PRECOMPUTED_DIR),
        )
        local_samples = loader.load()
        all_samples.extend(local_samples)
        logger.info(f"  Local samples: {len(local_samples):,}")

    # FMA
    fma_dir = DATASETS_DIR / "fma" / "fma_small"
    if fma_dir.exists():
        try:
            from ml.training.rpm_dataset import FMALoader
            fma_loader = FMALoader(str(DATASETS_DIR / "fma"))
            fma_samples = fma_loader.load()
            all_samples.extend(fma_samples)
            logger.info(f"  FMA samples: {len(fma_samples):,}")
        except Exception as e:
            logger.warning(f"Failed to load FMA: {e}")

    # NSynth
    nsynth_dir = DATASETS_DIR / "nsynth" / "nsynth-train"
    if nsynth_dir.exists():
        try:
            from ml.training.rpm_dataset import NSynthLoader
            nsynth_loader = NSynthLoader(str(DATASETS_DIR / "nsynth"))
            nsynth_samples = nsynth_loader.load()
            all_samples.extend(nsynth_samples)
            logger.info(f"  NSynth samples: {len(nsynth_samples):,}")
        except Exception as e:
            logger.warning(f"Failed to load NSynth: {e}")

    logger.info(f"Total training samples: {len(all_samples):,}")

    if not all_samples:
        logger.error("No training samples found!")
        sys.exit(1)

    # ── Train/Val split ──
    random.seed(42)
    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.95)  # 95/5 on large datasets
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]
    logger.info(f"Train: {len(train_samples):,}, Val: {len(val_samples):,}")

    # ════════════════════════════════════════════════════════════════
    # Phase A: Knowledge Distillation
    # ════════════════════════════════════════════════════════════════

    if "a" in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE A: Knowledge Distillation (backbone frozen)")
        logger.info("=" * 70)

        data_cfg = DatasetConfig(
            local_samples_dir=str(PRECOMPUTED_DIR),
            local_profiles_dir=str(PRECOMPUTED_DIR),
            batch_size=bs_frozen,
            num_workers=num_workers,
            pin_memory=True,
        )
        train_ds = RPMDataset(train_samples, data_cfg, feature_extractor, augment=True)
        val_ds = RPMDataset(val_samples, data_cfg, feature_extractor, augment=False)
        train_dl = build_dataloader(train_ds, data_cfg, shuffle=True)
        val_dl = build_dataloader(val_ds, data_cfg, shuffle=False)

        train_cfg = TrainingConfig(
            output_dir=str(OUTPUT_DIR),
            checkpoint_dir=str(CHECKPOINT_DIR),
            log_dir=str(LOG_DIR),
            device=device,
            phase_a_epochs=5,
            phase_a_batch_size=bs_frozen,
        )
        trainer = RPMTrainer(train_cfg)

        t0 = time.time()
        trainer.train_phase_a(model, train_dl, val_dl)
        logger.info(f"Phase A done in {(time.time()-t0)/60:.1f} min")

        torch.save(model.state_dict(), CHECKPOINT_DIR / "rpm_phaseA_done.pt")
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    # ════════════════════════════════════════════════════════════════
    # Phase B: Multi-Task (backbone frozen)
    # ════════════════════════════════════════════════════════════════

    if "b" in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE B: Multi-Task Training (backbone frozen)")
        logger.info("=" * 70)

        data_cfg = DatasetConfig(
            local_samples_dir=str(PRECOMPUTED_DIR),
            local_profiles_dir=str(PRECOMPUTED_DIR),
            batch_size=bs_frozen,
            num_workers=num_workers,
            pin_memory=True,
        )
        train_ds = RPMDataset(train_samples, data_cfg, feature_extractor, augment=True)
        val_ds = RPMDataset(val_samples, data_cfg, feature_extractor, augment=False)
        train_dl = build_dataloader(train_ds, data_cfg, shuffle=True)
        val_dl = build_dataloader(val_ds, data_cfg, shuffle=False)

        train_cfg = TrainingConfig(
            output_dir=str(OUTPUT_DIR),
            checkpoint_dir=str(CHECKPOINT_DIR),
            log_dir=str(LOG_DIR),
            device=device,
            phase_b_epochs=8,
            phase_b_batch_size=bs_frozen,
        )
        trainer = RPMTrainer(train_cfg)

        t0 = time.time()
        trainer.train_phase_b(model, train_dl, val_dl)
        logger.info(f"Phase B done in {(time.time()-t0)/60:.1f} min")

        torch.save(model.state_dict(), CHECKPOINT_DIR / "rpm_phaseB_done.pt")
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    # ════════════════════════════════════════════════════════════════
    # Phase C: Large-Scale Training (backbone UNFROZEN, AMP)
    # ════════════════════════════════════════════════════════════════

    if "c" in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE C: Large-Scale Training (backbone unfrozen, AMP)")
        logger.info("=" * 70)

        # Rebuild dataloaders with unfrozen batch size
        data_cfg = DatasetConfig(
            local_samples_dir=str(PRECOMPUTED_DIR),
            local_profiles_dir=str(PRECOMPUTED_DIR),
            batch_size=bs_unfrozen,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Include all datasets for Phase C
        train_ds = RPMDataset(train_samples, data_cfg, feature_extractor, augment=True)
        val_ds = RPMDataset(val_samples, data_cfg, feature_extractor, augment=False)
        train_dl = build_dataloader(train_ds, data_cfg, shuffle=True)
        val_dl = build_dataloader(val_ds, data_cfg, shuffle=False)

        train_cfg = TrainingConfig(
            output_dir=str(OUTPUT_DIR),
            checkpoint_dir=str(CHECKPOINT_DIR),
            log_dir=str(LOG_DIR),
            device=device,
            phase_c_epochs=10,
            phase_c_batch_size=bs_unfrozen,
            phase_c_use_amp=True,
            phase_c_gradient_accumulation=grad_accum,
            phase_c_backbone_lr=1e-5,
            phase_c_heads_lr=5e-4,
        )
        trainer = RPMTrainer(train_cfg)

        t0 = time.time()
        trainer.train_phase_c(model, train_dl, val_dl)
        logger.info(f"Phase C done in {(time.time()-t0)/60:.1f} min")

        torch.save(model.state_dict(), CHECKPOINT_DIR / "rpm_phaseC_done.pt")
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    # ════════════════════════════════════════════════════════════════
    # Phase D: Chart Intelligence Fine-Tune
    # ════════════════════════════════════════════════════════════════

    if "d" in phases:
        chart_db = CHARTS_DIR / "chart_features.db"
        if not chart_db.exists():
            logger.warning("No chart database found — skipping Phase D")
            logger.warning(f"  Expected at: {chart_db}")
        else:
            logger.info("\n" + "=" * 70)
            logger.info("PHASE D: Chart Intelligence Fine-Tune")
            logger.info("=" * 70)

            train_cfg = TrainingConfig(
                output_dir=str(OUTPUT_DIR),
                checkpoint_dir=str(CHECKPOINT_DIR),
                log_dir=str(LOG_DIR),
                device=device,
                phase_d_epochs=5,
                phase_d_batch_size=bs_frozen,
            )
            trainer = RPMTrainer(train_cfg)

            # Phase D uses chart data — filter to samples with chart features
            chart_samples = [s for s in all_samples if s.chart_potential is not None]
            if chart_samples:
                random.shuffle(chart_samples)
                split = int(len(chart_samples) * 0.9)
                chart_train = chart_samples[:split]
                chart_val = chart_samples[split:]

                data_cfg = DatasetConfig(
                    local_samples_dir=str(PRECOMPUTED_DIR),
                    local_profiles_dir=str(PRECOMPUTED_DIR),
                    batch_size=bs_frozen,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                train_ds = RPMDataset(chart_train, data_cfg, feature_extractor, augment=False)
                val_ds = RPMDataset(chart_val, data_cfg, feature_extractor, augment=False)
                train_dl = build_dataloader(train_ds, data_cfg, shuffle=True)
                val_dl = build_dataloader(val_ds, data_cfg, shuffle=False)

                t0 = time.time()
                trainer.train_phase_d(model, train_dl, val_dl)
                logger.info(f"Phase D done in {(time.time()-t0)/60:.1f} min")
            else:
                logger.warning("No samples with chart potential labels — skipping Phase D")

    # ════════════════════════════════════════════════════════════════
    # Save Final Model
    # ════════════════════════════════════════════════════════════════

    final_path = OUTPUT_DIR / "rpm_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"\nFinal model saved: {final_path}")

    # Also export for production
    try:
        logger.info("Exporting ONNX model...")
        from ml.training.rpm_export import export_to_onnx
        export_to_onnx(str(final_path), str(OUTPUT_DIR), export_mode="embedding")
        logger.info(f"ONNX model saved: {OUTPUT_DIR}/rpm_embedding.onnx")
    except Exception as e:
        logger.warning(f"ONNX export failed (non-fatal): {e}")

    logger.info("\n" + "=" * 70)
    logger.info("ALL TRAINING COMPLETE!")
    logger.info(f"Final model:  {final_path}")
    logger.info(f"Checkpoints:  {CHECKPOINT_DIR}")
    logger.info(f"")
    logger.info(f"Download your model:")
    logger.info(f"  scp root@<GPU_IP>:{final_path} ~/Desktop/")
    logger.info(f"  scp root@<GPU_IP>:{OUTPUT_DIR}/rpm_embedding.onnx ~/Desktop/")
    logger.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RESONATE RPM Cloud Training")
    parser.add_argument(
        "--phases", default="all",
        help="Which phases to run: 'all', 'ab', 'cd', 'c', 'a', 'b', 'd' (default: all)"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download datasets, don't train"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip dataset download, use what's already there"
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Override work directory (default: /workspace/resonate)"
    )

    args = parser.parse_args()

    if args.work_dir:
        global WORK_DIR, DATA_DIR, PRECOMPUTED_DIR, DATASETS_DIR, CHARTS_DIR
        global CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR
        WORK_DIR = Path(args.work_dir)
        DATA_DIR = WORK_DIR / "data"
        PRECOMPUTED_DIR = DATA_DIR / "precomputed"
        DATASETS_DIR = DATA_DIR / "datasets"
        CHARTS_DIR = DATA_DIR / "charts"
        CHECKPOINT_DIR = WORK_DIR / "checkpoints"
        OUTPUT_DIR = WORK_DIR / "output"
        LOG_DIR = WORK_DIR / "logs"

    ensure_dirs()

    logger.info("=" * 70)
    logger.info("RESONATE Production Model — Cloud GPU Training")
    logger.info("=" * 70)
    logger.info(f"Work dir: {WORK_DIR}")

    # Download datasets
    if not args.skip_download:
        logger.info("\nDownloading datasets (cloud network = fast)...")
        try:
            download_datasets()
        except Exception as e:
            logger.warning(f"Dataset download issue: {e}")
            logger.warning("Continuing with available data...")

    if args.download_only:
        logger.info("Download complete. Exiting.")
        return

    # Parse phases
    phases = args.phases.lower()
    if phases == "all":
        phases = "abcd"
    elif phases == "ab":
        phases = "ab"
    elif phases == "cd":
        phases = "cd"

    logger.info(f"Running phases: {phases.upper()}")

    # Auto-detect checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None and ("c" in phases or "d" in phases):
        # Look for best available checkpoint
        for name in ["rpm_phaseB_done.pt", "rpm_phase_ab.pt", "rpm_phaseA_done.pt"]:
            candidate = CHECKPOINT_DIR / name
            if candidate.exists():
                checkpoint = str(candidate)
                logger.info(f"Auto-detected checkpoint: {checkpoint}")
                break

    run_training(phases, checkpoint)


if __name__ == "__main__":
    main()
