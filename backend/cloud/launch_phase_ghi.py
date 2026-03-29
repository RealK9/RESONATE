#!/usr/bin/env python3
"""
RESONATE — Launch Phase G/H/I training on RunPod H100.

1. Builds data if not already built
2. Launches training using the AdvancedTrainer Python API

Usage:
    python3 /workspace/launch_phase_ghi.py
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

WORKSPACE = Path("/workspace")
RESONATE = WORKSPACE / "resonate"
DATASETS = WORKSPACE / "datasets"

# Add project to path
sys.path.insert(0, str(RESONATE))
sys.path.insert(0, str(RESONATE / "backend"))


def verify_data():
    """Check if Phase G/H/I data exists and is ready."""
    ready = True

    # Phase G
    g_dir = DATASETS / "phase_g"
    g_segments = 0
    for source in ["salami", "harmonix", "billboard_structure"]:
        ann_dir = g_dir / source / "annotations"
        audio_dir = g_dir / source / "audio"
        if ann_dir.exists() and audio_dir.exists():
            ann_count = sum(1 for _ in ann_dir.glob("*.json"))
            audio_count = sum(1 for _ in audio_dir.rglob("*.mp3"))
            g_segments += min(ann_count, audio_count)
            logger.info(f"  Phase G/{source}: {ann_count} ann, {audio_count} audio")

    if g_segments == 0:
        logger.warning("  Phase G: NO DATA — need to run build_phase_ghi_data.py first")
        ready = False
    else:
        logger.info(f"  Phase G: {g_segments} annotated segments ready")

    # Phase H
    h_triplets = DATASETS / "phase_h" / "triplets.jsonl"
    if h_triplets.exists():
        with open(h_triplets) as f:
            count = sum(1 for _ in f)
        logger.info(f"  Phase H: {count:,} triplets ready")
    else:
        logger.warning("  Phase H: NO DATA")
        ready = False

    # Phase I
    i_dir = DATASETS / "phase_i" / "audio"
    if i_dir.exists():
        i_count = 0
        for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg"]:
            i_count += sum(1 for _ in i_dir.rglob(ext))
        if i_count > 0:
            logger.info(f"  Phase I: {i_count:,} audio files ready")
        else:
            logger.warning("  Phase I: NO AUDIO FILES")
            ready = False
    else:
        logger.warning("  Phase I: audio directory missing")
        ready = False

    return ready


def build_data():
    """Run the data build script."""
    build_script = WORKSPACE / "build_phase_ghi_data.py"
    if not build_script.exists():
        build_script = RESONATE / "backend" / "cloud" / "build_phase_ghi_data.py"
    if not build_script.exists():
        logger.error("build_phase_ghi_data.py not found!")
        return False

    logger.info(f"Running data build: {build_script}")
    result = subprocess.run(
        [sys.executable, str(build_script)],
        cwd=str(WORKSPACE),
    )
    return result.returncode == 0


def launch_training():
    """Launch Phase G/H/I training using the AdvancedTrainer API."""
    import torch

    # Import the trainer
    from ml.training.advanced_training import (
        AdvancedTrainer,
        AdvancedTrainingConfig,
        AdvancedRPMModel,
    )

    # Find latest checkpoint
    checkpoint_dir = RESONATE / "checkpoints"
    checkpoint = None
    for name in ["rpm_phaseF_done.pt", "rpm_best.pt", "rpm_phaseE_done.pt"]:
        candidate = checkpoint_dir / name
        if candidate.exists():
            checkpoint = str(candidate)
            break

    if not checkpoint:
        logger.error("No checkpoint found! Need at least Phase F checkpoint.")
        return False

    logger.info(f"Using checkpoint: {checkpoint}")

    # Configure
    cfg = AdvancedTrainingConfig(
        phase_g_data_dir=str(DATASETS / "phase_g"),
        phase_h_data_dir=str(DATASETS / "phase_h"),
        phase_i_data_dir=str(DATASETS / "phase_i"),
        checkpoint_dir=str(checkpoint_dir),
        output_dir=str(RESONATE / "output"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Build trainer
    trainer = AdvancedTrainer(cfg)

    # Load model with checkpoint
    logger.info("Loading model...")
    model = AdvancedRPMModel()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Try loading — some keys may not match if model evolved
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info("  Checkpoint loaded (strict=False)")
    except Exception as e:
        logger.warning(f"  Partial checkpoint load: {e}")
    model = model.to(trainer.device)

    # Feature extractor
    try:
        from transformers import ASTFeatureExtractor
        feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
    except Exception:
        logger.warning("Could not load AST feature extractor, using None")
        feature_extractor = None

    # Run training (G, H, I only)
    logger.info("\nStarting Phase G/H/I training...")
    trainer.train_advanced(model, phases="GHI", feature_extractor=feature_extractor)
    logger.info("Training complete!")
    return True


def main():
    logger.info("=" * 70)
    logger.info("RESONATE Phase G/H/I Training Launcher")
    logger.info("=" * 70)

    # Step 1: Verify data
    logger.info("\nStep 1: Verifying training data...")
    if not verify_data():
        logger.info("\nStep 1b: Building missing data...")
        if not build_data():
            logger.error("Data build failed!")
            sys.exit(1)

        # Re-verify
        if not verify_data():
            logger.error("Data verification failed after build!")
            sys.exit(1)

    # Step 2: Launch training
    logger.info("\nStep 2: Launching training...")
    try:
        success = launch_training()
        if success:
            logger.info("\nTraining completed successfully!")
        else:
            logger.error("\nTraining failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"\nTraining crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
