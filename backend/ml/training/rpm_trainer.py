"""
RESONATE Production Model — 4-Phase Training Pipeline.

Phase A: Knowledge Distillation (backbone frozen)
    - 5 epochs on local 33k samples
    - Distill CLAP (512d) + PANNs (2048d) + AST (768d) → RPM embedding space
    - ~5 min with pre-computed teacher embeddings

Phase B: Multi-Task on Local Data (backbone frozen)
    - 8 epochs on 33k samples with all task heads active
    - Role, genre, perceptual, quality, theory predictions
    - ~5 min

Phase C: Large-Scale Training (backbone unfrozen, all datasets)
    - FMA + NSynth + MTG-Jamendo + MUSDB18 + local 33k + chart previews
    - All heads active including instrument recognition and genre classification
    - Text-audio alignment loss from CLAP knowledge injection
    - 10 epochs, discriminative LR (backbone 1e-5, heads 5e-4)
    - Mixed precision (AMP)
    - ~8-12 hours CPU, ~2-3 hours GPU

Phase D: Chart Intelligence Fine-Tune (backbone frozen again)
    - Fine-tune era head and chart potential head on chart data
    - 5 epochs
    - ~30 min
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the full RPM training pipeline."""

    # Paths
    output_dir: str = "~/.resonate/rpm_training"
    checkpoint_dir: str = "~/.resonate/rpm_checkpoints"
    log_dir: str = "~/.resonate/rpm_logs"

    # Device
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Phase A: Distillation
    phase_a_epochs: int = 5
    phase_a_lr: float = 5e-4
    phase_a_batch_size: int = 32

    # Phase B: Multi-task local
    phase_b_epochs: int = 8
    phase_b_lr: float = 5e-4
    phase_b_batch_size: int = 32

    # Phase C: Large-scale
    phase_c_epochs: int = 10
    phase_c_backbone_lr: float = 1e-5
    phase_c_heads_lr: float = 5e-4
    phase_c_batch_size: int = 16
    phase_c_use_amp: bool = True        # mixed precision
    phase_c_gradient_accumulation: int = 4  # effective batch = 16 * 4 = 64

    # Phase D: Chart fine-tune
    phase_d_epochs: int = 5
    phase_d_lr: float = 2e-4
    phase_d_batch_size: int = 32

    # General
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    save_every_n_epochs: int = 1
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 50
    seed: int = 42

    # Early stopping
    patience: int = 3
    min_delta: float = 1e-4


def resolve_device(device_str: str = "auto") -> str:
    """Resolve device string to actual device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str


class TrainingMetrics:
    """Track and log training metrics."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: list[dict] = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def log(self, phase: str, epoch: int, step: int, metrics: dict):
        """Log a metrics snapshot."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "epoch": epoch,
            "step": step,
            **metrics,
        }
        self.metrics.append(entry)

        # Print summary
        loss_str = f"loss={metrics.get('total_loss', 0):.4f}"
        detail_parts = []
        for k, v in metrics.items():
            if k != "total_loss" and isinstance(v, (int, float)):
                detail_parts.append(f"{k}={v:.4f}")
        detail = ", ".join(detail_parts[:5])  # top 5 metrics
        logger.info(f"[Phase {phase}] Epoch {epoch} Step {step}: {loss_str} | {detail}")

    def check_improvement(self, val_loss: float) -> bool:
        """Check if validation loss improved. Returns True if should stop."""
        if val_loss < self.best_val_loss - 1e-4:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= 3  # patience

    def save(self):
        """Save metrics to disk."""
        metrics_file = self.log_dir / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)


class RPMTrainer:
    """
    Full 4-phase training pipeline for the RESONATE Production Model.
    """

    def __init__(self, cfg: TrainingConfig = None):
        self.cfg = cfg or TrainingConfig()
        self.device = resolve_device(self.cfg.device)

        # Ensure directories exist
        for d in [self.cfg.output_dir, self.cfg.checkpoint_dir, self.cfg.log_dir]:
            Path(d).expanduser().mkdir(parents=True, exist_ok=True)

        self.metrics = TrainingMetrics(self.cfg.log_dir)
        self.global_step = 0

        logger.info(f"RPMTrainer initialized — device: {self.device}")

    def _build_model(self):
        """Build the RPM model."""
        from ml.training.rpm_model import RPMModel, RPMConfig
        cfg = RPMConfig()
        model = RPMModel(cfg)
        model = model.to(self.device)
        return model

    def _build_loss(self, phase: str):
        """Build loss function configured for the given phase."""
        from ml.training.rpm_model import RPMLoss, RPMConfig
        loss_fn = RPMLoss(RPMConfig())
        loss_fn.set_phase_weights(phase)
        return loss_fn

    def _build_optimizer(self, model, lr: float, backbone_lr: float = None):
        """Build optimizer with optional discriminative LR."""
        if backbone_lr is not None:
            from ml.training.rpm_model import get_param_groups, RPMConfig
            cfg = RPMConfig()
            cfg.learning_rate = lr
            cfg.backbone_lr = backbone_lr
            param_groups = get_param_groups(model, cfg)
            # Filter out empty param groups
            param_groups = [g for g in param_groups if len(g["params"]) > 0]
        else:
            param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

        return AdamW(param_groups, weight_decay=self.cfg.weight_decay)

    def _build_scheduler(self, optimizer, num_training_steps: int):
        """Build learning rate scheduler with warmup + cosine decay."""
        warmup_steps = min(self.cfg.warmup_steps, num_training_steps // 5)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(num_training_steps - warmup_steps, 1),
            T_mult=1,
        )
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    def save_checkpoint(self, model, optimizer, phase: str, epoch: int, extra: dict = None):
        """Save a training checkpoint."""
        ckpt_dir = Path(self.cfg.checkpoint_dir).expanduser()
        ckpt_path = ckpt_dir / f"rpm_phase{phase}_epoch{epoch}.pt"

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "phase": phase,
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.metrics.best_val_loss,
        }
        if extra:
            state.update(extra)

        torch.save(state, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        # Also save "best" if this is the best val loss
        best_path = ckpt_dir / "rpm_best.pt"
        if extra and extra.get("is_best", False):
            torch.save(state, best_path)
            logger.info(f"Best model saved: {best_path}")

    def load_checkpoint(self, model, optimizer, checkpoint_path: str):
        """Load a checkpoint to resume training."""
        state = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.metrics.best_val_loss = state.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from {checkpoint_path} (step {self.global_step})")
        return state.get("epoch", 0), state.get("phase", "A")

    def _train_epoch(self, model, dataloader, loss_fn, optimizer, scheduler,
                     phase: str, epoch: int, scaler=None, accum_steps: int = 1):
        """Run one training epoch."""
        model.train()
        epoch_losses = {}
        num_batches = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_values = batch["input_values"].to(self.device)

            # Forward pass
            use_amp = scaler is not None and self.device == "cuda"

            if use_amp:
                with autocast():
                    outputs = model(input_values, batch.get("genre_top", None))
                    targets = {k: v.to(self.device) for k, v in batch.items() if k != "input_values"}
                    losses = loss_fn(outputs, targets)
                    loss = losses["total"] / accum_steps
            else:
                outputs = model(input_values,
                              batch.get("genre_top", {}).to(self.device) if "genre_top" in batch else None)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != "input_values"}
                losses = loss_fn(outputs, targets)
                loss = losses["total"] / accum_steps

            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1

            # Track losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            num_batches += 1

            # Log periodically
            if self.global_step % self.cfg.log_every_n_steps == 0 and self.global_step > 0:
                avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
                self.metrics.log(phase, epoch, self.global_step, {
                    "total_loss": avg_losses.get("total", 0),
                    **{f"loss_{k}": v for k, v in avg_losses.items() if k != "total"},
                })

        # Epoch summary
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        return avg_losses

    @torch.no_grad()
    def _eval_epoch(self, model, dataloader, loss_fn):
        """Run evaluation epoch."""
        model.eval()
        epoch_losses = {}
        num_batches = 0

        for batch in dataloader:
            input_values = batch["input_values"].to(self.device)
            outputs = model(input_values,
                          batch.get("genre_top", {}).to(self.device) if "genre_top" in batch else None)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != "input_values"}
            losses = loss_fn(outputs, targets)

            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            num_batches += 1

        return {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}

    # ──────────────────────────────────────────────────────────────
    # Phase A: Knowledge Distillation
    # ──────────────────────────────────────────────────────────────

    def train_phase_a(self, model, train_loader, val_loader=None):
        """
        Phase A: Distill CLAP + PANNs + AST into RPM embedding space.
        Backbone frozen — only projection neck + distillation head train.
        """
        logger.info("=" * 70)
        logger.info("PHASE A: Knowledge Distillation")
        logger.info("=" * 70)

        model.freeze_backbone()
        loss_fn = self._build_loss("A")
        optimizer = self._build_optimizer(model, lr=self.cfg.phase_a_lr)
        num_steps = len(train_loader) * self.cfg.phase_a_epochs
        scheduler = self._build_scheduler(optimizer, num_steps)

        for epoch in range(self.cfg.phase_a_epochs):
            t0 = time.time()
            train_losses = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scheduler, "A", epoch
            )
            elapsed = time.time() - t0

            logger.info(
                f"Phase A Epoch {epoch+1}/{self.cfg.phase_a_epochs} — "
                f"loss={train_losses.get('total', 0):.4f} — {elapsed:.1f}s"
            )

            if val_loader:
                val_losses = self._eval_epoch(model, val_loader, loss_fn)
                logger.info(f"  Val loss: {val_losses.get('total', 0):.4f}")

            self.save_checkpoint(model, optimizer, "A", epoch)

        self.metrics.save()
        logger.info("Phase A complete.\n")

    # ──────────────────────────────────────────────────────────────
    # Phase B: Multi-Task on Local Data
    # ──────────────────────────────────────────────────────────────

    def train_phase_b(self, model, train_loader, val_loader=None):
        """
        Phase B: Train all task heads on local 33k samples.
        Backbone still frozen.
        """
        logger.info("=" * 70)
        logger.info("PHASE B: Multi-Task Training (Local Data)")
        logger.info("=" * 70)

        model.freeze_backbone()
        loss_fn = self._build_loss("B")
        optimizer = self._build_optimizer(model, lr=self.cfg.phase_b_lr)
        num_steps = len(train_loader) * self.cfg.phase_b_epochs
        scheduler = self._build_scheduler(optimizer, num_steps)

        for epoch in range(self.cfg.phase_b_epochs):
            t0 = time.time()
            train_losses = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scheduler, "B", epoch
            )
            elapsed = time.time() - t0

            logger.info(
                f"Phase B Epoch {epoch+1}/{self.cfg.phase_b_epochs} — "
                f"loss={train_losses.get('total', 0):.4f} — {elapsed:.1f}s"
            )

            if val_loader:
                val_losses = self._eval_epoch(model, val_loader, loss_fn)
                logger.info(f"  Val loss: {val_losses.get('total', 0):.4f}")

                if self.metrics.check_improvement(val_losses.get("total", float("inf"))):
                    logger.info("Early stopping triggered.")
                    break

            self.save_checkpoint(model, optimizer, "B", epoch)

        self.metrics.save()
        logger.info("Phase B complete.\n")

    # ──────────────────────────────────────────────────────────────
    # Phase C: Large-Scale Training (backbone unfrozen)
    # ──────────────────────────────────────────────────────────────

    def train_phase_c(self, model, train_loader, val_loader=None):
        """
        Phase C: Full training with all datasets and backbone unfrozen.
        This is the long one — uses mixed precision and gradient accumulation.
        """
        logger.info("=" * 70)
        logger.info("PHASE C: Large-Scale Training (Backbone Unfrozen)")
        logger.info("=" * 70)

        model.unfreeze_backbone()
        loss_fn = self._build_loss("C")
        optimizer = self._build_optimizer(
            model,
            lr=self.cfg.phase_c_heads_lr,
            backbone_lr=self.cfg.phase_c_backbone_lr,
        )
        num_steps = (len(train_loader) * self.cfg.phase_c_epochs) // self.cfg.phase_c_gradient_accumulation
        scheduler = self._build_scheduler(optimizer, num_steps)

        # Mixed precision scaler (CUDA only)
        scaler = GradScaler() if self.cfg.phase_c_use_amp and self.device == "cuda" else None

        from ml.training.rpm_model import count_parameters
        params = count_parameters(model)
        logger.info(f"Trainable parameters: {params['trainable']:,} / {params['total']:,}")

        for epoch in range(self.cfg.phase_c_epochs):
            t0 = time.time()
            train_losses = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scheduler, "C", epoch,
                scaler=scaler, accum_steps=self.cfg.phase_c_gradient_accumulation,
            )
            elapsed = time.time() - t0

            logger.info(
                f"Phase C Epoch {epoch+1}/{self.cfg.phase_c_epochs} — "
                f"loss={train_losses.get('total', 0):.4f} — {elapsed:.1f}s"
            )

            if val_loader:
                val_losses = self._eval_epoch(model, val_loader, loss_fn)
                logger.info(f"  Val loss: {val_losses.get('total', 0):.4f}")

                is_best = val_losses.get("total", float("inf")) < self.metrics.best_val_loss
                self.save_checkpoint(model, optimizer, "C", epoch, {"is_best": is_best})

                if self.metrics.check_improvement(val_losses.get("total", float("inf"))):
                    logger.info("Early stopping triggered.")
                    break
            else:
                self.save_checkpoint(model, optimizer, "C", epoch)

        self.metrics.save()
        logger.info("Phase C complete.\n")

    # ──────────────────────────────────────────────────────────────
    # Phase D: Chart Intelligence
    # ──────────────────────────────────────────────────────────────

    def train_phase_d(self, model, train_loader, val_loader=None):
        """
        Phase D: Fine-tune era + chart potential heads on chart data.
        Backbone frozen again to prevent catastrophic forgetting.
        """
        logger.info("=" * 70)
        logger.info("PHASE D: Chart Intelligence Fine-Tune")
        logger.info("=" * 70)

        model.freeze_backbone()
        loss_fn = self._build_loss("D")
        optimizer = self._build_optimizer(model, lr=self.cfg.phase_d_lr)
        num_steps = len(train_loader) * self.cfg.phase_d_epochs
        scheduler = self._build_scheduler(optimizer, num_steps)

        for epoch in range(self.cfg.phase_d_epochs):
            t0 = time.time()
            train_losses = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scheduler, "D", epoch
            )
            elapsed = time.time() - t0

            logger.info(
                f"Phase D Epoch {epoch+1}/{self.cfg.phase_d_epochs} — "
                f"loss={train_losses.get('total', 0):.4f} — {elapsed:.1f}s"
            )

            if val_loader:
                val_losses = self._eval_epoch(model, val_loader, loss_fn)
                logger.info(f"  Val loss: {val_losses.get('total', 0):.4f}")

            self.save_checkpoint(model, optimizer, "D", epoch)

        self.metrics.save()
        logger.info("Phase D complete.\n")

    # ──────────────────────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────────────────────

    def train_full(self, data_config=None, resume_from: str = None):
        """
        Run the complete 4-phase training pipeline.

        Args:
            data_config: DatasetConfig with paths to all data sources
            resume_from: path to checkpoint to resume from
        """
        from ml.training.rpm_dataset import (
            DatasetConfig, build_dataset, build_dataloader
        )

        if data_config is None:
            data_config = DatasetConfig()

        logger.info("=" * 70)
        logger.info("RESONATE Production Model — Full Training Pipeline")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Output: {self.cfg.output_dir}")

        # Build model
        model = self._build_model()

        # Get feature extractor for dataset
        feature_extractor = model.feature_extractor

        # Resume if specified
        start_phase = "A"
        if resume_from and os.path.exists(resume_from):
            optimizer_dummy = self._build_optimizer(model, lr=1e-4)
            _, start_phase = self.load_checkpoint(model, optimizer_dummy, resume_from)
            del optimizer_dummy

        # ── Phase A: Distillation (local samples only) ──
        if start_phase <= "A":
            logger.info("\nBuilding Phase A dataset (local samples for distillation)...")
            data_config.batch_size = self.cfg.phase_a_batch_size
            train_ds = build_dataset(data_config, feature_extractor, split="train")
            val_ds = build_dataset(data_config, feature_extractor, split="val", augment=False)
            train_loader = build_dataloader(train_ds, data_config, shuffle=True)
            val_loader = build_dataloader(val_ds, data_config, shuffle=False) if len(val_ds) > 0 else None
            self.train_phase_a(model, train_loader, val_loader)

        # ── Phase B: Multi-task local ──
        if start_phase <= "B":
            logger.info("\nBuilding Phase B dataset (local samples, all heads)...")
            data_config.batch_size = self.cfg.phase_b_batch_size
            train_ds = build_dataset(data_config, feature_extractor, split="train")
            val_ds = build_dataset(data_config, feature_extractor, split="val", augment=False)
            train_loader = build_dataloader(train_ds, data_config, shuffle=True)
            val_loader = build_dataloader(val_ds, data_config, shuffle=False) if len(val_ds) > 0 else None
            self.train_phase_b(model, train_loader, val_loader)

        # ── Phase C: Large-scale (all datasets) ──
        if start_phase <= "C":
            logger.info("\nBuilding Phase C dataset (all sources)...")
            data_config.batch_size = self.cfg.phase_c_batch_size
            train_ds = build_dataset(data_config, feature_extractor, split="train")
            val_ds = build_dataset(data_config, feature_extractor, split="val", augment=False)
            train_loader = build_dataloader(train_ds, data_config, shuffle=True)
            val_loader = build_dataloader(val_ds, data_config, shuffle=False) if len(val_ds) > 0 else None
            self.train_phase_c(model, train_loader, val_loader)

        # ── Phase D: Chart intelligence ──
        if start_phase <= "D":
            logger.info("\nBuilding Phase D dataset (chart previews)...")
            data_config.batch_size = self.cfg.phase_d_batch_size
            train_ds = build_dataset(data_config, feature_extractor, split="train")
            val_ds = build_dataset(data_config, feature_extractor, split="val", augment=False)
            train_loader = build_dataloader(train_ds, data_config, shuffle=True)
            val_loader = build_dataloader(val_ds, data_config, shuffle=False) if len(val_ds) > 0 else None
            self.train_phase_d(model, train_loader, val_loader)

        # ── Save final model ──
        final_path = Path(self.cfg.output_dir).expanduser() / "rpm_final.pt"
        torch.save(model.state_dict(), final_path)
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Training complete! Final model saved to {final_path}")
        logger.info(f"{'=' * 70}")

        return model


# ──────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train the RESONATE Production Model")
    parser.add_argument("--local-samples", type=str, default="", help="Path to local samples")
    parser.add_argument("--local-profiles", type=str, default="", help="Path to sample profiles")
    parser.add_argument("--fma-dir", type=str, default="", help="Path to FMA dataset")
    parser.add_argument("--nsyth-dir", type=str, default="", help="Path to NSynth dataset")
    parser.add_argument("--jamendo-dir", type=str, default="", help="Path to MTG-Jamendo")
    parser.add_argument("--chart-dir", type=str, default="", help="Path to chart previews")
    parser.add_argument("--output-dir", type=str, default="~/.resonate/rpm_training")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--phase-only", type=str, default="", help="Run only this phase (A/B/C/D)")

    args = parser.parse_args()

    from ml.training.rpm_dataset import DatasetConfig

    data_cfg = DatasetConfig(
        local_samples_dir=args.local_samples,
        local_profiles_dir=args.local_profiles,
        fma_dir=args.fma_dir,
        nsyth_dir=args.nsyth_dir,
        jamendo_dir=args.jamendo_dir,
        chart_dir=args.chart_dir,
    )

    train_cfg = TrainingConfig(
        output_dir=args.output_dir,
        device=args.device,
    )

    trainer = RPMTrainer(train_cfg)
    trainer.train_full(data_cfg, resume_from=args.resume or None)
