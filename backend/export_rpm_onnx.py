#!/usr/bin/env python3
"""
RESONATE Production Model — Standalone ONNX Export & Validation Script.

Exports the trained RPM model to two ONNX artifacts:
  1. rpm_full.onnx     — complete model with all 7 task heads
  2. rpm_embedding.onnx — embedding-only model (backbone + projection neck, 768-d)

Validates each export by running ORT inference and comparing against PyTorch
outputs (max absolute error < 1e-4).  Prints model sizes and a wall-clock
speedup comparison (ORT vs PyTorch).

Usage:
    python export_rpm_onnx.py                          # defaults
    python export_rpm_onnx.py --checkpoint ~/.resonate/rpm_training/rpm_best.pt
    python export_rpm_onnx.py --output-dir ./my_models --warmup 5 --runs 20
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Resolve imports — allow running from backend/ or project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ml.training.rpm_model import RPMModel, RPMConfig  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_TRAINING_DIR = Path("~/.resonate/rpm_training").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/.resonate/rpm_models").expanduser()

# AST feature-extractor output shape (from ASTFeatureExtractor @ 16 kHz, 10 s)
DUMMY_BATCH = 1
DUMMY_TIME_STEPS = 1024
DUMMY_MEL_BINS = 128


# ======================================================================
# Export wrapper modules
# ======================================================================

class RPMEmbeddingWrapper(nn.Module):
    """
    Thin wrapper for ONNX export — embedding path only.

    Input : input_values  [B, 1024, 128]
    Output: embedding     [B, 768]
    """

    def __init__(self, model: RPMModel):
        super().__init__()
        self.backbone = model.backbone          # triggers lazy load
        self.neck = model.neck

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        embedding = self.neck(cls_token)
        return embedding


class RPMFullWrapper(nn.Module):
    """
    Thin wrapper for ONNX export — all production heads (distillation excluded).

    Returns a *tuple* because torch.onnx.export needs positional outputs to
    map to output_names.  The order matches what RPMExtractor._run_onnx expects.

    Outputs (in order):
        embedding           [B, 768]
        role_logits          [B, 11]
        genre_top_logits     [B, 12]
        genre_sub_logits     [B, 500]
        instrument_probs     [B, 200]   (sigmoid applied)
        key_logits           [B, 24]
        chord_logits         [B, 12]
        mode_logits          [B, 7]
        perceptual           [B, 9]
        era_probs            [B, 8]
        chart_potential      [B, 1]
    """

    def __init__(self, model: RPMModel):
        super().__init__()
        self.backbone = model.backbone          # triggers lazy load
        self.neck = model.neck
        self.role_head = model.role_head
        self.genre_head = model.genre_head
        self.instrument_head = model.instrument_head
        self.theory_head = model.theory_head
        self.perceptual_head = model.perceptual_head
        self.era_head = model.era_head
        self.chart_head = model.chart_head

    def forward(self, input_values: torch.Tensor):
        outputs = self.backbone(input_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        embedding = self.neck(cls_token)

        role_logits = self.role_head(embedding)
        genre_out = self.genre_head(embedding)      # inference path (no label)
        instrument_probs = torch.sigmoid(self.instrument_head(embedding))
        theory_out = self.theory_head(embedding)
        perceptual = self.perceptual_head(embedding)
        era_out = self.era_head(embedding)
        chart = self.chart_head(embedding)

        return (
            embedding,                          # 0
            role_logits,                        # 1
            genre_out["top_logits"],            # 2
            genre_out["sub_logits"],            # 3
            instrument_probs,                   # 4
            theory_out["key_logits"],           # 5
            theory_out["chord_logits"],         # 6
            theory_out["mode_logits"],          # 7
            perceptual,                         # 8
            era_out["probabilities"],           # 9
            chart,                              # 10
        )


# ======================================================================
# ONNX output names — must match RPMExtractor._run_onnx expectations
# ======================================================================

FULL_OUTPUT_NAMES = [
    "embedding",
    "role_logits",
    "genre_top_logits",
    "genre_sub_logits",
    "instrument_probs",
    "key_logits",
    "chord_logits",
    "mode_logits",
    "perceptual",
    "era_probs",
    "chart_potential",
]

EMBEDDING_OUTPUT_NAMES = ["embedding"]


# ======================================================================
# Core helpers
# ======================================================================

def _find_checkpoint(explicit: Optional[str]) -> Path:
    """
    Resolve the checkpoint path.  Priority:
      1. Explicit argument
      2. ~/.resonate/rpm_training/rpm_final.pt
      3. ~/.resonate/rpm_training/rpm_best.pt
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    for name in ("rpm_final.pt", "rpm_best.pt"):
        p = DEFAULT_TRAINING_DIR / name
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No checkpoint found.  Looked in {DEFAULT_TRAINING_DIR} for "
        "rpm_final.pt or rpm_best.pt.  Pass --checkpoint explicitly."
    )


def _load_model(checkpoint_path: Path, device: str = "cpu") -> RPMModel:
    """Load RPMModel from a .pt checkpoint (handles both raw state_dict and
    training-checkpoint dict with 'model_state_dict' key)."""
    logger.info("Loading RPMModel from %s ...", checkpoint_path)
    cfg = RPMConfig(freeze_backbone=False)
    model = RPMModel(cfg)

    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Force backbone materialisation (lazy property)
    _ = model.backbone
    return model


def _build_dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    """Build dynamic-axes dict: batch dim is dynamic for input and every output."""
    axes: dict[str, dict[int, str]] = {"input_values": {0: "batch_size"}}
    for name in output_names:
        axes[name] = {0: "batch_size"}
    return axes


def _file_size_mb(path: Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


# ======================================================================
# Export functions
# ======================================================================

def export_embedding_onnx(model: RPMModel, output_dir: Path) -> Path:
    """Export the embedding-only model to rpm_embedding.onnx."""
    wrapper = RPMEmbeddingWrapper(model)
    wrapper.eval()

    dummy = torch.randn(DUMMY_BATCH, DUMMY_TIME_STEPS, DUMMY_MEL_BINS)
    onnx_path = output_dir / "rpm_embedding.onnx"

    logger.info("Exporting rpm_embedding.onnx ...")
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["input_values"],
        output_names=EMBEDDING_OUTPUT_NAMES,
        dynamic_axes=_build_dynamic_axes(EMBEDDING_OUTPUT_NAMES),
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info("  Saved: %s  (%.1f MB)", onnx_path, _file_size_mb(onnx_path))
    return onnx_path


def export_full_onnx(model: RPMModel, output_dir: Path) -> Path:
    """Export the full model (all heads) to rpm_full.onnx."""
    wrapper = RPMFullWrapper(model)
    wrapper.eval()

    dummy = torch.randn(DUMMY_BATCH, DUMMY_TIME_STEPS, DUMMY_MEL_BINS)
    onnx_path = output_dir / "rpm_full.onnx"

    logger.info("Exporting rpm_full.onnx ...")
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["input_values"],
        output_names=FULL_OUTPUT_NAMES,
        dynamic_axes=_build_dynamic_axes(FULL_OUTPUT_NAMES),
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info("  Saved: %s  (%.1f MB)", onnx_path, _file_size_mb(onnx_path))
    return onnx_path


# ======================================================================
# ONNX structural validation  (onnx.checker)
# ======================================================================

def validate_onnx_structure(onnx_path: Path) -> bool:
    """Run onnx.checker.check_model if the onnx package is available."""
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("  [PASS] ONNX structural validation: %s", onnx_path.name)
        return True
    except ImportError:
        logger.warning("  [SKIP] onnx package not installed — skipping structural check")
        return True
    except Exception as exc:
        logger.error("  [FAIL] ONNX structural validation: %s — %s", onnx_path.name, exc)
        return False


# ======================================================================
# Numerical validation  (ORT vs PyTorch)
# ======================================================================

def validate_embedding_numerically(
    model: RPMModel,
    onnx_path: Path,
    atol: float = 1e-4,
) -> bool:
    """Compare ORT vs PyTorch embedding output on random input."""
    import onnxruntime as ort

    dummy_np = np.random.randn(DUMMY_BATCH, DUMMY_TIME_STEPS, DUMMY_MEL_BINS).astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_emb = model.get_embedding(torch.from_numpy(dummy_np)).cpu().numpy()

    # ORT
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_emb = session.run(["embedding"], {"input_values": dummy_np})[0]

    max_err = float(np.max(np.abs(pt_emb - ort_emb)))
    passed = max_err < atol
    status = "PASS" if passed else "FAIL"
    logger.info(
        "  [%s] Embedding numerical check — max |diff| = %.2e  (tol %.0e)",
        status, max_err, atol,
    )
    return passed


def validate_full_numerically(
    model: RPMModel,
    onnx_path: Path,
    atol: float = 1e-4,
) -> bool:
    """Compare ORT vs PyTorch for every output head on the same random input."""
    import onnxruntime as ort

    dummy_np = np.random.randn(DUMMY_BATCH, DUMMY_TIME_STEPS, DUMMY_MEL_BINS).astype(np.float32)
    dummy_pt = torch.from_numpy(dummy_np)

    # --- PyTorch reference (through the wrapper, so we get sigmoid on instruments
    # and era probabilities instead of cumulative logits) ---
    wrapper = RPMFullWrapper(model)
    wrapper.eval()
    with torch.no_grad():
        pt_outputs_tuple = wrapper(dummy_pt)
    pt_outputs = {name: pt_outputs_tuple[i].cpu().numpy() for i, name in enumerate(FULL_OUTPUT_NAMES)}

    # --- ORT ---
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_raw = session.run(FULL_OUTPUT_NAMES, {"input_values": dummy_np})
    ort_outputs = dict(zip(FULL_OUTPUT_NAMES, ort_raw))

    all_passed = True
    for name in FULL_OUTPUT_NAMES:
        max_err = float(np.max(np.abs(pt_outputs[name] - ort_outputs[name])))
        ok = max_err < atol
        status = "PASS" if ok else "FAIL"
        logger.info("  [%s] %-20s  max |diff| = %.2e", status, name, max_err)
        if not ok:
            all_passed = False

    return all_passed


# ======================================================================
# Speedup benchmark
# ======================================================================

def benchmark(
    model: RPMModel,
    onnx_full_path: Path,
    onnx_emb_path: Path,
    warmup: int = 3,
    runs: int = 10,
):
    """
    Wall-clock comparison: PyTorch (CPU) vs ORT (CPU) for both full and
    embedding-only models.
    """
    import onnxruntime as ort

    dummy_np = np.random.randn(DUMMY_BATCH, DUMMY_TIME_STEPS, DUMMY_MEL_BINS).astype(np.float32)
    dummy_pt = torch.from_numpy(dummy_np)

    # Wrappers
    full_wrapper = RPMFullWrapper(model)
    full_wrapper.eval()
    emb_wrapper = RPMEmbeddingWrapper(model)
    emb_wrapper.eval()

    full_sess = ort.InferenceSession(str(onnx_full_path), providers=["CPUExecutionProvider"])
    emb_sess = ort.InferenceSession(str(onnx_emb_path), providers=["CPUExecutionProvider"])

    def _time_pytorch(wrapper, n_warmup, n_runs):
        with torch.no_grad():
            for _ in range(n_warmup):
                wrapper(dummy_pt)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                wrapper(dummy_pt)
            return (time.perf_counter() - t0) / n_runs

    def _time_ort(session, output_names, n_warmup, n_runs):
        for _ in range(n_warmup):
            session.run(output_names, {"input_values": dummy_np})
        t0 = time.perf_counter()
        for _ in range(n_runs):
            session.run(output_names, {"input_values": dummy_np})
        return (time.perf_counter() - t0) / n_runs

    pt_full = _time_pytorch(full_wrapper, warmup, runs)
    ort_full = _time_ort(full_sess, FULL_OUTPUT_NAMES, warmup, runs)

    pt_emb = _time_pytorch(emb_wrapper, warmup, runs)
    ort_emb = _time_ort(emb_sess, EMBEDDING_OUTPUT_NAMES, warmup, runs)

    print("\n" + "=" * 64)
    print("  SPEEDUP COMPARISON  (CPU, batch=1)")
    print("=" * 64)
    print(f"  {'Model':<20} {'PyTorch':>12} {'ORT':>12} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Full (all heads)':<20} {pt_full*1000:>9.1f} ms {ort_full*1000:>9.1f} ms {pt_full/ort_full:>8.2f}x")
    print(f"  {'Embedding only':<20} {pt_emb*1000:>9.1f} ms {ort_emb*1000:>9.1f} ms {pt_emb/ort_emb:>8.2f}x")
    print("=" * 64 + "\n")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export RPM to ONNX (full + embedding) with validation & benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt checkpoint (default: auto-discover in ~/.resonate/rpm_training/)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to write ONNX files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip numerical validation (ORT vs PyTorch comparison)",
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true",
        help="Skip wall-clock speedup benchmark",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-4,
        help="Absolute tolerance for numerical validation (default: 1e-4)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Benchmark warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Benchmark timed iterations (default: 10)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Resolve checkpoint ──
    try:
        ckpt_path = _find_checkpoint(args.checkpoint)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Checkpoint: %s", ckpt_path)

    # ── Output dir ──
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # ── Load model ──
    model = _load_model(ckpt_path, device="cpu")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded — %.1f M parameters", total_params / 1e6)

    # ── Export ──
    emb_path = export_embedding_onnx(model, output_dir)
    full_path = export_full_onnx(model, output_dir)

    # ── Structural validation ──
    validate_onnx_structure(emb_path)
    validate_onnx_structure(full_path)

    # ── Numerical validation ──
    if not args.skip_validation:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            logger.warning("onnxruntime not installed — skipping numerical validation")
            args.skip_validation = True

    if not args.skip_validation:
        logger.info("Running numerical validation (ORT vs PyTorch) ...")
        emb_ok = validate_embedding_numerically(model, emb_path, atol=args.atol)
        full_ok = validate_full_numerically(model, full_path, atol=args.atol)
        if not (emb_ok and full_ok):
            logger.warning("Some numerical checks FAILED — inspect the diffs above")
    else:
        logger.info("Numerical validation skipped.")

    # ── Benchmark ──
    if not args.skip_benchmark:
        try:
            import onnxruntime  # noqa: F401
            logger.info("Running speedup benchmark (warmup=%d, runs=%d) ...", args.warmup, args.runs)
            benchmark(model, full_path, emb_path, warmup=args.warmup, runs=args.runs)
        except ImportError:
            logger.warning("onnxruntime not installed — skipping benchmark")
    else:
        logger.info("Benchmark skipped.")

    # ── Summary ──
    print("\n" + "=" * 64)
    print("  EXPORT SUMMARY")
    print("=" * 64)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output dir : {output_dir}")
    print(f"  rpm_embedding.onnx : {_file_size_mb(emb_path):>8.1f} MB")
    print(f"  rpm_full.onnx      : {_file_size_mb(full_path):>8.1f} MB")
    print(f"  Precision          : float32")
    print(f"  Opset              : 17")
    print(f"  Input shape        : [B, {DUMMY_TIME_STEPS}, {DUMMY_MEL_BINS}]  (dynamic batch)")
    print(f"  Input name         : input_values")
    print(f"  Embedding dim      : 768")
    print("=" * 64)
    print("\n  Next steps:")
    print(f"    cp {output_dir}/rpm_full.onnx ~/.resonate/rpm_models/")
    print(f"    cp {output_dir}/rpm_embedding.onnx ~/.resonate/rpm_models/")
    print("    # RPMExtractor auto-detects ONNX when rpm_full.onnx is present\n")


if __name__ == "__main__":
    main()
