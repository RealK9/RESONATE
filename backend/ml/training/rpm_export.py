"""
RESONATE Production Model — ONNX Export.

Exports the trained RPM model to ONNX format for fast production inference.
FP32 precision — quality over speed. No quantization.

Usage:
    python -m ml.training.rpm_export --checkpoint ~/.resonate/rpm_training/rpm_final.pt
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RPMEmbeddingExporter(nn.Module):
    """
    Wrapper that exports only the embedding extraction path.
    Input: mel spectrogram features [B, T, F]
    Output: 768-d RPM embedding [B, 768]
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        embedding = self.neck(cls_token)
        return embedding


class RPMFullExporter(nn.Module):
    """
    Wrapper that exports the full model with all heads.
    Used when you need classification outputs in production.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
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

        role = self.role_head(embedding)
        genre = self.genre_head(embedding)
        instruments = torch.sigmoid(self.instrument_head(embedding))
        theory = self.theory_head(embedding)
        perceptual = self.perceptual_head(embedding)
        era = self.era_head(embedding)
        chart = self.chart_head(embedding)

        return (
            embedding,                      # 0: [B, 768]
            role,                           # 1: [B, 11]
            genre["top_logits"],            # 2: [B, 12]
            genre["sub_logits"],            # 3: [B, 500]
            instruments,                    # 4: [B, 200]
            theory["key_logits"],           # 5: [B, 24]
            theory["chord_logits"],         # 6: [B, 12]
            theory["mode_logits"],          # 7: [B, 7]
            perceptual,                     # 8: [B, 9]
            era["probabilities"],           # 9: [B, 8]
            chart,                          # 10: [B, 1]
        )


def export_to_onnx(checkpoint_path: str, output_dir: str = "~/.resonate/rpm_models",
                   export_mode: str = "both"):
    """
    Export RPM model to ONNX format.

    Args:
        checkpoint_path: path to trained .pt checkpoint
        output_dir: directory to save ONNX models
        export_mode: "embedding" (just embedding), "full" (all heads), or "both"
    """
    from ml.training.rpm_model import RPMModel, RPMConfig

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    cfg = RPMConfig(freeze_backbone=False)
    model = RPMModel(cfg)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Dummy input — AST expects [B, 1024, 128] mel spectrogram
    # The exact shape depends on ASTFeatureExtractor output
    dummy_input = torch.randn(1, 1024, 128)

    if export_mode in ("embedding", "both"):
        logger.info("Exporting embedding model to ONNX...")
        embedding_model = RPMEmbeddingExporter(model)
        embedding_model.eval()

        onnx_path = output_dir / "rpm_embedding.onnx"
        torch.onnx.export(
            embedding_model,
            dummy_input,
            str(onnx_path),
            input_names=["input_values"],
            output_names=["embedding"],
            dynamic_axes={
                "input_values": {0: "batch_size"},
                "embedding": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info(f"Embedding model exported: {onnx_path}")
        logger.info(f"  Size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

    if export_mode in ("full", "both"):
        logger.info("Exporting full model to ONNX...")
        full_model = RPMFullExporter(model)
        full_model.eval()

        onnx_path = output_dir / "rpm_full.onnx"
        torch.onnx.export(
            full_model,
            dummy_input,
            str(onnx_path),
            input_names=["input_values"],
            output_names=[
                "embedding", "role_logits", "genre_top_logits", "genre_sub_logits",
                "instrument_probs", "key_logits", "chord_logits", "mode_logits",
                "perceptual", "era_probs", "chart_potential",
            ],
            dynamic_axes={
                "input_values": {0: "batch_size"},
                "embedding": {0: "batch_size"},
                "role_logits": {0: "batch_size"},
                "genre_top_logits": {0: "batch_size"},
                "genre_sub_logits": {0: "batch_size"},
                "instrument_probs": {0: "batch_size"},
                "key_logits": {0: "batch_size"},
                "chord_logits": {0: "batch_size"},
                "mode_logits": {0: "batch_size"},
                "perceptual": {0: "batch_size"},
                "era_probs": {0: "batch_size"},
                "chart_potential": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info(f"Full model exported: {onnx_path}")
        logger.info(f"  Size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

    # Verify ONNX model
    try:
        import onnx
        for name in ["rpm_embedding.onnx", "rpm_full.onnx"]:
            path = output_dir / name
            if path.exists():
                onnx_model = onnx.load(str(path))
                onnx.checker.check_model(onnx_model)
                logger.info(f"✓ {name} passed ONNX validation")
    except ImportError:
        logger.info("Install 'onnx' package to validate exported models")

    logger.info(f"\nExport complete! Models saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Export RPM to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", default="~/.resonate/rpm_models")
    parser.add_argument("--mode", choices=["embedding", "full", "both"], default="both")

    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output_dir, args.mode)
