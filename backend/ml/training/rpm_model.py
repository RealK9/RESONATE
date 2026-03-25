"""
RESONATE Production Model (RPM) — The God Model of Music Production.

Architecture:
    Audio (16kHz, 10s max)
      → ASTFeatureExtractor → mel spectrogram
      → AST backbone (768d CLS token)
      → ProjectionNeck: Linear(768→1024) → GELU → LN → Dropout → Linear(1024→768) → LN
      → 768d RPM embedding
      ├── Head 1: Role Classification (11 classes)
      ├── Head 2: Genre Classification (hierarchical — top + sub)
      ├── Head 3: Instrument Recognition (200+ multi-label)
      ├── Head 4: Music Theory (key, mode, chord quality)
      ├── Head 5: Perceptual Quality (9 descriptors + production markers)
      ├── Head 6: Era/Decade (1950s–2020s ordinal regression)
      ├── Head 7: Chart Potential (regression 0–1)
      └── Head 8: Distillation (CLAP 512d + PANNs 2048d + AST 768d alignment)

Single model replaces CLAP + PANNs + AST with one purpose-built architecture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RPMConfig:
    """All hyperparameters for the RPM model."""

    # Backbone
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    backbone_dim: int = 768
    freeze_backbone: bool = True  # Frozen in phases A/B, unfrozen in C

    # Projection neck
    neck_hidden: int = 1024
    neck_output: int = 768
    neck_dropout: float = 0.1

    # Head 1: Role
    num_roles: int = 11  # kick/snare/clap/hat/perc/bass/lead/pad/fx/texture/vocal

    # Head 2: Genre (hierarchical)
    num_top_genres: int = 12        # electronic, hip-hop, r&b, pop, rock, jazz, country, latin, classical, folk, metal, other
    num_sub_genres: int = 500       # all subgenres
    genre_embed_dim: int = 128      # genre embedding for hierarchical prediction

    # Head 3: Instruments (multi-label)
    num_instruments: int = 200

    # Head 4: Music Theory
    num_keys: int = 24              # 12 major + 12 minor
    num_chord_qualities: int = 12   # major, minor, dom7, maj7, min7, dim, aug, sus2, sus4, dim7, minmaj7, other
    num_modes: int = 7              # Ionian through Locrian

    # Head 5: Perceptual (9 descriptors)
    num_perceptual: int = 9         # brightness, warmth, air, punch, body, bite, smoothness, width, depth

    # Head 6: Era (ordinal regression — 8 decades)
    num_eras: int = 8               # 1950s, 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s

    # Head 7: Chart potential
    # Single regression output (0–1)

    # Head 8: Distillation targets
    clap_dim: int = 512
    panns_dim: int = 2048
    ast_dim: int = 768

    # Training
    learning_rate: float = 5e-4
    backbone_lr: float = 1e-5      # discriminative LR when unfrozen
    weight_decay: float = 0.01
    warmup_steps: int = 500
    label_smoothing: float = 0.1


# ──────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────

class ProjectionNeck(nn.Module):
    """
    Projects the AST CLS token into the RPM embedding space.
    Linear(768→1024) → GELU → LayerNorm → Dropout → Linear(1024→768) → LayerNorm
    """

    def __init__(self, cfg: RPMConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.backbone_dim, cfg.neck_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.neck_hidden),
            nn.Dropout(cfg.neck_dropout),
            nn.Linear(cfg.neck_hidden, cfg.neck_output),
            nn.LayerNorm(cfg.neck_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationHead(nn.Module):
    """Standard classification head: Linear → ReLU → Dropout → Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    """Regression head: Linear → ReLU → Dropout → Linear → Sigmoid."""

    def __init__(self, in_dim: int, hidden_dim: int, num_outputs: int = 1, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class DistillationHead(nn.Module):
    """
    Projects RPM embedding to match teacher model embedding spaces.
    Three separate linear projections for CLAP (512), PANNs (2048), AST (768).
    """

    def __init__(self, in_dim: int, clap_dim: int = 512, panns_dim: int = 2048, ast_dim: int = 768):
        super().__init__()
        self.to_clap = nn.Linear(in_dim, clap_dim)
        self.to_panns = nn.Linear(in_dim, panns_dim)
        self.to_ast = nn.Linear(in_dim, ast_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "clap": self.to_clap(x),
            "panns": self.to_panns(x),
            "ast": self.to_ast(x),
        }


class HierarchicalGenreHead(nn.Module):
    """
    Hierarchical genre classification.
    Step 1: Predict top-level genre (12 classes)
    Step 2: Predict sub-genre conditioned on top-level genre embedding
    """

    def __init__(self, in_dim: int, num_top: int, num_sub: int,
                 genre_embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()

        # Top-level genre classifier
        self.top_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_top),
        )

        # Genre embedding lookup for conditioning
        self.genre_embed = nn.Embedding(num_top, genre_embed_dim)

        # Sub-genre classifier (conditioned on top genre embedding)
        self.sub_head = nn.Sequential(
            nn.Linear(in_dim + genre_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_sub),
        )

    def forward(self, x: torch.Tensor,
                top_genre_label: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Args:
            x: RPM embedding [B, D]
            top_genre_label: ground truth top genre during training [B] (use argmax at inference)
        Returns:
            dict with 'top_logits' and 'sub_logits'
        """
        top_logits = self.top_head(x)

        # During training use ground truth; at inference use predicted
        if top_genre_label is not None:
            genre_idx = top_genre_label
        else:
            genre_idx = top_logits.argmax(dim=-1)

        genre_emb = self.genre_embed(genre_idx)  # [B, genre_embed_dim]
        conditioned = torch.cat([x, genre_emb], dim=-1)  # [B, D + genre_embed_dim]
        sub_logits = self.sub_head(conditioned)

        return {"top_logits": top_logits, "sub_logits": sub_logits}


class MusicTheoryHead(nn.Module):
    """
    Multi-output music theory prediction.
    Predicts: key (24), chord quality (12), mode (7) simultaneously.
    """

    def __init__(self, in_dim: int, num_keys: int = 24,
                 num_chord_qualities: int = 12, num_modes: int = 7, dropout: float = 0.2):
        super().__init__()

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.key_head = nn.Linear(512, num_keys)
        self.chord_head = nn.Linear(512, num_chord_qualities)
        self.mode_head = nn.Linear(512, num_modes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(x)
        return {
            "key_logits": self.key_head(shared),
            "chord_logits": self.chord_head(shared),
            "mode_logits": self.mode_head(shared),
        }


class OrdinalEraHead(nn.Module):
    """
    Ordinal regression for era/decade prediction.
    Instead of independent softmax, uses cumulative logits to preserve
    the ordered relationship: 1950s < 1960s < ... < 2020s.

    Outputs K-1 binary classifiers: P(era > k) for k = 0..K-2
    Final probabilities derived from cumulative distribution.
    """

    def __init__(self, in_dim: int, num_eras: int = 8, dropout: float = 0.2):
        super().__init__()
        self.num_eras = num_eras
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_eras - 1),  # K-1 cumulative thresholds
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns:
            cumulative_logits: [B, K-1] — raw logits for P(era > k)
            probabilities: [B, K] — per-class probabilities
        """
        cum_logits = self.net(x)  # [B, K-1]
        cum_probs = torch.sigmoid(cum_logits)  # P(era > k)

        # Convert cumulative to per-class: P(era = k) = P(era > k-1) - P(era > k)
        # P(era = 0) = 1 - P(era > 0)
        # P(era = K-1) = P(era > K-2)
        ones = torch.ones(x.shape[0], 1, device=x.device)
        zeros = torch.zeros(x.shape[0], 1, device=x.device)
        extended = torch.cat([ones, cum_probs, zeros], dim=1)  # [B, K+1]
        probs = extended[:, :-1] - extended[:, 1:]  # [B, K]

        return {"cumulative_logits": cum_logits, "probabilities": probs}


# ──────────────────────────────────────────────────────────────────────
# Main RPM Model
# ──────────────────────────────────────────────────────────────────────

class RPMModel(nn.Module):
    """
    RESONATE Production Model — single unified model for music intelligence.

    Input: raw audio waveform (16kHz, up to 10s)
    Output: 768d embedding + predictions from 8 task heads

    Replaces: CLAP (512d) + PANNs (2048d) + AST (768d) with one model.
    """

    def __init__(self, cfg: RPMConfig = None):
        super().__init__()
        if cfg is None:
            cfg = RPMConfig()
        self.cfg = cfg

        # ── AST Backbone ──
        # Loaded lazily to avoid importing transformers at module load
        self._backbone = None
        self._feature_extractor = None

        # ── Projection Neck ──
        self.neck = ProjectionNeck(cfg)

        # ── Task Heads ──
        D = cfg.neck_output  # 768

        # Head 1: Role classification
        self.role_head = ClassificationHead(D, 256, cfg.num_roles)

        # Head 2: Hierarchical genre classification
        self.genre_head = HierarchicalGenreHead(
            D, cfg.num_top_genres, cfg.num_sub_genres,
            cfg.genre_embed_dim
        )

        # Head 3: Instrument recognition (multi-label)
        self.instrument_head = ClassificationHead(D, 512, cfg.num_instruments)

        # Head 4: Music theory
        self.theory_head = MusicTheoryHead(
            D, cfg.num_keys, cfg.num_chord_qualities, cfg.num_modes
        )

        # Head 5: Perceptual quality (regression)
        self.perceptual_head = RegressionHead(D, 256, cfg.num_perceptual)

        # Head 6: Era/decade (ordinal regression)
        self.era_head = OrdinalEraHead(D, cfg.num_eras)

        # Head 7: Chart potential (single regression)
        self.chart_head = RegressionHead(D, 128, 1)

        # Head 8: Distillation
        self.distill_head = DistillationHead(
            D, cfg.clap_dim, cfg.panns_dim, cfg.ast_dim
        )

    @property
    def backbone(self):
        if self._backbone is None:
            from transformers import ASTModel
            self._backbone = ASTModel.from_pretrained(self.cfg.ast_model_name)
            if self.cfg.freeze_backbone:
                for param in self._backbone.parameters():
                    param.requires_grad = False
        return self._backbone

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            from transformers import ASTFeatureExtractor
            self._feature_extractor = ASTFeatureExtractor.from_pretrained(
                self.cfg.ast_model_name
            )
        return self._feature_extractor

    def freeze_backbone(self):
        """Freeze the AST backbone parameters."""
        if self._backbone is not None:
            for param in self._backbone.parameters():
                param.requires_grad = False
        self.cfg.freeze_backbone = True

    def unfreeze_backbone(self):
        """Unfreeze the AST backbone for fine-tuning (Phase C)."""
        _ = self.backbone  # ensure loaded
        for param in self._backbone.parameters():
            param.requires_grad = True
        self.cfg.freeze_backbone = False

    def extract_backbone_features(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Run audio through AST backbone → CLS token.
        Args:
            input_values: mel spectrogram features [B, T, F] from ASTFeatureExtractor
        Returns:
            CLS token embedding [B, 768]
        """
        outputs = self.backbone(input_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        return cls_token

    def get_embedding(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Extract the 768-d RPM embedding (backbone → neck).
        This is the embedding stored in FAISS for retrieval.
        """
        cls_token = self.extract_backbone_features(input_values)
        embedding = self.neck(cls_token)
        return embedding

    def forward(self, input_values: torch.Tensor,
                top_genre_label: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Full forward pass through all heads.

        Args:
            input_values: mel spectrogram features [B, T, F]
            top_genre_label: ground truth top-level genre [B] (for training)

        Returns:
            dict with:
                'embedding': [B, 768] — the RPM embedding
                'role_logits': [B, 11]
                'genre_top_logits': [B, 12]
                'genre_sub_logits': [B, 500]
                'instrument_logits': [B, 200]
                'key_logits': [B, 24]
                'chord_logits': [B, 12]
                'mode_logits': [B, 7]
                'perceptual': [B, 9]
                'era_cumulative_logits': [B, 7]
                'era_probabilities': [B, 8]
                'chart_potential': [B, 1]
                'distill_clap': [B, 512]
                'distill_panns': [B, 2048]
                'distill_ast': [B, 768]
        """
        # Backbone + projection
        embedding = self.get_embedding(input_values)

        # Head 1: Role
        role_logits = self.role_head(embedding)

        # Head 2: Genre (hierarchical)
        genre_out = self.genre_head(embedding, top_genre_label)

        # Head 3: Instruments (multi-label)
        instrument_logits = self.instrument_head(embedding)

        # Head 4: Music theory
        theory_out = self.theory_head(embedding)

        # Head 5: Perceptual
        perceptual = self.perceptual_head(embedding)

        # Head 6: Era
        era_out = self.era_head(embedding)

        # Head 7: Chart potential
        chart = self.chart_head(embedding)

        # Head 8: Distillation
        distill_out = self.distill_head(embedding)

        return {
            "embedding": embedding,
            "role_logits": role_logits,
            "genre_top_logits": genre_out["top_logits"],
            "genre_sub_logits": genre_out["sub_logits"],
            "instrument_logits": instrument_logits,
            "key_logits": theory_out["key_logits"],
            "chord_logits": theory_out["chord_logits"],
            "mode_logits": theory_out["mode_logits"],
            "perceptual": perceptual,
            "era_cumulative_logits": era_out["cumulative_logits"],
            "era_probabilities": era_out["probabilities"],
            "chart_potential": chart,
            "distill_clap": distill_out["clap"],
            "distill_panns": distill_out["panns"],
            "distill_ast": distill_out["ast"],
        }


# ──────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────

class RPMLoss(nn.Module):
    """
    Multi-task loss for RPM training.

    Combines:
      - Cross-entropy for classification heads (role, genre, theory)
      - BCE for multi-label (instruments)
      - MSE for regression (perceptual, chart)
      - BCE for ordinal (era)
      - Cosine + MSE for distillation

    Loss weights are adjustable per training phase.
    """

    def __init__(self, cfg: RPMConfig = None, label_smoothing: float = 0.1):
        super().__init__()
        if cfg is None:
            cfg = RPMConfig()
        self.cfg = cfg

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()

        # Default loss weights — adjusted per training phase
        self.weights = {
            "role": 1.0,
            "genre_top": 1.0,
            "genre_sub": 0.5,
            "instrument": 1.0,
            "key": 1.0,
            "chord": 0.5,
            "mode": 0.5,
            "perceptual": 1.0,
            "era": 0.5,
            "chart": 0.5,
            "distill_clap": 2.0,
            "distill_panns": 2.0,
            "distill_ast": 2.0,
        }

    def set_phase_weights(self, phase: str):
        """
        Set loss weights per training phase.
        Phase A: Distillation only
        Phase B: Multi-task on local data
        Phase C: Full training
        Phase D: Chart intelligence fine-tune
        """
        if phase == "A":
            # Distillation only — zero out all task losses
            for k in self.weights:
                self.weights[k] = 0.0
            self.weights["distill_clap"] = 2.0
            self.weights["distill_panns"] = 2.0
            self.weights["distill_ast"] = 2.0

        elif phase == "B":
            # Multi-task on local data — all heads active, distillation reduced
            self.weights = {
                "role": 2.0,
                "genre_top": 1.5,
                "genre_sub": 0.8,
                "instrument": 1.0,
                "key": 1.0,
                "chord": 0.5,
                "mode": 0.5,
                "perceptual": 1.5,
                "era": 0.3,
                "chart": 0.0,  # no chart data yet
                "distill_clap": 1.0,
                "distill_panns": 1.0,
                "distill_ast": 1.0,
            }

        elif phase == "C":
            # Full training — everything at full weight
            self.weights = {
                "role": 1.5,
                "genre_top": 2.0,
                "genre_sub": 1.0,
                "instrument": 2.0,
                "key": 1.5,
                "chord": 1.0,
                "mode": 0.8,
                "perceptual": 1.5,
                "era": 1.0,
                "chart": 0.5,
                "distill_clap": 1.0,
                "distill_panns": 1.0,
                "distill_ast": 1.0,
            }

        elif phase == "D":
            # Chart intelligence — focus on era + chart heads
            for k in self.weights:
                self.weights[k] = 0.0
            self.weights["era"] = 2.0
            self.weights["chart"] = 3.0
            self.weights["genre_top"] = 0.5
            self.weights["perceptual"] = 0.5

    def forward(self, outputs: dict[str, torch.Tensor],
                targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss.

        Args:
            outputs: from RPMModel.forward()
            targets: dict with keys matching output names, containing ground truth

        Returns:
            dict with individual losses and 'total'
        """
        losses = {}
        total = torch.tensor(0.0, device=outputs["embedding"].device)

        # Head 1: Role (cross-entropy)
        if "role" in targets and self.weights["role"] > 0:
            loss = self.ce(outputs["role_logits"], targets["role"])
            losses["role"] = loss
            total = total + self.weights["role"] * loss

        # Head 2: Genre (hierarchical cross-entropy)
        if "genre_top" in targets and self.weights["genre_top"] > 0:
            loss = self.ce(outputs["genre_top_logits"], targets["genre_top"])
            losses["genre_top"] = loss
            total = total + self.weights["genre_top"] * loss

        if "genre_sub" in targets and self.weights["genre_sub"] > 0:
            loss = self.ce(outputs["genre_sub_logits"], targets["genre_sub"])
            losses["genre_sub"] = loss
            total = total + self.weights["genre_sub"] * loss

        # Head 3: Instruments (multi-label BCE)
        if "instruments" in targets and self.weights["instrument"] > 0:
            loss = self.bce(outputs["instrument_logits"], targets["instruments"].float())
            losses["instrument"] = loss
            total = total + self.weights["instrument"] * loss

        # Head 4: Music theory
        if "key" in targets and self.weights["key"] > 0:
            loss = self.ce(outputs["key_logits"], targets["key"])
            losses["key"] = loss
            total = total + self.weights["key"] * loss

        if "chord" in targets and self.weights["chord"] > 0:
            loss = self.ce(outputs["chord_logits"], targets["chord"])
            losses["chord"] = loss
            total = total + self.weights["chord"] * loss

        if "mode" in targets and self.weights["mode"] > 0:
            loss = self.ce(outputs["mode_logits"], targets["mode"])
            losses["mode"] = loss
            total = total + self.weights["mode"] * loss

        # Head 5: Perceptual (MSE regression)
        if "perceptual" in targets and self.weights["perceptual"] > 0:
            loss = self.mse(outputs["perceptual"], targets["perceptual"].float())
            losses["perceptual"] = loss
            total = total + self.weights["perceptual"] * loss

        # Head 6: Era (ordinal — BCE on cumulative logits)
        if "era" in targets and self.weights["era"] > 0:
            # Convert era label to cumulative binary targets
            # era=3 (1980s) → [1, 1, 1, 0, 0, 0, 0] (> 1950s, > 1960s, > 1970s)
            era_labels = targets["era"]  # [B]
            num_thresholds = self.cfg.num_eras - 1
            cum_targets = torch.zeros(
                era_labels.shape[0], num_thresholds,
                device=era_labels.device
            )
            for k in range(num_thresholds):
                cum_targets[:, k] = (era_labels > k).float()

            loss = self.bce(outputs["era_cumulative_logits"], cum_targets)
            losses["era"] = loss
            total = total + self.weights["era"] * loss

        # Head 7: Chart potential (MSE)
        if "chart_potential" in targets and self.weights["chart"] > 0:
            loss = self.mse(outputs["chart_potential"].squeeze(-1),
                          targets["chart_potential"].float())
            losses["chart"] = loss
            total = total + self.weights["chart"] * loss

        # Head 8: Distillation (cosine similarity + MSE)
        ones = torch.ones(outputs["embedding"].shape[0],
                         device=outputs["embedding"].device)

        if "clap_embedding" in targets and self.weights["distill_clap"] > 0:
            cosine_loss = self.cosine(outputs["distill_clap"],
                                     targets["clap_embedding"], ones)
            mse_loss = self.mse(outputs["distill_clap"], targets["clap_embedding"])
            loss = 0.7 * cosine_loss + 0.3 * mse_loss
            losses["distill_clap"] = loss
            total = total + self.weights["distill_clap"] * loss

        if "panns_embedding" in targets and self.weights["distill_panns"] > 0:
            cosine_loss = self.cosine(outputs["distill_panns"],
                                     targets["panns_embedding"], ones)
            mse_loss = self.mse(outputs["distill_panns"], targets["panns_embedding"])
            loss = 0.7 * cosine_loss + 0.3 * mse_loss
            losses["distill_panns"] = loss
            total = total + self.weights["distill_panns"] * loss

        if "ast_embedding" in targets and self.weights["distill_ast"] > 0:
            cosine_loss = self.cosine(outputs["distill_ast"],
                                     targets["ast_embedding"], ones)
            mse_loss = self.mse(outputs["distill_ast"], targets["ast_embedding"])
            loss = 0.7 * cosine_loss + 0.3 * mse_loss
            losses["distill_ast"] = loss
            total = total + self.weights["distill_ast"] * loss

        losses["total"] = total
        return losses


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def get_param_groups(model: RPMModel, cfg: RPMConfig) -> list[dict]:
    """
    Create parameter groups with discriminative learning rates.
    Backbone gets cfg.backbone_lr, everything else gets cfg.learning_rate.
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone") or name.startswith("_backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return [
        {"params": backbone_params, "lr": cfg.backbone_lr},
        {"params": head_params, "lr": cfg.learning_rate},
    ]


def build_rpm_model(cfg: RPMConfig = None, device: str = "cpu") -> RPMModel:
    """Factory function to build and initialize the RPM model."""
    if cfg is None:
        cfg = RPMConfig()
    model = RPMModel(cfg)
    model = model.to(device)
    return model


# ──────────────────────────────────────────────────────────────────────
# Quick validation
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("RESONATE Production Model (RPM) — Architecture Validation")
    print("=" * 70)

    cfg = RPMConfig()
    model = RPMModel(cfg)

    # Test forward pass with dummy input (skip backbone, test heads only)
    B = 4  # batch size
    D = cfg.neck_output  # 768

    # Simulate post-neck embeddings
    dummy_embedding = torch.randn(B, D)

    print("\n[Head Tests — forward pass with dummy embeddings]")

    # Role
    role_out = model.role_head(dummy_embedding)
    print(f"  Role head:        input={dummy_embedding.shape} → output={role_out.shape}")

    # Genre
    genre_out = model.genre_head(dummy_embedding)
    print(f"  Genre head (top): input={dummy_embedding.shape} → output={genre_out['top_logits'].shape}")
    print(f"  Genre head (sub): input={dummy_embedding.shape} → output={genre_out['sub_logits'].shape}")

    # Instruments
    inst_out = model.instrument_head(dummy_embedding)
    print(f"  Instrument head:  input={dummy_embedding.shape} → output={inst_out.shape}")

    # Theory
    theory_out = model.theory_head(dummy_embedding)
    print(f"  Theory head (key):   output={theory_out['key_logits'].shape}")
    print(f"  Theory head (chord): output={theory_out['chord_logits'].shape}")
    print(f"  Theory head (mode):  output={theory_out['mode_logits'].shape}")

    # Perceptual
    perc_out = model.perceptual_head(dummy_embedding)
    print(f"  Perceptual head:  input={dummy_embedding.shape} → output={perc_out.shape}")

    # Era
    era_out = model.era_head(dummy_embedding)
    print(f"  Era head (cum):   output={era_out['cumulative_logits'].shape}")
    print(f"  Era head (probs): output={era_out['probabilities'].shape}")

    # Chart
    chart_out = model.chart_head(dummy_embedding)
    print(f"  Chart head:       input={dummy_embedding.shape} → output={chart_out.shape}")

    # Distillation
    dist_out = model.distill_head(dummy_embedding)
    print(f"  Distill (CLAP):   output={dist_out['clap'].shape}")
    print(f"  Distill (PANNs):  output={dist_out['panns'].shape}")
    print(f"  Distill (AST):    output={dist_out['ast'].shape}")

    # Parameter counts (excluding backbone since it's lazy-loaded)
    neck_params = count_parameters(model.neck)
    print(f"\n[Parameter Counts]")
    print(f"  Projection neck:  {neck_params['total']:,}")

    heads_total = 0
    for name in ["role_head", "genre_head", "instrument_head", "theory_head",
                 "perceptual_head", "era_head", "chart_head", "distill_head"]:
        head = getattr(model, name)
        p = count_parameters(head)
        heads_total += p["total"]
        print(f"  {name:20s}: {p['total']:>10,}")

    print(f"  {'TOTAL (heads+neck)':20s}: {heads_total + neck_params['total']:>10,}")
    print(f"\n  AST backbone (frozen): ~86M parameters (loaded on demand)")
    print(f"  Total with backbone:   ~{86_000_000 + heads_total + neck_params['total']:,}")

    # Test loss computation
    print("\n[Loss Function Test]")
    loss_fn = RPMLoss(cfg)

    dummy_outputs = {
        "embedding": dummy_embedding,
        "role_logits": role_out,
        "genre_top_logits": genre_out["top_logits"],
        "genre_sub_logits": genre_out["sub_logits"],
        "instrument_logits": inst_out,
        "key_logits": theory_out["key_logits"],
        "chord_logits": theory_out["chord_logits"],
        "mode_logits": theory_out["mode_logits"],
        "perceptual": perc_out,
        "era_cumulative_logits": era_out["cumulative_logits"],
        "era_probabilities": era_out["probabilities"],
        "chart_potential": chart_out,
        "distill_clap": dist_out["clap"],
        "distill_panns": dist_out["panns"],
        "distill_ast": dist_out["ast"],
    }

    dummy_targets = {
        "role": torch.randint(0, cfg.num_roles, (B,)),
        "genre_top": torch.randint(0, cfg.num_top_genres, (B,)),
        "genre_sub": torch.randint(0, cfg.num_sub_genres, (B,)),
        "instruments": torch.randint(0, 2, (B, cfg.num_instruments)),
        "key": torch.randint(0, cfg.num_keys, (B,)),
        "chord": torch.randint(0, cfg.num_chord_qualities, (B,)),
        "mode": torch.randint(0, cfg.num_modes, (B,)),
        "perceptual": torch.rand(B, cfg.num_perceptual),
        "era": torch.randint(0, cfg.num_eras, (B,)),
        "chart_potential": torch.rand(B),
        "clap_embedding": torch.randn(B, cfg.clap_dim),
        "panns_embedding": torch.randn(B, cfg.panns_dim),
        "ast_embedding": torch.randn(B, cfg.ast_dim),
    }

    losses = loss_fn(dummy_outputs, dummy_targets)
    for k, v in losses.items():
        print(f"  {k:20s}: {v.item():.4f}")

    print("\n✓ All heads operational. RPM architecture validated.")
    print("=" * 70)
