"""
RESONATE Production Model — Advanced Training Objectives (Phases E–I)

Defines the training losses and model heads for Phases E through I.
These extend the existing RPM model with new capabilities:

Phase E: Contrastive loss — align audio and text in shared embedding space
Phase F: Stem separation loss — understand how stems combine into mixes
Phase G: Structure prediction — detect sections, beats, transitions
Phase H: Knowledge embedding — embed entities in a shared space with audio
Phase I: Self-supervised — masked spectrogram modeling + contrastive + predictive

Each objective is modular — can be combined with existing Phase A–D losses.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Phase E: Contrastive Text-Audio Alignment
# ═══════════════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """
    Lightweight text encoder for music captions.
    Maps text tokens to the same embedding space as audio.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.project_up = nn.Linear(embed_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.project_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token indices
            attention_mask: [B, L] 1=attend, 0=ignore
        Returns:
            [B, output_dim] text embedding
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.project_up(x)

        if attention_mask is not None:
            # TransformerEncoder expects True = ignore
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Pool: mean of non-masked positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.project_out(x)


class ContrastiveLoss(nn.Module):
    """
    InfoNCE / CLIP-style contrastive loss.
    Pulls matching (audio, text) pairs together, pushes non-matching apart.
    Learns a temperature parameter for scaling logits.
    """

    def __init__(self, embed_dim: int = 768, init_temperature: float = 0.07):
        super().__init__()
        # Learnable temperature (log-scale for stability)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

        # Projection heads for audio and text
        self.audio_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        audio_embeddings: torch.Tensor,   # [B, D]
        text_embeddings: torch.Tensor,     # [B, D]
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss."""
        # Project and normalize
        audio_z = F.normalize(self.audio_proj(audio_embeddings), dim=-1)
        text_z = F.normalize(self.text_proj(text_embeddings), dim=-1)

        # Cosine similarity matrix scaled by temperature
        temperature = self.log_temperature.exp().clamp(max=100.0)
        logits = audio_z @ text_z.T * temperature

        # Labels: diagonal entries are positives
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric loss
        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.T, labels)

        return (loss_a2t + loss_t2a) / 2


# ═══════════════════════════════════════════════════════════════════
# Phase F: Stem Understanding
# ═══════════════════════════════════════════════════════════════════

class StemHead(nn.Module):
    """
    Multi-task head for stem understanding:
      1. Stem identification: what instrument is this stem?
      2. Stem contribution: how much does this stem contribute to the mix?
      3. Mix compatibility: do these stems belong together?
    """

    STEM_CLASSES = [
        "drums", "bass", "vocals", "other",  # MUSDB18 standard
        "piano", "guitar", "strings", "brass", "woodwind", "synth",
        "percussion", "keys", "fx", "background_vocals",
    ]

    def __init__(self, embed_dim: int = 768, num_stems: int = 14, hidden_dim: int = 512):
        super().__init__()
        self.num_stems = num_stems

        # Stem classification: what instrument is this?
        self.stem_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_stems),
        )

        # Stem contribution: how prominent is this stem in the mix?
        self.contribution_regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # mix_emb + stem_emb
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Mix compatibility: do these stems sound good together?
        self.compatibility_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def classify_stem(self, stem_embedding: torch.Tensor) -> torch.Tensor:
        """Predict what instrument this stem is. Returns [B, num_stems] logits."""
        return self.stem_classifier(stem_embedding)

    def predict_contribution(
        self, mix_embedding: torch.Tensor, stem_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Predict how much this stem contributes to the mix. Returns [B, 1]."""
        combined = torch.cat([mix_embedding, stem_embedding], dim=-1)
        return self.contribution_regressor(combined)

    def predict_compatibility(
        self, stem_a_embedding: torch.Tensor, stem_b_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Predict if two stems would sound good together. Returns [B, 1]."""
        combined = torch.cat([stem_a_embedding, stem_b_embedding], dim=-1)
        return self.compatibility_head(combined)


class StemLoss(nn.Module):
    """Combined loss for stem understanding tasks."""

    def __init__(self, num_stems: int = 14):
        super().__init__()
        self.stem_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.contribution_mse = nn.MSELoss()
        self.compatibility_bce = nn.BCELoss()

    def forward(
        self,
        stem_logits: torch.Tensor,    # [B, num_stems]
        stem_labels: torch.Tensor,    # [B]
        contribution_pred: Optional[torch.Tensor] = None,  # [B, 1]
        contribution_target: Optional[torch.Tensor] = None,  # [B, 1]
        compatibility_pred: Optional[torch.Tensor] = None,  # [B, 1]
        compatibility_target: Optional[torch.Tensor] = None,  # [B, 1]
    ) -> torch.Tensor:
        loss = self.stem_ce(stem_logits, stem_labels)

        if contribution_pred is not None and contribution_target is not None:
            loss = loss + 0.5 * self.contribution_mse(contribution_pred, contribution_target)

        if compatibility_pred is not None and compatibility_target is not None:
            loss = loss + 0.5 * self.compatibility_bce(compatibility_pred, compatibility_target)

        return loss


# ═══════════════════════════════════════════════════════════════════
# Phase G: Song Structure Detection
# ═══════════════════════════════════════════════════════════════════

class StructureHead(nn.Module):
    """
    Predicts song structure from audio embeddings:
      1. Section type: verse, chorus, bridge, intro, outro, etc.
      2. Beat/downbeat detection
      3. Transition probability: is a section boundary near?
    """

    SECTION_TYPES = [
        "intro", "verse", "pre-chorus", "chorus", "post-chorus",
        "bridge", "breakdown", "drop", "build", "outro",
        "instrumental", "solo", "hook", "silence",
    ]

    def __init__(self, embed_dim: int = 768, num_sections: int = 14, hidden_dim: int = 512):
        super().__init__()

        # Section classifier (from a sequence of frame embeddings)
        self.section_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_sections),
        )

        # Beat/downbeat detector (frame-level binary)
        self.beat_detector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),  # [beat_prob, downbeat_prob]
            nn.Sigmoid(),
        )

        # Transition detector (is there a section boundary nearby?)
        self.transition_detector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            embeddings: [B, D] audio embedding for a segment
        Returns:
            dict with section_logits, beat_probs, transition_prob
        """
        return {
            "section_logits": self.section_classifier(embeddings),  # [B, num_sections]
            "beat_probs": self.beat_detector(embeddings),           # [B, 2]
            "transition_prob": self.transition_detector(embeddings),  # [B, 1]
        }


class StructureLoss(nn.Module):
    """Combined loss for structure prediction."""

    def __init__(self):
        super().__init__()
        self.section_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.beat_bce = nn.BCELoss()
        self.transition_bce = nn.BCELoss()

    def forward(
        self,
        section_logits: torch.Tensor,
        section_labels: torch.Tensor,
        beat_probs: Optional[torch.Tensor] = None,
        beat_labels: Optional[torch.Tensor] = None,
        transition_prob: Optional[torch.Tensor] = None,
        transition_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.section_ce(section_logits, section_labels)

        if beat_probs is not None and beat_labels is not None:
            loss = loss + 0.3 * self.beat_bce(beat_probs, beat_labels)

        if transition_prob is not None and transition_label is not None:
            loss = loss + 0.5 * self.transition_bce(transition_prob, transition_label)

        return loss


# ═══════════════════════════════════════════════════════════════════
# Phase H: Knowledge Graph Embedding
# ═══════════════════════════════════════════════════════════════════

class KnowledgeEmbeddingHead(nn.Module):
    """
    Embeds knowledge graph entities (genres, artists, instruments, techniques)
    into the same embedding space as audio. Enables:
      - Audio → closest genre in graph
      - Audio → most similar artist production style
      - Audio → instrument identification via graph traversal
    """

    def __init__(
        self,
        audio_dim: int = 768,
        entity_dim: int = 256,
        num_entity_types: int = 6,  # artist, genre, instrument, style, technique, era
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.entity_dim = entity_dim

        # Project audio embeddings to entity space
        self.audio_to_entity = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, entity_dim),
        )

        # Entity type embedding
        self.type_embedding = nn.Embedding(num_entity_types, entity_dim)

        # Relation scoring: TransE-style (h + r ≈ t)
        self.relation_projection = nn.Linear(entity_dim, entity_dim, bias=False)

    def project_audio(self, audio_embedding: torch.Tensor) -> torch.Tensor:
        """Project audio embedding to entity space."""
        return F.normalize(self.audio_to_entity(audio_embedding), dim=-1)

    def score_triplet(
        self,
        head: torch.Tensor,    # [B, D]
        relation: torch.Tensor,  # [B, D]
        tail: torch.Tensor,    # [B, D]
    ) -> torch.Tensor:
        """TransE score: ||h + r - t||. Lower = more likely."""
        translated = head + self.relation_projection(relation)
        return -torch.norm(translated - tail, p=2, dim=-1)


class KnowledgeLoss(nn.Module):
    """
    Margin-based ranking loss for knowledge graph embedding.
    Positive triplets should score higher than corrupted negatives.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pos_scores: torch.Tensor,  # [B]
        neg_scores: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        return F.relu(self.margin - pos_scores + neg_scores).mean()


# ═══════════════════════════════════════════════════════════════════
# Phase I: Self-Supervised Objectives
# ═══════════════════════════════════════════════════════════════════

class MaskedSpectrogramModeling(nn.Module):
    """
    BERT-style masked prediction for spectrograms.
    Randomly masks patches of the mel spectrogram and predicts them.
    Teaches the model deep audio structure without any labels.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 16,
        num_freq_bins: int = 128,
        num_time_bins: int = 100,
        mask_ratio: float = 0.4,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        num_patches = (num_freq_bins // patch_size) * (num_time_bins // patch_size)

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Reconstruction head
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, patch_size * patch_size),
        )

    def create_mask(self, batch_size: int, num_patches: int, device: torch.device) -> torch.Tensor:
        """Create random mask. Returns [B, N] boolean tensor (True = masked)."""
        num_masked = int(num_patches * self.mask_ratio)
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_sorted = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_sorted[:, :num_masked], True)
        return mask

    def forward(
        self,
        patch_embeddings: torch.Tensor,  # [B, N, D] from backbone
        target_patches: torch.Tensor,     # [B, N, P*P] original patch values
    ) -> torch.Tensor:
        """Compute MSE loss on masked patch reconstruction."""
        B, N, D = patch_embeddings.shape
        mask = self.create_mask(B, N, patch_embeddings.device)

        # Replace masked patches with mask token
        masked_embeddings = patch_embeddings.clone()
        masked_embeddings[mask] = self.mask_token.expand(B, N, D)[mask]

        # Predict original patch values
        predictions = self.decoder(masked_embeddings)  # [B, N, P*P]

        # Loss only on masked positions
        loss = F.mse_loss(
            predictions[mask],
            target_patches[mask],
        )
        return loss


class AudioContrastiveSSL(nn.Module):
    """
    Self-supervised contrastive: different augmented views of the same
    audio should produce similar embeddings.

    Augmentations: time shift, pitch shift, noise, gain, EQ.
    """

    def __init__(self, embed_dim: int = 768, projection_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        embedding_view1: torch.Tensor,  # [B, D]
        embedding_view2: torch.Tensor,  # [B, D]
    ) -> torch.Tensor:
        """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss."""
        z1 = F.normalize(self.projector(embedding_view1), dim=-1)
        z2 = F.normalize(self.projector(embedding_view2), dim=-1)

        B = z1.size(0)
        temperature = self.temperature.abs().clamp(min=0.01, max=1.0)

        # Similarity matrix: all pairs
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = z @ z.T / temperature  # [2B, 2B]

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=sim.device),
            torch.arange(B, device=sim.device),
        ])

        return F.cross_entropy(sim, labels)


class NextSegmentPrediction(nn.Module):
    """
    Predict if two audio segments are consecutive in the same song.
    Binary classification: [seg_A, seg_B] → {consecutive, random}.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        segment_a: torch.Tensor,  # [B, D]
        segment_b: torch.Tensor,  # [B, D]
        labels: torch.Tensor,     # [B] binary: 1=consecutive, 0=random
    ) -> torch.Tensor:
        combined = torch.cat([segment_a, segment_b], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, labels.float())


# ═══════════════════════════════════════════════════════════════════
# Combined Advanced Loss
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AdvancedLossWeights:
    """Weights for each advanced training objective."""
    contrastive: float = 1.0        # Phase E
    stem_classification: float = 0.5  # Phase F
    stem_contribution: float = 0.3  # Phase F
    structure: float = 0.5          # Phase G
    knowledge: float = 0.3          # Phase H
    masked_spectrogram: float = 1.0  # Phase I
    audio_contrastive: float = 0.5  # Phase I
    next_segment: float = 0.3      # Phase I


class AdvancedRPMLoss(nn.Module):
    """
    Combines all Phase E-I training objectives.
    Each objective is optional — only compute losses for available data.
    """

    def __init__(self, embed_dim: int = 768, weights: Optional[AdvancedLossWeights] = None):
        super().__init__()
        self.weights = weights or AdvancedLossWeights()

        # Phase E
        self.contrastive = ContrastiveLoss(embed_dim)

        # Phase F
        self.stem_head = StemHead(embed_dim)
        self.stem_loss = StemLoss()

        # Phase G
        self.structure_head = StructureHead(embed_dim)
        self.structure_loss = StructureLoss()

        # Phase H
        self.knowledge_head = KnowledgeEmbeddingHead(audio_dim=embed_dim)
        self.knowledge_loss = KnowledgeLoss()

        # Phase I
        self.masked_spec = MaskedSpectrogramModeling(embed_dim)
        self.audio_contrastive = AudioContrastiveSSL(embed_dim)
        self.next_segment = NextSegmentPrediction(embed_dim)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """
        Compute all applicable losses based on available inputs.
        Returns dict of individual losses + total weighted loss.
        """
        losses = {}
        total = torch.tensor(0.0, device=next(self.parameters()).device)

        # Phase E: Contrastive
        if "audio_embeddings" in kwargs and "text_embeddings" in kwargs:
            loss_e = self.contrastive(kwargs["audio_embeddings"], kwargs["text_embeddings"])
            losses["contrastive"] = loss_e
            total = total + self.weights.contrastive * loss_e

        # Phase F: Stems
        if "stem_embeddings" in kwargs and "stem_labels" in kwargs:
            stem_logits = self.stem_head.classify_stem(kwargs["stem_embeddings"])
            loss_f = self.stem_loss(stem_logits, kwargs["stem_labels"])
            losses["stem"] = loss_f
            total = total + self.weights.stem_classification * loss_f

        # Phase G: Structure
        if "segment_embeddings" in kwargs and "section_labels" in kwargs:
            struct_out = self.structure_head(kwargs["segment_embeddings"])
            loss_g = self.structure_loss(struct_out["section_logits"], kwargs["section_labels"])
            losses["structure"] = loss_g
            total = total + self.weights.structure * loss_g

        # Phase H: Knowledge
        if "kg_pos_scores" in kwargs and "kg_neg_scores" in kwargs:
            loss_h = self.knowledge_loss(kwargs["kg_pos_scores"], kwargs["kg_neg_scores"])
            losses["knowledge"] = loss_h
            total = total + self.weights.knowledge * loss_h

        # Phase I: Self-supervised
        if "patch_embeddings" in kwargs and "target_patches" in kwargs:
            loss_msm = self.masked_spec(kwargs["patch_embeddings"], kwargs["target_patches"])
            losses["masked_spectrogram"] = loss_msm
            total = total + self.weights.masked_spectrogram * loss_msm

        if "view1_embeddings" in kwargs and "view2_embeddings" in kwargs:
            loss_ac = self.audio_contrastive(kwargs["view1_embeddings"], kwargs["view2_embeddings"])
            losses["audio_contrastive"] = loss_ac
            total = total + self.weights.audio_contrastive * loss_ac

        if "segment_a" in kwargs and "segment_b" in kwargs and "consecutive_labels" in kwargs:
            loss_nsp = self.next_segment(kwargs["segment_a"], kwargs["segment_b"], kwargs["consecutive_labels"])
            losses["next_segment"] = loss_nsp
            total = total + self.weights.next_segment * loss_nsp

        losses["total"] = total
        return losses


def count_advanced_parameters(loss_module: AdvancedRPMLoss) -> dict[str, int]:
    """Count parameters in each advanced head."""
    counts = {}
    for name, module in [
        ("contrastive", loss_module.contrastive),
        ("stem_head", loss_module.stem_head),
        ("structure_head", loss_module.structure_head),
        ("knowledge_head", loss_module.knowledge_head),
        ("masked_spectrogram", loss_module.masked_spec),
        ("audio_contrastive", loss_module.audio_contrastive),
        ("next_segment", loss_module.next_segment),
    ]:
        counts[name] = sum(p.numel() for p in module.parameters())

    counts["total"] = sum(counts.values())
    return counts
