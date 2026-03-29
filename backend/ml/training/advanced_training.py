"""
RESONATE Production Model — Advanced Training Pipeline (Phases E-I)

Extends the base RPM model with advanced capabilities:
  Phase E: Contrastive text-audio alignment (CLIP-style)
  Phase F: Stem understanding (identification, contribution, compatibility)
  Phase G: Song structure detection (sections, beats, transitions)
  Phase H: Knowledge graph embedding (TransE-style entity alignment)
  Phase I: Self-supervised pre-training (masked spectrogram, contrastive, NSP)

Designed for H100 80GB — aggressive batch sizes, BF16, gradient accumulation.

SoniqLabs — No shortcuts. Highest quality. All out.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader

from ml.training.rpm_model import RPMModel, RPMConfig, count_parameters
from ml.training.rpm_trainer import TrainingConfig, TrainingMetrics, resolve_device
from ml.training.rpm_dataset import (
    DatasetConfig, load_audio, augment_audio, rpm_collate_fn,
)
from ml.training.data_pipelines.training_objectives import (
    TextEncoder,
    ContrastiveLoss,
    StemHead,
    StemLoss,
    StructureHead,
    StructureLoss,
    KnowledgeEmbeddingHead,
    KnowledgeLoss,
    MaskedSpectrogramModeling,
    AudioContrastiveSSL,
    NextSegmentPrediction,
)

logger = logging.getLogger(__name__)


# ======================================================================
# A) AdvancedRPMModel — wraps RPMModel + Phase E-I heads
# ======================================================================

class AdvancedRPMModel(nn.Module):
    """
    Wraps the trained Phase A-D RPMModel and adds Phase E-I heads.

    Architecture:
        RPMModel (frozen backbone + trained heads)
          -> 768d RPM embedding
          |-- Phase E: TextEncoder + ContrastiveLoss projections
          |-- Phase F: StemHead (classify, contribution, compatibility)
          |-- Phase G: StructureHead (sections, beats, transitions)
          |-- Phase H: KnowledgeEmbeddingHead (TransE entity alignment)
          |-- Phase I: MaskedSpectrogramModeling + AudioContrastiveSSL + NSP
    """

    def __init__(
        self,
        base_model: RPMModel,
        embed_dim: int = 768,
        text_vocab_size: int = 32000,
        text_max_seq_len: int = 128,
        num_stems: int = 14,
        num_sections: int = 14,
        num_entity_types: int = 6,
        entity_dim: int = 256,
    ):
        super().__init__()

        self.base_model = base_model
        self.embed_dim = embed_dim

        # Phase E: Text-audio contrastive alignment
        self.text_encoder = TextEncoder(
            vocab_size=text_vocab_size,
            embed_dim=256,
            hidden_dim=512,
            output_dim=embed_dim,
            num_layers=4,
            num_heads=8,
            max_seq_len=text_max_seq_len,
        )
        self.contrastive_loss = ContrastiveLoss(embed_dim=embed_dim)

        # Phase F: Stem understanding
        self.stem_head = StemHead(
            embed_dim=embed_dim,
            num_stems=num_stems,
        )
        self.stem_loss = StemLoss(num_stems=num_stems)

        # Phase G: Structure detection
        self.structure_head = StructureHead(
            embed_dim=embed_dim,
            num_sections=num_sections,
        )
        self.structure_loss = StructureLoss()

        # Phase H: Knowledge graph embedding
        self.knowledge_head = KnowledgeEmbeddingHead(
            audio_dim=embed_dim,
            entity_dim=entity_dim,
            num_entity_types=num_entity_types,
        )
        self.knowledge_loss = KnowledgeLoss()

        # Phase I: Self-supervised
        self.masked_spec = MaskedSpectrogramModeling(embed_dim=embed_dim)
        self.audio_contrastive = AudioContrastiveSSL(embed_dim=embed_dim)
        self.next_segment = NextSegmentPrediction(embed_dim=embed_dim)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        rpm_config: RPMConfig = None,
        **kwargs,
    ) -> "AdvancedRPMModel":
        """
        Load trained Phase A-D model from checkpoint and wrap with Phase E-I heads.

        Args:
            checkpoint_path: path to rpm_final.pt or any Phase A-D checkpoint
            device: target device
            rpm_config: RPMConfig for the base model (default if None)
            **kwargs: forwarded to AdvancedRPMModel.__init__
        """
        if rpm_config is None:
            rpm_config = RPMConfig()

        base_model = RPMModel(rpm_config)
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Handle both raw state_dict and checkpoint dict
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        cleaned = {}
        for k, v in state_dict.items():
            clean_key = k.replace("_orig_mod.", "")
            cleaned[clean_key] = v
        state_dict = cleaned

        base_model.load_state_dict(state_dict, strict=False)
        base_model = base_model.to(device)

        # Force-load backbone so it's on the right device
        _ = base_model.backbone

        logger.info(f"Loaded base RPM model from {checkpoint_path}")
        params = count_parameters(base_model)
        logger.info(f"  Base model: {params['total']:,} params ({params['trainable']:,} trainable)")

        model = cls(base_model=base_model, embed_dim=rpm_config.neck_output, **kwargs)
        model = model.to(device)
        return model

    def freeze_base_model(self):
        """Freeze the entire base RPM model (backbone + Phase A-D heads)."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Base RPM model frozen (backbone + Phase A-D heads)")

    def unfreeze_backbone(self):
        """Unfreeze the AST backbone for joint fine-tuning."""
        self.base_model.unfreeze_backbone()
        logger.info("AST backbone unfrozen for joint fine-tuning")

    def get_embedding(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extract 768d RPM embedding from audio input."""
        return self.base_model.get_embedding(input_values)

    def forward(
        self,
        input_values: torch.Tensor,
        active_phases: str = "EFGHI",
        # Phase E inputs
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        # Phase F inputs
        stem_input_values: Optional[torch.Tensor] = None,
        stem_labels: Optional[torch.Tensor] = None,
        # Phase G inputs (section_labels, beat_labels passed as targets)
        # Phase H inputs
        relation_embeddings: Optional[torch.Tensor] = None,
        tail_input_values: Optional[torch.Tensor] = None,
        neg_tail_input_values: Optional[torch.Tensor] = None,
        # Phase I inputs
        input_values_view2: Optional[torch.Tensor] = None,
        segment_b_input_values: Optional[torch.Tensor] = None,
        consecutive_labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass returning outputs based on which phases are active.

        Args:
            input_values: mel spectrogram [B, T, F] — always required
            active_phases: string of active phase letters, e.g. "E" or "EFGHI"
            [phase-specific inputs documented inline]

        Returns:
            dict of all computed outputs keyed by phase + output name
        """
        outputs = {}

        # Always compute the base embedding
        embedding = self.get_embedding(input_values)
        outputs["embedding"] = embedding

        # Phase E: Contrastive text-audio
        if "E" in active_phases and text_input_ids is not None:
            text_embedding = self.text_encoder(text_input_ids, text_attention_mask)
            outputs["text_embedding"] = text_embedding
            outputs["audio_embedding"] = embedding
            # Loss is computed externally via self.contrastive_loss

        # Phase F: Stem understanding
        if "F" in active_phases:
            # Classify stem from its embedding
            if stem_input_values is not None:
                stem_embedding = self.get_embedding(stem_input_values)
                outputs["stem_embedding"] = stem_embedding
                outputs["stem_logits"] = self.stem_head.classify_stem(stem_embedding)

                # Mix-stem contribution
                contribution = self.stem_head.predict_contribution(embedding, stem_embedding)
                outputs["stem_contribution"] = contribution

                # Compatibility (between mix embedding and stem)
                compatibility = self.stem_head.predict_compatibility(embedding, stem_embedding)
                outputs["stem_compatibility"] = compatibility

        # Phase G: Structure detection
        if "G" in active_phases:
            structure_out = self.structure_head(embedding)
            outputs["section_logits"] = structure_out["section_logits"]
            outputs["beat_probs"] = structure_out["beat_probs"]
            outputs["transition_prob"] = structure_out["transition_prob"]

        # Phase H: Knowledge graph embedding
        if "H" in active_phases:
            head_entity = self.knowledge_head.project_audio(embedding)
            outputs["head_entity"] = head_entity

            if tail_input_values is not None and relation_embeddings is not None:
                tail_embedding = self.get_embedding(tail_input_values)
                tail_entity = self.knowledge_head.project_audio(tail_embedding)
                outputs["tail_entity"] = tail_entity

                # Positive triplet score
                pos_score = self.knowledge_head.score_triplet(
                    head_entity, relation_embeddings, tail_entity
                )
                outputs["kg_pos_scores"] = pos_score

                # Negative triplet score
                if neg_tail_input_values is not None:
                    neg_tail_embedding = self.get_embedding(neg_tail_input_values)
                    neg_tail_entity = self.knowledge_head.project_audio(neg_tail_embedding)
                    neg_score = self.knowledge_head.score_triplet(
                        head_entity, relation_embeddings, neg_tail_entity
                    )
                    outputs["kg_neg_scores"] = neg_score

        # Phase I: Self-supervised
        if "I" in active_phases:
            # Audio contrastive (two views of same audio)
            if input_values_view2 is not None:
                embedding_v2 = self.get_embedding(input_values_view2)
                outputs["view1_embedding"] = embedding
                outputs["view2_embedding"] = embedding_v2

            # Next segment prediction
            if segment_b_input_values is not None and consecutive_labels is not None:
                seg_b_embedding = self.get_embedding(segment_b_input_values)
                outputs["segment_a_embedding"] = embedding
                outputs["segment_b_embedding"] = seg_b_embedding
                outputs["consecutive_labels"] = consecutive_labels

        return outputs


# ======================================================================
# B) Phase-Specific Dataset Classes
# ======================================================================

class PhaseEDataset(Dataset):
    """
    Text-audio pairs for contrastive learning (Phase E).

    Loads from downloaded data produced by phase_e_contrastive.py:
      - JSON metadata files with captions + audio paths
      - Audio files (WAV/MP3/FLAC)

    Returns: input_values (mel spec) + tokenized text (input_ids + attention_mask)
    """

    def __init__(
        self,
        pairs: list[dict],
        feature_extractor=None,
        tokenizer=None,
        sample_rate: int = 16000,
        max_duration: float = 10.0,
        max_text_len: int = 128,
        augment: bool = True,
    ):
        self.pairs = pairs
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_text_len = max_text_len
        self.augment = augment
        self.target_length = int(sample_rate * max_duration)

        logger.info(f"PhaseEDataset: {len(pairs):,} text-audio pairs")

    @classmethod
    def from_downloaded_data(
        cls,
        data_dir: str,
        feature_extractor=None,
        tokenizer=None,
        **kwargs,
    ) -> "PhaseEDataset":
        """
        Build dataset from downloaded Phase E data.
        Scans data_dir for JSON metadata and matching audio files.

        Expected structure:
            data_dir/
              wavcaps/
                metadata.json  (list of {audio_path, caption, ...})
                audio/
              musiccaps/
                metadata.json
                audio/
              lp_musiccaps/
                metadata.json
                audio/
        """
        data_path = Path(data_dir)
        pairs = []

        # Scan all subdirectories for metadata JSON files
        for source_dir in sorted(data_path.iterdir()):
            if not source_dir.is_dir():
                continue

            # Try multiple metadata file patterns
            for meta_name in ["metadata.json", "pairs.json", "captions.json"]:
                meta_file = source_dir / meta_name
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            entries = json.load(f)
                        if isinstance(entries, dict) and "data" in entries:
                            entries = entries["data"]
                        for entry in entries:
                            audio_path = entry.get("audio_path", "")
                            caption = entry.get("caption", entry.get("text", ""))
                            if not audio_path or not caption:
                                continue
                            # Resolve relative paths
                            if not os.path.isabs(audio_path):
                                audio_path = str(source_dir / audio_path)
                            if os.path.exists(audio_path):
                                pairs.append({
                                    "audio_path": audio_path,
                                    "caption": caption,
                                    "source": source_dir.name,
                                })
                        logger.info(f"  {source_dir.name}: loaded {len(entries)} entries from {meta_name}")
                    except Exception as e:
                        logger.warning(f"  Failed to load {meta_file}: {e}")

            # Also scan for JSONL format
            for jsonl_name in ["metadata.jsonl", "pairs.jsonl"]:
                jsonl_file = source_dir / jsonl_name
                if jsonl_file.exists():
                    count = 0
                    try:
                        with open(jsonl_file) as f:
                            for line in f:
                                entry = json.loads(line.strip())
                                audio_path = entry.get("audio_path", "")
                                caption = entry.get("caption", entry.get("text", ""))
                                if not audio_path or not caption:
                                    continue
                                if not os.path.isabs(audio_path):
                                    audio_path = str(source_dir / audio_path)
                                if os.path.exists(audio_path):
                                    pairs.append({
                                        "audio_path": audio_path,
                                        "caption": caption,
                                        "source": source_dir.name,
                                    })
                                    count += 1
                        logger.info(f"  {source_dir.name}: loaded {count} entries from {jsonl_name}")
                    except Exception as e:
                        logger.warning(f"  Failed to load {jsonl_file}: {e}")

        logger.info(f"PhaseEDataset.from_downloaded_data: {len(pairs):,} total pairs from {data_dir}")
        return cls(pairs, feature_extractor=feature_extractor, tokenizer=tokenizer, **kwargs)

    def __len__(self) -> int:
        return len(self.pairs)

    def _tokenize_text(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize caption text. Falls back to simple char-level if no tokenizer."""
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
        else:
            # Simple byte-level tokenization fallback
            tokens = [ord(c) % 32000 for c in text[:self.max_text_len]]
            pad_len = self.max_text_len - len(tokens)
            input_ids = tokens + [0] * pad_len
            attention_mask = [1] * len(tokens) + [0] * pad_len
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        # Load audio
        try:
            audio = load_audio(
                pair["audio_path"],
                sr=self.sample_rate,
                max_duration=self.max_duration,
            )
        except Exception:
            audio = np.zeros(self.target_length, dtype=np.float32)

        if self.augment:
            audio = augment_audio(audio, sr=self.sample_rate)

        # Extract mel features
        if self.feature_extractor is not None:
            inputs = self.feature_extractor(
                audio, sampling_rate=self.sample_rate, return_tensors="pt"
            )
            input_values = inputs.input_values.squeeze(0)
        else:
            input_values = torch.from_numpy(audio)

        # Tokenize text
        text_out = self._tokenize_text(pair["caption"])

        return {
            "input_values": input_values,
            "text_input_ids": text_out["input_ids"],
            "text_attention_mask": text_out["attention_mask"],
        }


class PhaseFDataset(Dataset):
    """
    Stem pairs for multi-track understanding (Phase F).

    Loads from downloaded data produced by phase_f_stems.py:
      - Multitrack songs with mix + individual stems
      - Returns mix audio + stem audio + stem label

    Stem label mapping matches StemHead.STEM_CLASSES.
    """

    STEM_LABEL_MAP = {
        "drums": 0, "bass": 1, "vocals": 2, "other": 3,
        "piano": 4, "guitar": 5, "strings": 6, "brass": 7,
        "woodwind": 8, "synth": 9, "percussion": 10, "keys": 11,
        "fx": 12, "background_vocals": 13,
    }

    def __init__(
        self,
        stem_pairs: list[dict],
        feature_extractor=None,
        sample_rate: int = 16000,
        max_duration: float = 10.0,
        augment: bool = True,
    ):
        self.stem_pairs = stem_pairs
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.augment = augment
        self.target_length = int(sample_rate * max_duration)

        logger.info(f"PhaseFDataset: {len(stem_pairs):,} stem pairs")

    @classmethod
    def from_downloaded_data(
        cls,
        data_dir: str,
        feature_extractor=None,
        **kwargs,
    ) -> "PhaseFDataset":
        """
        Build dataset from downloaded Phase F stem data.

        Expected structure:
            data_dir/
              musdb18hq/
                train/
                  song_name/
                    mixture.wav
                    drums.wav
                    bass.wav
                    vocals.wav
                    other.wav
              medleydb/
                ...
              slakh2100/
                ...
        """
        data_path = Path(data_dir)
        stem_pairs = []

        # Standard stem names across datasets
        standard_stems = {"drums", "bass", "vocals", "other", "piano", "guitar",
                          "strings", "brass", "woodwind", "synth", "percussion",
                          "keys", "fx", "background_vocals"}

        # Scan for multitrack directories
        for source_dir in sorted(data_path.iterdir()):
            if not source_dir.is_dir():
                continue

            # Scan for song directories containing mixture + stems
            for split_dir in [source_dir, source_dir / "train", source_dir / "test"]:
                if not split_dir.exists():
                    continue

                for song_dir in sorted(split_dir.iterdir()):
                    if not song_dir.is_dir():
                        continue

                    # Find mix file
                    mix_path = None
                    for mix_name in ["mixture.wav", "mix.wav", "full_mix.wav",
                                     "mixture.mp3", "mix.mp3"]:
                        candidate = song_dir / mix_name
                        if candidate.exists():
                            mix_path = str(candidate)
                            break

                    if mix_path is None:
                        continue

                    # Find stem files
                    for stem_file in sorted(song_dir.iterdir()):
                        if not stem_file.is_file():
                            continue
                        stem_name = stem_file.stem.lower()
                        if stem_name in ("mixture", "mix", "full_mix"):
                            continue
                        if stem_name in standard_stems or stem_name in cls.STEM_LABEL_MAP:
                            label = cls.STEM_LABEL_MAP.get(stem_name, 3)  # default "other"
                            stem_pairs.append({
                                "mix_path": mix_path,
                                "stem_path": str(stem_file),
                                "stem_name": stem_name,
                                "stem_label": label,
                                "source": source_dir.name,
                            })

        logger.info(f"PhaseFDataset.from_downloaded_data: {len(stem_pairs):,} stem pairs from {data_dir}")
        return cls(stem_pairs, feature_extractor=feature_extractor, **kwargs)

    def __len__(self) -> int:
        return len(self.stem_pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.stem_pairs[idx]

        # Load mix audio
        try:
            mix_audio = load_audio(pair["mix_path"], sr=self.sample_rate, max_duration=self.max_duration)
        except Exception:
            mix_audio = np.zeros(self.target_length, dtype=np.float32)

        # Load stem audio
        try:
            stem_audio = load_audio(pair["stem_path"], sr=self.sample_rate, max_duration=self.max_duration)
        except Exception:
            stem_audio = np.zeros(self.target_length, dtype=np.float32)

        # Extract features
        if self.feature_extractor is not None:
            mix_inputs = self.feature_extractor(
                mix_audio, sampling_rate=self.sample_rate, return_tensors="pt"
            )
            stem_inputs = self.feature_extractor(
                stem_audio, sampling_rate=self.sample_rate, return_tensors="pt"
            )
            mix_values = mix_inputs.input_values.squeeze(0)
            stem_values = stem_inputs.input_values.squeeze(0)
        else:
            mix_values = torch.from_numpy(mix_audio)
            stem_values = torch.from_numpy(stem_audio)

        return {
            "input_values": mix_values,
            "stem_input_values": stem_values,
            "stem_label": torch.tensor(pair["stem_label"], dtype=torch.long),
        }


class PhaseGDataset(Dataset):
    """
    Annotated songs with structure labels (Phase G).

    Loads from downloaded data produced by phase_g_structure.py:
      - Audio files with section annotations (SALAMI, Harmonix, etc.)
      - Returns audio segments + section labels + beat labels

    Each sample is a segment extracted from a full song at an annotated position.
    """

    SECTION_LABEL_MAP = {
        "intro": 0, "verse": 1, "pre-chorus": 2, "chorus": 3,
        "post-chorus": 4, "bridge": 5, "breakdown": 6, "drop": 7,
        "build": 8, "outro": 9, "instrumental": 10, "solo": 11,
        "hook": 12, "silence": 13,
    }

    def __init__(
        self,
        segments: list[dict],
        feature_extractor=None,
        sample_rate: int = 16000,
        segment_duration: float = 5.0,
        augment: bool = True,
    ):
        self.segments = segments
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.augment = augment
        self.target_length = int(sample_rate * segment_duration)

        logger.info(f"PhaseGDataset: {len(segments):,} annotated segments")

    @classmethod
    def from_downloaded_data(
        cls,
        data_dir: str,
        feature_extractor=None,
        **kwargs,
    ) -> "PhaseGDataset":
        """
        Build dataset from downloaded Phase G structure data.

        Expected structure:
            data_dir/
              salami/
                annotations/
                  {track_id}/
                    textfile1_uppercase.txt  (section labels)
                audio/
                  {track_id}.wav
              harmonix/
                annotations/
                  {track_id}.json  (beats + sections)
                audio/
        """
        data_path = Path(data_dir)
        segments = []

        for source_dir in sorted(data_path.iterdir()):
            if not source_dir.is_dir():
                continue

            annotations_dir = source_dir / "annotations"
            audio_dir = source_dir / "audio"

            if not annotations_dir.exists():
                continue

            # Scan annotation files
            for ann_file in sorted(annotations_dir.rglob("*.json")):
                try:
                    with open(ann_file) as f:
                        ann_data = json.load(f)

                    track_id = ann_file.stem
                    audio_path = None
                    for ext in [".wav", ".mp3", ".flac"]:
                        candidate = audio_dir / f"{track_id}{ext}"
                        if candidate.exists():
                            audio_path = str(candidate)
                            break

                    if audio_path is None:
                        continue

                    # Extract annotated sections
                    section_list = ann_data.get("sections", ann_data.get("segments", []))
                    beats_list = ann_data.get("beats", [])

                    for section in section_list:
                        label_str = section.get("label", "other").lower()
                        # Normalize common label variants
                        label_str = label_str.replace("_", "-").strip()
                        label_id = cls.SECTION_LABEL_MAP.get(label_str, 10)  # default "instrumental"

                        start_time = float(section.get("start", section.get("start_time", 0)))

                        # Check for beat/transition near this section
                        has_beat = False
                        is_transition = False
                        for beat in beats_list:
                            beat_time = float(beat.get("time", beat.get("t", 0)))
                            if abs(beat_time - start_time) < 0.5:
                                has_beat = True
                                break

                        segments.append({
                            "audio_path": audio_path,
                            "start_time": start_time,
                            "section_label": label_id,
                            "has_beat": has_beat,
                            "is_transition": is_transition,
                            "source": source_dir.name,
                        })

                except Exception as e:
                    logger.debug(f"Failed to load annotation {ann_file}: {e}")

            # Also support simple text annotations (SALAMI format)
            for txt_file in sorted(annotations_dir.rglob("*.txt")):
                try:
                    track_id = txt_file.parent.name if txt_file.parent != annotations_dir else txt_file.stem
                    audio_path = None
                    for ext in [".wav", ".mp3", ".flac"]:
                        candidate = audio_dir / f"{track_id}{ext}"
                        if candidate.exists():
                            audio_path = str(candidate)
                            break
                    if audio_path is None:
                        continue

                    with open(txt_file) as f:
                        for line in f:
                            parts = line.strip().split("\t")
                            if len(parts) >= 2:
                                try:
                                    start_time = float(parts[0])
                                    label_str = parts[1].lower().strip()
                                    label_id = cls.SECTION_LABEL_MAP.get(label_str, 10)
                                    segments.append({
                                        "audio_path": audio_path,
                                        "start_time": start_time,
                                        "section_label": label_id,
                                        "has_beat": False,
                                        "is_transition": False,
                                        "source": source_dir.name,
                                    })
                                except ValueError:
                                    continue
                except Exception:
                    continue

        logger.info(f"PhaseGDataset.from_downloaded_data: {len(segments):,} segments from {data_dir}")
        return cls(segments, feature_extractor=feature_extractor, **kwargs)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seg = self.segments[idx]

        # Load audio segment at the annotated position
        try:
            import librosa
            audio, _ = librosa.load(
                seg["audio_path"],
                sr=self.sample_rate,
                mono=True,
                offset=seg["start_time"],
                duration=self.segment_duration,
            )
            # Pad if short
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode="constant")
            else:
                audio = audio[:self.target_length]
        except Exception:
            audio = np.zeros(self.target_length, dtype=np.float32)

        if self.augment:
            audio = augment_audio(audio, sr=self.sample_rate)

        # Extract features
        if self.feature_extractor is not None:
            inputs = self.feature_extractor(
                audio, sampling_rate=self.sample_rate, return_tensors="pt"
            )
            input_values = inputs.input_values.squeeze(0)
        else:
            input_values = torch.from_numpy(audio)

        # Labels
        section_label = torch.tensor(seg["section_label"], dtype=torch.long)
        beat_labels = torch.tensor(
            [1.0 if seg["has_beat"] else 0.0, 0.0],  # [beat_prob, downbeat_prob]
            dtype=torch.float32,
        )
        transition_label = torch.tensor(
            [1.0 if seg["is_transition"] else 0.0],
            dtype=torch.float32,
        )

        return {
            "input_values": input_values,
            "section_label": section_label,
            "beat_labels": beat_labels,
            "transition_label": transition_label,
        }


class PhaseHDataset(Dataset):
    """
    Knowledge graph triplets for entity embedding (Phase H).

    Loads from downloaded data produced by phase_h_knowledge_graph.py:
      - Knowledge graph stored in SQLite DB
      - Audio files for entities that have associated sounds
      - Returns (head_audio, relation_type, tail_audio) triplets

    Negative sampling: for each positive triplet (h, r, t), corrupt the tail
    by replacing t with a random entity to create (h, r, t').
    """

    RELATION_TYPES = {
        "genre_of": 0, "influenced_by": 1, "performed_by": 2,
        "produced_by": 3, "similar_to": 4, "subgenre_of": 5,
    }

    def __init__(
        self,
        triplets: list[dict],
        entity_dim: int = 256,
        feature_extractor=None,
        sample_rate: int = 16000,
        max_duration: float = 10.0,
    ):
        self.triplets = triplets
        self.entity_dim = entity_dim
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.target_length = int(sample_rate * max_duration)

        # Build audio path pool for negative sampling
        self._audio_paths = list({t["head_audio"] for t in triplets} | {t["tail_audio"] for t in triplets})

        logger.info(f"PhaseHDataset: {len(triplets):,} triplets, {len(self._audio_paths)} unique audio files")

    @classmethod
    def from_downloaded_data(
        cls,
        data_dir: str,
        feature_extractor=None,
        **kwargs,
    ) -> "PhaseHDataset":
        """
        Build dataset from downloaded Phase H knowledge graph data.

        Expected structure:
            data_dir/
              knowledge_graph.db  (SQLite with entities + relations)
              audio/              (audio files for entity-linked tracks)
              triplets.jsonl      (pre-computed triplets with audio paths)
        """
        data_path = Path(data_dir)
        triplets = []

        # Try pre-computed triplets first
        triplets_file = data_path / "triplets.jsonl"
        if triplets_file.exists():
            with open(triplets_file) as f:
                for line in f:
                    t = json.loads(line.strip())
                    head_audio = t.get("head_audio", "")
                    tail_audio = t.get("tail_audio", "")
                    relation = t.get("relation", "similar_to")
                    if not os.path.isabs(head_audio):
                        head_audio = str(data_path / head_audio)
                    if not os.path.isabs(tail_audio):
                        tail_audio = str(data_path / tail_audio)
                    if os.path.exists(head_audio) and os.path.exists(tail_audio):
                        triplets.append({
                            "head_audio": head_audio,
                            "tail_audio": tail_audio,
                            "relation": relation,
                            "relation_id": cls.RELATION_TYPES.get(relation, 4),
                        })

        # Fallback: build triplets from SQLite knowledge graph
        if not triplets:
            kg_db = data_path / "knowledge_graph.db"
            if kg_db.exists():
                import sqlite3
                conn = sqlite3.connect(str(kg_db))
                conn.row_factory = sqlite3.Row
                try:
                    rows = conn.execute("""
                        SELECT r.source_id, r.target_id, r.relation_type,
                               e1.audio_path as head_audio, e2.audio_path as tail_audio
                        FROM relations r
                        JOIN entities e1 ON r.source_id = e1.entity_id
                        JOIN entities e2 ON r.target_id = e2.entity_id
                        WHERE e1.audio_path IS NOT NULL AND e2.audio_path IS NOT NULL
                    """).fetchall()
                    for row in rows:
                        head_audio = row["head_audio"]
                        tail_audio = row["tail_audio"]
                        if not os.path.isabs(head_audio):
                            head_audio = str(data_path / head_audio)
                        if not os.path.isabs(tail_audio):
                            tail_audio = str(data_path / tail_audio)
                        if os.path.exists(head_audio) and os.path.exists(tail_audio):
                            relation = row["relation_type"]
                            triplets.append({
                                "head_audio": head_audio,
                                "tail_audio": tail_audio,
                                "relation": relation,
                                "relation_id": cls.RELATION_TYPES.get(relation, 4),
                            })
                except Exception as e:
                    logger.warning(f"Failed to query knowledge graph DB: {e}")
                finally:
                    conn.close()

        logger.info(f"PhaseHDataset.from_downloaded_data: {len(triplets):,} triplets from {data_dir}")
        return cls(triplets, feature_extractor=feature_extractor, **kwargs)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        triplet = self.triplets[idx]

        # Load head audio
        try:
            head_audio = load_audio(triplet["head_audio"], sr=self.sample_rate, max_duration=self.max_duration)
        except Exception:
            head_audio = np.zeros(self.target_length, dtype=np.float32)

        # Load tail audio (positive)
        try:
            tail_audio = load_audio(triplet["tail_audio"], sr=self.sample_rate, max_duration=self.max_duration)
        except Exception:
            tail_audio = np.zeros(self.target_length, dtype=np.float32)

        # Negative sample: random audio from pool
        neg_path = random.choice(self._audio_paths)
        while neg_path == triplet["tail_audio"] and len(self._audio_paths) > 1:
            neg_path = random.choice(self._audio_paths)
        try:
            neg_audio = load_audio(neg_path, sr=self.sample_rate, max_duration=self.max_duration)
        except Exception:
            neg_audio = np.zeros(self.target_length, dtype=np.float32)

        # Extract features
        if self.feature_extractor is not None:
            head_feat = self.feature_extractor(head_audio, sampling_rate=self.sample_rate, return_tensors="pt")
            tail_feat = self.feature_extractor(tail_audio, sampling_rate=self.sample_rate, return_tensors="pt")
            neg_feat = self.feature_extractor(neg_audio, sampling_rate=self.sample_rate, return_tensors="pt")
            head_values = head_feat.input_values.squeeze(0)
            tail_values = tail_feat.input_values.squeeze(0)
            neg_values = neg_feat.input_values.squeeze(0)
        else:
            head_values = torch.from_numpy(head_audio)
            tail_values = torch.from_numpy(tail_audio)
            neg_values = torch.from_numpy(neg_audio)

        # Relation embedding (one-hot encoded, projected by the model)
        relation_id = triplet["relation_id"]

        return {
            "input_values": head_values,
            "tail_input_values": tail_values,
            "neg_tail_input_values": neg_values,
            "relation_id": torch.tensor(relation_id, dtype=torch.long),
        }


class PhaseIDataset(Dataset):
    """
    Unlabeled audio for self-supervised pre-training (Phase I).

    Three objectives combined:
      1. Masked spectrogram modeling (predict masked patches)
      2. Audio contrastive (two augmented views should be similar)
      3. Next segment prediction (are these segments consecutive?)

    Loads from any directory of audio files — no labels needed.
    """

    def __init__(
        self,
        audio_paths: list[str],
        feature_extractor=None,
        sample_rate: int = 16000,
        segment_duration: float = 5.0,
        augment: bool = True,
    ):
        self.audio_paths = audio_paths
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.augment = augment
        self.target_length = int(sample_rate * segment_duration)

        logger.info(f"PhaseIDataset: {len(audio_paths):,} audio files")

    @classmethod
    def from_downloaded_data(
        cls,
        data_dir: str,
        feature_extractor=None,
        **kwargs,
    ) -> "PhaseIDataset":
        """
        Build dataset from any directory containing audio files.
        Recursively scans for .wav, .mp3, .flac files.
        """
        data_path = Path(data_dir)
        audio_paths = []

        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg"]:
            for audio_file in data_path.rglob(ext):
                audio_paths.append(str(audio_file))

        logger.info(f"PhaseIDataset.from_downloaded_data: {len(audio_paths):,} audio files from {data_dir}")
        return cls(audio_paths, feature_extractor=feature_extractor, **kwargs)

    def __len__(self) -> int:
        return len(self.audio_paths)

    def _load_segment(self, filepath: str, offset: float = 0.0) -> np.ndarray:
        """Load a segment from an audio file."""
        try:
            import librosa
            audio, _ = librosa.load(
                filepath, sr=self.sample_rate, mono=True,
                offset=offset, duration=self.segment_duration,
            )
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode="constant")
            else:
                audio = audio[:self.target_length]
            return audio
        except Exception:
            return np.zeros(self.target_length, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        filepath = self.audio_paths[idx]

        # Get audio duration for segment sampling
        try:
            import librosa
            duration = librosa.get_duration(path=filepath)
        except Exception:
            duration = self.segment_duration * 3

        # Segment A: random position
        max_offset = max(0, duration - self.segment_duration * 2)
        offset_a = random.uniform(0, max_offset) if max_offset > 0 else 0

        audio_a = self._load_segment(filepath, offset=offset_a)

        # View 2: augmented version of segment A (for audio contrastive)
        audio_view2 = augment_audio(audio_a.copy(), sr=self.sample_rate)

        # Segment B: either consecutive (50%) or random (50%)
        is_consecutive = random.random() < 0.5
        if is_consecutive:
            offset_b = offset_a + self.segment_duration
            if offset_b + self.segment_duration > duration:
                offset_b = max(0, offset_a - self.segment_duration)
            audio_b = self._load_segment(filepath, offset=offset_b)
        else:
            # Random segment from a different file
            random_idx = random.randint(0, len(self.audio_paths) - 1)
            random_offset = random.uniform(0, max(0, duration - self.segment_duration))
            audio_b = self._load_segment(self.audio_paths[random_idx], offset=random_offset)

        # Extract features
        if self.feature_extractor is not None:
            feat_a = self.feature_extractor(audio_a, sampling_rate=self.sample_rate, return_tensors="pt")
            feat_v2 = self.feature_extractor(audio_view2, sampling_rate=self.sample_rate, return_tensors="pt")
            feat_b = self.feature_extractor(audio_b, sampling_rate=self.sample_rate, return_tensors="pt")
            values_a = feat_a.input_values.squeeze(0)
            values_v2 = feat_v2.input_values.squeeze(0)
            values_b = feat_b.input_values.squeeze(0)
        else:
            values_a = torch.from_numpy(audio_a)
            values_v2 = torch.from_numpy(audio_view2)
            values_b = torch.from_numpy(audio_b)

        return {
            "input_values": values_a,
            "input_values_view2": values_v2,
            "segment_b_input_values": values_b,
            "consecutive_labels": torch.tensor(1.0 if is_consecutive else 0.0, dtype=torch.float32),
        }


# ======================================================================
# C) AdvancedTrainingConfig
# ======================================================================

@dataclass
class AdvancedTrainingConfig(TrainingConfig):
    """
    Training configuration for Phases E-I.
    Extends TrainingConfig with phase-specific hyperparameters.
    H100 80GB defaults: aggressive batch sizes, BF16 mixed precision.
    """

    # Phase E: Contrastive text-audio alignment
    phase_e_epochs: int = 15
    phase_e_lr: float = 3e-4
    phase_e_batch_size: int = 128       # H100 can handle large contrastive batches
    phase_e_gradient_accumulation: int = 2  # effective batch = 256
    phase_e_warmup_steps: int = 1000
    phase_e_use_bf16: bool = True

    # Phase F: Stem understanding
    phase_f_epochs: int = 10
    phase_f_lr: float = 2e-4
    phase_f_batch_size: int = 48        # Two audio streams per sample (mix + stem)
    phase_f_gradient_accumulation: int = 2  # effective = 96
    phase_f_warmup_steps: int = 500
    phase_f_use_bf16: bool = True

    # Phase G: Structure detection
    phase_g_epochs: int = 10
    phase_g_lr: float = 2e-4
    phase_g_batch_size: int = 64
    phase_g_gradient_accumulation: int = 2  # effective = 128
    phase_g_warmup_steps: int = 500
    phase_g_use_bf16: bool = True

    # Phase H: Knowledge graph embedding
    phase_h_epochs: int = 8
    phase_h_lr: float = 1e-4
    phase_h_batch_size: int = 32        # Three audio streams per triplet
    phase_h_gradient_accumulation: int = 4  # effective = 128
    phase_h_warmup_steps: int = 500
    phase_h_use_bf16: bool = True

    # Phase I: Self-supervised
    phase_i_epochs: int = 20
    phase_i_lr: float = 1e-4
    phase_i_batch_size: int = 48        # Three audio segments per sample
    phase_i_gradient_accumulation: int = 4  # effective = 192
    phase_i_warmup_steps: int = 2000
    phase_i_use_bf16: bool = True

    # Data directories for Phase E-I
    phase_e_data_dir: str = ""
    phase_f_data_dir: str = ""
    phase_g_data_dir: str = ""
    phase_h_data_dir: str = ""
    phase_i_data_dir: str = ""

    # Number of dataloader workers
    advanced_num_workers: int = 8


# ======================================================================
# D) AdvancedTrainer
# ======================================================================

class AdvancedTrainer:
    """
    Training pipeline for Phases E-I.

    Each phase follows the same pattern:
      1. Build phase-specific dataset + dataloader
      2. Configure optimizer with appropriate LR
      3. Train with BF16 mixed precision + gradient accumulation
      4. Checkpoint after each epoch
      5. Log with phase labels

    Mirrors the structure of RPMTrainer for consistency.
    """

    def __init__(self, cfg: AdvancedTrainingConfig = None):
        self.cfg = cfg or AdvancedTrainingConfig()
        self.device = resolve_device(self.cfg.device)

        # Ensure directories exist
        for d in [self.cfg.output_dir, self.cfg.checkpoint_dir, self.cfg.log_dir]:
            Path(d).expanduser().mkdir(parents=True, exist_ok=True)

        self.metrics = TrainingMetrics(self.cfg.log_dir)
        self.global_step = 0

        logger.info(f"AdvancedTrainer initialized — device: {self.device}")

    def _build_optimizer(
        self,
        model: AdvancedRPMModel,
        lr: float,
        phase_params_only: bool = True,
        phase_modules: list[nn.Module] = None,
    ) -> AdamW:
        """
        Build optimizer for a specific phase's parameters.

        Args:
            model: the AdvancedRPMModel
            lr: learning rate
            phase_params_only: if True, only optimize phase-specific heads (not base model)
            phase_modules: list of modules whose params should be optimized
        """
        if phase_modules is not None:
            params = []
            for module in phase_modules:
                params.extend([p for p in module.parameters() if p.requires_grad])
        elif phase_params_only:
            # Optimize everything except base_model
            base_params = set(id(p) for p in model.base_model.parameters())
            params = [p for p in model.parameters() if p.requires_grad and id(p) not in base_params]
        else:
            params = [p for p in model.parameters() if p.requires_grad]

        if not params:
            logger.warning("No trainable parameters found! Check model freeze state.")
            # Return optimizer with a dummy param to avoid crash
            dummy = torch.nn.Parameter(torch.zeros(1, device=self.device))
            return AdamW([dummy], lr=lr, weight_decay=self.cfg.weight_decay)

        return AdamW(params, lr=lr, weight_decay=self.cfg.weight_decay)

    def _build_scheduler(self, optimizer, num_training_steps: int, warmup_steps: int = 500):
        """Build LR scheduler with warmup + cosine decay."""
        warmup = min(warmup_steps, num_training_steps // 5)

        warmup_sched = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup,
        )
        cosine_sched = CosineAnnealingWarmRestarts(
            optimizer, T_0=max(num_training_steps - warmup, 1), T_mult=1,
        )
        return SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])

    def _build_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Build a DataLoader for an advanced phase dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.advanced_num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def save_checkpoint(self, model: AdvancedRPMModel, optimizer, phase: str, epoch: int):
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
        torch.save(state, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

    def _get_scaler(self, use_bf16: bool):
        """Get AMP scaler. BF16 on H100 doesn't need loss scaling."""
        if use_bf16 and self.device == "cuda":
            # BF16 has enough dynamic range — no scaler needed
            # But we use autocast with dtype=bfloat16
            return None
        return None

    def _get_autocast_dtype(self, use_bf16: bool):
        """Get autocast dtype."""
        if use_bf16 and self.device == "cuda":
            return torch.bfloat16
        return torch.float32

    # ──────────────────────────────────────────────────────────────
    # Phase E: Contrastive Text-Audio Alignment
    # ──────────────────────────────────────────────────────────────

    def train_phase_e(self, model: AdvancedRPMModel, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Phase E: CLIP-style contrastive learning between text and audio.
        The most important advanced phase — enables natural language understanding
        of audio content for recommendation.
        """
        logger.info("=" * 70)
        logger.info("PHASE E: Contrastive Text-Audio Alignment")
        logger.info("  The model learns to understand WHY things sound the way they do.")
        logger.info("=" * 70)

        cfg = self.cfg
        model.freeze_base_model()

        # Optimize: text encoder + contrastive loss projections
        phase_modules = [model.text_encoder, model.contrastive_loss]
        optimizer = self._build_optimizer(model, lr=cfg.phase_e_lr, phase_modules=phase_modules)

        num_steps = (len(train_loader) * cfg.phase_e_epochs) // cfg.phase_e_gradient_accumulation
        scheduler = self._build_scheduler(optimizer, num_steps, cfg.phase_e_warmup_steps)
        autocast_dtype = self._get_autocast_dtype(cfg.phase_e_use_bf16)
        accum_steps = cfg.phase_e_gradient_accumulation

        for epoch in range(cfg.phase_e_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            t0 = time.time()

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                input_values = batch["input_values"].to(self.device)
                text_ids = batch["text_input_ids"].to(self.device)
                text_mask = batch["text_attention_mask"].to(self.device)

                with autocast(device_type="cuda", dtype=autocast_dtype) if self.device == "cuda" else torch.no_grad.__class__():
                    outputs = model(
                        input_values,
                        active_phases="E",
                        text_input_ids=text_ids,
                        text_attention_mask=text_mask,
                    )
                    loss = model.contrastive_loss(
                        outputs["audio_embedding"],
                        outputs["text_embedding"],
                    ) / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += loss.item() * accum_steps
                num_batches += 1

                if self.global_step > 0 and self.global_step % cfg.log_every_n_steps == 0:
                    avg = epoch_loss / num_batches
                    self.metrics.log("E", epoch, self.global_step, {"total_loss": avg, "contrastive": avg})

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(
                f"Phase E Epoch {epoch+1}/{cfg.phase_e_epochs} — "
                f"loss={avg_loss:.4f} — {elapsed:.1f}s"
            )

            # Validation
            if val_loader is not None:
                val_loss = self._eval_phase_e(model, val_loader, autocast_dtype)
                logger.info(f"  Val loss: {val_loss:.4f}")
                if self.metrics.check_improvement(val_loss):
                    logger.info("Early stopping triggered.")
                    break

            self.save_checkpoint(model, optimizer, "E", epoch)

        self.metrics.save()
        logger.info("Phase E complete.\n")

    @torch.no_grad()
    def _eval_phase_e(self, model, val_loader, autocast_dtype) -> float:
        model.eval()
        total_loss = 0.0
        n = 0
        for batch in val_loader:
            input_values = batch["input_values"].to(self.device)
            text_ids = batch["text_input_ids"].to(self.device)
            text_mask = batch["text_attention_mask"].to(self.device)

            with autocast(device_type="cuda", dtype=autocast_dtype) if self.device == "cuda" else torch.no_grad.__class__():
                outputs = model(input_values, active_phases="E",
                                text_input_ids=text_ids, text_attention_mask=text_mask)
                loss = model.contrastive_loss(outputs["audio_embedding"], outputs["text_embedding"])

            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    # ──────────────────────────────────────────────────────────────
    # Phase F: Stem Understanding
    # ──────────────────────────────────────────────────────────────

    def train_phase_f(self, model: AdvancedRPMModel, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Phase F: Teach the model how individual stems combine into a mix.
        Stem classification + mix contribution + compatibility.
        """
        logger.info("=" * 70)
        logger.info("PHASE F: Stem Understanding")
        logger.info("  The model learns how music is constructed from stems.")
        logger.info("=" * 70)

        cfg = self.cfg
        model.freeze_base_model()

        phase_modules = [model.stem_head]
        optimizer = self._build_optimizer(model, lr=cfg.phase_f_lr, phase_modules=phase_modules)

        num_steps = (len(train_loader) * cfg.phase_f_epochs) // cfg.phase_f_gradient_accumulation
        scheduler = self._build_scheduler(optimizer, num_steps, cfg.phase_f_warmup_steps)
        autocast_dtype = self._get_autocast_dtype(cfg.phase_f_use_bf16)
        accum_steps = cfg.phase_f_gradient_accumulation

        for epoch in range(cfg.phase_f_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            t0 = time.time()

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                input_values = batch["input_values"].to(self.device)
                stem_values = batch["stem_input_values"].to(self.device)
                stem_labels = batch["stem_label"].to(self.device)

                with autocast(device_type="cuda", dtype=autocast_dtype) if self.device == "cuda" else torch.no_grad.__class__():
                    outputs = model(
                        input_values,
                        active_phases="F",
                        stem_input_values=stem_values,
                    )
                    loss = model.stem_loss(
                        outputs["stem_logits"],
                        stem_labels,
                    ) / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += loss.item() * accum_steps
                num_batches += 1

                if self.global_step > 0 and self.global_step % cfg.log_every_n_steps == 0:
                    avg = epoch_loss / num_batches
                    self.metrics.log("F", epoch, self.global_step, {"total_loss": avg, "stem_cls": avg})

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(
                f"Phase F Epoch {epoch+1}/{cfg.phase_f_epochs} — "
                f"loss={avg_loss:.4f} — {elapsed:.1f}s"
            )

            self.save_checkpoint(model, optimizer, "F", epoch)

        self.metrics.save()
        logger.info("Phase F complete.\n")

    # ──────────────────────────────────────────────────────────────
    # Phase G: Structure Detection
    # ──────────────────────────────────────────────────────────────

    def train_phase_g(self, model: AdvancedRPMModel, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Phase G: Song structure prediction — sections, beats, transitions.
        """
        logger.info("=" * 70)
        logger.info("PHASE G: Song Structure Detection")
        logger.info("  The model learns song architecture — verse, chorus, drops.")
        logger.info("=" * 70)

        cfg = self.cfg
        model.freeze_base_model()

        phase_modules = [model.structure_head]
        optimizer = self._build_optimizer(model, lr=cfg.phase_g_lr, phase_modules=phase_modules)

        num_steps = (len(train_loader) * cfg.phase_g_epochs) // cfg.phase_g_gradient_accumulation
        scheduler = self._build_scheduler(optimizer, num_steps, cfg.phase_g_warmup_steps)
        autocast_dtype = self._get_autocast_dtype(cfg.phase_g_use_bf16)
        accum_steps = cfg.phase_g_gradient_accumulation

        for epoch in range(cfg.phase_g_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            t0 = time.time()

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                input_values = batch["input_values"].to(self.device)
                section_labels = batch["section_label"].to(self.device)
                beat_labels = batch.get("beat_labels")
                transition_labels = batch.get("transition_label")

                with autocast(device_type="cuda", dtype=autocast_dtype) if self.device == "cuda" else torch.no_grad.__class__():
                    outputs = model(input_values, active_phases="G")

                    # Compute structure loss
                    beat_probs = outputs.get("beat_probs")
                    transition_prob = outputs.get("transition_prob")

                    if beat_labels is not None:
                        beat_labels = beat_labels.to(self.device)
                    if transition_labels is not None:
                        transition_labels = transition_labels.to(self.device)

                    loss = model.structure_loss(
                        outputs["section_logits"],
                        section_labels,
                        beat_probs=beat_probs,
                        beat_labels=beat_labels,
                        transition_prob=transition_prob,
                        transition_label=transition_labels,
                    ) / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += loss.item() * accum_steps
                num_batches += 1

                if self.global_step > 0 and self.global_step % cfg.log_every_n_steps == 0:
                    avg = epoch_loss / num_batches
                    self.metrics.log("G", epoch, self.global_step, {"total_loss": avg, "structure": avg})

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(
                f"Phase G Epoch {epoch+1}/{cfg.phase_g_epochs} — "
                f"loss={avg_loss:.4f} — {elapsed:.1f}s"
            )

            self.save_checkpoint(model, optimizer, "G", epoch)

        self.metrics.save()
        logger.info("Phase G complete.\n")

    # ──────────────────────────────────────────────────────────────
    # Phase H: Knowledge Graph Embedding
    # ──────────────────────────────────────────────────────────────

    def train_phase_h(self, model: AdvancedRPMModel, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Phase H: TransE-style knowledge graph embedding.
        Aligns audio embeddings with entity embeddings in a shared space.
        """
        logger.info("=" * 70)
        logger.info("PHASE H: Knowledge Graph Embedding")
        logger.info("  The model learns genre lineage, artist style, production history.")
        logger.info("=" * 70)

        cfg = self.cfg
        model.freeze_base_model()

        phase_modules = [model.knowledge_head]
        optimizer = self._build_optimizer(model, lr=cfg.phase_h_lr, phase_modules=phase_modules)

        num_steps = (len(train_loader) * cfg.phase_h_epochs) // cfg.phase_h_gradient_accumulation
        scheduler = self._build_scheduler(optimizer, num_steps, cfg.phase_h_warmup_steps)
        autocast_dtype = self._get_autocast_dtype(cfg.phase_h_use_bf16)
        accum_steps = cfg.phase_h_gradient_accumulation

        # Learnable relation embeddings
        num_relations = len(PhaseHDataset.RELATION_TYPES)
        entity_dim = model.knowledge_head.entity_dim
        relation_embeddings = nn.Embedding(num_relations, entity_dim).to(self.device)
        relation_optimizer = AdamW(relation_embeddings.parameters(), lr=cfg.phase_h_lr)

        for epoch in range(cfg.phase_h_epochs):
            model.train()
            relation_embeddings.train()
            epoch_loss = 0.0
            num_batches = 0
            t0 = time.time()

            optimizer.zero_grad()
            relation_optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                input_values = batch["input_values"].to(self.device)
                tail_values = batch["tail_input_values"].to(self.device)
                neg_tail_values = batch["neg_tail_input_values"].to(self.device)
                relation_ids = batch["relation_id"].to(self.device)

                # Look up relation vectors
                rel_vectors = relation_embeddings(relation_ids)

                with autocast(device_type="cuda", dtype=autocast_dtype) if self.device == "cuda" else torch.no_grad.__class__():
                    outputs = model(
                        input_values,
                        active_phases="H",
                        relation_embeddings=rel_vectors,
                        tail_input_values=tail_values,
                        neg_tail_input_values=neg_tail_values,
                    )

                    loss = model.knowledge_loss(
                        outputs["kg_pos_scores"],
                        outputs["kg_neg_scores"],
                    ) / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(relation_embeddings.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    relation_optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    relation_optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += loss.item() * accum_steps
                num_batches += 1

                if self.global_step > 0 and self.global_step % cfg.log_every_n_steps == 0:
                    avg = epoch_loss / num_batches
                    self.metrics.log("H", epoch, self.global_step, {"total_loss": avg, "knowledge": avg})

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(
                f"Phase H Epoch {epoch+1}/{cfg.phase_h_epochs} — "
                f"loss={avg_loss:.4f} — {elapsed:.1f}s"
            )

            self.save_checkpoint(model, optimizer, "H", epoch)

        # Save relation embeddings separately
        rel_path = Path(self.cfg.checkpoint_dir).expanduser() / "relation_embeddings.pt"
        torch.save(relation_embeddings.state_dict(), rel_path)
        logger.info(f"Relation embeddings saved: {rel_path}")

        self.metrics.save()
        logger.info("Phase H complete.\n")

    # ──────────────────────────────────────────────────────────────
    # Phase I: Self-Supervised Pre-Training
    # ──────────────────────────────────────────────────────────────

    def train_phase_i(self, model: AdvancedRPMModel, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Phase I: Self-supervised learning on massive unlabeled audio.
        Combines: masked spectrogram modeling, audio contrastive, next segment prediction.
        """
        logger.info("=" * 70)
        logger.info("PHASE I: Self-Supervised Pre-Training")
        logger.info("  The model learns deep audio structure from pure sound.")
        logger.info("=" * 70)

        cfg = self.cfg

        # Phase I can optionally unfreeze backbone for deeper learning
        model.freeze_base_model()

        phase_modules = [model.masked_spec, model.audio_contrastive, model.next_segment]
        optimizer = self._build_optimizer(model, lr=cfg.phase_i_lr, phase_modules=phase_modules)

        num_steps = (len(train_loader) * cfg.phase_i_epochs) // cfg.phase_i_gradient_accumulation
        scheduler = self._build_scheduler(optimizer, num_steps, cfg.phase_i_warmup_steps)
        autocast_dtype = self._get_autocast_dtype(cfg.phase_i_use_bf16)
        accum_steps = cfg.phase_i_gradient_accumulation

        # Loss weights for the three self-supervised objectives
        w_contrastive = 0.5
        w_nsp = 0.3

        for epoch in range(cfg.phase_i_epochs):
            model.train()
            epoch_losses = {"contrastive": 0.0, "nsp": 0.0, "total": 0.0}
            num_batches = 0
            t0 = time.time()

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                input_values = batch["input_values"].to(self.device)
                view2_values = batch["input_values_view2"].to(self.device)
                seg_b_values = batch["segment_b_input_values"].to(self.device)
                consec_labels = batch["consecutive_labels"].to(self.device)

                with autocast(device_type="cuda", dtype=autocast_dtype) if self.device == "cuda" else torch.no_grad.__class__():
                    outputs = model(
                        input_values,
                        active_phases="I",
                        input_values_view2=view2_values,
                        segment_b_input_values=seg_b_values,
                        consecutive_labels=consec_labels,
                    )

                    # Audio contrastive loss (two views)
                    loss_contrastive = model.audio_contrastive(
                        outputs["view1_embedding"],
                        outputs["view2_embedding"],
                    )

                    # Next segment prediction loss
                    loss_nsp = model.next_segment(
                        outputs["segment_a_embedding"],
                        outputs["segment_b_embedding"],
                        outputs["consecutive_labels"],
                    )

                    total_loss = (
                        w_contrastive * loss_contrastive +
                        w_nsp * loss_nsp
                    ) / accum_steps

                total_loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                epoch_losses["contrastive"] += loss_contrastive.item()
                epoch_losses["nsp"] += loss_nsp.item()
                epoch_losses["total"] += (loss_contrastive.item() * w_contrastive + loss_nsp.item() * w_nsp)
                num_batches += 1

                if self.global_step > 0 and self.global_step % cfg.log_every_n_steps == 0:
                    avgs = {k: v / num_batches for k, v in epoch_losses.items()}
                    self.metrics.log("I", epoch, self.global_step, {
                        "total_loss": avgs["total"],
                        "contrastive": avgs["contrastive"],
                        "nsp": avgs["nsp"],
                    })

            elapsed = time.time() - t0
            avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
            logger.info(
                f"Phase I Epoch {epoch+1}/{cfg.phase_i_epochs} — "
                f"loss={avg_losses['total']:.4f} "
                f"(contrastive={avg_losses['contrastive']:.4f}, nsp={avg_losses['nsp']:.4f}) — "
                f"{elapsed:.1f}s"
            )

            self.save_checkpoint(model, optimizer, "I", epoch)

        self.metrics.save()
        logger.info("Phase I complete.\n")

    # ──────────────────────────────────────────────────────────────
    # Full Advanced Pipeline
    # ──────────────────────────────────────────────────────────────

    def train_advanced(
        self,
        model: AdvancedRPMModel,
        phases: str = "EFGHI",
        feature_extractor=None,
    ):
        """
        Run the full advanced training pipeline (Phases E-I).

        Args:
            model: AdvancedRPMModel with loaded Phase A-D weights
            phases: which phases to run (e.g. "E", "EF", "EFGHI")
            feature_extractor: AST feature extractor for building datasets
        """
        logger.info("=" * 70)
        logger.info("RESONATE Production Model — Advanced Training Pipeline (E-I)")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Phases: {phases}")

        params = count_parameters(model)
        logger.info(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

        cfg = self.cfg

        # Phase E
        if "E" in phases and cfg.phase_e_data_dir:
            logger.info(f"\nBuilding Phase E dataset from {cfg.phase_e_data_dir}...")
            ds = PhaseEDataset.from_downloaded_data(
                cfg.phase_e_data_dir,
                feature_extractor=feature_extractor,
                augment=True,
            )
            if len(ds) > 0:
                # Train/val split
                n = len(ds)
                indices = list(range(n))
                random.shuffle(indices)
                split = int(n * 0.95)
                train_ds = torch.utils.data.Subset(ds, indices[:split])
                val_ds = torch.utils.data.Subset(ds, indices[split:])

                train_dl = self._build_dataloader(train_ds, cfg.phase_e_batch_size, shuffle=True)
                val_dl = self._build_dataloader(val_ds, cfg.phase_e_batch_size, shuffle=False)
                self.train_phase_e(model, train_dl, val_dl)

                torch.save(model.state_dict(), Path(cfg.checkpoint_dir).expanduser() / "rpm_phaseE_done.pt")
            else:
                logger.warning("Phase E: No data found, skipping.")

        # Phase F
        if "F" in phases and cfg.phase_f_data_dir:
            logger.info(f"\nBuilding Phase F dataset from {cfg.phase_f_data_dir}...")
            ds = PhaseFDataset.from_downloaded_data(
                cfg.phase_f_data_dir,
                feature_extractor=feature_extractor,
            )
            if len(ds) > 0:
                train_dl = self._build_dataloader(ds, cfg.phase_f_batch_size, shuffle=True)
                self.train_phase_f(model, train_dl)

                torch.save(model.state_dict(), Path(cfg.checkpoint_dir).expanduser() / "rpm_phaseF_done.pt")
            else:
                logger.warning("Phase F: No data found, skipping.")

        # Phase G
        if "G" in phases and cfg.phase_g_data_dir:
            logger.info(f"\nBuilding Phase G dataset from {cfg.phase_g_data_dir}...")
            ds = PhaseGDataset.from_downloaded_data(
                cfg.phase_g_data_dir,
                feature_extractor=feature_extractor,
            )
            if len(ds) > 0:
                train_dl = self._build_dataloader(ds, cfg.phase_g_batch_size, shuffle=True)
                self.train_phase_g(model, train_dl)

                torch.save(model.state_dict(), Path(cfg.checkpoint_dir).expanduser() / "rpm_phaseG_done.pt")
            else:
                logger.warning("Phase G: No data found, skipping.")

        # Phase H
        if "H" in phases and cfg.phase_h_data_dir:
            logger.info(f"\nBuilding Phase H dataset from {cfg.phase_h_data_dir}...")
            ds = PhaseHDataset.from_downloaded_data(
                cfg.phase_h_data_dir,
                feature_extractor=feature_extractor,
            )
            if len(ds) > 0:
                train_dl = self._build_dataloader(ds, cfg.phase_h_batch_size, shuffle=True)
                self.train_phase_h(model, train_dl)

                torch.save(model.state_dict(), Path(cfg.checkpoint_dir).expanduser() / "rpm_phaseH_done.pt")
            else:
                logger.warning("Phase H: No data found, skipping.")

        # Phase I
        if "I" in phases and cfg.phase_i_data_dir:
            logger.info(f"\nBuilding Phase I dataset from {cfg.phase_i_data_dir}...")
            ds = PhaseIDataset.from_downloaded_data(
                cfg.phase_i_data_dir,
                feature_extractor=feature_extractor,
            )
            if len(ds) > 0:
                train_dl = self._build_dataloader(ds, cfg.phase_i_batch_size, shuffle=True)
                self.train_phase_i(model, train_dl)

                torch.save(model.state_dict(), Path(cfg.checkpoint_dir).expanduser() / "rpm_phaseI_done.pt")
            else:
                logger.warning("Phase I: No data found, skipping.")

        # Save final advanced model
        final_path = Path(cfg.output_dir).expanduser() / "rpm_advanced_final.pt"
        torch.save(model.state_dict(), final_path)
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Advanced training complete! Model saved to {final_path}")
        logger.info(f"{'=' * 70}")

        return model
