"""
PANNs (Pretrained Audio Neural Networks) embedding and tagging extractor.
Uses Cnn14 for robust audio tagging and 2048-dim embeddings.
"""
from __future__ import annotations

import numpy as np
import librosa


class PANNsExtractor:
    """Extract PANNs embeddings and audio tags."""

    def __init__(self, device: str | None = None):
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load PANNs Cnn14 model."""
        from panns_inference import AudioTagging

        self.model = AudioTagging(checkpoint_path=None, device=self.device)

    def extract_embedding(self, filepath: str) -> np.ndarray:
        """Extract 2048-dim embedding from an audio file."""
        audio = self._load_audio(filepath)
        _, embedding = self.model.inference(audio[np.newaxis, :])
        return embedding.flatten().astype(np.float32)

    def extract_tags(self, filepath: str, top_k: int = 20) -> dict[str, float]:
        """Extract AudioSet tags with confidence scores."""
        audio = self._load_audio(filepath)
        clipwise_output, _ = self.model.inference(audio[np.newaxis, :])
        probs = clipwise_output.flatten()

        label_list = self._get_labels(len(probs))

        # Top-k tags above threshold
        top_indices = np.argsort(probs)[::-1][:top_k]
        return {
            label_list[i]: round(float(probs[i]), 4)
            for i in top_indices
            if probs[i] > 0.01
        }

    def _get_labels(self, num_classes: int) -> list[str]:
        """Get AudioSet label names, trying multiple approaches for compatibility."""
        # Approach 1: panns_inference.config.labels (some versions)
        try:
            from panns_inference.config import labels

            if isinstance(labels, list) and len(labels) == num_classes:
                return labels
        except (ImportError, AttributeError):
            pass

        # Approach 2: load from bundled CSV via panns_inference package path
        try:
            import panns_inference
            from pathlib import Path
            import csv

            pkg_dir = Path(panns_inference.__file__).parent
            csv_path = pkg_dir / "metadata" / "class_labels_indices.csv"
            if not csv_path.exists():
                # Some versions put it at different locations
                csv_path = pkg_dir / "class_labels_indices.csv"
            if csv_path.exists():
                label_list = []
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label_list.append(row.get("display_name", row.get("mid", "")))
                if len(label_list) == num_classes:
                    return label_list
        except Exception:
            pass

        # Approach 3: try panns_inference.labels directly
        try:
            import panns_inference

            if hasattr(panns_inference, "labels"):
                labels_attr = panns_inference.labels
                if hasattr(labels_attr, "labels"):
                    label_list = labels_attr.labels
                elif isinstance(labels_attr, list):
                    label_list = labels_attr
                else:
                    label_list = None
                if label_list and len(label_list) == num_classes:
                    return label_list
        except Exception:
            pass

        # Fallback: generic class names
        return [f"class_{i}" for i in range(num_classes)]

    def extract_embedding_and_tags(self, filepath: str, top_k: int = 20
                                   ) -> tuple[np.ndarray, dict[str, float]]:
        """Extract both embedding and tags in one pass (single audio load)."""
        audio = self._load_audio(filepath)
        clipwise_output, embedding = self.model.inference(audio[np.newaxis, :])

        # Embedding
        emb = embedding.flatten().astype(np.float32)

        # Tags
        probs = clipwise_output.flatten()
        label_list = self._get_labels(len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]
        tags = {
            label_list[i]: round(float(probs[i]), 4)
            for i in top_indices
            if probs[i] > 0.01
        }

        return emb, tags

    def _load_audio(self, filepath: str) -> np.ndarray:
        """Load audio at 32kHz mono (PANNs requirement)."""
        audio, _ = librosa.load(filepath, sr=32000, mono=True)
        return audio.astype(np.float32)
