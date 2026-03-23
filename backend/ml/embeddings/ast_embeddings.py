"""
AST (Audio Spectrogram Transformer) embedding extractor.
Produces 768-dim embeddings from spectrogram representations.
Good for capturing timbral and structural features.
"""
from __future__ import annotations

import numpy as np
import librosa
import torch
from transformers import ASTModel, ASTFeatureExtractor


class ASTExtractor:
    """Extract AST embeddings from audio files."""

    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                 device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def extract(self, filepath: str) -> np.ndarray:
        """
        Extract 768-dim embedding from an audio file.
        Uses the [CLS] token output from AST.
        """
        # Load at 16kHz (AST requirement)
        audio, _ = librosa.load(filepath, sr=16000, mono=True)

        # Truncate to 10 seconds max
        max_samples = 16000 * 10
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Extract features (mel spectrogram)
        inputs = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            # Use the [CLS] token (first token) as the embedding
            embedding = outputs.last_hidden_state[:, 0, :]

        emb = embedding.cpu().numpy().flatten().astype(np.float32)
        return emb
