"""
CLAP (Contrastive Language-Audio Pretraining) embedding extractor.
Produces 512-dim embeddings in a joint audio-text space.
Useful for general audio similarity and text-based retrieval.
"""
from __future__ import annotations

import numpy as np
import librosa
import torch


class CLAPExtractor:
    """Extracts CLAP embeddings from audio files."""

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load the CLAP model."""
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.model.load_ckpt()  # Downloads checkpoint if needed
        self.model.eval()

    def extract(self, filepath: str) -> np.ndarray:
        """
        Extract a 512-dim L2-normalized embedding from an audio file.
        Audio is resampled to 48kHz as required by CLAP.
        """
        # Load and prepare audio
        audio, sr = librosa.load(filepath, sr=48000, mono=True)

        # Pad or truncate to 10 seconds (CLAP default)
        target_len = 48000 * 10
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))

        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()

        with torch.no_grad():
            embedding = self.model.get_audio_embedding_from_data(
                x=audio_tensor, use_tensor=True
            )

        emb = embedding.cpu().numpy().flatten()
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        return emb.astype(np.float32)

    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract embedding for a text query (for text-to-audio search)."""
        with torch.no_grad():
            embedding = self.model.get_text_embedding([text])
        emb = embedding.cpu().numpy().flatten()
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        return emb.astype(np.float32)
