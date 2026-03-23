"""
Orchestrates all embedding extractors. Loads models once, extracts all embeddings
for a given audio file in one call.
"""
from __future__ import annotations
import logging
from backend.ml.models.sample_profile import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages all embedding extractors with lazy loading."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._clap = None
        self._panns = None
        self._ast = None

    @property
    def clap(self):
        if self._clap is None:
            from backend.ml.embeddings.clap_embeddings import CLAPExtractor
            self._clap = CLAPExtractor(device=self.device)
            logger.info("CLAP model loaded")
        return self._clap

    @property
    def panns(self):
        if self._panns is None:
            from backend.ml.embeddings.panns_embeddings import PANNsExtractor
            self._panns = PANNsExtractor(device=self.device)
            logger.info("PANNs model loaded")
        return self._panns

    @property
    def ast(self):
        if self._ast is None:
            from backend.ml.embeddings.ast_embeddings import ASTExtractor
            self._ast = ASTExtractor(device=self.device)
            logger.info("AST model loaded")
        return self._ast

    def extract_all(self, filepath: str) -> Embeddings:
        """Extract all embeddings for a single audio file."""
        result = Embeddings()

        # CLAP
        try:
            emb = self.clap.extract(filepath)
            result.clap_general = emb.tolist()
        except Exception as e:
            logger.warning(f"CLAP extraction failed for {filepath}: {e}")

        # PANNs
        try:
            emb = self.panns.extract_embedding(filepath)
            result.panns_music = emb.tolist()
            result.panns_tags = self.panns.extract_tags(filepath)
        except Exception as e:
            logger.warning(f"PANNs extraction failed for {filepath}: {e}")

        # AST
        try:
            emb = self.ast.extract(filepath)
            result.ast_spectrogram = emb.tolist()
        except Exception as e:
            logger.warning(f"AST extraction failed for {filepath}: {e}")

        return result
