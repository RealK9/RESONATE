"""
Style priors trainer -- learns production norms from reference audio files.

Wraps ReferenceProfileBuilder and the full v2 analysis pipeline to provide
a high-level API:  add reference files (or entire directories), train a
ReferenceCorpus, and persist / reload it from disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ml.analysis.reference_profiles import (
    DefaultPriors,
    ReferenceProfileBuilder,
)
from ml.models.reference_profile import ReferenceCorpus


# Supported audio extensions (same set as backend/config.py)
_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}


class StylePriorsTrainer:
    """Trains style priors from reference audio files.

    Usage::

        trainer = StylePriorsTrainer(corpus_path="corpus.json")
        trainer.add_reference_file("ref_track.wav", genre="modern_trap")
        trainer.add_reference_directory("references/melodic_house/")
        corpus = trainer.train()          # builds + optionally saves
        corpus = trainer.load_or_default()  # reload or use defaults
    """

    def __init__(self, corpus_path: str | None = None) -> None:
        self._builder = ReferenceProfileBuilder()
        self._corpus_path = corpus_path

    # ------------------------------------------------------------------
    # Adding references
    # ------------------------------------------------------------------

    def add_reference_file(
        self,
        filepath: str,
        genre: str | None = None,
        era: str | None = None,
    ) -> None:
        """Analyze a reference file and add to the corpus.

        Parameters
        ----------
        filepath:
            Path to an audio file.
        genre:
            If given, override the style classifier and file under this
            cluster name directly.
        era:
            Optional era hint (not currently used by the builder, but
            reserved for future weighting).
        """
        # Import lazily so the module can be loaded even when heavy
        # audio dependencies (librosa, etc.) are not installed.
        from ml.analysis.mix_analyzer import analyze_mix
        from ml.analysis.style_classifier import StyleClassifier

        mix_profile = analyze_mix(filepath)

        # Classify style if no explicit genre override
        if not genre:
            StyleClassifier().classify(mix_profile)

        self._builder.add_reference(mix_profile, cluster_override=genre)

    def add_reference_directory(
        self,
        directory: str,
        genre: str | None = None,
    ) -> None:
        """Add all audio files in *directory* as references.

        Walks the directory recursively and adds every file whose
        extension is in the supported audio set.

        Parameters
        ----------
        directory:
            Path to a directory of reference tracks.
        genre:
            If given, all files in this directory are filed under this
            cluster name (useful for genre-labeled folders).
        """
        root = Path(directory)
        if not root.is_dir():
            return

        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() in _AUDIO_EXT:
                    self.add_reference_file(str(fpath), genre=genre)

    # ------------------------------------------------------------------
    # Train / persist / load
    # ------------------------------------------------------------------

    def train(self) -> ReferenceCorpus:
        """Build the corpus from all added references.

        If *corpus_path* was set at init time the corpus is also
        serialized to disk as JSON so it can be reloaded later.
        """
        corpus = self._builder.build_corpus()

        if self._corpus_path:
            path = Path(self._corpus_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(corpus.to_dict(), indent=2))

        return corpus

    def load_or_default(self) -> ReferenceCorpus:
        """Load a previously trained corpus from disk.

        Falls back to ``DefaultPriors.get_corpus()`` when no saved
        corpus exists (or no *corpus_path* was configured).
        """
        if self._corpus_path:
            p = Path(self._corpus_path)
            if p.exists():
                data = json.loads(p.read_text())
                return ReferenceCorpus.from_dict(data)

        return DefaultPriors.get_corpus()
