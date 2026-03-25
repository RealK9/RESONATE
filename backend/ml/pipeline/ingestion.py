"""
Single-sample analysis pipeline. Orchestrates all extractors into a complete SampleProfile.
"""
from __future__ import annotations
import logging
from pathlib import Path
from ml.models.sample_profile import SampleProfile
from ml.analysis.core_descriptors import extract_core_descriptors
from ml.analysis.spectral_descriptors import extract_spectral_descriptors
from ml.analysis.harmonic_descriptors import extract_harmonic_descriptors
from ml.analysis.transient_descriptors import extract_transient_descriptors
from ml.analysis.perceptual_descriptors import extract_perceptual_descriptors
from ml.analysis.loop_detection import detect_loop
from ml.classifiers.role_classifier import RoleClassifier
from ml.classifiers.genre_era_classifier import GenreEraClassifier
from ml.classifiers.style_tagger import StyleTagger
from ml.classifiers.quality_scorer import QualityScorer

logger = logging.getLogger(__name__)

# Singletons for classifiers (stateless, safe to share)
_role_clf = RoleClassifier()
_genre_clf = GenreEraClassifier()
_style_tagger = StyleTagger()
_quality_scorer = QualityScorer()


def analyze_sample(filepath: str, skip_embeddings: bool = False,
                   embedding_manager=None, file_hash: str = "",
                   source: str = "local") -> SampleProfile:
    """
    Run the complete analysis pipeline on a single audio file.

    Args:
        filepath: Path to the audio file.
        skip_embeddings: If True, skip ML embedding extraction (faster).
        embedding_manager: Optional EmbeddingManager instance (reuse across calls).
        file_hash: Pre-computed file hash for cache invalidation.
        source: Sample source label (local/splice/loopcloud).

    Returns:
        Complete SampleProfile with all descriptors populated.
    """
    path = Path(filepath)
    profile = SampleProfile(
        filepath=str(path),
        filename=path.name,
        file_hash=file_hash,
        source=source,
    )

    # Stage 1: DSP features (all local, no ML)
    try:
        profile.core = extract_core_descriptors(filepath)
    except Exception as e:
        logger.error(f"Core extraction failed for {filepath}: {e}")

    try:
        profile.spectral = extract_spectral_descriptors(filepath)
    except Exception as e:
        logger.error(f"Spectral extraction failed for {filepath}: {e}")

    try:
        profile.harmonic = extract_harmonic_descriptors(filepath)
    except Exception as e:
        logger.error(f"Harmonic extraction failed for {filepath}: {e}")

    try:
        profile.transients = extract_transient_descriptors(filepath)
    except Exception as e:
        logger.error(f"Transient extraction failed for {filepath}: {e}")

    try:
        profile.perceptual = extract_perceptual_descriptors(filepath)
    except Exception as e:
        logger.error(f"Perceptual extraction failed for {filepath}: {e}")

    # Stage 2: Loop detection
    try:
        is_loop, loop_conf = detect_loop(filepath)
        profile.labels.is_loop = is_loop
        profile.labels.loop_confidence = loop_conf
    except Exception as e:
        logger.error(f"Loop detection failed for {filepath}: {e}")

    # Stage 3: Embeddings (optional, requires ML models)
    if not skip_embeddings and embedding_manager is not None:
        try:
            profile.embeddings = embedding_manager.extract_all(filepath)
        except Exception as e:
            logger.error(f"Embedding extraction failed for {filepath}: {e}")

    # Stage 4: Classification
    try:
        panns_tags = profile.embeddings.panns_tags if profile.embeddings.panns_tags else None
        role, role_conf = _role_clf.classify(filepath, filename_hint=path.name,
                                              panns_tags=panns_tags)
        profile.labels.role = role
        profile.labels.role_confidence = role_conf
    except Exception as e:
        logger.error(f"Role classification failed for {filepath}: {e}")

    try:
        profile.labels.tonal = profile.harmonic.tonalness > 0.5
    except Exception:
        pass

    try:
        profile.labels.genre_affinity = _genre_clf.classify_genre(filepath)
        profile.labels.era_affinity = _genre_clf.classify_era(filepath)
    except Exception as e:
        logger.error(f"Genre/era classification failed for {filepath}: {e}")

    try:
        profile.labels.style_tags = _style_tagger.tag(filepath)
    except Exception as e:
        logger.error(f"Style tagging failed for {filepath}: {e}")

    try:
        profile.labels.commercial_readiness = _quality_scorer.score(filepath)
    except Exception as e:
        logger.error(f"Quality scoring failed for {filepath}: {e}")

    return profile
