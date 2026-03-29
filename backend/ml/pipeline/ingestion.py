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

# Lazy singletons for classifiers (instantiated on first use, not at import time)
_role_clf = None
_genre_clf = None
_style_tagger = None
_quality_scorer = None


def _get_role_clf() -> RoleClassifier:
    global _role_clf
    if _role_clf is None:
        _role_clf = RoleClassifier()
    return _role_clf


def _get_genre_clf() -> GenreEraClassifier:
    global _genre_clf
    if _genre_clf is None:
        _genre_clf = GenreEraClassifier()
    return _genre_clf


def _get_style_tagger() -> StyleTagger:
    global _style_tagger
    if _style_tagger is None:
        _style_tagger = StyleTagger()
    return _style_tagger


def _get_quality_scorer() -> QualityScorer:
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = QualityScorer()
    return _quality_scorer


def analyze_sample(filepath: str, skip_embeddings: bool = False,
                   embedding_manager=None, rpm_extractor=None,
                   file_hash: str = "",
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

    # Stage 3: Embeddings + Classification
    # If RPM extractor is available, use it (replaces CLAP+PANNs+AST+classifiers)
    if not skip_embeddings and rpm_extractor is not None:
        try:
            rpm_result = rpm_extractor.extract(filepath)

            # Store the unified 768-d RPM embedding
            profile.embeddings.rpm = rpm_result.embedding.tolist()

            # RPM replaces all classifiers with one forward pass
            profile.labels.role = rpm_result.role
            profile.labels.role_confidence = rpm_result.role_confidence
            profile.labels.rpm_genre_top = rpm_result.genre_top
            profile.labels.rpm_genre_sub = rpm_result.genre_sub
            profile.labels.rpm_instruments = rpm_result.instruments
            profile.labels.rpm_key = rpm_result.key
            profile.labels.rpm_chord_quality = rpm_result.chord_quality
            profile.labels.rpm_mode = rpm_result.mode
            profile.labels.rpm_era = rpm_result.era
            profile.labels.rpm_chart_potential = rpm_result.chart_potential

            # Map RPM outputs to legacy fields for backward compatibility
            profile.labels.genre_affinity = rpm_result.genre_distribution
            profile.labels.era_affinity = rpm_result.era_distribution
            profile.labels.commercial_readiness = rpm_result.chart_potential

            # RPM perceptual descriptors override DSP-based ones
            if rpm_result.perceptual:
                for key, val in rpm_result.perceptual.items():
                    if hasattr(profile.perceptual, key):
                        setattr(profile.perceptual, key, val)

            profile.labels.tonal = profile.harmonic.tonalness > 0.5

            logger.info(f"RPM analysis: {rpm_result.role} ({rpm_result.role_confidence:.0%}), "
                       f"genre={rpm_result.genre_top}, key={rpm_result.key}, "
                       f"era={rpm_result.era}, chart={rpm_result.chart_potential:.2f}")

        except Exception as e:
            logger.error(f"RPM extraction failed for {filepath}: {e}")
            # Fall back to legacy pipeline
            _run_legacy_analysis(profile, filepath, path, embedding_manager)
    elif not skip_embeddings:
        _run_legacy_analysis(profile, filepath, path, embedding_manager)

    return profile


def _run_legacy_analysis(profile: SampleProfile, filepath: str, path: Path,
                         embedding_manager=None):
    """Legacy analysis pipeline using CLAP + PANNs + AST + separate classifiers."""
    # Stage 3: Embeddings (optional, requires ML models)
    if embedding_manager is not None:
        try:
            profile.embeddings = embedding_manager.extract_all(filepath)
        except Exception as e:
            logger.error(f"Embedding extraction failed for {filepath}: {e}")

    # Stage 4: Classification
    try:
        panns_tags = profile.embeddings.panns_tags if profile.embeddings.panns_tags else None
        role, role_conf = _get_role_clf().classify(filepath, filename_hint=path.name,
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
        profile.labels.genre_affinity = _get_genre_clf().classify_genre(filepath)
        profile.labels.era_affinity = _get_genre_clf().classify_era(filepath)
    except Exception as e:
        logger.error(f"Genre/era classification failed for {filepath}: {e}")

    try:
        profile.labels.style_tags = _get_style_tagger().tag(filepath)
    except Exception as e:
        logger.error(f"Style tagging failed for {filepath}: {e}")

    try:
        profile.labels.commercial_readiness = _get_quality_scorer().score(filepath)
    except Exception as e:
        logger.error(f"Quality scoring failed for {filepath}: {e}")
