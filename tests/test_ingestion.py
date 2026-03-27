import pytest
from ml.pipeline.ingestion import analyze_sample
from ml.models.sample_profile import SampleProfile


def test_analyze_sample_returns_profile(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert isinstance(profile, SampleProfile)


def test_core_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert profile.core.duration > 0
    assert profile.core.sample_rate > 0


def test_spectral_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert profile.spectral.centroid > 0


def test_harmonic_populated(sine_440hz):
    profile = analyze_sample(str(sine_440hz), skip_embeddings=True)
    assert profile.harmonic.f0 > 0


def test_transients_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert profile.transients.onset_count >= 1


def test_perceptual_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert profile.perceptual.punch > 0


def test_labels_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    assert profile.labels.role != "unknown" or profile.labels.role_confidence == 0


def test_loop_detection_populated(kick_like):
    profile = analyze_sample(str(kick_like), skip_embeddings=True)
    # One-shot kick should not be a loop
    assert profile.labels.is_loop is False
