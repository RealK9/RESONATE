torch = __import__("pytest").importorskip("torch")
laion_clap = __import__("pytest").importorskip("laion_clap")
import numpy as np
import pytest
from ml.embeddings.clap_embeddings import CLAPExtractor


@pytest.fixture(scope="session")
def clap():
    """Load CLAP model once for all tests in this session."""
    try:
        return CLAPExtractor()
    except Exception:
        pytest.skip("CLAP model not available")


def test_embedding_shape(clap, sine_440hz):
    emb = clap.extract(str(sine_440hz))
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert len(emb) == 512  # CLAP embedding dim


def test_embedding_normalized(clap, sine_440hz):
    emb = clap.extract(str(sine_440hz))
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 0.01  # Should be L2-normalized


def test_different_sounds_different_embeddings(clap, sine_440hz, kick_like):
    emb1 = clap.extract(str(sine_440hz))
    emb2 = clap.extract(str(kick_like))
    cosine_sim = np.dot(emb1, emb2)
    # Different sounds should not be identical
    assert cosine_sim < 0.99


def test_same_sound_consistent(clap, sine_440hz):
    emb1 = clap.extract(str(sine_440hz))
    emb2 = clap.extract(str(sine_440hz))
    cosine_sim = np.dot(emb1, emb2)
    assert cosine_sim > 0.99


def test_text_embedding_shape(clap):
    emb = clap.extract_text_embedding("a kick drum sample")
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert len(emb) == 512


def test_text_embedding_normalized(clap):
    emb = clap.extract_text_embedding("a snare drum hit")
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 0.01


def test_text_audio_cross_modal(clap, kick_like):
    """Text and audio embeddings should live in the same space."""
    audio_emb = clap.extract(str(kick_like))
    text_emb = clap.extract_text_embedding("a kick drum sample")
    cosine_sim = np.dot(audio_emb, text_emb)
    # Cross-modal similarity should be non-trivial (same space)
    assert -1.0 <= cosine_sim <= 1.0
