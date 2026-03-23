import numpy as np
import pytest
from backend.ml.embeddings.panns_embeddings import PANNsExtractor


@pytest.fixture(scope="session")
def panns():
    """Load PANNs model once for all tests in this session."""
    try:
        return PANNsExtractor()
    except Exception:
        pytest.skip("PANNs model not available")


def test_embedding_shape(panns, sine_440hz):
    emb = panns.extract_embedding(str(sine_440hz))
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert len(emb) == 2048  # PANNs Cnn14 embedding dim


def test_tags_returned(panns, kick_like):
    tags = panns.extract_tags(str(kick_like))
    assert isinstance(tags, dict)
    assert len(tags) > 0
    # All confidences should be 0-1
    for tag, conf in tags.items():
        assert 0.0 <= conf <= 1.0


def test_embedding_deterministic(panns, sine_440hz):
    emb1 = panns.extract_embedding(str(sine_440hz))
    emb2 = panns.extract_embedding(str(sine_440hz))
    np.testing.assert_array_almost_equal(emb1, emb2, decimal=4)
