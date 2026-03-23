import numpy as np
import pytest
from backend.ml.embeddings.ast_embeddings import ASTExtractor


@pytest.fixture(scope="session")
def ast_model():
    try:
        return ASTExtractor()
    except Exception:
        pytest.skip("AST model not available")


def test_embedding_shape(ast_model, sine_440hz):
    emb = ast_model.extract(str(sine_440hz))
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert len(emb) == 768  # AST hidden dim


def test_different_sounds_differ(ast_model, sine_440hz, kick_like):
    emb1 = ast_model.extract(str(sine_440hz))
    emb2 = ast_model.extract(str(kick_like))
    cosine = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    assert cosine < 0.99


def test_deterministic(ast_model, sine_440hz):
    emb1 = ast_model.extract(str(sine_440hz))
    emb2 = ast_model.extract(str(sine_440hz))
    np.testing.assert_array_almost_equal(emb1, emb2, decimal=4)
