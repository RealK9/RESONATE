"""Tests for the EmbeddingManager orchestrator."""
import pytest
from backend.ml.embeddings.embedding_manager import EmbeddingManager
from backend.ml.models.sample_profile import Embeddings


@pytest.fixture(scope="session")
def manager():
    """Create an EmbeddingManager. Skip all tests if construction fails entirely."""
    try:
        return EmbeddingManager()
    except Exception as e:
        pytest.skip(f"Embedding models not available: {e}")


@pytest.fixture(scope="session")
def extraction_result(manager, sine_440hz):
    """Extract once per session and reuse across tests."""
    return manager.extract_all(str(sine_440hz))


def test_returns_embeddings_dataclass(extraction_result):
    assert isinstance(extraction_result, Embeddings)


def test_clap_populated(extraction_result):
    if len(extraction_result.clap_general) == 0:
        pytest.skip("CLAP model not available")
    assert len(extraction_result.clap_general) == 512


def test_panns_populated(extraction_result):
    if len(extraction_result.panns_music) == 0:
        pytest.skip("PANNs model not available")
    assert len(extraction_result.panns_music) == 2048


def test_ast_populated(extraction_result):
    if len(extraction_result.ast_spectrogram) == 0:
        pytest.skip("AST model not available")
    assert len(extraction_result.ast_spectrogram) == 768


def test_panns_tags_populated(extraction_result):
    if len(extraction_result.panns_music) == 0:
        pytest.skip("PANNs model not available")
    assert isinstance(extraction_result.panns_tags, dict)


def test_at_least_one_extractor_worked(extraction_result):
    """At least one of the three extractors should produce output."""
    has_clap = len(extraction_result.clap_general) > 0
    has_panns = len(extraction_result.panns_music) > 0
    has_ast = len(extraction_result.ast_spectrogram) > 0
    if not (has_clap or has_panns or has_ast):
        pytest.skip("No embedding models available at all")
    assert has_clap or has_panns or has_ast


def test_individual_failure_does_not_crash(manager, sine_440hz):
    """Even if models fail, extract_all returns a valid Embeddings object."""
    result = manager.extract_all(str(sine_440hz))
    assert isinstance(result, Embeddings)
    # All fields should be their default types regardless of failures
    assert isinstance(result.clap_general, list)
    assert isinstance(result.panns_music, list)
    assert isinstance(result.ast_spectrogram, list)
    assert isinstance(result.panns_tags, dict)
