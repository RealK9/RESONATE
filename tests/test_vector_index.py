import numpy as np
import pytest
from backend.ml.retrieval.vector_index import VectorIndex


@pytest.fixture
def index():
    return VectorIndex(dim=512)


@pytest.fixture
def populated_index():
    idx = VectorIndex(dim=8)
    rng = np.random.default_rng(42)
    for i in range(100):
        vec = rng.standard_normal(8).astype(np.float32)
        vec /= np.linalg.norm(vec)
        idx.add(f"sample_{i}.wav", vec)
    return idx


def test_empty_index(index):
    assert index.size() == 0


def test_add_and_size(index):
    rng = np.random.default_rng(99)
    vec = rng.standard_normal(512).astype(np.float32)
    index.add("test.wav", vec)
    assert index.size() == 1


def test_search_returns_results(populated_index):
    rng = np.random.default_rng(7)
    query = rng.standard_normal(8).astype(np.float32)
    query /= np.linalg.norm(query)
    results = populated_index.search(query, k=5)
    assert len(results) == 5


def test_search_result_format(populated_index):
    rng = np.random.default_rng(11)
    query = rng.standard_normal(8).astype(np.float32)
    results = populated_index.search(query, k=3)
    for filepath, score in results:
        assert isinstance(filepath, str)
        assert isinstance(score, float)


def test_search_self_is_top_result(populated_index):
    # The vector for sample_0 should be most similar to itself
    vec = populated_index.get_vector("sample_0.wav")
    assert vec is not None
    results = populated_index.search(vec, k=1)
    assert results[0][0] == "sample_0.wav"


def test_save_and_load(populated_index, tmp_path):
    save_path = tmp_path / "test_index"
    populated_index.save(str(save_path))
    loaded = VectorIndex.load(str(save_path))
    assert loaded.size() == 100
    # Search should work on loaded index
    rng = np.random.default_rng(13)
    query = rng.standard_normal(8).astype(np.float32)
    results = loaded.search(query, k=5)
    assert len(results) == 5


def test_remove(populated_index):
    populated_index.remove("sample_0.wav")
    assert populated_index.size() == 99
    assert populated_index.get_vector("sample_0.wav") is None
