import pytest
from ml.db.sample_store import SampleStore
from ml.models.sample_profile import SampleProfile, CoreDescriptors, PredictedLabels


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = SampleStore(str(db_path))
    s.init()
    return s


@pytest.fixture
def sample_profile():
    p = SampleProfile(filepath="/samples/kick.wav", filename="kick.wav", file_hash="abc123")
    p.core = CoreDescriptors(duration=0.15, sample_rate=44100, channels=1,
                              rms=0.3, lufs=-14.0, peak=0.9, crest_factor=9.5)
    p.labels = PredictedLabels(role="kick", role_confidence=0.85,
                                tonal=False, genre_affinity={"trap": 0.8})
    return p


def test_save_and_load(store, sample_profile):
    store.save(sample_profile)
    loaded = store.load("/samples/kick.wav")
    assert loaded is not None
    assert loaded.filepath == "/samples/kick.wav"
    assert loaded.core.duration == 0.15
    assert loaded.labels.role == "kick"


def test_load_nonexistent(store):
    result = store.load("/nonexistent.wav")
    assert result is None


def test_update_existing(store, sample_profile):
    store.save(sample_profile)
    sample_profile.core.rms = 0.5
    sample_profile.labels.role = "bass"
    store.save(sample_profile)
    loaded = store.load("/samples/kick.wav")
    assert loaded.core.rms == 0.5
    assert loaded.labels.role == "bass"


def test_list_all(store, sample_profile):
    store.save(sample_profile)
    p2 = SampleProfile(filepath="/samples/snare.wav", filename="snare.wav")
    store.save(p2)
    all_profiles = store.list_all()
    assert len(all_profiles) == 2


def test_count(store, sample_profile):
    assert store.count() == 0
    store.save(sample_profile)
    assert store.count() == 1


def test_delete(store, sample_profile):
    store.save(sample_profile)
    store.delete("/samples/kick.wav")
    assert store.load("/samples/kick.wav") is None


def test_search_by_role(store, sample_profile):
    store.save(sample_profile)
    results = store.search_by_role("kick")
    assert len(results) == 1
    assert results[0].filepath == "/samples/kick.wav"


def test_needs_reanalysis(store, sample_profile):
    store.save(sample_profile)
    assert not store.needs_reanalysis("/samples/kick.wav", "abc123")
    assert store.needs_reanalysis("/samples/kick.wav", "different_hash")
    assert store.needs_reanalysis("/nonexistent.wav", "any")
