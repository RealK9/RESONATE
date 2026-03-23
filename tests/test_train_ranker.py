"""Tests for backend.ml.training.train_ranker — RankerTrainer."""
import time

import pytest

from backend.ml.db.sample_store import SampleStore
from backend.ml.models.preference import PreferencePair, UserTasteModel
from backend.ml.models.sample_profile import (
    PredictedLabels,
    SampleProfile,
    SpectralDescriptors,
    TransientDescriptors,
)
from backend.ml.training.preference_dataset import PreferenceDataset
from backend.ml.training.train_ranker import RankerTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pref_db(tmp_path):
    """PreferenceDataset backed by a temporary SQLite DB."""
    ds = PreferenceDataset(str(tmp_path / "prefs.db"))
    ds.init()
    return ds


@pytest.fixture
def sample_store_db(tmp_path):
    """SampleStore with a handful of synthetic profiles."""
    store = SampleStore(str(tmp_path / "samples.db"))
    store.init()

    profiles = [
        SampleProfile(
            filepath="/samples/kick_01.wav",
            filename="kick_01.wav",
            labels=PredictedLabels(
                role="kick", role_confidence=0.9, commercial_readiness=0.8
            ),
            spectral=SpectralDescriptors(centroid=120.0),
            transients=TransientDescriptors(onset_rate=4.0),
        ),
        SampleProfile(
            filepath="/samples/snare_01.wav",
            filename="snare_01.wav",
            labels=PredictedLabels(
                role="snare", role_confidence=0.85, commercial_readiness=0.7
            ),
            spectral=SpectralDescriptors(centroid=3500.0),
            transients=TransientDescriptors(onset_rate=3.0),
        ),
        SampleProfile(
            filepath="/samples/pad_01.wav",
            filename="pad_01.wav",
            labels=PredictedLabels(
                role="pad", role_confidence=0.75, commercial_readiness=0.6
            ),
            spectral=SpectralDescriptors(centroid=800.0),
            transients=TransientDescriptors(onset_rate=0.5),
        ),
        SampleProfile(
            filepath="/samples/hat_01.wav",
            filename="hat_01.wav",
            labels=PredictedLabels(
                role="hat", role_confidence=0.88, commercial_readiness=0.5
            ),
            spectral=SpectralDescriptors(centroid=9000.0),
            transients=TransientDescriptors(onset_rate=8.0),
        ),
    ]
    for p in profiles:
        store.save(p)
    return store


def _make_pairs(n: int, context_style: str = "trap") -> list[PreferencePair]:
    """Generate *n* synthetic preference pairs."""
    pairs = []
    now = time.time()
    for i in range(n):
        pairs.append(
            PreferencePair(
                preferred_filepath="/samples/kick_01.wav",
                rejected_filepath="/samples/snare_01.wav",
                mix_filepath="/mixes/mix.wav",
                context_style=context_style,
                strength=0.8,
                timestamp=now + i,
            )
        )
    return pairs


def _populate(pref_db: PreferenceDataset, n: int = 15, style: str = "trap"):
    """Insert *n* preference pairs directly into the DB."""
    pairs = _make_pairs(n, context_style=style)
    for p in pairs:
        with pref_db._connect() as conn:
            conn.execute(
                """INSERT INTO preference_pairs
                   (preferred_filepath, rejected_filepath, mix_filepath,
                    context_style, strength, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    p.preferred_filepath,
                    p.rejected_filepath,
                    p.mix_filepath,
                    p.context_style,
                    p.strength,
                    p.timestamp,
                ),
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_returns_none_insufficient_data(pref_db):
    """Fewer than min_pairs returns None."""
    _populate(pref_db, n=3)
    trainer = RankerTrainer(pref_db)
    result = trainer.train(min_pairs=10)
    assert result is None


def test_train_produces_taste_model(pref_db, sample_store_db):
    """Sufficient pairs produce a valid UserTasteModel."""
    _populate(pref_db, n=15)
    trainer = RankerTrainer(pref_db, sample_store=sample_store_db)
    model = trainer.train(min_pairs=10)
    assert model is not None
    assert isinstance(model, UserTasteModel)
    assert model.user_id == "default"


def test_role_bias_computed(pref_db, sample_store_db):
    """role_bias is non-empty when a sample_store is provided."""
    _populate(pref_db, n=15)
    trainer = RankerTrainer(pref_db, sample_store=sample_store_db)
    model = trainer.train(min_pairs=10)
    assert model is not None
    assert len(model.role_bias) > 0
    # Preferred role ("kick") should have positive bias.
    assert model.role_bias.get("kick", 0) > 0


def test_style_bias_computed(pref_db):
    """style_bias reflects context styles present in the pairs."""
    _populate(pref_db, n=15, style="lofi_hiphop")
    trainer = RankerTrainer(pref_db)
    model = trainer.train(min_pairs=10)
    assert model is not None
    assert "lofi_hiphop" in model.style_bias
    assert model.style_bias["lofi_hiphop"] > 0


def test_weight_deltas_bounded(pref_db, sample_store_db):
    """All weight deltas must be between -0.05 and +0.05."""
    _populate(pref_db, n=20)
    trainer = RankerTrainer(pref_db, sample_store=sample_store_db)
    model = trainer.train(min_pairs=10)
    assert model is not None
    for key, delta in model.weight_deltas.items():
        assert -0.05 <= delta <= 0.05, (
            f"Delta for {key} out of range: {delta}"
        )


def test_model_version_increments(pref_db, sample_store_db):
    """Training twice increments the model version."""
    _populate(pref_db, n=15)
    trainer = RankerTrainer(pref_db, sample_store=sample_store_db)
    m1 = trainer.train(min_pairs=10)
    assert m1 is not None
    assert m1.model_version == 1

    m2 = trainer.train(min_pairs=10)
    assert m2 is not None
    assert m2.model_version == 2


def test_quality_threshold_reasonable(pref_db, sample_store_db):
    """quality_threshold should be between 0 and 1."""
    _populate(pref_db, n=15)
    trainer = RankerTrainer(pref_db, sample_store=sample_store_db)
    model = trainer.train(min_pairs=10)
    assert model is not None
    assert 0.0 <= model.quality_threshold <= 1.0


def test_training_pairs_count(pref_db, sample_store_db):
    """model.training_pairs matches the number of input pairs."""
    n = 18
    _populate(pref_db, n=n)
    trainer = RankerTrainer(pref_db, sample_store=sample_store_db)
    model = trainer.train(min_pairs=10)
    assert model is not None
    assert model.training_pairs == n


def test_train_without_store(pref_db):
    """Training without a SampleStore still produces style bias."""
    _populate(pref_db, n=15, style="house")
    trainer = RankerTrainer(pref_db, sample_store=None)
    model = trainer.train(min_pairs=10)
    assert model is not None
    # Role bias empty without a store.
    assert model.role_bias == {}
    # Style bias should still be populated.
    assert "house" in model.style_bias


def test_normalize_biases(pref_db, sample_store_db):
    """All bias values must be in [-1, 1]."""
    _populate(pref_db, n=20)
    trainer = RankerTrainer(pref_db, sample_store=sample_store_db)
    model = trainer.train(min_pairs=10)
    assert model is not None

    for role, val in model.role_bias.items():
        assert -1.0 <= val <= 1.0, f"role_bias[{role}] = {val} out of range"
    for style, val in model.style_bias.items():
        assert -1.0 <= val <= 1.0, f"style_bias[{style}] = {val} out of range"
