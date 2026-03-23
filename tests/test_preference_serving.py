"""Tests for backend.ml.training.preference_serving — PreferenceServer."""
import pytest

from backend.ml.models.preference import UserTasteModel
from backend.ml.training.preference_dataset import PreferenceDataset
from backend.ml.training.preference_serving import PreferenceServer


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
def taste_model() -> UserTasteModel:
    """A pre-built taste model for testing."""
    return UserTasteModel(
        user_id="default",
        role_bias={"kick": 0.8, "snare": -0.4, "pad": 0.2},
        style_bias={"trap": 0.9, "lofi_hiphop": -0.3},
        weight_deltas={"eta": 0.03, "gamma": -0.02},
        quality_threshold=0.65,
        training_pairs=50,
        model_version=3,
        last_trained=1000000.0,
    )


@pytest.fixture
def server_with_model(pref_db, taste_model) -> PreferenceServer:
    """A PreferenceServer with a model already loaded."""
    pref_db.save_taste_model(taste_model)
    server = PreferenceServer(pref_db)
    server.load("default")
    return server


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_returns_false_no_model(pref_db):
    """load() returns False when no model exists for the user."""
    server = PreferenceServer(pref_db)
    assert server.load("nonexistent_user") is False


def test_load_returns_true_with_model(pref_db, taste_model):
    """load() returns True after a model has been saved."""
    pref_db.save_taste_model(taste_model)
    server = PreferenceServer(pref_db)
    assert server.load("default") is True


def test_score_default_no_model(pref_db):
    """score() returns 0.5 when no model is loaded."""
    server = PreferenceServer(pref_db)
    result = server.score("/samples/kick.wav", "kick", "trap")
    assert result == 0.5


def test_score_with_role_bias(server_with_model):
    """Positive role bias increases score above 0.5."""
    # "kick" has role_bias = 0.8 → rescaled to 0.9
    # No matching style → only role contributes
    score = server_with_model.score(
        "/samples/kick.wav", "kick", "unknown_style"
    )
    assert score > 0.5


def test_score_with_negative_bias(server_with_model):
    """Negative role bias decreases score below 0.5."""
    # "snare" has role_bias = -0.4 → rescaled to 0.3
    score = server_with_model.score(
        "/samples/snare.wav", "snare", "unknown_style"
    )
    assert score < 0.5


def test_score_with_style_bias(server_with_model):
    """Style bias affects score when the context style is known."""
    # "trap" has style_bias = 0.9 → rescaled to 0.95
    # Use an unknown role so only style contributes.
    score = server_with_model.score(
        "/samples/unknown.wav", "unknown_role", "trap"
    )
    assert score > 0.5


def test_get_weight_adjustments_empty(pref_db):
    """get_weight_adjustments() returns empty dict when no model is loaded."""
    server = PreferenceServer(pref_db)
    assert server.get_weight_adjustments() == {}


def test_get_weight_adjustments_with_model(server_with_model):
    """get_weight_adjustments() returns weight deltas from the model."""
    adjustments = server_with_model.get_weight_adjustments()
    assert "eta" in adjustments
    assert adjustments["eta"] == pytest.approx(0.03)
    assert "gamma" in adjustments
    assert adjustments["gamma"] == pytest.approx(-0.02)


def test_is_loaded_property(pref_db, taste_model):
    """is_loaded property reflects whether a model has been loaded."""
    server = PreferenceServer(pref_db)
    assert server.is_loaded is False

    pref_db.save_taste_model(taste_model)
    server.load("default")
    assert server.is_loaded is True


def test_model_version_property(pref_db, taste_model):
    """model_version matches the version of the loaded model."""
    server = PreferenceServer(pref_db)
    assert server.model_version == 0

    pref_db.save_taste_model(taste_model)
    server.load("default")
    assert server.model_version == taste_model.model_version
