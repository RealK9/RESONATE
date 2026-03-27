"""Tests for the taste profile route (Producer DNA endpoints)."""
import time
from unittest.mock import MagicMock, patch

import pytest

from ml.models.preference import FeedbackEvent, PreferencePair, UserTasteModel
from ml.training.preference_dataset import PreferenceDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ds(tmp_path):
    """Return an initialised PreferenceDataset backed by a temp SQLite file."""
    db = str(tmp_path / "test_prefs.db")
    dataset = PreferenceDataset(db)
    dataset.init()
    return dataset


@pytest.fixture
def populated_ds(ds):
    """Dataset with feedback events and a trained model."""
    # Log enough feedback events to build pairs
    for i in range(6):
        ds.log_feedback(FeedbackEvent(
            sample_filepath=f"/samples/sample_{i}.wav",
            mix_filepath="/mixes/track.wav",
            session_id="s1",
            action="drag" if i < 3 else "skip",
            timestamp=1000.0 + i,
        ))

    # Save a taste model
    model = UserTasteModel(
        user_id="default",
        role_bias={"kick": 0.8, "snare_clap": 0.4, "bass": -0.2, "lead": 0.1},
        style_bias={"trap": 0.9, "house": -0.3, "dnb": 0.5},
        weight_deltas={"eta": 0.02, "gamma": -0.01},
        quality_threshold=0.65,
        training_pairs=15,
        model_version=2,
        last_trained=time.time(),
    )
    ds.save_taste_model(model)
    return ds


# ---------------------------------------------------------------------------
# GET /taste/profile
# ---------------------------------------------------------------------------

class TestGetProfile:
    def test_profile_with_model(self, populated_ds):
        """Profile endpoint should return structured data when a model exists."""
        from routes.taste_profile import _get_dataset, get_taste_profile

        with patch("routes.taste_profile._get_dataset", return_value=populated_ds):
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(get_taste_profile("default"))

        assert result["status"] == "ok"
        assert result["user_id"] == "default"
        assert result["model_version"] == 2
        assert result["training_pairs"] == 15

        # role_affinities should be a sorted list of dicts
        roles = result["role_affinities"]
        assert isinstance(roles, list)
        assert all("role" in r and "affinity" in r for r in roles)
        # Sorted by absolute affinity descending
        abs_vals = [abs(r["affinity"]) for r in roles]
        assert abs_vals == sorted(abs_vals, reverse=True)

        # style_preferences should be a sorted list of dicts
        styles = result["style_preferences"]
        assert isinstance(styles, list)
        assert all("style" in s and "preference" in s for s in styles)

        # quality_threshold
        assert 0 <= result["quality_threshold"] <= 1

        # weight_profile
        assert "eta" in result["weight_profile"]

        # action_breakdown should reflect logged events
        breakdown = result["action_breakdown"]
        assert isinstance(breakdown, dict)
        total = result["total_interactions"]
        assert total == sum(breakdown.values())

    def test_profile_no_model(self, ds):
        """Profile endpoint should return no_data when no model exists."""
        from routes.taste_profile import get_taste_profile

        with patch("routes.taste_profile._get_dataset", return_value=ds):
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(get_taste_profile("default"))

        assert result["status"] == "no_data"
        assert "message" in result

    def test_action_breakdown_counts(self, populated_ds):
        """Action breakdown should correctly count each action type."""
        from routes.taste_profile import _action_breakdown

        breakdown = _action_breakdown(populated_ds.db_path)
        assert breakdown.get("drag", 0) == 3
        assert breakdown.get("skip", 0) == 3


# ---------------------------------------------------------------------------
# POST /taste/train
# ---------------------------------------------------------------------------

class TestTrainTaste:
    def test_train_with_sufficient_pairs(self, populated_ds):
        """Training should succeed when enough pairs exist."""
        # First build pairs so get_training_data returns them
        populated_ds.build_pairs()

        from routes.taste_profile import train_taste

        with patch("routes.taste_profile._get_dataset", return_value=populated_ds):
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(train_taste("default"))

        assert result["status"] == "ok"
        assert result["training_pairs"] > 0
        assert result["model_version"] >= 1

    def test_train_insufficient_pairs(self, ds):
        """Training should return insufficient_data when not enough pairs."""
        from routes.taste_profile import train_taste

        with patch("routes.taste_profile._get_dataset", return_value=ds):
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(train_taste("default"))

        assert result["status"] == "insufficient_data"
        assert result["training_pairs"] == 0
        assert result["model_version"] == 0
