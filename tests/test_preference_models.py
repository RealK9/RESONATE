"""Tests for Phase 5 preference data models."""
import json

import numpy as np
import pytest

from ml.models.preference import (
    FeedbackEvent,
    PreferencePair,
    UserTasteModel,
)


class TestFeedbackEvent:
    def test_feedback_event_defaults(self):
        ev = FeedbackEvent()
        assert ev.sample_filepath == ""
        assert ev.mix_filepath == ""
        assert ev.session_id == ""
        assert ev.action == ""
        assert ev.rating is None
        assert ev.recommendation_rank == 0
        assert ev.context_style == ""
        assert ev.timestamp == 0.0


class TestPreferencePair:
    def test_preference_pair_defaults(self):
        pair = PreferencePair()
        assert pair.preferred_filepath == ""
        assert pair.rejected_filepath == ""
        assert pair.mix_filepath == ""
        assert pair.context_style == ""
        assert pair.strength == 1.0
        assert pair.timestamp == 0.0


class TestUserTasteModel:
    def test_taste_model_defaults(self):
        model = UserTasteModel()
        assert model.user_id == "default"
        assert model.role_bias == {}
        assert model.style_bias == {}
        assert model.weight_deltas == {}
        assert model.quality_threshold == 0.3
        assert model.training_pairs == 0
        assert model.model_version == 0
        assert model.last_trained == 0.0

    def test_taste_model_serialization(self):
        """to_dict / from_dict roundtrip preserves all fields."""
        model = UserTasteModel(
            user_id="user42",
            role_bias={"kick": 0.5, "snare": -0.2},
            style_bias={"techno": 0.8},
            weight_deltas={"spectral_sim": 0.1},
            quality_threshold=0.6,
            training_pairs=200,
            model_version=3,
            last_trained=1700000000.0,
        )
        d = model.to_dict()
        restored = UserTasteModel.from_dict(d)

        assert restored.user_id == "user42"
        assert restored.role_bias == {"kick": 0.5, "snare": -0.2}
        assert restored.style_bias == {"techno": 0.8}
        assert restored.weight_deltas == {"spectral_sim": 0.1}
        assert restored.quality_threshold == 0.6
        assert restored.training_pairs == 200
        assert restored.model_version == 3
        assert restored.last_trained == 1700000000.0

    def test_taste_model_json(self):
        """to_json handles numpy scalar types without raising."""
        model = UserTasteModel(
            user_id="np_user",
            role_bias={"kick": float(np.float32(0.75))},
            style_bias={"house": float(np.float64(0.9))},
            weight_deltas={"w": float(np.float32(-0.1))},
            quality_threshold=0.5,
            training_pairs=int(np.int64(50)),
        )
        raw = model.to_json()
        parsed = json.loads(raw)
        assert parsed["user_id"] == "np_user"
        assert abs(parsed["role_bias"]["kick"] - 0.75) < 1e-5
        assert parsed["training_pairs"] == 50

        # Also verify direct numpy values in dicts go through the serializer
        model2 = UserTasteModel(user_id="np2")
        # Manually set a numpy value that the custom serializer must handle
        model2.role_bias["x"] = np.float64(1.23)  # type: ignore[assignment]
        raw2 = model2.to_json()
        parsed2 = json.loads(raw2)
        assert abs(parsed2["role_bias"]["x"] - 1.23) < 1e-5
