"""Tests for Phase 5 preference dataset (SQLite-backed)."""
import os
import tempfile
import time

import pytest

from ml.models.preference import (
    FeedbackEvent,
    PreferencePair,
    UserTasteModel,
)
from ml.training.preference_dataset import PreferenceDataset


@pytest.fixture
def ds(tmp_path):
    """Return an initialised PreferenceDataset backed by a temp SQLite file."""
    db = str(tmp_path / "pref_test.db")
    dataset = PreferenceDataset(db)
    dataset.init()
    return dataset


# ---- table creation --------------------------------------------------------

class TestInit:
    def test_init_creates_tables(self, ds: PreferenceDataset):
        """All three tables should exist after init()."""
        import sqlite3

        conn = sqlite3.connect(ds.db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "feedback_events" in tables
        assert "preference_pairs" in tables
        assert "taste_models" in tables


# ---- feedback logging & retrieval ------------------------------------------

class TestFeedback:
    def test_log_and_retrieve_feedback(self, ds: PreferenceDataset):
        ev1 = FeedbackEvent(
            sample_filepath="/samples/kick.wav",
            mix_filepath="/mixes/track.wav",
            session_id="s1",
            action="click",
            timestamp=1000.0,
        )
        ev2 = FeedbackEvent(
            sample_filepath="/samples/snare.wav",
            mix_filepath="/mixes/track.wav",
            session_id="s1",
            action="drag",
            timestamp=1001.0,
        )
        ds.log_feedback(ev1)
        ds.log_feedback(ev2)

        results = ds.get_feedback()
        assert len(results) == 2
        # Most recent first
        assert results[0].action == "drag"
        assert results[1].action == "click"

    def test_filter_by_mix(self, ds: PreferenceDataset):
        ds.log_feedback(FeedbackEvent(
            sample_filepath="/s/a.wav", mix_filepath="/m/x.wav",
            session_id="s1", action="click", timestamp=1.0,
        ))
        ds.log_feedback(FeedbackEvent(
            sample_filepath="/s/b.wav", mix_filepath="/m/y.wav",
            session_id="s1", action="drag", timestamp=2.0,
        ))
        results = ds.get_feedback(mix_filepath="/m/x.wav")
        assert len(results) == 1
        assert results[0].sample_filepath == "/s/a.wav"

    def test_filter_by_session(self, ds: PreferenceDataset):
        ds.log_feedback(FeedbackEvent(
            sample_filepath="/s/a.wav", mix_filepath="/m/x.wav",
            session_id="s1", action="click", timestamp=1.0,
        ))
        ds.log_feedback(FeedbackEvent(
            sample_filepath="/s/b.wav", mix_filepath="/m/x.wav",
            session_id="s2", action="drag", timestamp=2.0,
        ))
        results = ds.get_feedback(session_id="s2")
        assert len(results) == 1
        assert results[0].session_filepath if False else results[0].session_id == "s2"


# ---- pair construction -----------------------------------------------------

class TestPairBuilding:
    def test_build_pairs_from_session(self, ds: PreferenceDataset):
        """drag + skip in one session -> preference pair (drag preferred)."""
        ds.log_feedback(FeedbackEvent(
            sample_filepath="/s/a.wav", mix_filepath="/m/x.wav",
            session_id="s1", action="drag", timestamp=1.0,
        ))
        ds.log_feedback(FeedbackEvent(
            sample_filepath="/s/b.wav", mix_filepath="/m/x.wav",
            session_id="s1", action="skip", timestamp=2.0,
        ))
        pairs = ds.build_pairs(session_id="s1")
        assert len(pairs) == 1
        assert pairs[0].preferred_filepath == "/s/a.wav"
        assert pairs[0].rejected_filepath == "/s/b.wav"
        # drag(1.0) - skip(0.1) = 0.9
        assert abs(pairs[0].strength - 0.9) < 1e-4

    def test_pair_strength_ordering(self, ds: PreferenceDataset):
        """drag > audition > click: three samples yield correct pair ordering."""
        ds.log_feedback(FeedbackEvent(
            sample_filepath="/s/drag.wav", mix_filepath="/m/x.wav",
            session_id="s1", action="drag", timestamp=1.0,
        ))
        ds.log_feedback(FeedbackEvent(
            sample_filepath="/s/aud.wav", mix_filepath="/m/x.wav",
            session_id="s1", action="audition", timestamp=2.0,
        ))
        ds.log_feedback(FeedbackEvent(
            sample_filepath="/s/click.wav", mix_filepath="/m/x.wav",
            session_id="s1", action="click", timestamp=3.0,
        ))
        pairs = ds.build_pairs(session_id="s1")

        # 3 samples: drag>aud, drag>click, aud>click => 3 pairs
        assert len(pairs) == 3

        pair_map = {
            (p.preferred_filepath, p.rejected_filepath): p.strength
            for p in pairs
        }
        # drag(1.0) - audition(0.6) = 0.4
        assert abs(pair_map[("/s/drag.wav", "/s/aud.wav")] - 0.4) < 1e-4
        # drag(1.0) - click(0.4) = 0.6
        assert abs(pair_map[("/s/drag.wav", "/s/click.wav")] - 0.6) < 1e-4
        # audition(0.6) - click(0.4) = 0.2
        assert abs(pair_map[("/s/aud.wav", "/s/click.wav")] - 0.2) < 1e-4

    def test_build_pairs_multiple_samples(self, ds: PreferenceDataset):
        """4 samples with distinct strengths -> C(4,2)=6 directed pairs."""
        actions = [
            ("drag", "/s/1.wav"),
            ("keep", "/s/2.wav"),
            ("audition", "/s/3.wav"),
            ("skip", "/s/4.wav"),
        ]
        for action, path in actions:
            ds.log_feedback(FeedbackEvent(
                sample_filepath=path, mix_filepath="/m/x.wav",
                session_id="s1", action=action, timestamp=time.time(),
            ))
        pairs = ds.build_pairs(session_id="s1")
        # 4 distinct strengths: 1.0, 0.9, 0.6, 0.1
        # directed pairs where A > B: C(4,2) = 6
        assert len(pairs) == 6


# ---- training data ---------------------------------------------------------

class TestTrainingData:
    def test_get_training_data(self, ds: PreferenceDataset):
        """get_training_data returns pairs when count >= min_pairs."""
        # Build enough pairs: 4 samples -> 6 pairs
        actions = ["drag", "keep", "audition", "skip"]
        for i, action in enumerate(actions):
            ds.log_feedback(FeedbackEvent(
                sample_filepath=f"/s/{i}.wav", mix_filepath="/m/x.wav",
                session_id="s1", action=action, timestamp=float(i),
            ))
        ds.build_pairs(session_id="s1")

        # min_pairs=6 should return data
        data = ds.get_training_data(min_pairs=6)
        assert len(data) == 6

        # min_pairs=100 should return empty
        data_empty = ds.get_training_data(min_pairs=100)
        assert len(data_empty) == 0


# ---- taste model persistence ----------------------------------------------

class TestTasteModel:
    def test_save_and_load_taste_model(self, ds: PreferenceDataset):
        model = UserTasteModel(
            user_id="testuser",
            role_bias={"kick": 0.3},
            style_bias={"techno": 0.7},
            weight_deltas={"w1": -0.1},
            quality_threshold=0.5,
            training_pairs=42,
            model_version=2,
            last_trained=1700000000.0,
        )
        ds.save_taste_model(model)
        loaded = ds.load_taste_model("testuser")

        assert loaded is not None
        assert loaded.user_id == "testuser"
        assert loaded.role_bias == {"kick": 0.3}
        assert loaded.style_bias == {"techno": 0.7}
        assert loaded.quality_threshold == 0.5
        assert loaded.training_pairs == 42
        assert loaded.model_version == 2

    def test_load_nonexistent_returns_none(self, ds: PreferenceDataset):
        result = ds.load_taste_model("ghost_user")
        assert result is None
