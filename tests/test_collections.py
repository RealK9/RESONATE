"""
Tests for the RESONATE Smart Collection Curation system.
Covers collection generation, grouping by policy, genre labels, and ZIP export.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field

import pytest

# ── Ensure backend is importable ──────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# ── Mock heavy dependencies that routes.samples pulls in ──────────────────
# We mock essentia, indexer, and the full samples route to avoid needing
# native audio libraries in the test environment.
_essentia_mock = MagicMock()
sys.modules.setdefault("essentia", _essentia_mock)
sys.modules.setdefault("essentia.standard", _essentia_mock)

import state  # noqa: E402  — must come after sys.path insert


# ---------------------------------------------------------------------------
# Helpers — build mock recommendations matching the Recommendation dataclass
# ---------------------------------------------------------------------------

@dataclass
class MockRecommendation:
    filepath: str = ""
    filename: str = ""
    score: float = 0.0
    explanation: str = ""
    policy: str = ""
    need_addressed: str = ""
    role: str = ""


@dataclass
class MockRecommendationResult:
    mix_filepath: str = ""
    recommendations: list = field(default_factory=list)
    needs_addressed: list = field(default_factory=list)
    total_candidates_considered: int = 0

    def to_dict(self):
        return {
            "mix_filepath": self.mix_filepath,
            "recommendations": [r.__dict__ for r in self.recommendations],
            "needs_addressed": self.needs_addressed,
            "total_candidates_considered": self.total_candidates_considered,
        }


def _make_recs(policies_and_roles):
    """Build a list of MockRecommendation from (policy, role, score) tuples."""
    recs = []
    for i, (policy, role, score) in enumerate(policies_and_roles):
        recs.append(MockRecommendation(
            filepath=f"/samples/{role}_{i}.wav",
            filename=f"{role}_{i}.wav",
            score=score,
            explanation=f"Test explanation for {role}",
            policy=policy,
            role=role,
            need_addressed=f"need_{i}",
        ))
    return recs


def _make_result(recs):
    return MockRecommendationResult(
        mix_filepath="/test/mix.wav",
        recommendations=recs,
        needs_addressed=list({r.need_addressed for r in recs}),
        total_candidates_considered=len(recs) * 3,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_state():
    """Reset global state before each test."""
    state.latest_recommendations = None
    state.latest_mix_profile = None
    state.latest_gap_result = None
    yield
    state.latest_recommendations = None
    state.latest_mix_profile = None
    state.latest_gap_result = None


@pytest.fixture
def populated_state():
    """Set up state with a full set of recommendations across policies."""
    recs = _make_recs([
        ("fill_missing_role", "kick", 0.95),
        ("fill_missing_role", "snare_clap", 0.90),
        ("fill_missing_role", "bass", 0.88),
        ("fill_missing_role", "lead", 0.85),
        ("fill_missing_role", "pad", 0.80),
        ("improve_polish", "hats_tops", 0.75),
        ("improve_polish", "lead", 0.72),
        ("enhance_lift", "vocal_texture", 0.70),
        ("enhance_lift", "pad", 0.68),
        ("enhance_groove", "kick", 0.65),
        ("enhance_groove", "hats_tops", 0.62),
        ("add_movement", "fx_transitions", 0.60),
        ("add_movement", "lead", 0.58),
    ])
    result = _make_result(recs)
    state.latest_recommendations = result
    state.latest_mix_profile = {
        "style": {"primary_cluster": "modern_trap"},
        "analysis": {"key": "Cm", "bpm": 140},
    }
    state.latest_gap_result = {
        "readiness_score": 65,
        "missing_roles": ["kick", "snare_clap", "bass"],
        "gap_summary": "Mix needs more percussion",
    }
    return state


# ---------------------------------------------------------------------------
# Genre Label Helper Tests
# ---------------------------------------------------------------------------

class TestGenreLabels:
    def test_known_cluster(self):
        from routes.collections import get_genre_label
        assert get_genre_label("modern_trap") == "Trap"
        assert get_genre_label("2010s_edm_drop") == "EDM"
        assert get_genre_label("r_and_b") == "R&B"
        assert get_genre_label("lo_fi_chill") == "Lo-Fi"

    def test_unknown_cluster_titlecased(self):
        from routes.collections import get_genre_label
        result = get_genre_label("future_bass")
        assert result == "Future Bass"

    def test_empty_cluster_returns_your(self):
        from routes.collections import get_genre_label
        assert get_genre_label("") == "Your"
        assert get_genre_label(None) == "Your"


# ---------------------------------------------------------------------------
# Collection Generation Tests
# ---------------------------------------------------------------------------

class TestGenerateCollections:
    def test_generates_all_four_collections(self, populated_state):
        from routes.collections import generate_collections
        collections = generate_collections()
        ids = [c["id"] for c in collections]
        assert "essentials" in ids
        assert "polish" in ids
        assert "groove" in ids
        assert "top-picks" in ids

    def test_essentials_uses_fill_missing_role(self, populated_state):
        from routes.collections import generate_collections
        collections = generate_collections()
        essentials = next(c for c in collections if c["id"] == "essentials")
        assert essentials["name"] == "Trap Essentials"
        assert essentials["icon"] == "target"
        assert len(essentials["samples"]) == 5  # only 5 fill_missing_role recs
        assert all(
            s["filepath"].startswith("/samples/")
            for s in essentials["samples"]
        )

    def test_polish_groups_polish_and_lift(self, populated_state):
        from routes.collections import generate_collections
        collections = generate_collections()
        polish = next(c for c in collections if c["id"] == "polish")
        assert polish["icon"] == "sparkle"
        assert len(polish["samples"]) == 4  # 2 polish + 2 lift

    def test_groove_groups_groove_and_movement(self, populated_state):
        from routes.collections import generate_collections
        collections = generate_collections()
        groove = next(c for c in collections if c["id"] == "groove")
        assert groove["icon"] == "rhythm"
        assert len(groove["samples"]) == 4  # 2 groove + 2 movement

    def test_top_picks_sorted_by_score(self, populated_state):
        from routes.collections import generate_collections
        collections = generate_collections()
        top = next(c for c in collections if c["id"] == "top-picks")
        assert top["icon"] == "crown"
        scores = [s["score"] for s in top["samples"]]
        assert scores == sorted(scores, reverse=True)
        assert len(top["samples"]) == 10

    def test_total_impact_calculated(self, populated_state):
        from routes.collections import generate_collections
        collections = generate_collections()
        for c in collections:
            expected = round(sum(s["score"] for s in c["samples"]), 2)
            assert c["total_impact"] == expected

    def test_empty_when_no_recommendations(self):
        from routes.collections import generate_collections
        state.latest_recommendations = _make_result([])
        state.latest_mix_profile = {"style": {"primary_cluster": "modern_trap"}}
        collections = generate_collections()
        assert collections == []

    def test_missing_roles_in_description(self, populated_state):
        from routes.collections import generate_collections
        collections = generate_collections()
        essentials = next(c for c in collections if c["id"] == "essentials")
        assert "Kick" in essentials["description"]
        assert "Snare Clap" in essentials["description"]

    def test_no_essentials_when_no_fill_policy(self):
        from routes.collections import generate_collections
        recs = _make_recs([
            ("improve_polish", "lead", 0.9),
            ("enhance_groove", "kick", 0.8),
        ])
        state.latest_recommendations = _make_result(recs)
        state.latest_mix_profile = {"style": {"primary_cluster": "ambient"}}
        collections = generate_collections()
        ids = [c["id"] for c in collections]
        assert "essentials" not in ids
        assert "top-picks" in ids

    def test_sample_entry_fields(self, populated_state):
        from routes.collections import generate_collections
        collections = generate_collections()
        sample = collections[0]["samples"][0]
        assert "filepath" in sample
        assert "name" in sample
        assert "role" in sample
        assert "score" in sample
        assert "explanation" in sample


# ---------------------------------------------------------------------------
# Route Tests (using FastAPI TestClient)
# ---------------------------------------------------------------------------

class TestCollectionRoutes:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routes.collections import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_generate_404_when_no_recommendations(self, client):
        response = client.get("/collections/generate")
        assert response.status_code == 404

    def test_generate_404_when_no_mix_profile(self, client):
        state.latest_recommendations = _make_result([MockRecommendation()])
        response = client.get("/collections/generate")
        assert response.status_code == 404

    def test_generate_returns_collections(self, client, populated_state):
        response = client.get("/collections/generate")
        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
        assert len(data["collections"]) == 4

    def test_export_404_unknown_collection(self, client, populated_state):
        response = client.post("/collections/export/nonexistent")
        assert response.status_code == 404

    def test_export_creates_valid_zip(self, client, populated_state, tmp_path):
        """Test that export produces a valid ZIP with metadata."""
        sample_file = tmp_path / "kick_0.wav"
        sample_file.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch("routes.collections.find_sample_file", return_value=sample_file):
            response = client.post("/collections/export/essentials")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        buf = io.BytesIO(response.content)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            assert "_resonate_collection.json" in names
            assert any(n.endswith(".wav") for n in names)
            metadata = json.loads(zf.read("_resonate_collection.json"))
            assert metadata["collection"] == "Trap Essentials"
            assert "samples" in metadata
            assert len(metadata["samples"]) > 0

    def test_export_404_when_no_files_found(self, client, populated_state):
        """When none of the sample files exist on disk, return 404."""
        with patch("routes.collections.find_sample_file", return_value=None):
            response = client.post("/collections/export/essentials")
        assert response.status_code == 404
