"""
Tests for Chart Intelligence 2.0 routes.
Verifies /charts/trends and /charts/compare endpoints with mocked ChartAnalyzer.
"""
from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import state
from ml.training.charts.chart_analysis import ChartAnalyzer, DecadeProfile, GenreProfile
from routes.chart_intelligence import router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _mock_analyzer():
    """Create a mock ChartAnalyzer with realistic test data."""
    analyzer = MagicMock(spec=ChartAnalyzer)

    decade_2020 = DecadeProfile(
        decade=2020, count=500, bpm_mean=128.0, bpm_std=18.0, bpm_median=126.0,
        energy_mean=0.68, energy_std=0.15, danceability_mean=0.72, danceability_std=0.12,
        valence_mean=0.45, valence_std=0.18, loudness_mean=-6.5, loudness_std=2.5,
        acousticness_mean=0.15, instrumentalness_mean=0.05, major_ratio=0.55,
        duration_mean=195.0, duration_std=35.0,
    )
    decade_2010 = DecadeProfile(
        decade=2010, count=800, bpm_mean=122.0, bpm_std=22.0, bpm_median=120.0,
        energy_mean=0.65, energy_std=0.16, danceability_mean=0.68, danceability_std=0.14,
        valence_mean=0.50, valence_std=0.20, loudness_mean=-7.0, loudness_std=2.8,
        acousticness_mean=0.20, instrumentalness_mean=0.08, major_ratio=0.58,
        duration_mean=220.0, duration_std=40.0,
    )

    pop_profile = GenreProfile(
        genre="pop", count=1200, bpm_mean=120.0, bpm_std=20.0,
        energy_mean=0.65, danceability_mean=0.70, valence_mean=0.55,
        acousticness_mean=0.18, instrumentalness_mean=0.04, speechiness_mean=0.08,
        loudness_mean=-6.0, major_ratio=0.60, avg_peak_position=35.0, avg_weeks_on_chart=12.0,
    )
    electronic_profile = GenreProfile(
        genre="electronic", count=600, bpm_mean=128.0, bpm_std=15.0,
        energy_mean=0.78, danceability_mean=0.80, valence_mean=0.42,
        acousticness_mean=0.05, instrumentalness_mean=0.30, speechiness_mean=0.06,
        loudness_mean=-5.5, major_ratio=0.48, avg_peak_position=42.0, avg_weeks_on_chart=8.0,
    )

    analyzer.get_decade_profiles.return_value = {2020: decade_2020, 2010: decade_2010}
    analyzer.get_genre_profiles.return_value = {"pop": pop_profile, "electronic": electronic_profile}
    analyzer.analyze.return_value = None

    return analyzer


@pytest.fixture
def client():
    app = _make_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset shared state before each test."""
    old = state.latest_mix_profile
    yield
    state.latest_mix_profile = old


# ---------------------------------------------------------------------------
# /charts/trends tests
# ---------------------------------------------------------------------------

class TestChartTrends:

    @patch("routes.chart_intelligence._load_analyzer")
    def test_trends_returns_decades_and_genres(self, mock_load, client):
        mock_load.return_value = _mock_analyzer()
        resp = client.get("/charts/trends")
        assert resp.status_code == 200
        data = resp.json()
        assert "decades" in data
        assert "genres" in data
        assert "2020" in data["decades"]
        assert "2010" in data["decades"]
        assert "pop" in data["genres"]
        assert "electronic" in data["genres"]
        # Verify decade data
        d2020 = data["decades"]["2020"]
        assert d2020["bpm_mean"] == 128.0
        assert d2020["count"] == 500
        assert d2020["energy_mean"] == 0.68

    @patch("routes.chart_intelligence._load_analyzer")
    def test_trends_genre_filter(self, mock_load, client):
        mock_load.return_value = _mock_analyzer()
        resp = client.get("/charts/trends?genre=pop")
        assert resp.status_code == 200
        data = resp.json()
        assert "pop" in data["genres"]
        assert "electronic" not in data["genres"]

    @patch("routes.chart_intelligence._load_analyzer")
    def test_trends_decade_filter(self, mock_load, client):
        mock_load.return_value = _mock_analyzer()
        resp = client.get("/charts/trends?decade=2020")
        assert resp.status_code == 200
        data = resp.json()
        assert "2020" in data["decades"]
        assert "2010" not in data["decades"]


# ---------------------------------------------------------------------------
# /charts/compare tests
# ---------------------------------------------------------------------------

class TestChartCompare:

    @patch("routes.chart_intelligence._load_analyzer")
    def test_compare_with_valid_mix(self, mock_load, client):
        mock_load.return_value = _mock_analyzer()
        state.latest_mix_profile = {
            "analysis": {"bpm": 125, "key": "C", "genre": "pop", "energy": 0.72},
            "style": {"primary_cluster": "pop"},
            "bpm": 125,
            "energy": 0.72,
            "valence": 0.50,
            "danceability": 0.65,
            "key": "C",
        }
        resp = client.get("/charts/compare")
        assert resp.status_code == 200
        data = resp.json()
        assert "your_mix" in data
        assert "chart_average" in data
        assert "insights" in data
        assert "decade_trends" in data
        assert data["your_mix"]["bpm"] == 125
        assert data["chart_average"]["genre"] == "pop"
        assert len(data["insights"]) > 0
        assert len(data["decade_trends"]) > 0

    def test_compare_no_mix_returns_404(self, client):
        state.latest_mix_profile = None
        resp = client.get("/charts/compare")
        assert resp.status_code == 404

    @patch("routes.chart_intelligence._load_analyzer")
    def test_compare_unknown_genre_falls_back_to_pop(self, mock_load, client):
        mock_load.return_value = _mock_analyzer()
        state.latest_mix_profile = {
            "analysis": {"bpm": 140, "genre": "unknown_genre"},
            "style": {"primary_cluster": "unknown_genre"},
            "bpm": 140,
            "energy": 0.80,
        }
        resp = client.get("/charts/compare")
        assert resp.status_code == 200
        data = resp.json()
        # Should fall back to pop
        assert data["chart_average"]["genre"] == "pop"

    @patch("routes.chart_intelligence._load_analyzer")
    def test_compare_insights_content(self, mock_load, client):
        """Insights should mention BPM when it diverges from average."""
        mock_load.return_value = _mock_analyzer()
        state.latest_mix_profile = {
            "bpm": 180,  # Way above pop average of 120
            "energy": 0.95,
            "valence": 0.10,
            "danceability": 0.30,
            "style": {"primary_cluster": "pop"},
        }
        resp = client.get("/charts/compare")
        data = resp.json()
        insights_text = " ".join(data["insights"]).lower()
        assert "bpm" in insights_text
        assert "faster" in insights_text

    @patch("routes.chart_intelligence._load_analyzer")
    def test_decade_trends_sorted_recent_first(self, mock_load, client):
        mock_load.return_value = _mock_analyzer()
        state.latest_mix_profile = {
            "bpm": 120,
            "style": {"primary_cluster": "pop"},
        }
        resp = client.get("/charts/compare")
        data = resp.json()
        decades = [d["decade"] for d in data["decade_trends"]]
        assert decades == sorted(decades, reverse=True)
