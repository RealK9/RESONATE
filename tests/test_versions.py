"""
Tests for the RESONATE version tracking system.
Covers DB layer (save, get, compare, list, auto-label) and route layer.
"""
from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ── Ensure backend is importable ──────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """Redirect versions DB to a temporary file for every test."""
    db_file = tmp_path / "test_versions.db"
    import db.versions as versions_mod
    monkeypatch.setattr(versions_mod, "VERSIONS_DB_PATH", db_file)
    versions_mod.init()
    yield versions_mod


@pytest.fixture
def vdb(_temp_db):
    """Convenience alias for the versions module."""
    return _temp_db


# ---------------------------------------------------------------------------
# DB Layer Tests
# ---------------------------------------------------------------------------

class TestSaveAndGetVersions:
    def test_save_returns_id(self, vdb):
        vid = vdb.save_version("my_beat", "v1", "/path/beat.wav", readiness_score=72.5)
        assert isinstance(vid, int)
        assert vid >= 1

    def test_get_versions_ordered_asc(self, vdb):
        vdb.save_version("proj", "v1", "/a.wav", readiness_score=50)
        time.sleep(0.01)
        vdb.save_version("proj", "v2", "/b.wav", readiness_score=70)
        versions = vdb.get_versions("proj")
        assert len(versions) == 2
        assert versions[0]["version_label"] == "v1"
        assert versions[1]["version_label"] == "v2"
        assert versions[0]["created_at"] <= versions[1]["created_at"]

    def test_get_versions_empty_project(self, vdb):
        versions = vdb.get_versions("nonexistent")
        assert versions == []

    def test_stores_and_retrieves_all_fields(self, vdb):
        missing = ["bass", "lead"]
        analysis = {"key": "Cm", "bpm": 128}
        vid = vdb.save_version(
            "beat", "v1", "/path.wav",
            readiness_score=65.0,
            gap_summary="Needs bass and lead",
            chart_potential=0.8,
            missing_roles=missing,
            analysis_json=analysis,
        )
        versions = vdb.get_versions("beat")
        v = versions[0]
        assert v["id"] == vid
        assert v["readiness_score"] == 65.0
        assert v["gap_summary"] == "Needs bass and lead"
        assert v["chart_potential"] == 0.8
        assert v["missing_roles"] == ["bass", "lead"]
        assert v["analysis_json"] == {"key": "Cm", "bpm": 128}

    def test_different_projects_isolated(self, vdb):
        vdb.save_version("proj_a", "v1", "/a.wav")
        vdb.save_version("proj_b", "v1", "/b.wav")
        assert len(vdb.get_versions("proj_a")) == 1
        assert len(vdb.get_versions("proj_b")) == 1


class TestGetLatest:
    def test_returns_most_recent(self, vdb):
        vdb.save_version("proj", "v1", "/a.wav", readiness_score=40)
        time.sleep(0.01)
        vdb.save_version("proj", "v2", "/b.wav", readiness_score=80)
        latest = vdb.get_latest("proj")
        assert latest is not None
        assert latest["version_label"] == "v2"
        assert latest["readiness_score"] == 80

    def test_returns_none_for_empty(self, vdb):
        assert vdb.get_latest("nothing") is None


class TestCompareVersions:
    def test_delta_readiness(self, vdb):
        id_a = vdb.save_version("proj", "v1", "/a.wav", readiness_score=45, missing_roles=["bass", "pad"])
        id_b = vdb.save_version("proj", "v2", "/b.wav", readiness_score=72, missing_roles=["pad"])
        result = vdb.compare_versions(id_a, id_b)
        assert result is not None
        assert result["delta_readiness"] == 27.0
        assert result["resolved_gaps"] == ["bass"]
        assert result["new_gaps"] == []
        assert result["unchanged_gaps"] == ["pad"]

    def test_negative_delta(self, vdb):
        id_a = vdb.save_version("proj", "v1", "/a.wav", readiness_score=80, missing_roles=[])
        id_b = vdb.save_version("proj", "v2", "/b.wav", readiness_score=60, missing_roles=["bass"])
        result = vdb.compare_versions(id_a, id_b)
        assert result["delta_readiness"] == -20.0
        assert result["new_gaps"] == ["bass"]

    def test_returns_none_for_missing_ids(self, vdb):
        assert vdb.compare_versions(999, 1000) is None

    def test_compare_with_null_missing_roles(self, vdb):
        id_a = vdb.save_version("proj", "v1", "/a.wav", readiness_score=50)
        id_b = vdb.save_version("proj", "v2", "/b.wav", readiness_score=70, missing_roles=["bass"])
        result = vdb.compare_versions(id_a, id_b)
        assert result["new_gaps"] == ["bass"]
        assert result["resolved_gaps"] == []


class TestListProjects:
    def test_lists_distinct_projects(self, vdb):
        vdb.save_version("alpha", "v1", "/a.wav")
        vdb.save_version("alpha", "v2", "/a2.wav")
        vdb.save_version("beta", "v1", "/b.wav")
        projects = vdb.list_projects()
        assert len(projects) == 2
        names = {p["project_name"] for p in projects}
        assert names == {"alpha", "beta"}
        alpha = next(p for p in projects if p["project_name"] == "alpha")
        assert alpha["version_count"] == 2

    def test_empty_returns_empty(self, vdb):
        assert vdb.list_projects() == []


class TestAutoLabel:
    def test_first_version_is_v1(self, vdb):
        label = vdb.next_label("newproj")
        assert label == "v1"

    def test_increments_correctly(self, vdb):
        vdb.save_version("proj", "v1", "/a.wav")
        assert vdb.next_label("proj") == "v2"
        vdb.save_version("proj", "v2", "/b.wav")
        assert vdb.next_label("proj") == "v3"

    def test_independent_per_project(self, vdb):
        vdb.save_version("a", "v1", "/a.wav")
        vdb.save_version("a", "v2", "/a2.wav")
        assert vdb.next_label("a") == "v3"
        assert vdb.next_label("b") == "v1"


# ---------------------------------------------------------------------------
# Route Layer Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client(vdb):
    """Create a FastAPI test client with the versions router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from routes.versions import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestVersionRoutes:
    def test_save_version_route(self, app_client):
        # Mock state so route can read gap/mix data
        import state
        state.latest_gap_result = {"readiness_score": 75, "gap_summary": "Missing bass", "chart_potential": 0.6, "missing_roles": ["bass"]}
        state.latest_mix_profile = {"key": "Am", "bpm": 120}
        resp = app_client.post("/versions", json={"project_name": "test_beat", "filepath": "beat.wav"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert data["version_label"] == "v1"
        assert data["readiness_score"] == 75

    def test_save_version_requires_project_name(self, app_client):
        resp = app_client.post("/versions", json={})
        assert resp.status_code == 400

    def test_list_versions_route(self, app_client):
        import state
        state.latest_gap_result = None
        state.latest_mix_profile = None
        app_client.post("/versions", json={"project_name": "proj", "filepath": "a.wav"})
        app_client.post("/versions", json={"project_name": "proj", "filepath": "b.wav"})
        resp = app_client.get("/versions/proj")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["versions"]) == 2

    def test_compare_route(self, app_client):
        import state
        state.latest_gap_result = {"readiness_score": 50, "gap_summary": "x", "chart_potential": 0.5, "missing_roles": ["bass"]}
        state.latest_mix_profile = {}
        resp1 = app_client.post("/versions", json={"project_name": "proj", "filepath": "a.wav"})
        id_a = resp1.json()["id"]

        state.latest_gap_result = {"readiness_score": 80, "gap_summary": "y", "chart_potential": 0.8, "missing_roles": []}
        resp2 = app_client.post("/versions", json={"project_name": "proj", "filepath": "b.wav"})
        id_b = resp2.json()["id"]

        resp = app_client.get(f"/versions/compare?id_a={id_a}&id_b={id_b}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["delta_readiness"] == 30.0
        assert "bass" in data["resolved_gaps"]

    def test_compare_not_found(self, app_client):
        resp = app_client.get("/versions/compare?id_a=999&id_b=1000")
        assert resp.status_code == 404

    def test_list_projects_route(self, app_client):
        import state
        state.latest_gap_result = None
        state.latest_mix_profile = None
        app_client.post("/versions", json={"project_name": "alpha", "filepath": "a.wav"})
        app_client.post("/versions", json={"project_name": "beta", "filepath": "b.wav"})
        resp = app_client.get("/versions/projects")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["projects"]) == 2

    def test_custom_version_label(self, app_client):
        import state
        state.latest_gap_result = None
        state.latest_mix_profile = None
        resp = app_client.post("/versions", json={"project_name": "proj", "version_label": "mix-down-1", "filepath": "a.wav"})
        data = resp.json()
        assert data["version_label"] == "mix-down-1"
