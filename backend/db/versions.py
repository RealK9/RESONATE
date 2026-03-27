"""
RESONATE — Version Tracking Database Layer.
Stores track version snapshots for timeline visualization and comparison.
"""

import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager

from config import BACKEND_DIR

VERSIONS_DB_PATH = BACKEND_DIR / "versions.db"


def _get_connection():
    """Get a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(VERSIONS_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def _get_db():
    """Context manager for database connections."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init():
    """Create the track_versions table if it doesn't exist."""
    with _get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS track_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                version_label TEXT NOT NULL,
                filepath TEXT NOT NULL,
                readiness_score REAL,
                gap_summary TEXT,
                chart_potential REAL,
                missing_roles TEXT,
                analysis_json TEXT,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_versions_project
                ON track_versions(project_name);
        """)
    print("  \u2713 Versions database initialized")


def save_version(project_name, version_label, filepath, readiness_score=None,
                 gap_summary=None, chart_potential=None, missing_roles=None,
                 analysis_json=None):
    """Save a new version snapshot. Returns the new row id."""
    with _get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO track_versions
               (project_name, version_label, filepath, readiness_score,
                gap_summary, chart_potential, missing_roles, analysis_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                project_name,
                version_label,
                filepath,
                readiness_score,
                gap_summary,
                chart_potential,
                json.dumps(missing_roles) if missing_roles is not None else None,
                json.dumps(analysis_json) if analysis_json is not None else None,
                time.time(),
            ),
        )
        return cursor.lastrowid


def get_versions(project_name):
    """Get all versions for a project, ordered by created_at ascending."""
    with _get_db() as conn:
        rows = conn.execute(
            """SELECT id, project_name, version_label, filepath,
                      readiness_score, gap_summary, chart_potential,
                      missing_roles, analysis_json, created_at
               FROM track_versions
               WHERE project_name = ?
               ORDER BY created_at ASC""",
            (project_name,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d["missing_roles"]:
                d["missing_roles"] = json.loads(d["missing_roles"])
            if d["analysis_json"]:
                d["analysis_json"] = json.loads(d["analysis_json"])
            results.append(d)
        return results


def get_latest(project_name):
    """Get the most recent version for a project, or None."""
    with _get_db() as conn:
        row = conn.execute(
            """SELECT id, project_name, version_label, filepath,
                      readiness_score, gap_summary, chart_potential,
                      missing_roles, analysis_json, created_at
               FROM track_versions
               WHERE project_name = ?
               ORDER BY created_at DESC
               LIMIT 1""",
            (project_name,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d["missing_roles"]:
            d["missing_roles"] = json.loads(d["missing_roles"])
        if d["analysis_json"]:
            d["analysis_json"] = json.loads(d["analysis_json"])
        return d


def compare_versions(id_a, id_b):
    """Compare two versions by id. Returns delta readiness, new/resolved gaps."""
    with _get_db() as conn:
        row_a = conn.execute(
            "SELECT * FROM track_versions WHERE id = ?", (id_a,)
        ).fetchone()
        row_b = conn.execute(
            "SELECT * FROM track_versions WHERE id = ?", (id_b,)
        ).fetchone()

    if not row_a or not row_b:
        return None

    a = dict(row_a)
    b = dict(row_b)

    # Parse missing_roles
    roles_a = set(json.loads(a["missing_roles"])) if a["missing_roles"] else set()
    roles_b = set(json.loads(b["missing_roles"])) if b["missing_roles"] else set()

    readiness_a = a["readiness_score"] or 0
    readiness_b = b["readiness_score"] or 0

    return {
        "version_a": {"id": a["id"], "label": a["version_label"], "readiness": readiness_a},
        "version_b": {"id": b["id"], "label": b["version_label"], "readiness": readiness_b},
        "delta_readiness": round(readiness_b - readiness_a, 2),
        "new_gaps": sorted(roles_b - roles_a),
        "resolved_gaps": sorted(roles_a - roles_b),
        "unchanged_gaps": sorted(roles_a & roles_b),
    }


def list_projects():
    """List distinct project names with version count."""
    with _get_db() as conn:
        rows = conn.execute(
            """SELECT project_name, COUNT(*) as version_count,
                      MAX(created_at) as last_updated
               FROM track_versions
               GROUP BY project_name
               ORDER BY last_updated DESC"""
        ).fetchall()
        return [dict(row) for row in rows]


def next_label(project_name):
    """Auto-generate the next version label (v1, v2, ...) for a project."""
    with _get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM track_versions WHERE project_name = ?",
            (project_name,),
        ).fetchone()
        return f"v{row['cnt'] + 1}"
