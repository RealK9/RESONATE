"""
Persistent storage for SampleProfile objects.
Uses SQLite with JSON columns for rich nested data.
Designed to coexist with the existing backend/db/database.py.
"""
from __future__ import annotations
import sqlite3
import json
from contextlib import contextmanager
import numpy as np
from ml.models.sample_profile import SampleProfile


def _numpy_serializer(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


class SampleStore:
    """SQLite-backed storage for sample profiles."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _db(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self):
        """Create the sample_profiles table."""
        with self._db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sample_profiles (
                    filepath TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT,
                    source TEXT DEFAULT 'local',
                    core TEXT DEFAULT '{}',
                    spectral TEXT DEFAULT '{}',
                    harmonic TEXT DEFAULT '{}',
                    transients TEXT DEFAULT '{}',
                    perceptual TEXT DEFAULT '{}',
                    embeddings TEXT DEFAULT '{}',
                    labels TEXT DEFAULT '{}',
                    created_at REAL DEFAULT (strftime('%s','now')),
                    updated_at REAL DEFAULT (strftime('%s','now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_profiles_hash
                ON sample_profiles(file_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_profiles_role
                ON sample_profiles(json_extract(labels, '$.role'))
            """)

    def save(self, profile: SampleProfile):
        """Insert or update a sample profile."""
        d = profile.to_dict()
        with self._db() as conn:
            conn.execute("""
                INSERT INTO sample_profiles
                    (filepath, filename, file_hash, source, core, spectral,
                     harmonic, transients, perceptual, embeddings, labels, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
                ON CONFLICT(filepath) DO UPDATE SET
                    filename=excluded.filename,
                    file_hash=excluded.file_hash,
                    source=excluded.source,
                    core=excluded.core,
                    spectral=excluded.spectral,
                    harmonic=excluded.harmonic,
                    transients=excluded.transients,
                    perceptual=excluded.perceptual,
                    embeddings=excluded.embeddings,
                    labels=excluded.labels,
                    updated_at=strftime('%s','now')
            """, (
                d["filepath"], d["filename"], d["file_hash"], d["source"],
                json.dumps(d["core"], default=_numpy_serializer),
                json.dumps(d["spectral"], default=_numpy_serializer),
                json.dumps(d["harmonic"], default=_numpy_serializer),
                json.dumps(d["transients"], default=_numpy_serializer),
                json.dumps(d["perceptual"], default=_numpy_serializer),
                json.dumps(d["embeddings"], default=_numpy_serializer),
                json.dumps(d["labels"], default=_numpy_serializer),
            ))

    def load(self, filepath: str) -> SampleProfile | None:
        """Load a sample profile by filepath."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM sample_profiles WHERE filepath = ?", (filepath,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_profile(row)

    def delete(self, filepath: str):
        """Delete a sample profile."""
        with self._db() as conn:
            conn.execute("DELETE FROM sample_profiles WHERE filepath = ?", (filepath,))

    def list_all(self, limit: int = 0) -> list[SampleProfile]:
        """List all sample profiles."""
        with self._db() as conn:
            query = "SELECT * FROM sample_profiles ORDER BY updated_at DESC"
            if limit > 0:
                query += f" LIMIT {limit}"
            rows = conn.execute(query).fetchall()
            return [self._row_to_profile(r) for r in rows]

    def count(self) -> int:
        """Count total profiles."""
        with self._db() as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM sample_profiles").fetchone()
            return row["c"]

    def search_by_role(self, role: str) -> list[SampleProfile]:
        """Find samples by their classified role."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM sample_profiles WHERE json_extract(labels, '$.role') = ?",
                (role,)
            ).fetchall()
            return [self._row_to_profile(r) for r in rows]

    def needs_reanalysis(self, filepath: str, current_hash: str) -> bool:
        """Check if a file needs (re)analysis based on its hash."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT file_hash FROM sample_profiles WHERE filepath = ?", (filepath,)
            ).fetchone()
            if not row:
                return True
            return row["file_hash"] != current_hash

    def _row_to_profile(self, row: sqlite3.Row) -> SampleProfile:
        """Convert a database row to a SampleProfile."""
        d = {
            "filepath": row["filepath"],
            "filename": row["filename"],
            "file_hash": row["file_hash"],
            "source": row["source"],
            "core": json.loads(row["core"]),
            "spectral": json.loads(row["spectral"]),
            "harmonic": json.loads(row["harmonic"]),
            "transients": json.loads(row["transients"]),
            "perceptual": json.loads(row["perceptual"]),
            "embeddings": json.loads(row["embeddings"]),
            "labels": json.loads(row["labels"]),
        }
        return SampleProfile.from_dict(d)
