"""
RESONATE — SQLite Database Layer.
Manages all persistent data: samples, sessions, ratings, preferences, collections.
"""

import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager

from config import DB_PATH


def get_connection():
    """Get a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections with auto-commit."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_read_db():
    """Read-only context manager — no commit/rollback overhead."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    with get_db() as conn:
        conn.executescript("""
            -- Sample index (replaces sample_index.json over time)
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                hash TEXT,
                duration REAL DEFAULT 0,
                bpm REAL DEFAULT 0,
                key TEXT DEFAULT 'N/A',
                rms REAL DEFAULT 0,
                spectral_centroid REAL DEFAULT 0,
                mfcc_profile TEXT DEFAULT '[]',
                frequency_bands TEXT DEFAULT '{}',
                sample_type TEXT DEFAULT 'unknown',
                created_at REAL DEFAULT (strftime('%s','now')),
                updated_at REAL DEFAULT (strftime('%s','now'))
            );

            -- Analysis sessions (track + results snapshot)
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                track_filename TEXT NOT NULL,
                track_profile TEXT NOT NULL,
                ai_analysis TEXT,
                created_at REAL DEFAULT (strftime('%s','now'))
            );

            -- User ratings on samples
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_filepath TEXT NOT NULL,
                session_id INTEGER,
                rating INTEGER CHECK(rating BETWEEN 1 AND 5),
                created_at REAL DEFAULT (strftime('%s','now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            -- Usage history (plays, drags, favorites)
            CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_filepath TEXT NOT NULL,
                session_id INTEGER,
                action TEXT NOT NULL CHECK(action IN ('play', 'drag', 'favorite', 'unfavorite')),
                created_at REAL DEFAULT (strftime('%s','now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            -- User preferences (theme, volumes, etc.)
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now'))
            );

            -- Sample collections (smart & manual)
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                is_smart INTEGER DEFAULT 0,
                smart_query TEXT,
                created_at REAL DEFAULT (strftime('%s','now'))
            );

            -- Collection membership
            CREATE TABLE IF NOT EXISTS collection_samples (
                collection_id INTEGER NOT NULL,
                sample_filepath TEXT NOT NULL,
                added_at REAL DEFAULT (strftime('%s','now')),
                PRIMARY KEY (collection_id, sample_filepath),
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_samples_type ON samples(sample_type);
            CREATE INDEX IF NOT EXISTS idx_samples_key ON samples(key);
            CREATE INDEX IF NOT EXISTS idx_ratings_filepath ON ratings(sample_filepath);
            CREATE INDEX IF NOT EXISTS idx_usage_filepath ON usage_history(sample_filepath);
            CREATE INDEX IF NOT EXISTS idx_usage_action ON usage_history(action);
            CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
        """)
    print("  ✓ Database initialized")


# ═══════════════════════════════════════════════════════════════════════════
# SESSION OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def save_session(track_filename, track_profile, ai_analysis, name=None):
    """Save an analysis session to the database."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO sessions (name, track_filename, track_profile, ai_analysis) VALUES (?, ?, ?, ?)",
            (
                name or track_filename,
                track_filename,
                json.dumps(track_profile),
                json.dumps(ai_analysis) if ai_analysis else None,
            )
        )
        return cursor.lastrowid


def get_sessions(limit=50):
    """Get recent analysis sessions."""
    with get_read_db() as conn:
        rows = conn.execute(
            "SELECT id, name, track_filename, created_at FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_session(session_id):
    """Get a single session with full data."""
    with get_read_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["track_profile"] = json.loads(result["track_profile"])
        if result["ai_analysis"]:
            result["ai_analysis"] = json.loads(result["ai_analysis"])
        return result


def delete_session(session_id):
    """Delete a session and its associated data."""
    with get_db() as conn:
        conn.execute("DELETE FROM ratings WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM usage_history WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# ═══════════════════════════════════════════════════════════════════════════
# RATING OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def rate_sample(sample_filepath, rating, session_id=None):
    """Rate a sample (1-5 stars). Updates if rating already exists for this session."""
    with get_db() as conn:
        # Upsert: delete old rating for same sample+session, insert new
        if session_id:
            conn.execute(
                "DELETE FROM ratings WHERE sample_filepath = ? AND session_id = ?",
                (sample_filepath, session_id)
            )
        conn.execute(
            "INSERT INTO ratings (sample_filepath, session_id, rating) VALUES (?, ?, ?)",
            (sample_filepath, session_id, rating)
        )


def get_sample_ratings(sample_filepath):
    """Get all ratings for a sample."""
    with get_read_db() as conn:
        rows = conn.execute(
            "SELECT rating, created_at FROM ratings WHERE sample_filepath = ? ORDER BY created_at DESC",
            (sample_filepath,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_average_rating(sample_filepath):
    """Get average rating for a sample."""
    with get_read_db() as conn:
        row = conn.execute(
            "SELECT AVG(rating) as avg_rating, COUNT(*) as count FROM ratings WHERE sample_filepath = ?",
            (sample_filepath,)
        ).fetchone()
        return {"avg_rating": row["avg_rating"] or 0, "count": row["count"]}


def get_top_rated_samples(limit=50):
    """Get top rated samples across all sessions."""
    with get_read_db() as conn:
        rows = conn.execute("""
            SELECT sample_filepath, AVG(rating) as avg_rating, COUNT(*) as rating_count
            FROM ratings
            GROUP BY sample_filepath
            HAVING rating_count >= 1
            ORDER BY avg_rating DESC, rating_count DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


# ═══════════════════════════════════════════════════════════════════════════
# USAGE HISTORY OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def log_usage(sample_filepath, action, session_id=None):
    """Log a usage event (play, drag, favorite, unfavorite)."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO usage_history (sample_filepath, session_id, action) VALUES (?, ?, ?)",
            (sample_filepath, session_id, action)
        )


def get_most_used_samples(action=None, limit=50):
    """Get most frequently used samples, optionally filtered by action type."""
    with get_read_db() as conn:
        if action:
            rows = conn.execute("""
                SELECT sample_filepath, COUNT(*) as use_count
                FROM usage_history WHERE action = ?
                GROUP BY sample_filepath
                ORDER BY use_count DESC LIMIT ?
            """, (action, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT sample_filepath, COUNT(*) as use_count
                FROM usage_history
                GROUP BY sample_filepath
                ORDER BY use_count DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(row) for row in rows]


def get_recently_used_samples(limit=50):
    """Get recently used samples (most recent first)."""
    with get_read_db() as conn:
        rows = conn.execute("""
            SELECT DISTINCT sample_filepath, MAX(created_at) as last_used
            FROM usage_history
            GROUP BY sample_filepath
            ORDER BY last_used DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


# ═══════════════════════════════════════════════════════════════════════════
# PREFERENCE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def set_preference(key, value):
    """Set a user preference."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time())
        )


def get_preference(key, default=None):
    """Get a user preference."""
    with get_read_db() as conn:
        row = conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return json.loads(row["value"])
        return default


def get_all_preferences():
    """Get all user preferences as a dict."""
    with get_read_db() as conn:
        rows = conn.execute("SELECT key, value FROM preferences").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTION OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_collection(name, description=None, is_smart=False, smart_query=None):
    """Create a new collection."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO collections (name, description, is_smart, smart_query) VALUES (?, ?, ?, ?)",
            (name, description, int(is_smart), json.dumps(smart_query) if smart_query else None)
        )
        return cursor.lastrowid


def get_collections():
    """Get all collections."""
    with get_read_db() as conn:
        rows = conn.execute(
            "SELECT c.*, COUNT(cs.sample_filepath) as sample_count "
            "FROM collections c LEFT JOIN collection_samples cs ON c.id = cs.collection_id "
            "GROUP BY c.id ORDER BY c.created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def add_to_collection(collection_id, sample_filepath):
    """Add a sample to a collection."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO collection_samples (collection_id, sample_filepath) VALUES (?, ?)",
            (collection_id, sample_filepath)
        )


def remove_from_collection(collection_id, sample_filepath):
    """Remove a sample from a collection."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM collection_samples WHERE collection_id = ? AND sample_filepath = ?",
            (collection_id, sample_filepath)
        )


def get_collection_samples(collection_id):
    """Get all sample filepaths in a collection."""
    with get_read_db() as conn:
        rows = conn.execute(
            "SELECT sample_filepath, added_at FROM collection_samples WHERE collection_id = ? ORDER BY added_at DESC",
            (collection_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def delete_collection(collection_id):
    """Delete a collection and its memberships."""
    with get_db() as conn:
        conn.execute("DELETE FROM collection_samples WHERE collection_id = ?", (collection_id,))
        conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
