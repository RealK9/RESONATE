"""Feedback collection and preference dataset construction for Phase 5.

Stores feedback events in SQLite, constructs preference pairs from implicit
signals, and provides the training dataset for the preference-aware reranker.

Signal-to-preference strength mapping:
    drag    = 1.0
    keep    = 0.9
    rate(5) = 1.0
    rate(4) = 0.8
    rate(3) = 0.5  (neutral, still used for pairs against negatives)
    rate(2) = 0.2
    rate(1) = 0.1
    audition = 0.6
    click    = 0.4
    skip     = 0.1
    discard  = 0.05
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from backend.ml.models.preference import (
    FeedbackEvent,
    PreferencePair,
    UserTasteModel,
)

# ---------------------------------------------------------------------------
# Action strength mapping
# ---------------------------------------------------------------------------

_ACTION_STRENGTH: dict[str, float] = {
    "drag": 1.0,
    "keep": 0.9,
    "audition": 0.6,
    "click": 0.4,
    "skip": 0.1,
    "discard": 0.05,
}


def _event_strength(event: FeedbackEvent) -> float:
    """Return the preference-signal strength for a feedback event."""
    if event.action == "rate" and event.rating is not None:
        return {5: 1.0, 4: 0.8, 3: 0.5, 2: 0.2, 1: 0.1}.get(event.rating, 0.3)
    return _ACTION_STRENGTH.get(event.action, 0.3)


# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feedback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_filepath TEXT NOT NULL,
    mix_filepath TEXT NOT NULL,
    session_id TEXT NOT NULL,
    action TEXT NOT NULL,
    rating INTEGER,
    recommendation_rank INTEGER DEFAULT 0,
    context_style TEXT DEFAULT '',
    timestamp REAL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS preference_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    preferred_filepath TEXT NOT NULL,
    rejected_filepath TEXT NOT NULL,
    mix_filepath TEXT NOT NULL,
    context_style TEXT DEFAULT '',
    strength REAL DEFAULT 1.0,
    timestamp REAL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS taste_models (
    user_id TEXT PRIMARY KEY,
    model_data TEXT NOT NULL,
    model_version INTEGER DEFAULT 0,
    updated_at REAL DEFAULT (strftime('%s','now'))
);
"""


# ---------------------------------------------------------------------------
# PreferenceDataset
# ---------------------------------------------------------------------------


class PreferenceDataset:
    """SQLite-backed store for feedback events, preference pairs, and taste models."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # -- lifecycle -----------------------------------------------------------

    def init(self) -> None:
        """Create tables if they don't already exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # -- feedback events -----------------------------------------------------

    def log_feedback(self, event: FeedbackEvent) -> None:
        """Persist a single feedback event."""
        ts = event.timestamp if event.timestamp else time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback_events
                    (sample_filepath, mix_filepath, session_id, action,
                     rating, recommendation_rank, context_style, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.sample_filepath,
                    event.mix_filepath,
                    event.session_id,
                    event.action,
                    event.rating,
                    event.recommendation_rank,
                    event.context_style,
                    ts,
                ),
            )

    def get_feedback(
        self,
        mix_filepath: str = "",
        session_id: str = "",
        limit: int = 100,
    ) -> list[FeedbackEvent]:
        """Retrieve feedback events, optionally filtered by mix or session."""
        clauses: list[str] = []
        params: list[object] = []
        if mix_filepath:
            clauses.append("mix_filepath = ?")
            params.append(mix_filepath)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT sample_filepath, mix_filepath, session_id, action,
                   rating, recommendation_rank, context_style, timestamp
            FROM feedback_events
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            FeedbackEvent(
                sample_filepath=r[0],
                mix_filepath=r[1],
                session_id=r[2],
                action=r[3],
                rating=r[4],
                recommendation_rank=r[5],
                context_style=r[6],
                timestamp=r[7],
            )
            for r in rows
        ]

    # -- preference pair construction ----------------------------------------

    def build_pairs(self, session_id: str = "") -> list[PreferencePair]:
        """Construct preference pairs from feedback events.

        Algorithm:
        1. Group events by session_id.
        2. Within each session, compute the strongest action per sample.
        3. For every ordered pair (A, B) where strength_A > strength_B,
           emit a PreferencePair with strength = strength_A - strength_B.
        4. Deduplicate: keep the strongest preference for each (preferred, rejected) pair.
        5. Persist new pairs to the preference_pairs table.
        """
        events = self._load_events_for_pairing(session_id)
        if not events:
            return []

        # Group by session
        sessions: dict[str, list[FeedbackEvent]] = {}
        for ev in events:
            sessions.setdefault(ev.session_id, []).append(ev)

        all_pairs: list[PreferencePair] = []
        now = time.time()

        for _sid, session_events in sessions.items():
            # Best strength per sample in this session
            best: dict[str, tuple[float, FeedbackEvent]] = {}
            for ev in session_events:
                s = _event_strength(ev)
                key = ev.sample_filepath
                if key not in best or s > best[key][0]:
                    best[key] = (s, ev)

            samples = list(best.keys())
            # Build all ordered pairs where A is strictly preferred over B
            seen: dict[tuple[str, str], float] = {}
            for i, a in enumerate(samples):
                sa, ev_a = best[a]
                for j, b in enumerate(samples):
                    if i == j:
                        continue
                    sb, _ev_b = best[b]
                    if sa > sb:
                        pair_key = (a, b)
                        strength = round(sa - sb, 4)
                        if pair_key not in seen or strength > seen[pair_key]:
                            seen[pair_key] = strength

            for (pref, rej), strength in seen.items():
                ev_pref = best[pref][1]
                pair = PreferencePair(
                    preferred_filepath=pref,
                    rejected_filepath=rej,
                    mix_filepath=ev_pref.mix_filepath,
                    context_style=ev_pref.context_style,
                    strength=strength,
                    timestamp=now,
                )
                all_pairs.append(pair)

        # Persist
        self._store_pairs(all_pairs)
        return all_pairs

    def get_training_data(self, min_pairs: int = 10) -> list[PreferencePair]:
        """Return all stored preference pairs if the total count meets the minimum."""
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM preference_pairs"
            ).fetchone()[0]
            if count < min_pairs:
                return []
            rows = conn.execute(
                """
                SELECT preferred_filepath, rejected_filepath, mix_filepath,
                       context_style, strength, timestamp
                FROM preference_pairs
                ORDER BY timestamp DESC
                """
            ).fetchall()
        return [
            PreferencePair(
                preferred_filepath=r[0],
                rejected_filepath=r[1],
                mix_filepath=r[2],
                context_style=r[3],
                strength=r[4],
                timestamp=r[5],
            )
            for r in rows
        ]

    # -- taste model persistence ---------------------------------------------

    def save_taste_model(self, model: UserTasteModel) -> None:
        """Persist a trained taste model (upsert by user_id)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO taste_models (user_id, model_data, model_version, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    model_data = excluded.model_data,
                    model_version = excluded.model_version,
                    updated_at = excluded.updated_at
                """,
                (
                    model.user_id,
                    model.to_json(),
                    model.model_version,
                    time.time(),
                ),
            )

    def load_taste_model(self, user_id: str = "default") -> UserTasteModel | None:
        """Load a taste model by user_id, or return None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT model_data FROM taste_models WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return UserTasteModel.from_dict(data)

    # -- internals -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _load_events_for_pairing(
        self, session_id: str = ""
    ) -> list[FeedbackEvent]:
        clause = "WHERE session_id = ?" if session_id else ""
        params: list[object] = [session_id] if session_id else []
        query = f"""
            SELECT sample_filepath, mix_filepath, session_id, action,
                   rating, recommendation_rank, context_style, timestamp
            FROM feedback_events
            {clause}
            ORDER BY timestamp ASC
        """
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            FeedbackEvent(
                sample_filepath=r[0],
                mix_filepath=r[1],
                session_id=r[2],
                action=r[3],
                rating=r[4],
                recommendation_rank=r[5],
                context_style=r[6],
                timestamp=r[7],
            )
            for r in rows
        ]

    def _store_pairs(self, pairs: list[PreferencePair]) -> None:
        if not pairs:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO preference_pairs
                    (preferred_filepath, rejected_filepath, mix_filepath,
                     context_style, strength, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        p.preferred_filepath,
                        p.rejected_filepath,
                        p.mix_filepath,
                        p.context_style,
                        p.strength,
                        p.timestamp,
                    )
                    for p in pairs
                ],
            )
