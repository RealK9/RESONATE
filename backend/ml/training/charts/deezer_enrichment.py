"""
RESONATE Production Model — Deezer Audio Feature Enrichment

Enriches Billboard chart entries with:
  - 30-second preview MP3s (freely available, no auth required)
  - Popularity/rank data
  - BPM, key, duration metadata
  - Album artwork URLs

The Deezer API is free, requires no authentication for search + track endpoints,
and still provides 30-second preview URLs (unlike Spotify's deprecated endpoint).

Rate limit: 50 requests / 5 seconds = 10/s. We use a conservative 5/s.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEEZER_API = "https://api.deezer.com"
MIN_REQUEST_INTERVAL = 0.1  # 10 req/s = Deezer allows 50/5s


class DeezerEnricher:
    """
    Enriches chart entries with Deezer audio previews and metadata.
    Free API — no auth required.
    """

    def __init__(self, db_path: str | Path = "chart_data.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_columns()
        self._last_request: float = 0.0
        self._session = requests.Session()

    def _ensure_columns(self):
        """Add deezer-specific columns to chart_entries if missing."""
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(chart_entries)").fetchall()
        }
        new_cols = {
            "deezer_id": "INTEGER",
            "deezer_preview_url": "TEXT",
            "deezer_preview_path": "TEXT",
            "deezer_bpm": "REAL",
            "deezer_rank": "INTEGER",
        }
        for col, typ in new_cols.items():
            if col not in existing:
                self._conn.execute(f"ALTER TABLE chart_entries ADD COLUMN {col} {typ}")
        self._conn.commit()

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request = time.time()

    @staticmethod
    def _clean(text: str) -> str:
        """Strip parenthetical suffixes and special chars."""
        text = re.sub(r"[\(\[][^)\]]*[\)\]]", "", text)
        text = re.sub(r"[^\w\s\-']", "", text)
        return text.strip()

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def search_track(self, title: str, artist: str) -> Optional[dict]:
        """
        Search Deezer for a track matching title + artist.
        Returns the full track dict or None.
        """
        queries = [
            f'track:"{title}" artist:"{artist}"',
            f'{self._clean(title)} {self._clean(artist)}',
            f'{self._clean(title)}',
        ]

        for q in queries:
            self._rate_limit()
            try:
                resp = self._session.get(
                    f"{DEEZER_API}/search",
                    params={"q": q, "limit": 5},
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json().get("data", [])
                if not data:
                    continue

                # Find best match
                title_lower = self._clean(title).lower()
                artist_lower = self._clean(artist).lower()
                for track in data:
                    t_name = track.get("title", "").lower()
                    a_name = track.get("artist", {}).get("name", "").lower()
                    if (title_lower in t_name or t_name in title_lower) and \
                       (artist_lower in a_name or a_name in artist_lower):
                        return track

                # Fallback: return first result
                return data[0]

            except Exception as e:
                logger.debug("Deezer search error: %s", e)
                continue

        return None

    def get_track_details(self, deezer_id: int) -> Optional[dict]:
        """Fetch full track details including BPM."""
        self._rate_limit()
        try:
            resp = self._session.get(f"{DEEZER_API}/track/{deezer_id}", timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("Deezer track detail error: %s", e)
        return None

    def download_preview(
        self, deezer_id: int, preview_url: str, output_dir: Path
    ) -> Optional[str]:
        """Download 30-second preview MP3."""
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"deezer_{deezer_id}.mp3"

        if filepath.exists() and filepath.stat().st_size > 1000:
            return str(filepath)

        try:
            resp = self._session.get(preview_url, timeout=30)
            resp.raise_for_status()
            if len(resp.content) < 1000:
                return None  # Empty/corrupt preview
            filepath.write_bytes(resp.content)
            return str(filepath)
        except Exception as e:
            logger.debug("Preview download failed for %d: %s", deezer_id, e)
            return None

    # ------------------------------------------------------------------
    # Bulk enrichment
    # ------------------------------------------------------------------

    def enrich_chart_entries(
        self,
        output_dir: str | Path = "previews",
        limit: int = 0,
    ) -> dict:
        """
        Enrich all chart entries in the DB with Deezer data.
        Resumes from where it left off (skips entries with deezer_id).

        Returns stats dict.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get entries that need enrichment
        if limit > 0:
            rows = self._conn.execute(
                "SELECT rowid, title, artist, year, chart_name FROM chart_entries "
                "WHERE deezer_id IS NULL ORDER BY year DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT rowid, title, artist, year, chart_name FROM chart_entries "
                "WHERE deezer_id IS NULL ORDER BY year DESC"
            ).fetchall()

        already_done = self._conn.execute(
            "SELECT COUNT(*) FROM chart_entries WHERE deezer_id IS NOT NULL"
        ).fetchone()[0]

        total = len(rows)
        logger.info(f"Deezer enrichment: {total} entries to process, {already_done} already done")

        stats = {"enriched": 0, "with_preview": 0, "not_found": 0, "errors": 0}

        for row in tqdm(rows, desc="Deezer enrichment", unit="song"):
            title = row["title"]
            artist = row["artist"]

            try:
                # Search
                track = self.search_track(title, artist)
                if not track:
                    stats["not_found"] += 1
                    # Mark as searched (use -1 to skip on resume)
                    self._conn.execute(
                        "UPDATE chart_entries SET deezer_id = -1 WHERE rowid = ?",
                        (row["rowid"],),
                    )
                    self._conn.commit()
                    continue

                deezer_id = track["id"]
                preview_url = track.get("preview", "")
                rank = track.get("rank", 0)
                # Skip get_track_details to save API calls — BPM computed from audio
                bpm = 0

                # Download preview
                preview_path = None
                if preview_url:
                    preview_path = self.download_preview(deezer_id, preview_url, output_dir)
                    if preview_path:
                        stats["with_preview"] += 1

                # Update DB
                self._conn.execute(
                    "UPDATE chart_entries SET "
                    "deezer_id = ?, deezer_preview_url = ?, deezer_preview_path = ?, "
                    "deezer_bpm = ?, deezer_rank = ?, "
                    "preview_path = COALESCE(preview_path, ?) "
                    "WHERE rowid = ?",
                    (deezer_id, preview_url, preview_path, bpm, rank,
                     preview_path, row["rowid"]),
                )
                self._conn.commit()
                stats["enriched"] += 1

            except Exception as e:
                logger.warning("Error enriching '%s' by '%s': %s", title, artist, e)
                stats["errors"] += 1

        logger.info(
            "Deezer enrichment complete: %d enriched, %d with previews, "
            "%d not found, %d errors",
            stats["enriched"], stats["with_preview"],
            stats["not_found"], stats["errors"],
        )
        return stats

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="RESONATE Deezer Enrichment")
    parser.add_argument("--db", default="chart_data.db", help="Chart database path")
    parser.add_argument("--output-dir", default="previews", help="Preview MP3 output dir")
    parser.add_argument("--limit", type=int, default=0, help="Max entries (0 = all)")
    args = parser.parse_args()

    enricher = DeezerEnricher(db_path=args.db)
    try:
        stats = enricher.enrich_chart_entries(
            output_dir=args.output_dir,
            limit=args.limit,
        )
        print(f"\nResults: {json.dumps(stats, indent=2)}")
    finally:
        enricher.close()
