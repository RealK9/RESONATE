from __future__ import annotations
"""
RESONATE Production Model — Billboard Chart Intelligence Scraper

Scrapes Billboard chart data from 1958-2025 using the billboard.py library.
Consolidates weekly chart entries into unique song records with peak positions,
weeks on chart, and genre chart associations.

Rate-limited with resume capability for reliable large-scale scraping.
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import billboard

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChartEntry:
    """A single consolidated chart entry representing one unique song."""

    title: str
    artist: str
    peak_position: int
    weeks_on_chart: int
    year: int
    chart_name: str  # "hot-100", "r-b-hip-hop-songs", etc.
    spotify_id: Optional[str] = None
    spotify_features: Optional[dict] = field(default=None, repr=False)

    @property
    def dedup_key(self) -> str:
        """Canonical key used for deduplication across weekly snapshots."""
        return f"{self.title.lower().strip()}||{self.artist.lower().strip()}||{self.chart_name}"


# ---------------------------------------------------------------------------
# SQLite persistence helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chart_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL,
    artist          TEXT    NOT NULL,
    peak_position   INTEGER NOT NULL,
    weeks_on_chart  INTEGER NOT NULL,
    year            INTEGER NOT NULL,
    chart_name      TEXT    NOT NULL,
    spotify_id      TEXT,
    spotify_features TEXT,
    UNIQUE(title, artist, chart_name)
);
"""

_CREATE_PROGRESS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scrape_progress (
    chart_name  TEXT NOT NULL,
    last_date   TEXT NOT NULL,
    PRIMARY KEY (chart_name)
);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE_SQL)
    conn.execute(_CREATE_PROGRESS_TABLE_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# BillboardScraper
# ---------------------------------------------------------------------------

class BillboardScraper:
    """
    Scrapes Billboard weekly charts and consolidates entries into unique
    song records.  Supports resume capability and polite rate limiting.
    """

    # Genre charts to scrape alongside Hot 100
    GENRE_CHARTS: dict[str, str] = {
        "r-b-hip-hop-songs": "Hot R&B/Hip-Hop Songs",
        "country-songs": "Hot Country Songs",
        "latin-songs": "Hot Latin Songs",
        "dance-electronic-songs": "Hot Dance/Electronic Songs",
        "rock-songs": "Hot Rock & Alternative Songs",
    }

    def __init__(
        self,
        db_path: str | Path = "chart_data.db",
        delay_seconds: float = 1.5,
    ) -> None:
        self.db_path = Path(db_path)
        self.delay_seconds = delay_seconds
        # In-memory dedup map: dedup_key -> ChartEntry
        self._entries: dict[str, ChartEntry] = {}

        # Initialise database
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        _init_db(self._conn)

        # Preload existing entries from DB so we never lose prior work
        self._load_existing_entries()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_existing_entries(self) -> None:
        """Load previously scraped entries from SQLite into memory."""
        rows = self._conn.execute("SELECT * FROM chart_entries").fetchall()
        for row in rows:
            entry = ChartEntry(
                title=row["title"],
                artist=row["artist"],
                peak_position=row["peak_position"],
                weeks_on_chart=row["weeks_on_chart"],
                year=row["year"],
                chart_name=row["chart_name"],
                spotify_id=row["spotify_id"],
                spotify_features=(
                    json.loads(row["spotify_features"])
                    if row["spotify_features"]
                    else None
                ),
            )
            self._entries[entry.dedup_key] = entry
        if rows:
            logger.info("Loaded %d existing entries from database.", len(rows))

    def _get_last_scraped_date(self, chart_name: str) -> Optional[str]:
        """Return the last successfully scraped date string for a chart."""
        row = self._conn.execute(
            "SELECT last_date FROM scrape_progress WHERE chart_name = ?",
            (chart_name,),
        ).fetchone()
        return row["last_date"] if row else None

    def _set_last_scraped_date(self, chart_name: str, date_str: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO scrape_progress (chart_name, last_date) VALUES (?, ?)",
            (chart_name, date_str),
        )
        self._conn.commit()

    def _merge_entry(
        self,
        title: str,
        artist: str,
        position: int,
        weeks: int,
        year: int,
        chart_name: str,
    ) -> None:
        """Merge a weekly observation into the consolidated entry set."""
        key = f"{title.lower().strip()}||{artist.lower().strip()}||{chart_name}"
        if key in self._entries:
            existing = self._entries[key]
            existing.peak_position = min(existing.peak_position, position)
            existing.weeks_on_chart = max(existing.weeks_on_chart, weeks)
            # Keep the earliest year
            existing.year = min(existing.year, year)
        else:
            self._entries[key] = ChartEntry(
                title=title,
                artist=artist,
                peak_position=position,
                weeks_on_chart=weeks,
                year=year,
                chart_name=chart_name,
            )

    def _scrape_chart(
        self,
        chart_name: str,
        start_year: int,
        end_year: int,
    ) -> None:
        """
        Scrape a single Billboard chart across the given year range.

        Uses billboard.py's ChartData to fetch each week.  Automatically
        walks backwards in time via the chart's ``previousDate`` link.
        Resumes from the last scraped date if a prior run was interrupted.
        """
        # Determine the starting date for scraping
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        last_scraped = self._get_last_scraped_date(chart_name)
        if last_scraped and last_scraped > start_date:
            logger.info(
                "[%s] Resuming — already scraped up to %s. Starting from next week.",
                chart_name,
                last_scraped,
            )
            # Move one week past the last scraped date
            resume_date = datetime.strptime(last_scraped, "%Y-%m-%d") + timedelta(days=7)
            current_date_str = resume_date.strftime("%Y-%m-%d")
        else:
            current_date_str = start_date

        total_weeks_scraped = 0

        while current_date_str <= end_date:
            try:
                logger.info(
                    "[%s] Fetching chart for week of %s …",
                    chart_name,
                    current_date_str,
                )
                chart = billboard.ChartData(chart_name, date=current_date_str)
            except Exception as exc:
                logger.warning(
                    "[%s] Failed to fetch %s: %s. Skipping week.",
                    chart_name,
                    current_date_str,
                    exc,
                )
                # Advance by one week and continue
                next_date = datetime.strptime(current_date_str, "%Y-%m-%d") + timedelta(days=7)
                current_date_str = next_date.strftime("%Y-%m-%d")
                time.sleep(self.delay_seconds)
                continue

            chart_year = int(current_date_str[:4])

            for entry in chart:
                self._merge_entry(
                    title=entry.title,
                    artist=entry.artist,
                    position=entry.peakPos if entry.peakPos else entry.rank,
                    weeks=entry.weeks if entry.weeks else 1,
                    year=chart_year,
                    chart_name=chart_name,
                )

            # Persist resume checkpoint
            self._set_last_scraped_date(chart_name, current_date_str)
            total_weeks_scraped += 1

            if total_weeks_scraped % 50 == 0:
                logger.info(
                    "[%s] Progress — %d weeks scraped, %d unique entries so far.",
                    chart_name,
                    total_weeks_scraped,
                    len(self._entries),
                )
                # Flush to DB periodically
                self._flush_to_db()

            # Advance to next week
            next_date = datetime.strptime(current_date_str, "%Y-%m-%d") + timedelta(days=7)
            current_date_str = next_date.strftime("%Y-%m-%d")

            # Polite delay
            time.sleep(self.delay_seconds)

        logger.info(
            "[%s] Complete — %d weeks scraped.",
            chart_name,
            total_weeks_scraped,
        )

    def _flush_to_db(self) -> None:
        """Upsert all in-memory entries into SQLite."""
        for entry in self._entries.values():
            self._conn.execute(
                """
                INSERT INTO chart_entries
                    (title, artist, peak_position, weeks_on_chart, year, chart_name,
                     spotify_id, spotify_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(title, artist, chart_name) DO UPDATE SET
                    peak_position   = MIN(excluded.peak_position, chart_entries.peak_position),
                    weeks_on_chart  = MAX(excluded.weeks_on_chart, chart_entries.weeks_on_chart),
                    year            = MIN(excluded.year, chart_entries.year),
                    spotify_id      = COALESCE(excluded.spotify_id, chart_entries.spotify_id),
                    spotify_features = COALESCE(excluded.spotify_features, chart_entries.spotify_features)
                """,
                (
                    entry.title,
                    entry.artist,
                    entry.peak_position,
                    entry.weeks_on_chart,
                    entry.year,
                    entry.chart_name,
                    entry.spotify_id,
                    json.dumps(entry.spotify_features) if entry.spotify_features else None,
                ),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_hot_100(
        self,
        start_year: int = 1958,
        end_year: int = 2025,
    ) -> None:
        """
        Scrape Billboard Hot 100 weekly charts from *start_year* to *end_year*.

        Deduplicates across weeks — the same song appearing on multiple weeks
        is consolidated into a single entry with the best peak position and
        total weeks on chart.
        """
        logger.info(
            "Starting Hot 100 scrape: %d–%d", start_year, end_year,
        )
        self._scrape_chart("hot-100", start_year, end_year)
        self._flush_to_db()
        logger.info("Hot 100 scrape complete. %d unique entries total.", len(self._entries))

    def scrape_genre_charts(
        self,
        start_year: int = 1958,
        end_year: int = 2025,
    ) -> None:
        """
        Scrape Billboard genre-specific charts.

        Charts scraped:
          - Hot R&B/Hip-Hop Songs
          - Hot Country Songs
          - Hot Latin Songs
          - Hot Dance/Electronic Songs
          - Hot Rock & Alternative Songs
        """
        for chart_id, chart_label in self.GENRE_CHARTS.items():
            logger.info("Starting genre chart scrape: %s (%s)", chart_label, chart_id)
            self._scrape_chart(chart_id, start_year, end_year)
            self._flush_to_db()
            logger.info(
                "Genre chart '%s' complete. %d unique entries total.",
                chart_label,
                len(self._entries),
            )

    def get_unique_songs(
        self,
    ) -> list[tuple[str, str, int, int, str]]:
        """
        Return all unique songs as a list of tuples:
            (title, artist, peak_position, year, chart_name)
        """
        return [
            (e.title, e.artist, e.peak_position, e.year, e.chart_name)
            for e in self._entries.values()
        ]

    def save_to_json(self, filepath: str | Path = "chart_entries.json") -> None:
        """Export all entries to a JSON file."""
        filepath = Path(filepath)
        data = [asdict(e) for e in self._entries.values()]
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Saved %d entries to %s", len(data), filepath)

    def save_to_db(self, filepath: str | Path | None = None) -> None:
        """
        Persist all entries to SQLite.

        If *filepath* differs from the working database, entries are copied
        to a new database file.  Otherwise the working DB is flushed.
        """
        if filepath is None or Path(filepath) == self.db_path:
            self._flush_to_db()
            logger.info("Flushed %d entries to %s", len(self._entries), self.db_path)
            return

        # Copy to a separate database
        target = Path(filepath)
        target_conn = sqlite3.connect(str(target))
        _init_db(target_conn)
        for entry in self._entries.values():
            target_conn.execute(
                """
                INSERT OR REPLACE INTO chart_entries
                    (title, artist, peak_position, weeks_on_chart, year, chart_name,
                     spotify_id, spotify_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.title,
                    entry.artist,
                    entry.peak_position,
                    entry.weeks_on_chart,
                    entry.year,
                    entry.chart_name,
                    entry.spotify_id,
                    json.dumps(entry.spotify_features) if entry.spotify_features else None,
                ),
            )
        target_conn.commit()
        target_conn.close()
        logger.info("Saved %d entries to %s", len(self._entries), target)

    @classmethod
    def load_from_db(cls, filepath: str | Path) -> "BillboardScraper":
        """
        Create a BillboardScraper instance pre-loaded from an existing
        SQLite database.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Database not found: {filepath}")
        instance = cls(db_path=filepath)
        logger.info(
            "Loaded scraper from %s with %d entries.",
            filepath,
            len(instance._entries),
        )
        return instance

    def close(self) -> None:
        """Flush and close the database connection."""
        self._flush_to_db()
        self._conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RESONATE Billboard Chart Scraper",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1958,
        help="First year to scrape (default: 1958)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last year to scrape (default: 2025)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="chart_data.db",
        help="SQLite database path (default: chart_data.db)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="chart_entries.json",
        help="JSON output path (default: chart_entries.json)",
    )
    parser.add_argument(
        "--genres",
        action="store_true",
        help="Also scrape genre-specific charts",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between requests (default: 1.5)",
    )
    args = parser.parse_args()

    scraper = BillboardScraper(db_path=args.db, delay_seconds=args.delay)

    try:
        # Always scrape Hot 100
        scraper.scrape_hot_100(start_year=args.start_year, end_year=args.end_year)

        # Optionally scrape genre charts
        if args.genres:
            scraper.scrape_genre_charts(start_year=args.start_year, end_year=args.end_year)

        # Export
        scraper.save_to_json(args.json_out)
        scraper.save_to_db()

        songs = scraper.get_unique_songs()
        logger.info("Total unique songs: %d", len(songs))
        logger.info("Sample entries:")
        for title, artist, peak, year, chart in songs[:10]:
            logger.info("  #%d  %s — %s (%d) [%s]", peak, artist, title, year, chart)
    finally:
        scraper.close()
