from __future__ import annotations
"""
RESONATE Production Model — Spotify Audio Feature Enrichment

Takes Billboard chart entries and enriches them with Spotify audio features
(key, tempo, energy, valence, etc.) and 30-second preview MP3s.

Uses the Spotipy library with client-credentials authentication.
Rate-limited with resume capability for reliable bulk enrichment.
"""

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

from .billboard_scraper import ChartEntry

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SpotifyFeatures:
    """Audio features and metadata retrieved from the Spotify Web API."""

    track_id: str
    key: int                    # 0-11 pitch class notation
    mode: int                   # 0 = minor, 1 = major
    tempo: float                # BPM
    time_signature: int         # beats per bar (3, 4, 5, …)
    duration_ms: int
    danceability: float         # 0.0 – 1.0
    energy: float               # 0.0 – 1.0
    speechiness: float          # 0.0 – 1.0
    acousticness: float         # 0.0 – 1.0
    instrumentalness: float     # 0.0 – 1.0
    liveness: float             # 0.0 – 1.0
    valence: float              # 0.0 – 1.0  (musical positiveness)
    loudness: float             # dB (typically -60 to 0)
    preview_url: Optional[str] = None
    preview_path: Optional[str] = None


# ---------------------------------------------------------------------------
# SpotifyEnricher
# ---------------------------------------------------------------------------

class SpotifyEnricher:
    """
    Enriches ChartEntry records with Spotify audio features and
    optional 30-second preview MP3 downloads.
    """

    # Spotify rate limit: be conservative to avoid 24hr bans.
    # New apps get lower limits. 1 req / 500ms = 120/min, well under threshold.
    _MIN_REQUEST_INTERVAL_S = 0.5

    # Feature keys we extract from the Spotify audio-features endpoint
    _FEATURE_KEYS = [
        "key", "mode", "tempo", "time_signature", "duration_ms",
        "danceability", "energy", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "loudness",
    ]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        db_path: str | Path = "chart_data.db",
    ) -> None:
        """
        Authenticate with the Spotify Web API via client-credentials flow.

        Credentials can be passed explicitly or read from the environment
        variables ``SPOTIPY_CLIENT_ID`` / ``SPOTIPY_CLIENT_SECRET``.
        """
        cid = client_id or os.environ.get("SPOTIPY_CLIENT_ID")
        csecret = client_secret or os.environ.get("SPOTIPY_CLIENT_SECRET")
        if not cid or not csecret:
            raise ValueError(
                "Spotify credentials required. Pass client_id/client_secret "
                "or set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET env vars."
            )

        auth_manager = SpotifyClientCredentials(
            client_id=cid,
            client_secret=csecret,
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_features_table()

        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_features_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS spotify_features (
                track_id            TEXT PRIMARY KEY,
                key                 INTEGER,
                mode                INTEGER,
                tempo               REAL,
                time_signature      INTEGER,
                duration_ms         INTEGER,
                danceability        REAL,
                energy              REAL,
                speechiness         REAL,
                acousticness        REAL,
                instrumentalness    REAL,
                liveness            REAL,
                valence             REAL,
                loudness            REAL,
                preview_url         TEXT,
                preview_path        TEXT
            )
        """)
        self._conn.commit()

    def _rate_limit(self) -> None:
        """Enforce minimum interval between Spotify API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._MIN_REQUEST_INTERVAL_S:
            time.sleep(self._MIN_REQUEST_INTERVAL_S - elapsed)
        self._last_request_time = time.time()

    @staticmethod
    def _clean_query(text: str) -> str:
        """Strip parenthetical suffixes and special characters for search."""
        # Remove anything in parentheses or brackets: "(feat. X)", "[Remix]"
        text = re.sub(r"[\(\[][^)\]]*[\)\]]", "", text)
        # Remove special chars that confuse Spotify search
        text = re.sub(r"[^\w\s\-']", "", text)
        return text.strip()

    def _is_already_enriched(self, title: str, artist: str, chart_name: str) -> Optional[str]:
        """
        Check if a chart entry already has a spotify_id stored in the
        chart_entries table.  Returns the track ID or None.
        """
        row = self._conn.execute(
            "SELECT spotify_id FROM chart_entries WHERE title = ? AND artist = ? AND chart_name = ?",
            (title, artist, chart_name),
        ).fetchone()
        if row and row["spotify_id"]:
            return row["spotify_id"]
        return None

    def _features_cached(self, track_id: str) -> Optional[SpotifyFeatures]:
        """Return cached SpotifyFeatures from DB if available."""
        row = self._conn.execute(
            "SELECT * FROM spotify_features WHERE track_id = ?",
            (track_id,),
        ).fetchone()
        if row is None:
            return None
        return SpotifyFeatures(**{k: row[k] for k in dict(row).keys()})

    def _save_features(self, features: SpotifyFeatures) -> None:
        d = asdict(features)
        cols = ", ".join(d.keys())
        placeholders = ", ".join(["?"] * len(d))
        self._conn.execute(
            f"INSERT OR REPLACE INTO spotify_features ({cols}) VALUES ({placeholders})",
            tuple(d.values()),
        )
        self._conn.commit()

    def _update_chart_entry(
        self,
        title: str,
        artist: str,
        chart_name: str,
        spotify_id: str,
        features_dict: dict,
    ) -> None:
        """Write spotify_id and audio features back to chart_entries."""
        # Update spotify_id always
        self._conn.execute(
            "UPDATE chart_entries SET spotify_id = ? WHERE title = ? AND artist = ? AND chart_name = ?",
            (spotify_id, title, artist, chart_name),
        )
        # Update individual feature columns if we have data
        if features_dict:
            feature_cols = [
                "key", "mode", "tempo", "time_signature", "duration_ms",
                "danceability", "energy", "speechiness", "acousticness",
                "instrumentalness", "liveness", "valence", "loudness",
            ]
            set_parts = []
            values = []
            for col in feature_cols:
                if col in features_dict and features_dict[col] is not None:
                    set_parts.append(f"{col} = ?")
                    values.append(features_dict[col])
            if "preview_url" in features_dict:
                set_parts.append("preview_url = ?")
                values.append(features_dict["preview_url"])
            if set_parts:
                values.extend([title, artist, chart_name])
                self._conn.execute(
                    f"UPDATE chart_entries SET {', '.join(set_parts)} "
                    f"WHERE title = ? AND artist = ? AND chart_name = ?",
                    values,
                )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_track(self, title: str, artist: str) -> Optional[str]:
        """
        Search Spotify for a track matching *title* and *artist*.

        Tries progressively looser queries:
          1. Exact title + artist
          2. Cleaned title + artist (no parenthetical suffixes)
          3. Cleaned title only (no artist filter)

        Returns the Spotify track ID or None.
        """
        strategies = [
            f'track:"{title}" artist:"{artist}"',
            f'track:"{self._clean_query(title)}" artist:"{self._clean_query(artist)}"',
            f'track:"{self._clean_query(title)}" {self._clean_query(artist)}',
            f'{self._clean_query(title)} {self._clean_query(artist)}',
        ]

        for query in strategies:
            self._rate_limit()
            try:
                results = self.sp.search(q=query, type="track", limit=1)
                items = results.get("tracks", {}).get("items", [])
                if items:
                    return items[0]["id"]
            except spotipy.exceptions.SpotifyException as exc:
                logger.warning("Spotify search error for '%s': %s", query, exc)
                # Back off on rate-limit errors
                if exc.http_status == 429:
                    retry_after = int(exc.headers.get("Retry-After", 5))
                    logger.warning("Rate limited. Sleeping %ds.", retry_after)
                    time.sleep(retry_after)
            except Exception as exc:
                logger.warning("Unexpected search error: %s", exc)

        return None

    def get_audio_features(self, track_id: str) -> Optional[dict]:
        """
        Fetch Spotify audio features for a single track.

        NOTE: As of Nov 2024, Spotify deprecated the audio-features endpoint
        for new developer apps.  This method is kept for backwards compatibility
        but will gracefully return None on 403 errors.
        """
        self._rate_limit()
        try:
            results = self.sp.audio_features([track_id])
            if not results or results[0] is None:
                return None
            raw = results[0]
            return {k: raw[k] for k in self._FEATURE_KEYS if k in raw}
        except spotipy.exceptions.SpotifyException as exc:
            if exc.http_status == 403:
                # Deprecated endpoint — silently skip
                return None
            if exc.http_status == 429:
                retry_after = int(exc.headers.get("Retry-After", 5))
                logger.warning("Rate limited on audio_features. Sleeping %ds.", retry_after)
                time.sleep(retry_after)
                return self.get_audio_features(track_id)
            logger.warning("Audio features error for %s: %s", track_id, exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected audio_features error: %s", exc)
            return None

    def get_track_metadata(self, track_id: str) -> Optional[dict]:
        """
        Fetch track metadata from the /tracks endpoint.

        Returns popularity, duration_ms, explicit, release_date, album_name.
        This works for all apps (unlike audio_features which is deprecated).
        """
        self._rate_limit()
        try:
            track = self.sp.track(track_id)
            if not track:
                return None
            album = track.get("album", {})
            return {
                "popularity": track.get("popularity", 0),
                "duration_ms": track.get("duration_ms", 0),
                "explicit": track.get("explicit", False),
                "release_date": album.get("release_date", ""),
                "album_name": album.get("name", ""),
                "preview_url": track.get("preview_url"),
            }
        except spotipy.exceptions.SpotifyException as exc:
            if exc.http_status == 429:
                retry_after = int(exc.headers.get("Retry-After", 5))
                logger.warning("Rate limited on track endpoint. Sleeping %ds.", retry_after)
                time.sleep(retry_after)
                return self.get_track_metadata(track_id)
            logger.warning("Track metadata error for %s: %s", track_id, exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected track metadata error: %s", exc)
            return None

    def get_preview_url(self, track_id: str) -> Optional[str]:
        """
        Return the 30-second preview MP3 URL for a track, or None if
        no preview is available.
        """
        self._rate_limit()
        try:
            track = self.sp.track(track_id)
            return track.get("preview_url")
        except spotipy.exceptions.SpotifyException as exc:
            if exc.http_status == 429:
                retry_after = int(exc.headers.get("Retry-After", 5))
                logger.warning("Rate limited on track endpoint. Sleeping %ds.", retry_after)
                time.sleep(retry_after)
                return self.get_preview_url(track_id)
            logger.warning("Track endpoint error for %s: %s", track_id, exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected track endpoint error: %s", exc)
            return None

    def download_preview(
        self,
        track_id: str,
        output_dir: str | Path = "previews",
    ) -> Optional[str]:
        """
        Download the 30-second preview MP3 for *track_id*.

        Returns the local file path, or None if no preview is available
        or the download fails.
        """
        preview_url = self.get_preview_url(track_id)
        if not preview_url:
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{track_id}.mp3"

        # Skip if already downloaded
        if filepath.exists() and filepath.stat().st_size > 0:
            return str(filepath)

        try:
            resp = requests.get(preview_url, timeout=30)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
            return str(filepath)
        except Exception as exc:
            logger.warning("Preview download failed for %s: %s", track_id, exc)
            return None

    def enrich_chart_entries(
        self,
        entries: list[ChartEntry],
        output_dir: str | Path = "previews",
        download_previews: bool = True,
    ) -> list[ChartEntry]:
        """
        Enrich a list of ChartEntry records with Spotify data.

        For each entry:
          1. Search Spotify for a matching track
          2. Fetch audio features
          3. Optionally download the 30-second preview MP3

        Features and preview paths are persisted to SQLite.
        Already-enriched entries are skipped (resume capability).

        Returns the list of entries with spotify_id and spotify_features
        populated where available.
        """
        output_dir = Path(output_dir)
        enriched = 0
        skipped = 0
        not_found = 0

        for entry in tqdm(entries, desc="Enriching chart entries", unit="song"):
            # --- Resume: skip already-enriched ---
            cached_id = self._is_already_enriched(entry.title, entry.artist, entry.chart_name)
            if cached_id:
                entry.spotify_id = cached_id
                cached_feat = self._features_cached(cached_id)
                if cached_feat:
                    entry.spotify_features = asdict(cached_feat)
                skipped += 1
                continue

            # --- Step 1: Search ---
            track_id = self.search_track(entry.title, entry.artist)
            if not track_id:
                not_found += 1
                continue

            entry.spotify_id = track_id

            # --- Step 2: Track metadata (always works, unlike audio_features) ---
            metadata = self.get_track_metadata(track_id)
            features_dict = {}
            if metadata:
                features_dict = metadata.copy()
                # Store popularity in chart_entries
                self._conn.execute(
                    "UPDATE chart_entries SET popularity = ?, duration_ms = ? "
                    "WHERE title = ? AND artist = ? AND chart_name = ?",
                    (metadata.get("popularity", 0), metadata.get("duration_ms", 0),
                     entry.title, entry.artist, entry.chart_name),
                )
                self._conn.commit()

            # --- Step 3: Audio features (may fail with 403 on new apps) ---
            cached_feat = self._features_cached(track_id)
            if cached_feat:
                features_dict.update(asdict(cached_feat))
            else:
                raw_features = self.get_audio_features(track_id)
                if raw_features:
                    features_dict.update(raw_features)
                    preview_url = metadata.get("preview_url") if metadata else None
                    sf = SpotifyFeatures(
                        track_id=track_id,
                        preview_url=preview_url,
                        preview_path=None,
                        **raw_features,
                    )
                    self._save_features(sf)

            entry.spotify_features = features_dict

            # Write back to chart_entries table
            self._update_chart_entry(
                entry.title, entry.artist, entry.chart_name,
                track_id, features_dict,
            )
            enriched += 1

        logger.info(
            "Enrichment complete: %d enriched, %d skipped (cached), %d not found on Spotify.",
            enriched,
            skipped,
            not_found,
        )
        return entries

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from .billboard_scraper import BillboardScraper

    parser = argparse.ArgumentParser(
        description="RESONATE Spotify Enrichment — enrich Billboard chart entries",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="chart_data.db",
        help="SQLite database with chart_entries table (default: chart_data.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="previews",
        help="Directory for 30-second preview MP3s (default: previews/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max entries to enrich in this run (default: 50, 0 = all)",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip downloading preview MP3s",
    )
    args = parser.parse_args()

    # Load chart entries from DB
    scraper = BillboardScraper.load_from_db(args.db)
    all_songs = scraper.get_unique_songs()
    logger.info("Loaded %d unique songs from database.", len(all_songs))

    # Convert tuples to ChartEntry objects
    entries = [
        ChartEntry(
            title=title,
            artist=artist,
            peak_position=peak,
            weeks_on_chart=0,  # Not stored in tuple form
            year=year,
            chart_name=chart,
        )
        for title, artist, peak, year, chart in all_songs
    ]

    # Apply limit
    if args.limit > 0:
        entries = entries[: args.limit]
        logger.info("Limiting to %d entries for this run.", args.limit)

    # Enrich
    enricher = SpotifyEnricher(db_path=args.db)
    try:
        enriched = enricher.enrich_chart_entries(
            entries,
            output_dir=args.output_dir,
            download_previews=not args.no_previews,
        )

        # Summary
        with_features = sum(1 for e in enriched if e.spotify_features)
        with_id = sum(1 for e in enriched if e.spotify_id)
        logger.info("Results: %d/%d matched on Spotify, %d with audio features.",
                     with_id, len(enriched), with_features)

        # Show a few examples
        logger.info("Sample enriched entries:")
        for entry in enriched[:5]:
            if entry.spotify_features:
                feats = entry.spotify_features
                logger.info(
                    "  %s — %s | tempo=%.0f bpm, key=%s, energy=%.2f, valence=%.2f",
                    entry.artist,
                    entry.title,
                    feats.get("tempo", 0),
                    feats.get("key", "?"),
                    feats.get("energy", 0),
                    feats.get("valence", 0),
                )
    finally:
        enricher.close()
        scraper.close()
