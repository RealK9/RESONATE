"""
RESONATE Production Model — Chart Analysis.

Analyzes chart data to extract:
  - Per-decade feature distributions (how BPM/key/energy changed over time)
  - Per-genre feature signatures (what makes a country hit vs. hip-hop hit)
  - Cross-genre hit patterns (what audio features correlate with crossover success)
  - Key/tempo/energy trends for each era

This data feeds the Era head and Chart Potential head during training.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecadeProfile:
    """Statistical profile of music from a specific decade."""
    decade: int  # e.g. 1980
    count: int = 0

    # BPM
    bpm_mean: float = 120.0
    bpm_std: float = 20.0
    bpm_median: float = 120.0

    # Key distribution (12 pitch classes)
    key_distribution: list[float] = field(default_factory=lambda: [1/12]*12)

    # Mode (major vs minor ratio)
    major_ratio: float = 0.6

    # Energy/Danceability/Valence means
    energy_mean: float = 0.5
    energy_std: float = 0.2
    danceability_mean: float = 0.5
    danceability_std: float = 0.2
    valence_mean: float = 0.5
    valence_std: float = 0.2

    # Loudness (LUFS/dB)
    loudness_mean: float = -10.0
    loudness_std: float = 3.0

    # Acousticness / Instrumentalness
    acousticness_mean: float = 0.3
    instrumentalness_mean: float = 0.1

    # Duration
    duration_mean: float = 210.0  # seconds
    duration_std: float = 60.0

    def to_feature_vector(self) -> np.ndarray:
        """Convert to numeric feature vector for training."""
        return np.array([
            self.bpm_mean / 200.0,  # normalize
            self.bpm_std / 50.0,
            self.major_ratio,
            self.energy_mean,
            self.danceability_mean,
            self.valence_mean,
            (self.loudness_mean + 60) / 60.0,  # normalize -20..0 → 0..1
            self.acousticness_mean,
            self.instrumentalness_mean,
            self.duration_mean / 600.0,
        ], dtype=np.float32)


@dataclass
class GenreProfile:
    """Statistical profile of a genre based on chart data."""
    genre: str
    count: int = 0

    bpm_mean: float = 120.0
    bpm_std: float = 20.0
    energy_mean: float = 0.5
    danceability_mean: float = 0.5
    valence_mean: float = 0.5
    acousticness_mean: float = 0.3
    instrumentalness_mean: float = 0.1
    speechiness_mean: float = 0.1
    loudness_mean: float = -10.0

    # Key tendencies
    key_distribution: list[float] = field(default_factory=lambda: [1/12]*12)
    major_ratio: float = 0.6

    # Chart performance
    avg_peak_position: float = 50.0
    avg_weeks_on_chart: float = 10.0

    def to_feature_vector(self) -> np.ndarray:
        """Convert to numeric feature vector."""
        return np.array([
            self.bpm_mean / 200.0,
            self.energy_mean,
            self.danceability_mean,
            self.valence_mean,
            self.acousticness_mean,
            self.instrumentalness_mean,
            self.speechiness_mean,
            (self.loudness_mean + 60) / 60.0,
            self.major_ratio,
            self.avg_peak_position / 100.0,
        ], dtype=np.float32)


class ChartAnalyzer:
    """Analyze chart data for era/genre intelligence."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._decade_profiles: dict[int, DecadeProfile] = {}
        self._genre_profiles: dict[str, GenreProfile] = {}

    def analyze(self):
        """Run full analysis on the chart database."""
        if not self.db_path.exists():
            logger.warning(f"Chart database not found: {self.db_path}")
            return

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        self._analyze_decades(conn)
        self._analyze_genres(conn)
        self._analyze_crossover(conn)

        conn.close()
        logger.info("Chart analysis complete")

    def _analyze_decades(self, conn):
        """Compute per-decade statistical profiles."""
        logger.info("Analyzing decade trends...")

        for decade_start in range(1950, 2030, 10):
            decade_end = decade_start + 9

            rows = conn.execute("""
                SELECT tempo, key, mode, energy, danceability, valence,
                       loudness, acousticness, instrumentalness, duration_ms
                FROM chart_entries
                WHERE year >= ? AND year <= ?
                AND tempo IS NOT NULL AND tempo > 0
            """, (decade_start, decade_end)).fetchall()

            if not rows:
                continue

            profile = DecadeProfile(decade=decade_start, count=len(rows))

            # BPM
            bpms = [r["tempo"] for r in rows if r["tempo"] and r["tempo"] > 0]
            if bpms:
                profile.bpm_mean = float(np.mean(bpms))
                profile.bpm_std = float(np.std(bpms))
                profile.bpm_median = float(np.median(bpms))

            # Key distribution
            keys = [r["key"] for r in rows if r["key"] is not None and 0 <= r["key"] <= 11]
            if keys:
                key_counts = np.bincount(keys, minlength=12).astype(float)
                profile.key_distribution = (key_counts / key_counts.sum()).tolist()

            # Mode
            modes = [r["mode"] for r in rows if r["mode"] is not None]
            if modes:
                profile.major_ratio = sum(1 for m in modes if m == 1) / len(modes)

            # Audio features
            for feat_name in ["energy", "danceability", "valence", "loudness",
                             "acousticness", "instrumentalness"]:
                vals = [r[feat_name] for r in rows if r[feat_name] is not None]
                if vals:
                    setattr(profile, f"{feat_name}_mean", float(np.mean(vals)))
                    if hasattr(profile, f"{feat_name}_std"):
                        setattr(profile, f"{feat_name}_std", float(np.std(vals)))

            # Duration
            durations = [r["duration_ms"] / 1000.0 for r in rows
                        if r["duration_ms"] and r["duration_ms"] > 0]
            if durations:
                profile.duration_mean = float(np.mean(durations))
                profile.duration_std = float(np.std(durations))

            self._decade_profiles[decade_start] = profile
            logger.info(
                f"  {decade_start}s: {profile.count} songs, "
                f"BPM={profile.bpm_mean:.0f}±{profile.bpm_std:.0f}, "
                f"energy={profile.energy_mean:.2f}, "
                f"major={profile.major_ratio:.0%}"
            )

    def _analyze_genres(self, conn):
        """Compute per-genre statistical profiles."""
        logger.info("Analyzing genre signatures...")

        charts = conn.execute(
            "SELECT DISTINCT chart_name FROM chart_entries"
        ).fetchall()

        genre_map = {
            "hot-100": "pop",
            "r-b-hip-hop-songs": "r&b/hip-hop",
            "country-songs": "country",
            "latin-songs": "latin",
            "dance-electronic-songs": "electronic",
            "rock-songs": "rock",
            "hot-r-and-b-hip-hop-songs": "r&b/hip-hop",
            "hot-country-songs": "country",
            "hot-latin-songs": "latin",
            "hot-dance-electronic-songs": "electronic",
            "hot-rock-alternative-songs": "rock",
        }

        for chart_row in charts:
            chart_name = chart_row["chart_name"]
            genre = genre_map.get(chart_name, chart_name)

            rows = conn.execute("""
                SELECT tempo, key, mode, energy, danceability, valence,
                       loudness, acousticness, instrumentalness, speechiness,
                       peak_position, weeks_on_chart
                FROM chart_entries
                WHERE chart_name = ? AND tempo IS NOT NULL AND tempo > 0
            """, (chart_name,)).fetchall()

            if not rows:
                continue

            profile = GenreProfile(genre=genre, count=len(rows))

            # BPM
            bpms = [r["tempo"] for r in rows if r["tempo"] and r["tempo"] > 0]
            if bpms:
                profile.bpm_mean = float(np.mean(bpms))
                profile.bpm_std = float(np.std(bpms))

            # Features
            for feat in ["energy", "danceability", "valence", "loudness",
                        "acousticness", "instrumentalness", "speechiness"]:
                vals = [r[feat] for r in rows if r[feat] is not None]
                if vals:
                    setattr(profile, f"{feat}_mean", float(np.mean(vals)))

            # Key distribution
            keys = [r["key"] for r in rows if r["key"] is not None and 0 <= r["key"] <= 11]
            if keys:
                key_counts = np.bincount(keys, minlength=12).astype(float)
                profile.key_distribution = (key_counts / key_counts.sum()).tolist()

            # Mode
            modes = [r["mode"] for r in rows if r["mode"] is not None]
            if modes:
                profile.major_ratio = sum(1 for m in modes if m == 1) / len(modes)

            # Chart performance
            peaks = [r["peak_position"] for r in rows if r["peak_position"]]
            weeks = [r["weeks_on_chart"] for r in rows if r["weeks_on_chart"]]
            if peaks:
                profile.avg_peak_position = float(np.mean(peaks))
            if weeks:
                profile.avg_weeks_on_chart = float(np.mean(weeks))

            self._genre_profiles[genre] = profile
            logger.info(
                f"  {genre}: {profile.count} songs, "
                f"BPM={profile.bpm_mean:.0f}, "
                f"energy={profile.energy_mean:.2f}, "
                f"dance={profile.danceability_mean:.2f}"
            )

    def _analyze_crossover(self, conn):
        """Identify what makes songs cross genre boundaries."""
        logger.info("Analyzing crossover patterns...")

        # Find songs appearing on multiple charts
        crossover = conn.execute("""
            SELECT title, artist, COUNT(DISTINCT chart_name) as chart_count,
                   MIN(peak_position) as best_peak,
                   AVG(energy) as energy, AVG(danceability) as dance,
                   AVG(valence) as valence
            FROM chart_entries
            WHERE energy IS NOT NULL
            GROUP BY title, artist
            HAVING chart_count > 1
            ORDER BY chart_count DESC, best_peak ASC
            LIMIT 100
        """).fetchall()

        if crossover:
            energies = [r["energy"] for r in crossover if r["energy"]]
            dances = [r["dance"] for r in crossover if r["dance"]]
            valences = [r["valence"] for r in crossover if r["valence"]]

            logger.info(f"  Found {len(crossover)} crossover hits")
            if energies:
                logger.info(f"  Crossover energy: {np.mean(energies):.2f} (vs typical ~0.50)")
            if dances:
                logger.info(f"  Crossover danceability: {np.mean(dances):.2f}")
            if valences:
                logger.info(f"  Crossover valence: {np.mean(valences):.2f}")

    # ── Getters ──

    def get_decade_profiles(self) -> dict[int, DecadeProfile]:
        return self._decade_profiles

    def get_genre_profiles(self) -> dict[str, GenreProfile]:
        return self._genre_profiles

    def get_era_feature_vectors(self) -> np.ndarray:
        """Get feature vectors for all decades, sorted chronologically."""
        decades = sorted(self._decade_profiles.keys())
        if not decades:
            return np.zeros((8, 10), dtype=np.float32)
        return np.stack([
            self._decade_profiles[d].to_feature_vector() for d in decades
        ])

    def get_genre_feature_vectors(self) -> dict[str, np.ndarray]:
        """Get feature vectors for all genres."""
        return {
            genre: profile.to_feature_vector()
            for genre, profile in self._genre_profiles.items()
        }

    def save_analysis(self, output_path: str):
        """Save analysis results to JSON."""
        output = {
            "decades": {},
            "genres": {},
        }

        for decade, profile in self._decade_profiles.items():
            output["decades"][str(decade)] = {
                "count": profile.count,
                "bpm_mean": profile.bpm_mean,
                "bpm_std": profile.bpm_std,
                "energy_mean": profile.energy_mean,
                "danceability_mean": profile.danceability_mean,
                "valence_mean": profile.valence_mean,
                "loudness_mean": profile.loudness_mean,
                "major_ratio": profile.major_ratio,
                "key_distribution": profile.key_distribution,
                "duration_mean": profile.duration_mean,
            }

        for genre, profile in self._genre_profiles.items():
            output["genres"][genre] = {
                "count": profile.count,
                "bpm_mean": profile.bpm_mean,
                "energy_mean": profile.energy_mean,
                "danceability_mean": profile.danceability_mean,
                "valence_mean": profile.valence_mean,
                "loudness_mean": profile.loudness_mean,
                "speechiness_mean": profile.speechiness_mean,
                "major_ratio": profile.major_ratio,
                "avg_peak_position": profile.avg_peak_position,
                "avg_weeks_on_chart": profile.avg_weeks_on_chart,
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Analysis saved to {output_path}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    db_path = sys.argv[1] if len(sys.argv) > 1 else "~/.resonate/charts/chart_features.db"
    db_path = str(Path(db_path).expanduser())

    analyzer = ChartAnalyzer(db_path)
    analyzer.analyze()

    if analyzer.get_decade_profiles():
        analyzer.save_analysis(str(Path("~/.resonate/charts/chart_analysis.json").expanduser()))
    else:
        print("No chart data found. Run billboard_scraper.py and spotify_enrichment.py first.")
