"""
RESONATE Production Model — Phase H: Music Knowledge Graph

Connects everything the model knows into a unified graph where
sound, theory, history, and taste all live together.

Data sources (all free):
  - MusicBrainz:    35M+ tracks — artist relationships, genres, instruments
  - Wikidata:       Artist influence chains, genre genealogy, instrument taxonomy
  - AcousticBrainz: 4M+ tracks with pre-computed audio features (key, BPM, timbre)
  - Discogs:        15M+ releases — labels, styles, credits, masters
  - AllMusic:       Genre trees, mood tags, similar artists (web data)

The knowledge graph enables:
  - "This sample sounds like 808 State meets Burial" → genre lineage
  - "This chord progression borrows from Coltrane changes" → theory lineage
  - "This production technique was pioneered by Lee Scratch Perry" → history
  - Temporal genre evolution: how did hip-hop production change 1988-2025?
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path.home() / ".resonate" / "datasets" / "phase_h"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MusicEntity:
    """A node in the music knowledge graph."""
    entity_id: str
    entity_type: str  # artist, release, genre, instrument, label, technique
    name: str
    properties: dict = field(default_factory=dict)
    source: str = ""


@dataclass
class MusicRelation:
    """An edge in the music knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str  # influenced_by, genre_of, performed_by, produced_by, etc.
    properties: dict = field(default_factory=dict)
    source: str = ""


# ---------------------------------------------------------------------------
# MusicBrainz (35M+ tracks)
# ---------------------------------------------------------------------------

MUSICBRAINZ_API = "https://musicbrainz.org/ws/2"
MUSICBRAINZ_DUMPS = "https://data.metabrainz.org/pub/musicbrainz/data/fullexport"

class MusicBrainzClient:
    """
    MusicBrainz API client — the world's largest open music database.
    35M+ tracks with artist relationships, genres, instruments.
    Rate limit: 1 req/sec with proper User-Agent.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "musicbrainz"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "RESONATE/1.0 (https://soniqlabs.com)"
        self._last_request = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request = time.time()

    def search_artists(self, query: str, limit: int = 25) -> list[dict]:
        """Search for artists by name."""
        self._rate_limit()
        resp = self._session.get(
            f"{MUSICBRAINZ_API}/artist",
            params={"query": query, "limit": limit, "fmt": "json"},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("artists", [])

    def get_artist(self, mbid: str, inc: str = "genres+tags+artist-rels") -> dict:
        """Get artist details with relationships."""
        self._rate_limit()
        resp = self._session.get(
            f"{MUSICBRAINZ_API}/artist/{mbid}",
            params={"inc": inc, "fmt": "json"},
            timeout=30,
        )
        if resp.status_code != 200:
            return {}
        return resp.json()

    def get_artist_releases(self, mbid: str, limit: int = 100) -> list[dict]:
        """Get all releases by an artist."""
        self._rate_limit()
        resp = self._session.get(
            f"{MUSICBRAINZ_API}/release",
            params={"artist": mbid, "limit": limit, "fmt": "json"},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("releases", [])

    def get_recording(self, mbid: str, inc: str = "genres+tags+artist-credits") -> dict:
        """Get recording details."""
        self._rate_limit()
        resp = self._session.get(
            f"{MUSICBRAINZ_API}/recording/{mbid}",
            params={"inc": inc, "fmt": "json"},
            timeout=30,
        )
        if resp.status_code != 200:
            return {}
        return resp.json()

    def build_artist_graph(self, seed_artists: list[str], depth: int = 2) -> tuple[list[MusicEntity], list[MusicRelation]]:
        """
        Build a knowledge graph starting from seed artists.
        Expands through artist relationships (influences, members, collaborations).
        """
        entities = []
        relations = []
        visited = set()
        queue = [(name, 0) for name in seed_artists]

        while queue:
            artist_name, current_depth = queue.pop(0)
            if artist_name in visited or current_depth > depth:
                continue
            visited.add(artist_name)

            # Search for artist
            results = self.search_artists(artist_name, limit=1)
            if not results:
                continue

            artist = results[0]
            mbid = artist.get("id", "")
            if not mbid:
                continue

            # Get full artist details with relationships
            details = self.get_artist(mbid)
            if not details:
                continue

            # Create entity
            entity = MusicEntity(
                entity_id=f"mb:artist:{mbid}",
                entity_type="artist",
                name=details.get("name", artist_name),
                properties={
                    "mbid": mbid,
                    "type": details.get("type", ""),
                    "country": details.get("country", ""),
                    "begin_year": details.get("life-span", {}).get("begin", ""),
                    "end_year": details.get("life-span", {}).get("end", ""),
                    "genres": [g.get("name", "") for g in details.get("genres", [])],
                    "tags": [t.get("name", "") for t in details.get("tags", [])],
                },
                source="musicbrainz",
            )
            entities.append(entity)

            # Extract relationships
            for rel in details.get("relations", []):
                rel_type = rel.get("type", "")
                target = rel.get("target", {}) if isinstance(rel.get("target"), dict) else {}
                target_artist = rel.get("artist", {})

                if target_artist:
                    target_name = target_artist.get("name", "")
                    target_mbid = target_artist.get("id", "")

                    relations.append(MusicRelation(
                        source_id=f"mb:artist:{mbid}",
                        target_id=f"mb:artist:{target_mbid}",
                        relation_type=rel_type,
                        properties={
                            "direction": rel.get("direction", ""),
                            "attributes": rel.get("attributes", []),
                        },
                        source="musicbrainz",
                    ))

                    # Add to exploration queue
                    if current_depth < depth and target_name not in visited:
                        queue.append((target_name, current_depth + 1))

            logger.info(f"  [{current_depth}] {artist_name}: {len(details.get('relations', []))} relationships")

        logger.info(f"  MusicBrainz graph: {len(entities)} entities, {len(relations)} relations")
        return entities, relations


# ---------------------------------------------------------------------------
# Wikidata (genre genealogy, artist influences)
# ---------------------------------------------------------------------------

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

class WikidataClient:
    """
    Queries Wikidata for music knowledge via SPARQL.
    Genre genealogy, artist influence chains, instrument taxonomy.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "wikidata"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "RESONATE/1.0 (https://soniqlabs.com)"

    def _sparql_query(self, query: str) -> list[dict]:
        """Execute a SPARQL query against Wikidata."""
        time.sleep(1)  # Rate limit
        resp = self._session.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            timeout=60,
        )
        if resp.status_code != 200:
            logger.warning(f"  Wikidata query failed: {resp.status_code}")
            return []
        return resp.json().get("results", {}).get("bindings", [])

    def get_genre_hierarchy(self) -> tuple[list[MusicEntity], list[MusicRelation]]:
        """Get the full music genre hierarchy from Wikidata."""
        query = """
        SELECT ?genre ?genreLabel ?parent ?parentLabel WHERE {
          ?genre wdt:P31/wdt:P279* wd:Q188451 .  # instance of music genre
          OPTIONAL { ?genre wdt:P279 ?parent . }  # subclass of
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 5000
        """

        cache_path = self.data_dir / "genre_hierarchy.json"
        if cache_path.exists():
            with open(cache_path) as f:
                results = json.load(f)
            logger.info(f"  Wikidata genre hierarchy loaded from cache: {len(results)} entries")
        else:
            logger.info("  Querying Wikidata for genre hierarchy...")
            results = self._sparql_query(query)
            with open(cache_path, "w") as f:
                json.dump(results, f)

        entities = []
        relations = []
        seen_genres = set()

        for row in results:
            genre_uri = row.get("genre", {}).get("value", "")
            genre_label = row.get("genreLabel", {}).get("value", "")
            parent_uri = row.get("parent", {}).get("value", "")
            parent_label = row.get("parentLabel", {}).get("value", "")

            genre_id = genre_uri.split("/")[-1] if genre_uri else ""

            if genre_id and genre_id not in seen_genres:
                entities.append(MusicEntity(
                    entity_id=f"wd:{genre_id}",
                    entity_type="genre",
                    name=genre_label,
                    source="wikidata",
                ))
                seen_genres.add(genre_id)

            if parent_uri:
                parent_id = parent_uri.split("/")[-1]
                relations.append(MusicRelation(
                    source_id=f"wd:{genre_id}",
                    target_id=f"wd:{parent_id}",
                    relation_type="subgenre_of",
                    source="wikidata",
                ))

        logger.info(f"  Wikidata: {len(entities)} genres, {len(relations)} hierarchy relations")
        return entities, relations

    def get_artist_influences(self, limit: int = 2000) -> tuple[list[MusicEntity], list[MusicRelation]]:
        """Get artist influence relationships from Wikidata."""
        query = f"""
        SELECT ?artist ?artistLabel ?influence ?influenceLabel WHERE {{
          ?artist wdt:P31 wd:Q5 .           # is a human
          ?artist wdt:P106 wd:Q639669 .     # occupation: musician
          ?artist wdt:P737 ?influence .      # influenced by
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT {limit}
        """

        cache_path = self.data_dir / "artist_influences.json"
        if cache_path.exists():
            with open(cache_path) as f:
                results = json.load(f)
            logger.info(f"  Wikidata influences loaded from cache: {len(results)} entries")
        else:
            logger.info("  Querying Wikidata for artist influences...")
            results = self._sparql_query(query)
            with open(cache_path, "w") as f:
                json.dump(results, f)

        entities = []
        relations = []
        seen = set()

        for row in results:
            artist_uri = row.get("artist", {}).get("value", "")
            artist_label = row.get("artistLabel", {}).get("value", "")
            influence_uri = row.get("influence", {}).get("value", "")
            influence_label = row.get("influenceLabel", {}).get("value", "")

            artist_id = artist_uri.split("/")[-1]
            influence_id = influence_uri.split("/")[-1]

            for eid, label in [(artist_id, artist_label), (influence_id, influence_label)]:
                if eid and eid not in seen:
                    entities.append(MusicEntity(
                        entity_id=f"wd:{eid}",
                        entity_type="artist",
                        name=label,
                        source="wikidata",
                    ))
                    seen.add(eid)

            relations.append(MusicRelation(
                source_id=f"wd:{artist_id}",
                target_id=f"wd:{influence_id}",
                relation_type="influenced_by",
                source="wikidata",
            ))

        logger.info(f"  Wikidata: {len(entities)} artists, {len(relations)} influence relations")
        return entities, relations

    def get_instrument_taxonomy(self) -> tuple[list[MusicEntity], list[MusicRelation]]:
        """Get the full musical instrument hierarchy from Wikidata."""
        query = """
        SELECT ?instrument ?instrumentLabel ?parent ?parentLabel WHERE {
          ?instrument wdt:P31/wdt:P279* wd:Q34379 .  # musical instrument
          OPTIONAL { ?instrument wdt:P279 ?parent . }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 3000
        """

        cache_path = self.data_dir / "instrument_taxonomy.json"
        if cache_path.exists():
            with open(cache_path) as f:
                results = json.load(f)
        else:
            logger.info("  Querying Wikidata for instrument taxonomy...")
            results = self._sparql_query(query)
            with open(cache_path, "w") as f:
                json.dump(results, f)

        entities = []
        relations = []
        seen = set()

        for row in results:
            inst_uri = row.get("instrument", {}).get("value", "")
            inst_label = row.get("instrumentLabel", {}).get("value", "")
            parent_uri = row.get("parent", {}).get("value", "")

            inst_id = inst_uri.split("/")[-1]
            if inst_id and inst_id not in seen:
                entities.append(MusicEntity(
                    entity_id=f"wd:{inst_id}",
                    entity_type="instrument",
                    name=inst_label,
                    source="wikidata",
                ))
                seen.add(inst_id)

            if parent_uri:
                parent_id = parent_uri.split("/")[-1]
                relations.append(MusicRelation(
                    source_id=f"wd:{inst_id}",
                    target_id=f"wd:{parent_id}",
                    relation_type="subtype_of",
                    source="wikidata",
                ))

        logger.info(f"  Wikidata: {len(entities)} instruments, {len(relations)} taxonomy relations")
        return entities, relations


# ---------------------------------------------------------------------------
# AcousticBrainz (4M+ tracks with audio features)
# ---------------------------------------------------------------------------

ACOUSTICBRAINZ_DUMPS = "https://data.metabrainz.org/pub/acousticbrainz/acousticbrainz-highlevel-json"

class AcousticBrainzLoader:
    """
    Loads AcousticBrainz data — 4M+ tracks with pre-computed audio features.
    Features include: key, BPM, danceability, genre, mood, timbre descriptors.
    Data available as bulk JSON dumps.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "acousticbrainz"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_dumps(self, max_files: int = 10) -> list[Path]:
        """Download AcousticBrainz high-level JSON dumps."""
        downloaded = []

        # AcousticBrainz provides tar.bz2 files with JSON
        # Each file contains ~400k track analyses
        logger.info("  Downloading AcousticBrainz dumps...")
        logger.info("  Note: AcousticBrainz was shut down in 2022.")
        logger.info("  Data available at: https://data.metabrainz.org/pub/acousticbrainz/")

        # Try to download available dumps
        for i in range(max_files):
            dump_name = f"acousticbrainz-highlevel-json-{i:02d}.tar.bz2"
            dump_path = self.data_dir / dump_name
            url = f"{ACOUSTICBRAINZ_DUMPS}/{dump_name}"

            if dump_path.exists():
                downloaded.append(dump_path)
                continue

            try:
                result = subprocess.run([
                    "wget", "-q", "--spider", url,
                ], capture_output=True, timeout=10)

                if result.returncode == 0:
                    subprocess.run([
                        "wget", "-q", "--show-progress", url, "-O", str(dump_path),
                    ], check=True, timeout=3600)
                    downloaded.append(dump_path)
                else:
                    break  # No more files
            except Exception:
                break

        logger.info(f"  AcousticBrainz: {len(downloaded)} dump files available")
        return downloaded

    def load_features(self, max_tracks: int = 100000) -> list[MusicEntity]:
        """Load audio features from AcousticBrainz JSON files."""
        entities = []

        for json_file in sorted(self.data_dir.glob("*.json")):
            with open(json_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        mbid = data.get("mbid", "")

                        hl = data.get("highlevel", {})
                        entities.append(MusicEntity(
                            entity_id=f"ab:{mbid}",
                            entity_type="recording_features",
                            name=mbid,
                            properties={
                                "danceability": hl.get("danceability", {}).get("all", {}).get("danceable", 0),
                                "genre": hl.get("genre_electronic", {}).get("value", ""),
                                "mood": hl.get("mood_happy", {}).get("all", {}).get("happy", 0),
                                "voice_instrumental": hl.get("voice_instrumental", {}).get("value", ""),
                                "timbre": hl.get("timbre", {}).get("value", ""),
                                "tonal_atonal": hl.get("tonal_atonal", {}).get("value", ""),
                            },
                            source="acousticbrainz",
                        ))

                        if len(entities) >= max_tracks:
                            break
                    except json.JSONDecodeError:
                        continue

            if len(entities) >= max_tracks:
                break

        logger.info(f"  AcousticBrainz: {len(entities):,} track features loaded")
        return entities


# ---------------------------------------------------------------------------
# Discogs (15M+ releases)
# ---------------------------------------------------------------------------

DISCOGS_DATA_URL = "https://discogs-data-dumps.s3-us-west-2.amazonaws.com/data/2024"

class DiscogsLoader:
    """
    Loads Discogs data dumps — 15M+ releases with labels, styles, credits.
    Monthly dumps available in XML and JSON formats.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_dir = data_root / "discogs"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_artist_dump(self) -> Optional[Path]:
        """Download Discogs artist data dump."""
        # Discogs provides monthly XML dumps
        # The artists file is ~1.5GB compressed
        dump_path = self.data_dir / "discogs_artists.xml.gz"

        if dump_path.exists():
            logger.info("  Discogs artist dump already downloaded")
            return dump_path

        # Try recent monthly dumps
        for month in range(12, 0, -1):
            url = f"{DISCOGS_DATA_URL}/discogs_2024{month:02d}01_artists.xml.gz"
            logger.info(f"  Trying Discogs artists dump: 2024-{month:02d}...")
            result = subprocess.run([
                "wget", "-q", "--spider", url,
            ], capture_output=True, timeout=10)

            if result.returncode == 0:
                subprocess.run([
                    "wget", "-q", "--show-progress", url, "-O", str(dump_path),
                ], check=True, timeout=7200)
                return dump_path

        logger.warning("  Could not find Discogs dump. Visit: https://data.discogs.com/")
        return None

    def load_style_genre_map(self) -> dict[str, list[str]]:
        """
        Extract genre → style mappings from Discogs.
        Discogs has the most granular genre/style taxonomy in the world.
        """
        # Discogs genres and their sub-styles
        # This is a curated subset — full extraction requires XML parsing
        genre_styles = {
            "Electronic": [
                "Ambient", "Breakbeat", "Disco", "Downtempo", "Drum n Bass",
                "Dub", "Electro", "Euro House", "Experimental", "Garage House",
                "Happy Hardcore", "Hard House", "Hard Trance", "House", "IDM",
                "Industrial", "Jungle", "Minimal", "Progressive House",
                "Progressive Trance", "Synth-pop", "Tech House", "Techno",
                "Trance", "Trap", "Trip Hop", "UK Garage",
            ],
            "Hip Hop": [
                "Bass Music", "Boom Bap", "Conscious", "Crunk", "Cut-up/DJ",
                "Gangsta", "Grime", "Instrumental", "Jazzy Hip-Hop", "Pop Rap",
                "RnB/Swing", "Trap", "Trip Hop", "Turntablism",
            ],
            "Rock": [
                "Alternative Rock", "Art Rock", "Black Metal", "Blues Rock",
                "Classic Rock", "Death Metal", "Doom Metal", "Emo", "Garage Rock",
                "Gothic Rock", "Grunge", "Hard Rock", "Hardcore", "Indie Rock",
                "Math Rock", "Metal", "New Wave", "Post-Punk", "Post-Rock",
                "Prog Rock", "Psychedelic Rock", "Punk", "Shoegaze", "Stoner Rock",
            ],
            "Jazz": [
                "Afro-Cuban Jazz", "Avant-garde Jazz", "Bebop", "Big Band",
                "Bossa Nova", "Contemporary Jazz", "Cool Jazz", "Free Jazz",
                "Fusion", "Hard Bop", "Latin Jazz", "Modal", "Post Bop",
                "Smooth Jazz", "Soul-Jazz", "Swing",
            ],
            "Classical": [
                "Baroque", "Chamber Music", "Choral", "Classical", "Contemporary",
                "Impressionist", "Medieval", "Modern", "Neo-Classical",
                "Neo-Romantic", "Opera", "Orchestral", "Romantic", "Symphonic",
            ],
            "Reggae": [
                "Dancehall", "Dub", "Lovers Rock", "Ragga", "Reggae", "Rocksteady", "Ska",
            ],
            "Latin": [
                "Bossa Nova", "Cumbia", "Latin Jazz", "MPB", "Reggaeton",
                "Salsa", "Samba", "Tango",
            ],
            "Funk / Soul": [
                "Afrobeat", "Boogie", "Disco", "Funk", "Gospel", "Neo Soul",
                "P.Funk", "Rhythm & Blues", "Soul", "Swingbeat",
            ],
        }
        return genre_styles


# ---------------------------------------------------------------------------
# Knowledge Graph Database
# ---------------------------------------------------------------------------

class MusicKnowledgeGraph:
    """
    SQLite-backed knowledge graph that unifies all Phase H data sources.
    Provides the foundation for embedding-space training in Phase H.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                source TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                source TEXT DEFAULT '',
                FOREIGN KEY (source_id) REFERENCES entities(entity_id),
                FOREIGN KEY (target_id) REFERENCES entities(entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_relation_type ON relations(relation_type);
            CREATE INDEX IF NOT EXISTS idx_source ON relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_target ON relations(target_id);
        """)
        self._conn.commit()

    def add_entities(self, entities: list[MusicEntity]):
        self._conn.executemany(
            "INSERT OR REPLACE INTO entities (entity_id, entity_type, name, properties, source) VALUES (?, ?, ?, ?, ?)",
            [(e.entity_id, e.entity_type, e.name, json.dumps(e.properties), e.source) for e in entities],
        )
        self._conn.commit()

    def add_relations(self, relations: list[MusicRelation]):
        self._conn.executemany(
            "INSERT INTO relations (source_id, target_id, relation_type, properties, source) VALUES (?, ?, ?, ?, ?)",
            [(r.source_id, r.target_id, r.relation_type, json.dumps(r.properties), r.source) for r in relations],
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        entities = self._conn.execute("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type").fetchall()
        relations = self._conn.execute("SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type").fetchall()
        return {
            "entities": dict(entities),
            "relations": dict(relations),
            "total_entities": sum(c for _, c in entities),
            "total_relations": sum(c for _, c in relations),
        }

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------

class PhaseHPipeline:
    """
    Master pipeline for Phase H music knowledge graph.
    Builds the unified graph from all data sources.
    """

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT):
        self.data_root = data_root
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.graph = MusicKnowledgeGraph(data_root / "music_knowledge.db")

    def build_graph(self, seed_artists: Optional[list[str]] = None) -> dict:
        """Build the full music knowledge graph."""
        if seed_artists is None:
            # Top influential producers/artists across eras
            seed_artists = [
                "The Beatles", "James Brown", "Kraftwerk", "Bob Marley",
                "Michael Jackson", "Prince", "Madonna", "Dr. Dre",
                "Aphex Twin", "Daft Punk", "Timbaland", "Pharrell Williams",
                "Kanye West", "Skrillex", "Metro Boomin", "Burial",
                "Brian Eno", "J Dilla", "Lee Scratch Perry", "Quincy Jones",
                "Max Martin", "Rick Rubin", "Arca", "Sophie",
                "Flying Lotus", "Bjork", "Radiohead", "Tame Impala",
            ]

        all_entities = []
        all_relations = []

        # 1. Wikidata — genre hierarchy + artist influences + instruments
        logger.info("Building Wikidata knowledge...")
        wd = WikidataClient(self.data_root)

        genres_e, genres_r = wd.get_genre_hierarchy()
        all_entities.extend(genres_e)
        all_relations.extend(genres_r)

        influences_e, influences_r = wd.get_artist_influences()
        all_entities.extend(influences_e)
        all_relations.extend(influences_r)

        instruments_e, instruments_r = wd.get_instrument_taxonomy()
        all_entities.extend(instruments_e)
        all_relations.extend(instruments_r)

        # 2. MusicBrainz — detailed artist relationships
        logger.info("Building MusicBrainz artist graph...")
        mb = MusicBrainzClient(self.data_root)
        mb_entities, mb_relations = mb.build_artist_graph(seed_artists, depth=2)
        all_entities.extend(mb_entities)
        all_relations.extend(mb_relations)

        # 3. Discogs genre/style taxonomy
        logger.info("Loading Discogs genre taxonomy...")
        discogs = DiscogsLoader(self.data_root)
        genre_styles = discogs.load_style_genre_map()
        for genre, styles in genre_styles.items():
            genre_entity = MusicEntity(
                entity_id=f"discogs:genre:{genre.lower().replace(' ', '_')}",
                entity_type="genre",
                name=genre,
                source="discogs",
            )
            all_entities.append(genre_entity)
            for style in styles:
                style_entity = MusicEntity(
                    entity_id=f"discogs:style:{style.lower().replace(' ', '_')}",
                    entity_type="style",
                    name=style,
                    source="discogs",
                )
                all_entities.append(style_entity)
                all_relations.append(MusicRelation(
                    source_id=style_entity.entity_id,
                    target_id=genre_entity.entity_id,
                    relation_type="style_of",
                    source="discogs",
                ))

        # Store in graph
        self.graph.add_entities(all_entities)
        self.graph.add_relations(all_relations)

        stats = self.graph.stats()
        logger.info(f"\nPhase H Knowledge Graph built:")
        logger.info(f"  Entities: {stats['total_entities']:,}")
        for etype, count in stats["entities"].items():
            logger.info(f"    {etype}: {count:,}")
        logger.info(f"  Relations: {stats['total_relations']:,}")
        for rtype, count in stats["relations"].items():
            logger.info(f"    {rtype}: {count:,}")

        return stats

    def close(self):
        self.graph.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="RESONATE Phase H: Knowledge Graph Pipeline")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    args = parser.parse_args()

    pipeline = PhaseHPipeline(data_root=Path(args.data_root))
    try:
        pipeline.build_graph()
    finally:
        pipeline.close()
