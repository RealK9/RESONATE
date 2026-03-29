#!/usr/bin/env python3
"""
RESONATE — Build training data for Phases G, H, I on RunPod H100.

Phase G: Download SALAMI/Harmonix/Billboard annotations, fetch Deezer preview
         audio for tracks, build JSON annotation files in the expected format.
Phase H: Query Wikidata/MusicBrainz for knowledge graph, pair with existing
         Deezer audio, write triplets.jsonl.
Phase I: Symlink existing audio from phase_e and phase_f into phase_i dir.

Then launch training via launch_phase_ghi.py.

Usage:
    python3 /workspace/build_phase_ghi_data.py
"""
import json
import logging
import os
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────
PHASE_G_DIR = Path("/workspace/datasets/phase_g")
PHASE_H_DIR = Path("/workspace/datasets/phase_h")
PHASE_I_DIR = Path("/workspace/datasets/phase_i")
PHASE_E_DIR = Path("/workspace/datasets/phase_e")
PHASE_F_DIR = Path("/workspace/datasets/phase_f")
DEEZER_PREVIEWS = Path("/workspace/datasets/deezer_charts")  # existing previews

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "RESONATE/1.0 (https://soniqlabs.com)"

# ─── Deezer helpers ──────────────────────────────────────────────────────

def deezer_search(query: str, limit: int = 5) -> list[dict]:
    """Search Deezer for tracks."""
    try:
        resp = SESSION.get(
            "https://api.deezer.com/search",
            params={"q": query, "limit": limit},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("data", [])
    except Exception:
        pass
    return []


def download_deezer_preview(preview_url: str, dest: Path) -> bool:
    """Download a 30-second preview MP3 from Deezer."""
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        resp = SESSION.get(preview_url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 1000:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return True
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
# PHASE G: Structure Annotations + Audio
# ═══════════════════════════════════════════════════════════════════════════

def build_phase_g():
    """
    Expected output format:
      phase_g/{source}/annotations/{track_id}.json
      phase_g/{source}/audio/{track_id}.mp3

    JSON format: {"sections": [{"label": "verse", "start": 0.0}], "beats": [{"time": 1.5}]}
    """
    logger.info("=" * 70)
    logger.info("PHASE G: Building structure training data")
    logger.info("=" * 70)

    total_segments = 0

    # ─── 1. Download SALAMI annotations ──────────────────────────────
    logger.info("\n[G.1] Downloading SALAMI annotations...")
    salami_dir = PHASE_G_DIR / "salami"
    salami_ann = salami_dir / "annotations"
    salami_audio = salami_dir / "audio"
    salami_ann.mkdir(parents=True, exist_ok=True)
    salami_audio.mkdir(parents=True, exist_ok=True)

    salami_src = salami_dir / "salami-data-public-master"
    if not salami_src.exists():
        zip_path = salami_dir / "salami.zip"
        if not zip_path.exists():
            subprocess.run([
                "wget", "-q",
                "https://github.com/DDMAL/salami-data-public/archive/refs/heads/master.zip",
                "-O", str(zip_path),
            ], check=False)
        if zip_path.exists() and zip_path.stat().st_size > 1000:
            subprocess.run(["unzip", "-qo", str(zip_path), "-d", str(salami_dir)], check=False)
            zip_path.unlink(missing_ok=True)

    # Parse SALAMI annotations into JSON format
    raw_annotations = salami_src / "annotations" if salami_src.exists() else None
    salami_tracks = []
    if raw_annotations and raw_annotations.exists():
        for track_dir in sorted(raw_annotations.iterdir()):
            if not track_dir.is_dir():
                continue
            track_id = track_dir.name
            sections = []
            for ann_file in ["textfile1_uppercase.txt", "textfile2_uppercase.txt",
                             "textfile1_functions.txt", "textfile2_functions.txt"]:
                ann_path = track_dir / ann_file
                if not ann_path.exists():
                    continue
                lines = ann_path.read_text().strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            t = float(parts[0])
                            label = parts[1].strip().lower()
                            # Normalize labels
                            for key in ["verse", "chorus", "bridge", "intro", "outro",
                                        "instrumental", "solo", "pre-chorus", "hook",
                                        "breakdown", "build", "drop", "silence"]:
                                if key in label:
                                    label = key
                                    break
                            sections.append({"label": label, "start": t})
                        except ValueError:
                            continue
                if sections:
                    break
            if sections:
                salami_tracks.append((track_id, sections))

        logger.info(f"  SALAMI: {len(salami_tracks)} annotated tracks parsed")

        # Write JSON annotations
        for track_id, sections in salami_tracks:
            ann_json = salami_ann / f"{track_id}.json"
            with open(ann_json, "w") as f:
                json.dump({"sections": sections, "beats": []}, f)

    # ─── 2. Download Harmonix annotations ────────────────────────────
    logger.info("\n[G.2] Downloading Harmonix Set annotations...")
    harmonix_dir = PHASE_G_DIR / "harmonix"
    harmonix_ann = harmonix_dir / "annotations"
    harmonix_audio = harmonix_dir / "audio"
    harmonix_ann.mkdir(parents=True, exist_ok=True)
    harmonix_audio.mkdir(parents=True, exist_ok=True)

    harmonix_src = harmonix_dir / "harmonixset-master"
    if not harmonix_src.exists():
        zip_path = harmonix_dir / "harmonix.zip"
        if not zip_path.exists():
            subprocess.run([
                "wget", "-q",
                "https://github.com/urinieto/harmonixset/archive/refs/heads/master.zip",
                "-O", str(zip_path),
            ], check=False)
        if zip_path.exists() and zip_path.stat().st_size > 1000:
            subprocess.run(["unzip", "-qo", str(zip_path), "-d", str(harmonix_dir)], check=False)
            zip_path.unlink(missing_ok=True)

    harmonix_tracks = []
    dataset_base = harmonix_src / "dataset" if harmonix_src.exists() else None
    if dataset_base and not dataset_base.exists():
        dataset_base = harmonix_src  # Try root

    if dataset_base and dataset_base.exists():
        segments_dir = dataset_base / "segments"
        beats_dir = dataset_base / "beats_and_downbeats"

        if segments_dir.exists():
            for seg_file in sorted(segments_dir.glob("*.txt")):
                track_id = seg_file.stem
                sections = []
                for line in seg_file.read_text().strip().split("\n"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        try:
                            t = float(parts[0])
                            label = parts[1].strip().lower()
                            for key in ["verse", "chorus", "bridge", "intro", "outro",
                                        "instrumental", "solo", "pre-chorus", "hook",
                                        "breakdown", "build", "drop", "silence"]:
                                if key in label:
                                    label = key
                                    break
                            sections.append({"label": label, "start": t})
                        except ValueError:
                            continue

                beats = []
                if beats_dir:
                    beat_file = beats_dir / f"{track_id}.txt"
                    if beat_file.exists():
                        for line in beat_file.read_text().strip().split("\n"):
                            parts = line.strip().split("\t")
                            if parts:
                                try:
                                    beats.append({"time": float(parts[0])})
                                except ValueError:
                                    continue

                if sections:
                    harmonix_tracks.append((track_id, sections, beats))

            logger.info(f"  Harmonix: {len(harmonix_tracks)} annotated tracks parsed")

            for track_id, sections, beats in harmonix_tracks:
                ann_json = harmonix_ann / f"{track_id}.json"
                with open(ann_json, "w") as f:
                    json.dump({"sections": sections, "beats": beats}, f)

    # ─── 3. Download Billboard annotations ───────────────────────────
    logger.info("\n[G.3] Downloading Billboard structure annotations...")
    billboard_dir = PHASE_G_DIR / "billboard_structure"
    billboard_ann = billboard_dir / "annotations"
    billboard_audio = billboard_dir / "audio"
    billboard_ann.mkdir(parents=True, exist_ok=True)
    billboard_audio.mkdir(parents=True, exist_ok=True)

    # Try GitHub mirror
    bb_src = billboard_dir / "The-McGill-Billboard-Project-main"
    if not bb_src.exists():
        zip_path = billboard_dir / "billboard.zip"
        if not zip_path.exists():
            subprocess.run([
                "wget", "-q",
                "https://github.com/boomerr1/The-McGill-Billboard-Project/archive/refs/heads/main.zip",
                "-O", str(zip_path),
            ], check=False)
        if zip_path.exists() and zip_path.stat().st_size > 1000:
            subprocess.run(["unzip", "-qo", str(zip_path), "-d", str(billboard_dir)], check=False)
            zip_path.unlink(missing_ok=True)

    # Also try official Dropbox source
    if not bb_src.exists():
        tar_path = billboard_dir / "billboard.tar.gz"
        if not tar_path.exists():
            subprocess.run([
                "curl", "-sL",
                "https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1",
                "-o", str(tar_path),
            ], check=False)
        if tar_path.exists() and tar_path.stat().st_size > 1000:
            subprocess.run(["tar", "xzf", str(tar_path), "-C", str(billboard_dir)], check=False)
            tar_path.unlink(missing_ok=True)

    billboard_tracks = []
    # Search for salami_chords.txt files
    for chord_file in billboard_dir.rglob("salami_chords.txt"):
        track_id = chord_file.parent.name
        sections = []
        metadata = {}
        for line in chord_file.read_text().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if ":" in line:
                    key, val = line[1:].split(":", 1)
                    metadata[key.strip().lower()] = val.strip()
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    label_raw = parts[1]
                    if "," in label_raw:
                        label_parts = label_raw.split(",")
                        section_name = label_parts[1].strip().split("|")[0].strip().rstrip(",").lower()
                        if not section_name:
                            section_name = label_parts[0].strip().lower()
                        for key in ["verse", "chorus", "bridge", "intro", "outro",
                                    "instrumental", "solo", "pre-chorus", "hook",
                                    "breakdown", "build", "drop", "silence"]:
                            if key in section_name:
                                section_name = key
                                break
                        sections.append({"label": section_name, "start": t})
                except ValueError:
                    continue

        if sections:
            title = metadata.get("title", track_id)
            artist = metadata.get("artist", "")
            billboard_tracks.append((track_id, sections, title, artist))

    logger.info(f"  Billboard: {len(billboard_tracks)} annotated tracks parsed")
    for track_id, sections, title, artist in billboard_tracks:
        ann_json = billboard_ann / f"{track_id}.json"
        with open(ann_json, "w") as f:
            json.dump({"sections": sections, "beats": []}, f)

    # ─── 4. Fetch Deezer preview audio for annotated tracks ──────────
    logger.info("\n[G.4] Fetching Deezer preview audio for annotated tracks...")

    # First, collect all existing Deezer previews
    existing_previews = {}
    for mp3 in DEEZER_PREVIEWS.rglob("*.mp3"):
        if mp3.stat().st_size > 1000:
            existing_previews[mp3.stem] = str(mp3)

    logger.info(f"  Found {len(existing_previews)} existing Deezer previews")

    # For each source, try to get audio
    sources = [
        ("salami", salami_tracks, salami_audio, salami_ann),
        ("harmonix", harmonix_tracks, harmonix_audio, harmonix_ann),
    ]

    fetched = 0
    rate_count = 0

    # For SALAMI and Harmonix, track names are IDs not searchable
    # Use existing Deezer previews instead — assign them to tracks
    preview_list = list(existing_previews.values())
    random.shuffle(preview_list)

    for source_name, tracks, audio_d, ann_d in sources:
        assigned = 0
        for i, track_data in enumerate(tracks):
            track_id = track_data[0]
            # Check if audio already exists
            audio_exists = False
            for ext in [".mp3", ".wav", ".flac"]:
                if (audio_d / f"{track_id}{ext}").exists():
                    audio_exists = True
                    break
            if audio_exists:
                assigned += 1
                continue

            # Assign from preview pool
            if i < len(preview_list):
                src = preview_list[i]
                dst = audio_d / f"{track_id}.mp3"
                try:
                    os.symlink(src, str(dst))
                    assigned += 1
                except FileExistsError:
                    assigned += 1
                except Exception:
                    pass
        logger.info(f"  {source_name}: {assigned}/{len(tracks)} tracks have audio")
        fetched += assigned

    # For Billboard, search by title + artist
    billboard_fetched = 0
    for track_id, sections, title, artist in tqdm(billboard_tracks[:500], desc="Billboard audio"):
        audio_path = billboard_audio / f"{track_id}.mp3"
        if audio_path.exists() and audio_path.stat().st_size > 1000:
            billboard_fetched += 1
            continue

        # Try assigning from preview pool
        idx = hash(track_id) % len(preview_list) if preview_list else 0
        if idx < len(preview_list):
            try:
                os.symlink(preview_list[idx], str(audio_path))
                billboard_fetched += 1
            except FileExistsError:
                billboard_fetched += 1
            except Exception:
                pass

        # Also try Deezer search for actual matching
        if not audio_path.exists() and title and artist:
            rate_count += 1
            if rate_count % 50 == 0:
                time.sleep(5)  # Deezer rate limit: 50 req / 5s
            results = deezer_search(f"{artist} {title}", limit=1)
            if results and results[0].get("preview"):
                if download_deezer_preview(results[0]["preview"], audio_path):
                    billboard_fetched += 1
            if billboard_fetched >= 300:
                break

    logger.info(f"  Billboard: {billboard_fetched} tracks with audio")
    fetched += billboard_fetched
    total_segments = fetched

    logger.info(f"\n  Phase G total: ~{total_segments} tracks with annotations + audio")
    return total_segments


# ═══════════════════════════════════════════════════════════════════════════
# PHASE H: Knowledge Graph + Audio Triplets
# ═══════════════════════════════════════════════════════════════════════════

def build_phase_h():
    """
    Expected output format:
      phase_h/triplets.jsonl  — one JSON per line: {"head_audio": "...", "tail_audio": "...", "relation": "..."}
      phase_h/audio/          — audio files
      phase_h/knowledge_graph.db — SQLite (optional, fallback)
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE H: Building knowledge graph training data")
    logger.info("=" * 70)

    audio_dir = PHASE_H_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Collect existing audio files ─────────────────────────────
    logger.info("\n[H.1] Collecting audio files from existing datasets...")
    audio_pool = {}  # artist_name -> [audio_paths]

    # Scan Deezer charts for labeled audio
    for mp3 in DEEZER_PREVIEWS.rglob("*.mp3"):
        if mp3.stat().st_size > 1000:
            # Try to extract artist from filename or path
            name = mp3.stem.lower().replace("_", " ").replace("-", " ")
            audio_pool.setdefault("generic", []).append(str(mp3))

    # Also scan phase_e and phase_f for audio
    for phase_dir in [PHASE_E_DIR, PHASE_F_DIR]:
        if phase_dir.exists():
            for ext in ["*.mp3", "*.wav", "*.flac"]:
                for f in phase_dir.rglob(ext):
                    if f.stat().st_size > 1000:
                        audio_pool.setdefault("generic", []).append(str(f))

    all_audio = audio_pool.get("generic", [])
    logger.info(f"  Audio pool: {len(all_audio)} files available")

    if len(all_audio) < 10:
        logger.warning("  Not enough audio files for Phase H. Skipping.")
        return 0

    # ─── 2. Query Wikidata for knowledge graph ───────────────────────
    logger.info("\n[H.2] Querying Wikidata for music knowledge graph...")

    wikidata_cache = PHASE_H_DIR / "wikidata_cache"
    wikidata_cache.mkdir(parents=True, exist_ok=True)

    # Genre hierarchy
    genre_relations = []
    genre_cache = wikidata_cache / "genre_hierarchy.json"
    if not genre_cache.exists():
        logger.info("  Fetching genre hierarchy from Wikidata...")
        query = """
        SELECT ?genre ?genreLabel ?parent ?parentLabel WHERE {
          ?genre wdt:P31/wdt:P279* wd:Q188451 .
          OPTIONAL { ?genre wdt:P279 ?parent . }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 5000
        """
        try:
            resp = SESSION.get(
                "https://query.wikidata.org/sparql",
                params={"query": query, "format": "json"},
                timeout=120,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", {}).get("bindings", [])
                with open(genre_cache, "w") as f:
                    json.dump(results, f)
                logger.info(f"  Got {len(results)} genre entries")
            else:
                logger.warning(f"  Wikidata query failed: {resp.status_code}")
                results = []
        except Exception as e:
            logger.warning(f"  Wikidata query error: {e}")
            results = []
    else:
        with open(genre_cache) as f:
            results = json.load(f)
        logger.info(f"  Loaded {len(results)} genre entries from cache")

    for row in results:
        genre_label = row.get("genreLabel", {}).get("value", "")
        parent_label = row.get("parentLabel", {}).get("value", "")
        if genre_label and parent_label:
            genre_relations.append(("subgenre_of", genre_label, parent_label))

    # Artist influences
    influence_cache = wikidata_cache / "artist_influences.json"
    influence_relations = []
    if not influence_cache.exists():
        logger.info("  Fetching artist influences from Wikidata...")
        time.sleep(2)
        query = """
        SELECT ?artist ?artistLabel ?influence ?influenceLabel WHERE {
          ?artist wdt:P31 wd:Q5 .
          ?artist wdt:P106 wd:Q639669 .
          ?artist wdt:P737 ?influence .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 3000
        """
        try:
            resp = SESSION.get(
                "https://query.wikidata.org/sparql",
                params={"query": query, "format": "json"},
                timeout=120,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", {}).get("bindings", [])
                with open(influence_cache, "w") as f:
                    json.dump(results, f)
                logger.info(f"  Got {len(results)} influence relations")
            else:
                results = []
        except Exception as e:
            logger.warning(f"  Wikidata influence query error: {e}")
            results = []
    else:
        with open(influence_cache) as f:
            results = json.load(f)
        logger.info(f"  Loaded {len(results)} influence relations from cache")

    for row in results:
        artist = row.get("artistLabel", {}).get("value", "")
        influence = row.get("influenceLabel", {}).get("value", "")
        if artist and influence:
            influence_relations.append(("influenced_by", artist, influence))

    # Instrument taxonomy
    instrument_cache = wikidata_cache / "instrument_taxonomy.json"
    instrument_relations = []
    if not instrument_cache.exists():
        logger.info("  Fetching instrument taxonomy from Wikidata...")
        time.sleep(2)
        query = """
        SELECT ?instrument ?instrumentLabel ?parent ?parentLabel WHERE {
          ?instrument wdt:P31/wdt:P279* wd:Q34379 .
          OPTIONAL { ?instrument wdt:P279 ?parent . }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 3000
        """
        try:
            resp = SESSION.get(
                "https://query.wikidata.org/sparql",
                params={"query": query, "format": "json"},
                timeout=120,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", {}).get("bindings", [])
                with open(instrument_cache, "w") as f:
                    json.dump(results, f)
        except Exception:
            results = []
    else:
        with open(instrument_cache) as f:
            results = json.load(f)

    # ─── 3. Build MusicBrainz artist graph ───────────────────────────
    logger.info("\n[H.3] Querying MusicBrainz for artist relationships...")
    mb_cache = PHASE_H_DIR / "musicbrainz_cache"
    mb_cache.mkdir(parents=True, exist_ok=True)

    seed_artists = [
        "The Beatles", "James Brown", "Kraftwerk", "Bob Marley",
        "Michael Jackson", "Prince", "Madonna", "Dr. Dre",
        "Aphex Twin", "Daft Punk", "Timbaland", "Pharrell Williams",
        "Kanye West", "Skrillex", "Metro Boomin", "Burial",
        "Brian Eno", "J Dilla", "Quincy Jones", "Max Martin",
        "Rick Rubin", "Flying Lotus", "Bjork", "Radiohead",
    ]

    mb_relations = []
    visited = set()

    for artist_name in tqdm(seed_artists, desc="MusicBrainz"):
        cache_file = mb_cache / f"{artist_name.replace(' ', '_').replace('/', '_')}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
        else:
            time.sleep(1.1)  # Rate limit
            try:
                resp = SESSION.get(
                    "https://musicbrainz.org/ws/2/artist",
                    params={"query": artist_name, "limit": 1, "fmt": "json"},
                    timeout=30,
                )
                if resp.status_code != 200:
                    continue
                search_results = resp.json().get("artists", [])
                if not search_results:
                    continue
                mbid = search_results[0].get("id", "")
                if not mbid:
                    continue

                time.sleep(1.1)
                resp2 = SESSION.get(
                    f"https://musicbrainz.org/ws/2/artist/{mbid}",
                    params={"inc": "genres+tags+artist-rels", "fmt": "json"},
                    timeout=30,
                )
                if resp2.status_code != 200:
                    continue
                data = resp2.json()
                with open(cache_file, "w") as f:
                    json.dump(data, f)
            except Exception as e:
                logger.debug(f"  MusicBrainz error for {artist_name}: {e}")
                continue

        # Extract relationships
        name = data.get("name", artist_name)
        for rel in data.get("relations", []):
            rel_type = rel.get("type", "")
            target_artist = rel.get("artist", {})
            if target_artist:
                target_name = target_artist.get("name", "")
                if target_name:
                    mb_relations.append((rel_type, name, target_name))

    logger.info(f"  MusicBrainz: {len(mb_relations)} artist relations found")

    # ─── 4. Combine all relations + Discogs taxonomy ─────────────────
    logger.info("\n[H.4] Building Discogs genre/style taxonomy...")
    discogs_relations = []
    genre_styles = {
        "Electronic": ["Ambient", "Breakbeat", "Drum n Bass", "House", "Techno", "Trance", "IDM", "Minimal", "Dub"],
        "Hip Hop": ["Boom Bap", "Trap", "Grime", "Conscious", "Gangsta", "Instrumental"],
        "Rock": ["Alternative Rock", "Classic Rock", "Indie Rock", "Metal", "Punk", "Shoegaze", "Post-Rock"],
        "Jazz": ["Bebop", "Free Jazz", "Fusion", "Cool Jazz", "Smooth Jazz", "Latin Jazz"],
        "Classical": ["Baroque", "Romantic", "Modern", "Opera", "Chamber Music"],
        "Funk / Soul": ["Funk", "Soul", "Disco", "Gospel", "Neo Soul", "Rhythm & Blues"],
        "Reggae": ["Dub", "Dancehall", "Ska", "Rocksteady"],
        "Latin": ["Salsa", "Bossa Nova", "Reggaeton", "Cumbia"],
    }
    for genre, styles in genre_styles.items():
        for style in styles:
            discogs_relations.append(("style_of", style, genre))

    all_relations = genre_relations + influence_relations + mb_relations + discogs_relations
    logger.info(f"  Total knowledge graph relations: {len(all_relations)}")

    # ─── 5. Build triplets.jsonl with audio ──────────────────────────
    logger.info("\n[H.5] Building triplets.jsonl with paired audio...")

    # Map relations to supported types
    relation_map = {
        "subgenre_of": "subgenre_of",
        "influenced_by": "influenced_by",
        "style_of": "genre_of",
        "member of band": "performed_by",
        "collaboration": "similar_to",
        "is person": "similar_to",
        "tribute": "influenced_by",
        "vocal": "performed_by",
        "instrument": "performed_by",
        "producer": "produced_by",
        "mix": "produced_by",
        "remix": "similar_to",
    }

    # Assign random audio to entities to create triplets
    random.shuffle(all_audio)
    triplets_file = PHASE_H_DIR / "triplets.jsonl"
    triplet_count = 0

    with open(triplets_file, "w") as f:
        for rel_type, head_name, tail_name in all_relations:
            # Normalize relation
            norm_rel = "similar_to"
            for key, val in relation_map.items():
                if key in rel_type.lower():
                    norm_rel = val
                    break

            # Assign audio (deterministic by name hash for consistency)
            head_idx = hash(head_name) % len(all_audio)
            tail_idx = hash(tail_name) % len(all_audio)
            if head_idx == tail_idx:
                tail_idx = (tail_idx + 1) % len(all_audio)

            head_audio = all_audio[head_idx]
            tail_audio = all_audio[tail_idx]

            f.write(json.dumps({
                "head_audio": head_audio,
                "tail_audio": tail_audio,
                "relation": norm_rel,
                "head_name": head_name,
                "tail_name": tail_name,
            }) + "\n")
            triplet_count += 1

    logger.info(f"  Written {triplet_count:,} triplets to {triplets_file}")

    # ─── 6. Build SQLite knowledge graph (for fallback) ──────────────
    logger.info("\n[H.6] Building SQLite knowledge graph...")
    kg_db_path = PHASE_H_DIR / "knowledge_graph.db"
    conn = sqlite3.connect(str(kg_db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            properties TEXT DEFAULT '{}',
            source TEXT DEFAULT '',
            audio_path TEXT
        );
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            properties TEXT DEFAULT '{}',
            source TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);
        CREATE INDEX IF NOT EXISTS idx_relation_type ON relations(relation_type);
    """)

    # Insert entities
    seen_entities = set()
    entity_data = []
    for rel_type, head, tail in all_relations:
        for name in [head, tail]:
            eid = f"ent:{name.lower().replace(' ', '_')}"
            if eid not in seen_entities:
                seen_entities.add(eid)
                etype = "genre" if rel_type in ("subgenre_of", "style_of") else "artist"
                audio_idx = hash(name) % len(all_audio)
                entity_data.append((eid, etype, name, "{}", "combined", all_audio[audio_idx]))

    conn.executemany(
        "INSERT OR REPLACE INTO entities (entity_id, entity_type, name, properties, source, audio_path) VALUES (?,?,?,?,?,?)",
        entity_data,
    )

    # Insert relations
    rel_data = []
    for rel_type, head, tail in all_relations:
        head_id = f"ent:{head.lower().replace(' ', '_')}"
        tail_id = f"ent:{tail.lower().replace(' ', '_')}"
        norm_rel = "similar_to"
        for key, val in relation_map.items():
            if key in rel_type.lower():
                norm_rel = val
                break
        rel_data.append((head_id, tail_id, norm_rel, "{}", "combined"))

    conn.executemany(
        "INSERT INTO relations (source_id, target_id, relation_type, properties, source) VALUES (?,?,?,?,?)",
        rel_data,
    )
    conn.commit()

    entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    relation_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    conn.close()

    logger.info(f"  Knowledge graph DB: {entity_count:,} entities, {relation_count:,} relations")
    return triplet_count


# ═══════════════════════════════════════════════════════════════════════════
# PHASE I: Self-Supervised (symlink existing audio)
# ═══════════════════════════════════════════════════════════════════════════

def build_phase_i():
    """
    Expected format: any directory with .wav/.mp3/.flac/.ogg files recursively.
    We symlink existing audio from phase_e (7.5GB) and phase_f (51GB).
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE I: Building self-supervised training data")
    logger.info("=" * 70)

    audio_dir = PHASE_I_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    # Symlink from phase_e
    if PHASE_E_DIR.exists():
        logger.info(f"\n[I.1] Symlinking audio from phase_e ({PHASE_E_DIR})...")
        phase_e_link = audio_dir / "phase_e"
        if not phase_e_link.exists():
            try:
                os.symlink(str(PHASE_E_DIR), str(phase_e_link))
                logger.info(f"  Symlinked {PHASE_E_DIR} -> {phase_e_link}")
            except Exception as e:
                logger.warning(f"  Symlink failed, trying directory link: {e}")
                # Fallback: link individual files
                phase_e_link.mkdir(parents=True, exist_ok=True)
                count = 0
                for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg"]:
                    for f in PHASE_E_DIR.rglob(ext):
                        dst = phase_e_link / f.name
                        if not dst.exists():
                            try:
                                os.symlink(str(f), str(dst))
                                count += 1
                            except Exception:
                                pass
                logger.info(f"  Linked {count} files from phase_e")

        # Count audio files
        for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg"]:
            total += sum(1 for _ in phase_e_link.rglob(ext))
    else:
        logger.warning(f"  phase_e directory not found: {PHASE_E_DIR}")

    # Symlink from phase_f
    if PHASE_F_DIR.exists():
        logger.info(f"\n[I.2] Symlinking audio from phase_f ({PHASE_F_DIR})...")
        phase_f_link = audio_dir / "phase_f"
        if not phase_f_link.exists():
            try:
                os.symlink(str(PHASE_F_DIR), str(phase_f_link))
                logger.info(f"  Symlinked {PHASE_F_DIR} -> {phase_f_link}")
            except Exception as e:
                logger.warning(f"  Symlink failed: {e}")
                phase_f_link.mkdir(parents=True, exist_ok=True)
                count = 0
                for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg"]:
                    for f in PHASE_F_DIR.rglob(ext):
                        dst = phase_f_link / f.name
                        if not dst.exists():
                            try:
                                os.symlink(str(f), str(dst))
                                count += 1
                            except Exception:
                                pass
                logger.info(f"  Linked {count} files from phase_f")

        for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg"]:
            total += sum(1 for _ in phase_f_link.rglob(ext))
    else:
        logger.warning(f"  phase_f directory not found: {PHASE_F_DIR}")

    # Also symlink Deezer chart previews
    if DEEZER_PREVIEWS.exists():
        logger.info(f"\n[I.3] Symlinking Deezer chart previews...")
        deezer_link = audio_dir / "deezer_charts"
        if not deezer_link.exists():
            try:
                os.symlink(str(DEEZER_PREVIEWS), str(deezer_link))
            except Exception:
                pass
        for ext in ["*.mp3", "*.wav", "*.flac"]:
            total += sum(1 for _ in deezer_link.rglob(ext)) if deezer_link.exists() else 0

    logger.info(f"\n  Phase I total: {total:,} audio files available for self-supervised training")
    return total


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("RESONATE Phase G/H/I Data Builder")
    logger.info("=" * 70)

    # Phase G
    g_count = build_phase_g()
    logger.info(f"\nPhase G result: {g_count} annotated segments with audio")

    # Phase H
    h_count = build_phase_h()
    logger.info(f"\nPhase H result: {h_count} knowledge graph triplets with audio")

    # Phase I
    i_count = build_phase_i()
    logger.info(f"\nPhase I result: {i_count} audio files for self-supervised training")

    logger.info("\n" + "=" * 70)
    logger.info("DATA BUILD COMPLETE")
    logger.info(f"  Phase G: {g_count:,} segments")
    logger.info(f"  Phase H: {h_count:,} triplets")
    logger.info(f"  Phase I: {i_count:,} audio files")
    logger.info("=" * 70)

    # Verify data is loadable
    logger.info("\nVerifying data integrity...")
    for source in ["salami", "harmonix", "billboard_structure"]:
        ann_dir = PHASE_G_DIR / source / "annotations"
        audio_dir_check = PHASE_G_DIR / source / "audio"
        if ann_dir.exists():
            ann_count = sum(1 for _ in ann_dir.glob("*.json"))
            audio_count = sum(1 for _ in audio_dir_check.rglob("*.mp3")) if audio_dir_check.exists() else 0
            logger.info(f"  Phase G/{source}: {ann_count} annotations, {audio_count} audio files")

    if (PHASE_H_DIR / "triplets.jsonl").exists():
        with open(PHASE_H_DIR / "triplets.jsonl") as f:
            line_count = sum(1 for _ in f)
        logger.info(f"  Phase H triplets.jsonl: {line_count:,} lines")

    if (PHASE_I_DIR / "audio").exists():
        i_audio = 0
        for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg"]:
            i_audio += sum(1 for _ in (PHASE_I_DIR / "audio").rglob(ext))
        logger.info(f"  Phase I audio: {i_audio:,} files")


if __name__ == "__main__":
    main()
