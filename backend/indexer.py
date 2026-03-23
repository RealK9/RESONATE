"""
RESONATE — Sample Indexer.
Background indexing and auto-organization of sample library.
"""

import json
import shutil
from pathlib import Path

from config import SAMPLE_DIR, SAMPLE_INDEX_FILE, AUDIO_EXT, SPLICE_DIRS, LOOPCLOUD_DIRS, PROFILE_DB_PATH
from utils import classify_type, file_hash
from analysis.sample_analyzer import analyze_sample


# ── Global state ────────────────────────────────────────────────────────────
sample_cache = {}
indexing_status = {"done": False, "total": 0, "processed": 0}


def load_disk_cache():
    """Load persistent sample index from disk."""
    if SAMPLE_INDEX_FILE.exists():
        try:
            data = json.loads(SAMPLE_INDEX_FILE.read_text())
            sample_cache.clear()
            sample_cache.update(data)
            print(f"  ✓ Loaded {len(sample_cache)} samples from disk cache")
        except Exception as e:
            print(f"  ✗ Cache load error: {e}")
            sample_cache.clear()


def save_disk_cache():
    """Save sample index to disk."""
    try:
        SAMPLE_INDEX_FILE.write_text(json.dumps(sample_cache))
    except Exception as e:
        print(f"  ✗ Cache save error: {e}")


def auto_organize_samples():
    """
    Auto-categorize all samples into proper subfolders based on filename analysis.
    Runs once on startup. Only moves files that are in the root or a generic folder.
    """
    TYPE_FOLDERS = {
        "melody": "Melody", "vocals": "Vocals", "hihat": "Hi-Hats",
        "pad": "Pads", "strings": "Strings", "fx": "FX",
        "percussion": "Percussion", "bass": "Bass", "kick": "Kick",
        "snare": "Snare", "unknown": "Other",
    }

    GENERIC_FOLDERS = {"sounds", "packs", "samples", "audio", "all", "misc", "uncategorized", "other"}

    ORGANIZED_FOLDERS = {v.lower() for v in TYPE_FOLDERS.values()}
    ORGANIZED_FOLDERS.update({"guitar", "synth", "piano", "vocals", "hi-hats",
                              "hihats", "clap", "drums", "leads", "fillers"})

    all_files = [f for f in SAMPLE_DIR.rglob("*") if f.suffix.lower() in AUDIO_EXT and f.is_file()]
    if not all_files:
        return

    needs_organizing = False
    for fp in all_files[:50]:
        try:
            rel = fp.relative_to(SAMPLE_DIR)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) <= 1:
            needs_organizing = True
            break
        parent_folder = parts[0].lower()
        if parent_folder in GENERIC_FOLDERS:
            needs_organizing = True
            break

    if not needs_organizing:
        print(f"  Samples already organized ({len(all_files)} files)")
        return

    print(f"  Auto-organizing {len(all_files)} samples into category folders...")
    moved = 0
    for fp in all_files:
        try:
            stype = classify_type(fp)
            name_lower = fp.stem.lower()
            if stype == "melody":
                if "guitar" in name_lower:
                    folder = "Guitar"
                elif "piano" in name_lower or "keys" in name_lower or "rhodes" in name_lower:
                    folder = "Piano"
                elif "synth" in name_lower:
                    folder = "Synth"
                elif "lead" in name_lower:
                    folder = "Leads"
                elif "flute" in name_lower or "sax" in name_lower or "brass" in name_lower or "horn" in name_lower:
                    folder = "Brass-Wind"
                else:
                    folder = "Melody"
            elif stype == "snare" and "clap" in name_lower:
                folder = "Clap"
            elif stype == "snare":
                folder = "Snare"
            else:
                folder = TYPE_FOLDERS.get(stype, "Other")

            try:
                rel = fp.relative_to(SAMPLE_DIR)
            except ValueError:
                continue
            current_parent = rel.parts[0] if len(rel.parts) > 1 else None

            if current_parent and current_parent == folder:
                continue
            if current_parent and current_parent.lower() in ORGANIZED_FOLDERS:
                continue

            target_dir = SAMPLE_DIR / folder
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / fp.name

            if target_path.exists():
                stem = fp.stem
                suffix = fp.suffix
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            shutil.move(str(fp), str(target_path))
            moved += 1
        except Exception as e:
            print(f"    Error moving {fp.name}: {e}")

    for d in sorted(SAMPLE_DIR.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass

    if moved > 0:
        print(f"  ✓ Organized {moved} samples into category folders")
        if SAMPLE_INDEX_FILE.exists():
            SAMPLE_INDEX_FILE.unlink()
            print(f"  Cache cleared (paths changed)")
    else:
        print(f"  No samples needed moving")


def _discover_external_libraries():
    """Find audio files in Splice and Loopcloud local libraries."""
    external_files = []

    # Splice
    for splice_dir in SPLICE_DIRS:
        if splice_dir.exists():
            found = [f for f in splice_dir.rglob("*") if f.suffix.lower() in AUDIO_EXT and f.is_file()]
            for f in found:
                external_files.append((f, "splice"))
            if found:
                print(f"  ✓ Splice: found {len(found)} samples in {splice_dir}")

    # Loopcloud
    for lc_dir in LOOPCLOUD_DIRS:
        if lc_dir.exists():
            found = [f for f in lc_dir.rglob("*") if f.suffix.lower() in AUDIO_EXT and f.is_file()]
            for f in found:
                external_files.append((f, "loopcloud"))
            if found:
                print(f"  ✓ Loopcloud: found {len(found)} samples in {lc_dir}")

    return external_files


def background_index():
    """Index all samples in background thread on startup."""
    global sample_cache, indexing_status

    # Local samples
    local_files = sorted(
        f for f in SAMPLE_DIR.rglob("*")
        if f.suffix.lower() in AUDIO_EXT and f.is_file()
    )

    # External libraries (Splice, Loopcloud)
    external = _discover_external_libraries()

    # Combine: local files tagged as "local", plus external
    all_indexed = [(f, "local") for f in local_files] + external
    indexing_status["total"] = len(all_indexed)

    new_count = 0
    cached_count = 0
    for i, (fp, source) in enumerate(all_indexed):
        cache_key = str(fp)
        fh = file_hash(fp)

        existing = sample_cache.get(cache_key)
        if existing and existing.get("_hash") == fh:
            cached_count += 1
            indexing_status["processed"] = i + 1
            continue

        try:
            sa = analyze_sample(fp)
            if sa:
                sa["_hash"] = fh
                sa["source"] = source
                # Add mood/energy if not already present
                if "mood" not in sa:
                    from analysis.sample_analyzer import classify_mood
                    mood_data = classify_mood(
                        sa.get("rms", 0),
                        sa.get("spectral_centroid", 0),
                        sa.get("key", "N/A"),
                        sa.get("frequency_bands", {}),
                    )
                    sa["mood"] = mood_data["mood"]
                    sa["energy"] = mood_data["energy"]
                sample_cache[cache_key] = sa
                new_count += 1
        except Exception as e:
            print(f"  Index error {fp.name}: {e}")

        indexing_status["processed"] = i + 1

        if new_count > 0 and new_count % 50 == 0:
            save_disk_cache()

    if new_count > 0:
        save_disk_cache()

    indexing_status["done"] = True
    source_counts = {}
    for sa in sample_cache.values():
        src = sa.get("source", "local")
        source_counts[src] = source_counts.get(src, 0) + 1
    source_str = ", ".join(f"{v} {k}" for k, v in sorted(source_counts.items()))
    print(f"\n  ✓ Indexing complete: {cached_count} cached, {new_count} new, {len(all_indexed)} total ({source_str})\n")


# ── V2 Pipeline Indexing ──────────────────────────────────────────────────

v2_indexing_status = {"done": False, "total": 0, "processed": 0}


def background_index_v2():
    """Background indexing using the new Phase 1 analysis pipeline."""
    global v2_indexing_status

    try:
        from backend.ml.pipeline.batch_processor import BatchProcessor
        from backend.ml.db.sample_store import SampleStore
    except ImportError as e:
        print(f"  ✗ V2 pipeline import error: {e}")
        v2_indexing_status["done"] = True
        return

    store = SampleStore(str(PROFILE_DB_PATH))
    store.init()

    processor = BatchProcessor(
        skip_embeddings=True,  # Start without embeddings for speed
        db_path=str(PROFILE_DB_PATH),
        max_workers=4,
    )

    # Index local samples
    if SAMPLE_DIR.exists():
        result = processor.process_directory(str(SAMPLE_DIR), source="local")
        print(f"  ✓ Indexed {result['processed']} local samples (v2 pipeline)")

    # Index external libraries
    for d in SPLICE_DIRS:
        if d.exists():
            result = processor.process_directory(str(d), source="splice")
            print(f"  ✓ Indexed {result['processed']} Splice samples (v2 pipeline)")
    for d in LOOPCLOUD_DIRS:
        if d.exists():
            result = processor.process_directory(str(d), source="loopcloud")
            print(f"  ✓ Indexed {result['processed']} Loopcloud samples (v2 pipeline)")

    v2_indexing_status["done"] = True
    total = store.count()
    print(f"\n  ✓ V2 indexing complete: {total} total profiles\n")
