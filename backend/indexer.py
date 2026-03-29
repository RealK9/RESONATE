"""
RESONATE — Sample Indexer.
Background indexing and auto-organization of sample library.
"""

import json
import shutil
from pathlib import Path

from config import SAMPLE_DIR, SAMPLE_INDEX_FILE, AUDIO_EXT, SPLICE_DIRS, LOOPCLOUD_DIRS, PROFILE_DB_PATH, VECTOR_INDEX_DIR
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

v2_indexing_status = {"done": False, "total": 0, "processed": 0, "phase": "dsp"}

# Global vector index — loaded at startup, updated during embedding pass
_vector_index = None
_rpm_extractor = None


def get_vector_index():
    """Get the current vector index (may be None if not yet built)."""
    return _vector_index


def get_rpm_extractor():
    """Get the RPM extractor singleton (may be None if RPM model not available).

    Lazily instantiated on first call so the analyze routes can extract
    RPM embeddings for uploaded mixes without waiting for background indexing.
    """
    global _rpm_extractor
    if _rpm_extractor is not None:
        return _rpm_extractor

    try:
        from ml.embeddings.rpm_extractor import RPMExtractor
        from pathlib import Path as _Path
        rpm_models_dir = _Path("~/.resonate/rpm_models").expanduser()
        if (rpm_models_dir / "rpm_final.pt").exists() or (rpm_models_dir / "rpm_embedding.onnx").exists():
            # Load genre and instrument label mappings
            genre_labels = {}
            instrument_labels = []
            try:
                from ml.training.knowledge.genre_taxonomy import get_top_level_genres, get_genre_labels
                top_genres = get_top_level_genres()
                genre_labels = {i: name for i, name in enumerate(top_genres)}
                for gid, gname in get_genre_labels():
                    genre_labels[gid] = gname
            except Exception:
                pass
            try:
                from ml.training.knowledge.instruments import get_instrument_labels
                instrument_labels = get_instrument_labels()
            except Exception:
                pass

            _rpm_extractor = RPMExtractor(
                genre_labels=genre_labels,
                instrument_labels=instrument_labels,
            )
    except ImportError:
        pass

    return _rpm_extractor


def background_index_v2():
    """
    Two-phase background indexing using the Phase 1 analysis pipeline.

    Phase 1 (DSP): Extract core descriptors, spectral, harmonic, transient,
    perceptual features + classify role/genre/style. Fast (~34k in minutes).

    Phase 2 (Embeddings): Load CLAP/PANNs/AST models (~2-3GB download on
    first use), extract embeddings, build FAISS vector index. Slow but
    non-blocking — server is usable during this phase.
    """
    global v2_indexing_status, _vector_index, _rpm_extractor

    try:
        from ml.pipeline.batch_processor import BatchProcessor
        from ml.db.sample_store import SampleStore
        from ml.retrieval.vector_index import VectorIndex
    except ImportError as e:
        print(f"  ✗ V2 pipeline import error: {e}")
        v2_indexing_status["done"] = True
        return

    store = SampleStore(str(PROFILE_DB_PATH))
    store.init()

    # ── Phase 1: DSP features (fast) ─────────────────────────────────────
    v2_indexing_status["phase"] = "dsp"
    print("  ⟳ V2 Phase 1: Extracting DSP features...")

    processor = BatchProcessor(
        skip_embeddings=True,
        db_path=str(PROFILE_DB_PATH),
        max_workers=4,
    )

    dirs_to_index = [("local", SAMPLE_DIR)]
    dirs_to_index += [("splice", d) for d in SPLICE_DIRS if d.exists()]
    dirs_to_index += [("loopcloud", d) for d in LOOPCLOUD_DIRS if d.exists()]

    for source, d in dirs_to_index:
        if d.exists():
            result = processor.process_directory(str(d), source=source)
            print(f"  ✓ DSP: {result['processed']} {source} samples")

    total = store.count()
    print(f"  ✓ Phase 1 complete: {total} profiles with DSP features")

    # ── Detect RPM model (replaces CLAP + PANNs + AST) ─────────────────
    rpm_available = False
    rpm_extractor = None
    try:
        from ml.embeddings.rpm_extractor import RPMExtractor
        from pathlib import Path as _Path
        rpm_models_dir = _Path("~/.resonate/rpm_models").expanduser()
        if (rpm_models_dir / "rpm_final.pt").exists() or (rpm_models_dir / "rpm_embedding.onnx").exists():
            rpm_available = True
            print("  ✓ RPM model detected — using unified model (replaces CLAP+PANNs+AST)")
        else:
            print("  ℹ No RPM model found — falling back to legacy CLAP+PANNs+AST")
    except ImportError:
        print("  ℹ RPM extractor not available — using legacy pipeline")

    # ── Load existing vector index if available ──────────────────────────
    # RPM uses 768-d embeddings, legacy uses 512-d CLAP
    vector_index_subdir = "rpm" if rpm_available else "clap"
    vector_index_path = str(VECTOR_INDEX_DIR / vector_index_subdir)
    try:
        if (VECTOR_INDEX_DIR / vector_index_subdir / "index.faiss").exists():
            _vector_index = VectorIndex.load(vector_index_path)
            print(f"  ✓ Loaded FAISS index: {_vector_index.size()} vectors ({vector_index_subdir})")
    except Exception as e:
        print(f"  ⚠ Could not load existing FAISS index: {e}")

    # ── Phase 2: Embeddings ──────────────────────────────────────────────
    v2_indexing_status["phase"] = "embeddings"

    try:
        import numpy as np

        if rpm_available:
            # ── RPM path: single model, faster, purpose-built ────────────

            # Check if pre-built FAISS index exists (from GPU extraction)
            prebuilt_faiss = (VECTOR_INDEX_DIR / "rpm" / "index.faiss").exists()

            # Check how many profiles already have RPM embeddings
            _sample = store.list_all(limit=1)
            has_prebuilt_embeddings = (
                _sample and _sample[0].embeddings and
                _sample[0].embeddings.rpm and len(_sample[0].embeddings.rpm) == 768
            )

            if prebuilt_faiss and has_prebuilt_embeddings:
                # Pre-built embeddings exist (from GPU extraction) — skip CPU re-extraction
                print("  ✓ Pre-built RPM embeddings detected — skipping CPU extraction")
                try:
                    _vector_index = VectorIndex.load(vector_index_path)
                    print(f"  ✓ Loaded pre-built FAISS index: {_vector_index.size()} vectors (768-dim RPM)")
                except Exception as e:
                    print(f"  ⚠ Could not load pre-built FAISS index: {e}")
            else:
                # Need to extract embeddings on CPU
                print("  ⟳ V2 Phase 2: Loading RPM model for embeddings...")

                # Load genre and instrument label mappings
                genre_labels = {}
                instrument_labels = []
                try:
                    from ml.training.knowledge.genre_taxonomy import get_top_level_genres, get_genre_labels
                    top_genres = get_top_level_genres()
                    genre_labels = {i: name for i, name in enumerate(top_genres)}
                    for gid, gname in get_genre_labels():
                        genre_labels[gid] = gname
                except Exception as e:
                    print(f"  ⚠ Could not load genre labels: {e}")
                try:
                    from ml.training.knowledge.instruments import get_instrument_labels
                    instrument_labels = get_instrument_labels()
                except Exception as e:
                    print(f"  ⚠ Could not load instrument labels: {e}")

                rpm_extractor = RPMExtractor(
                    genre_labels=genre_labels,
                    instrument_labels=instrument_labels,
                )
                _rpm_extractor = rpm_extractor  # store globally for analyze routes

                processor_emb = BatchProcessor(
                    skip_embeddings=False,
                    rpm_extractor=rpm_extractor,
                    db_path=str(PROFILE_DB_PATH),
                    max_workers=1,
                )

                for source, d in dirs_to_index:
                    if d.exists():
                        result = processor_emb.process_directory(str(d), source=source)
                        print(f"  ✓ RPM Embeddings: {result['processed']} {source} samples")

                # ── Build FAISS vector index from RPM embeddings (768-d) ─────
                print("  ⟳ Building FAISS vector index from RPM embeddings...")
                all_profiles = store.list_all()
                rpm_dim = 768

                vi = VectorIndex(dim=rpm_dim)
                indexed = 0
                for profile in all_profiles:
                    if profile.embeddings and profile.embeddings.rpm:
                        vec = np.array(profile.embeddings.rpm, dtype=np.float32)
                        if vec.shape[0] == rpm_dim:
                            vi.add(profile.filepath, vec)
                            indexed += 1

                if indexed > 0:
                    vi.save(vector_index_path)
                    _vector_index = vi
                    print(f"  ✓ FAISS index built: {indexed} vectors ({rpm_dim}-dim RPM)")
                else:
                    print("  ⚠ No RPM embeddings found — vector index empty")

            # Lazy-load RPM extractor for analyze routes (on first mix upload)
            if _rpm_extractor is None:
                print("  ℹ RPM extractor will load on first mix upload (lazy)")

        else:
            # ── Legacy path: CLAP + PANNs + AST ─────────────────────────
            print("  ⟳ V2 Phase 2: Loading ML models for embeddings...")
            print("    (CLAP, PANNs, AST — ~2-3GB download on first use)")

            from ml.embeddings.embedding_manager import EmbeddingManager
            emb_manager = EmbeddingManager()

            processor_emb = BatchProcessor(
                skip_embeddings=False,
                embedding_manager=emb_manager,
                db_path=str(PROFILE_DB_PATH),
                max_workers=2,
            )

            for source, d in dirs_to_index:
                if d.exists():
                    result = processor_emb.process_directory(str(d), source=source)
                    print(f"  ✓ Embeddings: {result['processed']} {source} samples")

            # ── Build FAISS vector index from CLAP embeddings (512-d) ────
            print("  ⟳ Building FAISS vector index from CLAP embeddings...")
            all_profiles = store.list_all()
            clap_dim = 512

            vi = VectorIndex(dim=clap_dim)
            indexed = 0
            for profile in all_profiles:
                if profile.embeddings and profile.embeddings.clap_general:
                    vec = np.array(profile.embeddings.clap_general, dtype=np.float32)
                    if vec.shape[0] == clap_dim:
                        vi.add(profile.filepath, vec)
                        indexed += 1

            if indexed > 0:
                vi.save(vector_index_path)
                _vector_index = vi
                print(f"  ✓ FAISS index built: {indexed} vectors ({clap_dim}-dim CLAP)")
            else:
                print("  ⚠ No CLAP embeddings found — vector index empty")

    except Exception as e:
        print(f"  ⚠ Embedding phase failed (non-fatal): {e}")
        print("    Server will use filter-based recommendations without similarity search")
        import traceback
        traceback.print_exc()

    v2_indexing_status["done"] = True
    v2_indexing_status["phase"] = "complete"
    total = store.count()
    print(f"\n  ✓ V2 indexing complete: {total} total profiles\n")
