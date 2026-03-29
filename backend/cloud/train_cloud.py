#!/usr/bin/env python3
"""
RESONATE Production Model — Cloud GPU Training Script.

God-tier training pipeline: runs all 4 phases on CUDA GPU with full optimizations.
  - Mixed precision (AMP) for 2x speed on RTX 5090 / A100
  - Multi-worker data loading with prefetch
  - Gradient accumulation for large effective batch sizes
  - Automatic dataset downloading (NSynth 289k, FMA 8k, Jamendo 55k)
  - Compact embeddings loader (500MB binary vs 66GB JSON)
  - Spotify enrichment for chart intelligence
  - Resume from any checkpoint

Usage:
    # Full training from scratch (downloads everything):
    python3 cloud/train_cloud.py --phases all

    # Resume Phase C+D from checkpoint:
    python3 cloud/train_cloud.py --phases cd --checkpoint /workspace/resonate/checkpoints/rpm_phaseB_done.pt

    # Just Phase C (large-scale):
    python3 cloud/train_cloud.py --phases c

    # Download datasets only:
    python3 cloud/train_cloud.py --download-only

    # Run Spotify enrichment only:
    python3 cloud/train_cloud.py --enrich-only
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rpm_cloud_training.log", mode="a"),
    ],
)
logger = logging.getLogger("RPM-Cloud")


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

WORK_DIR = Path(os.environ.get("RESONATE_WORK_DIR", "/workspace/resonate"))
DATA_DIR = WORK_DIR / "data"
PRECOMPUTED_DIR = DATA_DIR / "precomputed"
COMPACT_DIR = DATA_DIR / "precomputed_compact"
DATASETS_DIR = DATA_DIR / "datasets"
CHARTS_DIR = DATA_DIR / "charts"
CONFIG_DIR = DATA_DIR / "config"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
OUTPUT_DIR = WORK_DIR / "output"
LOG_DIR = WORK_DIR / "logs"

# Advanced phase data directories (Phase E-I)
ADVANCED_DATA_DIR = DATA_DIR / "advanced"
PHASE_E_DIR = ADVANCED_DATA_DIR / "phase_e"
PHASE_F_DIR = ADVANCED_DATA_DIR / "phase_f"
PHASE_G_DIR = ADVANCED_DATA_DIR / "phase_g"
PHASE_H_DIR = ADVANCED_DATA_DIR / "phase_h"
PHASE_I_DIR = ADVANCED_DATA_DIR / "phase_i"

# Ensure code is importable
BACKEND_DIR = WORK_DIR / "backend"
if BACKEND_DIR.exists():
    sys.path.insert(0, str(BACKEND_DIR))
else:
    # Running from within backend/
    sys.path.insert(0, str(Path(__file__).parent.parent))


def ensure_dirs():
    for d in [DATA_DIR, PRECOMPUTED_DIR, DATASETS_DIR, CHARTS_DIR,
              CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR,
              ADVANCED_DATA_DIR, PHASE_E_DIR, PHASE_F_DIR,
              PHASE_G_DIR, PHASE_H_DIR, PHASE_I_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Dataset Downloads (cloud network = fast)
# ══════════════════════════════════════════════════════════════════════

def download_datasets():
    """Download all training datasets on the cloud instance."""
    import subprocess

    # ── FMA-small (8k tracks, 7.2GB) ──
    fma_dir = DATASETS_DIR / "fma" / "fma_small"
    if not fma_dir.exists():
        logger.info("Downloading FMA-small (7.2GB)...")
        fma_zip = DATASETS_DIR / "fma" / "fma_small.zip"
        (DATASETS_DIR / "fma").mkdir(parents=True, exist_ok=True)

        subprocess.run([
            "wget", "-q", "--show-progress",
            "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
            "-O", str(fma_zip),
        ], check=True)

        logger.info("Extracting FMA-small...")
        subprocess.run(["unzip", "-q", str(fma_zip), "-d", str(DATASETS_DIR / "fma")], check=True)
        fma_zip.unlink()
        logger.info(f"FMA-small ready: {fma_dir}")

    # FMA metadata
    fma_meta = DATASETS_DIR / "fma" / "fma_metadata"
    if not fma_meta.exists():
        logger.info("Downloading FMA metadata...")
        meta_zip = DATASETS_DIR / "fma" / "fma_metadata.zip"
        subprocess.run([
            "wget", "-q", "--show-progress",
            "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
            "-O", str(meta_zip),
        ], check=True)
        subprocess.run(["unzip", "-q", str(meta_zip), "-d", str(DATASETS_DIR / "fma")], check=True)
        meta_zip.unlink()

    # ── NSynth (289k notes, ~25GB) ──
    for split, size in [("train", "25GB"), ("valid", "3GB"), ("test", "1GB")]:
        nsynth_split = DATASETS_DIR / "nsynth" / f"nsynth-{split}"
        if not nsynth_split.exists():
            logger.info(f"Downloading NSynth-{split} ({size})...")
            tar_path = DATASETS_DIR / "nsynth" / f"nsynth-{split}.jsonwav.tar.gz"
            (DATASETS_DIR / "nsynth").mkdir(parents=True, exist_ok=True)

            subprocess.run([
                "wget", "-q", "--show-progress",
                f"http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{split}.jsonwav.tar.gz",
                "-O", str(tar_path),
            ], check=True)

            logger.info(f"Extracting NSynth-{split}...")
            subprocess.run(["tar", "xzf", str(tar_path), "-C", str(DATASETS_DIR / "nsynth")], check=True)
            tar_path.unlink()
            logger.info(f"NSynth-{split} ready")

    # ── MTG-Jamendo (55k tracks, ~30GB) ──
    jamendo_dir = DATASETS_DIR / "jamendo"
    if not jamendo_dir.exists():
        logger.info("Downloading MTG-Jamendo dataset (~30GB)...")
        jamendo_dir.mkdir(parents=True, exist_ok=True)

        # Download tag annotations
        for tag_file in ["autotagging_genre.tsv", "autotagging_instrument.tsv",
                         "autotagging_moodtheme.tsv"]:
            subprocess.run([
                "wget", "-q",
                f"https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/splits/{tag_file}",
                "-O", str(jamendo_dir / tag_file),
            ], check=False)  # Non-fatal if annotations unavailable

        # Download audio splits
        audio_dir = jamendo_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        # Jamendo provides audio via their API or bulk downloads
        # Try the raw audio pack first
        raw_tar = jamendo_dir / "mtg-jamendo-raw.tar.gz"
        result = subprocess.run([
            "wget", "-q", "--show-progress",
            "https://zenodo.org/records/3826813/files/raw_30s.zip",
            "-O", str(jamendo_dir / "raw_30s.zip"),
        ], check=False)

        if result.returncode == 0 and (jamendo_dir / "raw_30s.zip").exists():
            logger.info("Extracting Jamendo audio...")
            subprocess.run([
                "unzip", "-q", str(jamendo_dir / "raw_30s.zip"),
                "-d", str(audio_dir),
            ], check=False)
            (jamendo_dir / "raw_30s.zip").unlink(missing_ok=True)
            logger.info("MTG-Jamendo ready")
        else:
            logger.warning("Jamendo audio download failed — will train without it")


# ══════════════════════════════════════════════════════════════════════
# Spotify Enrichment (cloud has fast network for API calls + downloads)
# ══════════════════════════════════════════════════════════════════════

def run_chart_enrichment():
    """
    Enrich chart entries with Deezer previews + metadata.
    Deezer API is free, no auth required, and provides 30-second previews.
    Falls back to Spotify if Deezer misses entries.
    """
    chart_db = CHARTS_DIR / "chart_features.db"
    if not chart_db.exists():
        logger.warning("No chart database found — skipping enrichment")
        return

    preview_dir = CHARTS_DIR / "previews"
    preview_dir.mkdir(exist_ok=True)

    # ── Deezer enrichment (primary — free, no auth) ──
    logger.info("Running Deezer enrichment on chart entries...")
    try:
        from ml.training.charts.deezer_enrichment import DeezerEnricher

        enricher = DeezerEnricher(db_path=str(chart_db))
        stats = enricher.enrich_chart_entries(output_dir=str(preview_dir))
        enricher.close()
        logger.info(f"  Deezer: {stats}")
    except Exception as e:
        logger.warning(f"Deezer enrichment failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Spotify fallback (for entries Deezer missed) ──
    env_file = CONFIG_DIR / "spotify.env"
    if env_file.exists():
        for line in env_file.read_text().strip().split("\n"):
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    if client_id:
        logger.info("Running Spotify enrichment (fallback for Deezer misses)...")
        try:
            from ml.training.charts.spotify_enrichment import SpotifyEnricher
            from ml.training.charts.billboard_scraper import BillboardScraper, ChartEntry

            scraper = BillboardScraper.load_from_db(str(chart_db))
            all_songs = scraper.get_unique_songs()

            entries = [
                ChartEntry(title=t, artist=a, peak_position=p, weeks_on_chart=0, year=y, chart_name=c)
                for t, a, p, y, c in all_songs
            ]

            enricher = SpotifyEnricher(db_path=str(chart_db))
            enricher.enrich_chart_entries(entries, output_dir=str(preview_dir), download_previews=True)
            enricher.close()
        except Exception as e:
            logger.warning(f"Spotify enrichment failed: {e}")

    # ── Summary ──
    import sqlite3
    conn = sqlite3.connect(str(chart_db))
    total = conn.execute("SELECT COUNT(*) FROM chart_entries").fetchone()[0]
    with_preview = conn.execute(
        "SELECT COUNT(*) FROM chart_entries WHERE preview_path IS NOT NULL AND preview_path != ''"
    ).fetchone()[0]
    with_deezer = conn.execute(
        "SELECT COUNT(*) FROM chart_entries WHERE deezer_id IS NOT NULL AND deezer_id > 0"
    ).fetchone()[0]
    conn.close()
    logger.info(f"Chart enrichment summary: {with_preview}/{total} with previews, {with_deezer} via Deezer")


# ══════════════════════════════════════════════════════════════════════
# Compact Embeddings Loader
# ══════════════════════════════════════════════════════════════════════

def load_compact_embeddings():
    """
    Load pre-computed teacher embeddings from compact binary format.
    Returns list of TrainingSample objects with teacher embeddings attached.
    """
    import numpy as np
    from ml.training.rpm_dataset import TrainingSample

    compact_file = COMPACT_DIR / "embeddings.npz"
    metadata_file = COMPACT_DIR / "metadata.jsonl"

    if not compact_file.exists():
        logger.info("No compact embeddings found — skipping local samples")
        return []

    logger.info("Loading compact teacher embeddings...")
    t0 = time.time()

    data = np.load(str(compact_file))
    clap = data["clap"]       # [N, 512]
    panns = data["panns"]     # [N, 2048]
    ast = data["ast"]         # [N, 768]
    role_ids = data["role_ids"]  # [N]

    # Load metadata
    import json
    filepaths = []
    roles_str = []
    with open(metadata_file) as f:
        for line in f:
            meta = json.loads(line)
            filepaths.append(meta["filepath"])
            roles_str.append(meta["role"])

    role_map = {
        "kick": 0, "snare": 1, "clap": 2, "hat": 3, "perc": 4,
        "bass": 5, "lead": 6, "pad": 7, "fx": 8, "texture": 9, "vocal": 10,
    }

    samples = []
    for i in range(len(filepaths)):
        # Local audio files won't exist on cloud — samples are embedding-only
        # The training loop will use teacher embeddings for distillation loss
        sample = TrainingSample(
            filepath=filepaths[i],
            source="local_compact",
        )
        rid = int(role_ids[i])
        sample.role = rid if rid >= 0 else role_map.get(roles_str[i])
        sample.clap_embedding = clap[i]
        sample.panns_embedding = panns[i]
        sample.ast_embedding = ast[i]
        samples.append(sample)

    elapsed = time.time() - t0
    logger.info(f"  Loaded {len(samples):,} samples with embeddings in {elapsed:.1f}s")
    return samples


# ══════════════════════════════════════════════════════════════════════
# Advanced Phase E-I Data Downloads
# ══════════════════════════════════════════════════════════════════════

def download_advanced_datasets():
    """Download data for Phases E-I on the cloud instance."""
    logger.info("Downloading advanced phase datasets (E-I)...")

    # Phase E: Text-audio contrastive data
    try:
        from ml.training.data_pipelines.phase_e_contrastive import (
            WavCapsDownloader, MusicCapsDownloader,
        )
        logger.info("Phase E: Downloading WavCaps + MusicCaps...")
        try:
            wc = WavCapsDownloader(data_root=PHASE_E_DIR)
            wc.download()
        except Exception as e:
            logger.warning(f"  WavCaps download failed: {e}")
        try:
            mc = MusicCapsDownloader(data_root=PHASE_E_DIR)
            mc.download()
        except Exception as e:
            logger.warning(f"  MusicCaps download failed: {e}")
    except ImportError as e:
        logger.warning(f"Phase E download skipped (import error): {e}")

    # Phase F: Stem data
    try:
        from ml.training.data_pipelines.phase_f_stems import MUSDB18Downloader
        logger.info("Phase F: Downloading MUSDB18-HQ...")
        try:
            dl = MUSDB18Downloader(data_root=PHASE_F_DIR)
            dl.download()
        except Exception as e:
            logger.warning(f"  MUSDB18 download failed: {e}")
    except ImportError as e:
        logger.warning(f"Phase F download skipped (import error): {e}")

    # Phase G: Structure annotations
    try:
        from ml.training.data_pipelines.phase_g_structure import SALAMIDownloader
        logger.info("Phase G: Downloading SALAMI annotations...")
        try:
            dl = SALAMIDownloader(data_root=PHASE_G_DIR)
            dl.download()
        except Exception as e:
            logger.warning(f"  SALAMI download failed: {e}")
    except ImportError as e:
        logger.warning(f"Phase G download skipped (import error): {e}")

    # Phase H: Knowledge graph
    try:
        from ml.training.data_pipelines.phase_h_knowledge_graph import MusicBrainzClient
        logger.info("Phase H: Building knowledge graph from MusicBrainz...")
        try:
            client = MusicBrainzClient(data_root=PHASE_H_DIR)
            client.build_graph()
        except Exception as e:
            logger.warning(f"  Knowledge graph build failed: {e}")
    except ImportError as e:
        logger.warning(f"Phase H download skipped (import error): {e}")

    # Phase I: Self-supervised audio (reuses FMA + Jamendo from Phase A-D)
    logger.info("Phase I: Reuses FMA/Jamendo audio from main datasets — no extra download needed.")

    logger.info("Advanced dataset download complete.")


# ══════════════════════════════════════════════════════════════════════
# Advanced Training (Phases E-I)
# ══════════════════════════════════════════════════════════════════════

def run_advanced_training(phases: str, checkpoint_path: str = None):
    """Run Phase E-I advanced training."""
    import torch
    from ml.training.rpm_model import RPMModel, RPMConfig, count_parameters
    from ml.training.advanced_training import (
        AdvancedRPMModel,
        AdvancedTrainer,
        AdvancedTrainingConfig,
    )

    # ── Device ──
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Falling back to CPU (will be very slow).")
        device = "cpu"
    else:
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("  TF32 + cuDNN benchmark enabled")

    # ── Batch sizes based on VRAM ──
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 70:  # H100 80GB
            bs_e = 128
            bs_f = 48
            bs_g = 64
            bs_h = 32
            bs_i = 48
            num_workers = 8
        elif vram >= 40:  # A100 40GB+
            bs_e = 64
            bs_f = 32
            bs_g = 48
            bs_h = 24
            bs_i = 32
            num_workers = 8
        elif vram >= 20:  # RTX 4090 24GB
            bs_e = 32
            bs_f = 16
            bs_g = 24
            bs_h = 12
            bs_i = 16
            num_workers = 6
        else:
            bs_e = 16
            bs_f = 8
            bs_g = 16
            bs_h = 8
            bs_i = 8
            num_workers = 4
    else:
        bs_e = 8
        bs_f = 4
        bs_g = 8
        bs_h = 4
        bs_i = 4
        num_workers = 0

    logger.info(f"Advanced batch sizes: E={bs_e}, F={bs_f}, G={bs_g}, H={bs_h}, I={bs_i}")

    # ── Find base model checkpoint ──
    base_checkpoint = checkpoint_path
    if base_checkpoint is None:
        # Look for best Phase A-D checkpoint
        for name in ["rpm_final.pt", "rpm_phaseD_done.pt", "rpm_phaseC_done.pt", "rpm_best.pt"]:
            candidate = CHECKPOINT_DIR / name
            if candidate.exists():
                base_checkpoint = str(candidate)
                logger.info(f"Auto-detected base checkpoint: {base_checkpoint}")
                break
        # Also check output dir
        if base_checkpoint is None:
            final = OUTPUT_DIR / "rpm_final.pt"
            if final.exists():
                base_checkpoint = str(final)
                logger.info(f"Auto-detected base checkpoint: {base_checkpoint}")

    if base_checkpoint is None:
        logger.error("No Phase A-D checkpoint found! Train phases A-D first.")
        logger.error("  Expected: rpm_final.pt or rpm_phaseD_done.pt in checkpoints/")
        sys.exit(1)

    # ── Build advanced model ──
    logger.info(f"Building AdvancedRPMModel from {base_checkpoint}...")
    model = AdvancedRPMModel.from_checkpoint(
        base_checkpoint, device=device, rpm_config=RPMConfig()
    )

    params = count_parameters(model)
    logger.info(f"Advanced model: {params['total']:,} total, {params['trainable']:,} trainable")

    feature_extractor = model.base_model.feature_extractor

    # ── Phase I data: symlink FMA/Jamendo into phase_i dir ──
    # Phase I reuses existing audio datasets for self-supervised learning
    if "i" in phases:
        fma_audio = DATASETS_DIR / "fma" / "fma_small"
        jamendo_audio = DATASETS_DIR / "jamendo" / "audio"
        for src, name in [(fma_audio, "fma"), (jamendo_audio, "jamendo")]:
            dst = PHASE_I_DIR / name
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                    logger.info(f"  Symlinked {src} -> {dst}")
                except Exception:
                    pass

    # ── Build training config ──
    train_cfg = AdvancedTrainingConfig(
        output_dir=str(OUTPUT_DIR),
        checkpoint_dir=str(CHECKPOINT_DIR),
        log_dir=str(LOG_DIR),
        device=device,

        phase_e_batch_size=bs_e,
        phase_f_batch_size=bs_f,
        phase_g_batch_size=bs_g,
        phase_h_batch_size=bs_h,
        phase_i_batch_size=bs_i,

        phase_e_data_dir=str(PHASE_E_DIR),
        phase_f_data_dir=str(PHASE_F_DIR),
        phase_g_data_dir=str(PHASE_G_DIR),
        phase_h_data_dir=str(PHASE_H_DIR),
        phase_i_data_dir=str(PHASE_I_DIR),

        advanced_num_workers=num_workers,
    )

    # ── Run training ──
    trainer = AdvancedTrainer(train_cfg)
    phase_str = phases.upper().replace("A", "").replace("B", "").replace("C", "").replace("D", "")
    if not phase_str:
        phase_str = "EFGHI"

    t0 = time.time()
    trainer.train_advanced(model, phases=phase_str, feature_extractor=feature_extractor)
    elapsed = (time.time() - t0) / 60
    logger.info(f"Advanced training complete in {elapsed:.1f} minutes")

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════
# Training (Phases A-D)
# ══════════════════════════════════════════════════════════════════════

def run_training(phases: str, checkpoint_path: str = None):
    """Run specified training phases."""
    import torch
    import numpy as np
    from ml.training.rpm_model import RPMModel, RPMConfig, RPMLoss, count_parameters
    from ml.training.rpm_dataset import (
        DatasetConfig, RPMDataset, build_dataloader,
    )
    from ml.training.rpm_trainer import RPMTrainer, TrainingConfig

    # ── Device ──
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Falling back to CPU (will be very slow).")
        device = "cpu"
    else:
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")

        # Enable TF32 for Ampere+ GPUs (RTX 30xx, 40xx, 50xx, A100, H100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("  TF32 + cuDNN benchmark enabled")

    # ── Batch sizes based on VRAM ──
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 40:  # A100 40GB+, H100
            bs_frozen = 64
            bs_unfrozen = 32
            num_workers = 8
            grad_accum = 2  # effective = 64
        elif vram >= 28:  # RTX 5090 (32GB), A5000
            bs_frozen = 64
            bs_unfrozen = 24
            num_workers = 8
            grad_accum = 3  # effective = 72
        elif vram >= 20:  # RTX 4090 (24GB)
            bs_frozen = 48
            bs_unfrozen = 16
            num_workers = 6
            grad_accum = 4  # effective = 64
        elif vram >= 10:  # RTX 3080
            bs_frozen = 32
            bs_unfrozen = 8
            num_workers = 4
            grad_accum = 8  # effective = 64
        else:
            bs_frozen = 16
            bs_unfrozen = 4
            num_workers = 2
            grad_accum = 16
    else:
        bs_frozen = 8
        bs_unfrozen = 4
        num_workers = 0
        grad_accum = 8

    logger.info(f"Batch sizes: frozen={bs_frozen}, unfrozen={bs_unfrozen}, "
                f"grad_accum={grad_accum}, workers={num_workers}")

    # ── Build model ──
    logger.info("Building RPM model...")
    cfg = RPMConfig()
    model = RPMModel(cfg).to(device)
    _ = model.backbone  # force load AST

    params = count_parameters(model)
    logger.info(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # ── Load checkpoint ──
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint loaded successfully")

    feature_extractor = model.feature_extractor

    # ── Load all available data sources ──
    all_samples = []

    # 1. Local pre-computed samples (compact binary format)
    compact_samples = load_compact_embeddings()
    if compact_samples:
        all_samples.extend(compact_samples)
        logger.info(f"  Local (compact): {len(compact_samples):,}")

    # 2. Local pre-computed samples (JSON fallback)
    if not compact_samples and PRECOMPUTED_DIR.exists() and any(PRECOMPUTED_DIR.glob("*.json")):
        from ml.training.rpm_dataset import LocalSampleLoader
        logger.info("Loading pre-computed local samples (JSON)...")
        loader = LocalSampleLoader(str(PRECOMPUTED_DIR), str(PRECOMPUTED_DIR))
        local_samples = loader.load()
        all_samples.extend(local_samples)
        logger.info(f"  Local (JSON): {len(local_samples):,}")

    # 3. NSynth
    nsynth_dir = DATASETS_DIR / "nsynth"
    if (nsynth_dir / "nsynth-train").exists():
        try:
            from ml.training.rpm_dataset import NSynthLoader
            nsynth_loader = NSynthLoader(str(nsynth_dir))
            nsynth_samples = nsynth_loader.load()
            all_samples.extend(nsynth_samples)
            logger.info(f"  NSynth: {len(nsynth_samples):,}")
        except Exception as e:
            logger.warning(f"Failed to load NSynth: {e}")

    # 4. FMA
    fma_dir = DATASETS_DIR / "fma"
    if (fma_dir / "fma_small").exists():
        try:
            from ml.training.rpm_dataset import FMALoader
            fma_loader = FMALoader(str(fma_dir))
            fma_samples = fma_loader.load()
            all_samples.extend(fma_samples)
            logger.info(f"  FMA: {len(fma_samples):,}")
        except Exception as e:
            logger.warning(f"Failed to load FMA: {e}")

    # 5. MTG-Jamendo
    jamendo_dir = DATASETS_DIR / "jamendo"
    if jamendo_dir.exists() and (jamendo_dir / "audio").exists():
        try:
            from ml.training.rpm_dataset import JamendoLoader
            jamendo_loader = JamendoLoader(str(jamendo_dir))
            jamendo_samples = jamendo_loader.load()
            all_samples.extend(jamendo_samples)
            logger.info(f"  Jamendo: {len(jamendo_samples):,}")
        except Exception as e:
            logger.warning(f"Failed to load Jamendo: {e}")

    # 6. Chart previews (for Phase D — need Spotify enrichment first)
    chart_db = CHARTS_DIR / "chart_features.db"
    chart_samples = []
    if chart_db.exists():
        try:
            from ml.training.rpm_dataset import ChartPreviewLoader
            chart_loader = ChartPreviewLoader(str(CHARTS_DIR))
            chart_samples = chart_loader.load()
            # Chart samples go into Phase D, not the main training pool
            logger.info(f"  Chart previews: {len(chart_samples):,} (for Phase D)")
        except Exception as e:
            logger.warning(f"Failed to load chart previews: {e}")

    logger.info(f"\n{'='*70}")
    logger.info(f"TOTAL TRAINING SAMPLES: {len(all_samples):,}")
    if chart_samples:
        logger.info(f"  + {len(chart_samples):,} chart samples for Phase D")
    logger.info(f"{'='*70}\n")

    if not all_samples:
        logger.error("No training samples found!")
        sys.exit(1)

    # ── Train/Val split ──
    random.seed(42)
    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.95)  # 95/5 on large datasets
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]
    logger.info(f"Train: {len(train_samples):,}, Val: {len(val_samples):,}")

    # ════════════════════════════════════════════════════════════════
    # Phase A: Knowledge Distillation
    # ════════════════════════════════════════════════════════════════

    if "a" in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE A: Knowledge Distillation (backbone frozen)")
        logger.info("=" * 70)

        data_cfg = DatasetConfig(
            local_samples_dir=str(PRECOMPUTED_DIR),
            local_profiles_dir=str(PRECOMPUTED_DIR),
            batch_size=bs_frozen,
            num_workers=num_workers,
            pin_memory=True,
        )
        train_ds = RPMDataset(train_samples, data_cfg, feature_extractor, augment=True)
        val_ds = RPMDataset(val_samples, data_cfg, feature_extractor, augment=False)
        train_dl = build_dataloader(train_ds, data_cfg, shuffle=True)
        val_dl = build_dataloader(val_ds, data_cfg, shuffle=False)

        train_cfg = TrainingConfig(
            output_dir=str(OUTPUT_DIR),
            checkpoint_dir=str(CHECKPOINT_DIR),
            log_dir=str(LOG_DIR),
            device=device,
            phase_a_epochs=5,
            phase_a_batch_size=bs_frozen,
        )
        trainer = RPMTrainer(train_cfg)

        t0 = time.time()
        trainer.train_phase_a(model, train_dl, val_dl)
        logger.info(f"Phase A done in {(time.time()-t0)/60:.1f} min")

        torch.save(model.state_dict(), CHECKPOINT_DIR / "rpm_phaseA_done.pt")
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    # ════════════════════════════════════════════════════════════════
    # Phase B: Multi-Task (backbone frozen)
    # ════════════════════════════════════════════════════════════════

    if "b" in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE B: Multi-Task Training (backbone frozen)")
        logger.info("=" * 70)

        data_cfg = DatasetConfig(
            local_samples_dir=str(PRECOMPUTED_DIR),
            local_profiles_dir=str(PRECOMPUTED_DIR),
            batch_size=bs_frozen,
            num_workers=num_workers,
            pin_memory=True,
        )
        train_ds = RPMDataset(train_samples, data_cfg, feature_extractor, augment=True)
        val_ds = RPMDataset(val_samples, data_cfg, feature_extractor, augment=False)
        train_dl = build_dataloader(train_ds, data_cfg, shuffle=True)
        val_dl = build_dataloader(val_ds, data_cfg, shuffle=False)

        train_cfg = TrainingConfig(
            output_dir=str(OUTPUT_DIR),
            checkpoint_dir=str(CHECKPOINT_DIR),
            log_dir=str(LOG_DIR),
            device=device,
            phase_b_epochs=8,
            phase_b_batch_size=bs_frozen,
        )
        trainer = RPMTrainer(train_cfg)

        t0 = time.time()
        trainer.train_phase_b(model, train_dl, val_dl)
        logger.info(f"Phase B done in {(time.time()-t0)/60:.1f} min")

        torch.save(model.state_dict(), CHECKPOINT_DIR / "rpm_phaseB_done.pt")
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    # ════════════════════════════════════════════════════════════════
    # Phase C: Large-Scale Training (backbone UNFROZEN, AMP)
    # ════════════════════════════════════════════════════════════════

    if "c" in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE C: Large-Scale Training (backbone unfrozen, AMP)")
        logger.info("  The model learns to HEAR music.")
        logger.info("=" * 70)

        # Rebuild dataloaders with unfrozen batch size
        data_cfg = DatasetConfig(
            local_samples_dir=str(PRECOMPUTED_DIR),
            local_profiles_dir=str(PRECOMPUTED_DIR),
            batch_size=bs_unfrozen,
            num_workers=num_workers,
            pin_memory=True,
        )

        train_ds = RPMDataset(train_samples, data_cfg, feature_extractor, augment=True)
        val_ds = RPMDataset(val_samples, data_cfg, feature_extractor, augment=False)
        train_dl = build_dataloader(train_ds, data_cfg, shuffle=True)
        val_dl = build_dataloader(val_ds, data_cfg, shuffle=False)

        train_cfg = TrainingConfig(
            output_dir=str(OUTPUT_DIR),
            checkpoint_dir=str(CHECKPOINT_DIR),
            log_dir=str(LOG_DIR),
            device=device,
            phase_c_epochs=10,
            phase_c_batch_size=bs_unfrozen,
            phase_c_use_amp=True,
            phase_c_gradient_accumulation=grad_accum,
            phase_c_backbone_lr=1e-5,
            phase_c_heads_lr=5e-4,
        )
        trainer = RPMTrainer(train_cfg)

        t0 = time.time()
        trainer.train_phase_c(model, train_dl, val_dl)
        logger.info(f"Phase C done in {(time.time()-t0)/60:.1f} min")

        torch.save(model.state_dict(), CHECKPOINT_DIR / "rpm_phaseC_done.pt")
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    # ════════════════════════════════════════════════════════════════
    # Phase D: Chart Intelligence Fine-Tune
    # ════════════════════════════════════════════════════════════════

    if "d" in phases:
        if not chart_samples:
            logger.warning("No chart samples with previews — skipping Phase D")
            logger.warning("  Run with --enrich-only first to get Spotify data")
        else:
            logger.info("\n" + "=" * 70)
            logger.info("PHASE D: Chart Intelligence Fine-Tune")
            logger.info(f"  {len(chart_samples):,} chart songs with era + popularity labels")
            logger.info("=" * 70)

            # Re-freeze backbone for Phase D
            model.freeze_backbone()

            random.shuffle(chart_samples)
            split = int(len(chart_samples) * 0.9)
            chart_train = chart_samples[:split]
            chart_val = chart_samples[split:]

            data_cfg = DatasetConfig(
                local_samples_dir=str(PRECOMPUTED_DIR),
                local_profiles_dir=str(PRECOMPUTED_DIR),
                batch_size=bs_frozen,
                num_workers=num_workers,
                pin_memory=True,
            )
            train_ds = RPMDataset(chart_train, data_cfg, feature_extractor, augment=False)
            val_ds = RPMDataset(chart_val, data_cfg, feature_extractor, augment=False)
            train_dl = build_dataloader(train_ds, data_cfg, shuffle=True)
            val_dl = build_dataloader(val_ds, data_cfg, shuffle=False)

            train_cfg = TrainingConfig(
                output_dir=str(OUTPUT_DIR),
                checkpoint_dir=str(CHECKPOINT_DIR),
                log_dir=str(LOG_DIR),
                device=device,
                phase_d_epochs=5,
                phase_d_batch_size=bs_frozen,
            )
            trainer = RPMTrainer(train_cfg)

            t0 = time.time()
            trainer.train_phase_d(model, train_dl, val_dl)
            logger.info(f"Phase D done in {(time.time()-t0)/60:.1f} min")

            torch.save(model.state_dict(), CHECKPOINT_DIR / "rpm_phaseD_done.pt")

    # ════════════════════════════════════════════════════════════════
    # Save Final Model
    # ════════════════════════════════════════════════════════════════

    final_path = OUTPUT_DIR / "rpm_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"\nFinal model saved: {final_path}")

    # Export ONNX for production inference
    try:
        logger.info("Exporting ONNX model...")
        from ml.training.rpm_export import export_to_onnx
        export_to_onnx(str(final_path), str(OUTPUT_DIR), export_mode="embedding")
        logger.info(f"ONNX model saved: {OUTPUT_DIR}/rpm_embedding.onnx")
    except Exception as e:
        logger.warning(f"ONNX export failed (non-fatal): {e}")

    logger.info("\n" + "=" * 70)
    logger.info("ALL TRAINING COMPLETE!")
    logger.info(f"Final model:  {final_path}")
    logger.info(f"Checkpoints:  {CHECKPOINT_DIR}")
    logger.info(f"")
    logger.info(f"Download your model:")
    logger.info(f"  scp -P <SSH_PORT> root@<GPU_IP>:{final_path} ~/Desktop/")
    logger.info(f"  scp -P <SSH_PORT> root@<GPU_IP>:{OUTPUT_DIR}/rpm_embedding.onnx ~/Desktop/")
    logger.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RESONATE RPM Cloud Training")
    parser.add_argument(
        "--phases", default="all",
        help=(
            "Which phases to run: 'all' (A-D), 'ab', 'cd', 'c', 'a', 'b', 'd', "
            "'advanced' (E-I), 'e', 'f', 'g', 'h', 'i', 'ef', 'efghi', "
            "'full' (A-I), or any combo like 'cde' (default: all = A-D)"
        )
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint to resume from (Phase A-D or advanced)"
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download datasets, don't train"
    )
    parser.add_argument(
        "--enrich-only", action="store_true",
        help="Only run Spotify enrichment, don't train"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip dataset download, use what's already there"
    )
    parser.add_argument(
        "--skip-enrich", action="store_true",
        help="Skip Spotify enrichment"
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Override work directory (default: /workspace/resonate)"
    )
    parser.add_argument(
        "--download-advanced", action="store_true",
        help="Download Phase E-I datasets"
    )
    parser.add_argument(
        "--skip-advanced-download", action="store_true",
        help="Skip Phase E-I dataset download"
    )

    args = parser.parse_args()

    if args.work_dir:
        global WORK_DIR, DATA_DIR, PRECOMPUTED_DIR, COMPACT_DIR, DATASETS_DIR
        global CHARTS_DIR, CONFIG_DIR, CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR
        global ADVANCED_DATA_DIR, PHASE_E_DIR, PHASE_F_DIR, PHASE_G_DIR
        global PHASE_H_DIR, PHASE_I_DIR
        WORK_DIR = Path(args.work_dir)
        DATA_DIR = WORK_DIR / "data"
        PRECOMPUTED_DIR = DATA_DIR / "precomputed"
        COMPACT_DIR = DATA_DIR / "precomputed_compact"
        DATASETS_DIR = DATA_DIR / "datasets"
        CHARTS_DIR = DATA_DIR / "charts"
        CONFIG_DIR = DATA_DIR / "config"
        CHECKPOINT_DIR = WORK_DIR / "checkpoints"
        OUTPUT_DIR = WORK_DIR / "output"
        LOG_DIR = WORK_DIR / "logs"
        ADVANCED_DATA_DIR = DATA_DIR / "advanced"
        PHASE_E_DIR = ADVANCED_DATA_DIR / "phase_e"
        PHASE_F_DIR = ADVANCED_DATA_DIR / "phase_f"
        PHASE_G_DIR = ADVANCED_DATA_DIR / "phase_g"
        PHASE_H_DIR = ADVANCED_DATA_DIR / "phase_h"
        PHASE_I_DIR = ADVANCED_DATA_DIR / "phase_i"

    ensure_dirs()

    logger.info("=" * 70)
    logger.info("RESONATE Production Model — Cloud GPU Training")
    logger.info("  The God of Music Production AI")
    logger.info("=" * 70)
    logger.info(f"Work dir: {WORK_DIR}")

    # Download datasets
    if not args.skip_download:
        logger.info("\nDownloading datasets (cloud network = fast)...")
        try:
            download_datasets()
        except Exception as e:
            logger.warning(f"Dataset download issue: {e}")
            logger.warning("Continuing with available data...")

    if args.download_only:
        logger.info("Download complete. Exiting.")
        return

    # Spotify enrichment
    if not args.skip_enrich:
        run_chart_enrichment()

    if args.enrich_only:
        logger.info("Enrichment complete. Exiting.")
        return

    # Parse phases
    phases = args.phases.lower()
    if phases == "all":
        phases = "abcd"
    elif phases == "advanced":
        phases = "efghi"
    elif phases == "full":
        phases = "abcdefghi"
    elif phases == "ab":
        phases = "ab"
    elif phases == "cd":
        phases = "cd"

    # Determine which phase groups are requested
    has_basic = any(p in phases for p in "abcd")
    has_advanced = any(p in phases for p in "efghi")

    logger.info(f"Running phases: {phases.upper()}")

    # Download advanced datasets if needed
    if has_advanced and not args.skip_advanced_download:
        if args.download_advanced or not args.skip_download:
            try:
                download_advanced_datasets()
            except Exception as e:
                logger.warning(f"Advanced dataset download issue: {e}")
                logger.warning("Continuing with available data...")

    # Auto-detect checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Look for best available checkpoint
        candidates = [
            "rpm_phaseC_done.pt",  # best for Phase D only
            "rpm_phaseB_done.pt",  # best for Phase C+D
            "rpm_phaseA_done.pt",  # best for Phase B+C+D
            "rpm_best.pt",         # fallback
        ]

        # Pick the best checkpoint for the requested phases
        if "c" in phases and "a" not in phases and "b" not in phases:
            # Running Phase C — want Phase B checkpoint
            candidates = ["rpm_phaseB_done.pt", "rpm_phaseA_done.pt", "rpm_best.pt"]
        elif "d" in phases and "c" not in phases:
            # Running Phase D only — want Phase C checkpoint
            candidates = ["rpm_phaseC_done.pt", "rpm_best.pt"]
        elif has_advanced and not has_basic:
            # Running advanced only — want Phase D or final checkpoint
            candidates = [
                "rpm_advanced_final.pt",  # previous advanced run
                "rpm_phaseI_done.pt",
                "rpm_phaseH_done.pt",
                "rpm_phaseG_done.pt",
                "rpm_phaseF_done.pt",
                "rpm_phaseE_done.pt",
                "rpm_phaseD_done.pt",
                "rpm_final.pt",
                "rpm_phaseC_done.pt",
                "rpm_best.pt",
            ]

        for name in candidates:
            candidate = CHECKPOINT_DIR / name
            if candidate.exists():
                checkpoint = str(candidate)
                logger.info(f"Auto-detected checkpoint: {checkpoint}")
                break

        # Also check output dir for rpm_final.pt
        if checkpoint is None:
            final = OUTPUT_DIR / "rpm_final.pt"
            if final.exists():
                checkpoint = str(final)
                logger.info(f"Auto-detected checkpoint: {checkpoint}")

    # Run Phase A-D training if requested
    if has_basic:
        basic_phases = "".join(p for p in phases if p in "abcd")
        run_training(basic_phases, checkpoint)

    # Run Phase E-I training if requested
    if has_advanced:
        advanced_phases = "".join(p for p in phases if p in "efghi")
        run_advanced_training(advanced_phases, checkpoint)


if __name__ == "__main__":
    main()
