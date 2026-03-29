#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# RESONATE — Autonomous Full Training Orchestrator
# ══════════════════════════════════════════════════════════════════════
# This script runs FULLY AUTONOMOUSLY on RunPod:
#   1. Waits for all downloads and Deezer enrichment to complete
#   2. Retrains Phase C on the FULL dataset (NSynth + FMA + Jamendo + compact)
#   3. Runs Phase D on chart preview audio
#   4. Exports the final model
#
# No shortcuts. No half-measures. Train on everything.
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail

WORK_DIR="/workspace/resonate"
LOG_DIR="$WORK_DIR/logs"
DATASETS_DIR="$WORK_DIR/data/datasets"
CHARTS_DIR="$WORK_DIR/data/charts"
CHECKPOINT_DIR="$WORK_DIR/checkpoints"

LOGFILE="$LOG_DIR/orchestrator_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

# ──────────────────────────────────────────────────────────────────
# Phase 1: Wait for all downloads + enrichment to finish
# ──────────────────────────────────────────────────────────────────

wait_for_downloads() {
    log "═══════════════════════════════════════════════════════════"
    log "PHASE 1: Waiting for all downloads and Deezer enrichment"
    log "═══════════════════════════════════════════════════════════"

    local max_wait=14400  # 4 hours max
    local elapsed=0
    local check_interval=30

    while [ $elapsed -lt $max_wait ]; do
        local all_done=true

        # Check wget processes (dataset downloads)
        local wget_count
        wget_count=$(pgrep -c wget 2>/dev/null || echo 0)
        if [ "$wget_count" -gt 0 ]; then
            all_done=false
            # Show download progress
            local nsynth_size fma_size jamendo_size
            nsynth_size=$(du -sh "$DATASETS_DIR/nsynth/" 2>/dev/null | cut -f1 || echo "0")
            fma_size=$(du -sh "$DATASETS_DIR/fma/" 2>/dev/null | cut -f1 || echo "0")
            jamendo_size=$(du -sh "$DATASETS_DIR/jamendo/" 2>/dev/null | cut -f1 || echo "0")
            log "  Downloads in progress ($wget_count active) — NSynth: $nsynth_size, FMA: $fma_size, Jamendo: $jamendo_size"
        fi

        # Check Deezer enrichment process
        if pgrep -f "deezer_enrichment" > /dev/null 2>&1; then
            all_done=false
            local enriched
            enriched=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$CHARTS_DIR/chart_features.db')
total = conn.execute('SELECT COUNT(*) FROM chart_entries').fetchone()[0]
done = conn.execute('SELECT COUNT(*) FROM chart_entries WHERE deezer_id IS NOT NULL').fetchone()[0]
previews = conn.execute('SELECT COUNT(*) FROM chart_entries WHERE length(deezer_preview_path) > 0').fetchone()[0]
print(f'{done}/{total} enriched, {previews} previews')
conn.close()
" 2>/dev/null || echo "unknown")
            log "  Deezer enrichment running — $enriched"
        fi

        if $all_done; then
            log "  All downloads and enrichment COMPLETE!"
            break
        fi

        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    if [ $elapsed -ge $max_wait ]; then
        log "WARNING: Max wait time reached. Proceeding with whatever data is available."
    fi

    # Log final data inventory
    log ""
    log "═══ DATA INVENTORY ═══"

    # NSynth
    for split in train valid test; do
        local count
        count=$(ls "$DATASETS_DIR/nsynth/nsynth-$split/audio/" 2>/dev/null | wc -l || echo 0)
        log "  NSynth-$split: $count audio files"
    done

    # FMA
    local fma_count
    fma_count=$(find "$DATASETS_DIR/fma/fma_small" -name "*.mp3" 2>/dev/null | wc -l || echo 0)
    log "  FMA-small: $fma_count MP3 files"

    # Jamendo
    local jamendo_count
    jamendo_count=$(find "$DATASETS_DIR/jamendo" -name "*.mp3" -o -name "*.wav" -o -name "*.ogg" 2>/dev/null | wc -l || echo 0)
    log "  Jamendo: $jamendo_count audio files"

    # Chart previews
    local preview_count
    preview_count=$(find "$CHARTS_DIR/previews" -name "*.mp3" 2>/dev/null | wc -l || echo 0)
    log "  Chart previews: $preview_count MP3 files"

    # Deezer enrichment stats
    python3 -c "
import sqlite3
conn = sqlite3.connect('$CHARTS_DIR/chart_features.db')
total = conn.execute('SELECT COUNT(*) FROM chart_entries').fetchone()[0]
deezer = conn.execute('SELECT COUNT(*) FROM chart_entries WHERE deezer_id IS NOT NULL AND deezer_id > 0').fetchone()[0]
previews = conn.execute('SELECT COUNT(*) FROM chart_entries WHERE length(deezer_preview_path) > 0').fetchone()[0]
print(f'  Chart DB: {total} entries, {deezer} Deezer-matched, {previews} with preview audio')
conn.close()
" 2>/dev/null | tee -a "$LOGFILE"

    local disk_used
    disk_used=$(df -h /workspace | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')
    log "  Disk: $disk_used"
    log ""
}

# ──────────────────────────────────────────────────────────────────
# Phase 2: Full retraining — Phase C on ALL data
# ──────────────────────────────────────────────────────────────────

run_full_training() {
    log "═══════════════════════════════════════════════════════════"
    log "PHASE 2: Full Retraining — Phase C + D on ALL data"
    log "  Loading from Phase B checkpoint (pre-backbone-unfreezing)"
    log "  This is the GOD-TIER training run."
    log "═══════════════════════════════════════════════════════════"

    cd "$WORK_DIR"

    # Use Phase B checkpoint as the base — we want to retrain Phase C
    # from scratch on the full dataset, not continue from the compact-only run
    local checkpoint="$CHECKPOINT_DIR/rpm_phaseB_done.pt"
    if [ ! -f "$checkpoint" ]; then
        log "ERROR: Phase B checkpoint not found at $checkpoint"
        log "Cannot retrain without it. Using best available checkpoint instead."
        checkpoint="$CHECKPOINT_DIR/rpm_best.pt"
    fi

    log "Starting from checkpoint: $checkpoint"
    log "Training phases: C + D"
    log ""

    # Increase Phase C epochs for full dataset — more data = more to learn
    # 15 epochs on ~276k samples vs the initial 10 epochs on ~26k samples
    python3 cloud/train_cloud.py \
        --phases cd \
        --checkpoint "$checkpoint" \
        --skip-download \
        --skip-enrich \
        --work-dir "$WORK_DIR" \
        2>&1 | tee -a "$LOGFILE"

    log ""
    log "Full training pipeline complete!"
}

# ──────────────────────────────────────────────────────────────────
# Phase 3: Post-training — save results
# ──────────────────────────────────────────────────────────────────

post_training() {
    log "═══════════════════════════════════════════════════════════"
    log "PHASE 3: Post-Training"
    log "═══════════════════════════════════════════════════════════"

    # List all checkpoints and output files
    log "Checkpoints:"
    ls -lh "$CHECKPOINT_DIR/" 2>/dev/null | tee -a "$LOGFILE"
    log ""
    log "Output:"
    ls -lh "$WORK_DIR/output/" 2>/dev/null | tee -a "$LOGFILE"
    log ""

    # GPU stats
    nvidia-smi 2>/dev/null | tee -a "$LOGFILE" || true

    log ""
    log "═══════════════════════════════════════════════════════════"
    log "ALL DONE — RESONATE Production Model fully trained!"
    log ""
    log "To download:"
    log "  scp -P 12560 root@40.142.99.119:/workspace/resonate/output/rpm_final.pt ~/Desktop/RESONATE/backend/ml/models/"
    log "  scp -P 12560 root@40.142.99.119:/workspace/resonate/output/rpm_embedding.onnx ~/Desktop/RESONATE/backend/ml/models/"
    log "  scp -P 12560 root@40.142.99.119:/workspace/resonate/checkpoints/rpm_phaseC_done.pt ~/Desktop/RESONATE/backend/ml/models/"
    log "  scp -P 12560 root@40.142.99.119:/workspace/resonate/checkpoints/rpm_phaseD_done.pt ~/Desktop/RESONATE/backend/ml/models/"
    log ""
    log "═══════════════════════════════════════════════════════════"
}

# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

main() {
    log "═══════════════════════════════════════════════════════════"
    log "RESONATE — Autonomous Full Training Orchestrator"
    log "  SoniqLabs — No shortcuts, highest quality, all out."
    log "═══════════════════════════════════════════════════════════"
    log ""

    wait_for_downloads
    run_full_training
    post_training
}

main "$@"
