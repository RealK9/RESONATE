#!/bin/bash
# ============================================================================
# RESONATE Cloud Pack — Package everything needed for cloud training
# ============================================================================
# Creates a single tarball with all code + data needed on the GPU instance.
# Run this locally on your Mac.
#
# Usage:
#   cd /path/to/RESONATE/backend
#   bash cloud/pack.sh
#
# Output:
#   ~/Desktop/resonate_cloud_pack.tar.gz
# ============================================================================
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACK_DIR="/tmp/resonate_cloud_pack"
OUTPUT="$HOME/Desktop/resonate_cloud_pack.tar.gz"

echo "========================================================================"
echo "  RESONATE Cloud Pack Builder"
echo "========================================================================"
echo "  Backend: $BACKEND_DIR"

# Clean previous
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR"

# ── 1. Code ──
echo "[1/6] Packing code..."
mkdir -p "$PACK_DIR/backend"

# Copy ML code (the important stuff)
cp -r "$BACKEND_DIR/ml" "$PACK_DIR/backend/"

# Copy cloud scripts
cp -r "$BACKEND_DIR/cloud" "$PACK_DIR/backend/"

# Copy training entry points
cp "$BACKEND_DIR/train_rpm.py" "$PACK_DIR/backend/" 2>/dev/null || true

echo "  Code: $(du -sh "$PACK_DIR/backend/ml" | cut -f1)"

# ── 2. Pre-computed teacher embeddings ──
echo "[2/6] Packing pre-computed embeddings..."
PRECOMPUTED="$HOME/.resonate/precomputed"
if [ -d "$PRECOMPUTED" ]; then
    mkdir -p "$PACK_DIR/data/precomputed"
    # Only copy real JSON files, skip ExFAT ._ files
    find "$PRECOMPUTED" -name "*.json" -not -name "._*" -exec cp {} "$PACK_DIR/data/precomputed/" \;
    COUNT=$(find "$PACK_DIR/data/precomputed" -name "*.json" | wc -l | tr -d ' ')
    echo "  Embeddings: $COUNT files ($(du -sh "$PACK_DIR/data/precomputed" | cut -f1))"
else
    echo "  WARNING: No pre-computed embeddings found at $PRECOMPUTED"
fi

# ── 3. Chart database ──
echo "[3/6] Packing chart data..."
CHARTS="$HOME/.resonate/charts"
if [ -d "$CHARTS" ]; then
    mkdir -p "$PACK_DIR/data/charts"
    cp "$CHARTS"/*.db "$PACK_DIR/data/charts/" 2>/dev/null || true
    cp "$CHARTS"/*.json "$PACK_DIR/data/charts/" 2>/dev/null || true
    echo "  Charts: $(du -sh "$PACK_DIR/data/charts" | cut -f1)"
else
    echo "  No chart data found (optional — needed for Phase D)"
fi

# ── 4. Phase A+B checkpoint (if exists) ──
echo "[4/6] Packing checkpoints..."
TRAINING="$HOME/.resonate/rpm_training"
CKPT="$HOME/.resonate/rpm_checkpoints"
mkdir -p "$PACK_DIR/data/checkpoints"

if [ -f "$TRAINING/rpm_phase_ab.pt" ]; then
    cp "$TRAINING/rpm_phase_ab.pt" "$PACK_DIR/data/checkpoints/"
    echo "  Phase A+B checkpoint: $(du -sh "$TRAINING/rpm_phase_ab.pt" | cut -f1)"
else
    # Copy any available checkpoints
    if [ -d "$CKPT" ]; then
        LATEST=$(ls -t "$CKPT"/*.pt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            cp "$LATEST" "$PACK_DIR/data/checkpoints/"
            echo "  Latest checkpoint: $(basename "$LATEST") ($(du -sh "$LATEST" | cut -f1))"
        else
            echo "  No checkpoints found (training will start from scratch)"
        fi
    fi
fi

# ── 5. Local samples list (for reference, not the audio files) ──
echo "[5/6] Packing sample manifest..."
SAMPLES_DIR="$BACKEND_DIR/samples"
if [ -d "$SAMPLES_DIR" ]; then
    find "$SAMPLES_DIR" -name "*.wav" -type f > "$PACK_DIR/data/sample_manifest.txt"
    echo "  Sample manifest: $(wc -l < "$PACK_DIR/data/sample_manifest.txt" | tr -d ' ') files"
fi

# ── 6. Create tarball ──
echo "[6/6] Creating tarball..."
cd /tmp
tar czf "$OUTPUT" -C "$PACK_DIR" .
SIZE=$(du -sh "$OUTPUT" | cut -f1)

echo ""
echo "========================================================================"
echo "  Pack complete!"
echo "  Output: $OUTPUT"
echo "  Size:   $SIZE"
echo ""
echo "  Contents:"
du -sh "$PACK_DIR"/* 2>/dev/null | sed 's|/tmp/resonate_cloud_pack/||'
echo ""
echo "  Upload to your GPU instance:"
echo "    scp $OUTPUT root@<GPU_IP>:/workspace/resonate/"
echo "    # or for RunPod:"
echo "    rsync -avP $OUTPUT root@<POD_IP>:/workspace/resonate/"
echo "========================================================================"

# Cleanup
rm -rf "$PACK_DIR"
