#!/bin/bash
# ============================================================================
# RESONATE — One-Shot RunPod Deployment
# ============================================================================
# Run this ON the RunPod instance after uploading the cloud pack.
# Does everything: setup → unpack → download datasets → train all phases.
#
# Usage:
#   # 1. Upload pack to RunPod (from your Mac):
#   #    scp -P <SSH_PORT> ~/Desktop/resonate_cloud_pack.tar.gz root@<POD_IP>:/workspace/
#   #
#   # 2. SSH into RunPod:
#   #    ssh -p <SSH_PORT> root@<POD_IP>
#   #
#   # 3. Run this script:
#   #    cd /workspace && bash resonate/backend/cloud/deploy_runpod.sh
#   #
#   # Or one-liner after SSH:
#   #    cd /workspace && tar xzf resonate_cloud_pack.tar.gz -C resonate/ && bash resonate/backend/cloud/deploy_runpod.sh
#
# Estimated time on RTX 5090:
#   Setup:     ~3 min
#   Download:  ~15 min (NSynth 25GB + FMA 7GB + Jamendo 30GB)
#   Phase C:   ~1.5-2 hours (229k+ samples, 10 epochs, backbone unfrozen)
#   Phase D:   ~10 min (chart fine-tune)
#   Total:     ~2-3 hours
# ============================================================================
set -euo pipefail

WORK_DIR="${RESONATE_WORK_DIR:-/workspace/resonate}"
export RESONATE_WORK_DIR="$WORK_DIR"

echo "════════════════════════════════════════════════════════════════════════"
echo "  RESONATE Production Model — RunPod Deployment"
echo "  The God of Music Production AI"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Work dir: $WORK_DIR"
echo "  Time:     $(date)"
echo ""

# ── Step 0: Unpack tarball OR pull from Cloudflare R2 ──
if [ -f "/workspace/resonate_cloud_pack.tar.gz" ] && [ ! -d "$WORK_DIR/backend/cloud" ]; then
    echo "[0/5] Unpacking cloud pack..."
    mkdir -p "$WORK_DIR"
    tar xzf /workspace/resonate_cloud_pack.tar.gz -C "$WORK_DIR"
    echo "  ✓ Unpacked from tarball"
elif command -v rclone &>/dev/null; then
    echo "[0/5] Pulling data from Cloudflare R2..."
    mkdir -p "$WORK_DIR"/{data,checkpoints,backend}
    # Pull charts (DB + preview MP3s)
    rclone sync r2:resonate-data/charts "$WORK_DIR/data/charts/" --progress --transfers 8 2>/dev/null || true
    # Pull checkpoints
    rclone sync r2:resonate-data/checkpoints "$WORK_DIR/checkpoints/" --progress --transfers 4 2>/dev/null || true
    # Pull compact embeddings
    rclone sync r2:resonate-data/embeddings/compact "$WORK_DIR/data/precomputed_compact/" --progress 2>/dev/null || true
    echo "  ✓ Pulled from R2"
fi

# Move checkpoints from data/ to checkpoints/
if [ -d "$WORK_DIR/data/checkpoints" ]; then
    mkdir -p "$WORK_DIR/checkpoints"
    cp -n "$WORK_DIR/data/checkpoints"/*.pt "$WORK_DIR/checkpoints/" 2>/dev/null || true
    echo "  ✓ Checkpoints staged"
fi

# ── Step 1: Environment setup ──
echo ""
echo "[1/5] Setting up environment..."
bash "$WORK_DIR/backend/cloud/setup.sh"

# ── Step 2: Load Spotify credentials ──
echo ""
echo "[2/5] Loading credentials..."
if [ -f "$WORK_DIR/data/config/spotify.env" ]; then
    set -a
    source "$WORK_DIR/data/config/spotify.env"
    set +a
    echo "  ✓ Spotify credentials loaded"
fi

# ── Step 3: Download datasets ──
echo ""
echo "[3/5] Downloading datasets (cloud network = fast)..."
cd "$WORK_DIR/backend"
python3 cloud/train_cloud.py --download-only --work-dir "$WORK_DIR"

# ── Step 4: Run Spotify enrichment ──
echo ""
echo "[4/5] Running Spotify enrichment..."
python3 cloud/train_cloud.py --enrich-only --skip-download --work-dir "$WORK_DIR" || \
    echo "  ⚠ Spotify enrichment had issues (non-fatal, continuing)"

# ── Step 5: TRAIN! ──
echo ""
echo "[5/5] Starting training..."
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Phase C: Large-Scale Training (backbone unfrozen)"
echo "  Phase D: Chart Intelligence Fine-Tune"
echo "  Starting from Phase B checkpoint (skip A+B — already done locally)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Use nohup so training survives SSH disconnects
# Also tee to log file for monitoring
nohup python3 cloud/train_cloud.py \
    --phases cd \
    --skip-download \
    --skip-enrich \
    --work-dir "$WORK_DIR" \
    2>&1 | tee "$WORK_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log" &

TRAIN_PID=$!
echo "Training started! PID: $TRAIN_PID"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Monitor training:"
echo "    tail -f $WORK_DIR/logs/train_*.log"
echo "    # or"
echo "    tail -f rpm_cloud_training.log"
echo ""
echo "  When done, download results to your Mac:"
echo "    bash cloud/fetch_results.sh <YOUR_MAC_IP_OR_LOCALHOST>"
echo "    # or manually:"
echo "    scp -P <SSH_PORT> root@<POD_IP>:$WORK_DIR/output/rpm_final.pt ~/Desktop/"
echo "════════════════════════════════════════════════════════════════════════"

# Wait for training to complete
wait $TRAIN_PID
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  ✅ TRAINING COMPLETE!"
    echo ""
    echo "  Final model: $WORK_DIR/output/rpm_final.pt"
    ls -lh "$WORK_DIR/output/rpm_final.pt" 2>/dev/null || true
    ls -lh "$WORK_DIR/output/rpm_embedding.onnx" 2>/dev/null || true
    echo ""
    echo "  Download to your Mac:"
    echo "    scp -P <SSH_PORT> root@<POD_IP>:$WORK_DIR/output/rpm_final.pt ~/Desktop/"
    echo "    scp -P <SSH_PORT> root@<POD_IP>:$WORK_DIR/output/rpm_embedding.onnx ~/Desktop/"
    echo "════════════════════════════════════════════════════════════════════════"
else
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  ❌ Training failed with exit code $EXIT_CODE"
    echo "  Check logs: tail -100 rpm_cloud_training.log"
    echo "════════════════════════════════════════════════════════════════════════"
    exit $EXIT_CODE
fi
