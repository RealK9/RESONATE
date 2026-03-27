#!/bin/bash
# ============================================================================
# RESONATE — Fetch trained model from cloud GPU back to local machine
# ============================================================================
# Usage:
#   bash fetch_results.sh <GPU_IP> [SSH_PORT]
#
# Example:
#   bash fetch_results.sh 123.456.789.0
#   bash fetch_results.sh 123.456.789.0 22222   # custom SSH port (RunPod)
# ============================================================================
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash fetch_results.sh <GPU_IP> [SSH_PORT]"
    echo "  GPU_IP:   IP address of your cloud GPU"
    echo "  SSH_PORT: SSH port (default: 22, RunPod often uses custom ports)"
    exit 1
fi

GPU_IP="$1"
SSH_PORT="${2:-22}"
LOCAL_DIR="$HOME/.resonate"
REMOTE_DIR="/workspace/resonate"

echo "========================================================================"
echo "  Fetching trained RPM model from $GPU_IP:$SSH_PORT"
echo "========================================================================"

# Final model
echo "[1/3] Downloading final model..."
mkdir -p "$LOCAL_DIR/rpm_training"
scp -P "$SSH_PORT" "root@$GPU_IP:$REMOTE_DIR/output/rpm_final.pt" \
    "$LOCAL_DIR/rpm_training/rpm_final.pt" 2>/dev/null && \
    echo "  Downloaded: rpm_final.pt" || \
    echo "  WARNING: rpm_final.pt not found"

# ONNX model
echo "[2/3] Downloading ONNX model..."
scp -P "$SSH_PORT" "root@$GPU_IP:$REMOTE_DIR/output/rpm_embedding.onnx" \
    "$LOCAL_DIR/rpm_training/rpm_embedding.onnx" 2>/dev/null && \
    echo "  Downloaded: rpm_embedding.onnx" || \
    echo "  WARNING: rpm_embedding.onnx not found (ONNX export may have failed)"

# All checkpoints
echo "[3/3] Downloading checkpoints..."
mkdir -p "$LOCAL_DIR/rpm_checkpoints"
rsync -avP --port="$SSH_PORT" \
    "root@$GPU_IP:$REMOTE_DIR/checkpoints/" \
    "$LOCAL_DIR/rpm_checkpoints/" 2>/dev/null || \
    echo "  WARNING: Could not sync checkpoints"

echo ""
echo "========================================================================"
echo "  Done! Model files:"
ls -lh "$LOCAL_DIR/rpm_training/rpm_final.pt" 2>/dev/null || true
ls -lh "$LOCAL_DIR/rpm_training/rpm_embedding.onnx" 2>/dev/null || true
echo ""
echo "  To use in RESONATE:"
echo "    export RPM_MODEL_PATH=$LOCAL_DIR/rpm_training/rpm_final.pt"
echo "    python3 -m backend.app"
echo "========================================================================"
