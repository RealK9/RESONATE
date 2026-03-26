#!/bin/bash
# ============================================================================
# RESONATE Cloud GPU Setup — Run this FIRST on your rented GPU instance
# ============================================================================
# Tested on: RunPod, Vast.ai, Lambda (Ubuntu 22.04 + CUDA 12.x)
#
# Usage:
#   bash setup.sh
# ============================================================================
set -euo pipefail

echo "========================================================================"
echo "  RESONATE Production Model — Cloud GPU Setup"
echo "========================================================================"

# ── System packages ──
echo "[1/5] Installing system packages..."
apt-get update -qq && apt-get install -y -qq \
    ffmpeg libsndfile1 sox git wget unzip curl pigz pv \
    2>/dev/null || true

# ── Python environment ──
echo "[2/5] Setting up Python environment..."
pip install --quiet --upgrade pip

pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    2>/dev/null || pip install --quiet torch torchvision torchaudio

pip install --quiet \
    transformers \
    librosa \
    soundfile \
    numpy \
    scipy \
    laion-clap \
    onnx \
    onnxruntime-gpu \
    tqdm \
    tensorboard

# ── Verify CUDA ──
echo "[3/5] Verifying GPU..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: No CUDA GPU detected!')
"

# ── Directory structure ──
echo "[4/5] Setting up directory structure..."
WORK_DIR="${RESONATE_WORK_DIR:-/workspace/resonate}"
mkdir -p "$WORK_DIR"/{data,checkpoints,logs,output}
mkdir -p "$WORK_DIR"/data/{precomputed,datasets,charts}

echo "[5/5] Setup complete!"
echo ""
echo "  Work directory: $WORK_DIR"
echo ""
echo "  Next steps:"
echo "    1. Upload your data pack:  rsync -avP resonate_cloud_pack.tar.gz <gpu-ip>:$WORK_DIR/"
echo "    2. Extract:                cd $WORK_DIR && tar xzf resonate_cloud_pack.tar.gz"
echo "    3. Train:                  python3 train_cloud.py --phases all"
echo ""
echo "========================================================================"
