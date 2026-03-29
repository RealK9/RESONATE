#!/bin/bash
# ============================================================================
# RESONATE Cloud GPU Setup — Run this FIRST on your rented GPU instance
# ============================================================================
# Tested on: RunPod, Vast.ai, Lambda (Ubuntu 22.04 + CUDA 12.x)
# Optimized for: RTX 5090 (32GB), also works on A100/H100/4090
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

# PyTorch with CUDA 12.8 (RTX 5090 / Blackwell support)
pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    2>/dev/null || pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
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
    tensorboard \
    spotipy \
    billboard.py \
    requests

# ── Verify CUDA ──
echo "[3/5] Verifying GPU..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  VRAM: {vram:.1f} GB')
    if vram >= 28:
        print(f'  Tier: RTX 5090 / A100 class — full speed training')
    elif vram >= 20:
        print(f'  Tier: RTX 4090 class — fast training')
    else:
        print(f'  Tier: Standard — moderate speed')
else:
    print('  WARNING: No CUDA GPU detected!')
"

# ── Directory structure ──
echo "[4/5] Setting up directory structure..."
WORK_DIR="${RESONATE_WORK_DIR:-/workspace/resonate}"
mkdir -p "$WORK_DIR"/{data,checkpoints,logs,output}
mkdir -p "$WORK_DIR"/data/{precomputed,datasets,charts,samples}

echo "[5/5] Setup complete!"
echo ""
echo "  Work directory: $WORK_DIR"
echo ""
echo "  Next steps:"
echo "    1. Upload your data pack:  rsync -avP resonate_cloud_pack.tar.gz root@\$POD_IP:\$WORK_DIR/"
echo "    2. Extract:                cd $WORK_DIR && tar xzf resonate_cloud_pack.tar.gz"
echo "    3. Train:                  cd $WORK_DIR/backend && python3 cloud/train_cloud.py --phases all"
echo ""
echo "========================================================================"
