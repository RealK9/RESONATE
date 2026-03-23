#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# RESONATE — Setup Script
# Sets up the Python backend, installs dependencies, and validates
# the environment so you can run the app with `npm run dev:full`.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
VENV="$BACKEND/venv"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║          RESONATE — Environment Setup        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Check Python ──────────────────────────────────────────────────
PYTHON=""
for cmd in python3.11 python3.10 python3.9 python3; do
  if command -v "$cmd" &>/dev/null; then
    PYTHON="$cmd"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "✗ Python 3.9+ not found. Install Python and try again."
  exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Found $PYTHON ($PY_VERSION)"

# ── 2. Check Node.js ────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
  echo "✗ Node.js not found. Install Node.js 18+ and try again."
  exit 1
fi
NODE_VERSION=$(node -v)
echo "✓ Found Node.js $NODE_VERSION"

# ── 3. Create / activate Python venv ────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo ""
  echo "Creating Python virtual environment..."
  $PYTHON -m venv "$VENV"
  echo "✓ Virtual environment created at backend/venv/"
else
  echo "✓ Virtual environment already exists"
fi

# Activate
source "$VENV/bin/activate"
echo "✓ Activated venv ($(python --version))"

# ── 4. Install base backend dependencies ─────────────────────────────
echo ""
echo "Installing backend dependencies..."
pip install --upgrade pip -q
pip install -r "$BACKEND/requirements.txt" -q
echo "✓ Base backend dependencies installed"

# ── 5. Install ML pipeline dependencies ──────────────────────────────
echo ""
echo "Installing ML pipeline dependencies..."
echo "  (This includes PyTorch, transformers, FAISS — may take a few minutes)"
pip install -r "$ROOT/requirements-ml.txt" -q
echo "✓ ML pipeline dependencies installed"

# ── 6. Install Node.js dependencies ──────────────────────────────────
echo ""
echo "Installing frontend dependencies..."
cd "$ROOT"
npm install --silent 2>/dev/null || npm install
echo "✓ Frontend dependencies installed"

# ── 7. Create required directories ───────────────────────────────────
mkdir -p "$BACKEND/samples" "$BACKEND/uploads" "$BACKEND/transposed_cache"
echo "✓ Backend directories ready"

# ── 8. Environment file ──────────────────────────────────────────────
if [ ! -f "$BACKEND/.env" ]; then
  cp "$BACKEND/.env.example" "$BACKEND/.env"
  echo "✓ Created backend/.env from template (add API keys if you have them)"
else
  echo "✓ backend/.env already exists"
fi

# ── 9. Validate imports ──────────────────────────────────────────────
echo ""
echo "Validating Python imports..."
IMPORT_OK=true

python -c "import fastapi" 2>/dev/null || { echo "  ✗ fastapi"; IMPORT_OK=false; }
python -c "import uvicorn" 2>/dev/null || { echo "  ✗ uvicorn"; IMPORT_OK=false; }
python -c "import librosa" 2>/dev/null || { echo "  ✗ librosa"; IMPORT_OK=false; }
python -c "import numpy" 2>/dev/null || { echo "  ✗ numpy"; IMPORT_OK=false; }
python -c "import soundfile" 2>/dev/null || { echo "  ✗ soundfile"; IMPORT_OK=false; }
python -c "import sklearn" 2>/dev/null || { echo "  ✗ scikit-learn"; IMPORT_OK=false; }
python -c "import torch" 2>/dev/null || { echo "  ⚠ torch (ML models will be limited)"; }
python -c "import faiss" 2>/dev/null || { echo "  ⚠ faiss-cpu (vector search disabled)"; }

if $IMPORT_OK; then
  echo "✓ All core imports validated"
fi

# ── 10. Run quick test ───────────────────────────────────────────────
echo ""
echo "Running quick validation..."
cd "$ROOT"
python -m pytest tests/test_models.py tests/test_sample_store.py -q --tb=line 2>/dev/null && echo "✓ Core tests pass" || echo "⚠ Some tests failed (non-critical for startup)"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo ""
echo "  Setup complete! To run RESONATE:"
echo ""
echo "    npm run dev:full     Full app (backend + frontend + Electron)"
echo "    npm run dev          Frontend only (start backend separately)"
echo ""
echo "  To add samples, either:"
echo "    1. Drop audio files into backend/samples/"
echo "    2. Configure sample directory in the app settings"
echo "    3. Install Splice or Loopcloud (auto-detected)"
echo ""
echo "  ML models (CLAP, PANNs, AST) download automatically on first use."
echo "  First analysis may take longer while models are cached."
echo ""
echo "══════════════════════════════════════════════════"
