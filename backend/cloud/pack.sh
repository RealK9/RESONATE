#!/bin/bash
# ============================================================================
# RESONATE Cloud Pack — Package everything needed for cloud GPU training
# ============================================================================
# Creates a tarball with code + compact data for fast upload.
# Datasets (NSynth 25GB, FMA 7GB) are downloaded on the cloud instance.
#
# Usage:
#   cd /path/to/RESONATE/backend
#   bash cloud/pack.sh
#
# Output:
#   ~/Desktop/resonate_cloud_pack.tar.gz  (~800MB—1.2GB)
# ============================================================================
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACK_DIR="/tmp/resonate_cloud_pack"
OUTPUT="$HOME/Desktop/resonate_cloud_pack.tar.gz"

echo "========================================================================"
echo "  RESONATE Cloud Pack Builder"
echo "========================================================================"
echo "  Backend: $BACKEND_DIR"
echo ""

# Clean previous
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR"

# ── 1. Code ──
echo "[1/6] Packing code..."
mkdir -p "$PACK_DIR/backend"

# ML code
cp -r "$BACKEND_DIR/ml" "$PACK_DIR/backend/"
# Cloud scripts
cp -r "$BACKEND_DIR/cloud" "$PACK_DIR/backend/"
# Training entry points
cp "$BACKEND_DIR/train_rpm.py" "$PACK_DIR/backend/" 2>/dev/null || true
cp "$BACKEND_DIR/train_rpm_phase_c.py" "$PACK_DIR/backend/" 2>/dev/null || true

echo "  Code: $(du -sh "$PACK_DIR/backend" | cut -f1)"

# ── 2. Compact teacher embeddings (66GB JSON → ~500MB binary) ──
echo "[2/6] Compacting teacher embeddings (66GB → ~500MB)..."
cd "$BACKEND_DIR"
python3 -c "
import json, os, struct, sys
import numpy as np
from pathlib import Path

precomputed = Path(os.path.expanduser('~/.resonate/precomputed'))
out_dir = Path('$PACK_DIR/data/precomputed_compact')
out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(precomputed.glob('*.json'))
print(f'  Processing {len(files)} profiles...')

# Collect into arrays for numpy
filepaths = []
roles = []
role_ids = []
clap_embs = []
panns_embs = []
ast_embs = []
panns_tags = []

skipped = 0
for i, f in enumerate(files):
    if f.name.startswith('._'):
        skipped += 1
        continue
    try:
        raw = f.read_bytes()
        if raw[0:1] != b'{':
            skipped += 1
            continue
        p = json.loads(raw)

        emb = p.get('embeddings', {})
        clap = emb.get('clap_general')
        panns = emb.get('panns_music')
        ast = emb.get('ast_spectrogram')
        tags = emb.get('panns_tags')

        if not clap or not panns or not ast:
            skipped += 1
            continue

        filepaths.append(p.get('filepath', ''))
        roles.append(p.get('role', 'unknown'))
        role_ids.append(p.get('role_id', -1))
        clap_embs.append(clap)
        panns_embs.append(panns)
        ast_embs.append(ast)
        # panns_tags can be list of dicts or list of floats — normalize
        if tags and isinstance(tags, list) and len(tags) > 0:
            if isinstance(tags[0], dict):
                panns_tags.append([0.0]*20)
            else:
                panns_tags.append(tags)
        else:
            panns_tags.append([0.0]*20)

    except Exception as e:
        skipped += 1
        continue

    if (i + 1) % 5000 == 0:
        print(f'    {i+1}/{len(files)}...')

print(f'  Valid: {len(filepaths)}, Skipped: {skipped}')

# Save as numpy arrays
np.savez_compressed(
    str(out_dir / 'embeddings.npz'),
    clap=np.array(clap_embs, dtype=np.float32),
    panns=np.array(panns_embs, dtype=np.float32),
    ast=np.array(ast_embs, dtype=np.float32),
    panns_tags=np.array(panns_tags, dtype=np.float32),
    role_ids=np.array(role_ids, dtype=np.int32),
)

# Save metadata as JSON lines (compact)
with open(str(out_dir / 'metadata.jsonl'), 'w') as fout:
    for fp, role in zip(filepaths, roles):
        fout.write(json.dumps({'filepath': fp, 'role': role}) + '\n')

print(f'  Saved to {out_dir}')
" 2>&1
echo "  Compact embeddings: $(du -sh "$PACK_DIR/data/precomputed_compact" | cut -f1)"

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

# ── 4. Best checkpoint for Phase C resume ──
echo "[4/6] Packing checkpoint..."
CKPT_DIR="$HOME/.resonate/rpm_checkpoints"
mkdir -p "$PACK_DIR/data/checkpoints"

# Phase B done = best starting point for re-running Phase C
if [ -f "$CKPT_DIR/rpm_phaseB_done.pt" ]; then
    cp "$CKPT_DIR/rpm_phaseB_done.pt" "$PACK_DIR/data/checkpoints/"
    echo "  Phase B checkpoint: $(du -sh "$CKPT_DIR/rpm_phaseB_done.pt" | cut -f1)"
elif [ -f "$CKPT_DIR/rpm_phaseC_done.pt" ]; then
    cp "$CKPT_DIR/rpm_phaseC_done.pt" "$PACK_DIR/data/checkpoints/"
    echo "  Phase C checkpoint: $(du -sh "$CKPT_DIR/rpm_phaseC_done.pt" | cut -f1)"
fi

# Also pack the best model as fallback
if [ -f "$CKPT_DIR/rpm_best.pt" ]; then
    cp "$CKPT_DIR/rpm_best.pt" "$PACK_DIR/data/checkpoints/"
    echo "  Best model: $(du -sh "$CKPT_DIR/rpm_best.pt" | cut -f1)"
fi

# ── 5. Spotify credentials ──
echo "[5/6] Packing Spotify config..."
mkdir -p "$PACK_DIR/data/config"
cat > "$PACK_DIR/data/config/spotify.env" << 'SPOTEOF'
SPOTIPY_CLIENT_ID=25b596b5d6d949d7974378d8b8223e78
SPOTIPY_CLIENT_SECRET=2a94726254d342ec85d42316a282cc54
SPOTEOF
echo "  Spotify credentials: ✓"

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
echo "  Upload to RunPod:"
echo "    scp -P <SSH_PORT> $OUTPUT root@<POD_IP>:/workspace/"
echo "========================================================================"

# Cleanup
rm -rf "$PACK_DIR"
