#!/usr/bin/env python3
"""
RESONATE — Cloudflare R2 Sync Manager
Manages data + model sync between local SSD and R2 cloud storage.

Usage:
    python r2_sync.py push [--all|--models|--embeddings|--datasets|--charts]
    python r2_sync.py pull [--all|--models|--embeddings|--datasets|--charts]
    python r2_sync.py status
    python r2_sync.py ls [prefix]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
RESONATE_HOME = Path.home() / ".resonate"
REMOTE = "r2:resonate-data"

# Data categories and their local paths
SYNC_MAP = {
    "models": {
        "local": RESONATE_HOME / "rpm_training",
        "remote": f"{REMOTE}/models",
        "patterns": ["*.pt", "*.onnx", "*.bin"],
        "description": "Trained RPM models + checkpoints",
    },
    "checkpoints": {
        "local": RESONATE_HOME / "rpm_checkpoints",
        "remote": f"{REMOTE}/checkpoints",
        "patterns": ["*.pt"],
        "description": "Phase A/B/C/D checkpoints",
    },
    "embeddings": {
        "local": RESONATE_HOME / "precomputed",
        "remote": f"{REMOTE}/embeddings/precomputed",
        "patterns": ["*.json"],
        "description": "Pre-computed teacher embeddings (CLAP+PANNs+AST)",
    },
    "datasets": {
        "local": RESONATE_HOME / "datasets",
        "remote": f"{REMOTE}/datasets",
        "patterns": None,  # sync everything
        "description": "FMA, NSynth, and other training datasets",
    },
    "charts": {
        "local": RESONATE_HOME / "charts",
        "remote": f"{REMOTE}/charts",
        "patterns": None,  # sync everything (DB + previews MP3s)
        "description": "Billboard chart data + Deezer/Spotify previews",
    },
    "profiles": {
        "local": Path(__file__).parent.parent / "sample_profiles.db",
        "remote": f"{REMOTE}/db/sample_profiles.db",
        "patterns": None,
        "description": "Sample analysis database",
        "single_file": True,
    },
    "vectors": {
        "local": Path(__file__).parent.parent / "vector_indexes",
        "remote": f"{REMOTE}/vector_indexes",
        "patterns": None,
        "description": "FAISS vector indexes",
    },
}


def run_rclone(args: list[str], dry_run: bool = False) -> int:
    """Run an rclone command."""
    cmd = ["rclone"] + args
    if dry_run:
        cmd.append("--dry-run")
    cmd += [
        "--progress",
        "--transfers", "8",
        "--checkers", "16",
        "--s3-upload-concurrency", "8",
        "--fast-list",
        "--exclude", "._*",       # skip ExFAT resource forks
        "--exclude", ".DS_Store",
    ]
    print(f"  $ {' '.join(cmd[:6])}...")
    return subprocess.call(cmd)


def sync_category(category: str, direction: str, dry_run: bool = False):
    """Sync a single category to/from R2."""
    info = SYNC_MAP[category]
    local = Path(info["local"])
    remote = info["remote"]

    print(f"\n{'─' * 60}")
    print(f"  {category.upper()}: {info['description']}")
    print(f"  Local:  {local}")
    print(f"  Remote: {remote}")

    if info.get("single_file"):
        # Single file sync
        if direction == "push":
            if not local.exists():
                print(f"  ⚠ Skipping — {local} not found")
                return
            run_rclone(["copyto", str(local), remote], dry_run)
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            run_rclone(["copyto", remote, str(local)], dry_run)
    else:
        if direction == "push":
            if not local.exists():
                print(f"  ⚠ Skipping — {local} not found")
                return
            run_rclone(["sync", str(local), remote], dry_run)
        else:
            local.mkdir(parents=True, exist_ok=True)
            run_rclone(["sync", remote, str(local)], dry_run)

    print(f"  ✓ {category} {'pushed' if direction == 'push' else 'pulled'}")


def cmd_sync(direction: str, categories: list[str], dry_run: bool = False):
    """Push or pull selected categories."""
    arrow = "→ R2" if direction == "push" else "← R2"
    print(f"\n{'=' * 60}")
    print(f"  RESONATE R2 Sync — {direction.upper()} {arrow}")
    print(f"  Categories: {', '.join(categories)}")
    print(f"{'=' * 60}")

    t0 = time.time()
    for cat in categories:
        if cat not in SYNC_MAP:
            print(f"\n  ⚠ Unknown category: {cat}")
            continue
        sync_category(cat, direction, dry_run)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"{'=' * 60}")


def cmd_status():
    """Show local data sizes and R2 bucket usage."""
    print(f"\n{'=' * 60}")
    print(f"  RESONATE Data Status")
    print(f"{'=' * 60}")

    total_local = 0
    for cat, info in SYNC_MAP.items():
        local = Path(info["local"])
        if info.get("single_file"):
            size = local.stat().st_size if local.exists() else 0
        elif local.exists():
            size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
        else:
            size = 0
        total_local += size
        size_str = _fmt_size(size) if size > 0 else "—"
        exists = "✓" if size > 0 else "✗"
        print(f"  {exists} {cat:<15} {size_str:>10}   {info['description']}")

    print(f"  {'─' * 55}")
    print(f"    {'Total local:':<15} {_fmt_size(total_local):>10}")

    # R2 bucket size
    print(f"\n  R2 Bucket:")
    result = subprocess.run(
        ["rclone", "size", REMOTE, "--json"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        import json
        data = json.loads(result.stdout)
        print(f"    Objects: {data.get('count', 0)}")
        print(f"    Size:    {_fmt_size(data.get('bytes', 0))}")
    else:
        print(f"    (unable to query)")

    print(f"{'=' * 60}")


def cmd_ls(prefix: str = ""):
    """List R2 bucket contents."""
    target = f"{REMOTE}/{prefix}" if prefix else REMOTE
    subprocess.call(["rclone", "ls", target, "--max-depth", "2"])


def _fmt_size(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def main():
    parser = argparse.ArgumentParser(description="RESONATE R2 Sync Manager")
    sub = parser.add_subparsers(dest="command")

    # push
    p_push = sub.add_parser("push", help="Push data to R2")
    p_push.add_argument("--all", action="store_true")
    p_push.add_argument("--dry-run", action="store_true")
    for cat in SYNC_MAP:
        p_push.add_argument(f"--{cat}", action="store_true")

    # pull
    p_pull = sub.add_parser("pull", help="Pull data from R2")
    p_pull.add_argument("--all", action="store_true")
    p_pull.add_argument("--dry-run", action="store_true")
    for cat in SYNC_MAP:
        p_pull.add_argument(f"--{cat}", action="store_true")

    # status
    sub.add_parser("status", help="Show data status")

    # ls
    p_ls = sub.add_parser("ls", help="List R2 contents")
    p_ls.add_argument("prefix", nargs="?", default="")

    args = parser.parse_args()

    if args.command in ("push", "pull"):
        if args.all:
            categories = list(SYNC_MAP.keys())
        else:
            categories = [c for c in SYNC_MAP if getattr(args, c, False)]
        if not categories:
            print("Specify --all or specific categories (e.g. --models --embeddings)")
            sys.exit(1)
        cmd_sync(args.command, categories, dry_run=getattr(args, "dry_run", False))

    elif args.command == "status":
        cmd_status()

    elif args.command == "ls":
        cmd_ls(args.prefix)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
