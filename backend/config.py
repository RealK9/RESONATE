"""
RESONATE — Configuration & constants.
Loads environment variables and defines shared paths.
"""

import os
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent
SAMPLE_DIR = BACKEND_DIR / "samples"
SAMPLE_DIR.mkdir(exist_ok=True)
TRANSPOSED_DIR = BACKEND_DIR / "transposed_cache"
TRANSPOSED_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = BACKEND_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_INDEX_FILE = BACKEND_DIR / "sample_index.json"
DB_PATH = BACKEND_DIR / "resonate.db"

# ── New analysis pipeline ─────────────────────────────────────────────────
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REFERENCE_CORPUS_PATH = BACKEND_DIR / "reference_corpus.json"
PROFILE_DB_PATH = BACKEND_DIR / "sample_profiles.db"
VECTOR_INDEX_DIR = BACKEND_DIR / "vector_indexes"
VECTOR_INDEX_DIR.mkdir(exist_ok=True)

# ── Supported audio formats ────────────────────────────────────────────────
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}

# ── External sample library paths ─────────────────────────────────────────
HOME = Path.home()
SPLICE_DIRS = [
    HOME / "Splice" / "sounds",
    HOME / "Splice" / "INSTRUMENT",
]
LOOPCLOUD_DIRS = [
    HOME / "Library" / "Application Support" / "Loopcloud" / "downloads",
    HOME / "Loopcloud",
]

# ── API Keys (loaded from environment) ─────────────────────────────────────
CYANITE_TOKEN = os.environ.get("CYANITE_API_KEY", "")
HAS_CYANITE = bool(CYANITE_TOKEN)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Try to load .env file if python-dotenv is available ────────────────────
try:
    from dotenv import load_dotenv
    env_path = BACKEND_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        # Reload after dotenv
        CYANITE_TOKEN = os.environ.get("CYANITE_API_KEY", "")
        HAS_CYANITE = bool(CYANITE_TOKEN)
        ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
except ImportError:
    pass

# ── Cloudflare R2 Storage ─────────────────────────────────────────────────
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "resonate-data")
R2_ENDPOINT_URL = os.environ.get(
    "R2_ENDPOINT_URL",
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else "",
)
HAS_R2 = bool(R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY)

# ── Spotify API ───────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
HAS_SPOTIFY = bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)

# ── Claude client ──────────────────────────────────────────────────────────
claude_client = None
HAS_CLAUDE = False
try:
    import anthropic
    claude_client = anthropic.Anthropic()
    HAS_CLAUDE = True
except Exception:
    pass

# ── Print status ───────────────────────────────────────────────────────────
if HAS_CYANITE:
    print("✓ Cyanite.ai connected (real audio AI)")
else:
    print("✗ Cyanite.ai unavailable — set CYANITE_API_KEY")

if HAS_CLAUDE:
    print("✓ Claude AI connected (text analysis fallback)")
else:
    print("  Claude AI unavailable — Essentia + genre profiles only")

# ── Detect external sample libraries ──────────────────────────────────────
_splice_found = [d for d in SPLICE_DIRS if d.exists()]
_loopcloud_found = [d for d in LOOPCLOUD_DIRS if d.exists()]
if _splice_found:
    print(f"✓ Splice library detected ({len(_splice_found)} directories)")
if _loopcloud_found:
    print(f"✓ Loopcloud library detected ({len(_loopcloud_found)} directories)")
