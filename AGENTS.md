## Cursor Cloud specific instructions

### Overview

RESONATE is an Electron + React (Vite) + Python (FastAPI) desktop app for AI-powered audio sample matching. It has three main components:

| Component | Path | Stack | Port |
|---|---|---|---|
| Python Backend | `backend/` | FastAPI + Uvicorn + Essentia | 8000 |
| React Frontend | `src/renderer/` | Vite + React | 5173 |
| Electron Shell | `src/main/` | Electron | N/A |

### Running the application

1. **Start backend**: `cd backend && source venv/bin/activate && python server.py`
2. **Start Vite dev server**: `npx vite` (from repo root)
3. **Start Electron**: `NODE_ENV=development npx electron .` (from repo root, after Vite is ready)

Or use the combined script: `npm run dev:full` (starts all three).

The backend must be started separately in dev mode (Electron skips auto-starting it when `NODE_ENV=development`).

### Key caveats

- **No `requirements.txt`**: Python dependencies are not pinned in a manifest file. The venv at `backend/venv/` must contain: `fastapi`, `uvicorn`, `numpy`, `essentia`, `librosa`, `soundfile`, `pydantic`, `python-dotenv`, `python-multipart`, `anthropic`, `pydub`.
- **No test framework**: The codebase has no automated tests (no pytest, jest, vitest, etc.).
- **No linter configuration**: No ESLint/Prettier/Ruff configured.
- **`.env` file**: Copy `backend/.env.example` to `backend/.env`. The `ANTHROPIC_API_KEY` and `CYANITE_API_KEY` are optional — the app falls back to Essentia-only analysis without them.
- **SQLite database**: `backend/resonate.db` is auto-created on first run. No migrations needed.
- **Bridge server**: A TCP socket server on port 9876 starts automatically inside the backend for VST3/AU plugin communication. No separate setup needed.
- **Electron on headless Linux**: dbus errors in the console are expected and non-blocking. Ensure `DISPLAY` is set (`:1` in Cloud Agent VMs).
- **`python-multipart`**: Required by FastAPI for file upload endpoints — not obvious from the codebase but will cause a startup crash if missing.
