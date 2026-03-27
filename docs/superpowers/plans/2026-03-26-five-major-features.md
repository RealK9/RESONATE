# Five Major Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship 5 features that transform RESONATE from a recommendation engine into a complete production intelligence platform: in-context sample preview, producer taste profiles, chart intelligence, multi-track sessions, and smart collection curation.

**Architecture:** Each feature is self-contained with its own backend routes, frontend components, and tests. Features share the existing MixProfile/SampleStore/PreferenceDataset infrastructure. Frontend uses React 19 with inline styles, custom fonts (SERIF, AF, MONO), and the existing theme system. Backend is FastAPI with SQLite persistence.

**Tech Stack:** React 19, Web Audio API, FastAPI, SQLite, Python 3.9+, FAISS, inline CSS

---

## Feature 1: In-Context Sample Preview

**What:** Users hear recommended samples layered over their mix in real-time. Click a sample → it plays mixed with the track at matched tempo/key. Volume faders for independent control.

**Why this works:** useAudioPlayer.js already has dual-channel Web Audio mixing (sample + track GainNodes), mix mode toggle, and independent volume control. We just need to wire it to the recommendation UX.

### File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/renderer/hooks/useAudioPlayer.js` | Add `previewInContext(sampleId)` method that auto-enables mix mode |
| Modify | `src/renderer/components/SampleRow.jsx` | Add in-context play button (layered icon) next to existing play |
| Create | `src/renderer/components/MixPreviewBar.jsx` | Floating bar at bottom: dual waveforms, volume faders, A/B toggle |
| Modify | `src/renderer/App.jsx` | Wire MixPreviewBar, manage preview state |
| Create | `tests/test_mix_preview_bar.test.jsx` | Component tests for MixPreviewBar |
| Modify | `backend/routes/samples.py` | Add `?context=mix` param to audio endpoint for pre-mixed preview |

### Task 1.1: In-Context Play Method

**Files:**
- Modify: `src/renderer/hooks/useAudioPlayer.js`

- [ ] **Step 1: Add `previewInContext` to useAudioPlayer**

Add a method that: loads the sample, enables mix mode, starts playback with track synced. This builds on the existing `toggle()` + `toggleMix()` flow but combines them into one action.

```javascript
// Add after the existing toggle function (~line 180)
const previewInContext = useCallback(async (sampleId, samplePath) => {
  // If already playing this sample in context, stop
  if (playing && currentId === sampleId && mixMode) {
    stop();
    return;
  }
  // Enable mix mode before starting playback
  if (!mixMode) toggleMix();
  // Load and play the sample
  await toggle(sampleId, samplePath);
}, [playing, currentId, mixMode, toggleMix, toggle, stop]);
```

Add to the return object: `previewInContext`.

- [ ] **Step 2: Verify existing mix mode still works**

Run the dev server, upload a track, play a sample, toggle mix mode manually. Confirm dual-channel playback works.

- [ ] **Step 3: Commit**

```bash
git add src/renderer/hooks/useAudioPlayer.js
git commit -m "feat: add previewInContext method to audio player"
```

### Task 1.2: MixPreviewBar Component

**Files:**
- Create: `src/renderer/components/MixPreviewBar.jsx`

- [ ] **Step 1: Create the MixPreviewBar component**

Floating bar at bottom of results screen. Shows:
- Track name (left) + Sample name (right) with dual mini-waveforms
- Two volume sliders (track / sample) using existing VolumeSlider pattern
- Play/pause, stop, A/B toggle (mute sample to compare)
- Progress bar spanning full width

```jsx
import { MONO, AF } from "../theme/fonts";

export function MixPreviewBar({ theme, isDark, audio, activeSample, fileName }) {
  if (!audio.playing || !audio.mixMode) return null;

  return (
    <div style={{
      position: "fixed", bottom: 0, left: 0, right: 0, height: 64,
      background: isDark ? "rgba(13,13,18,0.95)" : "rgba(255,255,255,0.95)",
      backdropFilter: "blur(20px)", borderTop: "1px solid " + theme.border,
      display: "flex", alignItems: "center", padding: "0 20px", gap: 16,
      zIndex: 100,
    }}>
      {/* Track info */}
      <div style={{ minWidth: 120 }}>
        <div style={{ fontSize: 8, color: theme.textFaint, fontFamily: AF, textTransform: "uppercase", letterSpacing: 1.5 }}>Your Mix</div>
        <div style={{ fontSize: 10, color: theme.text, fontFamily: AF, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{fileName || "Track"}</div>
      </div>

      {/* Track volume */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 8, color: theme.textMuted, fontFamily: AF }}>MIX</span>
        <input type="range" min="0" max="1" step="0.01"
          value={audio.trackVol}
          onChange={e => audio.setTrackVol(parseFloat(e.target.value))}
          style={{ width: 60, accentColor: theme.accent }} />
      </div>

      {/* Playback controls */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <button onClick={() => audio.stop()} style={{
          width: 28, height: 28, borderRadius: 14, border: "1px solid " + theme.border,
          background: "transparent", color: theme.text, cursor: "pointer",
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10,
        }}>
          {/* Stop icon */}
          <svg width="10" height="10" viewBox="0 0 10 10"><rect x="1" y="1" width="8" height="8" fill="currentColor" rx="1" /></svg>
        </button>
        {/* A/B toggle — mute sample to hear mix alone */}
        <button onClick={() => audio.setSampleVol(audio.sampleVol > 0 ? 0 : 0.8)} style={{
          padding: "4px 10px", borderRadius: 5, fontSize: 9, fontWeight: 700, fontFamily: AF,
          border: "1px solid " + (audio.sampleVol > 0 ? "rgba(217,70,239,0.3)" : theme.border),
          background: audio.sampleVol > 0 ? "rgba(217,70,239,0.1)" : "transparent",
          color: audio.sampleVol > 0 ? "#D946EF" : theme.textMuted, cursor: "pointer",
        }}>
          A/B
        </button>
      </div>

      {/* Progress bar */}
      <div style={{ flex: 1, height: 3, borderRadius: 2, background: theme.borderLight, cursor: "pointer", position: "relative" }}
        onClick={e => { const rect = e.currentTarget.getBoundingClientRect(); audio.seek((e.clientX - rect.left) / rect.width); }}>
        <div style={{ height: "100%", borderRadius: 2, background: "linear-gradient(90deg, #D946EF, #06B6D4)", width: `${(audio.progress || 0) * 100}%`, transition: "width 0.1s linear" }} />
      </div>

      {/* Sample volume */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 8, color: theme.textMuted, fontFamily: AF }}>SAMPLE</span>
        <input type="range" min="0" max="1" step="0.01"
          value={audio.sampleVol}
          onChange={e => audio.setSampleVol(parseFloat(e.target.value))}
          style={{ width: 60, accentColor: "#D946EF" }} />
      </div>

      {/* Sample info */}
      <div style={{ minWidth: 120, textAlign: "right" }}>
        <div style={{ fontSize: 8, color: theme.textFaint, fontFamily: AF, textTransform: "uppercase", letterSpacing: 1.5 }}>Sample</div>
        <div style={{ fontSize: 10, color: "#D946EF", fontFamily: AF, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{activeSample?.clean_name || activeSample?.name || "—"}</div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/renderer/components/MixPreviewBar.jsx
git commit -m "feat: add MixPreviewBar floating component"
```

### Task 1.3: Wire Into App + SampleRow

**Files:**
- Modify: `src/renderer/App.jsx`
- Modify: `src/renderer/components/SampleRow.jsx`

- [ ] **Step 1: Add in-context play button to SampleRow**

Read SampleRow.jsx. Add a second play button (layered waveform icon) that calls `onPreviewInContext(sample)` instead of `onPlay(sample)`. Show it only when `_isV2` is true (AI-matched samples).

- [ ] **Step 2: Import and render MixPreviewBar in App.jsx**

Add `import { MixPreviewBar } from "./components/MixPreviewBar"` and render it at the bottom of the results screen, passing `theme, isDark, audio, activeSample, fileName`.

- [ ] **Step 3: Wire previewInContext through props**

In App.jsx, pass `onPreviewInContext={(sample) => audio.previewInContext(sample.id, sample.path)}` to SampleRow.

- [ ] **Step 4: Add bottom padding to sample list when MixPreviewBar visible**

When `audio.playing && audio.mixMode`, add 64px bottom padding to the scroll container so the last samples aren't hidden behind the bar.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/App.jsx src/renderer/components/SampleRow.jsx
git commit -m "feat: wire in-context preview — layered play button + floating bar"
```

### Task 1.4: Feedback Integration

**Files:**
- Modify: `src/renderer/App.jsx`

- [ ] **Step 1: Log `audition_in_context` feedback events**

When a user plays a sample in context mode, fire `logFeedbackV2({ sample_filepath, action: "audition", ... })`. When they use A/B toggle, log that too. This feeds the preference learning system.

- [ ] **Step 2: Commit**

```bash
git add src/renderer/App.jsx
git commit -m "feat: log in-context audition events for preference learning"
```

---

## Feature 2: Producer DNA / Taste Profiles

**What:** Visual taste profile showing the user's sonic identity — genre affinities, role preferences, evolving taste over time. The backend already has `UserTasteModel` with `role_bias`, `style_bias`, and `weight_deltas`. We need a UI to display it and auto-train after sessions.

### File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/renderer/components/ProducerDNA.jsx` | Full taste profile visualization panel |
| Create | `backend/routes/taste_profile.py` | API endpoint for formatted taste data |
| Modify | `backend/server.py` | Register new route |
| Modify | `src/renderer/hooks/useApi.js` | Add taste profile API method |
| Modify | `src/renderer/App.jsx` | Add DNA tab/panel in results screen |
| Create | `tests/test_taste_profile_route.py` | Backend route tests |

### Task 2.1: Taste Profile API

**Files:**
- Create: `backend/routes/taste_profile.py`
- Modify: `backend/server.py` (or main app file that registers routers)

- [ ] **Step 1: Create taste_profile route**

```python
"""RESONATE — Taste Profile route. Returns formatted producer DNA data."""

from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/taste/profile")
async def get_taste_profile(user_id: str = "default"):
    """Return the producer's taste DNA — role affinities, genre leanings, evolution."""
    from ml.training.preference_dataset import PreferenceDataset
    from ml.training.preference_serving import PreferenceServer
    from config import PROFILE_DB_PATH

    pref_db_path = str(PROFILE_DB_PATH).replace(".db", "_prefs.db")
    ds = PreferenceDataset(pref_db_path)
    ds.init()

    model = ds.load_taste_model(user_id)
    if model is None:
        return {"status": "no_data", "message": "Use RESONATE more to build your taste profile"}

    # Format role affinities as sorted list
    role_affinities = sorted(
        [{"role": r, "affinity": round(v, 3)} for r, v in model.role_bias.items()],
        key=lambda x: abs(x["affinity"]), reverse=True
    )

    # Format style preferences
    style_prefs = sorted(
        [{"style": s, "preference": round(v, 3)} for s, v in model.style_bias.items()],
        key=lambda x: abs(x["preference"]), reverse=True
    )

    # Feedback stats — query the SQLite table directly
    total_interactions = 0
    action_breakdown = {}
    try:
        import sqlite3
        conn = sqlite3.connect(pref_db_path)
        rows = conn.execute(
            "SELECT action, COUNT(*) FROM feedback_events GROUP BY action"
        ).fetchall()
        conn.close()
        for action, count in rows:
            action_breakdown[action] = count
            total_interactions += count
    except Exception:
        pass  # Stats are optional — don't fail the whole endpoint

    return {
        "status": "ok",
        "user_id": model.user_id,
        "model_version": model.model_version,
        "training_pairs": model.training_pairs,
        "last_trained": model.last_trained,
        "role_affinities": role_affinities,
        "style_preferences": style_prefs,
        "quality_threshold": round(model.quality_threshold, 3),
        "weight_profile": model.weight_deltas,
        "total_interactions": total_interactions,
        "action_breakdown": action_breakdown,
    }

@router.post("/taste/train")
async def train_taste_profile(user_id: str = "default"):
    """Trigger a retrain of the taste model from accumulated feedback."""
    from ml.training.preference_dataset import PreferenceDataset
    from ml.training.train_ranker import RankerTrainer
    from ml.db.sample_store import SampleStore
    from config import PROFILE_DB_PATH

    pref_db_path = str(PROFILE_DB_PATH).replace(".db", "_prefs.db")
    ds = PreferenceDataset(pref_db_path)
    ds.init()

    store = SampleStore(str(PROFILE_DB_PATH))
    store.init()

    # Build pairs from recent feedback
    pairs = ds.build_pairs()

    trainer = RankerTrainer(dataset=ds, sample_store=store)
    model = trainer.train(user_id=user_id, min_pairs=5)

    if model is None:
        return {"status": "insufficient_data", "message": "Keep using RESONATE — need more feedback"}

    return {
        "status": "ok",
        "training_pairs": model.training_pairs,
        "model_version": model.model_version,
    }
```

- [ ] **Step 2: Register route in server**

Find the main FastAPI app file and add `from routes.taste_profile import router as taste_router` and `app.include_router(taste_router)`.

- [ ] **Step 3: Write tests**

Create `tests/test_taste_profile_route.py` testing both endpoints with mocked PreferenceDataset.

- [ ] **Step 4: Commit**

```bash
git add backend/routes/taste_profile.py backend/server.py tests/test_taste_profile_route.py
git commit -m "feat: taste profile API — producer DNA endpoint"
```

### Task 2.2: Add API Method

**Files:**
- Modify: `src/renderer/hooks/useApi.js`

- [ ] **Step 1: Add getTasteProfile and trainTaste methods**

```javascript
const getTasteProfile = async (userId = "default") => {
  const r = await fetch(API + `/taste/profile?user_id=${userId}`);
  return r.json();
};

const trainTaste = async (userId = "default") => {
  const r = await fetch(API + `/taste/train?user_id=${userId}`, { method: "POST" });
  return r.json();
};
```

Add both to the return object.

- [ ] **Step 2: Commit**

```bash
git add src/renderer/hooks/useApi.js
git commit -m "feat: add taste profile API methods"
```

### Task 2.3: ProducerDNA Component

**Files:**
- Create: `src/renderer/components/ProducerDNA.jsx`

- [ ] **Step 1: Build the ProducerDNA panel**

A full panel showing:
- **Radar chart** of role affinities (kick, snare, bass, lead, pad, etc.) using SVG polygon
- **Genre bars** showing style preferences as horizontal bars, sorted by strength
- **Stats row**: total interactions, training pairs, model version
- **Quality threshold** indicator — where the user's taste sits on the quality spectrum
- **Weight profile** — what the user values most (spectral complement vs tonal compatibility vs rhythmic feel)
- **Train button** to retrain the model

Use SVG for the radar chart. All inline styles matching existing RESONATE design.

```jsx
import { useState, useEffect } from "react";
import { AF, MONO, SERIF } from "../theme/fonts";

const ROLE_ORDER = ["kick", "snare_clap", "bass", "lead", "pad", "hats_tops", "vocal_texture", "fx_transitions", "percussion"];
const ROLE_SHORT = { kick: "Kick", snare_clap: "Snare", bass: "Bass", lead: "Lead", pad: "Pad", hats_tops: "Hats", vocal_texture: "Vocal", fx_transitions: "FX", percussion: "Perc" };

function RadarChart({ roleAffinities, size = 180, theme }) {
  const cx = size / 2, cy = size / 2, r = size * 0.38;
  const n = ROLE_ORDER.length;

  // Normalize affinities to 0-1 (from -1..1)
  const vals = ROLE_ORDER.map(role => {
    const entry = roleAffinities.find(a => a.role === role);
    return entry ? (entry.affinity + 1) / 2 : 0.5; // 0.5 = neutral
  });

  const getPoint = (i, value) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    return { x: cx + r * value * Math.cos(angle), y: cy + r * value * Math.sin(angle) };
  };

  const gridLevels = [0.25, 0.5, 0.75, 1.0];
  const polygon = vals.map((v, i) => getPoint(i, v)).map(p => `${p.x},${p.y}`).join(" ");

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* Grid rings */}
      {gridLevels.map(level => (
        <polygon key={level} fill="none" stroke={theme.borderLight} strokeWidth="0.5"
          points={ROLE_ORDER.map((_, i) => getPoint(i, level)).map(p => `${p.x},${p.y}`).join(" ")} />
      ))}
      {/* Axis lines */}
      {ROLE_ORDER.map((_, i) => {
        const p = getPoint(i, 1);
        return <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke={theme.borderLight} strokeWidth="0.5" />;
      })}
      {/* Data polygon */}
      <polygon points={polygon} fill="rgba(217,70,239,0.15)" stroke="#D946EF" strokeWidth="1.5" />
      {/* Data dots + labels */}
      {vals.map((v, i) => {
        const p = getPoint(i, v);
        const lp = getPoint(i, 1.18);
        return (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r="3" fill="#D946EF" />
            <text x={lp.x} y={lp.y} textAnchor="middle" dominantBaseline="middle"
              fill={theme.textMuted} fontSize="8" fontFamily="var(--font-af, sans-serif)">
              {ROLE_SHORT[ROLE_ORDER[i]] || ROLE_ORDER[i]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function ProducerDNA({ profile, onTrain, theme, isDark }) {
  if (!profile || profile.status === "no_data") {
    return (
      <div style={{ padding: 20, textAlign: "center" }}>
        <div style={{ fontSize: 13, color: theme.textMuted, fontFamily: AF, marginBottom: 8 }}>
          Your Producer DNA builds as you use RESONATE
        </div>
        <div style={{ fontSize: 10, color: theme.textFaint, fontFamily: AF }}>
          Keep auditioning, rating, and keeping samples to train your taste model
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: 16 }}>
      {/* Radar Chart */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <RadarChart roleAffinities={profile.role_affinities} theme={theme} />
      </div>

      {/* Genre Preferences */}
      {profile.style_preferences.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 8, color: theme.textFaint, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 6, fontFamily: AF }}>Genre Affinity</div>
          {profile.style_preferences.slice(0, 6).map(s => (
            <div key={s.style} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <span style={{ fontSize: 10, color: theme.textSec, fontFamily: AF, minWidth: 80 }}>{s.style}</span>
              <div style={{ flex: 1, height: 4, borderRadius: 2, background: theme.borderLight, overflow: "hidden" }}>
                <div style={{ height: "100%", borderRadius: 2, background: s.preference > 0 ? "#D946EF" : "#6366F1",
                  width: `${Math.abs(s.preference) * 100}%`, transition: "width 0.3s" }} />
              </div>
              <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: MONO, minWidth: 30, textAlign: "right" }}>
                {s.preference > 0 ? "+" : ""}{(s.preference * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Stats */}
      <div style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderTop: "1px solid " + theme.borderLight, marginBottom: 12 }}>
        {[
          ["Interactions", profile.total_interactions],
          ["Training Pairs", profile.training_pairs],
          ["Model v", profile.model_version],
        ].map(([label, val]) => (
          <div key={label} style={{ textAlign: "center" }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: theme.text, fontFamily: MONO }}>{val}</div>
            <div style={{ fontSize: 8, color: theme.textFaint, fontFamily: AF }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Train button */}
      <button onClick={onTrain} style={{
        width: "100%", padding: "8px 0", borderRadius: 6, border: "1px solid rgba(217,70,239,0.3)",
        background: "rgba(217,70,239,0.08)", color: "#D946EF", fontSize: 10, fontWeight: 700,
        fontFamily: AF, cursor: "pointer",
      }}>
        Retrain Taste Model
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/renderer/components/ProducerDNA.jsx
git commit -m "feat: ProducerDNA visual component — radar chart + taste bars"
```

### Task 2.4: Wire Into App

**Files:**
- Modify: `src/renderer/App.jsx`

- [ ] **Step 1: Add DNA tab to results screen**

Add a "Your DNA" tab next to "AI Matched" and "Favorites". When selected, fetch taste profile and render ProducerDNA.

- [ ] **Step 2: Auto-train after session save**

After `saveSession()` completes, fire `api.trainTaste()` in the background to keep the model up-to-date.

- [ ] **Step 3: Commit**

```bash
git add src/renderer/App.jsx
git commit -m "feat: wire ProducerDNA into results screen with auto-train"
```

---

## Feature 3: Chart Intelligence 2.0

**What:** Show producers how their production compares to what's charting. "Tracks in your genre that charted this year: avg BPM 142, mostly minor keys, high energy." The chart analysis module already exists — we need an API to serve it and a UI to display it.

### File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `backend/routes/chart_intelligence.py` | API endpoints for chart trends/comparisons |
| Create | `src/renderer/components/ChartIntel.jsx` | Chart intelligence visualization panel |
| Modify | `src/renderer/hooks/useApi.js` | Add chart intelligence API methods |
| Modify | `src/renderer/App.jsx` | Wire ChartIntel into sidebar or summary |
| Modify | `backend/server.py` | Register chart routes |
| Create | `tests/test_chart_intelligence.py` | Route tests |

### Task 3.1: Chart Intelligence API

**Files:**
- Create: `backend/routes/chart_intelligence.py`

- [ ] **Step 1: Create the chart intelligence route**

Endpoints:
- `GET /charts/trends` — Decade profiles, genre profiles, overall trends
- `GET /charts/compare` — Compare current mix against chart averages for its genre
- `GET /charts/insights` — AI-generated insights about the user's mix vs charts

The compare endpoint uses `state.latest_mix_profile` and matches it against chart analysis data.

```python
"""RESONATE — Chart Intelligence routes."""

from dataclasses import asdict
from fastapi import APIRouter, HTTPException
import state

router = APIRouter()

# Chart DB path — same directory as other DBs
def _get_chart_db_path():
    from config import BACKEND_DIR
    return str(BACKEND_DIR / "chart_features.db")

def _load_analyzer():
    """Load and run ChartAnalyzer. Returns analyzer or raises 503."""
    from ml.training.charts.chart_analysis import ChartAnalyzer
    try:
        analyzer = ChartAnalyzer(_get_chart_db_path())
        analyzer.analyze()  # Must call analyze() to populate profiles
        return analyzer
    except Exception as e:
        raise HTTPException(503, f"Chart database not available: {e}")

def _profile_to_dict(profile) -> dict:
    """Serialize a DecadeProfile or GenreProfile dataclass to dict."""
    d = asdict(profile)
    # Convert numpy arrays to lists if present
    for k, v in d.items():
        if hasattr(v, 'tolist'):
            d[k] = v.tolist()
    return d

@router.get("/charts/trends")
async def get_chart_trends(genre: str = "", decade: str = ""):
    """Return chart trend data — BPM/key/energy distributions over time."""
    analyzer = _load_analyzer()

    result = {"decades": {}, "genres": {}}

    # get_decade_profiles() returns dict[int, DecadeProfile]
    decade_profiles = analyzer.get_decade_profiles()
    if decade:
        decade_int = int(decade.rstrip("s"))  # "2020s" -> 2020
        if decade_int in decade_profiles:
            result["decades"][str(decade_int)] = _profile_to_dict(decade_profiles[decade_int])
    else:
        for d, profile in decade_profiles.items():
            result["decades"][str(d)] = _profile_to_dict(profile)

    # get_genre_profiles() returns dict[str, GenreProfile]
    genre_profiles = analyzer.get_genre_profiles()
    if genre:
        if genre in genre_profiles:
            result["genres"][genre] = _profile_to_dict(genre_profiles[genre])
    else:
        for g, profile in genre_profiles.items():
            result["genres"][g] = _profile_to_dict(profile)

    return result

@router.get("/charts/compare")
async def compare_to_charts():
    """Compare the latest analyzed mix against chart data for its genre."""
    if state.latest_mix_profile is None:
        raise HTTPException(404, "No mix analyzed yet")

    analyzer = _load_analyzer()

    mix = state.latest_mix_profile
    mix_dict = mix.to_dict() if hasattr(mix, 'to_dict') else mix
    genre = mix_dict.get("style", {}).get("primary_cluster", "")
    analysis = mix_dict.get("analysis", {})
    bpm = analysis.get("bpm", 0)
    key = analysis.get("key", "")

    genre_profiles = analyzer.get_genre_profiles()
    decade_profiles = analyzer.get_decade_profiles()

    comparison = {
        "your_mix": {"bpm": bpm, "key": key, "genre": genre},
        "chart_average": {},
        "insights": [],
    }

    # Match genre profile
    gp = genre_profiles.get(genre)
    if gp:
        comparison["chart_average"] = {
            "avg_bpm": round(gp.bpm_mean, 1),
            "avg_energy": round(gp.energy_mean, 3),
            "avg_valence": round(gp.valence_mean, 3),
            "avg_danceability": round(gp.danceability_mean, 3),
            "major_ratio": round(gp.major_ratio, 2),
            "total_chart_entries": gp.count,
        }

        # Generate insights
        if bpm and gp.bpm_mean:
            diff = bpm - gp.bpm_mean
            if abs(diff) > 10:
                direction = "faster" if diff > 0 else "slower"
                comparison["insights"].append(
                    f"Your BPM ({bpm:.0f}) is {abs(diff):.0f} BPM {direction} than the {genre} chart average ({gp.bpm_mean:.0f})"
                )
            else:
                comparison["insights"].append(
                    f"Your BPM ({bpm:.0f}) is right in the sweet spot for charting {genre} tracks ({gp.bpm_mean:.0f} avg)"
                )

    # Current decade trends
    dp = decade_profiles.get(2020)
    if dp:
        comparison["decade_trends"] = {
            "decade": "2020s",
            "avg_bpm": round(dp.bpm_mean, 1),
            "avg_energy": round(dp.energy_mean, 3),
            "major_ratio": round(dp.major_ratio, 2),
        }

    return comparison
```

- [ ] **Step 2: Register route, write tests**
- [ ] **Step 3: Commit**

```bash
git add backend/routes/chart_intelligence.py backend/server.py tests/test_chart_intelligence.py
git commit -m "feat: chart intelligence API — trends, comparison, insights"
```

### Task 3.2: ChartIntel Component

**Files:**
- Create: `src/renderer/components/ChartIntel.jsx`

- [ ] **Step 1: Build the ChartIntel panel**

Shows in the sidebar when chart data is available:
- **Your Mix vs Charts** comparison header
- BPM comparison bar (your BPM marker vs genre average range)
- Key match indicator
- Energy/valence comparison dots
- Text insights from the API
- Decade trend sparkline showing how BPM/energy evolved

All inline styles, matching RESONATE design language.

- [ ] **Step 2: Commit**

```bash
git add src/renderer/components/ChartIntel.jsx
git commit -m "feat: ChartIntel visualization component"
```

### Task 3.3: Wire Into App

**Files:**
- Modify: `src/renderer/hooks/useApi.js`
- Modify: `src/renderer/App.jsx`

- [ ] **Step 1: Add API methods**

```javascript
const getChartTrends = async (genre = "", decade = "") => {
  const params = new URLSearchParams();
  if (genre) params.set("genre", genre);
  if (decade) params.set("decade", decade);
  const r = await fetch(API + "/charts/trends?" + params.toString());
  return r.json();
};

const getChartComparison = async () => {
  const r = await fetch(API + "/charts/compare");
  return r.json();
};
```

- [ ] **Step 2: Fetch chart comparison after analysis completes**

In `runAnalysis()`, after gap analysis is set, fire `api.getChartComparison()` and store in `chartComparison` state. Render ChartIntel in the sidebar below the gap analysis panel.

- [ ] **Step 3: Commit**

```bash
git add src/renderer/hooks/useApi.js src/renderer/App.jsx
git commit -m "feat: wire chart intelligence into analysis flow"
```

---

## Feature 4: Multi-Track Session Support

**What:** Analyze multiple versions of a track or individual stems. Compare readiness scores across versions. Track a project's evolution from demo to final. Sessions already exist — we extend them with version tracking and comparison.

### File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `backend/routes/versions.py` | Version tracking API |
| Create | `backend/db/versions.py` | SQLite version storage |
| Create | `src/renderer/components/VersionTimeline.jsx` | Visual timeline of track versions |
| Modify | `src/renderer/App.jsx` | Version tracking, comparison mode |
| Modify | `src/renderer/hooks/useApi.js` | Version API methods |
| Modify | `backend/server.py` | Register version routes |
| Create | `tests/test_versions.py` | Version tracking tests |

### Task 4.1: Version Database

**Files:**
- Create: `backend/db/versions.py`

- [ ] **Step 1: Create version storage module**

SQLite table `track_versions`:
```sql
CREATE TABLE IF NOT EXISTS track_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,
    version_label TEXT NOT NULL,
    filepath TEXT NOT NULL,
    readiness_score REAL,
    gap_summary TEXT,
    chart_potential REAL,
    missing_roles TEXT,
    analysis_json TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_versions_project ON track_versions(project_name);
```

Methods: `save_version()`, `get_versions(project_name)`, `get_latest(project_name)`, `compare_versions(id_a, id_b)`, `list_projects()`.

- [ ] **Step 2: Write tests**
- [ ] **Step 3: Commit**

```bash
git add backend/db/versions.py tests/test_versions.py
git commit -m "feat: version tracking database"
```

### Task 4.2: Version Routes

**Files:**
- Create: `backend/routes/versions.py`

- [ ] **Step 1: Create version API endpoints**

```python
POST /versions            — Save current analysis as a version
GET  /versions/{project}  — List all versions for a project
GET  /versions/compare    — Compare two versions (readiness delta, new/resolved gaps)
GET  /versions/projects   — List all projects
```

Each save captures: readiness score, gap summary, chart potential, missing roles, full analysis JSON.

- [ ] **Step 2: Register route, write tests**
- [ ] **Step 3: Commit**

```bash
git add backend/routes/versions.py backend/server.py tests/test_versions_route.py
git commit -m "feat: version tracking API"
```

### Task 4.3: VersionTimeline Component

**Files:**
- Create: `src/renderer/components/VersionTimeline.jsx`

- [ ] **Step 1: Build the VersionTimeline component**

Visual timeline showing:
- Horizontal timeline with version dots
- Each dot shows readiness score with color coding
- Click a dot to see that version's analysis
- Delta arrows between versions (↑12 or ↓3)
- "Save Version" button at the end of the timeline

```jsx
export function VersionTimeline({ versions, currentReadiness, onSaveVersion, onSelectVersion, theme, isDark }) {
  // Horizontal timeline with dots, scores, and deltas
  // Each version: { id, version_label, readiness_score, created_at, missing_roles }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/renderer/components/VersionTimeline.jsx
git commit -m "feat: VersionTimeline visual component"
```

### Task 4.4: Wire Into App

**Files:**
- Modify: `src/renderer/hooks/useApi.js`
- Modify: `src/renderer/App.jsx`

- [ ] **Step 1: Add version API methods to useApi**

```javascript
const saveVersion = async (projectName, versionLabel) => { ... };
const getVersions = async (projectName) => { ... };
const compareVersions = async (idA, idB) => { ... };
const listProjects = async () => { ... };
```

- [ ] **Step 2: Add version timeline to results screen**

Render VersionTimeline above the AI Summary Card. Auto-detect project name from filename. Show "Save as v1/v2/v3..." based on existing versions.

- [ ] **Step 3: Auto-save version on analysis**

After each analysis completes (including re-analysis), auto-save a version with timestamp-based label.

- [ ] **Step 4: Commit**

```bash
git add src/renderer/hooks/useApi.js src/renderer/App.jsx
git commit -m "feat: wire version timeline into analysis flow"
```

---

## Feature 5: Smart Collection Curation

**What:** Auto-generate themed sample kits from gap analysis. "Your Trap Essentials — 10 samples that would most improve your track." One-click export as ZIP. The export endpoint already exists — we build the curation logic and UI.

### File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `backend/routes/collections.py` | Smart collection generation + export |
| Create | `src/renderer/components/SmartCollection.jsx` | Collection card with preview + export |
| Modify | `src/renderer/hooks/useApi.js` | Collection API methods |
| Modify | `src/renderer/App.jsx` | Render collections below recommendations |
| Create | `tests/test_collections.py` | Collection generation tests |

### Task 5.1: Collection Generation API

**Files:**
- Create: `backend/routes/collections.py`

- [ ] **Step 1: Create smart collection route**

```python
"""RESONATE — Smart Collection Curation."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import zipfile, io, os, json

import state

router = APIRouter()

@router.get("/collections/generate")
async def generate_collections():
    """Generate smart collections based on the latest gap analysis."""
    if state.latest_mix_profile is None:
        raise HTTPException(404, "No mix analyzed yet")
    if state.latest_recommendations is None:
        raise HTTPException(404, "No recommendations yet — run analysis first")

    recs = state.latest_recommendations.recommendations
    gap = state.latest_gap_result
    mix = state.latest_mix_profile
    mix_dict = mix.to_dict() if hasattr(mix, 'to_dict') else (mix or {})
    genre = mix_dict.get("style", {}).get("primary_cluster", "your")

    # Derive missing roles from gap result or from needs with fill_missing_role policy
    missing_roles = []
    if gap and isinstance(gap, dict):
        missing_roles = gap.get("missing_roles", [])
    if not missing_roles and mix_dict.get("needs"):
        missing_roles = [
            n.get("description", "").split()[-1]  # best-effort role extraction
            for n in mix_dict["needs"]
            if n.get("recommendation_policy") == "fill_missing_role"
        ]

    collections = []

    # Collection 1: Gap Fillers — samples that address missing roles
    if missing_roles:
        gap_fillers = [r for r in recs if r.policy == "fill_missing_role"][:10]
        if gap_fillers:
            collections.append({
                "id": "gap_fillers",
                "name": f"{_genre_label(genre)} Essentials",
                "description": f"Top {len(gap_fillers)} samples to fill the gaps in your mix",
                "icon": "target",
                "samples": [_rec_to_sample(r) for r in gap_fillers],
                "total_impact": round(sum(r.score for r in gap_fillers), 2),
            })

    # Collection 2: Polish Pack — samples that add commercial quality
    polish = [r for r in recs if r.policy in ("improve_polish", "enhance_lift")][:8]
    if polish:
        collections.append({
            "id": "polish_pack",
            "name": "Polish Pack",
            "description": "Samples that push your mix toward radio-ready",
            "icon": "sparkle",
            "samples": [_rec_to_sample(r) for r in polish],
            "total_impact": round(sum(r.score for r in polish), 2),
        })

    # Collection 3: Groove Kit — rhythmic elements
    groove = [r for r in recs if r.policy in ("enhance_groove", "add_movement")][:8]
    if groove:
        collections.append({
            "id": "groove_kit",
            "name": "Groove Kit",
            "description": "Lock in the rhythmic foundation",
            "icon": "rhythm",
            "samples": [_rec_to_sample(r) for r in groove],
            "total_impact": round(sum(r.score for r in groove), 2),
        })

    # Collection 4: Top 10 Overall
    top10 = recs[:10]
    if top10:
        collections.append({
            "id": "top_10",
            "name": "Top 10 Picks",
            "description": "The best samples for your mix, period",
            "icon": "crown",
            "samples": [_rec_to_sample(r) for r in top10],
            "total_impact": round(sum(r.score for r in top10), 2),
        })

    return {"collections": collections}


@router.post("/collections/export/{collection_id}")
async def export_collection(collection_id: str):
    """Export a smart collection as a ZIP file."""
    # Regenerate to get fresh data
    collections_data = await generate_collections()
    collection = next(
        (c for c in collections_data["collections"] if c["id"] == collection_id),
        None,
    )
    if not collection:
        raise HTTPException(404, f"Collection '{collection_id}' not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sample in collection["samples"]:
            filepath = sample["filepath"]
            if os.path.isfile(filepath):
                arcname = os.path.basename(filepath)
                zf.write(filepath, arcname)

        # Add metadata
        meta = {
            "collection": collection["name"],
            "description": collection["description"],
            "samples": [s["name"] for s in collection["samples"]],
            "generated_by": "RESONATE by SONIQlabs",
        }
        zf.writestr("_resonate_collection.json", json.dumps(meta, indent=2))

    buf.seek(0)
    filename = f"RESONATE-{collection['name'].replace(' ', '-')}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _genre_label(cluster: str) -> str:
    labels = {
        "modern_trap": "Trap", "modern_drill": "Drill",
        "lo_fi_hip_hop": "Lo-Fi", "boom_bap": "Boom Bap",
        "rnb_soul": "R&B", "pop_electronic": "Pop",
        "house_tech_house": "House", "future_bass": "Future Bass",
    }
    return labels.get(cluster, cluster.replace("_", " ").title())


def _rec_to_sample(rec) -> dict:
    return {
        "filepath": rec.filepath,
        "name": rec.filename or os.path.basename(rec.filepath),
        "role": rec.role,
        "score": round(rec.score, 3),
        "explanation": rec.explanation or "",
    }
```

- [ ] **Step 2: Register route, write tests**
- [ ] **Step 3: Commit**

```bash
git add backend/routes/collections.py backend/server.py tests/test_collections.py
git commit -m "feat: smart collection generation + ZIP export API"
```

### Task 5.2: SmartCollection Component

**Files:**
- Create: `src/renderer/components/SmartCollection.jsx`

- [ ] **Step 1: Build the SmartCollection card**

Each collection renders as a card:
- Title + icon + description
- Sample count + total impact score
- Expandable list of samples (click to preview)
- "Export Kit" button → downloads ZIP
- Subtle gradient background matching the collection type

```jsx
import { useState } from "react";
import { AF, MONO, SERIF } from "../theme/fonts";

const ICONS = {
  target: "M12 2a10 10 0 100 20 10 10 0 000-20zm0 4a6 6 0 100 12 6 6 0 000-12zm0 4a2 2 0 100 4 2 2 0 000-4z",
  sparkle: "M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z",
  rhythm: "M3 18h2V8H3zm4 0h2V3H7zm4 0h2v-8h-2zm4 0h2V7h-2zm4 0h2v-5h-2z",
  crown: "M2 20h20l-2-8-4 4-4-6-4 6-4-4z",
};

export function SmartCollection({ collection, onPreview, onExport, theme, isDark }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div style={{
      padding: 14, borderRadius: 10, background: theme.bg,
      border: "1px solid " + theme.borderLight, minWidth: 220, maxWidth: 280,
      flexShrink: 0, cursor: "pointer", transition: "border-color 0.2s",
    }}
    onClick={() => setExpanded(!expanded)}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill={theme.accent}>
          <path d={ICONS[collection.icon] || ICONS.crown} />
        </svg>
        <div>
          <div style={{ fontSize: 12, fontWeight: 700, color: theme.text, fontFamily: AF }}>{collection.name}</div>
          <div style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF }}>{collection.description}</div>
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF }}>{collection.samples.length} samples</span>
        <span style={{ fontSize: 9, color: theme.accent, fontFamily: MONO, fontWeight: 600 }}>
          Impact: {collection.total_impact}
        </span>
      </div>

      {/* Expanded sample list */}
      {expanded && (
        <div style={{ borderTop: "1px solid " + theme.borderLight, paddingTop: 8, marginBottom: 8 }}>
          {collection.samples.map((s, i) => (
            <div key={i} onClick={e => { e.stopPropagation(); onPreview?.(s); }}
              style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "4px 0", fontSize: 10, color: theme.textSec, fontFamily: AF,
                cursor: "pointer", borderBottom: i < collection.samples.length - 1 ? "1px solid " + theme.borderLight : "none",
              }}>
              <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.name}</span>
              <span style={{ fontSize: 8, color: theme.textFaint, fontFamily: MONO, marginLeft: 8 }}>{s.role}</span>
            </div>
          ))}
        </div>
      )}

      {/* Export button */}
      <button onClick={e => { e.stopPropagation(); onExport?.(collection.id); }}
        style={{
          width: "100%", padding: "6px 0", borderRadius: 6, border: "none",
          background: isDark ? theme.text : "#1A1A1A", color: isDark ? "#0D0D12" : "#fff",
          fontSize: 10, fontWeight: 700, fontFamily: AF, cursor: "pointer",
        }}>
        Export Kit
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/renderer/components/SmartCollection.jsx
git commit -m "feat: SmartCollection card component"
```

### Task 5.3: Wire Into App

**Files:**
- Modify: `src/renderer/hooks/useApi.js`
- Modify: `src/renderer/App.jsx`

- [ ] **Step 1: Add collection API methods**

```javascript
const getCollections = async () => {
  const r = await fetch(API + "/collections/generate");
  return r.json();
};

const exportCollection = async (collectionId) => {
  const r = await fetch(API + `/collections/export/${collectionId}`, { method: "POST" });
  const blob = await r.blob();
  // Trigger download
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `RESONATE-${collectionId}.zip`;
  a.click();
  URL.revokeObjectURL(url);
};
```

- [ ] **Step 2: Render collections below the sample list or in a dedicated tab**

After analysis completes, fetch collections. Show them as a horizontal scrollable row of cards above the sample list, or as a "Kits" tab.

- [ ] **Step 3: Commit**

```bash
git add src/renderer/hooks/useApi.js src/renderer/App.jsx
git commit -m "feat: wire smart collections into results screen"
```

---

## Integration & Final Polish

> **Note:** Individual feature tasks mention "Register route in server" — skip those steps. All route registration happens here in Task 6.1 to avoid merge conflicts.

### Task 6.1: Register All New Routes

**Files:**
- Modify: `backend/server.py`

- [ ] **Step 1: Import and register all new routers**

```python
from routes.taste_profile import router as taste_router
from routes.chart_intelligence import router as chart_router
from routes.versions import router as versions_router
from routes.collections import router as collections_router

app.include_router(taste_router)
app.include_router(chart_router)
app.include_router(versions_router)
app.include_router(collections_router)
```

- [ ] **Step 2: Commit**

### Task 6.2: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
cd /Users/krsn/Desktop/RESONATE/.claude/worktrees/competent-goldstine
python3 -m pytest tests/ -v --tb=short
```

Expected: All tests pass.

- [ ] **Step 2: Fix any failures**
- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: five major features — preview, DNA, charts, versions, collections"
```

### Task 6.3: Push and Create PR

- [ ] **Step 1: Push and create PR**

```bash
git push
gh pr create --title "feat: Five major features — complete production intelligence platform" --body "..."
```
