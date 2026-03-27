/**
 * RESONATE — Main Application Shell.
 * All features: similarity search, sessions, ratings, keyboard shortcuts,
 * toasts, virtual scrolling, smart collections, batch analysis, preference learning.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useTheme } from "./theme/ThemeProvider";
import { SERIF, AF, MONO } from "./theme/fonts";
import { useAudioPlayer } from "./hooks/useAudioPlayer";
import { useWaveformData } from "./hooks/useWaveformData";
import { useApi, API } from "./hooks/useApi";
import { mergeV2WithV1, formatNeeds, formatGapAnalysis, NEED_CATEGORY_COLORS, POLICY_LABELS } from "./utils/v2Adapter";
import { Titlebar } from "./components/Titlebar";
import { ResonateOrb } from "./components/ResonateOrb";
import { SkeletonRow } from "./components/SkeletonRow";
import { SampleRow } from "./components/SampleRow";
import { VolumeSlider } from "./components/VolumeSlider";
import { SpectrumViz } from "./components/SpectrumViz";
import { RealWaveform } from "./components/Waveform";
import { useToast } from "./components/Toast";
import { ShortcutOverlay } from "./components/ShortcutOverlay";
import { useBridge } from "./hooks/useBridge";
import { WaveformTooltip } from "./components/WaveformTooltip";
import { Modal } from "./components/Modal";

// ── Logo with background stripped + animated light tracing down the colored strokes ──
function LogoBlend({ size, isDark, animate = true }) {
  const canvasRef = useRef(null);
  const baseDataRef = useRef(null);
  const animRef = useRef(null);

  // Load image and strip background once
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const c = canvasRef.current;
      if (!c) return;
      const dpr = window.devicePixelRatio || 1;
      const rs = Math.round(size * dpr);
      c.width = rs;
      c.height = rs;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, 0, 0, rs, rs);
      const id = ctx.getImageData(0, 0, rs, rs);
      const d = id.data;
      if (isDark) {
        for (let i = 0; i < d.length; i += 4) {
          const mx = Math.max(d[i], d[i+1], d[i+2]);
          if (mx < 35) d[i+3] = 0;
          else if (mx < 65) d[i+3] = Math.floor(d[i+3] * ((mx - 35) / 30));
        }
      } else {
        for (let i = 0; i < d.length; i += 4) {
          const br = d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114;
          const mx = Math.max(d[i], d[i+1], d[i+2]);
          const mn = Math.min(d[i], d[i+1], d[i+2]);
          const sat = mx > 0 ? (mx - mn) / mx : 0;
          if (br > 225 && sat < 0.12) d[i+3] = 0;
          else if (br > 200 && sat < 0.15) d[i+3] = Math.floor(d[i+3] * Math.max(0, (225 - br) / 25));
        }
      }
      // Store the clean base image data for animation
      baseDataRef.current = new Uint8ClampedArray(d);
      ctx.putImageData(id, 0, 0);

      // Start trace animation
      if (animate) {
        const startTime = Date.now();
        const drawFrame = () => {
          const t = ((Date.now() - startTime) / 1000) % 3; // 3-second loop
          const norm = t / 3; // 0 → 1
          const base = baseDataRef.current;
          if (!base || !c) return;
          const frame = ctx.createImageData(rs, rs);
          const fd = frame.data;
          // The wave position moves top-to-bottom (y-axis normalized)
          const waveCenter = norm;
          const waveWidth = 0.25; // how wide the bright band is
          for (let y = 0; y < rs; y++) {
            const yNorm = y / rs;
            // Distance from wave center (wrapping)
            let dist = Math.abs(yNorm - waveCenter);
            if (dist > 0.5) dist = 1 - dist; // wrap around
            // Brightness boost: peaks at wave center, fades with distance
            const boost = dist < waveWidth ? Math.cos((dist / waveWidth) * Math.PI * 0.5) : 0;
            for (let x = 0; x < rs; x++) {
              const i = (y * rs + x) * 4;
              const alpha = base[i + 3];
              if (alpha === 0) {
                fd[i] = fd[i+1] = fd[i+2] = fd[i+3] = 0;
                continue;
              }
              // Only boost pixels that are visible (the colored strokes)
              const brighten = 1 + boost * 0.8; // up to 80% brighter
              fd[i]     = Math.min(255, Math.round(base[i] * brighten));
              fd[i + 1] = Math.min(255, Math.round(base[i + 1] * brighten));
              fd[i + 2] = Math.min(255, Math.round(base[i + 2] * brighten));
              fd[i + 3] = alpha;
            }
          }
          ctx.putImageData(frame, 0, 0);
          animRef.current = requestAnimationFrame(drawFrame);
        };
        drawFrame();
      }
    };
    img.src = isDark ? "/logo_dark.PNG" : "/logo_light.PNG";
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [isDark, size, animate]);

  return <canvas ref={canvasRef} style={{ width: size, height: size }} />;
}

const KEYS = ["Any","C","C#","D","D#","E","F","F#","G","G#","A","A#","B","Cm","C#m","Dm","D#m","Em","Fm","F#m","Gm","G#m","Am","A#m","Bm"];
const STAGES = ["Uploading audio...","Detecting tempo & groove...","Analyzing harmonic content...","Mapping frequency spectrum...","Identifying instrumentation...","Consulting AI engine...","Scoring samples..."];
const ROW_HEIGHT = 42;
const BUFFER_ROWS = 8;

function formatTime(s) { if (!s || isNaN(s)) return "0:00"; const m = Math.floor(s / 60), sec = Math.floor(s % 60); return m + ":" + String(sec).padStart(2, "0"); }

export default function App() {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const toast = useToast();
  const api = useApi();

  // ── Core State ──
  const [screen, setScreen] = useState("home");
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("");
  const [activeSample, setActiveSample] = useState(null);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const [category, setCategory] = useState("all");
  const [selectedKey, setSelectedKey] = useState("Any");
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState("matched");
  const [dragOver, setDragOver] = useState(false);
  const [favorites, setFavorites] = useState(new Set());
  const [fileName, setFileName] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [samples, setSamples] = useState([]);
  const [error, setError] = useState(null);
  const [backendOk, setBackendOk] = useState(false);
  const [indexProgress, setIndexProgress] = useState(null);
  const [sampleDir, setSampleDir] = useState("");
  const tmr = useRef(null);
  const audio = useAudioPlayer();
  const bridge = useBridge();

  // ── New Feature State ──
  const [ratings, setRatings] = useState({});          // { sampleId: 1-5 }
  const [similarSamples, setSimilarSamples] = useState(null);  // similarity panel
  const [layerSamples, setLayerSamples] = useState(null);      // layering panel
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [showSessions, setShowSessions] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [sourceFilter, setSourceFilter] = useState("all");      // all | local | splice | loopcloud
  const [moodFilter, setMoodFilter] = useState("all");          // all | dark | warm | bright | aggressive | chill
  const [hoverWaveform, setHoverWaveform] = useState({ path: null, rect: null });
  const [checkedSamples, setCheckedSamples] = useState(new Set());  // batch select
  const searchRef = useRef(null);

  // ── v2 ML Pipeline State ──
  const [mixProfile, setMixProfile] = useState(null);              // v2 MixProfile dict
  const [v2Recommendations, setV2Recommendations] = useState([]);  // raw v2 recs
  const [mixNeeds, setMixNeeds] = useState([]);                    // formatted needs for display
  const [gapAnalysis, setGapAnalysis] = useState(null);            // formatted gap analysis
  const [prevReadiness, setPrevReadiness] = useState(null);        // previous readiness for delta display
  const [reanalyzing, setReanalyzing] = useState(false);           // re-analysis in progress
  const [v2Available, setV2Available] = useState(true);            // v2 pipeline responded?
  const [v2Loading, setV2Loading] = useState(false);               // loading spinner for v2 recs
  const [viewMode, setViewMode] = useState("smart");               // "smart" | "all"

  // ── Virtual Scrolling State ──
  const scrollRef = useRef(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(600);

  const waveformUrl = useMemo(() => activeSample ? API + "/samples/audio/" + encodeURI(activeSample.path) : null, [activeSample]);
  const waveformPeaks = useWaveformData(waveformUrl);

  const iStyle = { padding: "7px 9px", borderRadius: 5, border: "1px solid " + theme.border, background: isDark ? "#1E1E28" : "#fff", color: theme.text, fontSize: 11, outline: "none", fontFamily: "'DM Sans', sans-serif" };
  const lbl = { fontSize: 8, color: theme.textMuted, textTransform: "uppercase", letterSpacing: 2, marginBottom: 5, fontWeight: 600, fontFamily: AF };

  // ── Health check + poll index status ──
  useEffect(() => {
    const check = () => fetch(API + "/health").then(r => r.json()).then(d => {
      setBackendOk(true);
      if (!d.indexed) setIndexProgress({ done: d.indexed, processed: d.index_progress, total: d.index_total });
      else setIndexProgress(null);
    }).catch(() => setBackendOk(false));
    check();
    const iv = setInterval(check, 3000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => { fetch(API + "/settings").then(r => r.json()).then(d => setSampleDir(d.sample_dir)).catch(() => {}); }, [backendOk]);

  const loadSamples = useCallback(async () => { try { const d = await (await fetch(API + "/samples")).json(); setSamples(d.samples || []); } catch {} }, []);
  useEffect(() => { if (backendOk) loadSamples(); }, [backendOk, loadSamples]);

  // ── Auto-refetch samples when DAW BPM changes (real-time bridge matching) ──
  const rescoreTimerRef = useRef(null);
  useEffect(() => {
    if (bridge.rescoreNeeded && backendOk && samples.length > 0) {
      // Debounce 500ms to avoid rapid re-fetches while BPM is being adjusted
      clearTimeout(rescoreTimerRef.current);
      rescoreTimerRef.current = setTimeout(() => {
        console.log(`[Bridge] BPM changed → re-scoring samples (DAW: ${bridge.dawBpm} BPM)`);
        loadSamples().then(() => bridge.clearRescore());
      }, 500);
    }
    return () => clearTimeout(rescoreTimerRef.current);
  }, [bridge.rescoreNeeded, bridge.dawBpm, backendOk, samples.length, loadSamples, bridge.clearRescore]);

  // ── Load sessions on mount ──
  useEffect(() => {
    if (backendOk) {
      fetch(API + "/sessions").then(r => r.json()).then(d => setSessions(d.sessions || [])).catch(() => {});
    }
  }, [backendOk]);

  const runAnalysis = useCallback(async (file) => {
    setScreen("analyzing"); setProgress(0); setError(null); setSimilarSamples(null);
    setMixProfile(null); setV2Recommendations([]); setMixNeeds([]); setGapAnalysis(null); setV2Loading(false);
    let p = 0, si = 0; setStage(STAGES[0]);
    tmr.current = setInterval(() => { p += Math.random() * 1.5 + 0.5; p = Math.min(p, 90); const nsi = Math.min(Math.floor(p / (90 / (STAGES.length - 1))), STAGES.length - 2); if (nsi !== si) { si = nsi; setStage(STAGES[nsi]); } setProgress(Math.round(p)); }, 100);
    try {
      // Try the full v2 pipeline first (analyze + gap + recommend in one call)
      let v2Success = false;
      try {
        const fullResult = await api.analyzeFullV2(file, 30);
        // Extract mix profile
        const mp = fullResult.mix_profile;
        setMixProfile(mp);
        setMixNeeds(formatNeeds(mp));
        // Extract gap analysis
        setGapAnalysis(formatGapAnalysis(fullResult.gap_analysis));
        // Extract recommendations
        if (fullResult.recommendations?.recommendations) {
          setV2Recommendations(fullResult.recommendations.recommendations);
        }
        v2Success = true;
        setV2Available(true);
        // Build v1-compatible analysis result
        const v1Compat = { analysis: { key: mp?.analysis?.key || "", bpm: mp?.analysis?.bpm || 0, genre: fullResult.summary?.blueprint_used || mp?.style?.primary_cluster || "", mood: "", energy_label: "", summary: fullResult.summary?.gap_summary || "", what_track_needs: (mp?.needs || []).map(n => n.description).slice(0, 6), frequency_bands: {}, frequency_gaps: [], detected_instruments: [] } };
        setAnalysisResult(v1Compat);
      } catch {
        // Full endpoint not available — try v2 analyze + separate recommend
        try {
          const v2Result = await api.analyzeTrackV2(file);
          setMixProfile(v2Result);
          setMixNeeds(formatNeeds(v2Result));
          // Try to get gap analysis separately
          api.getGapAnalysisV2().then(g => setGapAnalysis(formatGapAnalysis(g))).catch(() => {});
          v2Success = true;
          setV2Available(true);
          const v1Compat = { analysis: { key: v2Result?.analysis?.key || "", bpm: v2Result?.analysis?.bpm || 0, genre: v2Result?.style?.primary_cluster || "", mood: "", energy_label: "", summary: "", what_track_needs: (v2Result?.needs || []).map(n => n.description).slice(0, 6), frequency_bands: {}, frequency_gaps: [], detected_instruments: [] } };
          setAnalysisResult(v1Compat);
        } catch {
          // v2 not available — fall back to v1
          setV2Available(false);
          const fd = new FormData(); fd.append("file", file);
          const res = await fetch(API + "/analyze", { method: "POST", body: fd });
          if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Failed"); }
          const result = await res.json(); setAnalysisResult(result);
        }
      }
      clearInterval(tmr.current); setStage(STAGES[STAGES.length - 1]); setProgress(100);
      await loadSamples(); await audio.loadTrack();

      // Fetch v2 recommendations in the background if not already loaded
      if (v2Success && v2Recommendations.length === 0) {
        setV2Loading(true);
        api.getRecommendationsV2(30).then(recsResult => {
          if (recsResult?.recommendations) setV2Recommendations(recsResult.recommendations);
          setV2Loading(false);
        }).catch(() => setV2Loading(false));
      }

      setTimeout(() => {
        setScreen("results");
        const readiness = gapAnalysis?.readiness;
        toast.success(readiness != null ? `Analysis complete — ${readiness}/100 readiness` : "Analysis complete");
      }, 500);
    } catch (e) { clearInterval(tmr.current); setError(e.message); setProgress(0); setScreen("home"); toast.error(e.message); }
  }, [loadSamples, audio, toast, api]);

  const handleUpload = useCallback(async () => {
    if (!backendOk) { setError("Backend not running"); toast.error("Backend not running"); return; }
    if (window.electronAPI?.openFile) {
      const fp = await window.electronAPI.openFile(); if (!fp) return;
      setFileName(fp.split("/").pop());
      const r = await window.electronAPI.readFileAsBuffer(fp);
      if (r.error) { setError(r.error); return; }
      const b = atob(r.data), ab = new ArrayBuffer(b.length), ia = new Uint8Array(ab);
      for (let i = 0; i < b.length; i++) ia[i] = b.charCodeAt(i);
      runAnalysis(new File([new Blob([ab])], r.name));
    } else {
      const input = document.createElement("input"); input.type = "file"; input.accept = "audio/*";
      input.onchange = e => { const f = e.target.files[0]; if (f) { setFileName(f.name); runAnalysis(f); } };
      input.click();
    }
  }, [backendOk, runAnalysis, toast]);

  const handleDrop = useCallback(e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) { setFileName(f.name); runAnalysis(f); } }, [runAnalysis]);
  const toggleFav = useCallback((id) => setFavorites(p => { const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n; }), []);
  const toggleCheck = useCallback((id) => setCheckedSamples(p => { const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n; }), []);
  const handleHoverWaveform = useCallback((path, rect) => setHoverWaveform({ path, rect }), []);
  const clearChecked = useCallback(() => setCheckedSamples(new Set()), []);

  // ── Batch Export ──
  const exportChecked = useCallback(async () => {
    if (checkedSamples.size === 0) { toast.error("No samples selected"); return; }
    try {
      const paths = [...checkedSamples].map(id => {
        const s = samples.find(x => x.id === id);
        return s?.path || id;
      });
      const r = await fetch(API + "/samples/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sample_paths: paths }) });
      if (!r.ok) throw new Error("Export failed");
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = `resonate-kit-${checkedSamples.size}-samples.zip`; a.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported ${checkedSamples.size} samples`);
      clearChecked();
    } catch { toast.error("Export failed"); }
  }, [checkedSamples, samples, toast, clearChecked]);

  // ── Auto-Layering ──
  const findLayers = useCallback(async (sample) => {
    if (!sample) return;
    try {
      const r = await fetch(API + "/samples/layers/" + encodeURIComponent(sample.id));
      const d = await r.json();
      setLayerSamples(d.layers || []);
      toast.info(`Found ${(d.layers || []).length} layering suggestions`);
    } catch { toast.error("Layering search failed"); }
  }, [toast]);

  // ── v2 Feedback helper (fire-and-forget) ──
  const logV2Feedback = useCallback((sample, action, extras = {}) => {
    if (!v2Available || !sample) return;
    api.logFeedbackV2({
      sample_filepath: sample.id || sample.filepath || "",
      action,
      mix_filepath: mixProfile?.filepath || "",
      session_id: currentSessionId || "",
      recommendation_rank: sample.v2_rank ?? 0,
      ...extras,
    });
  }, [v2Available, api, mixProfile, currentSessionId]);

  // ── Rating ──
  const rateSample = useCallback((sampleId, rating) => {
    setRatings(prev => ({ ...prev, [sampleId]: rating }));
    fetch(API + "/ratings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sample_filepath: sampleId, rating, session_id: currentSessionId }) }).catch(() => {});
    // Also log to v2 feedback
    logV2Feedback({ id: sampleId }, "rate", { rating });
    toast.info(`Rated ${rating} star${rating > 1 ? "s" : ""}`);
  }, [currentSessionId, toast, logV2Feedback]);

  // ── Similarity Search ──
  const findSimilar = useCallback(async (sample) => {
    if (!sample) return;
    try {
      const r = await fetch(API + "/samples/similar/" + encodeURIComponent(sample.id));
      const d = await r.json();
      setSimilarSamples(d.similar || []);
      toast.info(`Found ${(d.similar || []).length} similar samples`);
    } catch { toast.error("Similarity search failed"); }
  }, [toast]);

  // ── Session Save/Load ──
  const saveSession = useCallback(async () => {
    try {
      const r = await fetch(API + "/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ track_filename: fileName, name: fileName }) });
      const d = await r.json();
      setCurrentSessionId(d.id);
      const sessR = await fetch(API + "/sessions"); const sessD = await sessR.json(); setSessions(sessD.sessions || []);
      toast.success("Session saved");
    } catch { toast.error("Failed to save session"); }
  }, [fileName, toast]);

  const loadSession = useCallback(async (sessionId) => {
    try {
      const r = await fetch(API + "/sessions/" + sessionId);
      const d = await r.json();
      setAnalysisResult({ analysis: d.ai_analysis || d.track_profile });
      setFileName(d.track_filename);
      setCurrentSessionId(d.id);
      setShowSessions(false);
      await loadSamples();
      await audio.loadTrack();
      setScreen("results");
      toast.success("Session loaded: " + d.name);
    } catch { toast.error("Failed to load session"); }
  }, [loadSamples, audio, toast]);

  const deleteSession = useCallback(async (sessionId) => {
    try {
      await fetch(API + "/sessions/" + sessionId, { method: "DELETE" });
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      toast.info("Session deleted");
    } catch { toast.error("Failed to delete session"); }
  }, [toast]);

  const a = analysisResult?.analysis || {};

  // Merge v2 recommendations with v1 library when in "smart" mode
  const displaySamples = useMemo(() => {
    if (viewMode === "smart" && v2Recommendations.length > 0) {
      return mergeV2WithV1(v2Recommendations, samples);
    }
    return samples;
  }, [viewMode, v2Recommendations, samples]);

  const filtered = useMemo(() => displaySamples.filter(s => {
    if (category !== "all" && s.category.toLowerCase() !== category.toLowerCase()) return false;
    if (selectedKey !== "Any" && s.key !== "—" && s.key !== selectedKey) return false;
    if (search && !(s.clean_name || s.name).toLowerCase().includes(search.toLowerCase())) return false;
    if (tab === "favorites" && !favorites.has(s.id)) return false;
    if (sourceFilter !== "all" && (s.source || "local") !== sourceFilter) return false;
    if (moodFilter !== "all" && (s.mood || "neutral") !== moodFilter) return false;
    return true;
  }).sort((a, b) => (b.match || 0) - (a.match || 0)), [displaySamples, category, selectedKey, search, tab, favorites, sourceFilter, moodFilter]);

  // Auto-sync to track key/BPM whenever we have an analysis (like Splice).
  // Also syncs when DAW bridge is connected with dawSync enabled.
  const isSynced = !!analysisResult || (bridge.connected && bridge.dawSync);
  const handlePlay = s => { setActiveSample(s); audio.toggle(s.path, s.id, isSynced); const idx = filtered.findIndex(x => x.id === s.id); if (idx >= 0) setSelectedIdx(idx); logV2Feedback(s, "audition"); };
  const selectAllVisible = useCallback(() => { setCheckedSamples(new Set(filtered.map(s => s.id))); }, [filtered]);

  // Count samples by source
  const sourceCounts = useMemo(() => {
    const c = { all: displaySamples.length, local: 0, splice: 0, loopcloud: 0 };
    for (const s of displaySamples) { const src = s.source || "local"; c[src] = (c[src] || 0) + 1; }
    return c;
  }, [displaySamples]);
  const cats = useMemo(() => ["all", ...new Set(displaySamples.filter(s => s.category).map(s => s.category.toLowerCase()))], [displaySamples]);

  // ── Virtual Scrolling ──
  const totalHeight = filtered.length * ROW_HEIGHT;
  const startIdx = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER_ROWS);
  const endIdx = Math.min(filtered.length, Math.ceil((scrollTop + containerHeight) / ROW_HEIGHT) + BUFFER_ROWS);
  const visibleSamples = filtered.slice(startIdx, endIdx);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => setContainerHeight(el.clientHeight);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [screen]);

  const handleScroll = useCallback((e) => setScrollTop(e.target.scrollTop), []);

  // ── Enhanced Keyboard Shortcuts ──
  useEffect(() => {
    if (screen !== "results") return;
    const handler = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

      // Close overlays
      if (showShortcuts) { setShowShortcuts(false); return; }
      if (showSessions) { if (e.key === "Escape") setShowSessions(false); return; }
      if (similarSamples) { if (e.key === "Escape") setSimilarSamples(null); return; }
      if (layerSamples) { if (e.key === "Escape") setLayerSamples(null); return; }

      if (e.code === "Space") {
        e.preventDefault();
        if (activeSample) audio.toggle(activeSample.path, activeSample.id, isSynced);
      } else if (e.code === "ArrowDown") {
        e.preventDefault();
        const next = Math.min(selectedIdx + 1, filtered.length - 1);
        if (filtered[next]) { setSelectedIdx(next); setActiveSample(filtered[next]); audio.toggle(filtered[next].path, filtered[next].id, isSynced); }
      } else if (e.code === "ArrowUp") {
        e.preventDefault();
        const prev = Math.max(selectedIdx - 1, 0);
        if (filtered[prev]) { setSelectedIdx(prev); setActiveSample(filtered[prev]); audio.toggle(filtered[prev].path, filtered[prev].id, isSynced); }
      } else if (e.key === "m" || e.key === "M") {
        audio.toggleMix();
      } else if (e.key === "f" || e.key === "F") {
        if (activeSample) toggleFav(activeSample.id);
      } else if (e.key >= "1" && e.key <= "5") {
        if (activeSample) rateSample(activeSample.id, parseInt(e.key));
      } else if (e.key === "s" || e.key === "S") {
        if (activeSample) findSimilar(activeSample);
      } else if (e.key === "r" || e.key === "R") {
        setScreen("home"); setActiveSample(null); audio.stop(); setAnalysisResult(null); setSelectedIdx(-1); setSimilarSamples(null);
      } else if (e.key === "?" || (e.key === "/" && e.shiftKey)) {
        setShowShortcuts(true);
      } else if (e.key === "Tab") {
        e.preventDefault();
        const ci = cats.indexOf(category);
        const ni = (ci + 1) % cats.length;
        setCategory(cats[ni]);
      } else if ((e.metaKey || e.ctrlKey) && e.key === "f") {
        e.preventDefault();
        searchRef.current?.focus();
      } else if ((e.metaKey || e.ctrlKey) && e.key === ",") {
        e.preventDefault();
        setShowSettings(true);
      } else if (e.key === "e" || e.key === "E") {
        saveSession();
      } else if (e.key === "l" || e.key === "L") {
        if (activeSample) findLayers(activeSample);
      } else if (e.key === "Escape") {
        if (checkedSamples.size > 0) clearChecked();
        else { setActiveSample(null); audio.stop(); setSelectedIdx(-1); }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [screen, activeSample, selectedIdx, filtered, audio, toggleFav, rateSample, findSimilar, findLayers, saveSession, showShortcuts, showSessions, similarSamples, layerSamples, cats, category, checkedSamples, clearChecked]);

  // ── Drag-to-DAW ──
  const handleDragStart = useCallback(async (e, sample) => {
    e.dataTransfer.setData("text/plain", sample.clean_name || sample.name);
    try {
      const syncParam = (bridge.connected && bridge.dawSync) ? "?sync=1" : "";
      const r = await fetch(API + "/samples/abspath/" + encodeURI(sample.path) + syncParam);
      const d = await r.json();
      if (d.path) {
        e.dataTransfer.setData("text/uri-list", "file://" + d.path);
        // Native Electron drag for DAW drop support
        if (window.electronAPI?.startDrag) window.electronAPI.startDrag(d.path);
      }
    } catch {}
    fetch(API + "/usage", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sample_filepath: sample.id, action: "drag", session_id: currentSessionId }) }).catch(() => {});
    logV2Feedback(sample, "drag");
  }, [currentSessionId, bridge.connected, bridge.dawSync, logV2Feedback]);

  const handleNew = () => { setScreen("home"); setActiveSample(null); audio.stop(); setAnalysisResult(null); setSelectedIdx(-1); setSimilarSamples(null); setMixProfile(null); setV2Recommendations([]); setMixNeeds([]); setGapAnalysis(null); setViewMode("smart"); };

  const analyzeFromBridge = useCallback(async () => {
    try {
      setScreen("analyzing"); setProgress(0); setStage(STAGES[0]);
      let p = 0, si = 0;
      tmr.current = setInterval(() => { p += Math.random() * 1.5 + 0.5; p = Math.min(p, 90); const nsi = Math.min(Math.floor(p / (90 / (STAGES.length - 1))), STAGES.length - 2); if (nsi !== si) { si = nsi; setStage(STAGES[nsi]); } setProgress(Math.round(p)); }, 100);
      const r = await fetch(API + "/analyze/bridge", { method: "POST" });
      clearInterval(tmr.current);
      if (r.ok) {
        const d = await r.json();
        setAnalysisResult(d); setStage(STAGES[STAGES.length - 1]); setProgress(100);
        setFileName(d.filename || "DAW Master");
        await loadSamples(); await audio.loadTrack();
        setTimeout(() => { setScreen("results"); toast.success("Bridge analysis complete"); }, 500);
      } else { setScreen("home"); toast.error("Bridge analysis failed — play some audio first"); }
    } catch { setScreen("home"); toast.error("Bridge analysis failed"); }
  }, [loadSamples, audio, toast]);

  // ── Re-Analyze from Results Screen (Bridge or File) ──
  const reAnalyze = useCallback(async () => {
    if (reanalyzing) return;
    setReanalyzing(true);

    // Save current readiness for delta comparison
    if (gapAnalysis?.readiness != null) {
      setPrevReadiness(gapAnalysis.readiness);
    }

    try {
      if (bridge.connected) {
        // Re-analyze from DAW bridge
        const r = await fetch(API + "/analyze/bridge", { method: "POST" });
        if (!r.ok) { toast.error("Bridge capture failed — play some audio first"); setReanalyzing(false); return; }
        const d = await r.json();
        setAnalysisResult(d);
        setFileName(d.filename || "DAW Master");
      }

      // Run full v2 pipeline on whatever we have
      if (v2Available) {
        try {
          // Use the gap endpoint to get fresh analysis (the /analyze/v2 endpoint was already called by bridge)
          const gapResult = await api.getGapAnalysisV2();
          const newGap = formatGapAnalysis(gapResult);
          setGapAnalysis(newGap);

          // Get fresh needs
          const needsResult = await api.getNeedsV2();
          setMixNeeds(formatNeeds(needsResult));

          // Get fresh recommendations
          const recsResult = await api.getRecommendationsV2(30);
          if (recsResult?.recommendations) setV2Recommendations(recsResult.recommendations);

          await loadSamples();

          // Show delta
          if (prevReadiness != null && newGap?.readiness != null) {
            const delta = newGap.readiness - prevReadiness;
            if (delta > 0) toast.success(`Re-analyzed! Readiness: ${prevReadiness} → ${newGap.readiness} (+${delta})`);
            else if (delta < 0) toast.info(`Re-analyzed. Readiness: ${prevReadiness} → ${newGap.readiness} (${delta})`);
            else toast.success(`Re-analyzed. Readiness: ${newGap.readiness}/100`);
          } else {
            toast.success("Re-analysis complete");
          }
        } catch {
          toast.info("Re-analyzed (basic mode)");
        }
      }
    } catch (e) {
      toast.error("Re-analysis failed: " + (e.message || "unknown error"));
    }

    setReanalyzing(false);
  }, [reanalyzing, bridge.connected, v2Available, gapAnalysis, prevReadiness, api, loadSamples, toast]);

  // ── Star Rating Display ──
  const StarRating = ({ sampleId }) => {
    const r = ratings[sampleId] || 0;
    return (
      <div style={{ display: "flex", gap: 1 }}>
        {[1, 2, 3, 4, 5].map(n => (
          <span key={n} onClick={(e) => { e.stopPropagation(); rateSample(sampleId, n); }} style={{ cursor: "pointer", fontSize: 10, color: n <= r ? "#D97706" : theme.textFaint }}>{n <= r ? "★" : "☆"}</span>
        ))}
      </div>
    );
  };

  return (
    <div style={{ width: "100%", height: "100vh", background: theme.bg, color: theme.text, fontFamily: "'DM Sans', sans-serif", overflow: "hidden" }}>
      <Titlebar backendOk={backendOk} indexProgress={indexProgress} screen={screen} onNew={handleNew} />
      {showShortcuts && <ShortcutOverlay onClose={() => setShowShortcuts(false)} />}
      <WaveformTooltip samplePath={hoverWaveform.path} visible={!!hoverWaveform.path} anchorRect={hoverWaveform.rect} />

      {/* SESSION HISTORY MODAL */}
      <Modal visible={showSessions} onClose={() => setShowSessions(false)}>
        <div style={{ fontSize: 14, fontWeight: 600, color: theme.text, fontFamily: AF, marginBottom: 16 }}>Session History</div>
        <div style={{ flex: 1, overflowY: "auto" }}>
          {sessions.length === 0 ? (
            <div style={{ padding: 20, textAlign: "center", color: theme.textMuted, fontSize: 11 }}>No saved sessions</div>
          ) : sessions.map(s => (
            <div key={s.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", borderRadius: 6, marginBottom: 4, background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)", border: "1px solid " + theme.borderLight }}>
              <div>
                <div style={{ fontSize: 11, color: theme.text, fontFamily: AF, fontWeight: 500 }}>{s.name || s.track_filename}</div>
                <div style={{ fontSize: 9, color: theme.textMuted, fontFamily: MONO }}>{new Date(s.created_at * 1000).toLocaleDateString()}</div>
              </div>
              <div style={{ display: "flex", gap: 4 }}>
                <button onClick={() => loadSession(s.id)} style={{ fontSize: 9, padding: "3px 10px", borderRadius: 4, border: "none", background: isDark ? theme.text : "#1A1A1A", color: isDark ? "#0D0D12" : "#fff", cursor: "pointer", fontWeight: 600 }}>Load</button>
                <button onClick={() => deleteSession(s.id)} style={{ fontSize: 9, padding: "3px 8px", borderRadius: 4, border: "1px solid " + theme.border, background: "transparent", color: theme.textMuted, cursor: "pointer" }}>×</button>
              </div>
            </div>
          ))}
        </div>
      </Modal>

      {/* SIMILARITY PANEL */}
      <Modal visible={!!similarSamples} onClose={() => setSimilarSamples(null)} width={520}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: theme.text, fontFamily: AF }}>Similar Samples</div>
          <button onClick={() => setSimilarSamples(null)} style={{ background: "none", border: "none", color: theme.textMuted, cursor: "pointer", fontSize: 16 }}>×</button>
        </div>
        <div style={{ flex: 1, overflowY: "auto" }}>
          {(similarSamples || []).length === 0 ? (
            <div style={{ padding: 20, textAlign: "center", color: theme.textMuted, fontSize: 11 }}>No similar samples found</div>
          ) : (similarSamples || []).map(s => (
            <div key={s.id} onClick={() => { setActiveSample(s); audio.toggle(s.path, s.id, isSynced); }} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", borderRadius: 6, marginBottom: 4, cursor: "pointer", background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)", border: "1px solid " + theme.borderLight }}>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 12, color: theme.text, fontFamily: SERIF, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.clean_name || s.name}</div>
                <div style={{ fontSize: 9, color: theme.textMuted }}>{s.type_label} · {s.key} · {s.bpm ? Math.round(s.bpm) : "—"}</div>
              </div>
              <span style={{ fontSize: 11, fontWeight: 700, color: theme.text, fontFamily: MONO, marginLeft: 8 }}>{s.similarity}%</span>
            </div>
          ))}
        </div>
      </Modal>

      {/* LAYERING PANEL */}
      <Modal visible={!!layerSamples} onClose={() => setLayerSamples(null)} width={520}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: theme.text, fontFamily: AF }}>Layer Suggestions</div>
          <button onClick={() => setLayerSamples(null)} style={{ background: "none", border: "none", color: theme.textMuted, cursor: "pointer", fontSize: 16 }}>×</button>
        </div>
        <div style={{ flex: 1, overflowY: "auto" }}>
          {(layerSamples || []).length === 0 ? (
            <div style={{ padding: 20, textAlign: "center", color: theme.textMuted, fontSize: 11 }}>No layering suggestions found</div>
          ) : (layerSamples || []).map(s => (
            <div key={s.id} onClick={() => { setActiveSample(s); audio.toggle(s.path, s.id, isSynced); }} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", borderRadius: 6, marginBottom: 4, cursor: "pointer", background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)", border: "1px solid " + theme.borderLight }}>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 12, color: theme.text, fontFamily: SERIF, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.clean_name || s.name}</div>
                <div style={{ fontSize: 9, color: theme.textMuted }}>{s.type_label} · {s.key} · {s.layer_reason}</div>
              </div>
              <span style={{ fontSize: 11, fontWeight: 700, color: theme.text, fontFamily: MONO, marginLeft: 8 }}>{Math.round(s.layer_score)}%</span>
            </div>
          ))}
        </div>
      </Modal>

      {/* HOME */}
      {screen === "home" && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "calc(100vh - 46px)", padding: 40, textAlign: "center", position: "relative", overflow: "hidden" }}>
          {/* Ambient background layers */}
          <div style={{ position: "absolute", inset: 0, background: isDark
            ? "radial-gradient(ellipse 80% 60% at 50% 30%, rgba(217,70,239,0.06) 0%, transparent 70%)"
            : "radial-gradient(ellipse 80% 60% at 50% 30%, rgba(217,70,239,0.05) 0%, transparent 70%)", pointerEvents: "none" }} />
          <div style={{ position: "absolute", inset: 0, background: isDark
            ? "radial-gradient(ellipse 50% 40% at 70% 80%, rgba(6,182,212,0.05) 0%, transparent 60%)"
            : "radial-gradient(ellipse 50% 40% at 70% 80%, rgba(6,182,212,0.04) 0%, transparent 60%)", pointerEvents: "none" }} />
          {/* Logo + Title */}
          <div style={{ marginBottom: 40, animation: "fadeInUp 0.5s ease", position: "relative", zIndex: 1 }}>
            <div style={{ position: "relative", width: 88, height: 88, margin: "0 auto 28px" }}>
              <LogoBlend size={88} isDark={isDark} />
            </div>
            <h1 className="gradient-text shimmer-text" style={{ fontSize: 30, fontWeight: 300, letterSpacing: 12, fontFamily: AF, margin: "0 0 8px" }}>RESONATE</h1>
            <p style={{ fontSize: 10, color: theme.textMuted, letterSpacing: 5, textTransform: "uppercase", fontFamily: AF }}>Smart Sampling</p>
          </div>

          {error && <div style={{ padding: "8px 14px", borderRadius: 6, background: isDark ? "rgba(220,38,38,0.12)" : "rgba(220,38,38,0.06)", border: "1px solid rgba(220,38,38,0.12)", color: theme.red, fontSize: 11, marginBottom: 16, maxWidth: 440, position: "relative", zIndex: 1 }}>{error}</div>}

          {/* Upload Card */}
          <div className="upload-card" onDragOver={e => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={handleDrop} onClick={handleUpload}
            style={{ width: "100%", maxWidth: 440, padding: "44px 32px", borderRadius: 16, cursor: "pointer",
              border: "1.5px solid transparent",
              background: isDark
                ? (dragOver ? "rgba(217,70,239,0.06)" : "linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.01) 100%)")
                : (dragOver ? "rgba(217,70,239,0.04)" : "linear-gradient(180deg, rgba(255,255,255,0.85) 0%, rgba(255,255,255,0.6) 100%)"),
              transition: "all 0.3s ease",
              boxShadow: isDark
                ? "0 2px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03)"
                : "0 2px 16px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.8)",
              backdropFilter: "blur(16px)",
              position: "relative", zIndex: 1 }}>
            <svg width="28" height="28" viewBox="0 0 44 44" fill="none" style={{ margin: "0 auto 14px", display: "block", opacity: 0.5 }}>
              <path d="M22 8v22M15 15l7-7 7 7" stroke={dragOver ? theme.text : theme.textMuted} strokeWidth="1.5" strokeLinecap="round" />
              <path d="M8 32v4a2 2 0 002 2h24a2 2 0 002-2v-4" stroke={dragOver ? theme.text : theme.textMuted} strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <div style={{ fontSize: 15, fontWeight: 400, color: theme.text, marginBottom: 5, fontFamily: SERIF }}>Drop your mixdown here</div>
            <div style={{ fontSize: 11, color: theme.textMuted }}>or click to browse</div>
            <div style={{ display: "flex", gap: 6, justifyContent: "center", marginTop: 16 }}>
              {["WAV", "MP3", "FLAC", "AIFF"].map(f => <span key={f} style={{ padding: "3px 10px", borderRadius: 4, fontSize: 8, fontWeight: 600, background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)", color: theme.tagText, letterSpacing: 1.2 }}>{f}</span>)}
            </div>
          </div>

          {/* Analyze from DAW — shown when bridge is connected */}
          {bridge.connected && (
            <button onClick={analyzeFromBridge} style={{ width: "100%", maxWidth: 440, marginTop: 12, padding: "12px 0", borderRadius: 10, border: "1px solid rgba(34,197,94,0.25)", background: "rgba(34,197,94,0.06)", color: "#22C55E", cursor: "pointer", fontWeight: 600, fontSize: 12, fontFamily: AF, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, position: "relative", zIndex: 1, backdropFilter: "blur(12px)" }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", animation: "pulse 2s ease infinite" }} />
              Analyze from DAW ({bridge.dawBpm.toFixed(0)} BPM)
            </button>
          )}

          {/* Thin accent divider */}
          <div style={{ width: 40, height: 1, background: "linear-gradient(90deg, #D946EF, #06B6D4)", opacity: 0.3, margin: "32px 0", position: "relative", zIndex: 1, borderRadius: 1 }} />

          {/* Stats */}
          <div style={{ display: "flex", gap: 44, position: "relative", zIndex: 1 }}>
            {[{ v: samples.length || "—", l: "Samples" }, { v: cats.length - 1 || "—", l: "Categories" }, { v: "AI", l: "Analysis" }].map((s, i) => (
              <div key={s.l} style={{ textAlign: "center", animation: `fadeInUp 0.5s ease ${0.1 + i * 0.08}s both` }}>
                <div style={{ fontSize: 22, fontWeight: 200, color: theme.text, fontFamily: SERIF }}>{s.v}</div>
                <div style={{ fontSize: 8, color: theme.textMuted, textTransform: "uppercase", letterSpacing: 2.5, marginTop: 4, fontFamily: AF, fontWeight: 500 }}>{s.l}</div>
              </div>
            ))}
          </div>

          {/* Session History + Shortcut Hints */}
          <div style={{ marginTop: 28, display: "flex", gap: 12, position: "relative", zIndex: 1 }}>
            {sessions.length > 0 && (
              <button onClick={() => setShowSessions(true)} style={{ fontSize: 10, padding: "6px 16px", borderRadius: 8, border: "1px solid " + theme.border, background: isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.6)", color: theme.textSec, cursor: "pointer", fontFamily: AF, backdropFilter: "blur(8px)", transition: "all 0.2s" }}>Load Session ({sessions.length})</button>
            )}
          </div>
          <div style={{ marginTop: 16, fontSize: 9, color: theme.textFaint, fontFamily: AF, position: "relative", zIndex: 1 }}>
            Press <span style={{ fontFamily: MONO, padding: "1px 5px", borderRadius: 3, background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)", fontSize: 10 }}>?</span> for keyboard shortcuts
          </div>
        </div>
      )}

      {/* ANALYZING */}
      {screen === "analyzing" && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "calc(100vh - 46px)", padding: 40, position: "relative", overflow: "hidden" }}>
          {/* Orb with shimmering R logo centered inside */}
          <div style={{ position: "relative", width: 320, height: 320 }}>
            <ResonateOrb progress={progress} size={320} />
            <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", width: 110, height: 110, zIndex: 2 }}>
              <LogoBlend size={110} isDark={isDark} />
            </div>
          </div>
          <div style={{ marginTop: 36, textAlign: "center", position: "relative", zIndex: 1 }}>
            <div style={{ fontSize: 32, fontWeight: 200, color: theme.text, fontFamily: SERIF, letterSpacing: -1, marginBottom: 12 }}>
              {progress}<span style={{ fontSize: 16, opacity: 0.4 }}>%</span>
            </div>
            <div className="gradient-text" style={{ fontSize: 11, fontWeight: 600, marginBottom: 6, fontFamily: AF, letterSpacing: 1.5, textTransform: "uppercase" }}>{stage}</div>
            <div style={{ fontSize: 9, color: theme.textMuted, fontFamily: MONO, opacity: 0.6 }}>{fileName}</div>
          </div>
        </div>
      )}

      {/* RESULTS */}
      {screen === "results" && (
        <div style={{ display: "flex", height: "calc(100vh - 46px)", fontFamily: SERIF }}>
          {/* Sidebar */}
          <div style={{ width: 250, borderRight: "1px solid " + theme.border, padding: 12, overflowY: "auto", flexShrink: 0, background: theme.surface }}>
            <div style={{ padding: 12, borderRadius: 8, marginBottom: 12, background: theme.bg, border: "1px solid " + theme.borderLight }}>
              <div style={lbl}>Analysis</div>
              <div style={{ fontSize: 9, color: theme.textMuted, marginBottom: 6, fontFamily: MONO, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{fileName}</div>
              {[["Key", a.key], ["BPM", a.bpm ? Math.round(a.bpm) : "—"], ["Genre", a.genre], ["Energy", a.energy_label], ["Mood", a.mood]].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 10, color: theme.textMuted, fontFamily: AF }}>{k}</span>
                  <span style={{ fontSize: 10, color: theme.text, fontWeight: 600, fontFamily: MONO }}>{v || "—"}</span>
                </div>
              ))}
              {/* Session + Re-Analyze actions */}
              <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
                <button onClick={saveSession} style={{ flex: 1, fontSize: 8, padding: "4px 0", borderRadius: 4, border: "1px solid " + theme.border, background: "transparent", color: theme.textSec, cursor: "pointer", fontFamily: AF }}>Save Session</button>
                {sessions.length > 0 && <button onClick={() => setShowSessions(true)} style={{ flex: 1, fontSize: 8, padding: "4px 0", borderRadius: 4, border: "1px solid " + theme.border, background: "transparent", color: theme.textSec, cursor: "pointer", fontFamily: AF }}>History</button>}
              </div>
              {bridge.connected && (
                <button onClick={reAnalyze} disabled={reanalyzing} style={{ width: "100%", fontSize: 9, padding: "6px 0", marginTop: 6, borderRadius: 5, border: "1px solid rgba(34,197,94,0.3)", background: reanalyzing ? "rgba(34,197,94,0.03)" : "rgba(34,197,94,0.08)", color: reanalyzing ? theme.textMuted : "#22C55E", cursor: reanalyzing ? "default" : "pointer", fontWeight: 600, fontFamily: AF, transition: "all 0.2s" }}>
                  {reanalyzing ? "Re-Analyzing..." : "Re-Analyze from DAW"}
                </button>
              )}
            </div>
            {/* v2 Mix Needs */}
            {mixNeeds.length > 0 && (
              <div style={{ padding: 12, borderRadius: 8, marginBottom: 12, background: theme.bg, border: "1px solid " + theme.borderLight }}>
                <div style={lbl}>Your Mix Needs</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {mixNeeds.slice(0, 8).map((need, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
                      <span style={{ width: 5, height: 5, borderRadius: "50%", background: NEED_CATEGORY_COLORS[need.category] || theme.textMuted, flexShrink: 0, marginTop: 4, opacity: 0.4 + need.severity * 0.6 }} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 10, color: theme.text, fontFamily: AF, lineHeight: 1.3 }}>{need.description}</div>
                        {need.policy && <div style={{ fontSize: 8, color: theme.textFaint, fontFamily: AF, marginTop: 1 }}>{POLICY_LABELS[need.policy] || need.policy}</div>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Gap Analysis Panel */}
            {gapAnalysis && (
              <div style={{ padding: 12, borderRadius: 8, marginBottom: 12, background: theme.bg, border: "1px solid " + theme.borderLight }}>
                <div style={lbl}>Production Readiness</div>
                {/* Score ring */}
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <div style={{ position: "relative", width: 44, height: 44, flexShrink: 0 }}>
                    <svg width="44" height="44" viewBox="0 0 44 44">
                      <circle cx="22" cy="22" r="18" fill="none" stroke={isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"} strokeWidth="3" />
                      <circle cx="22" cy="22" r="18" fill="none" stroke={gapAnalysis.readinessColor} strokeWidth="3" strokeDasharray={`${gapAnalysis.readiness * 1.131} 113.1`} strokeLinecap="round" transform="rotate(-90 22 22)" style={{ transition: "stroke-dasharray 0.6s ease" }} />
                    </svg>
                    <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 700, fontFamily: MONO, color: gapAnalysis.readinessColor }}>{gapAnalysis.readiness}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 600, color: gapAnalysis.readinessColor, fontFamily: AF }}>{gapAnalysis.readinessTier}</div>
                    <div style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF }}>{gapAnalysis.genre}</div>
                  </div>
                </div>
                {/* Chart potential bar */}
                <div style={{ marginBottom: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF }}>Chart Potential</span>
                    <span style={{ fontSize: 9, color: theme.text, fontFamily: MONO, fontWeight: 600 }}>{gapAnalysis.chartPotentialCurrent}<span style={{ color: theme.textFaint }}> / {gapAnalysis.chartPotentialCeiling}</span></span>
                  </div>
                  <div style={{ height: 4, borderRadius: 2, background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)", overflow: "hidden" }}>
                    <div style={{ height: "100%", borderRadius: 2, background: "linear-gradient(90deg, #D946EF, #06B6D4)", width: `${gapAnalysis.chartPotentialCeiling}%`, position: "relative" }}>
                      <div style={{ position: "absolute", left: `${(gapAnalysis.chartPotentialCurrent / Math.max(gapAnalysis.chartPotentialCeiling, 1)) * 100}%`, top: -2, width: 2, height: 8, background: "#fff", borderRadius: 1, boxShadow: "0 0 4px rgba(0,0,0,0.3)" }} />
                    </div>
                  </div>
                </div>
                {/* Missing roles */}
                {gapAnalysis.missingRoles.length > 0 && (
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 8, color: theme.textFaint, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 4, fontFamily: AF }}>Missing</div>
                    <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
                      {gapAnalysis.missingRoles.map(r => (
                        <span key={r} style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, background: "rgba(239,68,68,0.1)", color: "#EF4444", fontFamily: AF, fontWeight: 600 }}>{r}</span>
                      ))}
                    </div>
                  </div>
                )}
                {/* Top gaps */}
                {gapAnalysis.gaps.length > 0 && (
                  <div>
                    <div style={{ fontSize: 8, color: theme.textFaint, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 4, fontFamily: AF }}>Top Issues</div>
                    {gapAnalysis.gaps.slice(0, 5).map((g, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 6, marginBottom: 3 }}>
                        <span style={{ width: 5, height: 5, borderRadius: "50%", background: g.severityColor, flexShrink: 0, marginTop: 4 }} />
                        <div style={{ fontSize: 9, color: theme.textSec, fontFamily: AF, lineHeight: 1.3 }}>{g.message}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Bridge Status */}
            {bridge.connected && (
              <div style={{ padding: 10, borderRadius: 8, marginBottom: 12, background: isDark ? "rgba(34,197,94,0.06)" : "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.15)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", animation: "pulse 2s ease infinite" }} />
                  <span style={{ fontSize: 9, fontWeight: 600, color: "#22C55E", textTransform: "uppercase", letterSpacing: 1.5, fontFamily: AF }}>DAW Connected</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 10, color: theme.textMuted, fontFamily: AF }}>BPM</span>
                  <span style={{ fontSize: 14, color: theme.text, fontWeight: 700, fontFamily: MONO }}>{bridge.dawBpm.toFixed(1)}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 10, color: theme.textMuted, fontFamily: AF }}>Time Sig</span>
                  <span style={{ fontSize: 10, color: theme.text, fontWeight: 600, fontFamily: MONO }}>{bridge.dawTimeSig}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 10, color: theme.textMuted, fontFamily: AF }}>Transport</span>
                  <span style={{ fontSize: 10, color: bridge.dawPlaying ? "#22C55E" : theme.textMuted, fontFamily: MONO }}>{bridge.dawPlaying ? "Playing" : "Stopped"}</span>
                </div>
                {/* Sync Toggle */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", marginBottom: 6, borderTop: "1px solid rgba(34,197,94,0.1)" }}>
                  <span style={{ fontSize: 10, color: theme.textMuted, fontFamily: AF }}>Sync Key & BPM</span>
                  <button onClick={() => bridge.setDawSync(!bridge.dawSync)}
                    style={{ width: 32, height: 16, borderRadius: 8, border: "none", background: bridge.dawSync ? "#22C55E" : theme.borderLight, position: "relative", cursor: "pointer", transition: "background 0.2s" }}>
                    <div style={{ width: 12, height: 12, borderRadius: 6, background: "#fff", position: "absolute", top: 2, left: bridge.dawSync ? 18 : 2, transition: "left 0.2s" }} />
                  </button>
                </div>
                {bridge.dawSync && (
                  <div style={{ fontSize: 8, color: "#22C55E", fontFamily: AF, opacity: 0.8, marginBottom: 6 }}>
                    Playback & drag transposed to DAW key/BPM
                  </div>
                )}
                {/* Analyze from DAW */}
                {!analysisResult && (
                  <button onClick={analyzeFromBridge} style={{ width: "100%", fontSize: 9, padding: "6px 0", borderRadius: 5, border: "1px solid rgba(34,197,94,0.3)", background: "rgba(34,197,94,0.1)", color: "#22C55E", cursor: "pointer", fontWeight: 700, fontFamily: AF, marginBottom: 6 }}>
                    Analyze from DAW
                  </button>
                )}
                {/* Key Browser */}
                <div style={lbl}>Browse Key</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                  {bridge.allKeys.map(k => (
                    <button key={k} onClick={() => bridge.setBrowseKey(bridge.browseKey === k ? null : k)}
                      style={{ fontSize: 8, padding: "2px 5px", borderRadius: 3, border: "1px solid " + (bridge.browseKey === k ? "#22C55E" : theme.borderLight), background: bridge.browseKey === k ? "rgba(34,197,94,0.15)" : "transparent", color: bridge.browseKey === k ? "#22C55E" : theme.textMuted, cursor: "pointer", fontFamily: MONO, fontWeight: bridge.browseKey === k ? 700 : 400 }}>{k}</button>
                  ))}
                </div>
                {bridge.browseKey && (
                  <button onClick={() => bridge.sendKeyChange(bridge.browseKey)} style={{ marginTop: 6, width: "100%", fontSize: 9, padding: "5px 0", borderRadius: 5, border: "none", background: "#22C55E", color: "#0D0D12", cursor: "pointer", fontWeight: 700, fontFamily: AF }}>
                    Send {bridge.browseKey} to DAW
                  </button>
                )}
              </div>
            )}

            {a.frequency_bands && (
              <div style={{ padding: 10, borderRadius: 8, marginBottom: 12, background: theme.bg, border: "1px solid " + theme.borderLight }}>
                <div style={lbl}>Spectrum</div>
                <SpectrumViz trackBands={a.frequency_bands} sampleBands={activeSample?.frequency_bands} gaps={a.frequency_gaps} height={70} />
              </div>
            )}
            <div style={{ position: "relative", marginBottom: 12 }}>
              <input ref={searchRef} placeholder="Search... (Cmd+F)" value={search} onChange={e => setSearch(e.target.value)} style={{ ...iStyle, paddingLeft: 26, width: "100%", boxSizing: "border-box" }} />
              <svg width="10" height="10" viewBox="0 0 14 14" style={{ position: "absolute", left: 8, top: 11, opacity: 0.3 }}><circle cx="6" cy="6" r="5" fill="none" stroke="currentColor" strokeWidth="1.5" /><line x1="10" y1="10" x2="13" y2="13" stroke="currentColor" strokeWidth="1.5" /></svg>
            </div>
            <div style={{ marginBottom: 12 }}>
              <div style={lbl}>Category</div>
              {cats.map(cat => {
                const cnt = cat === "all" ? samples.length : samples.filter(s => s.category.toLowerCase() === cat).length;
                return <button key={cat} onClick={() => setCategory(cat)} style={{ display: "flex", justifyContent: "space-between", width: "100%", textAlign: "left", padding: "5px 7px", marginBottom: 1, borderRadius: 4, border: "none", background: category === cat ? (isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)") : "transparent", color: category === cat ? theme.text : theme.textSec, fontSize: 11, fontWeight: category === cat ? 600 : 400, cursor: "pointer", textTransform: "capitalize", fontFamily: AF }}><span>{cat === "all" ? "All" : cat}</span><span style={{ fontSize: 9, opacity: 0.5 }}>{cnt}</span></button>;
              })}
            </div>
            <div style={{ marginBottom: 12 }}><div style={lbl}>Key</div><select value={selectedKey} onChange={e => setSelectedKey(e.target.value)} style={{ ...iStyle, width: "100%", boxSizing: "border-box", cursor: "pointer" }}>{KEYS.map(k => <option key={k} value={k}>{k}</option>)}</select></div>

            {/* Source filter */}
            <div>
              <div style={lbl}>Source</div>
              {[
                { id: "all", label: "All Sources", color: null },
                { id: "local", label: "Local", color: null },
                { id: "splice", label: "Splice", color: "#6366F1" },
                { id: "loopcloud", label: "Loopcloud", color: "#EC4899" },
              ].filter(s => s.id === "all" || sourceCounts[s.id] > 0).map(s => (
                <button key={s.id} onClick={() => setSourceFilter(s.id)} style={{ display: "flex", justifyContent: "space-between", width: "100%", textAlign: "left", padding: "5px 7px", marginBottom: 1, borderRadius: 4, border: "none", background: sourceFilter === s.id ? (isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)") : "transparent", color: sourceFilter === s.id ? theme.text : theme.textSec, fontSize: 11, fontWeight: sourceFilter === s.id ? 600 : 400, cursor: "pointer", fontFamily: AF }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    {s.color && <span style={{ width: 5, height: 5, borderRadius: "50%", background: s.color, flexShrink: 0 }} />}
                    {s.label}
                  </span>
                  <span style={{ fontSize: 9, opacity: 0.5 }}>{sourceCounts[s.id] || 0}</span>
                </button>
              ))}
            </div>

            {/* Mood filter */}
            <div style={{ marginTop: 12 }}>
              <div style={lbl}>Mood</div>
              {[
                { id: "all", label: "All Moods", color: null },
                { id: "dark", label: "Dark", color: "#8B5CF6" },
                { id: "warm", label: "Warm", color: "#CA8A04" },
                { id: "bright", label: "Bright", color: "#3B82F6" },
                { id: "aggressive", label: "Aggressive", color: "#EF4444" },
                { id: "chill", label: "Chill", color: "#22C55E" },
              ].map(m => {
                const cnt = m.id === "all" ? samples.length : samples.filter(s => (s.mood || "neutral") === m.id).length;
                if (m.id !== "all" && cnt === 0) return null;
                return (
                  <button key={m.id} onClick={() => setMoodFilter(m.id)} style={{ display: "flex", justifyContent: "space-between", width: "100%", textAlign: "left", padding: "5px 7px", marginBottom: 1, borderRadius: 4, border: "none", background: moodFilter === m.id ? (isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)") : "transparent", color: moodFilter === m.id ? theme.text : theme.textSec, fontSize: 11, fontWeight: moodFilter === m.id ? 600 : 400, cursor: "pointer", fontFamily: AF }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      {m.color && <span style={{ width: 5, height: 5, borderRadius: "50%", background: m.color, flexShrink: 0 }} />}
                      {m.label}
                    </span>
                    <span style={{ fontSize: 9, opacity: 0.5 }}>{cnt}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Main */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", background: theme.bg }}>
            {/* AI Summary Card */}
            {(a.summary || gapAnalysis) && (
              <div style={{ padding: "12px 16px", background: theme.surface, borderBottom: "1px solid " + theme.border }}>
                {/* Gap analysis headline */}
                {gapAnalysis && (
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: (a.summary || a.what_track_needs?.length) ? 8 : 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ fontSize: 22, fontWeight: 200, color: gapAnalysis.readinessColor, fontFamily: SERIF }}>{gapAnalysis.readiness}</span>
                      <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF }}>/100</span>
                    </div>
                    <div style={{ width: 1, height: 24, background: theme.borderLight }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 11, color: theme.text, fontFamily: AF, fontWeight: 500 }}>{gapAnalysis.summary}</div>
                    </div>
                    {gapAnalysis.criticalGaps > 0 && (
                      <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 4, background: "rgba(239,68,68,0.1)", color: "#EF4444", fontFamily: AF, fontWeight: 600, flexShrink: 0 }}>{gapAnalysis.criticalGaps} critical</span>
                    )}
                  </div>
                )}
                {a.summary && !gapAnalysis && (
                  <div style={{ fontSize: 13, color: theme.text, fontFamily: SERIF, lineHeight: 1.5, marginBottom: a.what_track_needs?.length ? 8 : 0 }}>
                    {a.summary}
                  </div>
                )}
                {a.what_track_needs && a.what_track_needs.length > 0 && (
                  <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                    <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF, alignSelf: "center" }}>Needs:</span>
                    {a.what_track_needs.slice(0, 6).map((need, i) => (
                      <span key={i} style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)", color: theme.textSec, fontFamily: AF }}>{need}</span>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Tabs */}
            <div style={{ display: "flex", background: theme.surface, borderBottom: "1px solid " + theme.border, padding: "0 14px", alignItems: "center" }}>
              {[
                { id: "matched", l: "AI Matched", c: displaySamples.length },
                { id: "favorites", l: "Favorites", c: favorites.size },
              ].map(t => (
                <button key={t.id} onClick={() => setTab(t.id)} style={{ padding: "10px 14px", border: "none", background: "transparent", color: tab === t.id ? (t.color || theme.text) : theme.textMuted, fontSize: 11, fontWeight: tab === t.id ? 600 : 400, cursor: "pointer", borderBottom: tab === t.id ? "2px solid " + (t.color || theme.accent) : "2px solid transparent", fontFamily: AF, transition: "all 0.2s ease" }}>
                  {t.l}<span style={{ marginLeft: 4, fontSize: 9, padding: "1px 4px", borderRadius: 4, background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)" }}>{t.c}</span>
                </button>
              ))}
              {/* v2 view mode toggle */}
              {v2Available && (
                <div style={{ marginLeft: "auto", display: "flex", borderRadius: 5, overflow: "hidden", border: "1px solid " + theme.borderLight }}>
                  <button onClick={() => setViewMode("smart")} style={{ padding: "5px 10px", border: "none", background: viewMode === "smart" ? (isDark ? "rgba(217,70,239,0.15)" : "rgba(217,70,239,0.1)") : "transparent", color: viewMode === "smart" ? "#D946EF" : theme.textMuted, fontSize: 9, fontWeight: viewMode === "smart" ? 700 : 400, cursor: "pointer", fontFamily: AF }}>
                    Smart Match{v2Loading ? " ..." : ""}
                  </button>
                  <button onClick={() => setViewMode("all")} style={{ padding: "5px 10px", border: "none", borderLeft: "1px solid " + theme.borderLight, background: viewMode === "all" ? (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)") : "transparent", color: viewMode === "all" ? theme.text : theme.textMuted, fontSize: 9, fontWeight: viewMode === "all" ? 700 : 400, cursor: "pointer", fontFamily: AF }}>
                    Full Library
                  </button>
                </div>
              )}
            </div>

            {/* Batch Export Toolbar */}
            {checkedSamples.size > 0 && (
              <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 14px", background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)", borderBottom: "1px solid " + theme.borderLight }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: theme.text, fontFamily: AF }}>{checkedSamples.size} selected</span>
                <button onClick={exportChecked} style={{ fontSize: 9, padding: "4px 12px", borderRadius: 5, border: "none", background: isDark ? theme.text : "#1A1A1A", color: isDark ? "#0D0D12" : "#fff", cursor: "pointer", fontWeight: 600, fontFamily: AF }}>Export Kit (.zip)</button>
                <button onClick={selectAllVisible} style={{ fontSize: 9, padding: "4px 10px", borderRadius: 5, border: "1px solid " + theme.border, background: "transparent", color: theme.textSec, cursor: "pointer", fontFamily: AF }}>Select All</button>
                <button onClick={clearChecked} style={{ fontSize: 9, padding: "4px 10px", borderRadius: 5, border: "1px solid " + theme.border, background: "transparent", color: theme.textSec, cursor: "pointer", fontFamily: AF }}>Clear</button>
              </div>
            )}

            {(
              /* ── Sample List ── */
              <>
                <div style={{ display: "grid", gridTemplateColumns: "22px 30px 1fr 54px 40px 40px 42px", gap: 6, padding: "6px 14px", fontSize: 8, color: theme.textFaint, textTransform: "uppercase", letterSpacing: 1.5, background: theme.surface, borderBottom: "1px solid " + theme.borderLight, fontFamily: AF }}>
                  <span onClick={checkedSamples.size > 0 ? clearChecked : selectAllVisible} style={{ cursor: "pointer", textAlign: "center", fontSize: 7 }}>{checkedSamples.size > 0 ? "✓" : ""}</span><span /><span>Name</span><span>Match</span><span>Key</span><span>BPM</span><span>Len</span>
                </div>
                <div ref={scrollRef} onScroll={handleScroll} style={{ flex: 1, overflowY: "auto", background: theme.surface }}>
                  {samples.length === 0 && indexProgress && !indexProgress.done ? (
                    Array.from({ length: 12 }).map((_, i) => <SkeletonRow key={i} />)
                  ) : filtered.length > 0 ? (
                    <div style={{ height: totalHeight, position: "relative" }}>
                      <div style={{ position: "absolute", top: startIdx * ROW_HEIGHT, width: "100%" }}>
                        {visibleSamples.map((s, i) => (
                          <div key={s.id} draggable onDragStart={e => handleDragStart(e, s)} style={{ height: ROW_HEIGHT }}>
                            <SampleRow sample={s} isActive={activeSample?.id === s.id} isPlaying={audio.currentId === s.id && audio.playing} onPlay={handlePlay} isSelected={(startIdx + i) === selectedIdx} isChecked={checkedSamples.has(s.id)} onCheck={toggleCheck} onHoverWaveform={handleHoverWaveform} dawSync={isSynced} />
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : <div style={{ padding: 40, textAlign: "center", color: theme.textMuted, fontSize: 12 }}>{samples.length === 0 ? "No samples." : tab === "favorites" ? "No favorites." : "No matches."}</div>}
                </div>
              </>
            )}

            {/* Player */}
            {activeSample && (
              <div style={{ padding: "8px 14px", borderTop: "1px solid " + theme.border, background: theme.surface }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <button onClick={() => audio.toggle(activeSample.path, activeSample.id, isSynced)} style={{ width: 32, height: 32, borderRadius: "50%", border: "none", background: theme.gradient, cursor: "pointer", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 2px 12px rgba(217,70,239,0.2), 0 2px 6px rgba(6,182,212,0.15)" }}>
                    {audio.currentId === activeSample.id && audio.playing
                      ? <svg width="9" height="9" viewBox="0 0 14 14" fill="#fff"><rect x="2" y="1" width="4" height="12" rx="1" /><rect x="8" y="1" width="4" height="12" rx="1" /></svg>
                      : <svg width="9" height="9" viewBox="0 0 14 14" fill="#fff"><path d="M3 1v12l10-6z" /></svg>}
                  </button>
                  <div style={{ flex: "0 0 130px", minWidth: 0 }}>
                    <div style={{ fontSize: 12, fontWeight: 500, color: theme.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: SERIF }}>{activeSample.clean_name || activeSample.name}</div>
                    <div style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF }}>{activeSample.type_label} · {activeSample.key} · {activeSample.bpm ? Math.round(activeSample.bpm) : "—"}</div>
                  </div>
                  <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: MONO, flexShrink: 0 }}>{formatTime(audio.currentTime)}</span>
                  <div style={{ flex: 1, padding: "0 4px" }}>
                    <RealWaveform peaks={waveformPeaks} progress={audio.currentId === activeSample.id ? audio.progress : 0} height={28} onClick={audio.seek} />
                  </div>
                  <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: MONO, flexShrink: 0 }}>{formatTime(audio.duration)}</span>
                  <span style={{ fontSize: 11, fontWeight: 700, color: theme.text, fontFamily: MONO }}>{Math.round(activeSample.match || 50)}%</span>
                  <StarRating sampleId={activeSample.id} />
                  <button onClick={() => toggleFav(activeSample.id)} style={{ background: "none", border: "none", fontSize: 14, cursor: "pointer", color: favorites.has(activeSample.id) ? "#D97706" : theme.textFaint, padding: "2px" }}>
                    {favorites.has(activeSample.id) ? "★" : "☆"}
                  </button>
                  <button onClick={() => findSimilar(activeSample)} title="Find Similar (S)" style={{ background: "none", border: "none", color: theme.textMuted, cursor: "pointer", fontSize: 11, padding: "2px 4px", fontFamily: AF }}>≈</button>
                  <button onClick={() => findLayers(activeSample)} title="Layer Suggestions (L)" style={{ background: "none", border: "none", color: theme.textMuted, cursor: "pointer", fontSize: 9, padding: "2px 4px", fontFamily: AF }}>Layer</button>
                  <button onClick={audio.toggleMix} style={{ padding: "4px 8px", borderRadius: 4, fontSize: 9, fontWeight: 600, cursor: "pointer", border: "1px solid " + theme.border, background: audio.mixMode ? (isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)") : "transparent", color: audio.mixMode ? theme.text : theme.textMuted, fontFamily: AF }}>
                    {audio.mixMode ? "◉ Mix" : "Mix"}
                  </button>
                </div>
                {audio.mixMode && (
                  <div style={{ display: "flex", gap: 12, padding: "4px 40px 0" }}>
                    <div style={{ flex: 1 }}><VolumeSlider value={audio.trackVol} onChange={audio.setTrackVol} label="Track" /></div>
                    <div style={{ flex: 1 }}><VolumeSlider value={audio.sampleVol} onChange={audio.setSampleVol} label="Sample" /></div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
