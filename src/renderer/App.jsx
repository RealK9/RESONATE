/**
 * RESONATE — App Orchestrator.
 * Thin shell: all state management lives here, rendering delegated to
 * AppShell (persistent layout) + routed pages (HomePage, SoundsPage, etc.).
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useTheme } from "./theme/ThemeProvider";
import { AF, MONO } from "./theme/fonts";
import { useRouter } from "./router";
import { useAudioPlayer } from "./hooks/useAudioPlayer";
import { useWaveformData } from "./hooks/useWaveformData";
import { useApi, API } from "./hooks/useApi";
import { mergeV2WithV1, formatNeeds, formatGapAnalysis, buildV1Compat } from "./utils/v2Adapter";
import { useToast } from "./components/Toast";
import { useBridge } from "./hooks/useBridge";
import { usePitchTempo } from "./hooks/usePitchTempo";
import { AppShell } from "./layouts/AppShell";
import { AnalyzingOverlay } from "./components/AnalyzingOverlay";
import { ShortcutOverlay } from "./components/ShortcutOverlay";
import { Modal } from "./components/Modal";
import { HomePage } from "./pages/HomePage";
import { SoundsPage } from "./pages/SoundsPage";
import { CollectionsPage } from "./pages/CollectionsPage";
import { LibraryPage } from "./pages/LibraryPage";

// ── Logo with background stripped + animated light tracing down the colored strokes ──
function LogoBlend({ size, isDark, animate = true }) {
  const canvasRef = useRef(null);
  const baseDataRef = useRef(null);
  const animRef = useRef(null);

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
      baseDataRef.current = new Uint8ClampedArray(d);
      ctx.putImageData(id, 0, 0);

      if (animate) {
        const startTime = Date.now();
        const drawFrame = () => {
          const t = ((Date.now() - startTime) / 1000) % 3;
          const norm = t / 3;
          const base = baseDataRef.current;
          if (!base || !c) return;
          const frame = ctx.createImageData(rs, rs);
          const fd = frame.data;
          const waveCenter = norm;
          const waveWidth = 0.25;
          for (let y = 0; y < rs; y++) {
            const yNorm = y / rs;
            let dist = Math.abs(yNorm - waveCenter);
            if (dist > 0.5) dist = 1 - dist;
            const boost = dist < waveWidth ? Math.cos((dist / waveWidth) * Math.PI * 0.5) : 0;
            for (let x = 0; x < rs; x++) {
              const i = (y * rs + x) * 4;
              const alpha = base[i + 3];
              if (alpha === 0) {
                fd[i] = fd[i+1] = fd[i+2] = fd[i+3] = 0;
                continue;
              }
              const brighten = 1 + boost * 0.8;
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

const STAGES = ["Uploading audio...","Detecting tempo & groove...","Analyzing harmonic content...","Mapping frequency spectrum...","Identifying instrumentation...","Consulting AI engine...","Scoring samples..."];

export default function App() {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const toast = useToast();
  const api = useApi();
  const { page, navigate } = useRouter();

  // ── Core State ──
  const [analyzing, setAnalyzing] = useState(false);
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
  const [sortBy, setSortBy] = useState("relevant");
  const tmr = useRef(null);
  const audio = useAudioPlayer();
  const pitchTempo = usePitchTempo();
  const bridge = useBridge();

  audio.connectPitchTempo(pitchTempo);

  // ── Sidebar UI State ──
  const [analyzerExpanded, setAnalyzerExpanded] = useState(false);

  // ── New Feature State ──
  const [ratings, setRatings] = useState({});
  const [similarSamples, setSimilarSamples] = useState(null);
  const [layerSamples, setLayerSamples] = useState(null);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [showSessions, setShowSessions] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [sourceFilter, setSourceFilter] = useState("all");
  const [moodFilter, setMoodFilter] = useState("all");
  const [checkedSamples, setCheckedSamples] = useState(new Set());

  // ── v2 ML Pipeline State ──
  const [mixProfile, setMixProfile] = useState(null);
  const [v2Recommendations, setV2Recommendations] = useState([]);
  const [mixNeeds, setMixNeeds] = useState([]);
  const [gapAnalysis, setGapAnalysis] = useState(null);
  const [prevReadiness, setPrevReadiness] = useState(null);
  const [reanalyzing, setReanalyzing] = useState(false);
  const [v2Available, setV2Available] = useState(true);
  const [v2Loading, setV2Loading] = useState(false);
  const [viewMode, setViewMode] = useState("smart");
  const [collections, setCollections] = useState([]);

  // ── Producer DNA / Taste Profile State ──
  const [dnaProfile, setDnaProfile] = useState(null);
  const [dnaTraining, setDnaTraining] = useState(false);
  const [ringGlowing, setRingGlowing] = useState(false);
  const [chartComparison, setChartComparison] = useState(null);

  // ── Version Tracking State ──
  const [versions, setVersions] = useState([]);

  const waveformUrl = useMemo(() => activeSample ? API + "/samples/audio/" + encodeURI(activeSample.path) : null, [activeSample]);
  const waveformPeaks = useWaveformData(waveformUrl);

  // ── Readiness ring glow on change ──
  const prevReadinessRef = useRef(null);
  useEffect(() => {
    if (gapAnalysis?.readiness != null && prevReadinessRef.current != null && gapAnalysis.readiness !== prevReadinessRef.current) {
      setRingGlowing(true);
      const t = setTimeout(() => setRingGlowing(false), 1200);
      return () => clearTimeout(t);
    }
    if (gapAnalysis?.readiness != null) prevReadinessRef.current = gapAnalysis.readiness;
  }, [gapAnalysis?.readiness]);

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

  // ── Auto-refetch samples when DAW BPM changes ──
  const rescoreTimerRef = useRef(null);
  useEffect(() => {
    if (bridge.rescoreNeeded && backendOk && samples.length > 0) {
      clearTimeout(rescoreTimerRef.current);
      rescoreTimerRef.current = setTimeout(() => {
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

  // ── Analysis Pipeline ──
  const runAnalysis = useCallback(async (file) => {
    setAnalyzing(true); setProgress(0); setError(null); setSimilarSamples(null);
    setMixProfile(null); setV2Recommendations([]); setMixNeeds([]); setGapAnalysis(null); setChartComparison(null); setV2Loading(false);
    let p = 0, si = 0; setStage(STAGES[0]);
    tmr.current = setInterval(() => { p += Math.random() * 1.5 + 0.5; p = Math.min(p, 90); const nsi = Math.min(Math.floor(p / (90 / (STAGES.length - 1))), STAGES.length - 2); if (nsi !== si) { si = nsi; setStage(STAGES[nsi]); } setProgress(Math.round(p)); }, 100);
    try {
      let v2Success = false;
      let mp = null;
      try {
        const fullResult = await api.analyzeFullV2(file, 30);
        mp = fullResult.mix_profile;
        setMixProfile(mp);
        setMixNeeds(formatNeeds(mp));
        setGapAnalysis(formatGapAnalysis(fullResult.gap_analysis));
        if (fullResult.recommendations?.recommendations) {
          setV2Recommendations(fullResult.recommendations.recommendations);
        }
        v2Success = true;
        setV2Available(true);
        api.getChartComparison().then(c => setChartComparison(c)).catch(() => {});
        setAnalysisResult(buildV1Compat(mp, fullResult.summary));
      } catch {
        try {
          const v2Result = await api.analyzeTrackV2(file);
          mp = v2Result;
          setMixProfile(v2Result);
          setMixNeeds(formatNeeds(v2Result));
          api.getGapAnalysisV2().then(g => setGapAnalysis(formatGapAnalysis(g))).catch(() => {});
          v2Success = true;
          setV2Available(true);
          api.getChartComparison().then(c => setChartComparison(c)).catch(() => {});
          setAnalysisResult(buildV1Compat(v2Result));
        } catch {
          setV2Available(false);
          const fd = new FormData(); fd.append("file", file);
          const res = await fetch(API + "/analyze", { method: "POST", body: fd });
          if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Failed"); }
          const result = await res.json(); setAnalysisResult(result);
          pitchTempo.setUploadAnalysis(result?.analysis?.key, result?.analysis?.bpm);
        }
      }
      clearInterval(tmr.current); setStage(STAGES[STAGES.length - 1]); setProgress(100);
      await loadSamples(); await audio.loadTrack();

      if (mp) {
        pitchTempo.setUploadAnalysis(mp.analysis?.key, mp.analysis?.bpm);
      }

      if (v2Success && v2Recommendations.length === 0) {
        setV2Loading(true);
        api.getRecommendationsV2(30).then(recsResult => {
          if (recsResult?.recommendations) setV2Recommendations(recsResult.recommendations);
          setV2Loading(false);
          api.getCollections().then(res => {
            if (res?.collections) setCollections(res.collections);
          }).catch(() => {});
        }).catch(() => setV2Loading(false));
      }

      if (v2Success && v2Recommendations.length > 0) {
        api.getCollections().then(res => {
          if (res?.collections) setCollections(res.collections);
        }).catch(() => {});
      }

      const projectName = file.name.replace(/\.[^/.]+$/, "");
      api.saveVersion(projectName, null, file.name).then(() => {
        api.getVersions(projectName).then(res => {
          if (res?.versions) setVersions(res.versions);
        }).catch(() => {});
      }).catch(() => {});

      setTimeout(() => {
        setAnalyzing(false);
        navigate("sounds");
        const readiness = gapAnalysis?.readiness;
        toast.success(readiness != null ? `Analysis complete — ${readiness}/100 readiness` : "Analysis complete");
      }, 500);
    } catch (e) { clearInterval(tmr.current); setError(e.message); setProgress(0); setAnalyzing(false); navigate("home"); toast.error(e.message); }
  }, [loadSamples, audio, toast, api, navigate]);

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
  const clearChecked = useCallback(() => setCheckedSamples(new Set()), []);

  // ── Batch Export ──
  const exportChecked = useCallback(async () => {
    if (checkedSamples.size === 0) { toast.error("No samples selected"); return; }
    try {
      const paths = [...checkedSamples].map(id => { const s = samples.find(x => x.id === id); return s?.path || id; });
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

  // ── v2 Feedback ──
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
      api.trainTaste().catch(() => {});
    } catch { toast.error("Failed to save session"); }
  }, [fileName, toast, api]);

  const loadSession = useCallback(async (sessionId) => {
    try {
      const r = await fetch(API + "/sessions/" + sessionId);
      const d = await r.json();
      const sessAnalysis = { ...d.track_profile, ...d.ai_analysis };
      setAnalysisResult({ analysis: sessAnalysis });
      pitchTempo.setUploadAnalysis(sessAnalysis.key, sessAnalysis.bpm);
      setFileName(d.track_filename);
      setCurrentSessionId(d.id);
      setShowSessions(false);
      await loadSamples();
      await audio.loadTrack();
      navigate("sounds");
      toast.success("Session loaded: " + d.name);
    } catch { toast.error("Failed to load session"); }
  }, [loadSamples, audio, toast, navigate]);

  const deleteSession = useCallback(async (sessionId) => {
    try {
      await fetch(API + "/sessions/" + sessionId, { method: "DELETE" });
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      toast.info("Session deleted");
    } catch { toast.error("Failed to delete session"); }
  }, [toast]);

  // ── Merge v2 recommendations ──
  const displaySamples = useMemo(() => {
    if (viewMode === "smart" && v2Recommendations.length > 0) {
      return mergeV2WithV1(v2Recommendations, samples);
    }
    return samples;
  }, [viewMode, v2Recommendations, samples]);

  // ── Categories ──
  const cats = useMemo(() => ["all", ...new Set(displaySamples.filter(s => s.category).map(s => s.category.toLowerCase()))], [displaySamples]);

  // ── Source counts ──
  const sourceCounts = useMemo(() => {
    const c = { all: displaySamples.length, local: 0, splice: 0, loopcloud: 0 };
    for (const s of displaySamples) { const src = s.source || "local"; c[src] = (c[src] || 0) + 1; }
    return c;
  }, [displaySamples]);

  const makeMeta = (s) => ({ key: s.key, bpm: s.bpm });
  const handlePlay = useCallback((s) => {
    setActiveSample(s);
    audio.toggle(s.path, s.id, makeMeta(s));
    logV2Feedback(s, "audition");
  }, [audio, logV2Feedback]);

  const selectAllVisible = useCallback(() => {
    setCheckedSamples(new Set(displaySamples.map(s => s.id)));
  }, [displaySamples]);

  const analyzeFromBridge = useCallback(async () => {
    try {
      setAnalyzing(true); setProgress(0); setStage(STAGES[0]);
      let p = 0, si = 0;
      tmr.current = setInterval(() => { p += Math.random() * 1.5 + 0.5; p = Math.min(p, 90); const nsi = Math.min(Math.floor(p / (90 / (STAGES.length - 1))), STAGES.length - 2); if (nsi !== si) { si = nsi; setStage(STAGES[nsi]); } setProgress(Math.round(p)); }, 100);
      const r = await fetch(API + "/analyze/bridge", { method: "POST" });
      clearInterval(tmr.current);
      if (r.ok) {
        const d = await r.json();
        setAnalysisResult(d); setStage(STAGES[STAGES.length - 1]); setProgress(100);
        setFileName(d.filename || "DAW Master");
        await loadSamples(); await audio.loadTrack();
        setTimeout(() => { setAnalyzing(false); navigate("sounds"); toast.success("Bridge analysis complete"); }, 500);
      } else { setAnalyzing(false); navigate("home"); toast.error("Bridge analysis failed — play some audio first"); }
    } catch { setAnalyzing(false); navigate("home"); toast.error("Bridge analysis failed"); }
  }, [loadSamples, audio, toast, navigate]);

  // ── Re-Analyze ──
  const reAnalyze = useCallback(async () => {
    if (reanalyzing) return;
    setReanalyzing(true);
    if (gapAnalysis?.readiness != null) setPrevReadiness(gapAnalysis.readiness);
    try {
      if (bridge.connected) {
        const r = await fetch(API + "/analyze/bridge", { method: "POST" });
        if (!r.ok) { toast.error("Bridge capture failed — play some audio first"); setReanalyzing(false); return; }
        const d = await r.json();
        setAnalysisResult(d);
        setFileName(d.filename || "DAW Master");
      }
      if (v2Available) {
        try {
          const gapResult = await api.getGapAnalysisV2();
          const newGap = formatGapAnalysis(gapResult);
          setGapAnalysis(newGap);
          const needsResult = await api.getNeedsV2();
          setMixNeeds(formatNeeds(needsResult));
          const recsResult = await api.getRecommendationsV2(30);
          if (recsResult?.recommendations) setV2Recommendations(recsResult.recommendations);
          await loadSamples();
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

  // ── Drag-to-DAW ──
  const handleDragStart = useCallback(async (e, sample) => {
    e.dataTransfer.setData("text/plain", sample.clean_name || sample.name);
    try {
      const syncParam = (bridge.connected && bridge.dawSync) ? "?sync=1" : "";
      const r = await fetch(API + "/samples/abspath/" + encodeURI(sample.path) + syncParam);
      const d = await r.json();
      if (d.path) {
        e.dataTransfer.setData("text/uri-list", "file://" + d.path);
        if (window.electronAPI?.startDrag) window.electronAPI.startDrag(d.path);
      }
    } catch {}
    fetch(API + "/usage", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sample_filepath: sample.id, action: "drag", session_id: currentSessionId }) }).catch(() => {});
    logV2Feedback(sample, "drag");
  }, [currentSessionId, bridge.connected, bridge.dawSync, logV2Feedback]);

  const handleNew = useCallback(() => {
    navigate("home");
    setActiveSample(null); audio.stop(); setAnalysisResult(null); setSelectedIdx(-1);
    setSimilarSamples(null); setMixProfile(null); setV2Recommendations([]);
    setMixNeeds([]); setGapAnalysis(null); setChartComparison(null);
    setCollections([]); setViewMode("smart"); pitchTempo.setUploadAnalysis(null, 0);
  }, [audio, navigate]);

  // ── Tab change handler ──
  const handleTabChange = useCallback((newTab) => {
    setTab(newTab);
    if (newTab === "dna") api.getTasteProfile().then(setDnaProfile).catch(() => {});
  }, [api]);

  // ── Train DNA handler ──
  const handleTrainDNA = useCallback(async () => {
    setDnaTraining(true);
    try { await api.trainTaste(); const p = await api.getTasteProfile(); setDnaProfile(p); } catch {}
    setDnaTraining(false);
  }, [api]);

  // ── Version save handler ──
  const handleSaveVersion = useCallback(() => {
    const pName = fileName.replace(/\.[^/.]+$/, "");
    api.saveVersion(pName, null, fileName).then(() => {
      api.getVersions(pName).then(res => {
        if (res?.versions) setVersions(res.versions);
      }).catch(() => {});
      toast.success("Version saved");
    }).catch(() => toast.error("Failed to save version"));
  }, [fileName, api, toast]);

  const handleSelectVersion = useCallback((v) => {
    if (v.readiness_score != null) {
      toast.success(`${v.version_label}: readiness ${Math.round(v.readiness_score)}/100`);
    }
  }, [toast]);

  // ── Keyboard Shortcuts ──
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

      if (showShortcuts) { setShowShortcuts(false); return; }
      if (showSessions) { if (e.key === "Escape") setShowSessions(false); return; }
      if (similarSamples) { if (e.key === "Escape") setSimilarSamples(null); return; }
      if (layerSamples) { if (e.key === "Escape") setLayerSamples(null); return; }

      if (e.code === "Space" && activeSample) {
        e.preventDefault();
        audio.toggle(activeSample.path, activeSample.id, makeMeta(activeSample));
      } else if (e.key === "m" || e.key === "M") {
        audio.toggleMix();
      } else if (e.key === "f" || e.key === "F") {
        if (activeSample) toggleFav(activeSample.id);
      } else if (e.key >= "1" && e.key <= "5") {
        if (activeSample) rateSample(activeSample.id, parseInt(e.key));
      } else if (e.key === "s" || e.key === "S") {
        if (activeSample) findSimilar(activeSample);
      } else if (e.key === "r" || e.key === "R") {
        handleNew();
      } else if (e.key === "?" || (e.key === "/" && e.shiftKey)) {
        setShowShortcuts(true);
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
  }, [activeSample, audio, toggleFav, rateSample, findSimilar, findLayers, saveSession, showShortcuts, showSessions, similarSamples, layerSamples, checkedSamples, clearChecked, handleNew]);

  // ── Settings click (placeholder) ──
  const handleSettingsClick = useCallback(() => {
    // TODO: open settings modal
  }, []);

  // ── Render ──
  const renderPage = () => {
    switch (page) {
      case "sounds":
        return (
          <SoundsPage
            analysisResult={analysisResult} fileName={fileName}
            samples={samples} displaySamples={displaySamples}
            category={category} onCategoryChange={setCategory} categories={cats}
            selectedKey={selectedKey} onKeyChange={setSelectedKey}
            sourceFilter={sourceFilter} onSourceChange={setSourceFilter}
            sourceCounts={sourceCounts}
            moodFilter={moodFilter} onMoodChange={setMoodFilter}
            sortBy={sortBy} onSortChange={setSortBy}
            search={search}
            tab={tab} onTabChange={handleTabChange}
            viewMode={viewMode} onViewModeChange={setViewMode}
            v2Available={v2Available} v2Loading={v2Loading}
            audio={audio} activeSample={activeSample} onPlay={handlePlay}
            favorites={favorites} onToggleFav={toggleFav}
            onFindSimilar={findSimilar} onFindLayers={findLayers}
            checkedSamples={checkedSamples} onCheck={toggleCheck}
            onSelectAll={selectAllVisible} onClearChecked={clearChecked}
            onExportChecked={exportChecked}
            mixNeeds={mixNeeds} gapAnalysis={gapAnalysis}
            prevReadiness={prevReadiness} ringGlowing={ringGlowing}
            chartComparison={chartComparison} bridge={bridge}
            analyzerExpanded={analyzerExpanded}
            onToggleAnalyzer={() => setAnalyzerExpanded(!analyzerExpanded)}
            onSaveSession={saveSession}
            onShowSessions={() => setShowSessions(true)}
            sessions={sessions}
            onReAnalyze={reAnalyze} reanalyzing={reanalyzing}
            onAnalyzeFromBridge={analyzeFromBridge}
            collections={collections}
            versions={versions}
            onSaveVersion={handleSaveVersion}
            onSelectVersion={handleSelectVersion}
            dnaProfile={dnaProfile} dnaTraining={dnaTraining}
            onTrainDNA={handleTrainDNA}
            indexProgress={indexProgress}
            api={api}
            onDragStart={handleDragStart}
            pitchTempo={pitchTempo}
            mixProfile={mixProfile}
          />
        );
      case "collections":
        return (
          <CollectionsPage
            collections={collections} audio={audio}
            onPlay={handlePlay} api={api}
          />
        );
      case "library":
        return (
          <LibraryPage
            samples={samples} favorites={favorites} ratings={ratings}
            audio={audio} activeSample={activeSample}
            onPlay={handlePlay} onToggleFav={toggleFav}
            onFindSimilar={findSimilar} onFindLayers={findLayers}
          />
        );
      case "home":
      default:
        return (
          <HomePage
            LogoBlend={LogoBlend}
            error={error}
            dragOver={dragOver}
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onUpload={handleUpload}
            bridge={bridge}
            onAnalyzeFromBridge={analyzeFromBridge}
            samples={samples}
            sessions={sessions}
            onShowSessions={() => setShowSessions(true)}
          />
        );
    }
  };

  return (
    <>
      <AppShell
        search={search} onSearchChange={setSearch}
        onSettingsClick={handleSettingsClick}
        bridgeConnected={bridge.connected}
        backendOk={backendOk} indexProgress={indexProgress}
        audio={audio} activeSample={activeSample}
        pitchTempo={pitchTempo} waveformPeaks={waveformPeaks}
        onToggleFav={toggleFav}
        isFavorite={activeSample ? favorites.has(activeSample.id) : false}
        onFindSimilar={findSimilar}
        onToggleMix={audio.toggleMix}
        hasAnalysis={!!analysisResult}
      >
        {renderPage()}
      </AppShell>

      {/* Analyzing Overlay — shown on top of everything during analysis */}
      {analyzing && (
        <AnalyzingOverlay
          progress={progress} stage={stage}
          fileName={fileName} LogoBlend={LogoBlend}
        />
      )}

      {/* Shortcut Overlay */}
      {showShortcuts && <ShortcutOverlay onClose={() => setShowShortcuts(false)} />}

      {/* Session History Modal */}
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
                <button onClick={() => deleteSession(s.id)} style={{ fontSize: 9, padding: "3px 8px", borderRadius: 4, border: "1px solid " + theme.border, background: "transparent", color: theme.textMuted, cursor: "pointer" }}>&times;</button>
              </div>
            </div>
          ))}
        </div>
      </Modal>

      {/* Similarity Panel */}
      <Modal visible={!!similarSamples} onClose={() => setSimilarSamples(null)} width={520}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: theme.text, fontFamily: AF }}>Similar Samples</div>
          <button onClick={() => setSimilarSamples(null)} style={{ background: "none", border: "none", color: theme.textMuted, cursor: "pointer", fontSize: 16 }}>&times;</button>
        </div>
        <div style={{ flex: 1, overflowY: "auto" }}>
          {(similarSamples || []).length === 0 ? (
            <div style={{ padding: 20, textAlign: "center", color: theme.textMuted, fontSize: 11 }}>No similar samples found</div>
          ) : (similarSamples || []).map(s => (
            <div key={s.id} onClick={() => { setActiveSample(s); audio.toggle(s.path, s.id, makeMeta(s)); }} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", borderRadius: 6, marginBottom: 4, cursor: "pointer", background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)", border: "1px solid " + theme.borderLight }}>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 12, color: theme.text, fontFamily: "'EB Garamond', serif", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.clean_name || s.name}</div>
                <div style={{ fontSize: 9, color: theme.textMuted }}>{s.type_label} · {s.key} · {s.bpm ? Math.round(s.bpm) : "\u2014"}</div>
              </div>
              <span style={{ fontSize: 11, fontWeight: 700, color: theme.text, fontFamily: MONO, marginLeft: 8 }}>{s.similarity}%</span>
            </div>
          ))}
        </div>
      </Modal>

      {/* Layering Panel */}
      <Modal visible={!!layerSamples} onClose={() => setLayerSamples(null)} width={520}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: theme.text, fontFamily: AF }}>Layer Suggestions</div>
          <button onClick={() => setLayerSamples(null)} style={{ background: "none", border: "none", color: theme.textMuted, cursor: "pointer", fontSize: 16 }}>&times;</button>
        </div>
        <div style={{ flex: 1, overflowY: "auto" }}>
          {(layerSamples || []).length === 0 ? (
            <div style={{ padding: 20, textAlign: "center", color: theme.textMuted, fontSize: 11 }}>No layering suggestions found</div>
          ) : (layerSamples || []).map(s => (
            <div key={s.id} onClick={() => { setActiveSample(s); audio.toggle(s.path, s.id, makeMeta(s)); }} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", borderRadius: 6, marginBottom: 4, cursor: "pointer", background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)", border: "1px solid " + theme.borderLight }}>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 12, color: theme.text, fontFamily: "'EB Garamond', serif", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.clean_name || s.name}</div>
                <div style={{ fontSize: 9, color: theme.textMuted }}>{s.type_label} · {s.key} · {s.layer_reason}</div>
              </div>
              <span style={{ fontSize: 11, fontWeight: 700, color: theme.text, fontFamily: MONO, marginLeft: 8 }}>{Math.round(s.layer_score)}%</span>
            </div>
          ))}
        </div>
      </Modal>
    </>
  );
}
