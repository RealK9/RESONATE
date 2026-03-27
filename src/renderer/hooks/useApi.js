/**
 * RESONATE — API Hook.
 * Centralized backend API calls.
 */

// Configurable API URL — set VITE_RESONATE_API_URL for remote server
const API = (typeof import.meta !== "undefined" && import.meta.env?.VITE_RESONATE_API_URL)
  || "http://localhost:8000";

export function useApi() {
  const checkHealth = async () => {
    const r = await fetch(API + "/health");
    return r.json();
  };

  const getSettings = async () => {
    const r = await fetch(API + "/settings");
    return r.json();
  };

  const setSampleDir = async (path) => {
    const r = await fetch(API + "/settings/sample-dir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!r.ok) {
      const e = await r.json();
      throw new Error(e.detail);
    }
    return r.json();
  };

  const analyzeTrack = async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(API + "/analyze", { method: "POST", body: fd });
    if (!r.ok) {
      const e = await r.json();
      throw new Error(e.detail || "Analysis failed");
    }
    return r.json();
  };

  const getSamples = async () => {
    const r = await fetch(API + "/samples");
    return r.json();
  };

  const getSampleAbsPath = async (path) => {
    const r = await fetch(API + "/samples/abspath/" + encodeURI(path));
    return r.json();
  };

  // ---------------------------------------------------------------------------
  // v2 ML pipeline endpoints
  // ---------------------------------------------------------------------------

  /** Full v2 analysis (runs v1 internally for backward compat). */
  const analyzeTrackV2 = async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(API + "/analyze/v2", { method: "POST", body: fd });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || "v2 analysis failed");
    }
    return r.json();
  };

  /** Generate v2 recommendations for the latest analyzed mix. */
  const getRecommendationsV2 = async (maxResults = 30) => {
    const r = await fetch(API + `/recommend/v2?max_results=${maxResults}`, { method: "POST" });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || "Recommendation failed");
    }
    return r.json();
  };

  /** Log a user interaction with a recommended sample. */
  const logFeedbackV2 = (params) => {
    const qs = new URLSearchParams();
    qs.set("sample_filepath", params.sample_filepath);
    qs.set("action", params.action);
    if (params.mix_filepath) qs.set("mix_filepath", params.mix_filepath);
    if (params.session_id) qs.set("session_id", params.session_id);
    if (params.rating != null) qs.set("rating", String(params.rating));
    if (params.recommendation_rank != null) qs.set("recommendation_rank", String(params.recommendation_rank));
    // Fire-and-forget — don't await, don't block UI
    fetch(API + "/feedback/v2?" + qs.toString(), { method: "POST" }).catch(() => {});
  };

  /** Train the per-user preference model from accumulated feedback. */
  const trainPreferencesV2 = async (userId = "default", minPairs = 10) => {
    const r = await fetch(API + `/preference/v2/train?user_id=${userId}&min_pairs=${minPairs}`, { method: "POST" });
    return r.json();
  };

  /** Get the v2 needs vector from the latest analysis. */
  const getNeedsV2 = async () => {
    const r = await fetch(API + "/analyze/v2/needs");
    return r.json();
  };

  /** Get gap analysis results from the latest analysis. */
  const getGapAnalysisV2 = async () => {
    const r = await fetch(API + "/analyze/v2/gap");
    return r.json();
  };

  // ---------------------------------------------------------------------------
  // Smart Collections endpoints
  // ---------------------------------------------------------------------------

  /** Generate themed sample collections from current analysis. */
  const getCollections = async () => {
    const r = await fetch(API + "/collections/generate");
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || "Failed to generate collections");
    }
    return r.json();
  };

  /** Export a collection as a ZIP file and trigger download. */
  const exportCollection = async (collectionId) => {
    const r = await fetch(API + `/collections/export/${collectionId}`, { method: "POST" });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || "Export failed");
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `RESONATE-${collectionId}.zip`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ---------------------------------------------------------------------------
  // Chart Intelligence 2.0
  // ---------------------------------------------------------------------------

  /** Get chart trend data (decade + genre profiles). */
  const getChartTrends = async (genre = null, decade = null) => {
    const params = new URLSearchParams();
    if (genre) params.set("genre", genre);
    if (decade) params.set("decade", String(decade));
    const qs = params.toString();
    const r = await fetch(API + "/charts/trends" + (qs ? "?" + qs : ""));
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || "Failed to load chart trends");
    }
    return r.json();
  };

  /** Compare latest mix against chart averages. */
  const getChartComparison = async () => {
    const r = await fetch(API + "/charts/compare");
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || "Chart comparison failed");
    }
    return r.json();
  };

  /** Full RESONATE workflow: upload → analyze → gap → recommend in one call. */
  const analyzeFullV2 = async (file, maxResults = 30) => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(API + `/analyze/v2/full?max_results=${maxResults}`, {
      method: "POST", body: fd,
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || "Full analysis failed");
    }
    return r.json();
  };

  // ---------------------------------------------------------------------------
  // Taste profile / Producer DNA endpoints
  // ---------------------------------------------------------------------------

  const getTasteProfile = async (userId = "default") => {
    const r = await fetch(API + `/taste/profile?user_id=${userId}`);
    return r.json();
  };

  const trainTaste = async (userId = "default") => {
    const r = await fetch(API + `/taste/train?user_id=${userId}`, { method: "POST" });
    return r.json();
  };

  // ---------------------------------------------------------------------------
  // Version tracking endpoints
  // ---------------------------------------------------------------------------

  /** Save current analysis as a named version. */
  const saveVersion = async (projectName, versionLabel = null, filepath = "") => {
    const body = { project_name: projectName, filepath };
    if (versionLabel) body.version_label = versionLabel;
    const r = await fetch(API + "/versions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || "Failed to save version");
    }
    return r.json();
  };

  /** Get all versions for a project. */
  const getVersions = async (projectName) => {
    const r = await fetch(API + "/versions/" + encodeURIComponent(projectName));
    return r.json();
  };

  /** Compare two versions by id. */
  const compareVersions = async (idA, idB) => {
    const r = await fetch(API + `/versions/compare?id_a=${idA}&id_b=${idB}`);
    return r.json();
  };

  /** List all projects with version counts. */
  const listProjects = async () => {
    const r = await fetch(API + "/versions/projects");
    return r.json();
  };

  return {
    checkHealth, getSettings, setSampleDir,
    analyzeTrack, getSamples, getSampleAbsPath,
    // v2
    analyzeTrackV2, getRecommendationsV2, logFeedbackV2,
    trainPreferencesV2, getNeedsV2, getGapAnalysisV2, analyzeFullV2,
    // chart intelligence
    getChartTrends, getChartComparison,
    // taste profile
    getTasteProfile, trainTaste,
    // version tracking
    saveVersion, getVersions, compareVersions, listProjects,
    // smart collections
    getCollections, exportCollection,
    API,
  };
}

export { API };
