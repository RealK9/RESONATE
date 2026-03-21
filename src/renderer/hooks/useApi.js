/**
 * RESONATE — API Hook.
 * Centralized backend API calls.
 */

const API = "http://localhost:8000";

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

  return { checkHealth, getSettings, setSampleDir, analyzeTrack, getSamples, getSampleAbsPath, API };
}

export { API };
