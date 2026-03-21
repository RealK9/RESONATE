/**
 * RESONATE — Waveform Tooltip.
 * Mini waveform preview that appears on hover over a sample row.
 * Fetches peaks lazily from the API and caches them.
 */

import { useState, useEffect, useRef } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { API } from "../hooks/useApi";

const peakCache = new Map();

export function WaveformTooltip({ samplePath, visible, anchorRect }) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const canvasRef = useRef(null);
  const [peaks, setPeaks] = useState(null);

  useEffect(() => {
    if (!visible || !samplePath) return;
    const encoded = encodeURIComponent(samplePath).replace(/%2F/g, "/");
    if (peakCache.has(samplePath)) {
      setPeaks(peakCache.get(samplePath));
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      fetch(`${API}/samples/waveform/${encoded}?bars=60`)
        .then(r => r.json())
        .then(data => {
          if (!cancelled && data.peaks) {
            peakCache.set(samplePath, data.peaks);
            setPeaks(data.peaks);
          }
        })
        .catch(() => {});
    }, 120);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [visible, samplePath]);

  useEffect(() => {
    if (!peaks || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = 200, h = 36;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const barW = w / peaks.length;
    const color = isDark ? "rgba(232,230,227,0.6)" : "rgba(26,26,26,0.5)";
    ctx.fillStyle = color;
    for (let i = 0; i < peaks.length; i++) {
      const barH = Math.max(1, peaks[i] * (h - 4));
      const x = i * barW;
      const y = (h - barH) / 2;
      ctx.fillRect(x + 0.5, y, Math.max(1, barW - 1), barH);
    }
  }, [peaks, isDark]);

  if (!visible || !anchorRect) return null;

  return (
    <div style={{
      position: "fixed",
      left: anchorRect.left + 40,
      top: anchorRect.top - 48,
      width: 216,
      height: 48,
      background: isDark ? "#1E1E28" : "#FFFFFF",
      border: "1px solid " + theme.border,
      borderRadius: 6,
      boxShadow: isDark ? "0 4px 16px rgba(0,0,0,0.5)" : "0 4px 16px rgba(0,0,0,0.12)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "6px 8px",
      zIndex: 9999,
      pointerEvents: "none",
      opacity: peaks ? 1 : 0.4,
      transition: "opacity 0.15s",
    }}>
      {peaks ? (
        <canvas ref={canvasRef} style={{ width: 200, height: 36 }} />
      ) : (
        <div style={{ fontSize: 9, color: theme.textMuted, letterSpacing: 1 }}>Loading...</div>
      )}
    </div>
  );
}
