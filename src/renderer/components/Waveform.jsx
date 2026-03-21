/**
 * RESONATE — Waveform Visualization.
 * Canvas-based waveform renderer with click-to-seek and progress overlay.
 */

import { useRef, useEffect } from "react";
import { useTheme } from "../theme/ThemeProvider";

export function RealWaveform({ peaks, progress = 0, height = 28, onClick }) {
  const ref = useRef(null);
  const { theme } = useTheme();
  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  const bars = peaks ? peaks.length : 80;

  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    cv.width = bars * 5 * dpr; cv.height = height * dpr;
    const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
    const w = bars * 5, h = height, bW = w / bars, g = 1, pB = Math.floor(progress * bars);
    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < bars; i++) {
      const v = peaks ? peaks[i] || 0.02 : 0.02, bH = Math.max(v * h * 0.9, 2);
      ctx.fillStyle = i <= pB && progress > 0 ? theme.waveformActive : theme.waveformInactive;
      ctx.globalAlpha = i <= pB && progress > 0 ? theme.waveformActiveAlpha : theme.waveformInactiveAlpha;
      const x = i * bW + g, y = (h - bH) / 2, rw = bW - g * 2, r = Math.min(rw / 2, 2);
      ctx.beginPath(); ctx.moveTo(x + r, y); ctx.lineTo(x + rw - r, y); ctx.quadraticCurveTo(x + rw, y, x + rw, y + r);
      ctx.lineTo(x + rw, y + bH - r); ctx.quadraticCurveTo(x + rw, y + bH, x + rw - r, y + bH);
      ctx.lineTo(x + r, y + bH); ctx.quadraticCurveTo(x, y + bH, x, y + bH - r);
      ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }, [peaks, progress, height, bars, dpr, theme]);

  const handleClick = (e) => {
    if (!onClick || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onClick(pct);
  };

  return <canvas ref={ref} onClick={handleClick} style={{ width: bars * 5, height, maxWidth: "100%", cursor: onClick ? "pointer" : "default" }} />;
}
