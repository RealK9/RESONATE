/**
 * RESONATE — Analysis Progress Ring.
 * Animated circular progress indicator during track analysis.
 */

import { useTheme } from "../theme/ThemeProvider";
import { SERIF, AF } from "../theme/fonts";

export function AnalysisRing({ progress, size = 180 }) {
  const { theme } = useTheme();
  const r = (size - 14) / 2, circ = 2 * Math.PI * r, off = circ - (progress / 100) * circ;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={theme.borderLight} strokeWidth={6} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="url(#rg)" strokeWidth={6} strokeDasharray={circ} strokeDashoffset={off} strokeLinecap="round" style={{ transition: "stroke-dashoffset 0.25s" }} />
        <defs><linearGradient id="rg" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stopColor="#D946EF" /><stop offset="50%" stopColor="#8B5CF6" /><stop offset="100%" stopColor="#06B6D4" /></linearGradient></defs>
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 36, fontWeight: 300, color: theme.text, fontFamily: SERIF, letterSpacing: -1 }}>{progress}<span style={{ fontSize: 16, opacity: 0.4 }}>%</span></span>
        <span style={{ fontSize: 9, color: theme.textMuted, textTransform: "uppercase", letterSpacing: 3, marginTop: 3, fontFamily: AF }}>Analyzing</span>
      </div>
    </div>
  );
}
