/**
 * RESONATE — Timestretch Panel.
 * ½x / 1x / 2x tempo buttons with BPM display.
 * Splice-style tempo control — syncs sample BPM to upload BPM at selected multiplier.
 */

import { memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { MONO, AF } from "../theme/fonts";

export const TimestretchPanel = memo(function TimestretchPanel({
  enabled, onToggle,
  bpmOptions, tempoMultiplier, onSelectMultiplier,
  uploadBpm,
}) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  return (
    <div style={{
      background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.015)",
      border: "1px solid " + theme.border,
      borderRadius: 8,
      padding: "10px 12px",
      opacity: enabled ? 1 : 0.45,
      transition: "opacity 0.2s ease",
    }}>
      {/* Header: title + toggle */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 10, fontWeight: 600, color: theme.text, fontFamily: AF, textTransform: "uppercase", letterSpacing: 1 }}>Timestretch</span>
          {enabled && uploadBpm > 0 && (
            <span style={{ fontSize: 9, fontFamily: MONO, color: "#06B6D4", fontWeight: 700 }}>{Math.round(uploadBpm)} BPM</span>
          )}
        </div>
        <button
          onClick={onToggle}
          style={{
            width: 32, height: 16, borderRadius: 8, border: "none", cursor: "pointer",
            background: enabled ? "#06B6D4" : (isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"),
            position: "relative", transition: "background 0.2s",
          }}
        >
          <div style={{
            width: 12, height: 12, borderRadius: "50%", background: "#fff",
            position: "absolute", top: 2,
            left: enabled ? 18 : 2,
            transition: "left 0.2s ease",
          }} />
        </button>
      </div>

      {/* ½x / 1x / 2x buttons */}
      <div style={{ display: "flex", gap: 4 }}>
        {bpmOptions.map(opt => {
          const isActive = tempoMultiplier === opt.mult;
          return (
            <button
              key={opt.mult}
              onClick={() => enabled && onSelectMultiplier(opt.mult)}
              style={{
                flex: 1, padding: "8px 4px", borderRadius: 6,
                border: isActive ? "1px solid rgba(6,182,212,0.4)" : "1px solid " + theme.borderLight,
                background: isActive
                  ? (isDark ? "rgba(6,182,212,0.12)" : "rgba(6,182,212,0.08)")
                  : "transparent",
                cursor: enabled ? "pointer" : "default",
                transition: "all 0.15s",
                display: "flex", flexDirection: "column", alignItems: "center", gap: 2,
              }}
            >
              <span style={{
                fontSize: 11, fontWeight: 700, fontFamily: MONO,
                color: isActive ? "#06B6D4" : theme.textSec,
              }}>{opt.label}</span>
              {opt.bpm > 0 && (
                <span style={{
                  fontSize: 8, fontFamily: MONO,
                  color: isActive ? "rgba(6,182,212,0.7)" : theme.textMuted,
                }}>{opt.bpm}</span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
});
