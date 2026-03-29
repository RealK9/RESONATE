/**
 * RESONATE — Titlebar Component.
 * Frosted glass macOS window chrome with status beacon, theme toggle, gradient accents.
 */

import { useState } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { AF, MONO, SERIF } from "../theme/fonts";

export function Titlebar({ backendOk, indexProgress, screen, onNew }) {
  const { theme, mode, toggleTheme } = useTheme();
  const isDark = mode === "dark";
  const [themeHover, setThemeHover] = useState(false);
  const [newHover, setNewHover] = useState(false);

  return (
    <div className="titlebar-drag titlebar-glass" style={{
      height: 46,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "0 16px",
      background: isDark
        ? "rgba(10, 10, 16, 0.75)"
        : "rgba(245, 243, 239, 0.8)",
      position: "sticky",
      top: 0,
      zIndex: 100,
    }}>
      {/* Status beacon */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 70 }}>
        <div style={{
          width: 6, height: 6, borderRadius: "50%",
          background: backendOk ? theme.green : theme.red,
          boxShadow: backendOk
            ? "0 0 6px rgba(34,197,94,0.4), 0 0 12px rgba(34,197,94,0.15)"
            : "0 0 6px rgba(220,38,38,0.4)",
          transition: "all 0.4s ease",
        }} />
        <span style={{
          fontSize: 9, color: theme.textMuted, fontFamily: MONO, fontWeight: 500,
          letterSpacing: 1, textTransform: "uppercase",
          transition: "color 0.3s ease",
        }}>
          {backendOk ? "Online" : "Offline"}
        </span>
      </div>

      {/* Center: RESONATE branding */}
      <span className="gradient-text" style={{
        position: "absolute", left: "50%", transform: "translateX(-50%)",
        fontSize: 12, fontWeight: 600, letterSpacing: 4, fontFamily: AF,
        transition: "letter-spacing 0.3s ease",
      }}>
        RESONATE
      </span>

      {/* Right: controls */}
      <div className="titlebar-no-drag" style={{ display: "flex", gap: 4, alignItems: "center" }}>
        {/* Index progress indicator */}
        {indexProgress && !indexProgress.done && (
          <div style={{
            display: "flex", alignItems: "center", gap: 5,
            padding: "3px 8px", borderRadius: 6,
            background: isDark ? "rgba(217,70,239,0.06)" : "rgba(217,70,239,0.04)",
            border: "1px solid rgba(217,70,239,0.1)",
          }}>
            <svg width="10" height="10" viewBox="0 0 16 16" style={{ animation: "spinAnalyze 1.5s linear infinite", flexShrink: 0 }}>
              <circle cx="8" cy="8" r="6" fill="none" stroke="#D946EF" strokeWidth="1.5" strokeDasharray="28 10" strokeLinecap="round" opacity="0.6" />
            </svg>
            <span style={{ fontSize: 8, color: theme.textMuted, fontFamily: MONO }}>
              {indexProgress.processed}/{indexProgress.total}
            </span>
          </div>
        )}

        {/* New analysis button */}
        {screen === "results" && (
          <button
            onClick={onNew}
            onMouseEnter={() => setNewHover(true)}
            onMouseLeave={() => setNewHover(false)}
            style={{
              background: newHover
                ? (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)")
                : "transparent",
              border: "1px solid " + (newHover ? theme.border : "transparent"),
              borderRadius: 6,
              color: newHover ? theme.text : theme.textSec,
              cursor: "pointer",
              fontSize: 10,
              fontFamily: AF,
              fontWeight: 500,
              padding: "4px 10px",
              transition: "all 0.2s ease",
              display: "flex",
              alignItems: "center",
              gap: 4,
            }}
          >
            <svg width="10" height="10" viewBox="0 0 16 16" fill="none" style={{ transition: "transform 0.2s ease", transform: newHover ? "translateX(-1px)" : "none" }}>
              <path d="M10 2L4 8l6 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            New
          </button>
        )}

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          onMouseEnter={() => setThemeHover(true)}
          onMouseLeave={() => setThemeHover(false)}
          title={isDark ? "Light mode" : "Dark mode"}
          style={{
            width: 28, height: 28,
            borderRadius: 7,
            border: "1px solid " + (themeHover ? theme.border : "transparent"),
            background: themeHover
              ? (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)")
              : "transparent",
            color: themeHover ? theme.text : theme.textMuted,
            cursor: "pointer",
            fontSize: 13,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "all 0.25s ease",
            transform: themeHover ? "rotate(15deg)" : "rotate(0deg)",
          }}
        >
          {isDark ? "\u2600" : "\u263E"}
        </button>
      </div>
    </div>
  );
}
