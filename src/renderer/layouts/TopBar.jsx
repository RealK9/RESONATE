/**
 * RESONATE — Top Bar.
 * Central search bar (Splice's most prominent element) + theme toggle + settings.
 */

import { memo, useRef, useEffect } from "react";
import { useRouter } from "../router";
import { useTheme } from "../theme/ThemeProvider";
import { AF, MONO } from "../theme/fonts";

export const TopBar = memo(function TopBar({ search, onSearchChange, onSettingsClick, bridgeConnected, backendOk, indexProgress }) {
  const { page } = useRouter();
  const isHome = page === "home";
  const { theme, mode, toggleTheme } = useTheme();
  const isDark = mode === "dark";
  const inputRef = useRef(null);

  // Global Cmd+F / Ctrl+F → focus search
  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "f") {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === "/" && !e.metaKey && !e.ctrlKey && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div style={{
      height: 46, minHeight: 46,
      display: "flex", alignItems: "center", gap: 10,
      padding: "0 16px",
      background: isDark ? "#0A0A10" : "#F5F3EF",
      borderBottom: "1px solid " + theme.border,
      WebkitAppRegion: "drag",
    }}>
      {/* Spacer for macOS traffic lights */}
      <div style={{ width: 68, WebkitAppRegion: "drag" }} />

      {/* AI Online indicator */}
      <div style={{
        display: "flex", alignItems: "center", gap: 5,
        flexShrink: 0, WebkitAppRegion: "no-drag",
      }}>
        <div style={{
          width: 6, height: 6, borderRadius: "50%",
          background: backendOk ? "#22C55E" : "#EF4444",
          boxShadow: backendOk ? "0 0 6px rgba(34,197,94,0.4)" : "0 0 6px rgba(239,68,68,0.4)",
          transition: "background 0.3s, box-shadow 0.3s",
        }} />
        <span style={{
          fontSize: 8, fontWeight: 600, fontFamily: AF, letterSpacing: 0.5,
          color: backendOk ? (isDark ? "rgba(34,197,94,0.8)" : "#16A34A") : (isDark ? "rgba(239,68,68,0.8)" : "#DC2626"),
          transition: "color 0.3s",
        }}>
          {backendOk ? "AI Online" : "Offline"}
        </span>
        {indexProgress && !indexProgress.done && (
          <span style={{ fontSize: 7, color: theme.textMuted, fontFamily: MONO }}>
            {indexProgress.processed}/{indexProgress.total}
          </span>
        )}
      </div>

      {/* Center: RESONATE title on home, search bar on other pages */}
      {isHome ? (
        <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
          <span className="gradient-text shimmer-text" style={{
            fontSize: 12, fontWeight: 600, letterSpacing: 3, fontFamily: AF,
          }}>RESONATE</span>
        </div>
      ) : (
        <div style={{
          flex: 1, maxWidth: 520, margin: "0 auto",
          display: "flex", alignItems: "center", gap: 8,
          padding: "6px 12px", borderRadius: 8,
          background: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)",
          border: "1px solid " + theme.borderLight,
          WebkitAppRegion: "no-drag",
          transition: "border-color 0.15s",
        }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={theme.textMuted} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
          </svg>
          <input ref={inputRef} type="text" value={search} onChange={e => onSearchChange(e.target.value)}
            placeholder="Search sounds..."
            style={{
              flex: 1, background: "transparent", border: "none", outline: "none",
              color: theme.text, fontSize: 13, fontFamily: AF,
            }}
          />
          {!search && (
            <kbd style={{
              fontSize: 9, padding: "1px 5px", borderRadius: 3,
              background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)",
              color: theme.textMuted, fontFamily: MONO,
            }}>/</kbd>
          )}
          {search && (
            <button onClick={() => onSearchChange("")} style={{
              background: "none", border: "none", color: theme.textMuted, cursor: "pointer",
              fontSize: 12, padding: "0 2px", lineHeight: 1,
            }}>&times;</button>
          )}
        </div>
      )}

      {/* Right controls */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, WebkitAppRegion: "no-drag" }}>
        {bridgeConnected && (
          <div style={{
            display: "flex", alignItems: "center", gap: 4,
            padding: "3px 8px", borderRadius: 4,
            background: isDark ? "rgba(6,182,212,0.08)" : "rgba(6,182,212,0.06)",
          }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%", background: "#06B6D4",
              boxShadow: "0 0 6px rgba(6,182,212,0.4)",
            }} />
            <span style={{ fontSize: 8, color: "#06B6D4", fontWeight: 600, fontFamily: AF, letterSpacing: 0.5 }}>BRIDGE</span>
          </div>
        )}
        <button onClick={toggleTheme}
          style={{ background: "none", border: "none", cursor: "pointer", fontSize: 14, color: theme.textMuted, padding: 4 }}>
          {isDark ? "\u2600" : "\u263E"}
        </button>
        <button onClick={onSettingsClick}
          style={{ background: "none", border: "none", cursor: "pointer", color: theme.textMuted, padding: 4 }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
          </svg>
        </button>
      </div>
    </div>
  );
});
