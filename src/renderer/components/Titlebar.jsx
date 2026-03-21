/**
 * RESONATE — Titlebar Component.
 * macOS window chrome, health status, theme toggle.
 */

import { useTheme } from "../theme/ThemeProvider";
import { AF, MONO, SERIF } from "../theme/fonts";

export function Titlebar({ backendOk, indexProgress, screen, onNew }) {
  const { theme, mode, toggleTheme } = useTheme();
  return (
    <div className="titlebar-drag" style={{ height: 46, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 16px", background: theme.surface, borderBottom: "1px solid " + theme.border, position: "sticky", top: 0, zIndex: 100, position: "relative" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 4, marginLeft: 70 }}>
        <div style={{ width: 5, height: 5, borderRadius: "50%", background: backendOk ? theme.green : theme.red }} />
        <span style={{ fontSize: 8, color: theme.textMuted }}>{backendOk ? "Online" : "Offline"}</span>
      </div>
      <span className="gradient-text" style={{ position: "absolute", left: "50%", transform: "translateX(-50%)", fontSize: 12, fontWeight: 600, letterSpacing: 3, fontFamily: AF }}>RESONATE</span>
      <div className="titlebar-no-drag" style={{ display: "flex", gap: 6, alignItems: "center" }}>
        {indexProgress && !indexProgress.done && (
          <span style={{ fontSize: 8, color: theme.textMuted, fontFamily: MONO }}>Indexing {indexProgress.processed}/{indexProgress.total}</span>
        )}
        {screen === "results" && <button onClick={onNew} style={{ background: "none", border: "none", color: theme.textSec, cursor: "pointer", fontSize: 11, fontFamily: SERIF }}>← New</button>}
        <button onClick={toggleTheme} title={mode === "dark" ? "Light mode" : "Dark mode"} style={{ background: "none", border: "none", color: theme.textMuted, cursor: "pointer", fontSize: 13, padding: "2px 6px" }}>
          {mode === "dark" ? "☀" : "☾"}
        </button>
      </div>
    </div>
  );
}
