/**
 * RESONATE — Keyboard Shortcut Overlay.
 * Shows all available shortcuts when ? is pressed.
 */

import { useTheme } from "../theme/ThemeProvider";
import { AF, MONO } from "../theme/fonts";

const SHORTCUTS = [
  { key: "Space", desc: "Play / Pause" },
  { key: "↑ / ↓", desc: "Navigate samples" },
  { key: "M", desc: "Toggle mix mode" },
  { key: "F", desc: "Favorite sample" },
  { key: "1-5", desc: "Rate sample (stars)" },
  { key: "S", desc: "Find similar samples" },
  { key: "L", desc: "Layer suggestions" },
  { key: "E", desc: "Save session" },
  { key: "R", desc: "New analysis" },
  { key: "Esc", desc: "Deselect / clear selection" },
  { key: "Cmd+F", desc: "Focus search" },
  { key: "Tab", desc: "Cycle categories" },
  { key: "?", desc: "Show shortcuts" },
];

export function ShortcutOverlay({ onClose }) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", backdropFilter: "blur(8px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1500 }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{ width: 340, background: theme.surface, borderRadius: 12, padding: "24px 24px 20px", border: "1px solid " + theme.border, boxShadow: "0 24px 48px rgba(0,0,0,0.15)" }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: theme.text, fontFamily: AF, marginBottom: 16 }}>Keyboard Shortcuts</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {SHORTCUTS.map(s => (
            <div key={s.key} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 11, color: theme.textSec, fontFamily: AF }}>{s.desc}</span>
              <kbd style={{ fontSize: 10, fontFamily: MONO, padding: "2px 8px", borderRadius: 4, background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)", color: theme.textMuted, border: "1px solid " + theme.borderLight }}>{s.key}</kbd>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 16, textAlign: "center" }}>
          <button onClick={onClose} style={{ fontSize: 10, color: theme.textMuted, background: "none", border: "none", cursor: "pointer", fontFamily: AF }}>Press any key to close</button>
        </div>
      </div>
    </div>
  );
}
