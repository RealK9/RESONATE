/**
 * RESONATE — Settings Modal.
 * Configure sample library directory.
 */

import { useState } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { AF, MONO } from "../theme/fonts";

export function SettingsModal({ sampleDir, onSave, onClose }) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const [dir, setDir] = useState(sampleDir);

  const iStyle = { padding: "7px 9px", borderRadius: 5, border: "1px solid " + theme.border, background: isDark ? "#1E1E28" : "#fff", color: theme.text, fontSize: 11, outline: "none", fontFamily: "'DM Sans', sans-serif" };

  const save = async () => {
    try {
      await onSave(dir);
    } catch (e) {
      alert("Failed: " + e.message);
    }
  };

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", backdropFilter: "blur(8px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000 }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{ width: 460, background: theme.surface, borderRadius: 12, padding: "32px 28px", border: "1px solid " + theme.border, boxShadow: "0 24px 48px rgba(0,0,0,0.12)" }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: theme.text, fontFamily: AF, marginBottom: 4 }}>Settings</div>
        <div style={{ fontSize: 11, color: theme.textMuted, marginBottom: 20 }}>Configure your sample library location</div>
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 9, color: theme.textMuted, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 4, fontFamily: AF }}>Sample Library Path</div>
          <input value={dir} onChange={e => setDir(e.target.value)} style={{ ...iStyle, width: "100%", boxSizing: "border-box", fontSize: 12, fontFamily: MONO }} />
        </div>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button onClick={onClose} style={{ padding: "6px 16px", borderRadius: 6, border: "1px solid " + theme.border, background: theme.surface, color: theme.textSec, fontSize: 11, cursor: "pointer" }}>Cancel</button>
          <button onClick={save} style={{ padding: "6px 16px", borderRadius: 6, border: "none", background: isDark ? theme.text : "#1A1A1A", color: isDark ? "#0D0D12" : "#FFFFFF", fontSize: 11, cursor: "pointer", fontWeight: 600 }}>Save & Re-index</button>
        </div>
      </div>
    </div>
  );
}
