/**
 * RESONATE — Reusable Modal Component.
 * Glass morphism backdrop with brand-accented panel.
 */

import { useTheme } from "../theme/ThemeProvider";

export function Modal({ visible, onClose, width = 480, children }) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  if (!visible) return null;
  return (
    <div style={{ position: "fixed", inset: 0, background: isDark ? "rgba(0,0,0,0.5)" : "rgba(0,0,0,0.25)", backdropFilter: "blur(12px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, animation: "fadeIn 0.15s ease" }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{ width, maxHeight: 500, background: isDark ? "rgba(18,18,26,0.95)" : "rgba(255,255,255,0.95)", borderRadius: 14, padding: "24px", border: "1px solid " + theme.border, boxShadow: isDark ? "0 24px 48px rgba(0,0,0,0.4), 0 0 1px rgba(139,92,246,0.1)" : "0 24px 48px rgba(0,0,0,0.1), 0 0 1px rgba(139,92,246,0.08)", overflow: "hidden", display: "flex", flexDirection: "column", backdropFilter: "blur(20px)", animation: "fadeInUp 0.2s ease" }}>
        {children}
      </div>
    </div>
  );
}
