/**
 * RESONATE — Skeleton Loading Row.
 */

import { useTheme } from "../theme/ThemeProvider";

export function SkeletonRow() {
  const { mode } = useTheme();
  const isDark = mode === "dark";
  const bg1 = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)";
  const bg2 = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)";
  return (
    <div style={{ display: "grid", gridTemplateColumns: "30px 1fr 54px 40px 40px 42px", gap: 6, padding: "11px 14px", borderBottom: isDark ? "1px solid rgba(255,255,255,0.04)" : "1px solid rgba(0,0,0,0.04)" }}>
      <div style={{ width: 11, height: 11, borderRadius: 3, background: bg1, animation: "pulse 1.5s ease infinite" }} />
      <div>
        <div style={{ width: "70%", height: 10, borderRadius: 3, background: bg1, marginBottom: 4, animation: "pulse 1.5s ease infinite" }} />
        <div style={{ width: "40%", height: 8, borderRadius: 3, background: bg2, animation: "pulse 1.5s ease infinite" }} />
      </div>
      <div style={{ width: 32, height: 10, borderRadius: 3, background: bg1, animation: "pulse 1.5s ease infinite" }} />
      <div style={{ width: 20, height: 10, borderRadius: 3, background: bg2, animation: "pulse 1.5s ease infinite" }} />
      <div style={{ width: 20, height: 10, borderRadius: 3, background: bg2, animation: "pulse 1.5s ease infinite" }} />
      <div style={{ width: 24, height: 10, borderRadius: 3, background: bg2, animation: "pulse 1.5s ease infinite" }} />
    </div>
  );
}
