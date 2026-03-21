/**
 * RESONATE — Volume Slider Component.
 */

import { useTheme } from "../theme/ThemeProvider";
import { AF, MONO } from "../theme/fonts";

export function VolumeSlider({ value, onChange, label }) {
  const { theme } = useTheme();
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 9, fontFamily: AF }}>
      <span style={{ color: theme.textMuted, width: 38 }}>{label}</span>
      <input type="range" min="0" max="100" value={Math.round(value * 100)} onChange={e => onChange(parseInt(e.target.value) / 100)} style={{ flex: 1, height: 2, accentColor: theme.textSec, cursor: "pointer" }} />
      <span style={{ color: theme.textMuted, fontFamily: MONO, width: 24, textAlign: "right" }}>{Math.round(value * 100)}</span>
    </div>
  );
}
