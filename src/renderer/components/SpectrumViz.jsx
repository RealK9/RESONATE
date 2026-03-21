/**
 * RESONATE — Spectrum Visualization.
 * 7-band frequency comparison (track vs sample vs genre ideal).
 */

import { useTheme } from "../theme/ThemeProvider";

const BL = ["Sub", "Bass", "Lo-Mid", "Mid", "Hi-Mid", "Pres", "Air"];
const BK = ["sub_bass_20_80", "bass_80_250", "low_mid_250_500", "mid_500_2k", "upper_mid_2k_6k", "presence_6k_12k", "air_12k_20k"];

export function SpectrumViz({ trackBands, sampleBands, gaps, height = 80 }) {
  const { theme } = useTheme();
  if (!trackBands || !Object.keys(trackBands).length) return null;
  const mx = Math.max(...BK.map(k => trackBands[k] || 0), 0.25);
  return (
    <div style={{ padding: "8px 0" }}>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height, marginBottom: 4 }}>
        {BK.map(band => {
          const tv = trackBands[band] || 0, sv = sampleBands ? (sampleBands[band] || 0) : 0;
          const tH = (tv / mx) * height * 0.85, sH = sampleBands ? (sv / mx) * height * 0.85 : 0;
          const isGap = gaps && gaps.some(g => (g === "midrange_melody" && band === "mid_500_2k") || (g === "upper_mid_presence" && band === "upper_mid_2k_6k") || (g === "high_end_sparkle" && band === "presence_6k_12k") || (g === "low_mid_warmth" && band === "low_mid_250_500") || (g === "air" && band === "air_12k_20k"));
          return (
            <div key={band} style={{ flex: 1, position: "relative", height: "100%", display: "flex", alignItems: "flex-end" }}>
              {sampleBands && sH > 0 && <div style={{ position: "absolute", bottom: 0, width: "100%", height: Math.max(sH, 2), borderRadius: "2px 2px 0 0", background: theme.spectrumSample, transition: "height 0.3s" }} />}
              <div style={{ width: "100%", height: Math.max(tH, 2), borderRadius: "2px 2px 0 0", background: isGap ? theme.spectrumGap : theme.spectrumTrack, transition: "height 0.3s", position: "relative", zIndex: 1 }} />
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 2 }}>{BL.map(l => <div key={l} style={{ flex: 1, textAlign: "center", fontSize: 7, color: theme.textFaint }}>{l}</div>)}</div>
    </div>
  );
}
