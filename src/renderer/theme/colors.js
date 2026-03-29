/**
 * RESONATE — Color System.
 * Light and dark theme palettes with brand gradient (purple → cyan).
 */

// Brand gradient colors (matches logo)
const BRAND = {
  purple: "#8B5CF6",
  cyan: "#06B6D4",
  gradient: "linear-gradient(135deg, #8B5CF6, #06B6D4)",
  gradientHover: "linear-gradient(135deg, #A78BFA, #22D3EE)",
  gradientSubtle: "linear-gradient(135deg, rgba(139,92,246,0.12), rgba(6,182,212,0.12))",
  gradientSubtleDark: "linear-gradient(135deg, rgba(139,92,246,0.08), rgba(6,182,212,0.08))",
  glow: "0 0 20px rgba(139,92,246,0.15), 0 0 40px rgba(6,182,212,0.1)",
  glowStrong: "0 0 24px rgba(139,92,246,0.25), 0 0 48px rgba(6,182,212,0.15)",
};

export const brand = BRAND;

export const lightTheme = {
  bg: "#F5F3EF",
  surface: "#FFFFFF",
  surfaceHover: "rgba(0,0,0,0.018)",
  border: "rgba(0,0,0,0.07)",
  borderLight: "rgba(0,0,0,0.04)",
  text: "#1A1A1A",
  textSec: "#555555",
  textMuted: "#999999",
  textFaint: "#BBBBBB",
  tag: "rgba(0,0,0,0.05)",
  tagText: "#666666",
  red: "#DC2626",
  green: "#16A34A",
  accent: BRAND.purple,
  accentSec: BRAND.cyan,
  surfaceActive: "rgba(139,92,246,0.06)",
  gradient: BRAND.gradient,
  gradientHover: BRAND.gradientHover,
  gradientSubtle: BRAND.gradientSubtle,
  brand: BRAND,
  // Player
  waveformActive: BRAND.purple,
  waveformInactive: "#D4D0CC",
  waveformActiveAlpha: 0.85,
  waveformInactiveAlpha: 0.35,
  // Spectrum
  spectrumTrack: "rgba(139,92,246,0.2)",
  spectrumSample: "rgba(6,182,212,0.25)",
  spectrumGap: "rgba(239,68,68,0.2)",
  // Scrollbar
  scrollbarThumb: "rgba(0,0,0,0.08)",
  scrollbarThumbHover: "rgba(0,0,0,0.15)",
};

export const darkTheme = {
  bg: "#0A0A10",
  surface: "#12121A",
  surfaceHover: "rgba(255,255,255,0.03)",
  border: "rgba(255,255,255,0.06)",
  borderLight: "rgba(255,255,255,0.04)",
  text: "#E8E6E3",
  textSec: "#A0A0A0",
  textMuted: "#606068",
  textFaint: "#3A3A42",
  tag: "rgba(255,255,255,0.06)",
  tagText: "#909098",
  red: "#EF4444",
  green: "#22C55E",
  accent: BRAND.purple,
  accentSec: BRAND.cyan,
  surfaceActive: "rgba(139,92,246,0.08)",
  gradient: BRAND.gradient,
  gradientHover: BRAND.gradientHover,
  gradientSubtle: BRAND.gradientSubtleDark,
  brand: BRAND,
  // Player
  waveformActive: BRAND.purple,
  waveformInactive: "#2A2A32",
  waveformActiveAlpha: 0.85,
  waveformInactiveAlpha: 0.35,
  // Spectrum
  spectrumTrack: "rgba(139,92,246,0.18)",
  spectrumSample: "rgba(6,182,212,0.22)",
  spectrumGap: "rgba(239,68,68,0.25)",
  // Scrollbar
  scrollbarThumb: "rgba(255,255,255,0.08)",
  scrollbarThumbHover: "rgba(255,255,255,0.14)",
};
