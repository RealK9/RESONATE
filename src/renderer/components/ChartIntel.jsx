/**
 * RESONATE — Chart Intelligence 2.0 Panel.
 * Shows how the producer's mix compares to charting music:
 * BPM bar, energy/valence/danceability dots, text insights, decade trends.
 */

import { useTheme } from "../theme/ThemeProvider";
import { AF, MONO, SERIF } from "../theme/fonts";

// ── Comparison dot row ──────────────────────────────────────────────────────
function ComparisonDot({ label, yours, chart, isDark }) {
  const yVal = yours != null ? yours : 0;
  const cVal = chart != null ? chart : 0;
  const barBg = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)";
  const yourColor = "#A855F7";
  const chartColor = isDark ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.25)";
  const { theme } = useTheme();

  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
        <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF }}>{label}</span>
        <span style={{ fontSize: 9, color: theme.text, fontFamily: MONO, fontWeight: 600 }}>
          {yVal.toFixed(2)}
          <span style={{ color: theme.textFaint }}> / {cVal.toFixed(2)}</span>
        </span>
      </div>
      <div style={{ position: "relative", height: 6, borderRadius: 3, background: barBg, overflow: "visible" }}>
        {/* Chart average marker */}
        <div style={{
          position: "absolute",
          left: `${cVal * 100}%`,
          top: -1,
          width: 2,
          height: 8,
          borderRadius: 1,
          background: chartColor,
          transform: "translateX(-50%)",
          zIndex: 1,
        }} />
        {/* Your value fill */}
        <div style={{
          height: "100%",
          borderRadius: 3,
          background: `linear-gradient(90deg, ${yourColor}44, ${yourColor})`,
          width: `${Math.min(yVal * 100, 100)}%`,
          transition: "width 0.4s ease",
        }} />
        {/* Your value dot */}
        <div style={{
          position: "absolute",
          left: `${Math.min(yVal * 100, 100)}%`,
          top: -2,
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: yourColor,
          border: `2px solid ${isDark ? "#1a1a1a" : "#fff"}`,
          transform: "translateX(-50%)",
          zIndex: 2,
          boxShadow: `0 0 6px ${yourColor}66`,
        }} />
      </div>
    </div>
  );
}

// ── BPM comparison bar ──────────────────────────────────────────────────────
function BpmBar({ yourBpm, avgBpm, stdBpm, isDark }) {
  const { theme } = useTheme();
  if (!yourBpm || !avgBpm) return null;

  const lo = Math.max(avgBpm - stdBpm * 1.5, 60);
  const hi = avgBpm + stdBpm * 1.5;
  const range = hi - lo;
  const yourPct = Math.max(0, Math.min(100, ((yourBpm - lo) / range) * 100));
  const avgPct = ((avgBpm - lo) / range) * 100;
  const stdPctHalf = ((stdBpm) / range) * 100;

  const barBg = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)";
  const accentColor = "#06B6D4";

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF }}>BPM</span>
        <span style={{ fontSize: 9, color: theme.text, fontFamily: MONO, fontWeight: 600 }}>
          {yourBpm.toFixed(0)}
          <span style={{ color: theme.textFaint }}> / {avgBpm.toFixed(0)} avg</span>
        </span>
      </div>
      <div style={{ position: "relative", height: 8, borderRadius: 4, background: barBg, overflow: "visible" }}>
        {/* Genre average range band */}
        <div style={{
          position: "absolute",
          left: `${Math.max(0, avgPct - stdPctHalf)}%`,
          width: `${Math.min(100 - Math.max(0, avgPct - stdPctHalf), stdPctHalf * 2)}%`,
          height: "100%",
          borderRadius: 4,
          background: isDark ? "rgba(6,182,212,0.15)" : "rgba(6,182,212,0.12)",
        }} />
        {/* Average marker */}
        <div style={{
          position: "absolute",
          left: `${avgPct}%`,
          top: -2,
          width: 2,
          height: 12,
          borderRadius: 1,
          background: isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.2)",
          transform: "translateX(-50%)",
          zIndex: 1,
        }} />
        {/* Your BPM marker */}
        <div style={{
          position: "absolute",
          left: `${yourPct}%`,
          top: -3,
          width: 14,
          height: 14,
          borderRadius: "50%",
          background: accentColor,
          border: `2px solid ${isDark ? "#1a1a1a" : "#fff"}`,
          transform: "translateX(-50%)",
          zIndex: 2,
          boxShadow: `0 0 8px ${accentColor}66`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2 }}>
        <span style={{ fontSize: 7, color: theme.textFaint, fontFamily: MONO }}>{lo.toFixed(0)}</span>
        <span style={{ fontSize: 7, color: theme.textFaint, fontFamily: MONO }}>{hi.toFixed(0)}</span>
      </div>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────
export function ChartIntel({ data }) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  if (!data) return null;

  const { your_mix, chart_average, insights, decade_trends } = data;

  const lbl = {
    fontSize: 10,
    fontWeight: 700,
    color: theme.text,
    fontFamily: AF,
    textTransform: "uppercase",
    letterSpacing: 1.2,
    marginBottom: 8,
  };

  return (
    <div style={{
      padding: 12,
      borderRadius: 8,
      marginBottom: 12,
      background: theme.bg,
      border: "1px solid " + theme.borderLight,
    }}>
      {/* Header */}
      <div style={lbl}>
        <span style={{ fontSize: 11 }}>Your Mix vs Charts</span>
        {chart_average?.genre && (
          <span style={{
            fontSize: 8,
            fontWeight: 500,
            color: theme.textMuted,
            textTransform: "capitalize",
            marginLeft: 6,
            letterSpacing: 0.5,
          }}>
            {chart_average.genre}
          </span>
        )}
      </div>

      {/* BPM bar */}
      <BpmBar
        yourBpm={your_mix?.bpm}
        avgBpm={chart_average?.bpm_mean}
        stdBpm={chart_average?.bpm_std || 15}
        isDark={isDark}
      />

      {/* Energy / Valence / Danceability dots */}
      {your_mix?.energy != null && chart_average?.energy_mean != null && (
        <ComparisonDot label="Energy" yours={your_mix.energy} chart={chart_average.energy_mean} isDark={isDark} />
      )}
      {your_mix?.valence != null && chart_average?.valence_mean != null && (
        <ComparisonDot label="Valence" yours={your_mix.valence} chart={chart_average.valence_mean} isDark={isDark} />
      )}
      {your_mix?.danceability != null && chart_average?.danceability_mean != null && (
        <ComparisonDot label="Danceability" yours={your_mix.danceability} chart={chart_average.danceability_mean} isDark={isDark} />
      )}

      {/* Insights */}
      {insights && insights.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 8, color: theme.textFaint, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 4, fontFamily: AF }}>
            Insights
          </div>
          {insights.map((insight, i) => (
            <div key={i} style={{
              fontSize: 9,
              color: theme.textSec,
              fontFamily: AF,
              lineHeight: 1.4,
              padding: "5px 8px",
              marginBottom: 3,
              borderRadius: 5,
              background: isDark ? "rgba(168,85,247,0.06)" : "rgba(168,85,247,0.05)",
              borderLeft: "2px solid rgba(168,85,247,0.3)",
            }}>
              {insight}
            </div>
          ))}
        </div>
      )}

      {/* Decade trends */}
      {decade_trends && decade_trends.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 8, color: theme.textFaint, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 4, fontFamily: AF }}>
            Decade Trends
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {decade_trends.map((d) => (
              <div key={d.decade} style={{
                flex: 1,
                padding: "5px 4px",
                borderRadius: 5,
                background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)",
                textAlign: "center",
              }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: theme.text, fontFamily: MONO, marginBottom: 2 }}>
                  {d.decade}s
                </div>
                <div style={{ fontSize: 8, color: theme.textMuted, fontFamily: MONO }}>
                  {d.bpm_mean.toFixed(0)} bpm
                </div>
                <div style={{ fontSize: 8, color: theme.textMuted, fontFamily: MONO }}>
                  E {d.energy_mean.toFixed(2)}
                </div>
                <div style={{ fontSize: 8, color: theme.textMuted, fontFamily: MONO }}>
                  V {d.valence_mean.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Track count badge */}
      {chart_average?.count > 0 && (
        <div style={{
          marginTop: 8,
          fontSize: 8,
          color: theme.textFaint,
          fontFamily: AF,
          textAlign: "center",
        }}>
          Based on {chart_average.count.toLocaleString()} charting tracks
        </div>
      )}
    </div>
  );
}
