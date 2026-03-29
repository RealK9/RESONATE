/**
 * AnalysisSidebar — Compact single-page-view analysis panel.
 * All info visible without scrolling: analysis card, mix needs, gap analysis,
 * chart intel, bridge status. Condensed layout with tight spacing.
 */

import { useState, memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { SERIF, AF, MONO } from "../theme/fonts";
import { NEED_CATEGORY_COLORS, POLICY_LABELS } from "../utils/v2Adapter";
import { ChartIntel } from "./ChartIntel";
import { SpectrumViz } from "./SpectrumViz";

export const AnalysisSidebar = memo(function AnalysisSidebar({
  analysisResult,
  fileName,
  analyzerExpanded,
  setAnalyzerExpanded,
  mixNeeds,
  gapAnalysis,
  ringGlowing,
  prevReadiness,
  chartComparison,
  bridge,
  reanalyzing,
  reAnalyze,
  saveSession,
  sessions,
  setShowSessions,
  activeSample,
  analyzeFromBridge,
}) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const [needsExpanded, setNeedsExpanded] = useState(false);

  const a = analysisResult?.analysis || {};

  const lbl = { fontSize: 7, color: theme.textMuted, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 3, fontWeight: 600, fontFamily: AF };
  const card = { padding: "8px 10px", borderRadius: 6, marginBottom: 6, background: theme.bg, border: "1px solid " + theme.borderLight };

  return (
    <>
      {/* ── Analysis Card: Key / BPM / Genre ── */}
      <div style={card}>
        <div style={{ fontSize: 8, color: theme.textMuted, marginBottom: 4, fontFamily: MONO, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{fileName}</div>
        <div style={{ display: "flex", gap: 6, marginBottom: 4 }}>
          {[["Key", a.key], ["BPM", a.bpm ? Math.round(a.bpm) : "\u2014"], ["Genre", a.genre]].map(([k, v]) => (
            <div key={k} style={{ flex: k === "Genre" ? 2 : 1, textAlign: "center" }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: theme.text, fontFamily: AF, lineHeight: 1.1 }}>{v || "\u2014"}</div>
              <div style={{ fontSize: 6, color: theme.textFaint, fontFamily: AF, textTransform: "uppercase", letterSpacing: 1.5, marginTop: 1 }}>{k}</div>
            </div>
          ))}
        </div>
        {/* More Details toggle */}
        <button onClick={() => setAnalyzerExpanded(!analyzerExpanded)} style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 3, padding: "2px 0", border: "none", background: "transparent", color: theme.textMuted, fontSize: 7, fontFamily: AF, cursor: "pointer", letterSpacing: 1, textTransform: "uppercase" }}>
          {analyzerExpanded ? "Less" : "Details"}
          <svg width="7" height="7" viewBox="0 0 8 8" style={{ transition: "transform 0.2s", transform: analyzerExpanded ? "rotate(180deg)" : "rotate(0)" }}>
            <path d="M1 2.5l3 3 3-3" stroke="currentColor" strokeWidth="1.2" fill="none" strokeLinecap="round" />
          </svg>
        </button>
        {analyzerExpanded && (
          <div style={{ marginTop: 4, paddingTop: 4, borderTop: "1px solid " + theme.borderLight }}>
            {[["Energy", a.energy_label], ["Mood", a.mood], ["Duration", a.duration ? `${Math.floor(a.duration / 60)}:${String(Math.floor(a.duration % 60)).padStart(2, "0")}` : "\u2014"]].map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF }}>{k}</span>
                <span style={{ fontSize: 9, color: theme.text, fontWeight: 600, fontFamily: AF }}>{v || "\u2014"}</span>
              </div>
            ))}
            {a.detected_instruments?.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 2, marginTop: 3 }}>
                {a.detected_instruments.map((inst) => (
                  <span key={inst} style={{ fontSize: 8, padding: "1px 5px", borderRadius: 3, background: isDark ? "rgba(139,92,246,0.15)" : "rgba(139,92,246,0.1)", color: isDark ? "#A78BFA" : "#7C3AED", fontFamily: AF, fontWeight: 500 }}>{inst}</span>
                ))}
              </div>
            )}
          </div>
        )}
        {/* Actions row */}
        <div style={{ display: "flex", gap: 3, marginTop: 5 }}>
          <button onClick={saveSession} style={{ flex: 1, fontSize: 7, padding: "3px 0", borderRadius: 3, border: "1px solid " + theme.border, background: "transparent", color: theme.textSec, cursor: "pointer", fontFamily: AF }}>Save</button>
          {sessions.length > 0 && <button onClick={() => setShowSessions(true)} style={{ flex: 1, fontSize: 7, padding: "3px 0", borderRadius: 3, border: "1px solid " + theme.border, background: "transparent", color: theme.textSec, cursor: "pointer", fontFamily: AF }}>History</button>}
        </div>
        {bridge.connected && (
          <button onClick={reAnalyze} disabled={reanalyzing} style={{ width: "100%", fontSize: 8, padding: "4px 0", marginTop: 4, borderRadius: 4, border: "1px solid rgba(34,197,94,0.3)", background: reanalyzing ? "rgba(34,197,94,0.03)" : "rgba(34,197,94,0.08)", color: reanalyzing ? theme.textMuted : "#22C55E", cursor: reanalyzing ? "default" : "pointer", fontWeight: 600, fontFamily: AF, display: "flex", alignItems: "center", justifyContent: "center", gap: 4 }}>
            {reanalyzing && (
              <svg width="8" height="8" viewBox="0 0 16 16" style={{ animation: "spinAnalyze 0.8s linear infinite", flexShrink: 0 }}>
                <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="28 10" strokeLinecap="round" />
              </svg>
            )}
            {reanalyzing ? "Re-Analyzing..." : "Re-Analyze DAW"}
          </button>
        )}
      </div>

      {/* ── Production Readiness (compact) ── */}
      {gapAnalysis && (
        <div style={card}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            {/* Score ring — smaller */}
            <div style={{ position: "relative", width: 36, height: 36, flexShrink: 0, animation: ringGlowing ? "ringGlow 1.2s ease-out" : "none" }}>
              <svg width="36" height="36" viewBox="0 0 36 36">
                <circle cx="18" cy="18" r="14" fill="none" stroke={isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"} strokeWidth="2.5" />
                <circle cx="18" cy="18" r="14" fill="none" stroke={gapAnalysis.readinessColor} strokeWidth="2.5" strokeDasharray={`${gapAnalysis.readiness * 0.88} 88`} strokeLinecap="round" transform="rotate(-90 18 18)" style={{ transition: "stroke-dasharray 0.6s ease" }} />
              </svg>
              <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, fontFamily: MONO, color: gapAnalysis.readinessColor }}>
                {gapAnalysis.readiness}
              </div>
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: gapAnalysis.readinessColor, fontFamily: AF }}>{gapAnalysis.readinessTier}</div>
              <div style={{ fontSize: 8, color: theme.textMuted, fontFamily: AF }}>{gapAnalysis.genre}</div>
            </div>
          </div>
          {/* Compact stats row */}
          <div style={{ display: "flex", gap: 8, marginBottom: 4 }}>
            {gapAnalysis.genreCoherence > 0 && (
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 7, color: theme.textFaint, fontFamily: AF, marginBottom: 2 }}>Coherence</div>
                <div style={{ height: 3, borderRadius: 1.5, background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)", overflow: "hidden" }}>
                  <div style={{ height: "100%", borderRadius: 1.5, background: gapAnalysis.genreCoherence >= 75 ? "#22C55E" : gapAnalysis.genreCoherence >= 50 ? "#F59E0B" : "#EF4444", width: `${gapAnalysis.genreCoherence}%`, transition: "width 0.4s ease" }} />
                </div>
              </div>
            )}
            {gapAnalysis.chartPotentialCeiling > 0 && (
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 7, color: theme.textFaint, fontFamily: AF, marginBottom: 2 }}>Chart</div>
                <div style={{ height: 3, borderRadius: 1.5, background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)", overflow: "hidden" }}>
                  <div style={{ height: "100%", borderRadius: 1.5, background: "linear-gradient(90deg, #8B5CF6, #06B6D4)", width: `${gapAnalysis.chartPotentialCeiling}%`, transition: "width 0.4s ease" }} />
                </div>
              </div>
            )}
          </div>
          {/* Roles — inline compact */}
          {(gapAnalysis.missingRoles.length > 0 || (gapAnalysis.presentRoles && gapAnalysis.presentRoles.length > 0)) && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 2, marginBottom: 3 }}>
              {gapAnalysis.missingRoles.map(r => (
                <span key={r} style={{ fontSize: 7, padding: "1px 5px", borderRadius: 3, background: "rgba(239,68,68,0.1)", color: "#EF4444", fontFamily: AF, fontWeight: 600 }}>{r}</span>
              ))}
              {(gapAnalysis.presentRoles || []).map(r => (
                <span key={r} style={{ fontSize: 7, padding: "1px 5px", borderRadius: 3, background: isDark ? "rgba(34,197,94,0.08)" : "rgba(34,197,94,0.1)", color: "#22C55E", fontFamily: AF, fontWeight: 500 }}>{r}</span>
              ))}
            </div>
          )}
          {/* Top issues — 3 max */}
          {gapAnalysis.gaps.length > 0 && (
            <div>
              {gapAnalysis.gaps.slice(0, 3).map((g, i) => (
                <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 4, marginBottom: 2 }}>
                  <span style={{ width: 4, height: 4, borderRadius: "50%", background: g.severityColor, flexShrink: 0, marginTop: 3 }} />
                  <div style={{ fontSize: 8, color: theme.textSec, fontFamily: AF, lineHeight: 1.3 }}>{g.message}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Mix Needs (collapsible, shows 3 by default) ── */}
      {mixNeeds.length > 0 && (
        <div style={card}>
          <div style={{ ...lbl, display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", marginBottom: 4 }} onClick={() => setNeedsExpanded(!needsExpanded)}>
            <span>Mix Needs</span>
            <svg width="7" height="7" viewBox="0 0 8 8" style={{ transition: "transform 0.2s", transform: needsExpanded ? "rotate(180deg)" : "rotate(0)" }}>
              <path d="M1 2.5l3 3 3-3" stroke="currentColor" strokeWidth="1.2" fill="none" strokeLinecap="round" />
            </svg>
          </div>
          {(needsExpanded ? mixNeeds.slice(0, 8) : mixNeeds.slice(0, 3)).map((need, i) => (
            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 4, marginBottom: 2 }}>
              <span style={{ width: 4, height: 4, borderRadius: "50%", background: NEED_CATEGORY_COLORS[need.category] || theme.textMuted, flexShrink: 0, marginTop: 3, opacity: 0.4 + need.severity * 0.6 }} />
              <div style={{ fontSize: 8, color: theme.text, fontFamily: AF, lineHeight: 1.3 }}>{need.description}</div>
            </div>
          ))}
          {mixNeeds.length > 3 && !needsExpanded && (
            <div style={{ fontSize: 7, color: theme.textMuted, fontFamily: AF, cursor: "pointer", textAlign: "center", marginTop: 2 }} onClick={() => setNeedsExpanded(true)}>
              +{mixNeeds.length - 3} more
            </div>
          )}
        </div>
      )}

      {/* ── Chart Intelligence (compact) ── */}
      {chartComparison && <ChartIntel data={chartComparison} />}

      {/* ── Bridge Status (compact) ── */}
      {bridge.connected && (
        <div style={{ ...card, background: isDark ? "rgba(34,197,94,0.04)" : "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.12)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 4 }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#22C55E", animation: "pulse 2s ease infinite" }} />
            <span style={{ fontSize: 8, fontWeight: 600, color: "#22C55E", fontFamily: AF, letterSpacing: 1 }}>DAW</span>
            <span style={{ marginLeft: "auto", fontSize: 12, color: theme.text, fontWeight: 700, fontFamily: MONO }}>{bridge.dawBpm.toFixed(0)}</span>
            <span style={{ fontSize: 7, color: theme.textMuted, fontFamily: AF }}>BPM</span>
          </div>
          <div style={{ display: "flex", gap: 3, alignItems: "center", marginBottom: 3 }}>
            <span style={{ fontSize: 8, color: theme.textMuted, fontFamily: AF }}>Sync</span>
            <button onClick={() => bridge.setDawSync(!bridge.dawSync)}
              style={{ width: 26, height: 13, borderRadius: 7, border: "none", background: bridge.dawSync ? "#22C55E" : theme.borderLight, position: "relative", cursor: "pointer", transition: "background 0.2s" }}>
              <div style={{ width: 9, height: 9, borderRadius: 5, background: "#fff", position: "absolute", top: 2, left: bridge.dawSync ? 15 : 2, transition: "left 0.2s" }} />
            </button>
            <span style={{ fontSize: 8, color: bridge.dawPlaying ? "#22C55E" : theme.textMuted, fontFamily: MONO, marginLeft: "auto" }}>{bridge.dawPlaying ? "\u25B6" : "\u25A0"}</span>
          </div>
        </div>
      )}

      {/* ── Spectrum (compact) ── */}
      {a.frequency_bands && (
        <div style={card}>
          <div style={lbl}>Spectrum</div>
          <SpectrumViz trackBands={a.frequency_bands} sampleBands={activeSample?.frequency_bands} gaps={a.frequency_gaps} height={50} />
        </div>
      )}
    </>
  );
});
