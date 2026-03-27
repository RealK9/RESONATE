/**
 * RESONATE — Version Timeline.
 * Horizontal timeline showing readiness score progression across versions.
 * Compact strip that sits above the AI Summary Card.
 */

import { useState, useMemo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { AF, MONO, SERIF } from "../theme/fonts";

function scoreColor(score) {
  if (score == null) return "#6B7280";
  if (score >= 85) return "#22C55E";
  if (score >= 65) return "#3B82F6";
  if (score >= 40) return "#F59E0B";
  return "#EF4444";
}

function formatDate(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  const mo = d.toLocaleString("default", { month: "short" });
  const day = d.getDate();
  const h = d.getHours();
  const m = String(d.getMinutes()).padStart(2, "0");
  return `${mo} ${day}, ${h}:${m}`;
}

export function VersionTimeline({ versions = [], onSaveVersion, onSelectVersion }) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const [hoveredId, setHoveredId] = useState(null);
  const [selectedId, setSelectedId] = useState(null);

  // Compute deltas between consecutive versions
  const versionData = useMemo(() => {
    return versions.map((v, i) => {
      const prev = i > 0 ? versions[i - 1] : null;
      const delta =
        prev && v.readiness_score != null && prev.readiness_score != null
          ? Math.round(v.readiness_score - prev.readiness_score)
          : null;
      return { ...v, delta };
    });
  }, [versions]);

  const handleDotClick = (v) => {
    setSelectedId(v.id === selectedId ? null : v.id);
    if (onSelectVersion) onSelectVersion(v);
  };

  const containerStyle = {
    display: "flex",
    alignItems: "center",
    gap: 0,
    padding: "8px 16px",
    background: theme.surface,
    borderBottom: "1px solid " + theme.border,
    overflowX: "auto",
    overflowY: "hidden",
    minHeight: 44,
  };

  const labelStyle = {
    fontSize: 9,
    fontFamily: AF,
    fontWeight: 600,
    color: theme.textMuted,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginRight: 10,
    flexShrink: 0,
    userSelect: "none",
  };

  const lineStyle = {
    width: 24,
    height: 1,
    background: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.08)",
    flexShrink: 0,
  };

  const dotOuter = (v, isSelected) => ({
    position: "relative",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    cursor: "pointer",
    flexShrink: 0,
  });

  const dotStyle = (v, isHovered, isSelected) => ({
    width: isSelected ? 22 : 18,
    height: isSelected ? 22 : 18,
    borderRadius: "50%",
    background: scoreColor(v.readiness_score),
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 8,
    fontWeight: 700,
    fontFamily: MONO,
    color: "#fff",
    transition: "all 0.2s ease",
    boxShadow: isSelected
      ? `0 0 0 3px ${isDark ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.1)"}, 0 0 8px ${scoreColor(v.readiness_score)}40`
      : isHovered
        ? `0 0 6px ${scoreColor(v.readiness_score)}40`
        : "none",
    border: isSelected ? `2px solid ${isDark ? "#fff" : "#000"}20` : "2px solid transparent",
  });

  const versionLabelStyle = {
    fontSize: 8,
    fontFamily: MONO,
    color: theme.textMuted,
    marginTop: 2,
    whiteSpace: "nowrap",
  };

  const deltaStyle = (delta) => ({
    position: "absolute",
    top: -14,
    fontSize: 8,
    fontWeight: 700,
    fontFamily: MONO,
    color: delta > 0 ? "#22C55E" : delta < 0 ? "#EF4444" : theme.textMuted,
    whiteSpace: "nowrap",
  });

  const tooltipStyle = {
    position: "absolute",
    top: -52,
    left: "50%",
    transform: "translateX(-50%)",
    background: isDark ? "#2A2A38" : "#fff",
    border: "1px solid " + theme.border,
    borderRadius: 6,
    padding: "5px 8px",
    whiteSpace: "nowrap",
    zIndex: 100,
    boxShadow: isDark ? "0 4px 12px rgba(0,0,0,0.4)" : "0 4px 12px rgba(0,0,0,0.1)",
    pointerEvents: "none",
  };

  const saveBtnStyle = {
    marginLeft: 8,
    padding: "4px 10px",
    borderRadius: 5,
    border: "1px solid " + (isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.1)"),
    background: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.02)",
    color: theme.textSec,
    fontSize: 9,
    fontFamily: AF,
    fontWeight: 600,
    cursor: "pointer",
    flexShrink: 0,
    transition: "all 0.15s ease",
    whiteSpace: "nowrap",
  };

  return (
    <div style={containerStyle}>
      <span style={labelStyle}>Versions</span>

      {versionData.map((v, i) => (
        <div key={v.id} style={{ display: "flex", alignItems: "center", flexShrink: 0 }}>
          {i > 0 && (
            <div style={lineStyle} />
          )}
          <div
            style={dotOuter(v, v.id === selectedId)}
            onMouseEnter={() => setHoveredId(v.id)}
            onMouseLeave={() => setHoveredId(null)}
            onClick={() => handleDotClick(v)}
          >
            {/* Delta arrow */}
            {v.delta != null && v.delta !== 0 && (
              <span style={deltaStyle(v.delta)}>
                {v.delta > 0 ? "\u2191" : "\u2193"}{Math.abs(v.delta)}
              </span>
            )}

            {/* Dot */}
            <div style={dotStyle(v, hoveredId === v.id, v.id === selectedId)}>
              {v.readiness_score != null ? Math.round(v.readiness_score) : "?"}
            </div>

            {/* Label */}
            <span style={versionLabelStyle}>{v.version_label}</span>

            {/* Tooltip on hover */}
            {hoveredId === v.id && (
              <div style={tooltipStyle}>
                <div style={{ fontSize: 10, fontWeight: 600, fontFamily: AF, color: theme.text }}>
                  {v.version_label}
                </div>
                <div style={{ fontSize: 9, fontFamily: MONO, color: theme.textMuted, marginTop: 2 }}>
                  Readiness: {v.readiness_score != null ? Math.round(v.readiness_score) : "N/A"}
                  {v.gap_summary && (
                    <span style={{ marginLeft: 6, fontFamily: AF }}>{v.gap_summary.length > 40 ? v.gap_summary.slice(0, 40) + "..." : v.gap_summary}</span>
                  )}
                </div>
                <div style={{ fontSize: 8, fontFamily: MONO, color: theme.textFaint, marginTop: 1 }}>
                  {formatDate(v.created_at)}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}

      {/* Save Version button at the right end */}
      {onSaveVersion && (
        <button
          style={saveBtnStyle}
          onClick={onSaveVersion}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.02)";
          }}
        >
          + Save Version
        </button>
      )}

      {versions.length === 0 && !onSaveVersion && (
        <span style={{ fontSize: 9, fontFamily: AF, color: theme.textFaint, fontStyle: "italic" }}>
          No versions yet
        </span>
      )}
    </div>
  );
}
