/**
 * SmartCollection — Themed sample kit card with expand/collapse and one-click export.
 * Each collection renders as a premium card with icon, sample list, and export button.
 */
import { useState } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { SERIF, AF, MONO } from "../theme/fonts";

// ── SVG icon paths for each collection type ──

const ICONS = {
  target: (color) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke={color} strokeWidth="1.5" opacity="0.3" />
      <circle cx="12" cy="12" r="6" stroke={color} strokeWidth="1.5" opacity="0.6" />
      <circle cx="12" cy="12" r="2" fill={color} />
    </svg>
  ),
  sparkle: (color) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path d="M12 2l1.5 6.5L20 10l-6.5 1.5L12 18l-1.5-6.5L4 10l6.5-1.5L12 2z" stroke={color} strokeWidth="1.5" strokeLinejoin="round" fill={color} fillOpacity="0.15" />
      <path d="M19 15l.75 2.25L22 18l-2.25.75L19 21l-.75-2.25L16 18l2.25-.75L19 15z" fill={color} fillOpacity="0.5" />
    </svg>
  ),
  rhythm: (color) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <rect x="3" y="14" width="3" height="7" rx="1" fill={color} fillOpacity="0.4" />
      <rect x="8" y="10" width="3" height="11" rx="1" fill={color} fillOpacity="0.6" />
      <rect x="13" y="6" width="3" height="15" rx="1" fill={color} fillOpacity="0.8" />
      <rect x="18" y="3" width="3" height="18" rx="1" fill={color} />
    </svg>
  ),
  crown: (color) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path d="M3 18h18v2H3v-2z" fill={color} fillOpacity="0.3" />
      <path d="M3 18l2-10 4 4 3-8 3 8 4-4 2 10H3z" stroke={color} strokeWidth="1.5" strokeLinejoin="round" fill={color} fillOpacity="0.12" />
      <circle cx="12" cy="4" r="1" fill={color} />
    </svg>
  ),
};

const ICON_COLORS = {
  target: "#D946EF",
  sparkle: "#F59E0B",
  rhythm: "#22C55E",
  crown: "#3B82F6",
};

const ROLE_LABELS = {
  kick: "Kick", snare_clap: "Snare", hats_tops: "Hi-Hat",
  bass: "Bass", lead: "Lead", chord_support: "Chords",
  pad: "Pad", vocal_texture: "Vocal", fx_transitions: "FX",
  ambience: "Ambience", percussion: "Perc",
};

function SmartCollection({ collection, onPreview, onExport }) {
  const { theme, isDark } = useTheme();
  const [expanded, setExpanded] = useState(false);
  const [exporting, setExporting] = useState(false);

  if (!collection || !collection.samples?.length) return null;

  const iconColor = ICON_COLORS[collection.icon] || "#D946EF";
  const IconComponent = ICONS[collection.icon] || ICONS.target;

  const handleExport = async (e) => {
    e.stopPropagation();
    if (exporting) return;
    setExporting(true);
    try {
      await onExport(collection.id);
    } catch {
      // toast handled by caller
    } finally {
      setExporting(false);
    }
  };

  return (
    <div
      style={{
        minWidth: 240,
        maxWidth: 280,
        background: isDark
          ? "rgba(255,255,255,0.03)"
          : "rgba(0,0,0,0.02)",
        border: `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`,
        borderRadius: 10,
        overflow: "hidden",
        cursor: "pointer",
        transition: "all 0.2s ease",
        flexShrink: 0,
        backdropFilter: "blur(12px)",
      }}
      onClick={() => setExpanded(!expanded)}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = isDark
          ? "rgba(255,255,255,0.12)"
          : "rgba(0,0,0,0.12)";
        e.currentTarget.style.background = isDark
          ? "rgba(255,255,255,0.05)"
          : "rgba(0,0,0,0.03)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = isDark
          ? "rgba(255,255,255,0.06)"
          : "rgba(0,0,0,0.06)";
        e.currentTarget.style.background = isDark
          ? "rgba(255,255,255,0.03)"
          : "rgba(0,0,0,0.02)";
      }}
    >
      {/* Header */}
      <div style={{ padding: "12px 14px 10px", display: "flex", alignItems: "flex-start", gap: 10 }}>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: `${iconColor}12`,
          display: "flex", alignItems: "center", justifyContent: "center",
          flexShrink: 0,
        }}>
          {IconComponent(iconColor)}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 12, fontWeight: 600, color: theme.text,
            fontFamily: AF, lineHeight: 1.3,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {collection.name}
          </div>
          <div style={{
            fontSize: 9, color: theme.textMuted, fontFamily: AF,
            marginTop: 2, lineHeight: 1.4,
          }}>
            {collection.description}
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        padding: "0 14px 10px",
      }}>
        <span style={{
          fontSize: 9, fontFamily: MONO, color: theme.textSec,
          padding: "2px 6px", borderRadius: 4,
          background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)",
        }}>
          {collection.samples.length} samples
        </span>
        <span style={{
          fontSize: 9, fontFamily: MONO, color: iconColor,
          opacity: 0.8,
        }}>
          +{collection.total_impact.toFixed(1)} impact
        </span>
        <svg
          width="10" height="10" viewBox="0 0 10 10"
          style={{ marginLeft: "auto", opacity: 0.4, transition: "transform 0.2s ease", transform: expanded ? "rotate(180deg)" : "rotate(0deg)" }}
        >
          <path d="M2 3.5l3 3 3-3" stroke={theme.textMuted} strokeWidth="1.5" fill="none" strokeLinecap="round" />
        </svg>
      </div>

      {/* Expanded sample list */}
      {expanded && (
        <div style={{
          borderTop: `1px solid ${isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"}`,
          maxHeight: 220,
          overflowY: "auto",
        }}>
          {collection.samples.map((sample, i) => (
            <div
              key={i}
              onClick={(e) => {
                e.stopPropagation();
                if (onPreview) onPreview(sample);
              }}
              style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "7px 14px",
                cursor: "pointer",
                transition: "background 0.15s ease",
                borderBottom: i < collection.samples.length - 1
                  ? `1px solid ${isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)"}`
                  : "none",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = isDark
                  ? "rgba(255,255,255,0.04)"
                  : "rgba(0,0,0,0.03)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "transparent";
              }}
            >
              {/* Play icon */}
              <svg width="8" height="8" viewBox="0 0 8 8" style={{ flexShrink: 0, opacity: 0.4 }}>
                <path d="M1 0.5l6 3.5-6 3.5z" fill={theme.textSec} />
              </svg>
              {/* Name */}
              <span style={{
                flex: 1, fontSize: 10, color: theme.text, fontFamily: AF,
                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
              }}>
                {sample.name}
              </span>
              {/* Role tag */}
              {sample.role && (
                <span style={{
                  fontSize: 8, fontFamily: MONO, color: iconColor,
                  padding: "1px 5px", borderRadius: 3,
                  background: `${iconColor}10`,
                  flexShrink: 0, textTransform: "capitalize",
                }}>
                  {ROLE_LABELS[sample.role] || sample.role}
                </span>
              )}
            </div>
          ))}

          {/* Export button */}
          <div style={{ padding: "8px 14px 10px" }}>
            <button
              onClick={handleExport}
              disabled={exporting}
              style={{
                width: "100%",
                padding: "7px 0",
                borderRadius: 6,
                border: `1px solid ${iconColor}40`,
                background: `${iconColor}10`,
                color: iconColor,
                fontSize: 10,
                fontWeight: 600,
                fontFamily: AF,
                cursor: exporting ? "wait" : "pointer",
                transition: "all 0.2s ease",
                opacity: exporting ? 0.6 : 1,
              }}
              onMouseEnter={(e) => {
                if (!exporting) {
                  e.currentTarget.style.background = `${iconColor}20`;
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = `${iconColor}10`;
              }}
            >
              {exporting ? "Exporting..." : "Export Kit (.zip)"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export { SmartCollection };
