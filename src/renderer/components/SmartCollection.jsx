/**
 * SmartCollection — Themed sample kit card with RESONATE branded cover art,
 * expand/collapse, and one-click export.
 */
import { useState } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { SERIF, AF, MONO } from "../theme/fonts";

// ── Cover art palettes + unique artwork per collection type ──

const COVER_GRADIENTS = {
  target: ["#8B5CF6", "#A855F7", "#6366F1"],
  sparkle: ["#F59E0B", "#F97316", "#EC4899"],
  rhythm: ["#22C55E", "#06B6D4", "#3B82F6"],
  crown: ["#3B82F6", "#8B5CF6", "#8B5CF6"],
};

/** Unique SVG artwork per type — each cover is visually distinct */
const COVER_ART = {
  // Target/Essentials: concentric rings + crosshair
  target: (s) => (
    <svg width={s} height={s} viewBox="0 0 42 42" style={{ position: "absolute", inset: 0 }}>
      <circle cx="21" cy="21" r="16" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="0.6" />
      <circle cx="21" cy="21" r="11" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="0.7" />
      <circle cx="21" cy="21" r="6" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="0.8" />
      <circle cx="21" cy="21" r="2" fill="rgba(255,255,255,0.35)" />
      <line x1="21" y1="2" x2="21" y2="10" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
      <line x1="21" y1="32" x2="21" y2="40" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
      <line x1="2" y1="21" x2="10" y2="21" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
      <line x1="32" y1="21" x2="40" y2="21" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
    </svg>
  ),
  // Sparkle/Polish: starburst + scattered diamonds
  sparkle: (s) => (
    <svg width={s} height={s} viewBox="0 0 42 42" style={{ position: "absolute", inset: 0 }}>
      <path d="M21 4l2 8 8 2-8 2-2 8-2-8-8-2 8-2z" fill="rgba(255,255,255,0.2)" />
      <path d="M33 8l1 3 3 1-3 1-1 3-1-3-3-1 3-1z" fill="rgba(255,255,255,0.25)" />
      <path d="M10 28l1 3 3 1-3 1-1 3-1-3-3-1 3-1z" fill="rgba(255,255,255,0.15)" />
      <path d="M32 30l0.7 2 2 0.7-2 0.7-0.7 2-0.7-2-2-0.7 2-0.7z" fill="rgba(255,255,255,0.18)" />
      <circle cx="8" cy="10" r="0.6" fill="rgba(255,255,255,0.3)" />
      <circle cx="35" cy="20" r="0.5" fill="rgba(255,255,255,0.2)" />
      <circle cx="15" cy="36" r="0.4" fill="rgba(255,255,255,0.15)" />
      <line x1="0" y1="42" x2="42" y2="0" stroke="rgba(255,255,255,0.04)" strokeWidth="8" />
    </svg>
  ),
  // Rhythm/Groove: EQ bars + waveform
  rhythm: (s) => (
    <svg width={s} height={s} viewBox="0 0 42 42" style={{ position: "absolute", inset: 0 }}>
      {[6, 11, 16, 21, 26, 31, 36].map((x, i) => {
        const heights = [10, 16, 22, 18, 24, 14, 8];
        const h = heights[i];
        return <rect key={i} x={x - 1.2} y={21 - h / 2} width={2.4} height={h} rx="1.2" fill="rgba(255,255,255,0.15)" />;
      })}
      <path d="M3 21 Q8 14, 13 19 T23 17 T33 21 T40 18" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="0.8" />
      <path d="M3 24 Q10 28, 17 23 T27 26 T37 22 T42 25" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="0.6" />
      <circle cx="6" cy="6" r="0.5" fill="rgba(255,255,255,0.2)" />
      <circle cx="36" cy="36" r="0.5" fill="rgba(255,255,255,0.2)" />
    </svg>
  ),
  // Crown/Top Picks: crown silhouette + geometric rays
  crown: (s) => (
    <svg width={s} height={s} viewBox="0 0 42 42" style={{ position: "absolute", inset: 0 }}>
      <path d="M8 30l3-14 5 6 5-10 5 10 5-6 3 14z" fill="rgba(255,255,255,0.12)" stroke="rgba(255,255,255,0.2)" strokeWidth="0.7" strokeLinejoin="round" />
      <rect x="8" y="30" width="26" height="3" rx="1" fill="rgba(255,255,255,0.15)" />
      <circle cx="21" cy="10" r="1.5" fill="rgba(255,255,255,0.3)" />
      <circle cx="13" cy="22" r="0.8" fill="rgba(255,255,255,0.2)" />
      <circle cx="29" cy="22" r="0.8" fill="rgba(255,255,255,0.2)" />
      {[0, 1, 2, 3, 4].map(i => (
        <line key={i} x1="21" y1="5" x2={21 + Math.cos((i / 5) * Math.PI - Math.PI / 2) * 12} y2={5 + Math.sin((i / 5) * Math.PI - Math.PI / 2) * 12} stroke="rgba(255,255,255,0.08)" strokeWidth="0.5" />
      ))}
    </svg>
  ),
};

/** Branded cover art — each type gets a unique visual identity */
function CoverArt({ type, size = 42, isDark }) {
  const colors = COVER_GRADIENTS[type] || COVER_GRADIENTS.target;
  const r = size * 0.16;
  const ArtSvg = COVER_ART[type] || COVER_ART.target;
  return (
    <div style={{
      width: size, height: size, borderRadius: r, flexShrink: 0,
      background: `linear-gradient(135deg, ${colors[0]}, ${colors[1]}, ${colors[2]})`,
      position: "relative", overflow: "hidden",
      boxShadow: isDark
        ? `0 3px 12px ${colors[0]}35, 0 1px 4px rgba(0,0,0,0.5)`
        : `0 3px 12px ${colors[0]}25, 0 1px 4px rgba(0,0,0,0.12)`,
    }}>
      {ArtSvg(size)}
      {/* RSN watermark */}
      <div style={{
        position: "absolute", bottom: 2, right: 3,
        fontSize: Math.max(5, size * 0.14), fontWeight: 800, fontFamily: MONO,
        color: "rgba(255,255,255,0.18)", letterSpacing: 0.5, lineHeight: 1,
      }}>
        RSN
      </div>
    </div>
  );
}

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
  target: "#8B5CF6",
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

  const iconColor = ICON_COLORS[collection.icon] || "#8B5CF6";
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
        width: 220,
        background: isDark
          ? "rgba(255,255,255,0.03)"
          : "rgba(0,0,0,0.02)",
        border: `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`,
        borderRadius: 10,
        overflow: "hidden",
        cursor: "pointer",
        transition: "all 0.25s ease",
        flexShrink: 0,
        backdropFilter: "blur(12px)",
      }}
      onClick={() => setExpanded(!expanded)}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = isDark
          ? "rgba(255,255,255,0.14)"
          : "rgba(0,0,0,0.14)";
        e.currentTarget.style.background = isDark
          ? "rgba(255,255,255,0.05)"
          : "rgba(0,0,0,0.03)";
        e.currentTarget.style.transform = "translateY(-2px)";
        e.currentTarget.style.boxShadow = isDark
          ? `0 6px 20px ${iconColor}15, 0 2px 8px rgba(0,0,0,0.3)`
          : `0 6px 20px ${iconColor}10, 0 2px 8px rgba(0,0,0,0.06)`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = isDark
          ? "rgba(255,255,255,0.06)"
          : "rgba(0,0,0,0.06)";
        e.currentTarget.style.background = isDark
          ? "rgba(255,255,255,0.03)"
          : "rgba(0,0,0,0.02)";
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.boxShadow = "none";
      }}
    >
      {/* Header with cover art */}
      <div style={{ padding: "10px 12px 8px", display: "flex", alignItems: "center", gap: 10 }}>
        <CoverArt type={collection.icon || "target"} size={42} isDark={isDark} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 11, fontWeight: 600, color: theme.text,
            fontFamily: AF, lineHeight: 1.3,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {collection.name}
          </div>
          <div style={{
            fontSize: 9, color: theme.textMuted, fontFamily: MONO,
            marginTop: 1,
          }}>
            {collection.samples.length} samples · +{collection.total_impact.toFixed(1)}
          </div>
        </div>
        <svg
          width="10" height="10" viewBox="0 0 10 10"
          style={{ opacity: 0.4, transition: "transform 0.2s ease", transform: expanded ? "rotate(180deg)" : "rotate(0deg)", flexShrink: 0 }}
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
          {/* Cover art banner in dropdown */}
          <div style={{
            height: 38, position: "relative", overflow: "hidden",
            background: `linear-gradient(135deg, ${(COVER_GRADIENTS[collection.icon] || COVER_GRADIENTS.target).join(", ")})`,
          }}>
            {/* Dot grid overlay */}
            <svg width="100%" height="38" style={{ position: "absolute", inset: 0, opacity: 0.15 }}>
              {Array.from({ length: 20 }, (_, i) => (
                <circle key={i} cx={12 + (i % 10) * 22} cy={i < 10 ? 12 : 26} r="1" fill="#fff" />
              ))}
              <line x1="70%" y1="0" x2="100%" y2="38" stroke="rgba(255,255,255,0.2)" strokeWidth="0.8" />
            </svg>
            <div style={{
              position: "absolute", inset: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 8, fontWeight: 700, fontFamily: MONO,
              color: "rgba(255,255,255,0.5)", letterSpacing: 3, textTransform: "uppercase",
            }}>
              RESONATE · {collection.name}
            </div>
            <div style={{
              position: "absolute", bottom: 2, right: 4,
              fontSize: 6, fontWeight: 800, fontFamily: MONO,
              color: "rgba(255,255,255,0.15)", letterSpacing: 1,
            }}>
              RSN
            </div>
          </div>

          {collection.samples.map((sample, i) => (
            <div
              key={i}
              onClick={(e) => {
                e.stopPropagation();
                if (onPreview) onPreview(sample);
              }}
              style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "7px 12px",
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
          <div style={{ padding: "8px 12px 10px" }}>
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
