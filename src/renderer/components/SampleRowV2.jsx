/**
 * RESONATE — Sample Row V2.
 * Splice-style row: play icon, name + tags, match %, key, BPM, duration, hover actions.
 * Click anywhere to play. Hover reveals Similar/Favorite/Layer buttons.
 */

import { useState, useRef, memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { SERIF, MONO, AF } from "../theme/fonts";

const MOOD_COLORS = {
  dark: "#8B5CF6", warm: "#CA8A04", bright: "#3B82F6",
  aggressive: "#EF4444", chill: "#22C55E", neutral: "#94A3B8",
};
const ROLE_COLORS = {
  kick: "#F59E0B", snare_clap: "#EF4444", hats_tops: "#6366F1",
  bass: "#8B5CF6", lead: "#EC4899", chord_support: "#14B8A6",
  pad: "#06B6D4", vocal_texture: "#F472B6", fx_transitions: "#A78BFA",
  ambience: "#34D399", percussion: "#FB923C",
};

export const SampleRowV2 = memo(function SampleRowV2({
  sample, isActive, isPlaying, onPlay,
  isFavorite, onToggleFav, onFindSimilar, onFindLayers,
  isChecked, onCheck,
}) {
  const [hov, setHov] = useState(false);
  const rowRef = useRef(null);
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  const dur = d => {
    const m = Math.floor(d / 60), s = Math.floor(d % 60);
    return m + ":" + String(s).padStart(2, "0");
  };

  const mood = sample.mood || "neutral";
  const moodColor = MOOD_COLORS[mood] || MOOD_COLORS.neutral;

  return (
    <div ref={rowRef} onClick={() => onPlay(sample)}
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{
        display: "grid",
        gridTemplateColumns: "28px 1fr 50px 38px 38px 38px 80px",
        alignItems: "center", gap: 4, padding: "0 14px",
        background: isActive
          ? (isDark ? "rgba(139,92,246,0.05)" : "rgba(139,92,246,0.03)")
          : hov ? theme.surfaceHover : "transparent",
        borderLeft: isActive ? "2px solid #8B5CF6" : "2px solid transparent",
        borderBottom: "1px solid " + theme.borderLight,
        cursor: "pointer", transition: "all 0.1s ease",
        height: 46,
      }}
    >
      {/* Play icon / animated bars */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
        {isActive && isPlaying ? (
          <div style={{ display: "flex", gap: 1.5, alignItems: "end", height: 14 }}>
            {[7, 11, 9, 10, 6].map((h, i) => (
              <div key={i} style={{
                width: 2, height: h, background: "#8B5CF6", borderRadius: 1,
                animation: "barBounce 0.55s ease " + (i * 0.08) + "s infinite alternate",
              }} />
            ))}
          </div>
        ) : (
          <svg width="11" height="11" viewBox="0 0 16 16"
            fill={hov ? theme.text : theme.textFaint}
            style={{ transition: "fill 0.1s" }}>
            <path d="M4 2.5v11l9-5.5z" />
          </svg>
        )}
      </div>

      {/* Name + metadata tags */}
      <div style={{ minWidth: 0, overflow: "hidden" }}>
        <div style={{
          fontSize: 12.5, fontWeight: 400, color: theme.text,
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          fontFamily: SERIF,
        }}>
          {sample.clean_name || sample.name}
        </div>
        <div style={{ display: "flex", gap: 3, marginTop: 1, alignItems: "center" }}>
          <span style={{
            fontSize: 8, padding: "1px 5px", borderRadius: 3,
            background: theme.tag, color: theme.tagText, fontFamily: AF,
          }}>{sample.type_label || "Sound"}</span>
          {sample.source && sample.source !== "local" && (
            <span style={{
              fontSize: 7, padding: "1px 4px", borderRadius: 3,
              background: "rgba(99,102,241,0.1)", color: "#6366F1",
              fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5, fontFamily: AF,
            }}>{sample.source}</span>
          )}
          {mood !== "neutral" && (
            <span style={{
              fontSize: 7, padding: "1px 4px", borderRadius: 3,
              background: moodColor + "18", color: moodColor,
              fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5, fontFamily: AF,
            }}>{mood}</span>
          )}
          {sample._isV2 && sample.v2_role && (
            <span style={{
              fontSize: 7, padding: "1px 4px", borderRadius: 3,
              background: (ROLE_COLORS[sample.v2_role] || "#888") + "18",
              color: ROLE_COLORS[sample.v2_role] || "#888",
              fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5, fontFamily: AF,
            }}>{(sample.v2_role || "").replace("_", " ")}</span>
          )}
        </div>
      </div>

      {/* Match % */}
      <span style={{
        fontSize: 11, fontWeight: 700, fontFamily: MONO, textAlign: "center",
        color: (sample.match || 50) >= 70 ? "#8B5CF6" : (sample.match || 50) >= 55 ? theme.text : theme.textMuted,
      }}>
        {Math.round(sample.match || 50)}%
      </span>

      {/* Key */}
      <span style={{ fontSize: 10, color: theme.textMuted, fontFamily: MONO, textAlign: "center" }}>
        {sample.key || "\u2014"}
      </span>

      {/* BPM */}
      <span style={{ fontSize: 10, color: theme.textMuted, fontFamily: MONO, textAlign: "center" }}>
        {sample.bpm ? Math.round(sample.bpm) : "\u2014"}
      </span>

      {/* Duration */}
      <span style={{ fontSize: 9, color: theme.textFaint, fontFamily: MONO, textAlign: "center" }}>
        {typeof sample.duration === "number" ? dur(sample.duration) : "\u2014"}
      </span>

      {/* Hover actions */}
      <div style={{
        display: "flex", gap: 2, alignItems: "center", justifyContent: "flex-end",
        opacity: hov ? 1 : 0, transition: "opacity 0.1s",
      }}>
        {/* Similar Sounds */}
        <button onClick={e => { e.stopPropagation(); onFindSimilar?.(sample); }}
          title="Similar Sounds"
          style={{ background: "none", border: "none", cursor: "pointer", padding: 3, color: theme.textMuted }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="9" cy="12" r="5" /><circle cx="15" cy="12" r="5" />
          </svg>
        </button>
        {/* Favorite */}
        <button onClick={e => { e.stopPropagation(); onToggleFav?.(sample.id); }}
          style={{
            background: "none", border: "none", cursor: "pointer", padding: 3,
            fontSize: 11, color: isFavorite ? "#D97706" : theme.textMuted,
          }}>
          {isFavorite ? "\u2605" : "\u2606"}
        </button>
        {/* Layer */}
        <button onClick={e => { e.stopPropagation(); onFindLayers?.(sample); }}
          title="Layer Suggestions"
          style={{
            background: "none", border: "none", cursor: "pointer", padding: 3,
            color: theme.textMuted, fontSize: 12, fontWeight: 700,
          }}>
          +
        </button>
      </div>
    </div>
  );
});
