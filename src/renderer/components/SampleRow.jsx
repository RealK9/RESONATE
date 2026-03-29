/**
 * RESONATE — Sample Row Component.
 * Single sample in the list with name, match %, key, BPM, duration, tags, mood badge.
 */

import { useState, useRef, memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { SERIF, MONO, AF } from "../theme/fonts";

const MOOD_COLORS = {
  dark: { bg: "rgba(99,60,180,0.12)", text: "#8B5CF6" },
  warm: { bg: "rgba(234,179,8,0.12)", text: "#CA8A04" },
  bright: { bg: "rgba(59,130,246,0.12)", text: "#3B82F6" },
  aggressive: { bg: "rgba(239,68,68,0.12)", text: "#EF4444" },
  chill: { bg: "rgba(34,197,94,0.12)", text: "#22C55E" },
  neutral: { bg: "rgba(148,163,184,0.08)", text: "#94A3B8" },
};

const ROLE_COLORS = {
  kick: "#F59E0B", snare_clap: "#EF4444", hats_tops: "#6366F1",
  bass: "#8B5CF6", lead: "#EC4899", chord_support: "#14B8A6",
  pad: "#06B6D4", vocal_texture: "#F472B6", fx_transitions: "#A78BFA",
  ambience: "#34D399", percussion: "#FB923C",
};

export const SampleRow = memo(function SampleRow({ sample, isActive, isPlaying, onPlay, onPreviewInContext, isSelected, isChecked, onCheck, onHoverWaveform }) {
  const [hov, setHov] = useState(false);
  const rowRef = useRef(null);
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const dur = d => { const m = Math.floor(d / 60), s = Math.floor(d % 60); return m + ":" + String(s).padStart(2, "0"); };
  const mood = sample.mood || "neutral";
  const moodStyle = MOOD_COLORS[mood] || MOOD_COLORS.neutral;

  return (
    <div ref={rowRef} onClick={() => onPlay(sample)} onMouseEnter={() => { setHov(true); if (onHoverWaveform && rowRef.current) { const r = rowRef.current.getBoundingClientRect(); onHoverWaveform(sample.path, r); } }} onMouseLeave={() => { setHov(false); onHoverWaveform?.(null, null); }}
      className={isActive ? "sample-row-active" : ""}
      style={{ display: "grid", gridTemplateColumns: "22px 30px 1fr 54px 40px 40px 42px", alignItems: "center", gap: 6, padding: "9px 14px", background: isActive ? (isDark ? "rgba(139,92,246,0.04)" : "rgba(139,92,246,0.03)") : isSelected ? (isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.015)") : hov ? theme.surfaceHover : "transparent", borderLeft: isActive ? "2px solid #8B5CF6" : "2px solid transparent", cursor: "pointer", transition: "all 0.15s ease", borderBottom: "1px solid " + theme.border }}>
      {/* Checkbox for batch select */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }} onClick={e => { e.stopPropagation(); onCheck?.(sample.id); }}>
        <div style={{ width: 13, height: 13, borderRadius: 3, border: "1.5px solid " + (isChecked ? theme.text : theme.borderLight), background: isChecked ? theme.text : "transparent", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.15s", cursor: "pointer" }}>
          {isChecked && <svg width="8" height="8" viewBox="0 0 12 12" fill={isDark ? "#0D0D12" : "#fff"}><path d="M2 6l3 3 5-5" stroke={isDark ? "#0D0D12" : "#fff"} strokeWidth="2" fill="none" strokeLinecap="round" /></svg>}
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 3 }}>
        {isActive && isPlaying ? (
          <div style={{ display: "flex", gap: 1.5, alignItems: "end", height: 12 }}>
            {[7, 11, 9, 10, 6].map((h, i) => <div key={i} style={{ width: 2, height: h, background: theme.text, borderRadius: 1, animation: "barBounce 0.55s ease " + (i * 0.08) + "s infinite alternate" }} />)}
          </div>
        ) : <svg width="11" height="11" viewBox="0 0 16 16" fill={hov ? theme.textSec : theme.textFaint}><path d="M4 2.5v11l9-5.5z" /></svg>}
        {sample._isV2 && onPreviewInContext && (
          <div
            onClick={e => { e.stopPropagation(); onPreviewInContext(sample); }}
            title="Preview in context with your track"
            style={{ cursor: "pointer", opacity: hov ? 1 : 0.5, transition: "opacity 0.15s" }}
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
              <path d="M2 4v8l5-4z" fill={hov ? "#8B5CF6" : theme.textFaint} />
              <path d="M6 4v8l5-4z" fill={hov ? "#8B5CF6" : theme.textFaint} opacity="0.5" />
              <rect x="12" y="3" width="1.5" height="10" rx="0.5" fill={hov ? "#8B5CF6" : theme.textFaint} opacity="0.3" />
            </svg>
          </div>
        )}
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 400, color: theme.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: SERIF }}>{sample.clean_name || sample.name}</div>
        <div style={{ display: "flex", gap: 3, marginTop: 2, alignItems: "center" }}>
          <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 3, background: theme.tag, color: theme.tagText }}>{sample.type_label || "Sound"}</span>
          {sample.source && sample.source !== "local" && (
            <span style={{ fontSize: 7, padding: "1px 4px", borderRadius: 3, background: sample.source === "splice" ? "rgba(99,102,241,0.1)" : "rgba(236,72,153,0.1)", color: sample.source === "splice" ? "#6366F1" : "#EC4899", fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5 }}>{sample.source}</span>
          )}
          {mood !== "neutral" && (
            <span style={{ fontSize: 7, padding: "1px 4px", borderRadius: 3, background: moodStyle.bg, color: moodStyle.text, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5, fontFamily: AF }}>{mood}</span>
          )}
          {sample._isV2 && sample.v2_role && (
            <span style={{ fontSize: 7, padding: "1px 4px", borderRadius: 3, background: (ROLE_COLORS[sample.v2_role] || "#888") + "18", color: ROLE_COLORS[sample.v2_role] || "#888", fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5, fontFamily: AF }}>{sample.v2_role.replace("_", " ")}</span>
          )}
          {sample._isV2 && sample.v2_need_addressed ? (
            <span style={{ fontSize: 8, color: theme.textMuted, fontStyle: "italic" }} title={sample.v2_explanation}>{sample.v2_need_addressed}</span>
          ) : sample.match_reason && (
            <span style={{ fontSize: 8, color: theme.textMuted, fontStyle: "italic" }}>{sample.match_reason}</span>
          )}
        </div>
      </div>
      <span style={{ fontSize: 11, fontWeight: 700, color: (sample.match || 50) >= 70 ? "#8B5CF6" : (sample.match || 50) >= 55 ? theme.text : theme.textMuted, fontFamily: MONO }}>{Math.round(sample.match || 50)}%</span>
      <span style={{ fontSize: 10, color: theme.textMuted, fontFamily: MONO }}>{sample.key || "—"}</span>
      <span style={{ fontSize: 10, color: theme.textMuted, fontFamily: MONO }}>{sample.bpm ? Math.round(sample.bpm) : "—"}</span>
      <span style={{ fontSize: 9, color: theme.textFaint }}>{typeof sample.duration === "number" ? dur(sample.duration) : "—"}</span>
    </div>
  );
});
