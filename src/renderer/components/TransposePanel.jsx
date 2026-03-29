/**
 * RESONATE — Transpose Panel.
 * Chromatic key selector with flat/sharp tabs and +/- semitone offset.
 * Splice-style pitch control — user manually selects target key.
 */

import { memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { MONO, AF } from "../theme/fonts";

export const TransposePanel = memo(function TransposePanel({
  enabled, onToggle,
  noteNames, targetKey, onSelectKey,
  pitchOffset, onPitchOffset,
  useFlats, onToggleFlats,
  uploadKey, getSemitones, activeSampleKey,
}) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  const semitones = activeSampleKey ? getSemitones(activeSampleKey) : pitchOffset;
  const semiLabel = semitones === 0 ? "0" : (semitones > 0 ? "+" + semitones : String(semitones));

  return (
    <div style={{
      background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.015)",
      border: "1px solid " + theme.border,
      borderRadius: 8,
      padding: "10px 12px",
      opacity: enabled ? 1 : 0.45,
      transition: "opacity 0.2s ease",
    }}>
      {/* Header: title + toggle */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 10, fontWeight: 600, color: theme.text, fontFamily: AF, textTransform: "uppercase", letterSpacing: 1 }}>Transpose</span>
          {semitones !== 0 && enabled && (
            <span style={{ fontSize: 9, fontFamily: MONO, color: "#8B5CF6", fontWeight: 700 }}>{semiLabel} st</span>
          )}
        </div>
        <button
          onClick={onToggle}
          style={{
            width: 32, height: 16, borderRadius: 8, border: "none", cursor: "pointer",
            background: enabled ? "#8B5CF6" : (isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"),
            position: "relative", transition: "background 0.2s",
          }}
        >
          <div style={{
            width: 12, height: 12, borderRadius: "50%", background: "#fff",
            position: "absolute", top: 2,
            left: enabled ? 18 : 2,
            transition: "left 0.2s ease",
          }} />
        </button>
      </div>

      {/* Flat / Sharp toggle */}
      <div style={{ display: "flex", gap: 0, marginBottom: 6, borderRadius: 4, overflow: "hidden", border: "1px solid " + theme.borderLight }}>
        <button
          onClick={() => onToggleFlats(true)}
          style={{
            flex: 1, padding: "3px 0", border: "none", fontSize: 8, fontWeight: 600,
            background: useFlats ? (isDark ? "rgba(139,92,246,0.12)" : "rgba(139,92,246,0.08)") : "transparent",
            color: useFlats ? "#8B5CF6" : theme.textMuted, cursor: "pointer", fontFamily: AF,
          }}
        >FLATS</button>
        <button
          onClick={() => onToggleFlats(false)}
          style={{
            flex: 1, padding: "3px 0", border: "none", borderLeft: "1px solid " + theme.borderLight, fontSize: 8, fontWeight: 600,
            background: !useFlats ? (isDark ? "rgba(139,92,246,0.12)" : "rgba(139,92,246,0.08)") : "transparent",
            color: !useFlats ? "#8B5CF6" : theme.textMuted, cursor: "pointer", fontFamily: AF,
          }}
        >SHARPS</button>
      </div>

      {/* Chromatic key grid — 2 rows of 6 */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 3, marginBottom: 8 }}>
        {noteNames.map(note => {
          const isTarget = targetKey && note === targetKey.replace(/\s*(m|min|maj|major|minor)$/i, "").trim();
          const isUpload = uploadKey && note === uploadKey.replace(/\s*(m|min|maj|major|minor)$/i, "").trim();
          return (
            <button
              key={note}
              onClick={() => enabled && onSelectKey(note)}
              style={{
                padding: "5px 0", border: "none", borderRadius: 4, fontSize: 10, fontWeight: 600,
                fontFamily: MONO, cursor: enabled ? "pointer" : "default",
                background: isTarget
                  ? "linear-gradient(135deg, rgba(139,92,246,0.25), rgba(139,92,246,0.2))"
                  : (isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)"),
                color: isTarget ? "#8B5CF6" : isUpload ? "#06B6D4" : theme.textSec,
                outline: isUpload && !isTarget ? "1px solid rgba(6,182,212,0.3)" : "none",
                transition: "all 0.15s",
              }}
            >{note}</button>
          );
        })}
      </div>

      {/* Pitch offset: -/+ semitone buttons */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
        <button
          onClick={() => enabled && onPitchOffset(pitchOffset - 1)}
          style={{
            width: 26, height: 22, borderRadius: 4, border: "1px solid " + theme.borderLight,
            background: "transparent", color: theme.textSec, fontSize: 13, fontWeight: 700,
            cursor: enabled ? "pointer" : "default", fontFamily: MONO, display: "flex",
            alignItems: "center", justifyContent: "center",
          }}
        >−</button>
        <span style={{ fontSize: 11, fontFamily: MONO, color: theme.text, fontWeight: 600, minWidth: 32, textAlign: "center" }}>
          {pitchOffset === 0 ? "0 st" : (pitchOffset > 0 ? "+" : "") + pitchOffset + " st"}
        </span>
        <button
          onClick={() => enabled && onPitchOffset(pitchOffset + 1)}
          style={{
            width: 26, height: 22, borderRadius: 4, border: "1px solid " + theme.borderLight,
            background: "transparent", color: theme.textSec, fontSize: 13, fontWeight: 700,
            cursor: enabled ? "pointer" : "default", fontFamily: MONO, display: "flex",
            alignItems: "center", justifyContent: "center",
          }}
        >+</button>
      </div>
    </div>
  );
});
