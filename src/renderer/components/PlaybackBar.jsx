/**
 * RESONATE — Playback Bar.
 * Persistent bottom bar with KEY and BPM quick-access buttons that expand
 * to show TransposePanel and TimestretchPanel. Always visible on results screen.
 * Splice-style controls — user decides key/tempo for sample preview.
 */

import { useState, memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { MONO, AF } from "../theme/fonts";
import { TransposePanel } from "./TransposePanel";
import { TimestretchPanel } from "./TimestretchPanel";

export const PlaybackBar = memo(function PlaybackBar({ pitchTempo }) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const [showPanel, setShowPanel] = useState(null); // "key" | "bpm" | null

  const {
    transposeEnabled, setTransposeEnabled,
    timestrechEnabled, setTimestrechEnabled,
    targetKey, setTargetKey,
    pitchOffset, setPitchOffset,
    tempoMultiplier, setTempoMultiplier,
    uploadKey, uploadBpm,
    useFlats, setUseFlats,
    effectiveKey, noteNames, bpmOptions,
    getSemitones,
  } = pitchTempo;

  const togglePanel = (panel) => setShowPanel(prev => prev === panel ? null : panel);

  // Display values for the quick-access buttons
  const keyDisplay = effectiveKey || uploadKey || "—";
  const bpmDisplay = uploadBpm ? Math.round(uploadBpm) : "—";

  return (
    <div style={{ position: "relative" }}>
      {/* Expanded panel (slides up above the bar) */}
      {showPanel && (
        <div style={{
          position: "absolute", bottom: "100%", left: 0, right: 0,
          padding: "8px 14px 4px",
          background: isDark ? "rgba(10,10,16,0.95)" : "rgba(245,243,239,0.95)",
          backdropFilter: "blur(16px)",
          borderTop: "1px solid " + theme.border,
          animation: "fadeInUp 0.15s ease",
        }}>
          {showPanel === "key" && (
            <TransposePanel
              enabled={transposeEnabled}
              onToggle={() => setTransposeEnabled(!transposeEnabled)}
              noteNames={noteNames}
              targetKey={targetKey}
              onSelectKey={setTargetKey}
              pitchOffset={pitchOffset}
              onPitchOffset={setPitchOffset}
              useFlats={useFlats}
              onToggleFlats={setUseFlats}
              uploadKey={uploadKey}
              getSemitones={getSemitones}
              activeSampleKey={null}
            />
          )}
          {showPanel === "bpm" && (
            <TimestretchPanel
              enabled={timestrechEnabled}
              onToggle={() => setTimestrechEnabled(!timestrechEnabled)}
              bpmOptions={bpmOptions}
              tempoMultiplier={tempoMultiplier}
              onSelectMultiplier={setTempoMultiplier}
              uploadBpm={uploadBpm}
            />
          )}
        </div>
      )}

      {/* Quick-access bar */}
      <div style={{
        display: "flex", alignItems: "center", gap: 6,
        padding: "6px 14px",
        background: isDark ? "rgba(18,18,26,0.98)" : "rgba(255,255,255,0.98)",
        borderTop: "1px solid " + theme.border,
      }}>
        {/* KEY button */}
        <button
          onClick={() => togglePanel("key")}
          style={{
            display: "flex", alignItems: "center", gap: 4,
            padding: "4px 10px", borderRadius: 5,
            border: "1px solid " + (showPanel === "key" ? "rgba(217,70,239,0.4)" : theme.borderLight),
            background: showPanel === "key"
              ? (isDark ? "rgba(217,70,239,0.1)" : "rgba(217,70,239,0.06)")
              : "transparent",
            cursor: "pointer", transition: "all 0.15s",
          }}
        >
          <span style={{ fontSize: 8, fontWeight: 600, color: theme.textMuted, fontFamily: AF, textTransform: "uppercase", letterSpacing: 0.5 }}>KEY</span>
          <span style={{ fontSize: 11, fontWeight: 700, color: transposeEnabled ? "#D946EF" : theme.text, fontFamily: MONO }}>{keyDisplay}</span>
          {transposeEnabled && <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#D946EF" }} />}
        </button>

        {/* BPM button */}
        <button
          onClick={() => togglePanel("bpm")}
          style={{
            display: "flex", alignItems: "center", gap: 4,
            padding: "4px 10px", borderRadius: 5,
            border: "1px solid " + (showPanel === "bpm" ? "rgba(6,182,212,0.4)" : theme.borderLight),
            background: showPanel === "bpm"
              ? (isDark ? "rgba(6,182,212,0.1)" : "rgba(6,182,212,0.06)")
              : "transparent",
            cursor: "pointer", transition: "all 0.15s",
          }}
        >
          <span style={{ fontSize: 8, fontWeight: 600, color: theme.textMuted, fontFamily: AF, textTransform: "uppercase", letterSpacing: 0.5 }}>BPM</span>
          <span style={{ fontSize: 11, fontWeight: 700, color: timestrechEnabled ? "#06B6D4" : theme.text, fontFamily: MONO }}>{bpmDisplay}</span>
          {timestrechEnabled && <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#06B6D4" }} />}
        </button>

        {/* Pitch offset indicator (compact) */}
        {pitchOffset !== 0 && (
          <span style={{ fontSize: 9, fontFamily: MONO, color: "#D946EF", fontWeight: 600 }}>
            {pitchOffset > 0 ? "+" : ""}{pitchOffset} st
          </span>
        )}

        {/* Tempo multiplier indicator */}
        {timestrechEnabled && tempoMultiplier !== 1 && (
          <span style={{ fontSize: 9, fontFamily: MONO, color: "#06B6D4", fontWeight: 600 }}>
            {tempoMultiplier === 0.5 ? "½×" : "2×"}
          </span>
        )}

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* Status indicators */}
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {transposeEnabled && (
            <span style={{ fontSize: 7, padding: "2px 5px", borderRadius: 3, background: "rgba(217,70,239,0.1)", color: "#D946EF", fontWeight: 700, fontFamily: AF, textTransform: "uppercase", letterSpacing: 0.5 }}>PITCH</span>
          )}
          {timestrechEnabled && (
            <span style={{ fontSize: 7, padding: "2px 5px", borderRadius: 3, background: "rgba(6,182,212,0.1)", color: "#06B6D4", fontWeight: 700, fontFamily: AF, textTransform: "uppercase", letterSpacing: 0.5 }}>TEMPO</span>
          )}
        </div>
      </div>
    </div>
  );
});
