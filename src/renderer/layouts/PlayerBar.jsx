/**
 * RESONATE — Persistent Bottom Player Bar.
 * Splice-style: play/pause, sample info, waveform scrubber, KEY/BPM controls, actions.
 * Always visible when a sample is active. Persists across page navigation.
 */

import { useState, memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { SERIF, MONO, AF } from "../theme/fonts";
import { RealWaveform } from "../components/Waveform";
import { VolumeSlider } from "../components/VolumeSlider";
import { TransposePanel } from "../components/TransposePanel";
import { TimestretchPanel } from "../components/TimestretchPanel";

function formatTime(s) {
  if (!s || !isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ":" + String(sec).padStart(2, "0");
}

export const PlayerBar = memo(function PlayerBar({
  audio, activeSample, pitchTempo, waveformPeaks,
  onToggleFav, isFavorite, onFindSimilar, onToggleMix,
  bridgeConnected,
}) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const [expandedPanel, setExpandedPanel] = useState(null); // "key" | "bpm" | null

  if (!activeSample) return null;

  const isPlaying = audio.currentId === activeSample.id && audio.playing;
  const togglePanel = (p) => setExpandedPanel(prev => prev === p ? null : p);
  const pt = pitchTempo || {};

  return (
    <div style={{
      position: "relative", flexShrink: 0,
      borderTop: "1px solid " + theme.border,
      background: isDark ? "rgba(10,10,16,0.98)" : "rgba(255,255,255,0.98)",
      backdropFilter: "blur(20px) saturate(1.4)",
      zIndex: 50,
    }}>
      {/* Expanded transpose/timestretch panel */}
      {expandedPanel && (
        <div style={{
          position: "absolute", bottom: "100%", left: 0, right: 0,
          padding: "10px 16px 6px",
          background: isDark ? "rgba(10,10,16,0.96)" : "rgba(245,243,239,0.96)",
          backdropFilter: "blur(16px)",
          borderTop: "1px solid " + theme.border,
          animation: "fadeInUp 0.15s ease",
        }}>
          {expandedPanel === "key" && pt.noteNames && (
            <TransposePanel
              enabled={pt.transposeEnabled} onToggle={() => pt.setTransposeEnabled(!pt.transposeEnabled)}
              noteNames={pt.noteNames} targetKey={pt.targetKey} onSelectKey={pt.setTargetKey}
              pitchOffset={pt.pitchOffset} onPitchOffset={pt.setPitchOffset}
              useFlats={pt.useFlats} onToggleFlats={pt.setUseFlats}
              uploadKey={pt.uploadKey} getSemitones={pt.getSemitones} activeSampleKey={activeSample.key}
            />
          )}
          {expandedPanel === "bpm" && pt.bpmOptions && (
            <TimestretchPanel
              enabled={pt.timestrechEnabled} onToggle={() => pt.setTimestrechEnabled(!pt.timestrechEnabled)}
              bpmOptions={pt.bpmOptions} tempoMultiplier={pt.tempoMultiplier}
              onSelectMultiplier={pt.setTempoMultiplier} uploadBpm={pt.uploadBpm}
            />
          )}
        </div>
      )}

      {/* Main player row */}
      <div style={{
        display: "flex", alignItems: "center", gap: 10,
        padding: "8px 16px", minHeight: 56,
      }}>
        {/* Play/Pause */}
        <button
          onClick={() => audio.toggle(activeSample.path, activeSample.id, { key: activeSample.key, bpm: activeSample.bpm })}
          style={{
            width: 34, height: 34, borderRadius: "50%", border: "none", flexShrink: 0,
            background: theme.gradient, cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 2px 12px rgba(139,92,246,0.2), 0 2px 6px rgba(6,182,212,0.15)",
            transition: "transform 0.1s",
          }}
        >
          {isPlaying
            ? <svg width="10" height="10" viewBox="0 0 14 14" fill="#fff"><rect x="2" y="1" width="4" height="12" rx="1" /><rect x="8" y="1" width="4" height="12" rx="1" /></svg>
            : <svg width="10" height="10" viewBox="0 0 14 14" fill="#fff"><path d="M3 1v12l10-6z" /></svg>
          }
        </button>

        {/* Sample info */}
        <div style={{ flex: "0 0 150px", minWidth: 0 }}>
          <div style={{
            fontSize: 12, fontWeight: 500, color: theme.text,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            fontFamily: SERIF,
          }}>
            {activeSample.clean_name || activeSample.name}
          </div>
          <div style={{
            fontSize: 9, color: theme.textMuted, fontFamily: AF,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {activeSample.type_label || "Sound"}
            {activeSample.source && activeSample.source !== "local" ? " \u00B7 " + activeSample.source : ""}
          </div>
        </div>

        {/* Current time */}
        <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: MONO, flexShrink: 0 }}>
          {formatTime(audio.currentTime)}
        </span>

        {/* Waveform scrubber */}
        <div style={{ flex: 1, minWidth: 80, padding: "0 2px" }}>
          <RealWaveform
            peaks={waveformPeaks}
            progress={isPlaying ? audio.progress : 0}
            height={28}
            onClick={audio.seek}
          />
        </div>

        {/* Duration */}
        <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: MONO, flexShrink: 0 }}>
          {formatTime(audio.duration)}
        </span>

        {/* KEY button */}
        <button onClick={() => togglePanel("key")}
          style={{
            display: "flex", alignItems: "center", gap: 3,
            padding: "4px 8px", borderRadius: 5,
            border: "1px solid " + (expandedPanel === "key" ? "rgba(139,92,246,0.4)" : theme.borderLight),
            background: expandedPanel === "key" ? (isDark ? "rgba(139,92,246,0.1)" : "rgba(139,92,246,0.06)") : "transparent",
            cursor: "pointer", transition: "all 0.15s",
          }}
        >
          <span style={{ fontSize: 8, fontWeight: 600, color: theme.textMuted, fontFamily: AF }}>KEY</span>
          <span style={{ fontSize: 10, fontWeight: 700, color: pt.transposeEnabled ? "#8B5CF6" : theme.text, fontFamily: MONO }}>
            {activeSample.key || pt.uploadKey || "\u2014"}
          </span>
          {pt.transposeEnabled && <span style={{ width: 4, height: 4, borderRadius: "50%", background: "#8B5CF6" }} />}
        </button>

        {/* BPM button */}
        <button onClick={() => togglePanel("bpm")}
          style={{
            display: "flex", alignItems: "center", gap: 3,
            padding: "4px 8px", borderRadius: 5,
            border: "1px solid " + (expandedPanel === "bpm" ? "rgba(6,182,212,0.4)" : theme.borderLight),
            background: expandedPanel === "bpm" ? (isDark ? "rgba(6,182,212,0.1)" : "rgba(6,182,212,0.06)") : "transparent",
            cursor: "pointer", transition: "all 0.15s",
          }}
        >
          <span style={{ fontSize: 8, fontWeight: 600, color: theme.textMuted, fontFamily: AF }}>BPM</span>
          <span style={{ fontSize: 10, fontWeight: 700, color: pt.timestrechEnabled ? "#06B6D4" : theme.text, fontFamily: MONO }}>
            {activeSample.bpm ? Math.round(activeSample.bpm) : pt.uploadBpm ? Math.round(pt.uploadBpm) : "\u2014"}
          </span>
          {pt.timestrechEnabled && <span style={{ width: 4, height: 4, borderRadius: "50%", background: "#06B6D4" }} />}
        </button>

        {/* Bridge indicator */}
        {bridgeConnected && (
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#06B6D4", boxShadow: "0 0 6px rgba(6,182,212,0.5)", flexShrink: 0 }} title="Bridge Connected" />
        )}

        {/* Separator */}
        <div style={{ width: 1, height: 20, background: theme.borderLight, flexShrink: 0 }} />

        {/* Favorite */}
        <button onClick={() => onToggleFav?.(activeSample.id)}
          style={{ background: "none", border: "none", fontSize: 14, cursor: "pointer", color: isFavorite ? "#D97706" : theme.textFaint, padding: "2px 3px" }}>
          {isFavorite ? "\u2605" : "\u2606"}
        </button>

        {/* Similar Sounds */}
        <button onClick={() => onFindSimilar?.(activeSample)}
          title="Similar Sounds" style={{ background: "none", border: "none", color: theme.textMuted, cursor: "pointer", padding: "2px 3px" }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="9" cy="12" r="5" /><circle cx="15" cy="12" r="5" />
          </svg>
        </button>

        {/* Mix toggle */}
        <button onClick={onToggleMix}
          style={{
            padding: "4px 8px", borderRadius: 4, fontSize: 9, fontWeight: 600, cursor: "pointer",
            border: "1px solid " + theme.border, fontFamily: AF,
            background: audio.mixMode ? (isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)") : "transparent",
            color: audio.mixMode ? theme.text : theme.textMuted,
            transition: "all 0.15s",
          }}
        >{audio.mixMode ? "\u25C9 Mix" : "Mix"}</button>
      </div>

      {/* Mix mode volume sliders */}
      {audio.mixMode && (
        <div style={{ display: "flex", gap: 12, padding: "0 60px 6px" }}>
          <div style={{ flex: 1 }}><VolumeSlider value={audio.trackVol} onChange={audio.setTrackVol} label="Track" /></div>
          <div style={{ flex: 1 }}><VolumeSlider value={audio.sampleVol} onChange={audio.setSampleVol} label="Sample" /></div>
        </div>
      )}
    </div>
  );
});
