/**
 * RESONATE — Mix Preview Bar.
 * Fixed-position bar at the bottom of the screen shown during in-context
 * sample preview (mix mode). Dual volume sliders, A/B toggle, progress bar.
 */

import { useState, useCallback, memo } from "react";
import { SERIF, AF, MONO } from "../theme/fonts";

function formatTime(s) {
  if (!s || !isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ":" + String(sec).padStart(2, "0");
}

export const MixPreviewBar = memo(function MixPreviewBar({ theme, isDark, audio, activeSample, fileName }) {
  const [abMuted, setAbMuted] = useState(false);
  const [hovStop, setHovStop] = useState(false);
  const [hovAB, setHovAB] = useState(false);

  const handleSeek = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audio.seek(pct);
  }, [audio]);

  const toggleAB = useCallback(() => {
    setAbMuted(prev => {
      const next = !prev;
      // Mute or unmute the sample channel for A/B comparison
      audio.setSampleVol(next ? 0 : 1.0);
      return next;
    });
  }, [audio]);

  const handleStop = useCallback(() => {
    audio.stop();
    setAbMuted(false);
  }, [audio]);

  if (!audio.mixMode || !audio.playing || !activeSample) return null;

  const sampleName = activeSample.clean_name || activeSample.name || "Sample";
  const trackName = fileName || "Track";
  const progress = audio.progress || 0;

  return (
    <div style={{
      position: "fixed",
      bottom: 0,
      left: 0,
      right: 0,
      height: 64,
      zIndex: 9999,
      display: "flex",
      alignItems: "center",
      gap: 12,
      padding: "0 20px",
      background: isDark
        ? "rgba(13, 13, 18, 0.85)"
        : "rgba(255, 255, 255, 0.85)",
      backdropFilter: "blur(24px) saturate(1.4)",
      WebkitBackdropFilter: "blur(24px) saturate(1.4)",
      borderTop: "1px solid " + (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"),
      boxShadow: isDark
        ? "0 -4px 24px rgba(0,0,0,0.4)"
        : "0 -4px 24px rgba(0,0,0,0.08)",
    }}>
      {/* Track name (left) */}
      <div style={{ flex: "0 0 120px", minWidth: 0 }}>
        <div style={{
          fontSize: 9,
          color: theme.textFaint,
          fontFamily: AF,
          textTransform: "uppercase",
          letterSpacing: 1,
          marginBottom: 2,
        }}>Track</div>
        <div style={{
          fontSize: 11,
          color: theme.textSec,
          fontFamily: SERIF,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}>{trackName}</div>
      </div>

      {/* Track volume slider */}
      <div style={{ flex: "0 0 80px", display: "flex", flexDirection: "column", gap: 2 }}>
        <div style={{ fontSize: 8, color: theme.textFaint, fontFamily: AF, textTransform: "uppercase", letterSpacing: 0.8 }}>Mix</div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={audio.trackVol}
          onChange={e => audio.setTrackVol(parseFloat(e.target.value))}
          style={{
            width: "100%",
            height: 3,
            appearance: "none",
            WebkitAppearance: "none",
            background: `linear-gradient(to right, ${theme.textSec} ${audio.trackVol * 100}%, ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"} ${audio.trackVol * 100}%)`,
            borderRadius: 2,
            outline: "none",
            cursor: "pointer",
            accentColor: "#D946EF",
          }}
        />
      </div>

      {/* Progress section */}
      <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: MONO, flexShrink: 0 }}>
          {formatTime(audio.currentTime)}
        </span>
        <div
          onClick={handleSeek}
          style={{
            flex: 1,
            height: 4,
            borderRadius: 2,
            background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)",
            cursor: "pointer",
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div style={{
            position: "absolute",
            top: 0,
            left: 0,
            height: "100%",
            width: (progress * 100) + "%",
            borderRadius: 2,
            background: "linear-gradient(90deg, #D946EF, #06B6D4)",
            transition: "width 0.1s linear",
          }} />
        </div>
        <span style={{ fontSize: 9, color: theme.textMuted, fontFamily: MONO, flexShrink: 0 }}>
          {formatTime(audio.duration)}
        </span>
      </div>

      {/* Sample volume slider */}
      <div style={{ flex: "0 0 80px", display: "flex", flexDirection: "column", gap: 2 }}>
        <div style={{ fontSize: 8, color: theme.textFaint, fontFamily: AF, textTransform: "uppercase", letterSpacing: 0.8 }}>Sample</div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={abMuted ? 0 : audio.sampleVol}
          onChange={e => { const v = parseFloat(e.target.value); audio.setSampleVol(v); if (v > 0) setAbMuted(false); }}
          style={{
            width: "100%",
            height: 3,
            appearance: "none",
            WebkitAppearance: "none",
            background: `linear-gradient(to right, #D946EF ${(abMuted ? 0 : audio.sampleVol) * 100}%, ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"} ${(abMuted ? 0 : audio.sampleVol) * 100}%)`,
            borderRadius: 2,
            outline: "none",
            cursor: "pointer",
            accentColor: "#D946EF",
          }}
        />
      </div>

      {/* Sample name (right) */}
      <div style={{ flex: "0 0 120px", minWidth: 0, textAlign: "right" }}>
        <div style={{
          fontSize: 9,
          color: theme.textFaint,
          fontFamily: AF,
          textTransform: "uppercase",
          letterSpacing: 1,
          marginBottom: 2,
        }}>Sample</div>
        <div style={{
          fontSize: 11,
          color: "#D946EF",
          fontFamily: SERIF,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}>{sampleName}</div>
      </div>

      {/* A/B Toggle */}
      <button
        onClick={toggleAB}
        onMouseEnter={() => setHovAB(true)}
        onMouseLeave={() => setHovAB(false)}
        title={abMuted ? "Unmute sample (A/B)" : "Mute sample to compare (A/B)"}
        style={{
          width: 32,
          height: 32,
          borderRadius: 6,
          border: "1px solid " + (abMuted ? "#D946EF" : theme.border),
          background: abMuted
            ? "rgba(217,70,239,0.12)"
            : hovAB
              ? (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)")
              : "transparent",
          color: abMuted ? "#D946EF" : theme.textSec,
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          fontSize: 10,
          fontWeight: 700,
          fontFamily: AF,
          transition: "all 0.15s ease",
        }}
      >
        A/B
      </button>

      {/* Stop button */}
      <button
        onClick={handleStop}
        onMouseEnter={() => setHovStop(true)}
        onMouseLeave={() => setHovStop(false)}
        title="Stop preview"
        style={{
          width: 32,
          height: 32,
          borderRadius: 6,
          border: "1px solid " + theme.border,
          background: hovStop
            ? (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)")
            : "transparent",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          transition: "all 0.15s ease",
        }}
      >
        <svg width="10" height="10" viewBox="0 0 14 14" fill={theme.textSec}>
          <rect x="2" y="2" width="10" height="10" rx="1.5" />
        </svg>
      </button>
    </div>
  );
});
