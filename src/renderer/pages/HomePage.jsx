/**
 * RESONATE — Home Page.
 * Upload landing screen: LogoBlend animation, upload card, drag/drop,
 * bridge analyze button, stats row, session load button, keyboard shortcut hint.
 */

import { useState, useMemo, memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { SERIF, MONO, AF } from "../theme/fonts";

export const HomePage = memo(function HomePage({
  LogoBlend,
  error,
  dragOver, onDragOver, onDragLeave, onDrop,
  onUpload,
  bridge, onAnalyzeFromBridge,
  samples,
  sessions, onShowSessions,
}) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  // Compute category count from actual samples data
  const categoryCount = useMemo(() => {
    if (!samples || samples.length === 0) return 0;
    const unique = new Set();
    for (const s of samples) {
      if (s.category) unique.add(s.category.toLowerCase());
      if (s.sub_category) unique.add(s.sub_category.toLowerCase());
    }
    return unique.size;
  }, [samples]);

  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", flex: 1, padding: 40, textAlign: "center",
      position: "relative", overflow: "hidden",
    }}>
      {/* Ambient background layers */}
      <div style={{
        position: "absolute", inset: 0, pointerEvents: "none",
        background: isDark
          ? "radial-gradient(ellipse 80% 60% at 50% 30%, rgba(139,92,246,0.06) 0%, transparent 70%)"
          : "radial-gradient(ellipse 80% 60% at 50% 30%, rgba(139,92,246,0.05) 0%, transparent 70%)",
      }} />
      <div style={{
        position: "absolute", inset: 0, pointerEvents: "none",
        background: isDark
          ? "radial-gradient(ellipse 50% 40% at 70% 80%, rgba(6,182,212,0.05) 0%, transparent 60%)"
          : "radial-gradient(ellipse 50% 40% at 70% 80%, rgba(6,182,212,0.04) 0%, transparent 60%)",
      }} />

      {/* Logo + Title */}
      <div style={{ marginBottom: 40, animation: "fadeInUp 0.5s ease", position: "relative", zIndex: 1 }}>
        <div style={{ position: "relative", width: 100, height: 100, margin: "0 auto 28px" }}>
          {LogoBlend && <LogoBlend size={100} isDark={isDark} />}
          {/* Circular equalizer — radial bars that pulse like music */}
          <div style={{ position: "absolute", inset: -40, width: 180, height: 180, pointerEvents: "none" }}>
            {Array.from({ length: 48 }, (_, i) => {
              const angle = (i / 48) * 360;
              const colors = ["#8B5CF6", "#7C3AED", "#A855F7", "#6366F1", "#06B6D4", "#22D3EE"];
              const c = colors[i % colors.length];
              const dur = 0.8 + (i % 6) * 0.2;
              const delay = (i % 9) * 0.1;
              return (
                <div key={i} style={{
                  position: "absolute", left: "50%", top: "50%",
                  width: 2.5, height: 18,
                  transformOrigin: "center 0px",
                  transform: `rotate(${angle}deg) translateY(-76px)`,
                }}>
                  <div style={{
                    width: "100%", height: "100%", borderRadius: 1.5,
                    background: `linear-gradient(to top, ${c}, ${c}80)`,
                    opacity: isDark ? 0.5 : 0.4,
                    transformOrigin: "bottom center",
                    animation: `eqBar ${dur}s ease-in-out infinite ${delay}s`,
                    boxShadow: isDark ? `0 0 6px ${c}50` : `0 0 3px ${c}30`,
                  }} />
                </div>
              );
            })}
          </div>
          {/* Warm glow behind logo */}
          <div style={{
            position: "absolute", inset: -30, borderRadius: "50%", pointerEvents: "none",
            background: isDark
              ? "radial-gradient(circle, rgba(139,92,246,0.1) 0%, rgba(139,92,246,0.05) 35%, transparent 65%)"
              : "radial-gradient(circle, rgba(139,92,246,0.04) 0%, transparent 60%)",
            filter: "blur(18px)",
            animation: "logoPulse 4s ease-in-out infinite",
          }} />
        </div>
        <h1 className="gradient-text shimmer-text" style={{
          fontSize: 30, fontWeight: 300, letterSpacing: 12, fontFamily: AF, margin: "0 0 8px",
        }}>RESONATE</h1>
        <p style={{
          fontSize: 9, color: theme.textMuted, letterSpacing: 6,
          textTransform: "uppercase", fontFamily: AF, opacity: 0.5,
        }}>Production Intelligence</p>
      </div>

      {error && (
        <div style={{
          padding: "8px 14px", borderRadius: 6, fontSize: 11, marginBottom: 16,
          maxWidth: 440, position: "relative", zIndex: 1, color: theme.red,
          background: isDark ? "rgba(220,38,38,0.12)" : "rgba(220,38,38,0.06)",
          border: "1px solid rgba(220,38,38,0.12)",
        }}>{error}</div>
      )}

      {/* Upload Card */}
      <div
        className="upload-card"
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={onUpload}
        style={{
          width: "100%", maxWidth: 440, padding: "44px 32px", borderRadius: 16,
          cursor: "pointer", border: "1.5px solid transparent",
          background: isDark
            ? (dragOver ? "rgba(139,92,246,0.06)" : "linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.01) 100%)")
            : (dragOver ? "rgba(139,92,246,0.04)" : "linear-gradient(180deg, rgba(255,255,255,0.85) 0%, rgba(255,255,255,0.6) 100%)"),
          transition: "all 0.3s ease",
          boxShadow: isDark
            ? "0 2px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03)"
            : "0 2px 16px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.8)",
          backdropFilter: "blur(16px)",
          position: "relative", zIndex: 1,
        }}
      >
        <svg width="28" height="28" viewBox="0 0 44 44" fill="none" style={{ margin: "0 auto 14px", display: "block", opacity: 0.5 }}>
          <path d="M22 8v22M15 15l7-7 7 7" stroke={dragOver ? theme.text : theme.textMuted} strokeWidth="1.5" strokeLinecap="round" />
          <path d="M8 32v4a2 2 0 002 2h24a2 2 0 002-2v-4" stroke={dragOver ? theme.text : theme.textMuted} strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <div style={{ fontSize: 15, fontWeight: 400, color: theme.text, marginBottom: 5, fontFamily: SERIF }}>
          Drop your mixdown here
        </div>
        <div style={{ fontSize: 11, color: theme.textMuted }}>or click to browse</div>
        <div style={{ display: "flex", gap: 6, justifyContent: "center", marginTop: 16 }}>
          {["WAV", "MP3", "FLAC", "AIFF"].map(f => (
            <span key={f} style={{
              padding: "3px 10px", borderRadius: 4, fontSize: 8, fontWeight: 600,
              background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)",
              color: theme.tagText, letterSpacing: 1.2,
            }}>{f}</span>
          ))}
        </div>
      </div>

      {/* Analyze from DAW — shown when bridge is connected */}
      {bridge?.connected && (
        <button onClick={onAnalyzeFromBridge} style={{
          width: "100%", maxWidth: 440, marginTop: 12, padding: "12px 0",
          borderRadius: 10, border: "1px solid rgba(34,197,94,0.25)",
          background: "rgba(34,197,94,0.06)", color: "#22C55E",
          cursor: "pointer", fontWeight: 600, fontSize: 12, fontFamily: AF,
          display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
          position: "relative", zIndex: 1, backdropFilter: "blur(12px)",
        }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", animation: "pulse 2s ease infinite" }} />
          Analyze from DAW ({bridge.dawBpm.toFixed(0)} BPM)
        </button>
      )}

      {/* Thin accent divider — polished glow */}
      <div style={{
        width: 40, height: 1, margin: "32px 0", position: "relative", zIndex: 1,
        borderRadius: 1,
        background: "linear-gradient(90deg, #8B5CF6, #A5F3FC, #06B6D4)",
        boxShadow: "0 0 8px rgba(6,182,212,0.2), 0 0 4px rgba(139,92,246,0.15)",
        opacity: 0.5,
      }} />

      {/* Stats */}
      <div style={{ display: "flex", gap: 56, position: "relative", zIndex: 1 }}>
        {[
          { v: samples?.length || "\u2014", l: "Samples" },
          { v: categoryCount || "\u2014", l: "Categories" },
          { v: "RPM", l: "Engine" },
        ].map((s, i) => (
          <div key={s.l} style={{ textAlign: "center", animation: `fadeInUp 0.5s ease ${0.1 + i * 0.08}s both` }}>
            <div className="gradient-text" style={{
              fontSize: 28, fontWeight: 200, fontFamily: MONO, lineHeight: 1, letterSpacing: -1,
            }}>{typeof s.v === "number" ? s.v.toLocaleString() : s.v}</div>
            <div style={{
              fontSize: 7, color: theme.textFaint, textTransform: "uppercase",
              letterSpacing: 4, marginTop: 8, fontFamily: AF, fontWeight: 500,
            }}>{s.l}</div>
          </div>
        ))}
      </div>

      {/* Session History + Shortcut Hints */}
      <div style={{ marginTop: 32, display: "flex", gap: 12, position: "relative", zIndex: 1 }}>
        {sessions?.length > 0 && (
          <button onClick={onShowSessions} style={{
            fontSize: 10, padding: "7px 18px", borderRadius: 8,
            border: "1px solid " + theme.border, color: theme.textSec,
            cursor: "pointer", fontFamily: AF, transition: "all 0.2s",
            backdropFilter: "blur(8px)",
            background: isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.7)",
          }}>Load Session ({sessions.length})</button>
        )}
      </div>
      <div style={{
        marginTop: 18, fontSize: 9, color: theme.textFaint, fontFamily: AF,
        position: "relative", zIndex: 1, display: "flex", alignItems: "center", gap: 5,
      }}>
        Press <span style={{
          fontFamily: MONO, padding: "2px 7px", borderRadius: 4, fontSize: 10,
          background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)",
          border: "1px solid " + theme.borderLight,
        }}>?</span> for shortcuts
      </div>
      <div style={{
        marginTop: 24, fontSize: 8, fontFamily: MONO, position: "relative", zIndex: 1,
        letterSpacing: 2, textTransform: "uppercase", fontWeight: 500,
        color: isDark ? "rgba(255,255,255,0.4)" : "#111111",
      }}>
        SONIQlabs · v0.1
      </div>
    </div>
  );
});
