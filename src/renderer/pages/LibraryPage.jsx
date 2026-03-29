/**
 * RESONATE — Library Page.
 * Favorites, rated samples, and recent history.
 * Three tabs: Favorites / Rated / Recent.
 */

import { useState, memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { SERIF, MONO, AF } from "../theme/fonts";
import { SampleRowV2 } from "../components/SampleRowV2";

export const LibraryPage = memo(function LibraryPage({
  samples, favorites, ratings, audio, activeSample,
  onPlay, onToggleFav, onFindSimilar, onFindLayers,
}) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const [libTab, setLibTab] = useState("favorites");

  const favoriteSamples = (samples || []).filter(s => favorites.has(s.id));
  const ratedSamples = (samples || []).filter(s => ratings[s.id]);
  const displayList = libTab === "favorites" ? favoriteSamples
    : libTab === "rated" ? ratedSamples
    : samples || [];

  const tabs = [
    { id: "favorites", label: "Favorites", count: favoriteSamples.length },
    { id: "rated", label: "Rated", count: ratedSamples.length },
    { id: "recent", label: "All Samples", count: samples?.length || 0 },
  ];

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Header */}
      <div style={{ padding: "20px 24px 0" }}>
        <h2 style={{
          fontSize: 20, fontWeight: 300, color: theme.text,
          fontFamily: SERIF, letterSpacing: -0.5, margin: "0 0 6px",
        }}>Library</h2>
        <p style={{ fontSize: 11, color: theme.textMuted, fontFamily: AF, margin: "0 0 16px" }}>
          Your saved and rated samples
        </p>
      </div>

      {/* Tabs */}
      <div style={{
        display: "flex", padding: "0 24px",
        borderBottom: "1px solid " + theme.border,
      }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setLibTab(t.id)} style={{
            padding: "10px 14px", border: "none", background: "transparent",
            color: libTab === t.id ? theme.text : theme.textMuted,
            fontSize: 11, fontWeight: libTab === t.id ? 600 : 400,
            cursor: "pointer", fontFamily: AF,
            borderBottom: libTab === t.id ? "2px solid " + theme.accent : "2px solid transparent",
            transition: "all 0.2s ease",
          }}>
            {t.label}
            <span style={{
              marginLeft: 4, fontSize: 9, padding: "1px 4px", borderRadius: 4,
              background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)",
            }}>{t.count}</span>
          </button>
        ))}
      </div>

      {/* Column Headers */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "28px 1fr 50px 38px 38px 38px 80px",
        alignItems: "center", gap: 4, padding: "6px 14px",
        fontSize: 8, color: theme.textFaint, textTransform: "uppercase",
        letterSpacing: 1.5, background: theme.surface,
        borderBottom: "1px solid " + theme.borderLight, fontFamily: AF,
      }}>
        <span /><span>Name</span><span style={{ textAlign: "center" }}>Match</span>
        <span style={{ textAlign: "center" }}>Key</span><span style={{ textAlign: "center" }}>BPM</span>
        <span style={{ textAlign: "center" }}>Len</span><span />
      </div>

      {/* Sample List */}
      <div style={{ flex: 1, overflowY: "auto", background: theme.surface }}>
        {displayList.length === 0 ? (
          <div style={{ padding: 40, textAlign: "center", color: theme.textMuted, fontSize: 12, fontFamily: AF }}>
            {libTab === "favorites" ? "No favorites yet. Click the star on any sample." :
             libTab === "rated" ? "No rated samples yet." : "No samples in library."}
          </div>
        ) : (
          displayList.map(s => (
            <div key={s.id} style={{ height: 46 }}>
              <SampleRowV2
                sample={s}
                isActive={activeSample?.id === s.id}
                isPlaying={audio.currentId === s.id && audio.playing}
                onPlay={onPlay}
                isFavorite={favorites.has(s.id)}
                onToggleFav={onToggleFav}
                onFindSimilar={onFindSimilar}
                onFindLayers={onFindLayers}
              />
            </div>
          ))
        )}
      </div>
    </div>
  );
});
