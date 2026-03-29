/**
 * RESONATE — Sounds Page.
 * Main browse experience: FilterBar + AnalysisSidebar + virtual-scrolled SampleRowV2 list.
 * Splice-style results view with AI summary card, smart collections, tabs, batch export.
 */

import { useState, useRef, useEffect, useCallback, useMemo, memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { SERIF, MONO, AF } from "../theme/fonts";
import { FilterBar } from "../components/FilterBar";
import { SampleRowV2 } from "../components/SampleRowV2";
import { AnalysisSidebar } from "../components/AnalysisSidebar";
import { SkeletonRow } from "../components/SkeletonRow";
import { VersionTimeline } from "../components/VersionTimeline";
import { SmartCollection } from "../components/SmartCollection";
import { ProducerDNA } from "../components/ProducerDNA";

const ROW_HEIGHT = 46;
const BUFFER_ROWS = 8;

export const SoundsPage = memo(function SoundsPage({
  // Data
  analysisResult, fileName, samples, displaySamples,
  // Filters
  category, onCategoryChange, categories,
  selectedKey, onKeyChange,
  sourceFilter, onSourceChange, sourceCounts,
  moodFilter, onMoodChange,
  sortBy, onSortChange,
  search,
  // Tabs
  tab, onTabChange,
  viewMode, onViewModeChange, v2Available, v2Loading,
  // Player
  audio, activeSample, onPlay,
  // Favorites & Ratings
  favorites, onToggleFav,
  // Similarity & Layering
  onFindSimilar, onFindLayers,
  // Batch
  checkedSamples, onCheck, onSelectAll, onClearChecked, onExportChecked,
  // Analysis sidebar
  mixNeeds, gapAnalysis, prevReadiness, ringGlowing,
  chartComparison, bridge,
  analyzerExpanded, onToggleAnalyzer,
  onSaveSession, onShowSessions, sessions,
  onReAnalyze, reanalyzing,
  // Collections
  collections,
  // Versions
  versions, onSaveVersion, onSelectVersion,
  // ProducerDNA
  dnaProfile, dnaTraining, onTrainDNA,
  // Index status
  indexProgress,
  // API
  api,
  // Drag
  onDragStart,
  // Bridge analyze
  onAnalyzeFromBridge,
  // Pitch/Tempo display in sidebar
  pitchTempo,
  // v2 state
  mixProfile,
}) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  // Virtual scrolling
  const scrollRef = useRef(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(600);

  const a = analysisResult?.analysis || {};

  // Filter samples
  const filtered = useMemo(() => (displaySamples || []).filter(s => {
    if (category !== "all" && s.category?.toLowerCase() !== category.toLowerCase()) return false;
    if (selectedKey !== "Any" && s.key !== "\u2014" && s.key !== selectedKey) return false;
    if (search && !(s.clean_name || s.name).toLowerCase().includes(search.toLowerCase())) return false;
    if (tab === "favorites" && !favorites.has(s.id)) return false;
    if (sourceFilter !== "all" && (s.source || "local") !== sourceFilter) return false;
    if (moodFilter !== "all" && (s.mood || "neutral") !== moodFilter) return false;
    return true;
  }).sort((a, b) => (b.match || 0) - (a.match || 0)), [displaySamples, category, selectedKey, search, tab, favorites, sourceFilter, moodFilter]);

  const totalHeight = filtered.length * ROW_HEIGHT;
  const startIdx = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER_ROWS);
  const endIdx = Math.min(filtered.length, Math.ceil((scrollTop + containerHeight) / ROW_HEIGHT) + BUFFER_ROWS);
  const visibleSamples = filtered.slice(startIdx, endIdx);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => setContainerHeight(el.clientHeight);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const handleScroll = useCallback((e) => setScrollTop(e.target.scrollTop), []);

  // Build category objects for FilterBar
  const categoryObjects = useMemo(() => {
    if (!displaySamples) return [];
    const counts = {};
    for (const s of displaySamples) {
      const cat = (s.category || "").toLowerCase();
      if (cat && cat !== "all") {
        counts[cat] = (counts[cat] || 0) + 1;
      }
    }
    return Object.entries(counts).map(([name, count]) => ({ name, count }));
  }, [displaySamples]);

  return (
    <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
      {/* Analysis Sidebar */}
      <div style={{
        width: 250, borderRight: "1px solid " + theme.border,
        padding: 12, overflowY: "auto", flexShrink: 0,
        background: theme.surface,
      }}>
      <AnalysisSidebar
        analysisResult={analysisResult}
        fileName={fileName}
        analyzerExpanded={analyzerExpanded}
        setAnalyzerExpanded={onToggleAnalyzer}
        mixNeeds={mixNeeds}
        gapAnalysis={gapAnalysis}
        prevReadiness={prevReadiness}
        ringGlowing={ringGlowing}
        chartComparison={chartComparison}
        bridge={bridge}
        reanalyzing={reanalyzing}
        reAnalyze={onReAnalyze}
        saveSession={onSaveSession}
        sessions={sessions}
        setShowSessions={onShowSessions}
        activeSample={activeSample}
        analyzeFromBridge={onAnalyzeFromBridge}
      />
      </div>

      {/* Main content */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Version Timeline */}
        {versions?.length > 0 && (
          <VersionTimeline
            versions={versions}
            onSaveVersion={onSaveVersion}
            onSelectVersion={onSelectVersion}
          />
        )}

        {/* Curated Packs — sleek horizontal strip */}
        {collections?.length > 0 && collections.some(c => c.samples?.length > 0) && (
          <div style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: "8px 14px",
            borderBottom: "1px solid " + theme.borderLight,
            background: isDark ? "rgba(255,255,255,0.015)" : "rgba(0,0,0,0.01)",
            overflowX: "auto", scrollbarWidth: "none",
          }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 4, flexShrink: 0,
              padding: "2px 8px 2px 0", borderRight: "1px solid " + theme.borderLight,
              marginRight: 2,
            }}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.7 }}>
                <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
                <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
              </svg>
              <span style={{ fontSize: 8, fontWeight: 700, fontFamily: AF, color: theme.textMuted, letterSpacing: 1, textTransform: "uppercase" }}>Packs</span>
            </div>
            {collections.filter(c => c.samples?.length > 0).map(c => {
              const PACK_COLORS = { target: "#8B5CF6", sparkle: "#F59E0B", rhythm: "#22C55E", crown: "#3B82F6" };
              const color = PACK_COLORS[c.icon] || "#8B5CF6";
              return (
                <SmartCollection
                  key={c.id}
                  collection={c}
                  onPreview={(sample) => {
                    if (sample.filepath) {
                      const s = { path: sample.filepath, id: sample.filepath, name: sample.name };
                      onPlay(s);
                    }
                  }}
                  onExport={(collectionId) => api?.exportCollection(collectionId)}
                />
              );
            })}
          </div>
        )}

        {/* Filter Bar */}
        <FilterBar
          category={category} onCategoryChange={onCategoryChange}
          categories={categoryObjects}
          selectedKey={selectedKey} onKeyChange={onKeyChange}
          sourceFilter={sourceFilter} onSourceChange={onSourceChange}
          sourceCounts={sourceCounts}
          moodFilter={moodFilter} onMoodChange={onMoodChange}
          sortBy={sortBy} onSortChange={onSortChange}
          totalCount={displaySamples?.length || 0}
        />

        {/* Tabs */}
        <div style={{ display: "flex", background: theme.surface, borderBottom: "1px solid " + theme.border, padding: "0 14px", alignItems: "center" }}>
          {[
            { id: "matched", l: "AI Matched", c: displaySamples?.length || 0 },
            { id: "favorites", l: "Favorites", c: favorites.size },
            { id: "dna", l: "Your DNA", c: null, color: "#8B5CF6" },
          ].map(t => (
            <button key={t.id} onClick={() => onTabChange(t.id)} style={{
              padding: "10px 14px", border: "none", background: "transparent",
              color: tab === t.id ? (t.color || theme.text) : theme.textMuted,
              fontSize: 11, fontWeight: tab === t.id ? 600 : 400, cursor: "pointer",
              borderBottom: tab === t.id ? "2px solid " + (t.color || theme.accent) : "2px solid transparent",
              fontFamily: AF, transition: "all 0.2s ease",
            }}>
              {t.l}{t.c != null && <span style={{ marginLeft: 4, fontSize: 9, padding: "1px 4px", borderRadius: 4, background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)" }}>{t.c}</span>}
            </button>
          ))}
          {/* v2 view mode toggle */}
          {v2Available && (
            <div style={{ marginLeft: "auto", display: "flex", borderRadius: 5, overflow: "hidden", border: "1px solid " + theme.borderLight }}>
              <button onClick={() => onViewModeChange("smart")} style={{
                padding: "5px 10px", border: "none", cursor: "pointer", fontFamily: AF, fontSize: 9,
                background: viewMode === "smart" ? (isDark ? "rgba(139,92,246,0.15)" : "rgba(139,92,246,0.1)") : "transparent",
                color: viewMode === "smart" ? "#8B5CF6" : theme.textMuted,
                fontWeight: viewMode === "smart" ? 700 : 400,
              }}>
                Smart Match{v2Loading ? " ..." : ""}
              </button>
              <button onClick={() => onViewModeChange("all")} style={{
                padding: "5px 10px", border: "none", borderLeft: "1px solid " + theme.borderLight, cursor: "pointer", fontFamily: AF, fontSize: 9,
                background: viewMode === "all" ? (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)") : "transparent",
                color: viewMode === "all" ? theme.text : theme.textMuted,
                fontWeight: viewMode === "all" ? 700 : 400,
              }}>
                Full Library
              </button>
            </div>
          )}
        </div>

        {/* Batch Export Toolbar */}
        {checkedSamples.size > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 14px", background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)", borderBottom: "1px solid " + theme.borderLight }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: theme.text, fontFamily: AF }}>{checkedSamples.size} selected</span>
            <button onClick={onExportChecked} style={{ fontSize: 9, padding: "4px 12px", borderRadius: 5, border: "none", background: isDark ? theme.text : "#1A1A1A", color: isDark ? "#0D0D12" : "#fff", cursor: "pointer", fontWeight: 600, fontFamily: AF }}>Export Kit (.zip)</button>
            <button onClick={onSelectAll} style={{ fontSize: 9, padding: "4px 10px", borderRadius: 5, border: "1px solid " + theme.border, background: "transparent", color: theme.textSec, cursor: "pointer", fontFamily: AF }}>Select All</button>
            <button onClick={onClearChecked} style={{ fontSize: 9, padding: "4px 10px", borderRadius: 5, border: "1px solid " + theme.border, background: "transparent", color: theme.textSec, cursor: "pointer", fontFamily: AF }}>Clear</button>
          </div>
        )}

        {/* Content Area */}
        {tab === "dna" ? (
          <ProducerDNA
            data={dnaProfile} training={dnaTraining}
            onTrain={onTrainDNA}
            theme={theme} isDark={isDark}
          />
        ) : (
          <>
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

            {/* Virtual-scrolled sample list */}
            <div ref={scrollRef} onScroll={handleScroll} style={{ flex: 1, overflowY: "auto", background: theme.surface }}>
              {samples?.length === 0 && indexProgress && !indexProgress.done ? (
                Array.from({ length: 12 }).map((_, i) => <SkeletonRow key={i} />)
              ) : filtered.length > 0 ? (
                <div style={{ height: totalHeight, position: "relative" }}>
                  <div style={{ position: "absolute", top: startIdx * ROW_HEIGHT, width: "100%" }}>
                    {visibleSamples.map((s) => (
                      <div key={s.id} draggable onDragStart={e => onDragStart?.(e, s)} style={{ height: ROW_HEIGHT }}>
                        <SampleRowV2
                          sample={s}
                          isActive={activeSample?.id === s.id}
                          isPlaying={audio.currentId === s.id && audio.playing}
                          onPlay={onPlay}
                          isFavorite={favorites.has(s.id)}
                          onToggleFav={onToggleFav}
                          onFindSimilar={onFindSimilar}
                          onFindLayers={onFindLayers}
                          isChecked={checkedSamples.has(s.id)}
                          onCheck={onCheck}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div style={{ padding: 40, textAlign: "center", color: theme.textMuted, fontSize: 12 }}>
                  {samples?.length === 0 ? "No samples." : tab === "favorites" ? "No favorites." : "No matches."}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
});
