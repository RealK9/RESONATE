/**
 * RESONATE — Application Shell.
 * Splice-style 3-zone layout: TopBar + [NavSidebar | Content] + PlayerBar.
 * Wraps all pages. Player and nav persist across navigation.
 */

import { memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { NavSidebar } from "./NavSidebar";
import { TopBar } from "./TopBar";
import { PlayerBar } from "./PlayerBar";

export const AppShell = memo(function AppShell({
  children,
  // TopBar props
  search, onSearchChange, onSettingsClick, bridgeConnected, backendOk, indexProgress,
  // PlayerBar props
  audio, activeSample, pitchTempo, waveformPeaks,
  onToggleFav, isFavorite, onFindSimilar, onToggleMix,
  // Nav state
  hasAnalysis,
}) {
  const { theme } = useTheme();

  return (
    <div style={{
      display: "flex", flexDirection: "column",
      height: "100vh", background: theme.bg,
      overflow: "hidden",
    }}>
      {/* Top bar with central search */}
      <TopBar
        search={search} onSearchChange={onSearchChange}
        onSettingsClick={onSettingsClick} bridgeConnected={bridgeConnected}
        backendOk={backendOk} indexProgress={indexProgress}
      />

      {/* Middle: nav sidebar + page content */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <NavSidebar hasAnalysis={hasAnalysis} />
        <div style={{
          flex: 1, display: "flex", flexDirection: "column",
          overflow: "hidden", position: "relative",
        }}>
          {children}
        </div>
      </div>

      {/* Persistent bottom player */}
      <PlayerBar
        audio={audio} activeSample={activeSample}
        pitchTempo={pitchTempo} waveformPeaks={waveformPeaks}
        onToggleFav={onToggleFav} isFavorite={isFavorite}
        onFindSimilar={onFindSimilar} onToggleMix={onToggleMix}
        bridgeConnected={bridgeConnected}
      />
    </div>
  );
});
