/**
 * RESONATE — Analyzing Overlay.
 * Full-screen overlay with progress orb + centered R logo.
 * Gradient percentage display with elegant serif typography.
 */

import { memo } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { ResonateOrb } from "./ResonateOrb";
import { SERIF, MONO, AF } from "../theme/fonts";

export const AnalyzingOverlay = memo(function AnalyzingOverlay({ progress, stage, fileName, LogoBlend }) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  return (
    <div style={{
      position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      background: isDark ? "rgba(10,10,16,0.92)" : "rgba(245,243,239,0.92)",
      backdropFilter: "blur(24px)",
      zIndex: 500,
    }}>
      {/* Orb with logo centered inside */}
      <div style={{ position: "relative", width: 320, height: 320 }}>
        <ResonateOrb progress={progress} size={320} />
        {/* Logo absolutely positioned in center of orb */}
        {LogoBlend && (
          <div style={{
            position: "absolute", top: "50%", left: "50%",
            transform: "translate(-50%, -50%)",
            width: 110, height: 110, zIndex: 2,
          }}>
            <LogoBlend size={110} isDark={isDark} />
          </div>
        )}
      </div>

      {/* Gradient percentage — elegant serif with polished glow */}
      <div className="gradient-text" style={{
        fontSize: 56, fontWeight: 200, letterSpacing: -3,
        marginTop: 28, fontFamily: SERIF, lineHeight: 1,
      }}>
        {progress}<span style={{ fontSize: 24, fontWeight: 300, opacity: 0.6 }}>%</span>
      </div>

      {/* Stage label */}
      <div className="gradient-text" style={{
        fontSize: 10, fontWeight: 600, marginTop: 12,
        fontFamily: AF, letterSpacing: 2, textTransform: "uppercase",
        animation: "fadeIn 0.3s ease",
      }}>
        {stage}
      </div>

      {/* Filename */}
      <div style={{
        fontSize: 9, color: theme.textFaint, marginTop: 10,
        fontFamily: MONO, opacity: 0.5,
      }}>
        {fileName}
      </div>
    </div>
  );
});
