/**
 * RESONATE — Left Navigation Sidebar.
 * Context-aware: pre-analyze shows Analyze + Library.
 * Post-analyze shows Analyze + Results + Library.
 */

import { memo } from "react";
import { useRouter } from "../router";
import { useTheme } from "../theme/ThemeProvider";
import { AF, MONO } from "../theme/fonts";

/**
 * MiniOrb — Twisted ribbon ring icon matching the RESONATE orb visual.
 * Multiple parallel curved strands forming a twisted Möbius-like ring,
 * purple→cyan gradient with a bright highlight point. Static, polished.
 */
function MiniOrb({ active }) {
  const a = active ? 1 : 0.35;
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" style={{ overflow: "visible", filter: active ? "drop-shadow(0 0 3px rgba(6,182,212,0.3))" : "none" }}>
      <defs>
        <linearGradient id="orb-g1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#06B6D4" />
          <stop offset="35%" stopColor="#22D3EE" />
          <stop offset="50%" stopColor="#A5F3FC" />
          <stop offset="65%" stopColor="#8B5CF6" />
          <stop offset="100%" stopColor="#7C3AED" />
        </linearGradient>
        <linearGradient id="orb-g2" x1="100%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#8B5CF6" />
          <stop offset="40%" stopColor="#A855F7" />
          <stop offset="60%" stopColor="#06B6D4" />
          <stop offset="100%" stopColor="#22D3EE" />
        </linearGradient>
        <radialGradient id="orb-hl" cx="25%" cy="25%" r="30%">
          <stop offset="0%" stopColor="white" stopOpacity="0.9" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </radialGradient>
      </defs>
      {/* Twisted ribbon strands — outer set (cyan-dominant) */}
      {[0, 0.6, 1.2, 1.8, 2.4].map((offset, i) => (
        <ellipse key={`a${i}`} cx="11" cy="11"
          rx={8 - offset * 0.3} ry={6.5 - offset * 0.25}
          fill="none" stroke="url(#orb-g1)"
          strokeWidth={0.55} opacity={a * (0.9 - i * 0.12)}
          transform={`rotate(${-30 + offset * 4}, 11, 11)`}
        />
      ))}
      {/* Twisted ribbon strands — inner set (purple-dominant, rotated) */}
      {[0, 0.6, 1.2, 1.8, 2.4].map((offset, i) => (
        <ellipse key={`b${i}`} cx="11" cy="11"
          rx={7.5 - offset * 0.3} ry={6 - offset * 0.25}
          fill="none" stroke="url(#orb-g2)"
          strokeWidth={0.5} opacity={a * (0.75 - i * 0.1)}
          transform={`rotate(${60 + offset * 4}, 11, 11)`}
        />
      ))}
      {/* Cross strands to create twist illusion */}
      {[0, 0.7, 1.4].map((offset, i) => (
        <ellipse key={`c${i}`} cx="11" cy="11"
          rx={7 - offset * 0.4} ry={5.5 - offset * 0.3}
          fill="none" stroke="url(#orb-g1)"
          strokeWidth={0.4} opacity={a * (0.5 - i * 0.1)}
          transform={`rotate(${150 + offset * 5}, 11, 11)`}
        />
      ))}
      {/* Bright highlight point — top-left like light catching the surface */}
      {active && <circle cx="5.5" cy="5" r="1.5" fill="url(#orb-hl)" opacity="0.7" />}
    </svg>
  );
}

const PRE_ANALYZE = [
  { id: "home", label: "Analyze", isOrb: true },
  { id: "library", label: "Library", icon: "M5 3v18l7-5 7 5V3H5z" },
];

const POST_ANALYZE = [
  { id: "home", label: "Analyze", isOrb: true },
  { id: "sounds", label: "Results", icon: "M9 18V5l12-2v13M9 18c0 1.66-1.34 3-3 3s-3-1.34-3-3 1.34-3 3-3 3 1.34 3 3zm12-2c0 1.66-1.34 3-3 3s-3-1.34-3-3 1.34-3 3-3 3 1.34 3 3z" },
  { id: "library", label: "Library", icon: "M5 3v18l7-5 7 5V3H5z" },
];

export const NavSidebar = memo(function NavSidebar({ hasAnalysis }) {
  const { page, navigate } = useRouter();
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  const items = hasAnalysis ? POST_ANALYZE : PRE_ANALYZE;

  return (
    <nav style={{
      width: 64, minWidth: 64, height: "100%",
      display: "flex", flexDirection: "column", alignItems: "center",
      paddingTop: 12, gap: 4,
      background: isDark ? "#08080E" : "#EDEAE6",
      borderRight: "1px solid " + theme.border,
      zIndex: 10,
    }}>
      {items.map(item => {
        const active = page === item.id || (item.id === "sounds" && page === "results");
        return (
          <button key={item.id} onClick={() => navigate(item.id)}
            title={item.label}
            style={{
              width: 48, height: 48, borderRadius: 10, border: "none",
              background: active ? (isDark ? "rgba(139,92,246,0.1)" : "rgba(139,92,246,0.06)") : "transparent",
              cursor: "pointer", display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center", gap: 2,
              transition: "all 0.15s",
            }}
          >
            {item.isOrb ? (
              <MiniOrb active={active} />
            ) : (
              <svg width="18" height="18" viewBox={item.viewBox || "0 0 24 24"} fill="none"
                stroke={active ? "#8B5CF6" : theme.textMuted} strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round">
                <path d={item.icon} />
              </svg>
            )}
            <span style={{
              fontSize: 8, fontFamily: AF, fontWeight: active ? 700 : 400,
              color: active ? "#8B5CF6" : theme.textMuted,
              letterSpacing: 0.3,
            }}>{item.label}</span>
          </button>
        );
      })}
      <div style={{ flex: 1 }} />
      <a
        href="https://www.SONIQlabs.org"
        target="_blank"
        rel="noopener noreferrer"
        onClick={(e) => {
          e.preventDefault();
          if (window.electronAPI?.openExternal) {
            window.electronAPI.openExternal("https://www.SONIQlabs.org");
          } else {
            window.open("https://www.SONIQlabs.org", "_blank");
          }
        }}
        style={{
          padding: "6px 0 10px", textDecoration: "none",
          display: "flex", flexDirection: "column", alignItems: "center", gap: 0,
          cursor: "pointer", transition: "opacity 0.2s",
          opacity: 0.5,
        }}
        onMouseEnter={e => e.currentTarget.style.opacity = "0.9"}
        onMouseLeave={e => e.currentTarget.style.opacity = "0.5"}
      >
        <span className="gradient-text" style={{
          fontSize: 7, fontWeight: 700, fontFamily: MONO,
          letterSpacing: 1.5, lineHeight: 1.6,
        }}>SONIQlabs</span>
      </a>
    </nav>
  );
});
