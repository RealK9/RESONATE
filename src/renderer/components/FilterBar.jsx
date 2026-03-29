/**
 * RESONATE — Filter Bar.
 * Splice-style horizontal filter pills: categories, key, source, mood, sort.
 */

import { memo, useState, useRef, useEffect } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { AF, MONO } from "../theme/fonts";

const KEYS = ["Any", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const MOODS = ["all", "dark", "warm", "bright", "aggressive", "chill"];
const MOOD_DOTS = { dark: "#8B5CF6", warm: "#CA8A04", bright: "#3B82F6", aggressive: "#EF4444", chill: "#22C55E" };
const SORTS = [
  { id: "relevant", label: "Relevant" },
  { id: "popular", label: "Popular" },
  { id: "recent", label: "Recent" },
  { id: "random", label: "Random" },
];

export const FilterBar = memo(function FilterBar({
  category, onCategoryChange, categories,
  selectedKey, onKeyChange,
  sourceFilter, onSourceChange, sourceCounts,
  moodFilter, onMoodChange,
  sortBy, onSortChange,
  totalCount,
}) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";
  const [dropdown, setDropdown] = useState(null); // "key" | "mood" | "source" | null
  const dropRef = useRef(null);

  // Close dropdown on outside click
  useEffect(() => {
    if (!dropdown) return;
    const handler = (e) => {
      if (dropRef.current && !dropRef.current.contains(e.target)) setDropdown(null);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [dropdown]);

  const Pill = ({ label, active, count, onClick, color, dot }) => (
    <button onClick={onClick} style={{
      padding: "4px 10px", borderRadius: 14,
      border: "1px solid " + (active ? (color || "rgba(139,92,246,0.35)") : theme.borderLight),
      background: active
        ? (isDark ? (color ? color + "18" : "rgba(139,92,246,0.1)") : (color ? color + "12" : "rgba(139,92,246,0.06)"))
        : "transparent",
      color: active ? (color || "#8B5CF6") : theme.textSec,
      fontSize: 10, fontWeight: active ? 600 : 400,
      cursor: "pointer", fontFamily: AF,
      display: "flex", alignItems: "center", gap: 4,
      transition: "all 0.12s", whiteSpace: "nowrap", flexShrink: 0,
    }}>
      {dot && <span style={{ width: 5, height: 5, borderRadius: "50%", background: dot }} />}
      {label}
      {count != null && <span style={{ fontSize: 8, opacity: 0.5, fontFamily: MONO }}>{count}</span>}
    </button>
  );

  return (
    <div ref={dropRef} style={{
      display: "flex", alignItems: "center", gap: 5,
      padding: "6px 14px", position: "relative",
      borderBottom: "1px solid " + theme.borderLight,
      overflowX: "auto", flexShrink: 0,
    }}>
      {/* Category pills */}
      <Pill label="All" active={category === "all"} count={totalCount} onClick={() => onCategoryChange("all")} />
      {(categories || []).slice(0, 8).map(c => (
        <Pill key={c.name} label={c.name} active={category === c.name} count={c.count} onClick={() => onCategoryChange(c.name)} />
      ))}

      <div style={{ width: 1, height: 18, background: theme.borderLight, flexShrink: 0 }} />

      {/* Key */}
      <Pill label={"Key: " + selectedKey} active={selectedKey !== "Any"} onClick={() => setDropdown(dropdown === "key" ? null : "key")} />

      {/* Mood */}
      <Pill
        label={moodFilter === "all" ? "Mood" : moodFilter.charAt(0).toUpperCase() + moodFilter.slice(1)}
        active={moodFilter !== "all"}
        dot={MOOD_DOTS[moodFilter]}
        onClick={() => setDropdown(dropdown === "mood" ? null : "mood")}
      />

      {/* Source */}
      {sourceCounts && Object.keys(sourceCounts).length > 1 && (
        <Pill
          label={sourceFilter === "all" ? "Source" : sourceFilter.charAt(0).toUpperCase() + sourceFilter.slice(1)}
          active={sourceFilter !== "all"}
          onClick={() => setDropdown(dropdown === "source" ? null : "source")}
        />
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Sort */}
      <div style={{ display: "flex", gap: 1, flexShrink: 0 }}>
        {SORTS.map(s => (
          <button key={s.id} onClick={() => onSortChange?.(s.id)} style={{
            padding: "3px 8px", borderRadius: 4, border: "none", fontSize: 9,
            background: sortBy === s.id ? (isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)") : "transparent",
            color: sortBy === s.id ? theme.text : theme.textMuted,
            cursor: "pointer", fontFamily: AF, fontWeight: sortBy === s.id ? 600 : 400,
            transition: "all 0.12s",
          }}>{s.label}</button>
        ))}
      </div>

      {/* Dropdowns */}
      {dropdown === "key" && (
        <div style={{
          position: "absolute", top: "100%", left: 120, zIndex: 100,
          padding: 8, borderRadius: 8, display: "flex", flexWrap: "wrap", gap: 4, width: 220,
          background: isDark ? "#14141E" : "#fff",
          border: "1px solid " + theme.border,
          boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
        }}>
          {KEYS.map(k => (
            <button key={k} onClick={() => { onKeyChange(k); setDropdown(null); }} style={{
              padding: "5px 10px", borderRadius: 4, border: "none", fontSize: 10, fontFamily: MONO, fontWeight: 600,
              background: selectedKey === k ? "rgba(139,92,246,0.15)" : (isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)"),
              color: selectedKey === k ? "#8B5CF6" : theme.textSec, cursor: "pointer",
              transition: "all 0.1s",
            }}>{k}</button>
          ))}
        </div>
      )}
      {dropdown === "mood" && (
        <div style={{
          position: "absolute", top: "100%", left: 200, zIndex: 100,
          padding: 8, borderRadius: 8, display: "flex", flexDirection: "column", gap: 2, width: 140,
          background: isDark ? "#14141E" : "#fff",
          border: "1px solid " + theme.border,
          boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
        }}>
          {MOODS.map(m => (
            <button key={m} onClick={() => { onMoodChange(m); setDropdown(null); }} style={{
              padding: "5px 10px", borderRadius: 4, border: "none", fontSize: 10, fontFamily: AF,
              background: moodFilter === m ? (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)") : "transparent",
              color: moodFilter === m ? theme.text : theme.textSec, cursor: "pointer", textAlign: "left",
              display: "flex", alignItems: "center", gap: 6,
            }}>
              {MOOD_DOTS[m] && <span style={{ width: 6, height: 6, borderRadius: "50%", background: MOOD_DOTS[m] }} />}
              {m === "all" ? "All Moods" : m.charAt(0).toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>
      )}
      {dropdown === "source" && sourceCounts && (
        <div style={{
          position: "absolute", top: "100%", right: 160, zIndex: 100,
          padding: 8, borderRadius: 8, display: "flex", flexDirection: "column", gap: 2, width: 140,
          background: isDark ? "#14141E" : "#fff",
          border: "1px solid " + theme.border,
          boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
        }}>
          {["all", ...Object.keys(sourceCounts)].map(s => (
            <button key={s} onClick={() => { onSourceChange(s); setDropdown(null); }} style={{
              padding: "5px 10px", borderRadius: 4, border: "none", fontSize: 10, fontFamily: AF,
              background: sourceFilter === s ? (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)") : "transparent",
              color: sourceFilter === s ? theme.text : theme.textSec, cursor: "pointer", textAlign: "left",
              display: "flex", justifyContent: "space-between",
            }}>
              <span>{s === "all" ? "All Sources" : s.charAt(0).toUpperCase() + s.slice(1)}</span>
              {s !== "all" && <span style={{ fontSize: 8, color: theme.textMuted, fontFamily: MONO }}>{sourceCounts[s]}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
});
