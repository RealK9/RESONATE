/**
 * ProducerDNA — Visual taste profile showing the user's sonic identity.
 * Radar chart of role affinities, genre preference bars, stats row, retrain button.
 */
import { SERIF, AF, MONO } from "../theme/fonts";

// ── Radar chart roles (canonical order) ──
const ROLES = [
  "kick", "snare_clap", "bass", "lead", "pad",
  "hats_tops", "vocal_texture", "fx_transitions", "percussion",
];

const ROLE_LABELS = {
  kick: "Kick",
  snare_clap: "Snare/Clap",
  bass: "Bass",
  lead: "Lead",
  pad: "Pad",
  hats_tops: "Hats/Tops",
  vocal_texture: "Vocal",
  fx_transitions: "FX",
  percussion: "Perc",
};

// ── Normalize affinity from [-1, 1] to [0, 1] ──
function norm(v) {
  return Math.max(0, Math.min(1, (v + 1) / 2));
}

// ── Radar Chart (SVG) ──
function RadarChart({ affinities, isDark }) {
  const cx = 120, cy = 120, maxR = 95;
  const n = ROLES.length;
  const affinityMap = {};
  for (const a of (affinities || [])) affinityMap[a.role] = a.affinity;

  // Compute points for each role
  const points = ROLES.map((role, i) => {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    const val = norm(affinityMap[role] || 0);
    const r = val * maxR;
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
      ax: cx + maxR * Math.cos(angle),
      ay: cy + maxR * Math.sin(angle),
      lx: cx + (maxR + 14) * Math.cos(angle),
      ly: cy + (maxR + 14) * Math.sin(angle),
      role,
      val,
    };
  });

  const polygon = points.map(p => `${p.x},${p.y}`).join(" ");

  // Grid rings at 25%, 50%, 75%, 100%
  const rings = [0.25, 0.5, 0.75, 1.0];

  return (
    <svg width="240" height="240" viewBox="0 0 240 240" style={{ display: "block", margin: "0 auto" }}>
      {/* Grid rings */}
      {rings.map(r => (
        <polygon
          key={r}
          points={ROLES.map((_, i) => {
            const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
            return `${cx + maxR * r * Math.cos(angle)},${cy + maxR * r * Math.sin(angle)}`;
          }).join(" ")}
          fill="none"
          stroke={isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}
          strokeWidth="0.5"
        />
      ))}

      {/* Axis lines */}
      {points.map((p, i) => (
        <line key={i} x1={cx} y1={cy} x2={p.ax} y2={p.ay} stroke={isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"} strokeWidth="0.5" />
      ))}

      {/* Data polygon */}
      <polygon
        points={polygon}
        fill="rgba(217,70,239,0.15)"
        stroke="#D946EF"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />

      {/* Data dots */}
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={p.val > 0.05 ? 3 : 1.5} fill="#D946EF" opacity={p.val > 0.05 ? 1 : 0.3} />
      ))}

      {/* Axis labels */}
      {points.map((p, i) => (
        <text
          key={i}
          x={p.lx}
          y={p.ly}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.45)"}
          fontSize="8"
          fontFamily={AF}
        >
          {ROLE_LABELS[p.role] || p.role}
        </text>
      ))}
    </svg>
  );
}

// ── Genre Preference Bars ──
function GenreBars({ preferences, isDark, theme }) {
  if (!preferences || preferences.length === 0) return null;

  // Sort by absolute preference, take top 10
  const sorted = [...preferences]
    .sort((a, b) => Math.abs(b.preference) - Math.abs(a.preference))
    .slice(0, 10);

  const maxAbs = Math.max(...sorted.map(s => Math.abs(s.preference)), 0.01);

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 9, color: isDark ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.35)", fontFamily: AF, textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 8 }}>
        Genre Preferences
      </div>
      {sorted.map(s => {
        const pct = Math.abs(s.preference) / maxAbs;
        const isPositive = s.preference >= 0;
        const color = isPositive ? "#D946EF" : "#6366F1";
        return (
          <div key={s.style} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span style={{ width: 70, fontSize: 10, color: theme.textSec, fontFamily: AF, textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flexShrink: 0 }}>
              {s.style}
            </span>
            <div style={{ flex: 1, height: 6, borderRadius: 3, background: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)", overflow: "hidden" }}>
              <div style={{ width: `${Math.round(pct * 100)}%`, height: "100%", borderRadius: 3, background: color, transition: "width 0.4s ease" }} />
            </div>
            <span style={{ width: 32, fontSize: 9, color: theme.textMuted, fontFamily: MONO, textAlign: "right", flexShrink: 0 }}>
              {isPositive ? "+" : ""}{Math.round(s.preference * 100)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Stats Row ──
function StatsRow({ data, theme }) {
  const stats = [
    { label: "Interactions", value: data.total_interactions || 0 },
    { label: "Training Pairs", value: data.training_pairs || 0 },
    { label: "Model v", value: data.model_version || 0 },
  ];
  return (
    <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: 20 }}>
      {stats.map(s => (
        <div key={s.label} style={{ textAlign: "center" }}>
          <div style={{ fontSize: 18, fontWeight: 200, color: theme.text, fontFamily: SERIF }}>{s.value}</div>
          <div style={{ fontSize: 8, color: theme.textMuted, fontFamily: AF, textTransform: "uppercase", letterSpacing: 1 }}>{s.label}</div>
        </div>
      ))}
    </div>
  );
}

// ── Empty State ──
function EmptyState({ isDark, theme }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "60px 30px", textAlign: "center" }}>
      <svg width="48" height="48" viewBox="0 0 48 48" style={{ opacity: 0.25, marginBottom: 16 }}>
        <circle cx="24" cy="24" r="20" fill="none" stroke={isDark ? "#fff" : "#000"} strokeWidth="1" strokeDasharray="3 3" />
        <circle cx="24" cy="24" r="3" fill={isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.15)"} />
      </svg>
      <div style={{ fontSize: 14, fontWeight: 500, color: theme.text, fontFamily: SERIF, marginBottom: 6 }}>
        Your Producer DNA
      </div>
      <div style={{ fontSize: 11, color: theme.textMuted, fontFamily: AF, lineHeight: 1.6, maxWidth: 280 }}>
        Keep using RESONATE to build your sonic identity. Audition, keep, and rate samples — your taste profile will emerge.
      </div>
    </div>
  );
}

// ── Main Component ──
export function ProducerDNA({ data, onTrain, training, theme, isDark }) {
  if (!data || data.status === "no_data") {
    return <EmptyState isDark={isDark} theme={theme} />;
  }

  return (
    <div style={{ padding: "16px 20px", overflowY: "auto", flex: 1 }}>
      {/* Section title */}
      <div style={{ textAlign: "center", marginBottom: 4 }}>
        <div style={{ fontSize: 14, fontWeight: 500, color: theme.text, fontFamily: SERIF }}>Producer DNA</div>
        <div style={{ fontSize: 9, color: theme.textMuted, fontFamily: AF }}>
          Your sonic identity across {data.total_interactions || 0} interactions
        </div>
      </div>

      {/* Radar chart */}
      <RadarChart affinities={data.role_affinities} isDark={isDark} />

      {/* Genre bars */}
      <GenreBars preferences={data.style_preferences} isDark={isDark} theme={theme} />

      {/* Stats */}
      <StatsRow data={data} theme={theme} />

      {/* Retrain button */}
      <div style={{ display: "flex", justifyContent: "center", marginTop: 20 }}>
        <button
          onClick={onTrain}
          disabled={training}
          style={{
            padding: "6px 20px",
            borderRadius: 6,
            border: "1px solid " + (isDark ? "rgba(217,70,239,0.3)" : "rgba(217,70,239,0.25)"),
            background: isDark ? "rgba(217,70,239,0.08)" : "rgba(217,70,239,0.05)",
            color: "#D946EF",
            fontSize: 10,
            fontWeight: 600,
            fontFamily: AF,
            cursor: training ? "wait" : "pointer",
            opacity: training ? 0.6 : 1,
            transition: "all 0.2s ease",
          }}
        >
          {training ? "Training..." : "Retrain Model"}
        </button>
      </div>

      {/* Quality threshold & weight profile */}
      {(data.quality_threshold != null || (data.weight_profile && Object.keys(data.weight_profile).length > 0)) && (
        <div style={{ marginTop: 20, padding: "10px 12px", borderRadius: 6, background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)" }}>
          <div style={{ fontSize: 8, color: theme.textMuted, fontFamily: AF, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>Model Details</div>
          {data.quality_threshold != null && (
            <div style={{ fontSize: 10, color: theme.textSec, fontFamily: MONO, marginBottom: 3 }}>
              Quality threshold: {Math.round(data.quality_threshold * 100)}%
            </div>
          )}
          {data.weight_profile && Object.entries(data.weight_profile).map(([k, v]) => (
            <div key={k} style={{ fontSize: 10, color: theme.textSec, fontFamily: MONO, marginBottom: 2 }}>
              {k}: {v > 0 ? "+" : ""}{v.toFixed(4)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
