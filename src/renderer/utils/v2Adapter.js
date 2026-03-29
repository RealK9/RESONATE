/**
 * RESONATE — v2 Adapter
 *
 * Translates v2 ML recommendation objects into the v1 sample shape
 * that SampleRow and the rest of the UI expect.
 */

/**
 * Derive a clean display name from a filepath or filename.
 */
function deriveName(filepath, filename) {
  const raw = filename || filepath.split("/").pop() || "Unknown";
  const stem = raw.replace(/\.[^.]+$/, "");
  return stem.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
}

// ── Intelligent Sample Naming ──

const MOOD_DESCRIPTORS = {
  dark: "Dark", warm: "Warm", bright: "Bright", aggressive: "Hard",
  chill: "Smooth", neutral: "Clean", melancholic: "Mellow",
  ethereal: "Ethereal", uplifting: "Bright", gritty: "Gritty",
  dreamy: "Dreamy", tense: "Tense", playful: "Playful",
};

const ENERGY_DESCRIPTORS = {
  high: ["Punchy", "Driving", "Hard", "Crisp"],
  medium: ["Smooth", "Warm", "Steady", "Solid"],
  low: ["Soft", "Gentle", "Subtle", "Muted"],
};

const INSTRUMENT_NAMES = {
  kick: "Kick", snare_clap: "Snare", hats_tops: "Hi-Hat",
  bass: "Bass", lead: "Synth", chord_support: "Keys",
  pad: "Pad", vocal_texture: "Vocal", fx_transitions: "FX",
  ambience: "Texture", percussion: "Perc",
};

const PURPOSE_MAP = {
  kick: "Drums", snare_clap: "Drums", hats_tops: "Drums",
  bass: "Low End", lead: "Lead Melody", chord_support: "Harmony",
  pad: "Atmosphere", vocal_texture: "Vocal Layer", fx_transitions: "Transition",
  ambience: "Ambience", percussion: "Rhythm",
};

const TYPE_MAP = {
  kick: "One-shot", snare_clap: "One-shot", hats_tops: "One-shot",
  percussion: "One-shot", fx_transitions: "One-shot",
  bass: "Loop", lead: "Loop", chord_support: "Loop",
  pad: "Loop", vocal_texture: "Loop", ambience: "Loop",
};

/** Map v1 category/sub_category names to internal role keys */
const CATEGORY_TO_ROLE = {
  kick: "kick", kicks: "kick", "kick-sub": "kick",
  snare: "snare_clap", snares: "snare_clap", clap: "snare_clap", claps: "snare_clap", "snare-clap": "snare_clap",
  "hi-hat": "hats_tops", hihat: "hats_tops", hat: "hats_tops", hats: "hats_tops", cymbal: "hats_tops", cymbals: "hats_tops", tops: "hats_tops", "open-hat": "hats_tops", "closed-hat": "hats_tops",
  bass: "bass", "808": "bass", sub: "bass", "sub-bass": "bass",
  lead: "lead", synth: "lead", melody: "lead", pluck: "lead", arp: "lead",
  chord: "chord_support", chords: "chord_support", keys: "chord_support", piano: "chord_support", organ: "chord_support", strings: "chord_support", guitar: "chord_support",
  pad: "pad", pads: "pad", atmosphere: "pad",
  vocal: "vocal_texture", vocals: "vocal_texture", vox: "vocal_texture", voice: "vocal_texture",
  fx: "fx_transitions", sfx: "fx_transitions", transition: "fx_transitions", riser: "fx_transitions", impact: "fx_transitions", downlifter: "fx_transitions",
  ambience: "ambience", ambient: "ambience", texture: "ambience", noise: "ambience", foley: "ambience",
  percussion: "percussion", perc: "percussion", shaker: "percussion", tambourine: "percussion", conga: "percussion", bongo: "percussion", rim: "percussion",
  drum: "kick", drums: "kick", "drum-loop": "kick",
  brass: "lead", "brass-wind": "lead", horn: "lead", trumpet: "lead", sax: "lead", flute: "lead", woodwind: "lead",
};

/** Resolve role from v2_role, sample_type, or category fallback */
function resolveRole(sample) {
  if (sample.v2_role && INSTRUMENT_NAMES[sample.v2_role]) return sample.v2_role;
  if (sample.sample_type && INSTRUMENT_NAMES[sample.sample_type]) return sample.sample_type;
  const cat = (sample.category || sample.sub_category || "").toLowerCase().replace(/[\s/]+/g, "-");
  if (CATEGORY_TO_ROLE[cat]) return CATEGORY_TO_ROLE[cat];
  // Try splitting hyphenated categories
  for (const part of cat.split("-")) {
    if (CATEGORY_TO_ROLE[part]) return CATEGORY_TO_ROLE[part];
  }
  return "";
}

/** Deterministic hash for consistent descriptor selection */
function simpleHash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = ((h << 5) - h + str.charCodeAt(i)) | 0;
  return Math.abs(h);
}

/** Pick a descriptor from mood/energy with deterministic variety */
function pickDescriptor(mood, energy, name) {
  if (mood && MOOD_DESCRIPTORS[mood]) return MOOD_DESCRIPTORS[mood];
  const level = energy || "medium";
  const pool = ENERGY_DESCRIPTORS[level] || ENERGY_DESCRIPTORS.medium;
  return pool[simpleHash(name) % pool.length];
}

/**
 * Generate a smart display name: "Descriptor Instrument - Purpose - Type"
 */
function generateSmartName(sample) {
  const role = resolveRole(sample);
  const mood = sample.mood || "";
  const energy = sample.energy || "medium";
  const origName = sample.name || sample.filename || "Sample";

  const descriptor = pickDescriptor(mood, energy, origName);
  const instrument = INSTRUMENT_NAMES[role] || "Sample";
  const purpose = PURPOSE_MAP[role] || "General";
  const type = TYPE_MAP[role] || "Loop";

  return `${descriptor} ${instrument} - ${purpose} - ${type}`;
}

/**
 * Post-process array to add #1, #2 numbering for duplicate names.
 */
function deduplicateNames(samples) {
  const counts = {};
  const indices = {};
  // First pass: count
  for (const s of samples) {
    counts[s.name] = (counts[s.name] || 0) + 1;
  }
  // Second pass: number duplicates
  for (const s of samples) {
    if (counts[s.name] > 1) {
      indices[s.name] = (indices[s.name] || 0) + 1;
      s.name = `${s.name.replace(/ - ([^-]+)$/, ` #${indices[s.name]} - $1`)}`;
    }
  }
  return samples;
}

/**
 * Build v1-compatible analysis object from v2 MixProfile.
 * Derives mood, energy, duration, and detected instruments from v2 data
 * so the sidebar analyzer box displays real values instead of dashes.
 */
export function buildV1Compat(mp, summary) {
  const roles = mp?.source_roles?.roles || {};
  const style = mp?.style || {};

  // Derive mood from style cluster name and spectral characteristics
  const cluster = (style.primary_cluster || "").toLowerCase();
  let mood = "Neutral";
  if (/dark|drill|phonk|trap/.test(cluster)) mood = "Dark";
  else if (/chill|lo.?fi|ambient/.test(cluster)) mood = "Chill";
  else if (/pop|uplifting|future.?bass/.test(cluster)) mood = "Bright";
  else if (/boom.?bap|rock|punk/.test(cluster)) mood = "Aggressive";
  else if (/r.?n.?b|soul|jazz/.test(cluster)) mood = "Warm";
  else if (/cinematic|orchestral/.test(cluster)) mood = "Cinematic";
  else if (/house|techno|edm/.test(cluster)) mood = "Energetic";
  else if (/latin|reggaeton|afro/.test(cluster)) mood = "Groovy";

  // Derive energy from percussive role presence + onset density
  const kickConf = roles.kick || 0;
  const snareConf = roles.snare_clap || 0;
  const hatsConf = roles.hats_tops || 0;
  const percTotal = kickConf + snareConf + hatsConf;
  let energy = "Medium";
  if (percTotal > 1.2) energy = "High";
  else if (percTotal > 0.6) energy = "Medium-High";
  else if (percTotal < 0.2) energy = "Low";
  else if (percTotal < 0.4) energy = "Medium-Low";

  // Detected instruments: roles with confidence > 0.15
  const ROLE_DISPLAY = {
    kick: "Kick", snare_clap: "Snare", hats_tops: "Hi-Hats",
    bass: "Bass", lead: "Lead", chord_support: "Chords",
    pad: "Pad", vocal_texture: "Vocal", fx_transitions: "FX",
    ambience: "Ambience",
  };
  const detected = Object.entries(roles)
    .filter(([, v]) => v > 0.15)
    .sort((a, b) => b[1] - a[1])
    .map(([k]) => ROLE_DISPLAY[k] || k);

  // Spectral occupancy → frequency bands (map 10-band analyzer names to 7-band SpectrumViz keys)
  const BAND_MAP = {
    sub: "sub_bass_20_80",
    bass: "bass_80_250",
    low_mid: "low_mid_250_500",
    mid: "mid_500_2k",
    upper_mid: "upper_mid_2k_6k",
    presence: "presence_6k_12k",
    brilliance: "air_12k_20k",   // merge brilliance into air
    air: "air_12k_20k",          // both map to air
    ultra_high: "air_12k_20k",   // merge ultra_high into air
    ceiling: "air_12k_20k",      // merge ceiling into air
  };
  const bands = {};
  const bandNames = mp?.spectral_occupancy?.bands || [];
  const bandMeans = mp?.spectral_occupancy?.mean_by_band || [];
  bandNames.forEach((name, i) => {
    if (i < bandMeans.length) {
      const key = BAND_MAP[name] || name;
      const val = Math.round(bandMeans[i] * 100);
      // For merged bands (air), take the max
      bands[key] = Math.max(bands[key] || 0, val);
    }
  });

  return {
    analysis: {
      key: mp?.analysis?.key || "",
      bpm: mp?.analysis?.bpm || 0,
      genre: summary?.blueprint_used || style.primary_cluster || "",
      mood,
      energy_label: energy,
      duration: mp?.analysis?.duration || 0,
      summary: summary?.gap_summary || "",
      what_track_needs: (mp?.needs || []).map(n => n.description).slice(0, 6),
      frequency_bands: bands,
      frequency_gaps: [],
      detected_instruments: detected,
    },
  };
}

/**
 * Category label map for roles.
 */
const ROLE_LABELS = {
  kick: "Kick", snare_clap: "Snare", hats_tops: "Hi-Hat",
  bass: "Bass", lead: "Lead", chord_support: "Chords",
  pad: "Pad", vocal_texture: "Vocal", fx_transitions: "FX",
  ambience: "Ambience", percussion: "Percussion",
};

/**
 * Merge v2 Recommendation objects with v1 sample metadata.
 *
 * @param {Array} v2Recs - Array of recommendation objects from POST /recommend/v2
 * @param {Array} v1Samples - Array of sample objects from GET /samples
 * @returns {Array} Merged sample objects compatible with SampleRow
 */
export function mergeV2WithV1(v2Recs, v1Samples) {
  // Build lookup from v1 samples keyed by id (absolute filepath)
  const v1Map = new Map();
  for (const s of v1Samples) {
    v1Map.set(s.id, s);
    // Also index by filename for fallback matching
    if (s.filename) v1Map.set(s.filename, s);
  }

  const results = v2Recs.map((rec, idx) => {
    const v1 = v1Map.get(rec.filepath) || v1Map.get(rec.filename) || null;

    const merged = {
      // v1 fields (from library metadata)
      id: rec.filepath,
      name: v1?.name || deriveName(rec.filepath, rec.filename),
      clean_name: v1?.clean_name || deriveName(rec.filepath, rec.filename),
      filename: v1?.filename || rec.filename,
      path: v1?.path || encodeURIComponent(rec.filepath),
      category: v1?.category || ROLE_LABELS[rec.role] || rec.role || "",
      sub_category: v1?.sub_category || "",
      duration: v1?.duration || 0,
      bpm: v1?.bpm || 0,
      key: v1?.key || "",
      original_bpm: v1?.original_bpm || v1?.bpm || 0,
      original_key: v1?.original_key || v1?.key || "",
      synced_key: v1?.synced_key || "",
      synced_bpm: v1?.synced_bpm || 0,
      sample_type: v1?.sample_type || rec.role || "",
      type_label: v1?.type_label || ROLE_LABELS[rec.role] || "",
      frequency_bands: v1?.frequency_bands || {},
      source: v1?.source || "local",
      mood: v1?.mood || "",
      energy: v1?.energy || "",

      // v2 mapped fields
      match: Math.round(rec.score * 100),
      match_reason: rec.explanation || "",

      // v2-specific fields for enhanced display
      v2_explanation: rec.explanation || "",
      v2_policy: rec.policy || "",
      v2_need_addressed: rec.need_addressed || "",
      v2_role: rec.role || "",
      v2_breakdown: rec.breakdown || {},
      v2_rank: idx,
      _isV2: true,
    };

    // Generate smart name from v2 ML data
    merged.name = generateSmartName(merged);
    merged.clean_name = merged.name;

    return merged;
  });

  return deduplicateNames(results);
}

/**
 * Format the needs array from a MixProfile for display.
 *
 * @param {Object} mixProfileDict - The MixProfile dict from /analyze/v2
 * @returns {Array} Sorted needs with {category, description, severity, policy}
 */
export function formatNeeds(mixProfileDict) {
  const needs = mixProfileDict?.needs;
  if (!Array.isArray(needs) || needs.length === 0) return [];

  return needs
    .map((n) => ({
      category: n.category || "general",
      description: n.description || "",
      severity: n.severity || 0,
      policy: n.recommendation_policy || "",
    }))
    .sort((a, b) => b.severity - a.severity);
}

/**
 * Category color map for needs diagnosis pills.
 */
export const NEED_CATEGORY_COLORS = {
  spectral: "#818CF8",     // indigo
  role: "#F472B6",         // pink
  dynamic: "#FB923C",      // orange
  spatial: "#34D399",      // emerald
  arrangement: "#A78BFA",  // violet
};

/**
 * Human-readable labels for recommendation policies.
 */
export const POLICY_LABELS = {
  fill_missing_role: "Fill Gap",
  reinforce_existing: "Reinforce",
  improve_polish: "Polish",
  increase_contrast: "Contrast",
  add_movement: "Movement",
  reduce_emptiness: "Fill Space",
  support_transition: "Transition",
  enhance_groove: "Groove",
  enhance_lift: "Lift",
};

/**
 * Genre display names for the gap analysis panel.
 */
const GENRE_DISPLAY = {
  modern_trap: "Trap",
  modern_drill: "Drill",
  "2010s_edm_drop": "EDM",
  "2020s_melodic_house": "Melodic House",
  melodic_techno: "Melodic Techno",
  dnb: "Drum & Bass",
  afro_house: "Afro House",
  pop_production: "Pop",
  "2000s_pop_chorus": "2000s Pop",
  r_and_b: "R&B",
  "1990s_boom_bap": "Boom Bap",
  lo_fi_chill: "Lo-fi",
  cinematic: "Cinematic",
  ambient: "Ambient",
  lo_fi_hip_hop: "Lo-Fi Hip Hop",
  boom_bap: "Boom Bap",
  rnb_soul: "R&B/Soul",
  pop_electronic: "Pop",
  indie_rock: "Indie Rock",
  ambient_textural: "Ambient",
  latin_reggaeton: "Reggaeton",
  house_tech_house: "House/Tech House",
  future_bass: "Future Bass",
  phonk: "Phonk",
  jazz_fusion: "Jazz Fusion",
};

/**
 * Gap severity label + color.
 */
const GAP_SEVERITY = [
  { threshold: 0.7, label: "Critical", color: "#EF4444" },
  { threshold: 0.4, label: "Moderate", color: "#F59E0B" },
  { threshold: 0, label: "Minor", color: "#6B7280" },
];

/**
 * Format gap analysis results for display.
 *
 * @param {Object} gapAnalysis - Raw gap analysis dict from API
 * @returns {Object} Formatted gap data for UI
 */
export function formatGapAnalysis(gapAnalysis) {
  if (!gapAnalysis) return null;

  const readiness = gapAnalysis.production_readiness_score ?? 0;
  const genre = GENRE_DISPLAY[gapAnalysis.blueprint_name] || gapAnalysis.genre_detected || "Unknown";

  // Determine readiness tier
  let readinessTier, readinessColor;
  if (readiness >= 85) {
    readinessTier = "Chart-Ready";
    readinessColor = "#22C55E";
  } else if (readiness >= 65) {
    readinessTier = "Getting Close";
    readinessColor = "#3B82F6";
  } else if (readiness >= 40) {
    readinessTier = "In Progress";
    readinessColor = "#F59E0B";
  } else {
    readinessTier = "Early Stage";
    readinessColor = "#EF4444";
  }

  // Format gaps
  const gaps = (gapAnalysis.gaps || []).map((g) => {
    const sev = GAP_SEVERITY.find((s) => g.severity >= s.threshold) || GAP_SEVERITY[2];
    return {
      category: g.category,
      dimension: g.dimension,
      severity: g.severity,
      severityLabel: sev.label,
      severityColor: sev.color,
      message: g.message,
      direction: g.direction,
    };
  });

  return {
    readiness: Math.round(readiness),
    readinessTier,
    readinessColor,
    genre,
    chartPotentialCurrent: Math.round(gapAnalysis.chart_potential_current ?? 0),
    chartPotentialCeiling: Math.round(gapAnalysis.chart_potential_ceiling ?? 0),
    genreCoherence: Math.round((gapAnalysis.genre_coherence_score ?? 0) * 100),
    missingRoles: (gapAnalysis.missing_roles || []).map((r) => ROLE_LABELS[r] || r),
    presentRoles: (gapAnalysis.present_roles || []).map((r) => ROLE_LABELS[r] || r),
    totalGaps: gapAnalysis.total_gaps ?? 0,
    criticalGaps: gapAnalysis.critical_gaps ?? 0,
    moderateGaps: gapAnalysis.moderate_gaps ?? 0,
    gaps: gaps.slice(0, 8), // top 8 for display
    summary: gapAnalysis.summary || "",
  };
}
