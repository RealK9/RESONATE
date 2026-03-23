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

  return v2Recs.map((rec, idx) => {
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

    return merged;
  });
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
