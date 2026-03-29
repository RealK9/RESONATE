/**
 * RESONATE — Pitch & Tempo Engine.
 *
 * Manages real-time pitch shifting and timestretching for sample preview
 * using @soundtouchjs/audio-worklet. Splice-style controls:
 *   - Transpose: chromatic key selector + manual semitone offset
 *   - Timestretch: ½x / 1x / 2x of upload BPM
 *   - Both toggleable independently
 *
 * The engine sits between AudioBufferSourceNode and GainNode in the
 * Web Audio graph, processing audio in real-time via AudioWorklet.
 */

import { useState, useRef, useCallback, useMemo } from "react";

// ── Music theory constants ──────────────────────────────────────────────────

const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const NOTE_NAMES_FLAT  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

/** Map any key string to a semitone index 0-11. */
function keyToSemitone(key) {
  if (!key || key === "N/A" || key === "--" || key === "—") return null;
  // Strip mode suffix (m, min, maj, major, minor)
  const root = key.replace(/\s*(m|min|maj|major|minor)$/i, "").trim();
  const sharp = NOTE_NAMES_SHARP.indexOf(root);
  if (sharp >= 0) return sharp;
  const flat = NOTE_NAMES_FLAT.indexOf(root);
  if (flat >= 0) return flat;
  // Try lowercase
  const upper = root.charAt(0).toUpperCase() + root.slice(1);
  const s2 = NOTE_NAMES_SHARP.indexOf(upper);
  if (s2 >= 0) return s2;
  const f2 = NOTE_NAMES_FLAT.indexOf(upper);
  if (f2 >= 0) return f2;
  return null;
}

/** Extract mode from key string: "minor" or "major". */
function keyMode(key) {
  if (!key) return "major";
  if (/m$|min|minor/i.test(key)) return "minor";
  return "major";
}

/** Calculate shortest-path semitone distance from source key to target key. */
export function semitoneDist(sourceKey, targetKey) {
  const s = keyToSemitone(sourceKey);
  const t = keyToSemitone(targetKey);
  if (s === null || t === null) return 0;
  let diff = (t - s) % 12;
  if (diff < 0) diff += 12;
  if (diff > 6) diff -= 12;
  return diff;
}

/** Convert semitones to pitch ratio: 2^(n/12). */
export function semitonesToRatio(semitones) {
  return Math.pow(2, semitones / 12);
}

/** Calculate tempo ratio, normalizing with half/double time. */
export function tempoRatio(sampleBpm, targetBpm, multiplier = 1) {
  if (!sampleBpm || !targetBpm || sampleBpm <= 0) return 1;
  const effectiveTarget = targetBpm * multiplier;
  return effectiveTarget / sampleBpm;
}

/**
 * Determine the best default tempo multiplier (½x, 1x, 2x) so the
 * ratio stays in a reasonable range [0.75, 1.5].
 */
export function bestTempoMultiplier(sampleBpm, targetBpm) {
  if (!sampleBpm || !targetBpm) return 1;
  const raw = targetBpm / sampleBpm;
  if (raw > 1.5) return 0.5;   // sample is much slower — use 2x target
  if (raw < 0.75) return 2;    // sample is much faster — use ½x target
  return 1;
}

// ── Hook ────────────────────────────────────────────────────────────────────

export function usePitchTempo() {
  // ── State ─────────────────────────────────────────────────────────────
  const [transposeEnabled, setTransposeEnabled] = useState(false);
  const [timestrechEnabled, setTimestrechEnabled] = useState(true);
  const [targetKey, setTargetKey] = useState(null);       // null = use upload key
  const [pitchOffset, setPitchOffset] = useState(0);      // manual +/- semitones
  const [tempoMultiplier, setTempoMultiplier] = useState(1); // 0.5, 1, 2
  const [uploadKey, setUploadKey] = useState(null);        // detected from upload
  const [uploadBpm, setUploadBpm] = useState(0);           // detected from upload
  const [useFlats, setUseFlats] = useState(true);          // flat vs sharp display

  // ── Refs for real-time access in audio callbacks ──────────────────────
  const stNodeRef = useRef(null);       // SoundTouchNode instance
  const registeredRef = useRef(false);  // processor registered?

  // ── Derived values ────────────────────────────────────────────────────

  /** The effective target key (user-selected or upload key). */
  const effectiveKey = targetKey ?? uploadKey;

  /** Note names for the key selector grid. */
  const noteNames = useMemo(() => useFlats ? NOTE_NAMES_FLAT : NOTE_NAMES_SHARP, [useFlats]);

  /** Display BPMs for ½x / 1x / 2x buttons. */
  const bpmOptions = useMemo(() => {
    if (!uploadBpm) return [{ label: "½x", bpm: 0, mult: 0.5 }, { label: "1x", bpm: 0, mult: 1 }, { label: "2x", bpm: 0, mult: 2 }];
    return [
      { label: "½x", bpm: Math.round(uploadBpm * 0.5), mult: 0.5 },
      { label: "1x", bpm: Math.round(uploadBpm), mult: 1 },
      { label: "2x", bpm: Math.round(uploadBpm * 2), mult: 2 },
    ];
  }, [uploadBpm]);

  // ── SoundTouch AudioWorklet lifecycle ─────────────────────────────────

  /**
   * Register the SoundTouch AudioWorklet processor.
   * Must be called once per AudioContext before creating nodes.
   */
  const registerProcessor = useCallback(async (audioCtx) => {
    if (registeredRef.current) return;
    try {
      // In Electron, use blob URL to avoid file:// CORS issues
      const resp = await fetch("/soundtouch-processor.js");
      const src = await resp.text();
      const blob = new Blob([src], { type: "text/javascript" });
      const url = URL.createObjectURL(blob);
      await audioCtx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      registeredRef.current = true;
    } catch (e) {
      console.error("[PitchTempo] Failed to register SoundTouch processor:", e);
      // Fallback: try direct URL (works in Vite dev mode)
      try {
        await audioCtx.audioWorklet.addModule("/soundtouch-processor.js");
        registeredRef.current = true;
      } catch (e2) {
        console.error("[PitchTempo] Fallback registration also failed:", e2);
      }
    }
  }, []);

  /**
   * Create a SoundTouchNode and wire it into the audio graph.
   * Returns the node, or null if creation fails (fallback to direct connection).
   *
   * @param {AudioContext} audioCtx
   * @returns {AudioWorkletNode|null}
   */
  const createNode = useCallback(async (audioCtx) => {
    await registerProcessor(audioCtx);
    if (!registeredRef.current) return null;

    try {
      const { SoundTouchNode } = await import("@soundtouchjs/audio-worklet");
      const node = new SoundTouchNode(audioCtx);
      stNodeRef.current = node;
      return node;
    } catch (e) {
      console.error("[PitchTempo] Failed to create SoundTouchNode:", e);
      return null;
    }
  }, [registerProcessor]);

  /**
   * Calculate and apply pitch/tempo parameters to the SoundTouch node.
   *
   * @param {string|null} sampleKey - The sample's original key
   * @param {number} sampleBpm - The sample's original BPM
   */
  const applyParams = useCallback((sampleKey, sampleBpm) => {
    const node = stNodeRef.current;
    if (!node) return;

    // Pitch: transpose semitones + manual offset
    let semitones = 0;
    if (transposeEnabled && effectiveKey && sampleKey) {
      semitones = semitoneDist(sampleKey, effectiveKey);
    }
    semitones += pitchOffset;

    // Apply pitch as ratio
    const pitchRatio = semitonesToRatio(semitones);

    // Tempo: stretch to match upload BPM at selected multiplier
    let tRatio = 1;
    if (timestrechEnabled && sampleBpm > 0 && uploadBpm > 0) {
      tRatio = tempoRatio(sampleBpm, uploadBpm, tempoMultiplier);
      // Clamp to reasonable range
      tRatio = Math.max(0.25, Math.min(4.0, tRatio));
    }

    // Set SoundTouch params
    try {
      node.pitch.value = pitchRatio;
      node.tempo.value = tRatio;
    } catch (e) {
      // Fallback: try pitchSemitones if available
      try {
        if (node.pitchSemitones) node.pitchSemitones.value = semitones;
        if (node.tempo) node.tempo.value = tRatio;
      } catch (_) {}
    }
  }, [transposeEnabled, effectiveKey, pitchOffset, timestrechEnabled, uploadBpm, tempoMultiplier]);

  /**
   * Get the effective tempo ratio for a given sample (used for duration calculation).
   */
  const getTempoRatio = useCallback((sampleBpm) => {
    if (!timestrechEnabled || !sampleBpm || !uploadBpm) return 1;
    const r = tempoRatio(sampleBpm, uploadBpm, tempoMultiplier);
    return Math.max(0.25, Math.min(4.0, r));
  }, [timestrechEnabled, uploadBpm, tempoMultiplier]);

  /**
   * Get the total semitone shift for a given sample key.
   */
  const getSemitones = useCallback((sampleKey) => {
    let s = 0;
    if (transposeEnabled && effectiveKey && sampleKey) {
      s = semitoneDist(sampleKey, effectiveKey);
    }
    return s + pitchOffset;
  }, [transposeEnabled, effectiveKey, pitchOffset]);

  // ── Set upload analysis results ───────────────────────────────────────

  const setUploadAnalysis = useCallback((key, bpm) => {
    setUploadKey(key || null);
    setUploadBpm(bpm || 0);
    // Auto-set target key to upload key
    if (key && key !== "N/A") {
      setTargetKey(key);
    }
  }, []);

  return {
    // State
    transposeEnabled, setTransposeEnabled,
    timestrechEnabled, setTimestrechEnabled,
    targetKey, setTargetKey,
    pitchOffset, setPitchOffset,
    tempoMultiplier, setTempoMultiplier,
    uploadKey, uploadBpm,
    useFlats, setUseFlats,

    // Derived
    effectiveKey,
    noteNames,
    bpmOptions,

    // Audio engine
    createNode,
    applyParams,
    getTempoRatio,
    getSemitones,
    setUploadAnalysis,

    // Constants (for UI)
    NOTE_NAMES_SHARP,
    NOTE_NAMES_FLAT,
  };
}
