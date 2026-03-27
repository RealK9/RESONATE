/**
 * RESONATE — Audio Player Hook.
 * Dual-channel playback (sample + track mix) using Web Audio API for
 * sample-accurate synchronization. Supports seek, volume, and playback rate.
 */

import { useState, useRef, useCallback, useEffect } from "react";

const API = "http://localhost:8000";

/**
 * Fetch an audio file and decode it into an AudioBuffer.
 */
async function fetchBuffer(ctx, url) {
  const res = await fetch(url);
  const arr = await res.arrayBuffer();
  return ctx.decodeAudioData(arr);
}

export function useAudioPlayer() {
  // ── State ────────────────────────────────────────────────────────────
  const [playing, setPlaying] = useState(false);
  const [currentId, setCurrentId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [mixMode, setMixMode] = useState(false);
  const [trackVol, setTrackVol] = useState(0.7);
  const [sampleVol, setSampleVol] = useState(1.0);
  const [playbackRate, setPlaybackRate] = useState(1.0);

  // ── Refs (Web Audio) ─────────────────────────────────────────────────
  const ctxRef = useRef(null);           // AudioContext
  const sSourceRef = useRef(null);       // BufferSourceNode for sample
  const tSourceRef = useRef(null);       // BufferSourceNode for track
  const sGainRef = useRef(null);         // GainNode for sample
  const tGainRef = useRef(null);         // GainNode for track
  const trackBufferRef = useRef(null);   // Decoded AudioBuffer for track
  const sampleBufferRef = useRef(null);  // Decoded AudioBuffer for current sample
  const startTimeRef = useRef(0);        // AudioContext.currentTime when playback started
  const rafRef = useRef(null);           // requestAnimationFrame id
  const currentPathRef = useRef(null);
  const mixModeRef = useRef(false);
  const playbackRateRef = useRef(1.0);

  // Keep refs in sync with state
  useEffect(() => { mixModeRef.current = mixMode; }, [mixMode]);
  useEffect(() => { playbackRateRef.current = playbackRate; }, [playbackRate]);

  /** Lazy-init the AudioContext. */
  const getCtx = useCallback(() => {
    if (!ctxRef.current) {
      ctxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      sGainRef.current = ctxRef.current.createGain();
      tGainRef.current = ctxRef.current.createGain();
      sGainRef.current.connect(ctxRef.current.destination);
      tGainRef.current.connect(ctxRef.current.destination);
    }
    // Resume if suspended (browser autoplay policy)
    if (ctxRef.current.state === "suspended") ctxRef.current.resume();
    return ctxRef.current;
  }, []);

  /** Stop the progress animation loop. */
  const stopRaf = useCallback(() => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
  }, []);

  /** Start the progress animation loop. */
  const startRaf = useCallback((buf) => {
    stopRaf();
    const tick = () => {
      const ctx = ctxRef.current;
      if (!ctx || !buf) return;
      const elapsed = (ctx.currentTime - startTimeRef.current) * playbackRateRef.current;
      const dur = buf.duration;
      if (elapsed >= dur) {
        // Sample ended
        setPlaying(false);
        setCurrentId(null);
        setProgress(0);
        setCurrentTime(0);
        stopRaf();
        // Stop track too
        if (tSourceRef.current) { try { tSourceRef.current.stop(); } catch (_) {} tSourceRef.current = null; }
        return;
      }
      setProgress(elapsed / dur);
      setCurrentTime(elapsed);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [stopRaf]);

  /** Stop all currently playing sources. */
  const stopSources = useCallback(() => {
    if (sSourceRef.current) { try { sSourceRef.current.stop(); } catch (_) {} sSourceRef.current = null; }
    if (tSourceRef.current) { try { tSourceRef.current.stop(); } catch (_) {} tSourceRef.current = null; }
    stopRaf();
  }, [stopRaf]);

  // ── Play ──────────────────────────────────────────────────────────────
  const play = useCallback(async (path, id, sync = false) => {
    const ctx = getCtx();
    stopSources();

    try {
      // Fetch and decode the sample
      const url = API + "/samples/audio/" + encodeURI(path) + (sync ? "?sync=1" : "");
      const buf = await fetchBuffer(ctx, url);
      sampleBufferRef.current = buf;
      setDuration(buf.duration);

      // Create sample source
      const sSource = ctx.createBufferSource();
      sSource.buffer = buf;
      sSource.playbackRate.value = playbackRateRef.current;
      sSource.connect(sGainRef.current);
      sGainRef.current.gain.value = sampleVol;
      sSourceRef.current = sSource;

      // When sample ends naturally
      sSource.onended = () => {
        if (sSourceRef.current === sSource) {
          setPlaying(false);
          setCurrentId(null);
          setProgress(0);
          setCurrentTime(0);
          stopRaf();
          if (tSourceRef.current) { try { tSourceRef.current.stop(); } catch (_) {} tSourceRef.current = null; }
        }
      };

      // Schedule both at EXACTLY the same time
      const startAt = ctx.currentTime + 0.01; // tiny lookahead for precision
      sSource.start(startAt);

      // If mix mode is on and we have the track loaded, play it in sync
      if (mixModeRef.current && trackBufferRef.current) {
        const tSource = ctx.createBufferSource();
        tSource.buffer = trackBufferRef.current;
        tSource.loop = true;
        tSource.playbackRate.value = playbackRateRef.current;
        tSource.connect(tGainRef.current);
        tGainRef.current.gain.value = trackVol;
        tSource.start(startAt); // Same exact start time — perfectly synced
        tSourceRef.current = tSource;
      }

      startTimeRef.current = startAt;
      currentPathRef.current = path;
      setPlaying(true);
      setCurrentId(id);
      startRaf(buf);
    } catch (e) {
      console.error("Playback error:", e);
      setPlaying(false);
      setCurrentId(null);
      setProgress(0);
    }
  }, [getCtx, stopSources, sampleVol, trackVol, startRaf, stopRaf]);

  // ── Pause / Toggle / Stop ─────────────────────────────────────────────
  const pause = useCallback(() => {
    stopSources();
    setPlaying(false);
  }, [stopSources]);

  const toggle = useCallback((path, id, sync = false) => {
    if (currentId === id && playing) pause();
    else play(path, id, sync);
  }, [currentId, playing, play, pause]);

  const stop = useCallback(() => {
    stopSources();
    setPlaying(false);
    setCurrentId(null);
    setProgress(0);
    setCurrentTime(0);
    setDuration(0);
    currentPathRef.current = null;
    sampleBufferRef.current = null;
  }, [stopSources]);

  // ── Seek ──────────────────────────────────────────────────────────────
  const seek = useCallback((pct) => {
    if (!sampleBufferRef.current || !ctxRef.current) return;
    const ctx = ctxRef.current;
    const buf = sampleBufferRef.current;
    const offset = pct * buf.duration;

    // Stop current sources
    stopSources();

    // Re-create sample source at new offset
    const sSource = ctx.createBufferSource();
    sSource.buffer = buf;
    sSource.playbackRate.value = playbackRateRef.current;
    sSource.connect(sGainRef.current);
    sSourceRef.current = sSource;

    sSource.onended = () => {
      if (sSourceRef.current === sSource) {
        setPlaying(false);
        setCurrentId(null);
        setProgress(0);
        setCurrentTime(0);
        stopRaf();
        if (tSourceRef.current) { try { tSourceRef.current.stop(); } catch (_) {} tSourceRef.current = null; }
      }
    };

    const startAt = ctx.currentTime + 0.005;
    sSource.start(startAt, offset);

    // Re-create track source at same offset if mix mode
    if (mixModeRef.current && trackBufferRef.current) {
      const tSource = ctx.createBufferSource();
      tSource.buffer = trackBufferRef.current;
      tSource.loop = true;
      tSource.playbackRate.value = playbackRateRef.current;
      tSource.connect(tGainRef.current);
      tSource.start(startAt, offset % trackBufferRef.current.duration);
      tSourceRef.current = tSource;
    }

    // Adjust startTimeRef so progress calculation accounts for the seek offset
    startTimeRef.current = startAt - (offset / playbackRateRef.current);
    startRaf(buf);
  }, [stopSources, startRaf, stopRaf]);

  // ── Load Track (pre-decode for instant sync) ──────────────────────────
  const loadTrack = useCallback(async () => {
    const ctx = getCtx();
    try {
      const buf = await fetchBuffer(ctx, API + "/track/audio");
      trackBufferRef.current = buf;
    } catch (e) {
      console.warn("Failed to load track audio:", e);
      trackBufferRef.current = null;
    }
  }, [getCtx]);

  // ── Toggle Mix Mode ───────────────────────────────────────────────────
  const toggleMix = useCallback(() => {
    const nm = !mixMode;
    setMixMode(nm);

    if (!nm) {
      // Turning mix OFF — stop the track channel
      if (tSourceRef.current) { try { tSourceRef.current.stop(); } catch (_) {} tSourceRef.current = null; }
    } else if (playing && ctxRef.current) {
      if (!trackBufferRef.current) {
        console.warn("MIX: Track not loaded yet — loading now...");
        // Attempt to load track on the fly
        fetchBuffer(ctxRef.current, API + "/track/audio")
          .then(buf => { trackBufferRef.current = buf; })
          .catch(e => console.warn("MIX: Failed to load track:", e));
        return;
      }
      // Turning mix ON while sample is playing — start track in sync
      const ctx = ctxRef.current;
      const elapsed = (ctx.currentTime - startTimeRef.current) * playbackRateRef.current;

      const tSource = ctx.createBufferSource();
      tSource.buffer = trackBufferRef.current;
      tSource.loop = true;
      tSource.playbackRate.value = playbackRateRef.current;
      tSource.connect(tGainRef.current);
      // Start at the same elapsed offset so they're in sync
      tSource.start(0, elapsed % trackBufferRef.current.duration);
      tSourceRef.current = tSource;
    }
  }, [mixMode, playing]);

  // ── Volume reactivity ─────────────────────────────────────────────────
  useEffect(() => { if (sGainRef.current) sGainRef.current.gain.value = sampleVol; }, [sampleVol]);
  useEffect(() => { if (tGainRef.current) tGainRef.current.gain.value = trackVol; }, [trackVol]);

  // ── Playback rate reactivity ──────────────────────────────────────────
  useEffect(() => {
    if (sSourceRef.current) sSourceRef.current.playbackRate.value = playbackRate;
    if (tSourceRef.current) tSourceRef.current.playbackRate.value = playbackRate;
  }, [playbackRate]);

  // ── Preview In Context (enable mix + play in one action) ─────────────
  const previewInContext = useCallback(async (id, path) => {
    // Ensure mix mode is on
    if (!mixModeRef.current) {
      setMixMode(true);
      mixModeRef.current = true;
    }
    // Ensure track is loaded
    if (!trackBufferRef.current) {
      const ctx = getCtx();
      try {
        const buf = await fetchBuffer(ctx, API + "/track/audio");
        trackBufferRef.current = buf;
      } catch (e) {
        console.warn("previewInContext: Failed to load track:", e);
      }
    }
    // Now play — mix mode is on so track will play in sync
    await play(path, id, false);
  }, [getCtx, play]);

  // ── Cleanup on unmount ────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopRaf();
      if (ctxRef.current) ctxRef.current.close().catch(() => {});
    };
  }, [stopRaf]);

  return {
    playing, currentId, toggle, stop, progress, duration, currentTime, seek,
    mixMode, toggleMix, trackVol, setTrackVol, sampleVol, setSampleVol,
    loadTrack, playbackRate, setPlaybackRate, previewInContext,
  };
}
