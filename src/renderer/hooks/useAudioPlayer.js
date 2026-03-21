/**
 * RESONATE — Audio Player Hook.
 * Dual-channel playback (sample + track mix), seek, volume control.
 * Supports real-time playback rate adjustment for DAW tempo sync.
 */

import { useState, useRef, useCallback, useEffect } from "react";

const API = "http://localhost:8000";

export function useAudioPlayer() {
  const sRef = useRef(null);
  const tRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [currentId, setCurrentId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [mixMode, setMixMode] = useState(false);
  const [trackVol, setTrackVol] = useState(0.7);
  const [sampleVol, setSampleVol] = useState(1.0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const currentPathRef = useRef(null);

  const play = useCallback((path, id, sync = false) => {
    if (sRef.current) { sRef.current.pause(); sRef.current = null; }
    const a = new Audio(API + "/samples/audio/" + encodeURI(path) + (sync ? "?sync=1" : ""));
    a.volume = sampleVol;
    a.preservesPitch = false;  // Allow pitch to shift with tempo
    a.playbackRate = playbackRate;
    a.onloadedmetadata = () => setDuration(a.duration || 0);
    a.ontimeupdate = () => { if (a.duration) { setProgress(a.currentTime / a.duration); setCurrentTime(a.currentTime); } };
    a.onended = () => { setPlaying(false); setCurrentId(null); setProgress(0); setCurrentTime(0); if (tRef.current) tRef.current.pause(); };
    a.onerror = () => { setPlaying(false); setCurrentId(null); setProgress(0); };
    a.play(); sRef.current = a; currentPathRef.current = path; setPlaying(true); setCurrentId(id);
    if (mixMode && tRef.current) { tRef.current.currentTime = 0; tRef.current.volume = trackVol; tRef.current.play().catch(() => {}); }
  }, [mixMode, trackVol, sampleVol, playbackRate]);

  const pause = useCallback(() => { if (sRef.current) sRef.current.pause(); if (tRef.current) tRef.current.pause(); setPlaying(false); }, []);
  const toggle = useCallback((path, id, sync = false) => { if (currentId === id && playing) pause(); else play(path, id, sync); }, [currentId, playing, play, pause]);
  const stop = useCallback(() => { if (sRef.current) { sRef.current.pause(); sRef.current = null; } if (tRef.current) tRef.current.pause(); setPlaying(false); setCurrentId(null); setProgress(0); setCurrentTime(0); setDuration(0); currentPathRef.current = null; }, []);
  const seek = useCallback((pct) => { if (sRef.current && sRef.current.duration) { sRef.current.currentTime = pct * sRef.current.duration; } }, []);
  const loadTrack = useCallback(() => { if (tRef.current) { tRef.current.pause(); tRef.current = null; } const a = new Audio(API + "/track/audio"); a.loop = true; a.volume = trackVol; tRef.current = a; }, [trackVol]);
  const toggleMix = useCallback(() => { const nm = !mixMode; setMixMode(nm); if (!nm && tRef.current) tRef.current.pause(); if (nm && playing && tRef.current) { tRef.current.currentTime = 0; tRef.current.volume = trackVol; tRef.current.play().catch(() => {}); } }, [mixMode, playing, trackVol]);

  // Live playback rate sync (for DAW tempo changes)
  useEffect(() => { if (sRef.current) { sRef.current.preservesPitch = false; sRef.current.playbackRate = playbackRate; } }, [playbackRate]);
  useEffect(() => { if (sRef.current) sRef.current.volume = sampleVol; }, [sampleVol]);
  useEffect(() => { if (tRef.current) tRef.current.volume = trackVol; }, [trackVol]);

  return { playing, currentId, toggle, stop, progress, duration, currentTime, seek, mixMode, toggleMix, trackVol, setTrackVol, sampleVol, setSampleVol, loadTrack, playbackRate, setPlaybackRate };
}
