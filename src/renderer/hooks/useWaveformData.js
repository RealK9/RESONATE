/**
 * RESONATE — Waveform Data Hook.
 * Decodes audio via Web Audio API and extracts peak data for visualization.
 */

import { useState, useEffect } from "react";

export function useWaveformData(url) {
  const [peaks, setPeaks] = useState(null);
  useEffect(() => {
    if (!url) { setPeaks(null); return; }
    let dead = false;
    (async () => {
      try {
        const buf = await (await fetch(url)).arrayBuffer();
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const audio = await ctx.decodeAudioData(buf);
        const raw = audio.getChannelData(0), n = 80, bs = Math.floor(raw.length / n), out = [];
        for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < bs; j++) s += Math.abs(raw[i * bs + j]); out.push(s / bs); }
        const pk = Math.max(...out) || 1;
        if (!dead) setPeaks(out.map(v => v / pk));
      } catch { if (!dead) setPeaks(null); }
    })();
    return () => { dead = true; };
  }, [url]);
  return peaks;
}
