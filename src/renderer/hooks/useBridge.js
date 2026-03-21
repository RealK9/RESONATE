/**
 * RESONATE — Bridge Hook.
 * Polls backend for DAW bridge state (BPM, key, transport).
 * Updates in real-time when VST3 plugin is connected.
 * Detects BPM changes to trigger sample re-scoring.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { API } from "./useApi";

const KEYS_ALL = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                  "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm"];

export function useBridge() {
  const [connected, setConnected] = useState(false);
  const [dawBpm, setDawBpm] = useState(120);
  const [dawPlaying, setDawPlaying] = useState(false);
  const [dawTimeSig, setDawTimeSig] = useState("4/4");
  const [dawPosition, setDawPosition] = useState(0);
  const [browseKey, setBrowseKey] = useState(null);  // Key being browsed (null = use DAW key)
  const [rescoreNeeded, setRescoreNeeded] = useState(false);
  const [dawSync, setDawSync] = useState(false);  // Sync playback & drag to DAW key/BPM
  const pollRef = useRef(null);

  // Poll bridge status
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(API + "/bridge/status");
        const d = await r.json();
        setConnected(d.connected);
        if (d.connected) {
          setDawBpm(d.bpm);
          setDawPlaying(d.playing);
          setDawTimeSig(`${d.timeSigNum}/${d.timeSigDen}`);
          setDawPosition(d.position);
          // Backend reports when BPM has drifted enough to warrant re-scoring
          if (d.rescoreNeeded) {
            setRescoreNeeded(true);
          }
        } else {
          // If bridge disconnects after being connected, trigger rescore to revert to track BPM
          if (connected && !d.connected) {
            setRescoreNeeded(true);
            setDawSync(false);  // Auto-disable sync when bridge disconnects
          }
        }
      } catch {
        setConnected(false);
      }
    };

    // Poll faster when connected (150ms), slower when not (2s)
    const start = () => {
      poll();
      pollRef.current = setInterval(poll, connected ? 150 : 2000);
    };
    start();
    return () => clearInterval(pollRef.current);
  }, [connected]);

  // Allow App to clear the rescore flag after refetching samples
  const clearRescore = useCallback(() => setRescoreNeeded(false), []);

  // Send key change to DAW
  const sendKeyChange = useCallback(async (key) => {
    try {
      await fetch(API + "/bridge/key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
    } catch {}
  }, []);

  return {
    connected,
    dawBpm,
    dawPlaying,
    dawTimeSig,
    dawPosition,
    browseKey,
    setBrowseKey,
    sendKeyChange,
    rescoreNeeded,
    clearRescore,
    dawSync,
    setDawSync,
    allKeys: KEYS_ALL,
  };
}
