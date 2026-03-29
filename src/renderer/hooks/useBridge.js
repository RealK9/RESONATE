/**
 * RESONATE — Bridge Hook.
 * Polls backend for DAW bridge state (BPM, key, transport).
 * Updates in real-time when VST3 plugin is connected.
 * Detects BPM changes, plugin crashes, and connection drops.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { API } from "./useApi";

const KEYS_ALL = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                  "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm"];

const POLL_FAST = 200;    // ms when connected
const POLL_SLOW = 3000;   // ms when disconnected
const STALE_THRESHOLD = 5; // consecutive failures before marking disconnected

export function useBridge() {
  const [connected, setConnected] = useState(false);
  const [dawBpm, setDawBpm] = useState(120);
  const [dawPlaying, setDawPlaying] = useState(false);
  const [dawTimeSig, setDawTimeSig] = useState("4/4");
  const [dawPosition, setDawPosition] = useState(0);
  const [browseKey, setBrowseKey] = useState(null);
  const [rescoreNeeded, setRescoreNeeded] = useState(false);
  const [dawSync, setDawSync] = useState(false);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);
  const failCountRef = useRef(0);
  const wasConnectedRef = useRef(false);

  const connectedRef = useRef(false);

  // Poll bridge status with crash detection
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(API + "/bridge/status");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();

        failCountRef.current = 0;
        setError(null);
        const wasConn = connectedRef.current;
        connectedRef.current = d.connected;
        setConnected(d.connected);

        if (d.connected) {
          setDawBpm(d.bpm);
          setDawPlaying(d.playing);
          setDawTimeSig(`${d.timeSigNum}/${d.timeSigDen}`);
          setDawPosition(d.position);

          if (d.rescoreNeeded) {
            setRescoreNeeded(true);
          }

          wasConnectedRef.current = true;
        } else if (wasConnectedRef.current && !d.connected) {
          // Plugin disconnected after being connected — likely crash or DAW close
          setRescoreNeeded(true);
          setDawSync(false);
          wasConnectedRef.current = false;
        }

        // Adjust polling rate based on connection state change
        if (d.connected !== wasConn) {
          clearInterval(pollRef.current);
          pollRef.current = setInterval(poll, d.connected ? POLL_FAST : POLL_SLOW);
        }
      } catch (e) {
        failCountRef.current++;
        if (failCountRef.current >= STALE_THRESHOLD) {
          const wasConn = connectedRef.current;
          connectedRef.current = false;
          setConnected(false);
          setError("Backend unreachable");
          if (wasConnectedRef.current) {
            setDawSync(false);
            wasConnectedRef.current = false;
          }
          if (wasConn) {
            clearInterval(pollRef.current);
            pollRef.current = setInterval(poll, POLL_SLOW);
          }
        }
      }
    };

    poll();
    pollRef.current = setInterval(poll, POLL_SLOW);
    return () => clearInterval(pollRef.current);
  }, []);

  const clearRescore = useCallback(() => setRescoreNeeded(false), []);

  // Send key change to DAW with error reporting
  const sendKeyChange = useCallback(async (key) => {
    try {
      const r = await fetch(API + "/bridge/key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      if (!r.ok) {
        setError("Failed to send key change");
        return false;
      }
      return true;
    } catch {
      setError("Bridge communication failed");
      return false;
    }
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
    error,
    allKeys: KEYS_ALL,
  };
}
