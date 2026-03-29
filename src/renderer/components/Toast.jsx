/**
 * RESONATE — Toast Notification System.
 * Frosted glass toasts with smooth slide-in, progress bar, auto-dismiss.
 */

import { useState, useMemo, useCallback, useEffect, createContext, useContext } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { AF } from "../theme/fonts";

const ToastContext = createContext();

let toastId = 0;

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((message, type = "info", duration = 3000) => {
    const id = ++toastId;
    setToasts(prev => [...prev, { id, message, type, duration, created: Date.now() }]);
    if (duration > 0) {
      setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), duration);
    }
    return id;
  }, []);

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const api = useMemo(() => ({
    success: (msg, dur) => addToast(msg, "success", dur),
    error: (msg, dur) => addToast(msg, "error", dur || 5000),
    info: (msg, dur) => addToast(msg, "info", dur),
    remove: removeToast,
  }), [addToast, removeToast]);

  return (
    <ToastContext.Provider value={api}>
      {children}
      <ToastContainer toasts={toasts} onDismiss={removeToast} />
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}

function ToastItem({ toast, onDismiss, isDark, theme }) {
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    if (toast.duration <= 0) return;
    const start = toast.created;
    const interval = setInterval(() => {
      const elapsed = Date.now() - start;
      const remaining = Math.max(0, 100 - (elapsed / toast.duration) * 100);
      setProgress(remaining);
      if (remaining <= 0) clearInterval(interval);
    }, 50);
    return () => clearInterval(interval);
  }, [toast.created, toast.duration]);

  const colors = {
    success: {
      bg: isDark ? "rgba(34,197,94,0.1)" : "rgba(22,163,74,0.06)",
      border: isDark ? "rgba(34,197,94,0.2)" : "rgba(22,163,74,0.15)",
      icon: theme.green,
      bar: "#22C55E",
    },
    error: {
      bg: isDark ? "rgba(239,68,68,0.1)" : "rgba(220,38,38,0.06)",
      border: isDark ? "rgba(239,68,68,0.2)" : "rgba(220,38,38,0.15)",
      icon: theme.red,
      bar: "#EF4444",
    },
    info: {
      bg: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)",
      border: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)",
      icon: theme.textMuted,
      bar: "#D946EF",
    },
  };

  const c = colors[toast.type] || colors.info;

  return (
    <div
      style={{
        pointerEvents: "auto",
        padding: "10px 14px 10px 14px",
        borderRadius: 10,
        background: c.bg,
        border: "1px solid " + c.border,
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        display: "flex",
        alignItems: "center",
        gap: 8,
        animation: "screenFadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1)",
        cursor: "pointer",
        maxWidth: 360,
        position: "relative",
        overflow: "hidden",
        boxShadow: isDark
          ? "0 8px 24px rgba(0,0,0,0.3)"
          : "0 8px 24px rgba(0,0,0,0.06)",
      }}
      onClick={() => onDismiss(toast.id)}
    >
      {/* Icon */}
      <div style={{
        width: 7, height: 7, borderRadius: "50%", background: c.icon, flexShrink: 0,
        boxShadow: `0 0 6px ${c.icon}40`,
      }} />
      <span style={{ fontSize: 11, color: theme.text, fontFamily: AF, fontWeight: 500 }}>{toast.message}</span>
      {/* Progress bar */}
      {toast.duration > 0 && (
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0, height: 2,
          background: "transparent",
        }}>
          <div style={{
            height: "100%",
            width: progress + "%",
            background: c.bar,
            opacity: 0.3,
            transition: "width 0.1s linear",
            borderRadius: "0 0 0 10px",
          }} />
        </div>
      )}
    </div>
  );
}

function ToastContainer({ toasts, onDismiss }) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  if (toasts.length === 0) return null;

  return (
    <div style={{
      position: "fixed", bottom: 16, right: 16, zIndex: 2000,
      display: "flex", flexDirection: "column", gap: 8, pointerEvents: "none",
    }}>
      {toasts.map(t => (
        <ToastItem key={t.id} toast={t} onDismiss={onDismiss} isDark={isDark} theme={theme} />
      ))}
    </div>
  );
}
