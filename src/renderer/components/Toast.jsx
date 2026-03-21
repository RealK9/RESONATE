/**
 * RESONATE — Toast Notification System.
 * Success, error, info toasts with auto-dismiss.
 */

import { useState, useMemo, useCallback, createContext, useContext } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { AF } from "../theme/fonts";

const ToastContext = createContext();

let toastId = 0;

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((message, type = "info", duration = 3000) => {
    const id = ++toastId;
    setToasts(prev => [...prev, { id, message, type, duration }]);
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

function ToastContainer({ toasts, onDismiss }) {
  const { theme, mode } = useTheme();
  const isDark = mode === "dark";

  if (toasts.length === 0) return null;

  const colors = {
    success: { bg: isDark ? "rgba(34,197,94,0.15)" : "rgba(22,163,74,0.08)", border: isDark ? "rgba(34,197,94,0.3)" : "rgba(22,163,74,0.2)", icon: theme.green },
    error: { bg: isDark ? "rgba(239,68,68,0.15)" : "rgba(220,38,38,0.08)", border: isDark ? "rgba(239,68,68,0.3)" : "rgba(220,38,38,0.2)", icon: theme.red },
    info: { bg: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)", border: theme.border, icon: theme.textMuted },
  };

  return (
    <div style={{ position: "fixed", bottom: 16, right: 16, zIndex: 2000, display: "flex", flexDirection: "column", gap: 6, pointerEvents: "none" }}>
      {toasts.map(t => {
        const c = colors[t.type] || colors.info;
        return (
          <div key={t.id} style={{ pointerEvents: "auto", padding: "8px 14px", borderRadius: 8, background: c.bg, border: "1px solid " + c.border, backdropFilter: "blur(12px)", display: "flex", alignItems: "center", gap: 8, animation: "fadeInUp 0.2s ease", cursor: "pointer", maxWidth: 340 }} onClick={() => onDismiss(t.id)}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: c.icon, flexShrink: 0 }} />
            <span style={{ fontSize: 11, color: theme.text, fontFamily: AF }}>{t.message}</span>
          </div>
        );
      })}
    </div>
  );
}
