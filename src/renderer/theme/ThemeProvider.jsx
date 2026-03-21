/**
 * RESONATE — Theme Provider.
 * React Context for light/dark theme with system preference detection.
 */

import { createContext, useContext, useState, useEffect, useCallback } from "react";
import { lightTheme, darkTheme } from "./colors";

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
  const [mode, setMode] = useState(() => {
    const saved = localStorage.getItem("resonate_theme");
    return saved || "light";
  });

  const theme = mode === "dark" ? darkTheme : lightTheme;

  const toggleTheme = useCallback(() => {
    setMode(prev => {
      const next = prev === "dark" ? "light" : "dark";
      localStorage.setItem("resonate_theme", next);
      return next;
    });
  }, []);


  // Apply theme to CSS custom properties for scrollbar styling etc.
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", mode);
  }, [mode]);

  return (
    <ThemeContext.Provider value={{ theme, mode, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
