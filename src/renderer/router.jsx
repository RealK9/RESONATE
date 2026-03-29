/**
 * RESONATE — Simple Hash Router.
 * Routes: home, sounds, collections, library.
 */

import { createContext, useContext, useState, useEffect, useCallback } from "react";

const RouterContext = createContext({ page: "home", navigate: () => {}, params: {} });

export function useRouter() { return useContext(RouterContext); }

export function Router({ children }) {
  const [page, setPage] = useState(() => {
    const h = window.location.hash.slice(1) || "home";
    return h.split("?")[0];
  });
  const [params, setParams] = useState({});

  useEffect(() => {
    const handler = () => {
      const h = window.location.hash.slice(1) || "home";
      const [p, query] = h.split("?");
      setPage(p);
      setParams(Object.fromEntries(new URLSearchParams(query || "")));
    };
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);

  const navigate = useCallback((p, q) => {
    const qs = q ? "?" + new URLSearchParams(q).toString() : "";
    window.location.hash = p + qs;
  }, []);

  return (
    <RouterContext.Provider value={{ page, navigate, params }}>
      {children}
    </RouterContext.Provider>
  );
}
