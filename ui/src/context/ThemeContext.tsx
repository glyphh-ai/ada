import React, { createContext, useContext, useState, useEffect } from "react";

type ThemePreference = "dark" | "light" | "system";
type ResolvedTheme = "dark" | "light";

interface ThemeContextValue {
  theme: ResolvedTheme;
  preference: ThemePreference;
  setPreference: (pref: ThemePreference) => void;
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

const getSystemTheme = (): ResolvedTheme =>
  window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const [preference, setPreference] = useState<ThemePreference>(
    (localStorage.getItem("theme") as ThemePreference) || "system"
  );
  const [resolved, setResolved] = useState<ResolvedTheme>(
    preference === "system" ? getSystemTheme() : (preference as ResolvedTheme)
  );

  useEffect(() => {
    if (preference === "system") {
      const update = () => setResolved(getSystemTheme());
      update();
      const mq = window.matchMedia("(prefers-color-scheme: light)");
      mq.addEventListener("change", update);
      return () => mq.removeEventListener("change", update);
    } else {
      setResolved(preference as ResolvedTheme);
    }
  }, [preference]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", resolved);
    localStorage.setItem("theme", preference);
  }, [resolved, preference]);

  return (
    <ThemeContext.Provider value={{ theme: resolved, preference, setPreference }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside ThemeProvider");
  return ctx;
};
