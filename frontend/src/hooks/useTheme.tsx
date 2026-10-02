/**
 * Theme ownership — the single source of truth for light/dark across the app.
 *
 * Why this exists: photogrammetric output is a *photograph*. A textured mesh
 * and a coloured point cloud are read against whatever surface sits behind
 * them, and the panel chrome shares that surface. On the dark obsidian shell
 * a dark-textured model of dark terrain loses contrast entirely; on a light
 * shell a pale scan washes out. So the choice belongs to the viewer, not to
 * the stylesheet — it is applied as `data-theme` on <html> and every surface
 * (chrome, panels, 3D backdrop, grid) reads it.
 *
 * The user's choice is persisted; "system" follows the OS setting and keeps
 * following it while the app is open.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type ThemeChoice = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

/** Persisted choice. Deliberately a different key space from auth/run state. */
export const THEME_STORAGE_KEY = "strata.theme";

interface ThemeValue {
  /** What the user picked (may be "system"). */
  choice: ThemeChoice;
  /** What is actually painted right now. */
  resolved: ResolvedTheme;
  setChoice: (choice: ThemeChoice) => void;
  /** Flip between light and dark, pinning the choice (leaves "system"). */
  toggle: () => void;
}

const ThemeContext = createContext<ThemeValue | null>(null);

function prefersLight(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return window.matchMedia("(prefers-color-scheme: light)").matches;
}

function readStoredChoice(): ThemeChoice {
  try {
    const raw = localStorage.getItem(THEME_STORAGE_KEY);
    if (raw === "light" || raw === "dark" || raw === "system") return raw;
  } catch {
    // Storage can be unavailable (private mode); fall back to the OS setting.
  }
  return "system";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoiceState] = useState<ThemeChoice>(readStoredChoice);
  const [systemPrefersLight, setSystemPrefersLight] = useState<boolean>(prefersLight);

  // Keep following the OS while the choice is "system".
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const query = window.matchMedia("(prefers-color-scheme: light)");
    const onChange = (event: MediaQueryListEvent) => setSystemPrefersLight(event.matches);
    query.addEventListener?.("change", onChange);
    return () => query.removeEventListener?.("change", onChange);
  }, []);

  const resolved: ResolvedTheme =
    choice === "system" ? (systemPrefersLight ? "light" : "dark") : choice;

  // Apply to the document so CSS (and the viewer's grid/backdrop) can react.
  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.dataset.theme = resolved;
    document.documentElement.style.colorScheme = resolved;
  }, [resolved]);

  const setChoice = useCallback((next: ThemeChoice) => {
    setChoiceState(next);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Non-fatal: the choice still applies for this session.
    }
  }, []);

  const toggle = useCallback(() => {
    setChoiceState((current) => {
      const now =
        (current === "system" ? (prefersLight() ? "light" : "dark") : current) === "dark"
          ? "light"
          : "dark";
      try {
        localStorage.setItem(THEME_STORAGE_KEY, now);
      } catch {
        // Non-fatal.
      }
      return now;
    });
  }, []);

  const value = useMemo<ThemeValue>(
    () => ({ choice, resolved, setChoice, toggle }),
    [choice, resolved, setChoice, toggle],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeValue {
  const value = useContext(ThemeContext);
  if (!value) {
    throw new Error("useTheme must be used inside <ThemeProvider>");
  }
  return value;
}
