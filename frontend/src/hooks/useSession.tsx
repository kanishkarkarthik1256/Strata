/**
 * Session — the app's single owner of "who is signed in".
 *
 * The console gates everything behind an account, so this hook is the one
 * place that holds the token + profile. On boot it validates any stored token
 * against `GET /api/auth/me`: a revoked or expired token is discarded rather
 * than left to fail every request in the app with a 401.
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
import { api, clearToken, getToken, setToken } from "../lib/api";
import type { AuthUser } from "../lib/types";

const USER_KEY = "strata.user";

export type SessionStatus = "checking" | "signed-out" | "signed-in";

interface SessionValue {
  status: SessionStatus;
  user: AuthUser | null;
  signIn: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const SessionContext = createContext<SessionValue | null>(null);

function readStoredUser(): AuthUser | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  } catch {
    return null;
  }
}

function writeStoredUser(user: AuthUser | null): void {
  try {
    if (user) localStorage.setItem(USER_KEY, JSON.stringify(user));
    else localStorage.removeItem(USER_KEY);
  } catch {
    /* storage unavailable — session-only auth */
  }
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<SessionStatus>(() => (getToken() ? "checking" : "signed-out"));
  const [user, setUser] = useState<AuthUser | null>(() => readStoredUser());

  // Validate the stored token once at boot.
  useEffect(() => {
    let cancelled = false;
    if (!getToken()) {
      setStatus("signed-out");
      return;
    }
    api
      .me()
      .then((me) => {
        if (cancelled) return;
        setUser(me);
        writeStoredUser(me);
        setStatus("signed-in");
      })
      .catch(() => {
        if (cancelled) return;
        // Expired/revoked token: drop it so the login page is honest about
        // the state instead of the app 401-ing on every page.
        clearToken();
        writeStoredUser(null);
        setUser(null);
        setStatus("signed-out");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const adopt = useCallback((token: string, profile: AuthUser) => {
    setToken(token);
    writeStoredUser(profile);
    setUser(profile);
    setStatus("signed-in");
  }, []);

  const signIn = useCallback(
    async (email: string, password: string) => {
      const res = await api.login(email, password);
      adopt(res.token, res.user);
    },
    [adopt],
  );

  const register = useCallback(
    async (email: string, password: string) => {
      await api.register(email, password);
      await signIn(email, password);
    },
    [signIn],
  );

  const signOut = useCallback(async () => {
    await api.logout();
    clearToken();
    writeStoredUser(null);
    setUser(null);
    setStatus("signed-out");
  }, []);

  const value = useMemo<SessionValue>(
    () => ({ status, user, signIn, register, signOut }),
    [status, user, signIn, register, signOut],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionValue {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used inside SessionProvider");
  return ctx;
}
