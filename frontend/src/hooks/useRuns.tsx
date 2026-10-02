import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { friendlyMessage } from "../lib/errors";
import type { RunSummary } from "../lib/types";

const SELECTED_KEY = "strata.selectedRun";

interface RunsState {
  runs: RunSummary[];
  loading: boolean;
  error: string | null;
  selected: RunSummary | null;
  selectRun: (runId: string) => void;
  refresh: () => void;
}

const RunsCtx = createContext<RunsState>({
  runs: [],
  loading: true,
  error: null,
  selected: null,
  selectRun: () => {},
  refresh: () => {},
});

function storedSelection(): string | null {
  try {
    return localStorage.getItem(SELECTED_KEY);
  } catch {
    return null;
  }
}

export function RunsProvider({ children }: { children: React.ReactNode }) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(storedSelection());

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getRuns()
      .then((res) => setRuns(res.runs ?? []))
      .catch((err) => setError(friendlyMessage(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(refresh, [refresh]);

  const selectRun = useCallback((runId: string) => {
    setSelectedId(runId);
    try {
      localStorage.setItem(SELECTED_KEY, runId);
    } catch {
      /* noop */
    }
  }, []);

  const selected = useMemo(() => {
    if (selectedId) {
      // The explicit selection wins only when it actually exists in the
      // loaded run list. While a just-selected run is still being discovered
      // (refresh in flight) there is nothing to show yet — silently falling
      // back to some *other* run is exactly the "the viewer shows the demo
      // / someone else's reconstruction" bug this must never reintroduce.
      return runs.find((r) => r.run_id === selectedId) ?? null;
    }
    return runs.find((r) => !r.is_demo) ?? runs[0] ?? null;
  }, [runs, selectedId]);

  const value = useMemo(
    () => ({ runs, loading, error, selected, selectRun, refresh }),
    [runs, loading, error, selected, selectRun, refresh],
  );

  return <RunsCtx.Provider value={value}>{children}</RunsCtx.Provider>;
}

export function useRuns(): RunsState {
  return useContext(RunsCtx);
}