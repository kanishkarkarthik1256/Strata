import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { friendlyMessage, isApiError } from "../lib/errors";
import type { PipelineStatusResponse } from "../lib/types";

/**
 * One value for what the run is doing, computed once from the poll payload.
 * These were four booleans (`isRunning`/`isCompleted`/`isFailed`/
 * `isCancelled`) derived ad hoc at every render site.
 */
export type RunPhase = "running" | "completed" | "failed" | "cancelled" | "missing";

export interface RunMonitor {
  status: PipelineStatusResponse | null;
  phase: RunPhase;
  /** Backend-stated error, or the message from the last failed poll. */
  error: string | null;
  /** The last poll failed in transport (not a 404) — retrying is meaningful. */
  connLost: boolean;
  /** Id of the stage the backend reports as running; null when none is. */
  runningStage: string | null;
  elapsedSec: number;
  eta: { remainingSec: number; confidence: "Low" | "Moderate" | "High" } | null;
  cancelling: boolean;
  cancel: () => Promise<void>;
  retry: () => Promise<void>;
}

/** Poll cadence while the tab is visible (ms). */
const POLL_VISIBLE_MS = 2000;
/**
 * Poll cadence while the tab is hidden. Chrome intensive-throttles hidden-tab
 * timer chains to ~1/min after 5 minutes, so a 2s chain silently dies in the
 * background and the page looks frozen when the user returns. A 30s chain is
 * at or above every throttle floor (it survives), keeps the run's state
 * roughly current for a tab-switch, and costs one cheap request.
 */
const POLL_HIDDEN_MS = 30_000;

/**
 * The run's observable state while it processes: polls
 * GET /api/pipeline/status/{jobId}, and owns everything that can be said about
 * it — phase, stage, ETA, connection loss, and whether the id exists at all.
 *
 * What it deliberately does NOT own: routing, run selection, and any stage
 * label or copy — the page decides what to do with a completed run.
 *
 * Background-tab behaviour: elapsed time is derived from the wall clock
 * (`Date.now()` at mount), never incremented by a timer, so throttled or
 * suspended background timers cannot make it drift — the display is exact the
 * moment the tab returns. Polling drops to a slow heartbeat while hidden and
 * catches up immediately on `visibilitychange`.
 */
export function useRunMonitor(jobId: string): RunMonitor {
  const [status, setStatus] = useState<PipelineStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connLost, setConnLost] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  // A run id the backend does not know (stale link, deleted run) used to
  // render the full live screen forever: every stage "Processing…" over a
  // 404 dressed up as "Connection temporarily unavailable — retrying…".
  const [notFound, setNotFound] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);

  // Progress-rate history for the ETA estimator: one sample per poll while
  // the running stage reports a numeric progress fraction.
  const rateHistory = useRef<Array<{ progress: number; elapsedSec: number }>>([]);
  const lastStageRef = useRef<string | null>(null);
  const notFoundStreak = useRef(0);
  // Wall-clock second the monitor mounted — the elapsed display's anchor.
  const startedAtRef = useRef(Date.now());
  // Readable from the polling callback, which is not re-created each second.
  const elapsedSecRef = useRef(0);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await api.getPipelineStatus(jobId);
      setStatus(res);
      setConnLost(false);
      setNotFound(false);
      notFoundStreak.current = 0;
      if (res.error) setError(res.error);
      const runningStage = Object.entries(res.stages ?? {}).find(
        ([, s]) => s?.status === "running",
      )?.[0];
      const progress = res.stages?.[runningStage ?? ""]?.progress;
      if (
        runningStage &&
        typeof progress === "number" &&
        (lastStageRef.current === null || lastStageRef.current === runningStage)
      ) {
        lastStageRef.current = runningStage;
        const hist = rateHistory.current;
        const last = hist[hist.length - 1];
        if (!last || last.progress !== progress) {
          hist.push({ progress, elapsedSec: elapsedSecRef.current });
        }
      } else if (runningStage && lastStageRef.current !== runningStage) {
        // New stage → reset the estimator; per-stage rates don't transfer.
        lastStageRef.current = runningStage;
        rateHistory.current = [];
      }
    } catch (err) {
      const missing = isApiError(err) && err.status === 404;
      // Two consecutive 404s rule out the brief window between an upload
      // returning a job id and the project row being visible to this read.
      // A real 404 never resolves, so the screen must stop pretending.
      notFoundStreak.current = missing ? notFoundStreak.current + 1 : 0;
      if (notFoundStreak.current >= 2) setNotFound(true);
      // Temporary polling failure: keep the last known state, never flip the
      // whole screen to an error, and never mark the run failed.
      if (!missing) {
        setConnLost(true);
        setError(friendlyMessage(err));
      }
    }
  }, [jobId]);

  // Elapsed time from the wall clock: recomputed against Date.now(), so a
  // throttled background interval can never make it lag or jump — it is
  // simply correct whenever it next runs.
  useEffect(() => {
    const tick = () => {
      const sec = Math.max(0, Math.floor((Date.now() - startedAtRef.current) / 1000));
      elapsedSecRef.current = sec;
      setElapsedSec(sec);
    };
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, []);

  // Polling with visibility-aware cadence: 2s while the user watches, a 30s
  // heartbeat while hidden, and an immediate catch-up poll the moment the
  // tab becomes visible again.
  useEffect(() => {
    if (notFound) return;
    let timer: number | undefined;
    let visible = !document.hidden;

    const schedule = () => {
      if (timer !== undefined) clearInterval(timer);
      timer = window.setInterval(fetchStatus, visible ? POLL_VISIBLE_MS : POLL_HIDDEN_MS);
    };

    const onVisibility = () => {
      const nowVisible = !document.hidden;
      if (nowVisible && !visible) {
        // Returned to the tab: refresh now instead of waiting out the
        // background heartbeat, then resume the fast cadence.
        void fetchStatus();
      }
      visible = nowVisible;
      schedule();
    };

    fetchStatus();
    schedule();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      if (timer !== undefined) clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [fetchStatus, notFound]);

  const cancel = useCallback(async () => {
    try {
      setCancelling(true);
      await api.cancelPipeline(jobId);
      await fetchStatus();
    } catch (err) {
      setError(friendlyMessage(err));
    } finally {
      setCancelling(false);
    }
  }, [jobId, fetchStatus]);

  const retry = useCallback(async () => {
    try {
      setError(null);
      rateHistory.current = [];
      lastStageRef.current = null;
      startedAtRef.current = Date.now();
      await api.startPipeline(jobId);
      await fetchStatus();
    } catch (err) {
      setError(friendlyMessage(err));
    }
  }, [jobId, fetchStatus]);

  const missing = notFound && status === null;
  const backendStatus = status?.status;
  const phase: RunPhase = missing
    ? "missing"
    : backendStatus === "completed"
      ? "completed"
      : backendStatus === "failed"
        ? "failed"
        : backendStatus === "cancelled"
          ? "cancelled"
          : "running";

  const runningStage =
    Object.entries(status?.stages ?? {}).find(([, s]) => s?.status === "running")?.[0] ?? null;

  const eta = useMemo(() => estimateEta(rateHistory.current), [status]);

  return {
    status,
    phase,
    error,
    connLost,
    runningStage,
    elapsedSec,
    eta,
    cancelling,
    cancel,
    retry,
  };
}

/**
 * ETA estimator — honest by construction.
 *
 * The backend's per-stage `progress` fraction (0..1, from real stage events)
 * is the only input. With k >= 2 observations of (progress, elapsed) we take
 * the *observed* rate between the last two samples and project the remaining
 * stage time; a single observation yields nothing. Fewer than 2 observations →
 * null ("Estimating…"). A longer history raises confidence, never the number.
 * We never use wall-clock guesses, and the result is clamped non-negative.
 */
export function estimateEta(
  history: Array<{ progress: number; elapsedSec: number }>,
): { remainingSec: number; confidence: "Low" | "Moderate" | "High" } | null {
  if (history.length < 2) return null;
  const last = history[history.length - 1];
  const prev = history[history.length - 2];
  const dProgress = last.progress - prev.progress;
  const dElapsed = last.elapsedSec - prev.elapsedSec;
  if (dProgress <= 0 || dElapsed <= 0) {
    // No measurable forward movement — refuse to invent a number.
    return null;
  }
  const rate = dProgress / dElapsed; // fraction of stage per second (recent)
  const remaining = Math.max(0, (1 - last.progress) / rate);
  const n = history.length;
  const confidence = n >= 5 ? "High" : n >= 3 ? "Moderate" : "Low";
  return { remainingSec: remaining, confidence };
}
