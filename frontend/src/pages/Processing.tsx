import { useEffect, useState, type CSSProperties } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { useRuns } from "../hooks/useRuns";
// Everything the screen can say about the run comes from the monitor; this
// file only decides how to show it.
import { useRunMonitor } from "../hooks/useRunMonitor";
import { Icon, type IconName } from "../components/ui/Icon";
import TelemetryBlock, { type TelemetrySyncView } from "../components/telemetry/TelemetryBlock";
import { depthBackendLabel, depthBackendNote } from "../lib/depthBackend";
import { STAGES, overallProgress, stageProgressList } from "../lib/pipelineStages";
import { formatClock } from "../lib/format";
import StageFailureDetails, { depthBackendId } from "../components/processing/StageFailureDetails";

function formatEta(sec: number): string {
  // Rounded, clamped, never negative — "~" marks it as an estimate.
  const clamped = Math.max(0, Math.round(sec / 30) * 30);
  if (clamped < 60) return `~${clamped}s remaining`;
  const mins = Math.round(clamped / 60);
  return `~${mins} min remaining`;
}

/**
 * Full-screen loading experience for a running reconstruction.
 * Signature: the stage timeline — one connected rail whose nodes fill emerald
 * as the backend completes each stage. All progress comes from the real
 * /api/pipeline/status payload; nothing is fabricated: an indeterminate
 * stage shows "Processing…", the overall bar averages the real per-stage
 * fractions, and the ETA appears only once real progress rate data exists.
 */
export default function ProcessingScreen() {
  const { jobId = "" } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  // This screen owns the run the user just started: selecting it here keeps
  // the viewer/analysis/reports pages pointed at THIS run instead of whatever
  // was last selected (e.g. the demo) in localStorage.
  const { selectRun, refresh } = useRuns();
  const missionName =
    (location.state as { missionName?: string } | null)?.missionName ?? `Mission ${jobId.substring(0, 8)}`;

  const [showTechDetails, setShowTechDetails] = useState(false);

  // Polling, phase, ETA, cancellation and retry all belong to the monitor.
  const { status, phase, error, connLost, runningStage, elapsedSec, eta, cancelling, cancel, retry } =
    useRunMonitor(jobId);

  const isCompleted = phase === "completed";
  const isFailed = phase === "failed";
  const isRunning = phase === "running";
  const isQueued = isRunning && status?.status === "queued";
  // This id does not exist: never claim a stage is running for it.
  const missing = phase === "missing";

  // On completion this run becomes the active selection — the completion
  // banner's OPEN 3D MODEL / VIEW ANALYSIS / VIEW REPORT buttons (and the
  // top-level viewer) must render THIS run's artifacts, never a stale or
  // demo selection left over in localStorage. The refresh also pulls the
  // just-finished run into the shared list (TopBar dropdown, Missions table).
  useEffect(() => {
    if (isCompleted && jobId) {
      selectRun(jobId);
      refresh();
    }
  }, [isCompleted, jobId, selectRun, refresh]);

  // Telemetry summary comes from the georef stage detail the poll already
  // carries; while georef is pending there is nothing to show yet.
  const georefDetail = status?.stages?.georef?.detail as
    | (TelemetrySyncView & Record<string, unknown>)
    | undefined;
  const telemetryView: TelemetrySyncView | null =
    georefDetail && (georefDetail.telemetry_mode || georefDetail.sync)
      ? { telemetry_mode: georefDetail.telemetry_mode ?? "VIDEO_ONLY", sync: georefDetail.sync ?? null, note: georefDetail.note ?? null }
      : null;

  const depthBackend = depthBackendId(status?.stages?.depth?.detail);
  // The monitor reports the stage the backend says is running; the label is
  // this page's business, and only a running run shows one.
  const runningStageLabel =
    isRunning && runningStage ? (STAGES.find((s) => s.id === runningStage)?.label ?? runningStage) : null;

  // Stage states come from the backend. Before the first status payload
  // arrives, the first stage shows "running" (the run has been started);
  // stages the backend has not reported yet stay "pending" — never claimed
  // as running ahead of the events.
  const stageState = (id: string, idx: number): string => {
    const st = status?.stages?.[id]?.status;
    if (st) return st;
    return isRunning && idx === 0 ? "running" : "pending";
  };

  const overall = overallProgress(status);
  const progressByStage = stageProgressList(status);
  const progressFor = (id: string): number | null =>
    progressByStage?.find((s) => s.id === id)?.fraction ?? null;
  const connBanner = connLost && isRunning;

  return (
    <div className="proc-wrap">
      {/* Top line */}
      <div className="proc-topline">
        <div>
          <div className="proc-meta">
            <span className={`badge ${missing || isFailed ? "badge-failed" : isRunning ? "badge-processing" : "badge-completed"}`}>
              {missing ? "RUN NOT FOUND" : isRunning ? "PROCESSING LIVE" : (status?.status?.toUpperCase() ?? "PROCESSING")}
            </span>
            <span>RUN {jobId.substring(0, 12)}</span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              <Icon name="clock" size={12} /> {formatClock(elapsedSec)}
            </span>
          </div>
          <h1 className="nm-title" style={{ marginTop: 6 }}>
            {missionName}
          </h1>
          <p className="nm-sub" style={{ marginBottom: 0 }}>
            STRATA end-to-end autonomous pipeline execution
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {isRunning && (
            <button className="btn btn-danger" onClick={() => void cancel()} disabled={cancelling}>
              {cancelling ? "Cancelling…" : "Cancel run"}
            </button>
          )}
          <button className="btn btn-secondary" onClick={() => navigate("/missions")}>
            Back to missions
          </button>
        </div>
      </div>

      {/* Unknown run id: say so instead of polling forever behind a stage
          list that describes nothing. */}
      {missing && (
        <div className="proc-banner fail" role="alert">
          <div className="proc-banner-icon"><Icon name="x" size={26} /></div>
          <div>
            <h2>RUN NOT FOUND</h2>
            <p>
              This server has no run with the ID <strong>{jobId}</strong>. It may have been
              deleted, or the link may point at a different workspace.
            </p>
            <div className="proc-banner-actions">
              <button className="btn btn-secondary" onClick={() => navigate("/missions")}>
                <Icon name="report" size={14} /><span>Back to missions</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Connection banner — temporary poll failure, never a fake failure */}
      {connBanner && (
        <div className="proc-note" role="status">
          <Icon name="info" size={14} />
          <div>Connection temporarily unavailable — retrying…</div>
        </div>
      )}

      {/* Completion banner */}
      {isCompleted && (
        <div className="proc-banner ok">
          <div className="proc-banner-icon"><Icon name="check" size={26} /></div>
          <div>
            <h2>RECONSTRUCTION COMPLETE</h2>
            <p>3D model, camera poses, point cloud, and georeferenced outputs generated successfully.</p>
            <div className="proc-banner-actions">
              <button className="btn btn-primary" onClick={() => navigate("/viewer")}>
                <Icon name="cube" size={14} /><span>OPEN 3D MODEL</span>
              </button>
              <button className="btn btn-secondary" onClick={() => navigate("/analysis")}>
                <Icon name="chart" size={14} /><span>VIEW ANALYSIS</span>
              </button>
              <button className="btn btn-secondary" onClick={() => navigate("/reports")}>
                <Icon name="report" size={14} /><span>VIEW REPORT</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Failure banner — real backend error, no demo fallback */}
      {isFailed && (
        <div className="proc-banner fail">
          <div className="proc-banner-icon"><Icon name="x" size={26} /></div>
          <div>
            <h2>RECONSTRUCTION STOPPED</h2>
            <p style={{ fontWeight: 600 }}>
              Stage: {(() => {
                const failedStage = Object.entries(status?.stages ?? {}).find(
                  ([, s]) => s?.status === "failed",
                )?.[0];
                return failedStage
                  ? (STAGES.find((s) => s.id === failedStage)?.label ?? failedStage)
                  : "unknown stage";
              })()}
            </p>
            <p>{error || status?.error || "An error occurred during pipeline execution."}</p>
            <div className="proc-banner-actions">
              <button className="btn btn-danger" onClick={() => void retry()}>
                <Icon name="refresh" size={14} /><span>Retry Reconstruction</span>
              </button>
              <button className="btn btn-secondary" onClick={() => setShowTechDetails(true)}>
                <Icon name="report" size={14} /><span>View Diagnostics</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Stage timeline — one connected rail replaces the old five-bar strata
          stack and its duplicate card list. The rail fills with the same real
          per-stage fractions the overall bar counts; nothing is invented. */}
      {!missing && (
        <section className="proc-timeline" aria-label="Pipeline stages">
          <div className="proc-tl-head">
            <div>
              <h3 className="section-title" style={{ marginBottom: 0 }}>
                {isCompleted
                  ? "All stages completed"
                  : isRunning
                    ? runningStageLabel
                      ? `Processing: ${runningStageLabel}`
                      : isQueued
                        ? "Queued — waiting for the worker"
                        : "Processing…"
                    : (status?.status ?? "")}
              </h3>
              {isRunning && (
                <div className="proc-tl-sub" aria-live="polite">
                  {eta ? (
                    <>
                      {formatEta(eta.remainingSec)}{" "}
                      <span style={{ color: "var(--gray-400)" }}>· ETA confidence: {eta.confidence}</span>
                    </>
                  ) : (
                    "Estimating time remaining…"
                  )}
                  {" · "}Elapsed: {formatClock(elapsedSec)}
                </div>
              )}
            </div>
            {isRunning && overall !== null && (
              <div className="proc-tl-pct">{Math.round(overall * 100)}% of pipeline complete</div>
            )}
          </div>
          {/* Determinate overall bar: completed stages + live per-stage
              fractions from the backend. Never counts a fraction the backend
              has not reported. */}
          {isRunning && overall !== null && (
            <div
              className="proc-progress proc-tl-overall"
              role="progressbar"
              aria-valuenow={Math.round(overall * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Overall pipeline progress"
            >
              <span className="proc-progress-track">
                <span className="proc-progress-fill" style={{ width: `${overall * 100}%` }} />
              </span>
            </div>
          )}
          <ol className="proc-timeline-list">
            {STAGES.map((st, idx) => {
              const stStatus = stageState(st.id, idx);
              const stageData = status?.stages?.[st.id];
              const fraction = progressFor(st.id);
              const isLast = idx === STAGES.length - 1;
              const stageIcon: IconName =
                stStatus === "completed" || stStatus === "skipped" ? "check"
                : stStatus === "running" ? "gear"
                : stStatus === "failed" ? "x"
                : "clock";
              return (
                <li key={st.id} className={`proc-tl-item ${stStatus}`}>
                  <span className="proc-tl-rail" aria-hidden="true">
                    <span className="proc-tl-node">
                      <Icon name={stageIcon} size={12} />
                    </span>
                    {!isLast && (
                      <span
                        className="proc-tl-link"
                        style={
                          stStatus === "running" && fraction
                            ? ({ "--tl-fill": `${Math.min(100, Math.max(0, fraction * 100))}%` } as CSSProperties)
                            : undefined
                        }
                      />
                    )}
                  </span>
                  <div className="proc-tl-body">
                    <div className="proc-stage-name">{st.label}</div>
                    <div className="proc-stage-engine">
                      {st.id === "depth" && depthBackend
                        ? depthBackendLabel(depthBackend)
                        : st.fallback}
                    </div>
                    {(stStatus === "failed" || stStatus === "completed" || stStatus === "skipped") &&
                      stageData && <StageFailureDetails stageId={st.id} detail={stageData.detail} />}
                  </div>
                  <div className="proc-stage-side">
                    {stStatus === "skipped" && typeof (stageData?.detail as any)?.reason === "string" && (
                      <div>Reason: {(stageData!.detail as any).reason}</div>
                    )}
                    {/* Live per-stage fraction from the backend, when available.
                        Falls back to "Processing…" only when the stage is running but
                        has not reported a numeric fraction yet. */}
                    {stStatus === "running" && typeof stageData?.progress === "number" ? (
                      <span className="proc-progress">
                        <span className="proc-progress-track">
                          <span className="proc-progress-fill" style={{ width: `${Math.min(100, Math.max(0, stageData.progress * 100))}%` }} />
                        </span>
                        <span className="proc-pct">{Math.round(stageData.progress * 100)}%</span>
                      </span>
                    ) : stStatus === "running" ? (
                      <span className="proc-pct">Processing…</span>
                    ) : (
                      stStatus && <span className="proc-pct">{stStatus}</span>
                    )}
                    {stageData && stageData.duration_ms > 0 && <div>Duration: {(stageData.duration_ms / 1000).toFixed(1)}s</div>}
                    {stageData && stageData.count > 0 && <div>{stageData.count.toLocaleString()} outputs</div>}
                  </div>
                </li>
              );
            })}
          </ol>
        </section>
      )}

      {/* Honest telemetry summary (video-only shows GPS — Not available) */}
      {telemetryView && <TelemetryBlock data={telemetryView} />}

      {/* Honest capability note — names the backend this run actually used */}
      {depthBackend && (
        <div className="proc-note">
          <Icon name="info" size={14} />
          <div>
            <strong style={{ color: "var(--silver)" }}>Depth &amp; metric scale.</strong>{" "}
            {depthBackendNote(depthBackend) ?? `Depth backend: ${depthBackend}.`} Metric scale:{" "}
            <strong>ESTIMATED</strong>. Metric validation: <strong>NOT VALIDATED</strong>.
          </div>
        </div>
      )}

      {/* Expandable technical details */}
      <div className="proc-tech">
        <button className="proc-tech-toggle" onClick={() => setShowTechDetails((v) => !v)}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <Icon name="gear" size={14} /> TECHNICAL DETAILS
          </span>
          <span>{showTechDetails ? "▲" : "▼"}</span>
        </button>
        {showTechDetails && (
          <div className="proc-tech-body">
            <div className="proc-tech-grid">
              <div className="proc-tech-cell">
                <span>Run</span>
                <strong>{jobId}</strong>
              </div>
              <div className="proc-tech-cell">
                <span>Poll interval</span>
                <strong>2s — /api/pipeline/status</strong>
              </div>
            </div>
            {status && <pre className="proc-raw">{JSON.stringify(status, null, 2)}</pre>}
          </div>
        )}
      </div>
    </div>
  );
}
