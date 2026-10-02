import { useRuns } from "../hooks/useRuns";
import { Icon } from "../components/ui/Icon";
import TelemetryBlock, {
  telemetryViewFromManifest,
} from "../components/telemetry/TelemetryBlock";
import { useApi } from "../hooks/useApi";
import { api } from "../lib/api";
import { friendlyMessage } from "../lib/errors";
import { depthBackendLabel } from "../lib/depthBackend";
import type { RunManifest } from "../lib/types";

/** Accuracy report served by /api/reconstruction/metric-validation/{run_id}.
 * The backend 404s when a run has no measurement artifact, which lands in the
 * honest "no report exists" state below — a run that measured nothing is never
 * rendered as though it had. */
interface ValidationReport {
  certification_status?: string;
  certification_reason?: string;
  dataset_split?: string;
  accuracy_summary?: { relative_reconstruction?: string; metric_scale?: string; absolute?: string };
  reference_comparison?: {
    strata_to_reference_m?: Record<string, number>;
    reference_to_strata_coverage_m?: Record<string, number>;
    reference_coverage_of_recon_bbox_pct?: number;
  } | null;
  error_decomposition?: {
    horizontal_median_m?: number;
    vertical_abs_median_m?: number;
    vertical_signed_median_m?: number;
    recon_self_nn_median_m?: number;
  };
  scale_validation?: { scale_error?: number; reconstructed_m?: number; ground_truth_m?: number };
  registration?: { matched_frames?: number; sufficient?: boolean; reason?: string };
  figures?: { error_map?: string; histogram?: string; cdf?: string };
}

/** Ground-truth DSM accuracy from /api/reconstruction/dsm-accuracy/{run_id}.
 * The backend 404s when the run's dataset has no registered reference grid —
 * the panel then renders nothing (never an unmeasured claim). Mirrors the
 * shared DsmAccuracy type (lib/types.ts). */
interface DsmAccuracy {
  status?: string;
  mae_m?: number;
  p95_m?: number;
  height_datum_offset_m?: number;
  coverage?: number;
  reference?: string;
  gate_mae_m?: number;
  measured_offset_m?: [number, number];
  registration_coverage?: number;
}

function accuracyBadge(status?: string): { icon: string; className: string } {
  if (!status) return { icon: "⚠", className: "badge-warning" };
  if (status.startsWith("CERTIFIED")) return { icon: "✓", className: "badge-completed" };
  return { icon: "⚠", className: "badge-warning" };
}

export default function Reports() {
  const { selected } = useRuns();
  const runId = selected?.run_id ?? null;
  // Never request a manifest for a null run id — before the run selector
  // resolves there is nothing to fetch (the old behavior produced a doomed
  // /api/runs/null/manifest request on every mount).
  const manifest = useApi(
    () => (runId ? api.getRunManifest(runId) : Promise.resolve(null)),
    [runId],
  );
  // Formal metric-validation layer. Read through the API, which serves the
  // single accuracy artifact produced by the metric-validation engine; when a
  // run has no measurement artifact the backend returns an honest NOT
  // CERTIFIED envelope in the same shape. Never a fabricated accuracy claim.
  const validation = useApi(
    () =>
      runId
        ? api.getMetricValidation(runId).catch(() => null)
        : Promise.resolve(null),
    [runId],
  );
  const valReport: ValidationReport | null =
    validation.state.status === "ready"
      ? (validation.state.data as unknown as ValidationReport | null)
      : null;
  // Ground-truth DSM accuracy (only runs whose dataset ships a reference
  // grid answer; the 404 is the "nothing to claim here" signal).
  const dsmAccuracy = useApi(
    () =>
      runId
        ? api.getDsmAccuracy(runId).catch(() => null)
        : Promise.resolve(null),
    [runId],
  );
  const dsm: DsmAccuracy | null =
    dsmAccuracy.state.status === "ready"
      ? (dsmAccuracy.state.data as unknown as DsmAccuracy | null)
      : null;
  const data: RunManifest | null =
    manifest.state.status === "ready" ? (manifest.state.data as unknown as RunManifest) : null;

  // The manifest carries the georef telemetry summary (flat sync dict for
  // external runs, mode-only stub otherwise).
  const telemetryView = telemetryViewFromManifest(data?.telemetry);

  // The depth backend this run actually executed (recorded by the pipeline
  // in the depth stage detail) — never a hardcoded model name.
  const depthDetail = data?.stages?.depth?.detail as
    | { backend?: unknown; checkpoint?: unknown }
    | undefined;
  const depthCheckpoint = depthDetail?.checkpoint;
  const depthModelLabel = depthBackendLabel(depthDetail?.backend);

  const downloadJson = () => {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${data.run_id ?? "run"}-manifest.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ maxWidth: 960, margin: "0 auto" }}>
      <div className="page-header">
        <div>
          <h2>Mission Report</h2>
          <p>
            Run: <span style={{ fontFamily: "monospace" }}>{runId ?? "none selected"}</span>
          </p>
        </div>
        {data && (
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-secondary" onClick={downloadJson}>
              Export Data (JSON)
            </button>
          </div>
        )}
      </div>

      {!runId ? (
        <div className="empty-state">
          <div className="empty-state-icon"><Icon name="report" size={44} /></div>
          <p>Select a run to view its report.</p>
        </div>
      ) : manifest.state.status === "loading" ? (
        <p style={{ fontSize: 13, color: "var(--gray-400)" }}>Loading report…</p>
      ) : manifest.state.status === "error" ? (
        <div className="card">
          <p style={{ fontSize: 13, color: "#dc2626" }}>{friendlyMessage(manifest.state.error)}</p>
        </div>
      ) : data ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <section className="card">
            <h3 className="card-title">Accuracy</h3>
            {valReport ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <table>
                  <tbody>
                    <tr>
                      <td style={{ fontWeight: 500 }}>Relative reconstruction quality</td>
                      <td>
                        {valReport.accuracy_summary?.relative_reconstruction === "measured"
                          ? "Available (reprojection / cross-view / dense→mesh diagnostics)"
                          : "Unavailable"}
                      </td>
                    </tr>
                    <tr>
                      <td style={{ fontWeight: 500 }}>Metric scale</td>
                      <td>{valReport.accuracy_summary?.metric_scale ?? "not validated"}</td>
                    </tr>
                    <tr>
                      <td style={{ fontWeight: 500 }}>Absolute spatial accuracy</td>
                      <td>
                        <span className={`badge ${accuracyBadge(valReport.certification_status).className}`}>
                          {accuracyBadge(valReport.certification_status).icon} {valReport.certification_status ?? "NOT CERTIFIED"}
                        </span>
                      </td>
                    </tr>
                    {valReport.certification_reason && (
                      <tr>
                        <td style={{ fontWeight: 500 }}>Reason</td>
                        <td style={{ fontSize: 13, color: "var(--gray-600)" }}>{valReport.certification_reason}</td>
                      </tr>
                    )}
                    {valReport.dataset_split && (
                      <tr>
                        <td style={{ fontWeight: 500 }}>Validation split</td>
                        <td style={{ fontSize: 13, color: "var(--gray-600)" }}>{valReport.dataset_split}</td>
                      </tr>
                    )}
                  </tbody>
                </table>
                {valReport.reference_comparison?.strata_to_reference_m && (
                  <table>
                    <thead>
                      <tr>
                        <th colSpan={2}>vs independent LiDAR (STRATA → reference)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {([
                        ["Median", valReport.reference_comparison.strata_to_reference_m.median],
                        ["RMSE", valReport.reference_comparison.strata_to_reference_m.rmse],
                        ["P95", valReport.reference_comparison.strata_to_reference_m.p95],
                        ["Within 1 m", valReport.reference_comparison.strata_to_reference_m["within_1.0m_pct"]],
                      ] as [string, number | undefined][]).map(([label, v]) =>
                        v != null ? (
                          <tr key={label}>
                            <td style={{ fontWeight: 500 }}>{label}</td>
                            <td>{typeof v === "number" ? `${v.toFixed(v < 10 ? 2 : 1)}${label === "Within 1 m" ? "%" : " m"}` : v}</td>
                          </tr>
                        ) : null,
                      )}
                    </tbody>
                  </table>
                )}
                {valReport.error_decomposition && (
                  <p style={{ fontSize: 12, color: "var(--gray-500)" }}>
                    Error decomposition: horizontal median {valReport.error_decomposition.horizontal_median_m ?? "—"} m ·
                    vertical |Δz| median {valReport.error_decomposition.vertical_abs_median_m ?? "—"} m ·
                    reconstruction self-spacing {valReport.error_decomposition.recon_self_nn_median_m ?? "—"} m.
                    Relative quality is never evidence of absolute accuracy.
                  </p>
                )}
                {valReport.figures?.error_map && runId && (
                  <div>
                    <a
                      href={api.artifactUrl(runId, `validation/${valReport.figures.error_map}`)}
                      target="_blank"
                      rel="noreferrer"
                      style={{ fontSize: 13 }}
                    >
                      View spatial error map →
                    </a>
                  </div>
                )}
              </div>
            ) : (
              <table>
                <tbody>
                  <tr>
                    <td style={{ fontWeight: 500 }}>Absolute spatial accuracy</td>
                    <td>
                      <span className="badge badge-warning">⚠ NOT CERTIFIED — NO INDEPENDENT REFERENCE</span>
                    </td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 500 }}>Reason</td>
                    <td style={{ fontSize: 13, color: "var(--gray-600)" }}>
                      No metric-validation report exists for this run (no independent reference evaluated).
                    </td>
                  </tr>
                </tbody>
              </table>
            )}
            {dsm && dsm.status === "ok" && (
              <table>
                <thead>
                  <tr>
                    <th colSpan={2}>vs reference DSM (ground truth, height-shape)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ fontWeight: 500 }}>Height MAE</td>
                    <td>{dsm.mae_m?.toFixed(2)} m</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 500 }}>Height error p95</td>
                    <td>{dsm.p95_m?.toFixed(2)} m</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 500 }}>Coverage of reference grid</td>
                    <td>{((dsm.coverage ?? 0) * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 500 }}>Height datum offset</td>
                    <td style={{ fontSize: 13, color: "var(--gray-600)" }}>
                      {dsm.height_datum_offset_m?.toFixed(1)} m (convention, removed)
                    </td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 500 }}>Horizontal placement</td>
                    <td style={{ fontSize: 13, color: "var(--gray-600)" }}>
                      measured (co-registered): offset {dsm.measured_offset_m?.[0] ?? 0} m E, {dsm.measured_offset_m?.[1] ?? 0} m N
                    </td>
                  </tr>
                </tbody>
              </table>
            )}
            {dsm && dsm.status === "alignment_gate_failed" && (
              <table>
                <thead>
                  <tr>
                    <th colSpan={2}>vs reference DSM — no score available</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ fontWeight: 500 }}>Status</td>
                    <td>
                      <span className="badge badge-warning">⚠ NOT REGISTRABLE</span>
                    </td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 500 }}>Reason</td>
                    <td style={{ fontSize: 13, color: "var(--gray-600)" }}>
                      The reconstruction could not be matched to the reference grid better than
                      decorrelated terrain (measured MAE {dsm.mae_m?.toFixed(2)} m vs null {dsm.gate_mae_m?.toFixed(2)} m) —
                      serving a score here would manufacture accuracy.
                    </td>
                  </tr>
                </tbody>
              </table>
            )}
          </section>

          <section className="card">
            <h3 className="card-title">Overview</h3>
            <table>
              <tbody>
                <tr><td style={{ fontWeight: 500 }}>Dataset / Mission</td><td>{data.dataset ?? "—"} / {data.mission ?? "—"}</td></tr>
                <tr><td style={{ fontWeight: 500 }}>Status</td><td><span className={`badge ${data.status === "PASS" ? "badge-completed" : "badge-warning"}`}>{data.status ?? "?"}</span></td></tr>
                <tr><td style={{ fontWeight: 500 }}>Pipeline version</td><td>{data.pipeline_version ?? "—"}</td></tr>
                <tr><td style={{ fontWeight: 500 }}>Started</td><td>{data.started_at ? new Date(data.started_at).toLocaleString() : "—"}</td></tr>
                <tr><td style={{ fontWeight: 500 }}>Completed</td><td>{data.completed_at ? new Date(data.completed_at).toLocaleString() : "—"}</td></tr>
                {data.git_commit ? <tr><td style={{ fontWeight: 500 }}>Git commit</td><td style={{ fontFamily: "monospace", fontSize: 12 }}>{data.git_commit}</td></tr> : null}
              </tbody>
            </table>
          </section>

          {telemetryView && <TelemetryBlock data={telemetryView} />}

          <section className="card">
            <h3 className="card-title">Pipeline Stages</h3>
            <table>
              <thead><tr><th>Stage</th><th>Status</th></tr></thead>
              <tbody>
                {Object.entries(data.stages ?? {}).map(([name, st]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>
                      <span className={`badge ${st?.status === "PASS" ? "badge-completed" : st?.status === "BLOCKED" ? "badge-failed" : "badge-warning"}`}>
                        {st?.status ?? "?"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {data.metrics && Object.keys(data.metrics).length > 0 && (
            <section className="card">
              <h3 className="card-title">Reconstruction Statistics</h3>
              <div className="mission-grid">
                {selected?.dense_points != null && (
                  <div className="mission-card"><div style={{ fontWeight: 600 }}>Dense points</div><div style={{ fontSize: 13, color: "var(--gray-500)" }}>{selected.dense_points.toLocaleString()}</div></div>
                )}
                {selected?.sparse_points != null && (
                  <div className="mission-card"><div style={{ fontWeight: 600 }}>Sparse points</div><div style={{ fontSize: 13, color: "var(--gray-500)" }}>{selected.sparse_points.toLocaleString()}</div></div>
                )}
                {selected?.cameras != null && (
                  <div className="mission-card"><div style={{ fontWeight: 600 }}>Registered cameras</div><div style={{ fontSize: 13, color: "var(--gray-500)" }}>{selected.cameras}</div></div>
                )}
                {selected?.gps_points != null && (
                  <div className="mission-card"><div style={{ fontWeight: 600 }}>GPS fixes</div><div style={{ fontSize: 13, color: "var(--gray-500)" }}>{selected.gps_points}</div></div>
                )}
                {selected?.mean_reprojection_error_px != null && (
                  <div className="mission-card"><div style={{ fontWeight: 600 }}>Mean reprojection error</div><div style={{ fontSize: 13, color: "var(--gray-500)" }}>{selected.mean_reprojection_error_px.toFixed(2)} px</div></div>
                )}
              </div>
            </section>
          )}

          <section className="card">
            <h3 className="card-title">Depth Backend</h3>
            <table>
              <tbody>
                {/* Read from the manifest's depth stage detail — the record of
                    what actually ran. Absent data reports "unknown"; a stereo
                    run must never claim Depth Anything. */}
                <tr><td style={{ fontWeight: 500 }}>Depth backend</td><td>{depthModelLabel}</td></tr>
                <tr><td style={{ fontWeight: 500 }}>Checkpoint</td><td>{typeof depthCheckpoint === "string" && depthCheckpoint ? depthCheckpoint : "unknown"}</td></tr>
              </tbody>
            </table>
          </section>

          {data.limitations && data.limitations.length > 0 && (
            <section className="card">
              <h3 className="card-title">Limitations</h3>
              <ul style={{ fontSize: 13, color: "var(--gray-600)", paddingLeft: 16, lineHeight: 1.8 }}>
                {data.limitations && data.limitations.length > 0 && (
                  <ul style={{ fontSize: 13, color: "var(--gray-600)", paddingLeft: 16, lineHeight: 1.8 }}>
                    {data.limitations.map((l: string, i: number) => <li key={i}>{l}</li>)}
                  </ul>
                )}
              </ul>
            </section>
          )}
        </div>
      ) : null}
    </div>
  );
}