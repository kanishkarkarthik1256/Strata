import { useRuns } from "../hooks/useRuns";
import { Icon } from "../components/ui/Icon";
import { useApi } from "../hooks/useApi";
import { api } from "../lib/api";
import { friendlyMessage } from "../lib/errors";
import { hostCapabilities, metricValidationFromReport, systemStatusRows } from "../lib/capabilities";
import type { CapabilityState, UiCapability } from "../lib/capabilities";

function StateBadge({ state }: { state: CapabilityState }) {
  const cls =
    state === "READY" ? "badge-completed" : state === "UNAVAILABLE" ? "badge-failed" : "badge-warning";
  const label = state === "READY" ? "Ready" : state === "UNAVAILABLE" ? "Unavailable" : "Not validated";
  return <span className={`badge ${cls}`}>{label}</span>;
}

/**
 * The one System Status panel: live host measurements merged with the selected
 * run's dependency record, de-duplicated by the rule in `systemStatusRows`.
 * Each row carries its measured detail (the reason a row is not Ready).
 */
function SystemStatus({ items }: { items: UiCapability[] }) {
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-title" style={{ marginBottom: 12 }}>System Status</div>
      {items.length === 0 ? (
        <p style={{ fontSize: 13, color: "var(--gray-400)" }}>No data available.</p>
      ) : (
        <table>
          <thead><tr><th>Component</th><th>Status</th><th>Details</th></tr></thead>
          <tbody>
            {items.map((c) => (
              <tr key={c.label}>
                <td style={{ fontWeight: 500 }}>{c.label}</td>
                <td><StateBadge state={c.state} /></td>
                <td style={{ fontSize: 12, color: "var(--gray-500)" }}>{c.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function Settings() {
  const { selected } = useRuns();
  const caps = useApi(() => api.getCapabilities(), []);
  const health = useApi(() => api.getHealth(), []);
  // The validation row asserts only what the selected run's report contains;
  // the endpoint returns the honest empty envelope when there is none.
  const validation = useApi(
    () => (selected ? api.getMetricValidation(selected.run_id) : Promise.resolve(null)),
    [selected?.run_id],
  );

  const hostData = caps.state.status === "ready" ? caps.state.data : null;
  const capabilities: UiCapability[] = systemStatusRows(
    hostCapabilities(hostData),
    selected?.dependencies,
    hostData,
    metricValidationFromReport(
      (validation.state.status === "ready" ? validation.state.data : null) as never,
    ),
  );

  return (
    <div style={{ maxWidth: 860, margin: "0 auto" }}>
      <div className="page-header">
        <h2>Settings</h2>
        <p>System configuration and capability status for this machine.</p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-title" style={{ marginBottom: 12 }}>STRATA Desktop Application & Engine</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, fontSize: 13 }}>
          <div>
            <div style={{ color: "var(--gray-400)", fontSize: 11, textTransform: "uppercase" }}>STRATA Version</div>
            {/* No colour: --gray-100 is a surface alias (--stratum-2), which made
                this value invisible on the dark card. Inherit like its siblings. */}
            <div style={{ fontWeight: 600 }}>1.0.0 (Desktop Runtime)</div>
          </div>
          <div>
            <div style={{ color: "var(--gray-400)", fontSize: 11, textTransform: "uppercase" }}>Engine Status</div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600 }}>
              <span
                className="status-dot"
                style={{
                  background:
                    health.state.status === "ready" ? "var(--green-500)"
                    : health.state.status === "loading" ? "var(--gray-300)" : "#ef4444",
                }}
              />
              {health.state.status === "ready" ? "READY (Connected)" : "STARTING / OFF-LINE"}
            </div>
          </div>
          <div>
            <div style={{ color: "var(--gray-400)", fontSize: 11, textTransform: "uppercase" }}>Storage Location (Read-Only)</div>
            <div style={{ fontFamily: "monospace", fontSize: 12 }}>data/storage</div>
          </div>
          <div>
            <div style={{ color: "var(--gray-400)", fontSize: 11, textTransform: "uppercase" }}>Backend Endpoint</div>
            <div style={{ fontFamily: "monospace", fontSize: 12 }}>http://127.0.0.1:8000</div>
          </div>
        </div>
        <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
          <button
            onClick={() => alert("Application Logs Location: data/storage/logs/\nBackend Log: app.log")}
            className="btn btn-secondary"
            style={{ fontSize: 12, padding: "6px 12px" }}
          >
            <Icon name="report" /> Open Logs Location
          </button>
        </div>
      </div>

      <SystemStatus items={capabilities} />

      <p style={{ fontSize: 12, color: "var(--gray-500)", marginBottom: 20 }}>
        Metric 3D Validation stays Not validated until the selected run has a
        report; measurements in the UI are labeled "Estimated Measurement"
        until ground truth exists.
      </p>

      {caps.state.status === "error" && (
        <div className="card" style={{ marginBottom: 20 }}>
          <p style={{ fontSize: 13, color: "#dc2626" }}>Capabilities error: {friendlyMessage(caps.state.error)}</p>
        </div>
      )}

    </div>
  );
}