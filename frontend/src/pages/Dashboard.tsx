import { useNavigate } from "react-router-dom";
import { useRuns } from "../hooks/useRuns";
import { Icon } from "../components/ui/Icon";

// System Status lives in Settings (it is machine configuration, not the
// mission overview). The Dashboard reads runs only.
export default function Dashboard() {
  const navigate = useNavigate();
  const { runs, loading, error, selected, selectRun } = useRuns();

  const total = runs.length;
  const completed = runs.filter((r) => r.status === "PASS" || r.status === "completed").length;
  const partial = runs.filter((r) => r.status && r.status !== "PASS" && r.status !== "completed").length;

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Welcome to STRATA</h2>
          <p>AI-enabled Drone Video Ingestion & 3D Model Generation Platform</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate("/new-mission")}>
          <Icon name="play" /> New mission
        </button>
      </div>

      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-icon green"><Icon name="missions" size={20} /></div>
          <div>
            <div className="stat-value">{total}</div>
            <div className="stat-label">Total Missions</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon blue"><Icon name="check" size={20} /></div>
          <div>
            <div className="stat-value">{completed}</div>
            <div className="stat-label">Completed</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon amber"><Icon name="clock" size={20} /></div>
          <div>
            <div className="stat-value">{partial}</div>
            <div className="stat-label">Partial / Processing</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon purple"><Icon name="cube" size={20} /></div>
          <div>
            <div className="stat-value">
              {selected?.dense_points != null
                ? `${(selected.dense_points / 1_000_000).toFixed(2)}M`
                : "—"}
            </div>
            <div className="stat-label">Dense Points (selected)</div>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 24 }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Recent Missions & Runs</span>
            <button className="btn btn-secondary" onClick={() => navigate("/missions")}>
              View All →
            </button>
          </div>
          {loading ? (
            <p style={{ fontSize: 13, color: "var(--gray-400)" }}>Loading…</p>
          ) : error ? (
            <p style={{ fontSize: 13, color: "#dc2626" }}>{error}</p>
          ) : runs.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon"><Icon name="folder" size={44} /></div>
              <p>No missions found. Click + NEW MISSION to upload drone footage.</p>
            </div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Mission / Run</th>
                  <th>Status</th>
                  <th>Points</th>
                  <th>Cameras</th>
                  <th>GPS</th>
                </tr>
              </thead>
              <tbody>
                {runs.slice(0, 6).map((r) => (
                  <tr
                    key={r.run_id}
                    style={{ cursor: "pointer", background: r.run_id === selected?.run_id ? "var(--green-50)" : undefined }}
                    onClick={() => {
                      selectRun(r.run_id);
                      navigate("/viewer");
                    }}
                  >
                    <td style={{ fontWeight: 500 }}>
                      {r.dataset ?? r.mission ?? "Uploaded mission"}
                      <div className="row-sub mono">{r.run_id}</div>
                    </td>
                    <td>
                      <span className={`badge ${r.status === "PASS" || r.status === "completed" ? "badge-completed" : r.status === "PARTIAL" ? "badge-warning" : "badge-failed"}`}>
                        {r.status ?? "unknown"}
                      </span>
                    </td>
                    <td>{r.dense_points != null ? r.dense_points.toLocaleString() : "—"}</td>
                    <td>{r.cameras ?? "—"}</td>
                    <td>{r.gps_points ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card">
            <div className="card-title" style={{ marginBottom: 8 }}>
              Quick Actions
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button
                className="btn btn-primary"
                style={{ width: "100%", justifyContent: "center" }}
                onClick={() => navigate("/new-mission")}
              >
                New mission
              </button>
              <button
                className="btn btn-secondary"
                style={{ width: "100%", justifyContent: "center" }}
                onClick={() => navigate("/missions")}
              >
                Browse missions
              </button>
              <button
                className="btn btn-secondary"
                style={{ width: "100%", justifyContent: "center" }}
                onClick={() => navigate("/viewer")}
              >
                Open viewer
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}