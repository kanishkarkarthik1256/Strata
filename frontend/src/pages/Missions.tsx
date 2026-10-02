import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useRuns } from "../hooks/useRuns";
import { useApi } from "../hooks/useApi";
import { api } from "../lib/api";
import { friendlyMessage } from "../lib/errors";
import type { Mission } from "../lib/types";
import { Icon } from "../components/ui/Icon";

export default function Missions() {
  const navigate = useNavigate();
  const { runs, selected, selectRun, loading, error, refresh } = useRuns();
  const [tab, setTab] = useState<"runs" | "missions">("runs");
  const [filter, setFilter] = useState<string>("ALL");
  const [search, setSearch] = useState("");

  // The app only renders this page with a live session, so the registered
  // missions list can be fetched directly (no second sign-in gate here).
  const missions = useApi<Mission[]>(
    () => api.getMissions().then((res) => res.missions ?? []),
    [],
  );

  const filteredRuns = runs.filter((r) => {
    if (filter === "ALL") return true;
    if (filter === "COMPLETED") return r.status === "PASS" || r.status === "completed";
    if (filter === "PROCESSING") return r.status === "running" || r.status === "queued";
    if (filter === "FAILED") return r.status === "failed";
    return true;
  });

  const q = search.trim().toLowerCase();
  const searchedRuns = q
    ? filteredRuns.filter((r) =>
        [r.run_id, r.dataset, r.mission].some((f) => (f ?? "").toLowerCase().includes(q)),
      )
    : filteredRuns;

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>STRATA Missions & Runs</h2>
          <p>Manage aerial photogrammetry missions and process drone footage</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate("/new-mission")}>
          <Icon name="play" /> New mission
        </button>
      </div>

      <div className="tabs">
        <div className="flex gap-2">
          <button className={`tab ${tab === "runs" ? "active" : ""}`} onClick={() => setTab("runs")}>
            Mission Runs ({runs.length})
          </button>
          <button className={`tab ${tab === "missions" ? "active" : ""}`} onClick={() => setTab("missions")}>
            Registered Missions ({missions.state.status === "ready" ? missions.state.data.length : "—"})
          </button>
        </div>

        {tab === "runs" && (
          <div className="flex items-center gap-3" style={{ flexWrap: "wrap" }}>
            <input
              type="search"
              placeholder="Search run ID, dataset, mission…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label="Search missions"
              style={{
                padding: "6px 12px",
                fontSize: 13,
                border: "1px solid var(--border)",
                borderRadius: 8,
                minWidth: 220,
              }}
            />
            <div className="tabs" style={{ margin: 0, borderBottom: "none" }}>
              {["ALL", "COMPLETED", "PROCESSING", "FAILED"].map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`tab ${filter === f ? "active" : ""}`}
                >
                  {f}
                </button>
              ))
              }
            </div>
          </div>
        )}
      </div>

      {tab === "runs" ? (
        loading ? (
          <p style={{ fontSize: 13, color: "var(--gray-400)" }}>Loading…</p>
        ) : error ? (
          <p style={{ fontSize: 13, color: "#dc2626" }}>{error}</p>
        ) : searchedRuns.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon"><Icon name="folder" size={44} /></div>
            <p>
              No runs found
              {q ? ` matching "${search}"` : ` matching filter '${filter}'`}. Upload drone
              footage to start a new mission.
            </p>
          </div>
        ) : (
          <div className="card">
            <table>
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Dataset</th>
                  <th>Mission</th>
                  <th>Status</th>
                  <th>Dense Points</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {searchedRuns.map((r) => (
                  <tr
                    key={r.run_id}
                    style={{
                      cursor: "pointer",
                      background: r.run_id === selected?.run_id ? "var(--green-50)" : undefined,
                    }}
                    onClick={() => {
                      selectRun(r.run_id);
                      navigate("/viewer");
                    }}
                  >
                    <td className="mono" style={{ fontSize: 12 }}>{r.run_id}</td>
                    <td>{r.dataset ?? "—"}</td>
                    <td>{r.mission ?? "—"}</td>
                    <td>
                      <span className={`badge ${r.status === "PASS" || r.status === "completed" ? "badge-completed" : r.status === "PARTIAL" ? "badge-warning" : "badge-failed"}`}>
                        {r.status ?? "?"}
                      </span>
                    </td>
                    <td>{r.dense_points != null ? r.dense_points.toLocaleString() : "—"}</td>
                    <td>
                      <button
                        className="btn btn-secondary"
                        style={{ fontSize: 11, padding: "4px 8px" }}
                        onClick={(e) => {
                          e.stopPropagation();
                          selectRun(r.run_id);
                          navigate("/viewer");
                        }}
                      >
                        Open 3D Model
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button className="btn btn-secondary" style={{ marginTop: 12 }} onClick={refresh}>
              Refresh
            </button>
          </div>
        )
      ) : missions.state.status === "loading" ? (
        <p style={{ fontSize: 13, color: "var(--gray-400)" }}>Loading missions…</p>
      ) : missions.state.status === "error" ? (
        <div className="card">
          <p style={{ fontSize: 13, color: "#dc2626" }}>{friendlyMessage(missions.state.error)}</p>
        </div>
      ) : missions.state.data.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon"><Icon name="folder" size={44} /></div>
          {/* Honest empty state: the count is real (the API answers with the
              list), but a Mission record is only written when a run starts
              from an account whose role may mutate missions — reconstruction
              itself does not need one, so runs can exist with no mission at
              all. "Create a mission around an uploaded project" pointed at a
              screen the app does not have. */}
          <p>
            No missions are registered. A mission record is written when a run is started from an
            account with the operator role; reconstruction itself does not require one, so runs
            can exist here without a mission.
          </p>
        </div>
      ) : (
        <div className="mission-grid">
          {missions.state.data.map((m) => (
            <div key={m.id} className="mission-card">
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontWeight: 600 }}>{m.name}</span>
                <span className={`badge ${m.status === "COMPLETED" ? "badge-completed" : "badge-warning"}`}>{m.status}</span>
              </div>
              <div style={{ fontSize: 13, color: "var(--gray-500)" }}>
                Project: {m.project_id ?? "—"} · Priority {m.priority}
              </div>
              <div style={{ fontSize: 12, color: "var(--gray-400)", marginTop: 4 }}>
                Created {m.created_at ? new Date(m.created_at).toLocaleString() : "—"}
                {m.current_stage ? ` · Stage: ${m.current_stage}` : ""}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}