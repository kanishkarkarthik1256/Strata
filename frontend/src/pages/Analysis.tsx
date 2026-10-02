import { useMemo } from "react";
import { MapContainer, TileLayer, Polygon, Polyline, Marker, Tooltip } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useRuns } from "../hooks/useRuns";
import { useApi } from "../hooks/useApi";
import { api } from "../lib/api";
import { friendlyMessage } from "../lib/errors";
import { Icon } from "../components/ui/Icon";
import type { RunAnalysis } from "../lib/types";

function fmt(n: number | null | undefined, digits = 2): string {
  return n == null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ padding: 20 }}>
      <div className="section-title">{title}</div>
      {children}
    </div>
  );
}

function Row({ label, value, title }: { label: string; value: React.ReactNode; title?: string }) {
  return (
    <div className="sys-row" title={title}>
      <span className="sys-label">{label}</span>
      <span className="sys-value">{value}</span>
    </div>
  );
}

function ErrorHeatmap({ map }: { map: NonNullable<RunAnalysis["accuracy_map"]> }) {
  if (!map.available || map.cells.length === 0) {
    return <p style={{ fontSize: 13, color: "var(--gray-400)" }}>No sparse↔dense error grid for this run.</p>;
  }
  const medians = map.cells.map((c) => c[2]);
  const lo = Math.min(...medians);
  const hi = Math.max(...medians);
  const color = (m: number) => {
    const t = hi > lo ? (m - lo) / (hi - lo) : 0;
    // green (good) → yellow → red (worst measured cell)
    const stops = [
      [34, 197, 94],
      [234, 179, 8],
      [239, 68, 68],
    ] as const;
    const idx = Math.min(Math.floor(t * 2), 1);
    const f = t * 2 - idx;
    const [r, g, b] = stops[idx].map((c, i) => Math.round(c + (stops[idx + 1][i] - c) * f));
    return `rgb(${r},${g},${b})`;
  };
  const { cell_size_m } = map;
  return (
    <div>
      <svg
        viewBox={`0 0 ${Math.max(...map.cells.map((c) => c[0])) + 2} ${Math.max(...map.cells.map((c) => c[1])) + 2}`}
        style={{ width: "100%", background: "var(--obsidian)", borderRadius: 8 }}
        role="img"
        aria-label="Per-cell sparse-to-dense error heat map"
      >
        {map.cells.map(([ix, iy, med], i) => (
          <rect key={i} x={ix} y={iy} width={0.95} height={0.95} fill={color(med)} opacity={0.9}>
            <title>{`cell (${ix}, ${iy}) · median ${med.toFixed(2)} m`}</title>
          </rect>
        ))}
      </svg>
      <div className="heatmap-labels" style={{ marginTop: 6 }}>
        <span>{`best ${lo.toFixed(2)} m`}</span>
        <span>{`${map.cells.length} cells ≥ ${map.min_points_per_cell} pts · ${cell_size_m} m`}</span>
        <span>{`worst ${hi.toFixed(2)} m`}</span>
      </div>
    </div>
  );
}

function SiteMap({ analysis }: { analysis: RunAnalysis }) {
  const outline = analysis.site?.outline_wgs84 ?? null;
  const track = analysis.flight?.track_wgs84 ?? null;
  const anchor = analysis.frame.anchor_wgs84;

  // WGS84 positions [lat, lon] for leaflet; outline is [lat, lon] already.
  const outlineLatLngs = outline ?? null;
  const trackLatLngs = track ?? null;
  const center: [number, number] | null =
    analysis.site?.centroid?.wgs84
      ? [analysis.site.centroid.wgs84.lat, analysis.site.centroid.wgs84.lon]
      : anchor
        ? [anchor.lat, anchor.lon]
        : null;
  if (!center) {
    return <p style={{ fontSize: 13, color: "var(--gray-400)" }}>No georeferenced anchor — map unavailable.</p>;
  }
  const startIcon = L.divIcon({ className: "", html: '<div style="width:10px;height:10px;border-radius:50%;background:#22c55e;border:2px solid white"></div>', iconSize: [10, 10] });
  return (
    <div style={{ height: 360, borderRadius: 8, overflow: "hidden" }}>
      <MapContainer center={center} zoom={15} style={{ height: "100%", width: "100%" }} scrollWheelZoom>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {outlineLatLngs && outlineLatLngs.length > 2 && (
          <Polygon positions={outlineLatLngs} pathOptions={{ color: "#34d399", weight: 2, fillOpacity: 0.15 }}>
            <Tooltip sticky>Modelled outline (convex hull of occupied cells)</Tooltip>
          </Polygon>
        )}
        {trackLatLngs && trackLatLngs.length > 1 && (
          <Polyline positions={trackLatLngs} pathOptions={{ color: "#60a5fa", weight: 3, opacity: 0.9 }}>
            <Tooltip sticky>Flight track ({trackLatLngs.length} GPS fixes)</Tooltip>
          </Polyline>
        )}
        {trackLatLngs && trackLatLngs.length > 0 && (
          <Marker position={trackLatLngs[0]} icon={startIcon}>
            <Tooltip>Flight start</Tooltip>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
}

export default function Analysis() {
  const { selected } = useRuns();
  const runId = selected?.run_id ?? null;
  const { state } = useApi(() => (runId ? api.getRunAnalysis(runId) : Promise.resolve(null)), [runId]);

  const a = state.status === "ready" ? state.data : null;

  const relief = a?.elevation;
  const site = a?.site;
  const flight = a?.flight;
  const validation = a?.metric_validation ?? null;

  const extremes = useMemo(() => {
    if (!relief?.lowest || !relief?.highest) return null;
    return { low: relief.lowest, high: relief.highest };
  }, [relief]);

  if (!runId) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon"><Icon name="chart" size={44} /></div>
        <p>Select a run to analyse.</p>
      </div>
    );
  }
  if (state.status === "loading") {
    return <p style={{ fontSize: 13, color: "var(--gray-400)" }}>Loading analysis…</p>;
  }
  if (state.status === "error") {
    return <p style={{ fontSize: 13, color: "#dc2626" }}>{friendlyMessage(state.error)}</p>;
  }
  if (!a || !a.available) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon"><Icon name="chart" size={44} /></div>
        <p>This run has no analysable reconstruction yet (no dense surface artifact).</p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h2 style={{ margin: 0, color: "var(--white)" }}>Analysis</h2>
        <p style={{ fontSize: 13, color: "var(--gray-400)", margin: "4px 0 0" }}>
          {runId} · computed from {a.sources.surface} ({fmt(a.sources.surface_points, 0)} points)
          {a.notes.length > 0 ? ` · ${a.notes.join(" · ")}` : ""}
        </p>
      </div>

      {site && (
        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-icon green"><Icon name="pin" size={20} /></div>
            <div>
              <div className="stat-value">{site.area_km2 >= 1 ? `${fmt(site.area_km2)} km²` : `${fmt(site.area_m2, 0)} m²`}</div>
              <div className="stat-label">Modelled area ({fmt(site.area_ha)} ha)</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon purple"><Icon name="cube" size={20} /></div>
            <div>
              <div className="stat-value">{fmt(site.outline_area_km2)} km²</div>
              <div className="stat-label">Site covered (incl. gaps)</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon blue"><Icon name="chart" size={20} /></div>
            <div>
              <div className="stat-value">{relief ? `${fmt(relief.robust_relief)} m` : "—"}</div>
              <div className="stat-label">Robust relief (P1→P99)</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon amber"><Icon name="dashboard" size={20} /></div>
            <div>
              <div className="stat-value">{a.density ? `${fmt(a.density.points_per_m2, 1)} /m²` : "—"}</div>
              <div className="stat-label">Point density</div>
            </div>
          </div>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, alignItems: "start" }}>
        {site && (
          <Section title="Represented area">
            <Row label="Modelled (occupied cells)" value={`${fmt(site.area_m2, 0)} m² · ${fmt(site.area_km2)} km²`} title={site.area_method} />
            <Row label="Site outline (convex hull)" value={`${fmt(site.outline_area_m2, 0)} m² · ${fmt(site.outline_area_km2)} km²`} title={site.outline_method} />
            <Row label="Grid" value={`${site.cell_size_m} m cells · ${fmt(site.occupied_cells, 0)} occupied`} />
            <Row label="Extent (E×N)" value={`${fmt(site.bbox.size[0], 1)} × ${fmt(site.bbox.size[1], 1)} m`} />
            <Row label="Frame" value={`${a.frame.kind}${a.frame.metric ? " (metric)" : ""} · scale ${fmt(a.frame.alignment_scale, 5)}`} />
          </Section>
        )}

        {extremes && relief && (
          <Section title="Highest & lowest points">
            <Row
              label="Highest"
              value={`${fmt(relief.max)} m ${relief.datum}`}
              title={extremes.high.wgs84 ? `WGS84 ${extremes.high.wgs84.lat.toFixed(6)}, ${extremes.high.wgs84.lon.toFixed(6)}` : undefined}
            />
            <Row label="Lowest" value={`${fmt(relief.min)} m ${relief.datum}`} />
            <Row label="Total relief" value={`${fmt(relief.relief)} m`} />
            <Row label="Robust (P1→P99)" value={`${fmt(relief.robust_relief)} m`} title="Excludes outlier specks" />
            <Row label="Mean / median" value={`${fmt(relief.mean)} / ${fmt(relief.median)} m`} />
          </Section>
        )}
      </div>

      {(site?.outline_wgs84 || flight?.track_wgs84) && (
        <Section title="Map — modelled area & flight track">
          <SiteMap analysis={a} />
          <p style={{ fontSize: 12, color: "var(--gray-500)", margin: "8px 0 0" }}>
            Green outline: the area captured as a 3D model (convex hull of modelled ground).
            Blue line: the drone's GPS flight track.
          </p>
        </Section>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, alignItems: "start" }}>
        {flight && (
          <Section title="Flight">
            {flight.available ? (
              <>
                <Row label="GPS fixes" value={fmt(flight.points, 0)} />
                <Row label="Path length" value={`${fmt(flight.path_length_m, 0)} m`} />
                <Row label="Speed (mean/max)" value={`${fmt(flight.mean_speed_m_s, 1)} / ${fmt(flight.max_speed_m_s, 1)} m/s`} />
                <Row label="Altitude AGL (min–max)" value={`${fmt(flight.agl_min_m, 0)}–${fmt(flight.agl_max_m, 0)} m`} />
                <Row label="Telemetry drift" value={`${fmt(flight.drift_m, 1)} m`} />
                <Row label="GPS quality" value={`${flight.grade ?? "—"} (${fmt(flight.gps_score, 1)})`} />
                {flight.gsd_m_per_px != null && <Row label="GSD" value={`≈ ${fmt(flight.gsd_m_per_px, 2)} m/px`} />}
              </>
            ) : (
              <p style={{ fontSize: 13, color: "var(--gray-400)" }}>No telemetry for this run.</p>
            )}
          </Section>
        )}

        {a.consistency && (
          <Section title="Internal consistency">
            <Row label={a.consistency.measure} value="" />
            <Row label="Median / P95" value={`${fmt(a.consistency.median_m)} / ${fmt(a.consistency.p95_m)} m`} />
            <Row label="Within 3 m" value={`${fmt(a.consistency.within_3m_pct, 1)}%`} />
            <Row label="Correspondences" value={fmt(a.consistency.correspondences, 0)} />
            {a.density && <Row label="Coverage / occlusion" value={`${fmt(a.density.coverage_percent, 1)}% / ${fmt(a.density.occlusion_percent, 1)}%`} />}
          </Section>
        )}
      </div>

      {a.accuracy_map && (
        <Section title="Where the model is least reliable (sparse↔dense error grid)">
          <ErrorHeatmap map={a.accuracy_map} />
        </Section>
      )}

      <Section title="Metric validation">
        {validation && validation.internal_validation ? (
          <>
            <Row label="Status" value={validation.certification_status} />
            {validation.internal_validation.checks.map((c) => (
              <Row
                key={c.name}
                label={c.name}
                value={
                  c.measured ? (
                    <span style={{ color: c.pass ? "var(--emerald-bright)" : "#fca5a5" }}>
                      {c.value != null ? `${fmt(c.value, 3)} ${c.unit}` : "—"} · {c.pass ? "pass" : "FAIL"}
                    </span>
                  ) : (
                    <span style={{ color: "var(--gray-500)" }}>not measured</span>
                  )
                }
                title={c.note ?? c.criterion}
              />
            ))}
            <p style={{ fontSize: 12, color: "var(--gray-500)", margin: "8px 0 0" }}>{validation.internal_validation.label}</p>
          </>
        ) : (
          <p style={{ fontSize: 13, color: "var(--gray-400)" }}>
            No metric-validation report exists for this run.
          </p>
        )}
      </Section>
    </div>
  );
}
