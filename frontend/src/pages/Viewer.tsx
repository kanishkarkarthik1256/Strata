import { useState } from "react";
import { Icon, type IconName } from "../components/ui/Icon";
import { useRuns } from "../hooks/useRuns";
import { useViewerArtifacts } from "../hooks/useViewerArtifacts";
import { useApi } from "../hooks/useApi";
import { api } from "../lib/api";
import ViewerCanvas, { type ViewCommand } from "../components/viewer/ViewerCanvas";
import ViewerToolbar from "../components/viewer/ViewerToolbar";
import ViewerLayers from "../components/viewer/ViewerLayers";
import ViewerStats from "../components/viewer/ViewerStats";
import ConfidenceLegend from "../components/viewer/ConfidenceLegend";
import GroundLevelBlock from "../components/viewer/GroundLevelBlock";
import type { ViewMode } from "../components/viewer/ViewerToolbar";

interface ValidationLike {
  certification_status?: string;
  certification_reason?: string;
  capability_state?: string;
  internal_validation?: {
    label?: string;
    verified?: boolean;
    checks?: { name: string; measured: boolean; pass: boolean | null; value: number | null; unit: string }[];
  } | null;
}

/** Metric Scale/Validation block — asserts only what the run's validation
 *  report (or its honest absence envelope) actually contains. */
function MetricValidationBlock({ runId }: { runId: string | null }) {
  const { state } = useApi(
    () => (runId ? api.getMetricValidation(runId) : Promise.resolve(null)),
    [runId],
  );
  if (state.status === "loading") {
    return <div style={{ fontSize: 12, color: "var(--gray-500)" }}>Checking validation…</div>;
  }
  if (state.status === "error") {
    return (
      <div style={{ fontSize: 12, color: "var(--gray-500)" }}>
        Validation status unavailable.
      </div>
    );
  }
  const report = (state.data ?? null) as ValidationLike | null;
  const internal = report?.internal_validation ?? null;
  if (!report || !internal || !internal.checks?.length) {
    return (
      <div style={{ fontSize: 12, color: "var(--gray-400)", lineHeight: "1.6" }}>
        <div>
          <strong>Metric Scale:</strong> <span style={{ color: "#f59e0b" }}>ESTIMATED</span>
        </div>
        <div>
          <strong>Metric Validation:</strong> <span style={{ color: "var(--gray-500)" }}>NOT MEASURED</span>
        </div>
        <div style={{ marginTop: 6, fontStyle: "italic" }}>
          No metric-validation report exists for this run.
        </div>
      </div>
    );
  }
  const passed = internal.checks.filter((c) => c.measured && c.pass).length;
  const total = internal.checks.length;
  return (
    <div style={{ fontSize: 12, color: "var(--gray-400)", lineHeight: "1.6" }}>
      <div>
        <strong>Metric Scale:</strong>{" "}
        <span style={{ color: "#34d399" }}>
          {report.capability_state === "GPS_GEOREFERENCED" ? "GPS GEOREFERENCED" : "TELEMETRY-ALIGNED"}
        </span>
      </div>
      <div>
        <strong>Metric Validation:</strong>{" "}
        <span style={{ color: internal.verified ? "#34d399" : "#f59e0b" }}>
          {internal.verified ? `CONSISTENT (${passed}/${total} checks)` : `ISSUES (${passed}/${total} checks pass)`}
        </span>
      </div>
      <div style={{ marginTop: 6, fontStyle: "italic" }}>{report.certification_status}</div>
    </div>
  );
}

export default function Viewer() {
  const { selected } = useRuns();
  const runId = selected?.run_id ?? null;
  const { meshUrl, denseUrl, sparseUrl, combinedUrl, poses, alignment, loading, error, points, meshFaces, cameras, gps, reprojectionErrorPx, sparseDense, meanSpacingM } =
    useViewerArtifacts(runId);
  // Open on the MESH when the run has one: the textured GLB is the artifact
  // that shows the reconstructed scene as imagery, and it was previously only
  // reachable by finding the toolbar toggle. Runs without a mesh fall back to
  // the point cloud automatically (ViewerCanvas loads the cloud when meshUrl
  // is null), so this default is safe for every run.
  const [mode, setMode] = useState<ViewMode>("mesh");
  const effectiveMeshUrl = mode === "mesh" ? meshUrl : null;
  // A run with no mesh at all would otherwise sit in an empty "Mesh" mode.
  const shownMode: ViewMode = meshUrl === null && mode === "mesh" ? "pointcloud" : mode;
  const [viewCommand, setViewCommand] = useState<ViewCommand | null>(null);
  const [layersOpen, setLayersOpen] = useState(true);
  const [infoOpen, setInfoOpen] = useState(true);
  const [layers, setLayers] = useState([
    { name: "Camera Path", visible: true, enabled: true, icon: "camera" as IconName },
    { name: "GPS", visible: true, enabled: true, icon: "pin" as IconName },
    // `enabled` flips on once the loaded cloud reports a confidence attribute.
    { name: "Confidence Heatmap", visible: false, enabled: false, icon: "map" as IconName },
  ]);
  const [threshold, setThreshold] = useState(60);
  const [livePoints, setLivePoints] = useState<{ points: number; cameras: number; gps: number } | null>(null);
  /** What the loaded cloud actually carries — measured, not assumed. */
  const [confAvailable, setConfAvailable] = useState<boolean | null>(null);
  const [confRange, setConfRange] = useState<{ min: number; max: number } | null>(null);

  const toggleLayer = (idx: number) =>
    setLayers((ls) => ls.map((l, i) => (i === idx ? { ...l, visible: !l.visible } : l)));

  return (
    <div className="viewer-container">
      <div className="viewer-stage">
        {!runId ? (
          <div className="viewer-empty">
            No runs yet. Start a mission to produce a reconstruction.
          </div>
        ) : loading ? (
          <div className="viewer-empty">Loading reconstruction…</div>
        ) : error ? (
          <div className="viewer-empty error">Failed to load reconstruction: {error}</div>
        ) : (
          <ViewerCanvas
            meshUrl={effectiveMeshUrl}
            denseUrl={denseUrl}
            sparseUrl={sparseUrl}
            combinedUrl={combinedUrl}
            poses={poses}
            alignment={alignment}
            viewCommand={viewCommand}
            showCameraPath={layers[0].visible}
            showGps={layers[1].visible}
            heatmap={shownMode === "heatmap"}
            meanSpacingM={meanSpacingM}
            confidenceThreshold={threshold}
            onConfidenceAvailable={(available, min, max) => {
              setConfAvailable(available);
              setConfRange(available ? { min, max } : null);
              setLayers((ls) => ls.map((l, i) => (i === 2 ? { ...l, enabled: available } : l)));
            }}
            onLoaded={(p, c, g) => setLivePoints({ points: p, cameras: c, gps: g })}
            onError={(_message) => {
              /* overlay handled via stats; PLY failures surface as error text below */
            }}
          />
        )}
        {shownMode === "heatmap" && runId && !loading && !error && confAvailable === false && (
          <div className="viewer-heatmap-notice">
            Heatmap visualization requires per-point confidence, which this
            reconstruction does not contain. Only aggregate reprojection
            statistics are available.
          </div>
        )}
        {/* One HUD row over the stage: mode/view controls on the left, panel
            toggles on the right. As one wrapping row they never overlap each
            other in a narrow stage, which stacked absolute controls did. */}
        <div className="viewer-hud">
        <ViewerToolbar mode={shownMode} onChange={setMode} onView={setViewCommand} />
        {/* Collapsible side panels — the canvas keeps its width in narrow windows. */}
        <div className="viewer-panel-toggles">
          <button
            className={`viewer-panel-toggle ${layersOpen ? "" : "closed"}`}
            onClick={() => setLayersOpen((o) => !o)}
            title={layersOpen ? "Hide layers panel" : "Show layers panel"}
            aria-label={layersOpen ? "Hide layers panel" : "Show layers panel"}
          >
            <Icon name="map" />
            <span>Layers</span>
          </button>
          <button
            className={`viewer-panel-toggle ${infoOpen ? "" : "closed"}`}
            onClick={() => setInfoOpen((o) => !o)}
            title={infoOpen ? "Hide info panel" : "Show info panel"}
            aria-label={infoOpen ? "Hide info panel" : "Show info panel"}
          >
            <Icon name="info" />
            <span>Info</span>
          </button>
        </div>
        </div>
        {/* The run summary counts the telemetry fixes the georef stage matched;
            the canvas overlay can only count markers baked into poses.json, which
            is zero for telemetry-assisted runs. Prefer the authoritative count. */}
        <ViewerStats
          points={livePoints?.points ?? points}
          // The canvas reports what it loaded: point samples in Point Cloud
          // mode, the GLB's own vertex count in Mesh mode. Label it for what
          // it is rather than calling a mesh's vertices "points".
          pointsLabel={shownMode === "mesh" ? "Mesh Vertices" : "Points"}
          meshFaces={meshFaces}
          cameras={livePoints?.cameras ?? cameras}
          gps={gps ?? livePoints?.gps ?? null}
          reprojectionErrorPx={reprojectionErrorPx}
          sparseDense={sparseDense}
          runId={runId}
        />
      </div>

      {layersOpen && (
        <div className="viewer-sidebar">
          <ViewerLayers layers={layers} onToggle={toggleLayer} />
          <div style={{ marginTop: 20 }}>
            <ConfidenceLegend
              threshold={threshold}
              onThreshold={setThreshold}
              available={confAvailable}
              range={confRange}
            />
          </div>
          <div className="section-title" style={{ marginTop: 20 }}>
            Ground Level
          </div>
          <GroundLevelBlock alignment={alignment} />
        </div>
      )}

      {infoOpen && (
        <div className="viewer-sidebar">
          <div className="section-title">Metric Scale & Validation</div>
          <MetricValidationBlock runId={runId} />
        </div>
      )}
    </div>
  );
}
