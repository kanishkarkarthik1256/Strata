import type { ViewerAlignment } from "../../lib/types";

/** Human labels for the evidence that fixed the viewer's up axis. */
const SOURCE_LABELS: Record<string, string> = {
  enu_vertical: "telemetry ENU vertical (metric)",
  camera_facing_surface: "camera-facing surface fit",
  camera_image_up: "camera orientation prior",
  smallest_extent_axis: "bounding-box axis",
};

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "#10b981",
  medium: "#f59e0b",
  low: "#ef4444",
};

export function levelSourceLabel(source: string | null | undefined): string | null {
  if (!source) return null;
  return SOURCE_LABELS[source] ?? source;
}

/**
 * Ground-level provenance for the viewer.
 *
 * The viewer rotates the model so its ground sits on the y=0 grid. THAT
 * LEVELING IS A CLAIM, and it is worth showing where the up axis came from: a
 * telemetry-placed run knows the true vertical (ENU), while a run with no
 * telemetry falls back to an assumption that can be tens of degrees wrong. A
 * run with no alignment at all must say so instead of rendering unlevelled
 * geometry without explanation.
 */
export default function GroundLevelBlock({ alignment }: { alignment: ViewerAlignment | null }) {
  const leveling = alignment?.leveling ?? null;

  if (!alignment || alignment.method === "identity_fallback" || !leveling) {
    return (
      <div style={{ fontSize: 12, color: "var(--gray-400)", lineHeight: "1.6" }}>
        <div>
          <strong>Ground Level:</strong>{" "}
          <span style={{ color: "#9ca3af" }}>NOT ESTIMATED</span>
        </div>
        <div style={{ marginTop: 6, fontStyle: "italic", color: "var(--gray-500)" }}>
          {alignment?.fallback_reason ??
            "No levelling estimate for this run — the model is shown in its reconstructed coordinates."}
        </div>
      </div>
    );
  }

  const source = levelSourceLabel(leveling.up_prior_source);
  const color = CONFIDENCE_COLORS[leveling.confidence] ?? "#9ca3af";
  const verified = leveling.plane_measured;

  return (
    <div style={{ fontSize: 12, color: "var(--gray-400)", lineHeight: "1.6" }}>
      <div>
        <strong>Ground Level:</strong>{" "}
        <span style={{ color }}>
          {source ? source.toUpperCase() : "UNKNOWN"}
        </span>
      </div>
      <div>
        <strong>Confidence:</strong> <span style={{ color }}>{leveling.confidence.toUpperCase()}</span>
      </div>
      <div>
        <strong>Plane Measured:</strong>{" "}
        <span style={{ color: verified ? "#10b981" : "#9ca3af" }}>
          {verified ? "YES" : "NO"}
        </span>
      </div>
      <div style={{ marginTop: 6, fontStyle: "italic", color: "var(--gray-500)" }}>
        {verified
          ? "Up axis confirmed against the reconstructed surface, so the grid is the real ground plane."
          : "Up axis taken from leveling evidence, not a measured plane — the grid may be tilted relative to the ground."}
      </div>
    </div>
  );
}
