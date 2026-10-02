import type {
  DenseStageDetail,
  DepthStageDetail,
  DepthViewExclusion,
  DepthViewFailure,
  PlacementStageDetail,
} from "../../lib/types";

/**
 * Reading the backend's per-stage `detail` payloads and rendering what they
 * actually contain. Nothing here invents a value: a payload that predates a
 * key renders nothing for that key.
 */

/** Narrow an unknown stage detail to the depth diagnostics shape. */
function depthDetail(detail: unknown): DepthStageDetail | null {
  if (!detail || typeof detail !== "object") return null;
  return detail as DepthStageDetail;
}

/** Narrow an unknown stage detail to the dense refusal shape. */
function denseDetail(detail: unknown): DenseStageDetail | null {
  if (!detail || typeof detail !== "object") return null;
  return detail as DenseStageDetail;
}

/** Narrow an unknown stage detail to the telemetry-placement shape. */
function placementDetail(detail: unknown): PlacementStageDetail | null {
  if (!detail || typeof detail !== "object") return null;
  const p = (detail as { placement?: unknown }).placement;
  if (!p || typeof p !== "object") return null;
  return p as PlacementStageDetail;
}

/** Raw backend id the backend reports in the stage detail (depth only). */
export function depthBackendId(detail: unknown): string | null {
  if (detail && typeof detail === "object") {
    const backend = (detail as { backend?: unknown }).backend;
    if (typeof backend === "string" && backend) return backend;
  }
  return null;
}

/**
 * A refused per-window placement still completes the sparse stage, so it renders
 * here rather than under a failure: the run continues on the rigid global
 * similarity, and the user is told what was refused and on what evidence.
 */
function PlacementNotice({ detail }: { detail: unknown }) {
  const p = placementDetail(detail);
  const refused = p?.refused;
  if (!p || !refused) return null;
  const retained =
    typeof refused.points_retained_fraction === "number"
      ? `${(refused.points_retained_fraction * 100).toFixed(1)}%`
      : null;
  const floor =
    typeof refused.retained_floor === "number" ? `${(refused.retained_floor * 100).toFixed(0)}%` : null;
  return (
    <div className="proc-stage-failures">
      <div>
        <strong>Per-window telemetry placement refused:</strong> {refused.reason ?? "unspecified"}
      </div>
      {typeof refused.points_before === "number" && typeof refused.points_after === "number" && (
        <div>
          <strong>Points:</strong> {refused.points_before.toLocaleString()} →{" "}
          {refused.points_after.toLocaleString()}
          {retained ? ` (${retained} retained${floor ? `, floor ${floor}` : ""})` : ""}
        </div>
      )}
      {typeof refused.observations_dropped === "number" && (
        <div>
          <strong>Observations dropped:</strong> {refused.observations_dropped.toLocaleString()}
        </div>
      )}
      <div>
        <strong>Continued with:</strong> {p.mode ?? "global similarity"}
      </div>
      {refused.note && <div>{refused.note}</div>}
    </div>
  );
}

/** Per-view depth failures joined with their reasons. The backend stores the
 *  frame ids in `failed` and the reasons in `failure_reasons`; older payloads
 *  may have only the ids — render the id alone rather than nothing. */
function depthViewFailures(detail: DepthStageDetail | null): DepthViewFailure[] {
  if (!detail) return [];
  const ids = Array.isArray(detail.failed) ? detail.failed : [];
  const reasons = detail.failure_reasons ?? {};
  return ids
    .filter((id): id is string => typeof id === "string")
    .map((frame) => ({ frame, reason: reasons[frame] ?? "unavailable" }));
}

/** Excluded views, tolerating payloads that predate the array shape. */
function depthViewExclusions(detail: DepthStageDetail | null): DepthViewExclusion[] {
  if (!detail || !Array.isArray(detail.excluded)) return [];
  return detail.excluded.filter(
    (e): e is DepthViewExclusion =>
      !!e && typeof e === "object" && typeof (e as DepthViewExclusion).frame === "string",
  );
}

/** Compact list rendering with a consistent "name: reason" line style. */
function DetailLineList({ items, label }: { items: Array<{ frame: string; reason: string }>; label: string }) {
  if (!items.length) return null;
  return (
    <div>
      <strong>{label} ({items.length}):</strong>{" "}
      {items.map((it, i) => (
        <span key={`${it.frame}-${i}`} className="proc-stage-fail-line">
          {it.frame}: {it.reason}
          {i < items.length - 1 ? "; " : ""}
        </span>
      ))}
    </div>
  );
}

/**
 * Failure surfacing for the depth and dense stages.
 *
 * Renders only what the backend actually reported, and only for a failed
 * (or partially failed) stage: per-view depth failures with their reasons,
 * conditioning-excluded views, and the dense audit's named failing criteria
 * (falling back to the dominant failure cause when nothing generated).
 * Older payloads without these keys render nothing.
 */
export default function StageFailureDetails({ stageId, detail }: { stageId: string; detail: unknown }) {
  if (stageId === "sparse") {
    return <PlacementNotice detail={detail} />;
  }
  if (stageId === "depth") {
    const d = depthDetail(detail);
    const failures = depthViewFailures(d);
    const exclusions = depthViewExclusions(d);
    if (!failures.length && !exclusions.length) return null;
    return (
      <div className="proc-stage-failures">
        <DetailLineList items={failures} label="Failed views" />
        <DetailLineList items={exclusions} label="Excluded views (conditioning)" />
      </div>
    );
  }
  if (stageId === "dense") {
    const d = denseDetail(detail);
    const criteria = d?.failing_criteria ?? [];
    const dominant = typeof d?.dominant_failure_reason === "string" ? d.dominant_failure_reason : null;
    if (!criteria.length && !dominant) return null;
    return (
      <div className="proc-stage-failures">
        {criteria.length > 0 && (
          <div>
            <strong>Failed audit criteria ({criteria.length}):</strong> {criteria.join(", ")}
          </div>
        )}
        {dominant && (
          <div>
            <strong>Dominant cause:</strong> {dominant}
          </div>
        )}
      </div>
    );
  }
  return null;
}
