/**
 * Single owner for depth-backend display: maps the backend recorded in the
 * run's depth stage detail (manifest / live status) to a human-readable
 * label, plus the honest fallback. Both Processing and Reports import from
 * here — the labels must never diverge between pages.
 *
 * Honesty rules:
 * - a stereo run's label never contains "Depth Anything";
 * - Depth Anything is always qualified as SfM-aligned and NOT metric (SfM
 *   scale is arbitrary; metric validation needs independent ground truth);
 * - unknown/missing data reports "unknown" rather than a guess.
 */

export const DEPTH_BACKEND_LABELS: Record<string, string> = {
  depth_anything: "Depth Anything V2 (SfM-aligned, not metric)",
  stereo: "Stereo SGBM (SfM-scaled)",
};

/** Label for a backend value from run data; "unknown" when absent/unrecognized-but-empty. */
export function depthBackendLabel(backend: unknown): string {
  if (typeof backend === "string" && backend.length > 0) {
    return DEPTH_BACKEND_LABELS[backend] ?? backend;
  }
  return "unknown";
}

/** The note shown on Processing for the run's depth backend; null when none applies. */
export function depthBackendNote(backend: unknown): string | null {
  if (backend === "depth_anything") {
    return "Depth Anything V2 output is aligned to SfM camera-space geometry (1/Z = a·D_raw + b) before dense integration.";
  }
  if (backend === "stereo") {
    return "Stereo SGBM depth is scaled to the SfM reconstruction geometry before dense integration.";
  }
  return null;
}
