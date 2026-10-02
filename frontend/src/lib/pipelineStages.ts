import type { PipelineStatusResponse } from "./types";

/**
 * The pipeline's stage vocabulary — one owner for which stages a run has, the
 * label to show for each, and the engine shown when the stage reports none.
 * The backend keys its stage map by `id`; nothing here may invent a stage.
 *
 * `fallback` names the ENGINE (what the stage uses), never an outcome: the
 * stage's own record already reports whether it ran. georef's used to read
 * "skipped without telemetry", so a completed run whose georeferenced ENU
 * artifacts were on disk still showed "skipped" next to its GPS telemetry
 * panel.
 */
export const STAGES = [
  { id: "frames", label: "Frame Extraction", fallback: "OpenCV" },
  { id: "sparse", label: "Camera Localization & SfM", fallback: "pycolmap" },
  { id: "depth", label: "Depth Estimation", fallback: "depth backend" },
  { id: "dense", label: "Dense Reconstruction", fallback: "multi-view fusion" },
  { id: "georef", label: "Georeferencing & ENU Alignment", fallback: "GPS telemetry → local ENU" },
] as const;

/** One stage's contribution to the overall fraction. */
export interface StageProgress {
  id: string;
  /** 0..1 — the real backend fraction where one exists, else 0. */
  fraction: number;
}

/**
 * Per-stage fractions: completed/skipped = 1.0, a running stage contributes
 * its real backend `progress` fraction (0..1, from stage events) when one is
 * reported, and anything not yet reported contributes 0. Fractions the
 * backend has not published are never invented.
 */
export function stageProgressList(status: PipelineStatusResponse | null): StageProgress[] | null {
  if (!status) return null;
  const list = STAGES.map((s) => {
    const st = status.stages?.[s.id];
    const stStatus = st?.status;
    const progress = st?.progress;
    if (stStatus === "completed" || stStatus === "skipped") return { id: s.id, fraction: 1 };
    if (stStatus === "running" && typeof progress === "number") {
      return { id: s.id, fraction: Math.min(1, Math.max(0, progress)) };
    }
    return { id: s.id, fraction: 0 };
  });
  // An empty stage map means no signal at all yet — same as no status.
  if (!status.stages || Object.keys(status.stages).length === 0) return null;
  return list;
}

/**
 * Overall progress = the mean of the per-stage fractions. While a stage is
 * mid-flight this moves continuously (that is the "stuck at 0%" fix); the
 * old completed-counts-only version held 0% until the first stage finished.
 */
export function overallProgress(status: PipelineStatusResponse | null): number | null {
  const list = stageProgressList(status);
  if (!list) return null;
  return list.reduce((acc, s) => acc + s.fraction, 0) / list.length;
}
