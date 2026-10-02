/**
 * Capability model for the UI — derived from the REAL backend responses:
 * host capabilities (GET /api/system/capabilities) plus the selected run's
 * dependency record (manifest.dependencies, the Phase 9.5.1 validation state).
 *
 * States are honest: READY only when the actual record says the capability
 * exists; NOT_VALIDATED where only installation is known; UNAVAILABLE when
 * absent. Metric 3D accuracy is NOT validated until a run provides evidence.
 */

import type { HostCapabilities, RunDependencies } from "./types";

export type CapabilityState = "READY" | "UNAVAILABLE" | "NOT_VALIDATED" | "WARNING";

export interface UiCapability {
  label: string;
  state: CapabilityState;
  detail: string;
}

const fmt = (n: number | null | undefined, suffix = ""): string =>
  n === null || n === undefined ? "Not available" : `${n}${suffix}`;

/** Host-level capabilities from GET /api/system/capabilities. */
export function hostCapabilities(caps: HostCapabilities | null): UiCapability[] {
  if (!caps) return [];
  // Defensive: the backend may return a partial body (missing sections);
  // every access is optional-chained so the UI never crashes on shape drift.
  const cpu = caps.compute?.cpu;
  const gpu = caps.compute?.gpu;
  const memory = caps.memory?.process_rss_bytes;
  const disk = caps.disk;
  const deps = caps.dependencies;
  const smoke = caps.smoke_tests;
  const depth = caps.depth_model;
  const fmtVer = (v: string | null | undefined): string => (v ? v : "Unknown");

  // Video pipeline facts: the decode path is OpenCV (FFI into bundled
  // FFmpeg libs); a standalone CLI is optional extra capability.
  const decodeOk = Boolean(smoke?.video_decode?.ok);
  const ffmpegBin = Boolean(caps.tools?.ffmpeg);
  const ffmpegState: CapabilityState = ffmpegBin ? "READY" : decodeOk ? "READY" : "UNAVAILABLE";
  const ffmpegDetail = ffmpegBin
    ? smoke?.video_decode?.ok
      ? `Binary + OpenCV decode verified (${smoke.video_decode.resolution ?? ""})`
      : "Binary available"
    : decodeOk
      ? "OpenCV decode verified (standalone CLI absent)"
      : "No decoder verified";

  // COLMAP: the embedded pycolmap engine IS the reconstruction path (run
  // records carry its "COLMAP Engine" row, which is where a real SfM smoke
  // result is reported). The standalone CLI binary is not part of the
  // pipeline: it read as a broken capability on a machine that never needs it,
  // so it is no longer a row.

  // Depth model: READY only when a real inference smoke succeeded.
  const depthSmoke = smoke?.depth ?? null;
  const depthState: CapabilityState = depthSmoke?.ok
    ? "READY"
    : depth?.detected
      ? "NOT_VALIDATED"
      : "UNAVAILABLE";
  const depthDetail = depthSmoke?.ok
    ? `Checkpoint ${depth?.checkpoint ?? "?"} · CPU inference ${depthSmoke.infer_seconds ?? "?"}s (relative depth, not metric)`
    : depth?.detected
      ? `Checkpoint ${depth.checkpoint} detected; inference not verified`
      : "Checkpoint not found";

  return [
    {
      label: "CPU",
      state: cpu?.cores ? "READY" : "UNAVAILABLE",
      detail: `${fmt(cpu?.cores, " cores")} · load ${fmt(cpu?.load_1m)}`,
    },
    {
      label: "Memory (process RSS)",
      state: memory ? "READY" : "UNAVAILABLE",
      detail: memory ? `${(memory / (1024 ** 3)).toFixed(2)} GB` : "Not available",
    },
    {
      label: "Disk",
      state: disk?.total_gb ? "READY" : "UNAVAILABLE",
      detail: `${fmt(disk?.free_gb, " GB free")} of ${fmt(disk?.total_gb, " GB")}`,
    },
    {
      label: "GPU Acceleration",
      state: gpu?.available ? "READY" : "UNAVAILABLE",
      detail: gpu?.available ? (gpu.device ?? "GPU / Metal detected") : "PyTorch CPU Engine",
    },
    {
      label: "FFmpeg",
      state: ffmpegState,
      detail: ffmpegDetail,
    },
    // Measured runtime facts (backend imports/probes the actual environment).
    {
      label: "Runtime",
      state: deps?.python || deps?.torch ? "READY" : "UNAVAILABLE",
      detail: `Python ${fmtVer(deps?.python)} · PyTorch ${fmtVer(deps?.torch)} · OpenCV ${fmtVer(deps?.opencv)}`,
    },
    {
      label: "COLMAP Version",
      state: deps?.pycolmap ? "READY" : "UNAVAILABLE",
      detail: deps?.pycolmap ? `pycolmap ${deps.pycolmap}` : "pycolmap version Unknown",
    },
    {
      label: "Depth Model",
      state: depthState,
      detail: depthDetail,
    },
  ];
}

/** Pipeline-level capabilities from the selected run's dependency record. */
export function runCapabilities(deps: RunDependencies | null, hostCaps?: HostCapabilities | null): UiCapability[] {
  if (!deps) return [];
  // Every fact comes from the run's dependency record or the live host
  // response — unknown values render as "Unknown", never a guessed version.
  const torchVer = deps.torch_version ?? "Unknown";
  const depthCkpt = deps.depth_anything_checkpoint ?? "Unknown";
  const pycolmapVer = deps.pycolmap_version ?? "Unknown";
  const gpuOk = Boolean(deps.cuda_available || deps.mps_available || hostCaps?.compute?.gpu?.available);
  const gpuDetail =
    deps.gpu_name ?? hostCaps?.compute?.gpu?.device ?? (gpuOk ? "GPU acceleration reported" : "Not available");

  return [
    {
      label: "AI Models",
      state: deps.torch_version && deps.depth_anything_checkpoint ? "READY" : "NOT_VALIDATED",
      detail: `PyTorch ${torchVer} · ${depthCkpt}`,
    },
    {
      label: "COLMAP Engine",
      state: deps.pycolmap_version ? "READY" : "NOT_VALIDATED",
      detail:
        pycolmapVer === "Unknown"
          ? "pycolmap version Unknown"
          : hostCaps?.smoke_tests?.colmap?.ok
            ? `pycolmap ${pycolmapVer} (SfM smoke: ${hostCaps.smoke_tests.colmap.images_registered ?? "?"} imgs, ${hostCaps.smoke_tests.colmap.points3d ?? "?"} pts)`
            : `pycolmap ${pycolmapVer} (installed; SfM smoke pending)`,
    },
    {
      label: "FFmpeg",
      state: deps.ffmpeg_available ? "READY" : "UNAVAILABLE",
      detail: deps.ffmpeg_available ? "Video frame extraction engine ready" : "Not available",
    },
    {
      label: "GPU Acceleration",
      state: gpuOk ? "READY" : "UNAVAILABLE",
      detail: gpuDetail,
    },
    {
      label: "Depth Model",
      state: deps.depth_anything_checkpoint ? "READY" : "NOT_VALIDATED",
      detail: deps.depth_anything_checkpoint
        ? "Depth Anything V2 (relative depth)"
        : "Depth model Unknown",
    },
    // NOTE: Metric 3D Validation is deliberately NOT emitted here — a static
    // row would beat the real, report-derived row in the Dashboard's dedup
    // and assert a capability nothing measured. The Dashboard derives that
    // row from the selected run's validation report via
    // metricValidationFromReport().
  ];
}

/**
 * The rows of the one "System Status" panel, in display order.
 *
 * Precedence is explicit, and the live host measurement comes first. A run's
 * dependency record is a point-in-time snapshot that is frequently empty (the
 * API returns `{}` for runs that predate it), so letting it shadow the host
 * made the panel report "FFmpeg: Unavailable" on a machine whose decode smoke
 * had just passed and "Depth Model: Not validated" beside a measured
 * inference — and made the verdict change with run selection.
 *
 * A run contributes rows only when its record carries facts: an empty record
 * is an absent measurement, not a negative one. Settings still shows the two
 * sources as separate tables, so no evidence is lost.
 */
export function systemStatusRows(
  host: UiCapability[],
  runDeps: RunDependencies | null | undefined,
  hostCaps: HostCapabilities | null,
  metric: UiCapability,
): UiCapability[] {
  const rows = [...host];
  if (runDeps && Object.keys(runDeps).length > 0) {
    rows.push(...runCapabilities(runDeps, hostCaps));
  }
  rows.push(metric);
  return rows.filter((c, i, arr) => arr.findIndex((x) => x.label === c.label) === i);
}

export const metricValidationState: UiCapability = {
  label: "Metric 3D Validation",
  state: "NOT_VALIDATED",
  detail: "No validation report checked — open a run to measure it",
};

/** Derive the Metric 3D Validation row from the selected run's real report.
 *  `null` (report not loaded / run has none) keeps the honest NOT_VALIDATED
 *  state instead of asserting a capability nothing measured. */
export function metricValidationFromReport(report: {
  validation_kind?: string;
  certification_status?: string;
  internal_validation?: { verified?: boolean; checks?: { measured: boolean; pass: boolean | null }[] } | null;
} | null): UiCapability {
  const checks = report?.internal_validation?.checks ?? [];
  if (!report || checks.length === 0) {
    return {
      label: "Metric 3D Validation",
      state: "NOT_VALIDATED",
      detail: "No metric-validation report exists for this run",
    };
  }
  const verified = report.internal_validation?.verified === true;
  return {
    label: "Metric 3D Validation",
    state: verified ? "READY" : "WARNING",
    detail: report.certification_status ?? "Internal-consistency report present",
  };
}