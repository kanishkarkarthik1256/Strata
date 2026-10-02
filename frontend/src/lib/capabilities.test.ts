import { describe, expect, it } from "vitest";
import { hostCapabilities, runCapabilities, systemStatusRows } from "./capabilities";
import type { HostCapabilities, RunDependencies } from "./types";

const REAL_CAPS: HostCapabilities = {
  compute: {
    cpu: { cores: 8, load_1m: 1.2, count: 8 },
    gpu: { available: false, reason: "cuda_unavailable" },
    device: "cpu",
  },
  memory: { process_rss_bytes: 734_003_200 },
  disk: { path: "/x", total_gb: 500, used_gb: 120, free_gb: 380 },
  tools: { colmap: false, ffprobe: false, ffmpeg: true },
};

const REAL_DEPS: RunDependencies = {
  torch_version: "2.2.2",
  device: "cpu",
  cuda_available: false,
  mps_available: false,
  pycolmap_version: "3.12.5",
  ffmpeg_available: true,
  depth_anything_checkpoint: "models/weights/depth_anything_v2_vits.pth",
  status: "PASS",
};

describe("capability derivation from real backend shapes", () => {
  it("never crashes on the real capabilities shape (regression for the audit crash)", () => {
    const items = hostCapabilities(REAL_CAPS);
    const labels = items.map((i) => i.label);
    expect(labels).toContain("GPU Acceleration");
    expect(labels).toContain("FFmpeg");
    // The standalone COLMAP CLI is not part of the pipeline: its absence is
    // not a capability gap, so it is not a row (the embedded pycolmap engine
    // is reported through "COLMAP Version" / the run's "COLMAP Engine").
    expect(labels).not.toContain("COLMAP CLI");
    expect(labels).toContain("COLMAP Version");
    const gpu = items.find((i) => i.label === "GPU Acceleration");
    expect(gpu?.state).toBe("UNAVAILABLE");
  });

  it("returns [] when capabilities are missing", () => {
    expect(hostCapabilities(null)).toEqual([]);
  });

  it("derives run-level states from the Phase 9.5.1 dependency record", () => {
    const items = runCapabilities(REAL_DEPS);
    const byLabel = Object.fromEntries(items.map((i) => [i.label, i.state]));
    expect(byLabel["COLMAP Engine"]).toBe("READY");
    expect(byLabel["AI Models"]).toBe("READY");
    expect(byLabel["FFmpeg"]).toBe("READY");
    expect(byLabel["GPU Acceleration"]).toBe("UNAVAILABLE");
    // Metric 3D Validation is intentionally absent from the static deps
    // derivation: the Dashboard derives that row from the run's real
    // validation report (metricValidationFromReport), so a static row can
    // never override measured state.
    expect(byLabel["Metric 3D Validation"]).toBeUndefined();
  });

  it("returns [] when no run is selected", () => {
    expect(runCapabilities(null)).toEqual([]);
  });

  it("never fabricates versions, checkpoints, or availability (honesty regression)", () => {
    // A dependency record with nothing filled in must produce no invented
    // facts: no "2.x", no "3.12.5", no hardcoded checkpoint filename.
    const items = runCapabilities({} as RunDependencies);
    const text = items.map((i) => `${i.label}: ${i.detail}`).join(" | ");
    expect(text).not.toContain("2.x");
    expect(text).not.toContain("3.12.5");
    expect(text).not.toContain("depth_anything_v2_vits.pth");
    expect(text).not.toContain("MPS Metal Acceleration");
    // Unknown facts render as Unknown and the depth model is not claimed READY.
    expect(items.find((i) => i.label === "AI Models")?.state).toBe("NOT_VALIDATED");
    expect(items.find((i) => i.label === "Depth Model")?.detail).toContain("Unknown");
    expect(items.find((i) => i.label === "FFmpeg")?.state).toBe("UNAVAILABLE");
  });

  it("reports real dependency values verbatim when present", () => {
    const items = runCapabilities(REAL_DEPS);
    const models = items.find((i) => i.label === "AI Models");
    expect(models?.detail).toContain("PyTorch 2.2.2");
    expect(models?.detail).toContain("depth_anything_v2_vits.pth");
    expect(items.find((i) => i.label === "COLMAP Engine")?.detail).toContain("pycolmap 3.12.5");
    expect(items.find((i) => i.label === "FFmpeg")?.state).toBe("READY");
  });
});

describe("systemStatusRows — one row per label, live host wins", () => {
  const READY_HOST: HostCapabilities = {
    ...REAL_CAPS,
    depth_model: {
      installed: true,
      detected: true,
      smoke_tested: true,
      checkpoint: "models/weights/depth_anything_v2_vitb.pth",
      status_level: "READY",
    },
    smoke_tests: {
      video_decode: { ok: true, resolution: "1920x1080" },
      depth: { ok: true, device: "cpu", infer_seconds: 5.9 },
    },
  };

  it("the run's empty record no longer shadows the host's measured truth", () => {
    // The live defect: the API returns `dependencies: {}` for older runs, and
    // the run rows (FFmpeg UNAVAILABLE, Depth Model NOT_VALIDATED) beat the host
    // rows (both verified by smoke test) purely by array order.
    const rows = systemStatusRows(
      hostCapabilities(READY_HOST),
      {},
      READY_HOST,
      { label: "Metric 3D Validation", state: "NOT_VALIDATED", detail: "none" },
    );
    const byLabel = Object.fromEntries(rows.map((r) => [r.label, r.state]));
    expect(byLabel["FFmpeg"]).toBe("READY");
    expect(byLabel["Depth Model"]).toBe("READY");
    // An empty run record contributes nothing rather than five Unknown rows.
    expect(byLabel["AI Models"]).toBeUndefined();
    expect(rows.filter((r) => r.label === "FFmpeg")).toHaveLength(1);
    expect(rows.filter((r) => r.label === "Depth Model")).toHaveLength(1);
  });

  it("a run with real dependencies adds its own rows and still cannot override the host", () => {
    const stale = { ...REAL_DEPS, ffmpeg_available: false };
    const rows = systemStatusRows(
      hostCapabilities(READY_HOST),
      stale,
      READY_HOST,
      { label: "Metric 3D Validation", state: "READY", detail: "certified" },
    );
    const byLabel = Object.fromEntries(rows.map((r) => [r.label, r.state]));
    expect(byLabel["FFmpeg"]).toBe("READY"); // host measurement, not the record
    expect(byLabel["AI Models"]).toBe("READY"); // run-owned row is kept
    expect(byLabel["COLMAP Engine"]).toBe("READY");
    expect(byLabel["Metric 3D Validation"]).toBe("READY");
    expect(new Set(rows.map((r) => r.label)).size).toBe(rows.length);
  });

  it("falls back to the run's rows when the host call failed", () => {
    const rows = systemStatusRows(
      hostCapabilities(null),
      REAL_DEPS,
      null,
      { label: "Metric 3D Validation", state: "NOT_VALIDATED", detail: "none" },
    );
    const byLabel = Object.fromEntries(rows.map((r) => [r.label, r.state]));
    expect(byLabel["FFmpeg"]).toBe("READY");
    expect(byLabel["AI Models"]).toBe("READY");
  });
});