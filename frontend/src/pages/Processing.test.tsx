import { afterEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import ProcessingScreen from "./Processing";
import { estimateEta } from "../hooks/useRunMonitor";
import { mockFetchFailure, mockFetchJson, mockFetchNetworkError } from "../test/setup";

/**
 * The real mid-run payload shape from GET /api/pipeline/status when the
 * backend reports live stage events (see routes/pipeline.py `_live_status`):
 * completed stages carry duration/count but no progress fraction; the running
 * stage carries a real `progress` value when the stage publishes one.
 */
const LIVE_MID_RUN = {
  job_id: "londonjob1234",
  status: "running",
  run_time_ms: 0,
  error: "",
  stages: {
    frames: { status: "completed", duration_ms: 5210.4, count: 112, error: "", detail: {} },
    sparse: { status: "completed", duration_ms: 67483.1, count: 2930, error: "", detail: {} },
    depth: { status: "running", duration_ms: 0, count: 0, error: "", detail: {}, progress: 0.4 },
    dense: { status: "pending", duration_ms: 0, count: 0, error: "", detail: {} },
    georef: { status: "pending", duration_ms: 0, count: 0, error: "", detail: {} },
  },
  profile: {},
  resume: {},
};

function renderAtRoute() {
  // Mirror how the app navigates here: /new-mission pushes the mission name
  // in router state; the page falls back to "Mission <id>" without it.
  return render(
    <MemoryRouter
      initialEntries={[
        { pathname: "/processing/londonjob1234", state: { missionName: "london.mp4" } },
      ]}
    >
      <Routes>
        <Route path="/processing/:jobId" element={<ProcessingScreen />} />
      </Routes>
    </MemoryRouter>,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Realistic failed-dense payload: the depth stage completed with cached maps
 *  plus real per-view failures/exclusions, and the dense stage refused with
 *  named audit criteria — exactly what the backend emits (DepthSummary.to_dict
 *  + dense refusal detail). */
const FAILED_DENSE = {
  job_id: "londonjob1234",
  status: "failed",
  run_time_ms: 117469.18,
  error: "Depth maps failed diagnostic quality audit (Criteria A-G) — cannot feed into TSDF reconstruction",
  stages: {
    frames: { status: "completed", duration_ms: 5210.4, count: 112, error: "", detail: {} },
    sparse: { status: "completed", duration_ms: 67483.1, count: 2930, error: "", detail: {} },
    depth: {
      status: "completed", duration_ms: 3642.77, count: 27, error: "",
      detail: {
        generated: ["frame_000029"],
        cached: ["frame_000032", "frame_000033", "frame_000037"],
        failed: ["frame_000011", "frame_000012", "frame_000015"],
        failure_reasons: {
          frame_000011: "source_image_missing (searched: selected/, frames/)",
          frame_000012: "source_image_missing (searched: selected/, frames/)",
          frame_000015: "inference_error: CUDA out of memory",
        },
        excluded: [
          { frame: "frame_000047", reason: "hover segment: no usable baseline at any depth" },
          { frame: "frame_000050", reason: "signed depth-structure gate: inverted ramp dz/dv=+0.31" },
        ],
        count_generated: 1,
        count_excluded: 2,
        backend: "depth_anything",
      },
    },
    dense: {
      status: "failed", duration_ms: 53133.51, count: 0, error: "Depth maps failed diagnostic quality audit (Criteria A-G) — cannot feed into TSDF reconstruction",
      detail: {
        failing_criteria: ["B_depth_range_vs_sfm_geometry", "F_depth_artifacts_present"],
        audited_count: 27,
        reason: "depth maps failed the A–G audit",
      },
    },
    georef: { status: "pending", duration_ms: 0, count: 0, error: "", detail: {} },
  },
  profile: {},
  resume: { frames: true, sparse: true, depth: false, dense: false, georef: false },
};

describe("ProcessingScreen — stage failure surfacing", () => {
  it("failed dense stage shows failed views with reasons, excluded views, and audit criteria", async () => {
    mockFetchJson(FAILED_DENSE);
    renderAtRoute();

    expect(await screen.findByText("RECONSTRUCTION STOPPED")).toBeTruthy();

    // Depth per-view failures render with their backend-stated reasons.
    // (frame and reason are separate inline spans, and the list separator
    // trails each line — match across them with a prefix check.)
    expect(
      await screen.findByText((_, el) => {
        const cls = el && typeof el.className === "string" ? el.className : "";
        return cls.includes("proc-stage-fail-line") && (el?.textContent ?? "").startsWith("frame_000011: source_image_missing (searched: selected/, frames/)");
      }),
    ).toBeTruthy();
    expect(screen.getByText(/frame_000015: inference_error: CUDA out of memory/)).toBeTruthy();

    // Conditioning-excluded views render with the named gate reason.
    expect(screen.getByText(/Failed views \(3\):/)).toBeTruthy();
    expect(screen.getByText(/frame_000047: hover segment: no usable baseline at any depth/)).toBeTruthy();
    expect(screen.getByText(/Excluded views \(conditioning\) \(2\):/)).toBeTruthy();

    // Dense refusal names the exact failing audit criteria (label is in a
    // <strong>, the criterion list is the sibling text node — assert both).
    expect(screen.getByText(/Failed audit criteria \(2\):/)).toBeTruthy();
    expect(screen.getByText(/B_depth_range_vs_sfm_geometry, F_depth_artifacts_present/)).toBeTruthy();
  });

  it("partially-failed depth stage surfaces per-view failures while dense is pending", async () => {
    mockFetchJson({
      ...FAILED_DENSE,
      status: "failed",
      stages: {
        ...FAILED_DENSE.stages,
        depth: { ...FAILED_DENSE.stages.depth, status: "failed" },
        dense: { status: "pending", duration_ms: 0, count: 0, error: "", detail: {} },
      },
    });
    renderAtRoute();

    expect(await screen.findByText("RECONSTRUCTION STOPPED")).toBeTruthy();
    expect(screen.getByText(/Failed views \(3\):/)).toBeTruthy();
    expect(screen.getByText(/frame_000012: source_image_missing/)).toBeTruthy();
    // Dense has no detail yet — nothing may render for it.
    expect(screen.queryByText(/Failed audit criteria/)).toBeNull();
  });

  it("renders no failure details when the payload lacks the newer keys (older backend)", async () => {
    mockFetchJson({
      ...LIVE_MID_RUN,
      status: "failed",
      error: "Depth maps failed diagnostic quality audit (Criteria A-G)",
      stages: {
        ...LIVE_MID_RUN.stages,
        depth: { status: "failed", duration_ms: 35029, count: 0, error: "", detail: {} },
      },
    });
    renderAtRoute();

    expect(await screen.findByText("RECONSTRUCTION STOPPED")).toBeTruthy();
    // No diagnostics in the payload → no failure-details blocks at all.
    expect(screen.queryByText(/Failed views/)).toBeNull();
    expect(screen.queryByText(/Excluded views/)).toBeNull();
    expect(screen.queryByText(/Failed audit criteria/)).toBeNull();
  });

  it("a refused per-window placement is surfaced with its measured evidence", async () => {
    // Real shape: the sparse stage COMPLETED, but trajectory_sync refused the
    // piecewise correction and fell back to the rigid global similarity. The
    // numbers come from the placement block the backend now records.
    const refusedPlacement = {
      ...LIVE_MID_RUN,
      status: "completed",
      stages: {
        ...LIVE_MID_RUN.stages,
        sparse: {
          status: "completed",
          duration_ms: 305006.9,
          count: 89,
          error: "",
          detail: {
            localization: { mode: "telemetry_assisted" },
            placement: {
              mode: "global_similarity",
              matched_cameras: 89,
              match_percent: 100,
              refused: {
                reason: "low_point_retention",
                points_before: 52882,
                points_after: 52882,
                points_retained_fraction: 0.0067,
                retained_floor: 0.3,
                observations_dropped: 569903,
                note: "per-window transforms left the observations unexplainable",
              },
            },
          },
        },
      },
    };
    mockFetchJson(refusedPlacement);
    renderAtRoute();

    expect(await screen.findByText(/Per-window telemetry placement refused/)).toBeTruthy();
    expect(screen.getByText(/low_point_retention/)).toBeTruthy();
    expect(screen.getByText(/52,882 → 52,882/)).toBeTruthy();
    expect(screen.getByText(/0.7% retained/)).toBeTruthy();
    expect(screen.getByText(/569,903/)).toBeTruthy();
    expect(screen.getByText(/global_similarity/)).toBeTruthy();
  });

  it("a successful placement renders no placement notice", async () => {
    mockFetchJson({
      ...LIVE_MID_RUN,
      status: "completed",
      stages: Object.fromEntries(
        Object.entries(LIVE_MID_RUN.stages).map(([k, v]) => [
          k,
          { ...v, status: "completed", detail: k === "sparse" ? { placement: { mode: "piecewise_rigid", refused: null } } : {} },
        ]),
      ),
    });
    renderAtRoute();

    expect(await screen.findByText("RECONSTRUCTION COMPLETE")).toBeTruthy();
    expect(screen.queryByText(/placement refused/)).toBeNull();
  });

  it("a passing depth stage with no failures renders no failure-details block", async () => {
    mockFetchJson({
      ...LIVE_MID_RUN,
      status: "completed",
      stages: Object.fromEntries(
        Object.entries(LIVE_MID_RUN.stages).map(([k, v]) => [k, { ...v, status: "completed" }]),
      ),
    });
    renderAtRoute();

    expect(await screen.findByText("RECONSTRUCTION COMPLETE")).toBeTruthy();
    expect(screen.queryByText(/Failed views/)).toBeNull();
    expect(screen.queryByText(/Excluded views/)).toBeNull();
    expect(screen.queryByText(/Failed audit criteria/)).toBeNull();
  });
});

describe("ProcessingScreen — live pipeline status", () => {
  it("renders the real mid-run payload: completed stages, running stage with progress bar", async () => {
    mockFetchJson(LIVE_MID_RUN);
    renderAtRoute();

    // Banner reflects the live run state.
    expect(await screen.findByText("PROCESSING LIVE")).toBeTruthy();
    expect(screen.getByText("london.mp4")).toBeTruthy();

    // Header names the actually-running stage (real backend state, not invented).
    expect(await screen.findByText("Processing: Depth Estimation")).toBeTruthy();

    // Completed stages show real durations from the events.
    expect(screen.getByText("Duration: 67.5s")).toBeTruthy();

    // The running stage shows the real backend fraction — no invented percentage.
    expect(screen.getByText("40%")).toBeTruthy();

    // Signature: the stage timeline renders one node per stage.
    expect(document.querySelectorAll(".proc-tl-node").length).toBe(5);
  });

  it("shows 'Processing…' without a percentage when the backend reports no progress", async () => {
    const noProgress = {
      ...LIVE_MID_RUN,
      stages: {
        ...LIVE_MID_RUN.stages,
        depth: { status: "running", duration_ms: 0, count: 0, error: "", detail: {} },
      },
    };
    mockFetchJson(noProgress);
    renderAtRoute();

    expect(await screen.findByText("Processing: Depth Estimation")).toBeTruthy();
    expect(screen.getByText("Processing…")).toBeTruthy();
    // No fabricated percentage anywhere.
    expect(screen.queryByText(/%\s*$/)).toBeNull();
    expect(screen.queryByText("40%")).toBeNull();
  });

  it("renders the failed state with the backend error and retry affordance", async () => {
    mockFetchJson({
      ...LIVE_MID_RUN,
      status: "failed",
      error: "Depth maps failed diagnostic quality audit (Criteria A-G)",
      stages: {
        ...LIVE_MID_RUN.stages,
        depth: { status: "failed", duration_ms: 35029, count: 0, error: "Depth maps failed diagnostic quality audit (Criteria A-G)", detail: {} },
      },
    });
    renderAtRoute();

    expect(await screen.findByText("RECONSTRUCTION STOPPED")).toBeTruthy();
    expect(screen.getByText(/^Stage: /)).toBeTruthy();
    expect(screen.getByText(/Depth maps failed diagnostic quality audit/)).toBeTruthy();
    expect(screen.getByText("Retry Reconstruction")).toBeTruthy();
    expect(screen.getByText("View Diagnostics")).toBeTruthy();
  });

  it("renders the completed state with the open-model CTA", async () => {
    mockFetchJson({
      ...LIVE_MID_RUN,
      status: "completed",
      run_time_ms: 794640.69,
      stages: Object.fromEntries(
        Object.entries(LIVE_MID_RUN.stages).map(([k, v]) => [k, { ...v, status: "completed" }]),
      ),
    });
    renderAtRoute();

    expect(await screen.findByText("RECONSTRUCTION COMPLETE")).toBeTruthy();
    expect(screen.getByText("OPEN 3D MODEL")).toBeTruthy();
    expect(screen.getByText("All stages completed")).toBeTruthy();
  });

  it("fabricates no telemetry claims before the georef stage reports", async () => {
    mockFetchJson(LIVE_MID_RUN);
    renderAtRoute();

    // While georef is pending its detail is empty — the mode is not yet
    // known, so no telemetry block may render at all.
    expect(await screen.findByText("Processing: Depth Estimation")).toBeTruthy();
    expect(document.querySelector(".tel-block")).toBeNull();
  });

  it("shows GPS — Not available for a video-only run (georef detail landed)", async () => {
    mockFetchJson({
      ...LIVE_MID_RUN,
      status: "completed",
      stages: {
        ...LIVE_MID_RUN.stages,
        georef: {
          status: "completed",
          duration_ms: 40,
          count: 0,
          error: "",
          detail: {
            telemetry_mode: "VIDEO_ONLY",
            note: "no GPS telemetry available — georeferencing skipped (outputs stay in local coordinates)",
          },
        },
      },
    });
    renderAtRoute();

    expect(await screen.findByText("Not available")).toBeTruthy();
    expect(screen.getByText("video-only")).toBeTruthy();
    expect(screen.getByText(/georeferencing skipped/)).toBeTruthy();
    expect(screen.queryByText("Samples")).toBeNull();
  });

  it("shows the telemetry sync stats for a telemetry-enabled run", async () => {
    mockFetchJson({
      ...LIVE_MID_RUN,
      status: "completed",
      stages: {
        ...LIVE_MID_RUN.stages,
        georef: {
          status: "completed",
          duration_ms: 1200,
          count: 10,
          error: "",
          detail: {
            telemetry_mode: "VIDEO_WITH_EXTERNAL_TELEMETRY",
            sync: {
              telemetry_samples: 22,
              matched_frames: 10,
              unmatched_frames: 0,
              timestamp_offset_sec: 0.25,
              gps_available: true,
              telemetry_quality: "good",
              invalid_samples_dropped: 0,
              median_sample_spacing_sec: 0.5,
              sufficient_for_georeferencing: true,
            },
          },
        },
      },
    });
    renderAtRoute();

    expect(await screen.findByText("Available")).toBeTruthy();
    expect(screen.getByText("22")).toBeTruthy();
    expect(screen.getByText("+0.25s")).toBeTruthy();
    expect(screen.getByText("good")).toBeTruthy();
    // Offset within half the sample spacing — no clock warning.
    expect(screen.queryByText(/CSV clock likely/)).toBeNull();
  });

  it("labels the depth stage with the backend the run actually used (depth_anything)", async () => {
    const daRun = {
      ...LIVE_MID_RUN,
      status: "completed",
      stages: {
        ...LIVE_MID_RUN.stages,
        depth: {
          status: "completed", duration_ms: 30000, count: 9, error: "",
          detail: { backend: "depth_anything", count_generated: 9 },
        },
      },
    };
    mockFetchJson(daRun);
    renderAtRoute();

    expect(await screen.findByText("RECONSTRUCTION COMPLETE")).toBeTruthy();
    expect(screen.getByText("Depth Anything V2 (SfM-aligned, not metric)")).toBeTruthy();
    // The honest capability note names the same backend.
    expect(screen.getByText(/Depth Anything V2 output is aligned/)).toBeTruthy();
  });

  it("a stereo-fallback run must NOT claim Depth Anything", async () => {
    const stereoRun = {
      ...LIVE_MID_RUN,
      status: "completed",
      stages: {
        ...LIVE_MID_RUN.stages,
        depth: {
          status: "completed", duration_ms: 30000, count: 9, error: "",
          detail: { backend: "stereo", count_generated: 9 },
        },
      },
    };
    mockFetchJson(stereoRun);
    renderAtRoute();

    expect(await screen.findByText("RECONSTRUCTION COMPLETE")).toBeTruthy();
    expect(screen.getByText("Stereo SGBM (SfM-scaled)")).toBeTruthy();
    expect(screen.queryByText("Depth Anything V2 (SfM-aligned, not metric)")).toBeNull();
    expect(screen.queryByText(/Depth Anything V2 output is aligned/)).toBeNull();
    // Stereo note instead, same honest metric-scale stance.
    expect(screen.getByText(/Stereo SGBM depth is scaled/)).toBeTruthy();
  });

  it("shows no depth-specific claim before the depth stage reports a backend", async () => {
    mockFetchJson(LIVE_MID_RUN);
    renderAtRoute();

    expect(await screen.findByText("Processing: Depth Estimation")).toBeTruthy();
    // Depth detail is empty so far — nothing may claim a backend yet.
    expect(screen.queryByText(/Depth Anything/)).toBeNull();
    expect(screen.queryByText(/Depth & metric scale/)).toBeNull();
  });

  it("warns when the clock offset is large relative to sample spacing", async () => {
    const warnPayload = {
      ...LIVE_MID_RUN,
      status: "completed",
      stages: {
        ...LIVE_MID_RUN.stages,
        georef: {
          status: "completed",
          duration_ms: 1200,
          count: 10,
          error: "",
          detail: {
            telemetry_mode: "VIDEO_WITH_EXTERNAL_TELEMETRY",
            sync: {
              telemetry_samples: 11,
              matched_frames: 10,
              unmatched_frames: 1,
              timestamp_offset_sec: 0.9,
              gps_available: true,
              telemetry_quality: "good",
              invalid_samples_dropped: 0,
              median_sample_spacing_sec: 1.0,
              sufficient_for_georeferencing: true,
            },
          },
        },
      },
    };
    mockFetchJson(warnPayload);
    renderAtRoute();

    expect(await screen.findByText(/CSV clock likely doesn't match the video/)).toBeTruthy();
    expect(screen.getByText("+0.9s")).toBeTruthy();
  });
});

describe("estimateEta — honest ETA estimator", () => {
  it("returns null (Estimating…) with fewer than two observations", () => {
    expect(estimateEta([])).toBeNull();
    expect(estimateEta([{ progress: 0.2, elapsedSec: 10 }])).toBeNull();
  });

  it("projects remaining time from the observed rate", () => {
    const eta = estimateEta([
      { progress: 0.4, elapsedSec: 40 },
      { progress: 0.5, elapsedSec: 50 }, // 0.01 stage-fraction/s → 50s left
    ]);
    expect(eta).not.toBeNull();
    expect(eta!.remainingSec).toBeCloseTo(50, 0);
    expect(eta!.confidence).toBe("Low");
  });

  it("never returns a negative ETA", () => {
    const eta = estimateEta([
      { progress: 0.9, elapsedSec: 10 },
      { progress: 1.0, elapsedSec: 20 },
    ]);
    expect(eta).not.toBeNull();
    expect(eta!.remainingSec).toBeGreaterThanOrEqual(0);
  });

  it("refuses to estimate when progress is not advancing", () => {
    expect(
      estimateEta([
        { progress: 0.4, elapsedSec: 10 },
        { progress: 0.4, elapsedSec: 40 },
      ]),
    ).toBeNull();
  });

  it("raises confidence with more history", () => {
    const hist = [
      { progress: 0.1, elapsedSec: 1 },
      { progress: 0.2, elapsedSec: 2 },
      { progress: 0.3, elapsedSec: 3 },
      { progress: 0.4, elapsedSec: 4 },
      { progress: 0.5, elapsedSec: 5 },
      { progress: 0.6, elapsedSec: 6 },
    ];
    expect(estimateEta(hist)!.confidence).toBe("High");
    expect(estimateEta(hist.slice(0, 3))!.confidence).toBe("Moderate");
  });
});

describe("ProcessingScreen — queued, connection, and honesty states", () => {
  it("shows the queued state while the job waits for the worker (no fake progress)", async () => {
    mockFetchJson({
      job_id: "londonjob1234",
      status: "queued",
      run_time_ms: 0,
      error: "",
      stages: {},
      profile: {},
      resume: {},
    });
    renderAtRoute();

    expect(await screen.findByText(/Queued — waiting for the worker/)).toBeTruthy();
    // No percentage claim anywhere: nothing has run yet.
    expect(screen.queryByText(/% of pipeline complete/)).toBeNull();
    expect(screen.getByText(/Estimating time remaining/)).toBeTruthy();
  });

  it("keeps the last known state (not failed) when one poll fails", async () => {
    mockFetchJson(LIVE_MID_RUN);
    renderAtRoute();
    expect(await screen.findByText("PROCESSING LIVE")).toBeTruthy();

    // Next poll throws — the screen must show the retry banner, keep the
    // running state, and NOT mark the run failed. The poll interval is 2s:
    // wait out one real cycle (fake timers are not installed in this file).
    mockFetchNetworkError();
    await vi.waitFor(
      () => {
        expect(screen.getByText(/Connection temporarily unavailable — retrying/)).toBeTruthy();
      },
      { timeout: 4000, interval: 200 },
    );
    expect(screen.getByText("PROCESSING LIVE")).toBeTruthy();
    expect(screen.queryByText("RECONSTRUCTION STOPPED")).toBeNull();
  });

  it("shows the overall progress from the mean of real per-stage fractions", async () => {
    mockFetchJson(LIVE_MID_RUN);
    renderAtRoute();
    // frames+sparse completed (1.0 each), depth running at 0.4, rest pending
    // → mean fraction 2.4/5 = 48%, never a fabricated number.
    expect(await screen.findByText("48% of pipeline complete")).toBeTruthy();
  });

  it("carries no hardcoded run id — the route param drives the UI", async () => {
    mockFetchJson({ ...LIVE_MID_RUN, job_id: "someOtherRun" });
    renderAtRoute();
    expect(await screen.findByText(/RUN londonjob123/)).toBeTruthy();
  });

  it("an unknown run id says so instead of dressing a 404 up as a live run", async () => {
    // GET /api/pipeline/status/{id} answers 404 for an id this backend does
    // not know (stale link, deleted run). The screen used to keep polling
    // forever: "PROCESSING LIVE", every stage "Processing…", and the 404
    // reported as "Connection temporarily unavailable — retrying…".
    mockFetchFailure(404, "Project not found");
    renderAtRoute();

    // Two consecutive 404 polls are required, and the poll interval is 2s.
    await vi.waitFor(
      () => {
        expect(screen.getAllByText("RUN NOT FOUND").length).toBeGreaterThan(0);
      },
      { timeout: 6000, interval: 200 },
    );
    expect(screen.getByText(/no run with the ID/)).toBeTruthy();
    // Nothing may claim to be processing, and no stage row may render.
    expect(screen.queryByText("PROCESSING LIVE")).toBeNull();
    expect(screen.queryByText("Frame Extraction")).toBeNull();
    expect(screen.queryByText(/Connection temporarily unavailable/)).toBeNull();
    // The panel's own escape hatch plus the topline's nav button.
    expect(screen.getAllByText("Back to missions").length).toBeGreaterThan(1);
  });
});
