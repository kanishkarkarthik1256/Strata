import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import Reports from "./Reports";
import { RunsProvider } from "../hooks/useRuns";
import type { RunSummary } from "../lib/types";

/**
 * Regression tests for the Depth Backend card: it must render from the run's
 * actual depth stage detail (never a hardcoded model name), show the honest
 * "unknown" fallback when the detail is absent, and a stereo run must never
 * claim Depth Anything. Guards against the row regressing into the old
 * always-dead `data.dependencies` conditional, which no manifest satisfies.
 */

function runWithStages(stages: RunSummary["stages"]): RunSummary {
  return {
    run_id: "run_test",
    dataset: "airport1/video.mp4",
    mission: "Mission run_test",
    status: "completed",
    stages,
    metrics: {},
    dependencies: {},
    limitations: [],
    artifacts: [],
  };
}

function mockManifestFetch(
  manifest: Record<string, unknown>,
  accuracy?: { status: number; body: unknown },
  dsm?: { status: number; body: unknown },
): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (url: string) => {
    // Accuracy is served by the metric-validation endpoint, which returns 404
    // when the run has no measurement artifact (the default here, matching the
    // real backend).
    if (url.includes("/api/reconstruction/metric-validation/")) {
      return new Response(
        JSON.stringify(accuracy ? accuracy.body : { detail: "not found" }),
        { status: accuracy ? accuracy.status : 404 },
      );
    }
    // Ground-truth DSM accuracy: only runs with a registered reference grid
    // answer; the default 404 is the honest "nothing to claim here".
    if (url.includes("/api/reconstruction/dsm-accuracy/")) {
      return new Response(
        JSON.stringify(dsm ? dsm.body : { detail: "not found" }),
        { status: dsm ? dsm.status : 404 },
      );
    }
    if (url.endsWith("/api/runs")) {
      return new Response(
        JSON.stringify({ runs: [runWithStages({ depth: "completed" })] }),
        { status: 200 },
      );
    }
    if (url.endsWith("/api/runs/run_test/manifest")) {
      return new Response(JSON.stringify(manifest), { status: 200 });
    }
    // RunsProvider mounts with no selection and briefly requests
    // /api/runs/null/manifest — the real backend 404s that path.
    return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderReports() {
  return render(
    <MemoryRouter>
      <RunsProvider>
        <Reports />
      </RunsProvider>
    </MemoryRouter>,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("Reports — Depth Backend card", () => {
  it("shows the real backend label from the depth stage detail (depth_anything run)", async () => {
    mockManifestFetch({
      run_id: "run_test",
      status: "COMPLETED",
      stages: {
        depth: { status: "PASS", detail: { backend: "depth_anything", checkpoint: "depth_anything_v2_vits.pth" } },
      },
    });
    renderReports();

    // The runs list and the manifest resolve in sequence before the card
    // renders; waitFor re-queries so React's re-render can't detach nodes.
    await waitFor(() => expect(screen.getByText("Depth Backend")).toBeInTheDocument(), { timeout: 5000 });
    expect(screen.getByText("Depth Anything V2 (SfM-aligned, not metric)")).toBeInTheDocument();
    expect(screen.getByText("depth_anything_v2_vits.pth")).toBeInTheDocument();
    // A DA run must not show the stereo label.
    expect(screen.queryByText("Stereo SGBM (SfM-scaled)")).not.toBeInTheDocument();
  });

  it("never claims Depth Anything for a stereo run", async () => {
    mockManifestFetch({
      run_id: "run_test",
      status: "COMPLETED",
      stages: {
        depth: { status: "PASS", detail: { backend: "stereo" } },
      },
    });
    renderReports();

    await waitFor(() => expect(screen.getByText("Stereo SGBM (SfM-scaled)")).toBeInTheDocument(), { timeout: 5000 });
    expect(screen.queryByText(/Depth Anything/)).not.toBeInTheDocument();
  });

  it("shows the honest unknown fallback when the manifest has no depth detail", async () => {
    mockManifestFetch({ run_id: "run_test", status: "COMPLETED", stages: {} });
    renderReports();

    await waitFor(() => expect(screen.getByText("Depth Backend")).toBeInTheDocument(), { timeout: 5000 });
    const unknowns = screen.getAllByText("unknown");
    expect(unknowns.length).toBe(2);
    expect(screen.queryByText("No dependency record in this manifest.")).not.toBeInTheDocument();
  });

  it("never requests /api/runs/null/manifest before a run is selected", async () => {
    // Regression: RunsProvider boots with no selection and the page used to
    // fire a doomed /api/runs/null/manifest request on every mount.
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith("/api/runs/run_test/manifest")) {
        return new Response(JSON.stringify({ run_id: "run_test", stages: {} }), { status: 200 });
      }
      if (url.endsWith("/api/runs")) {
        return new Response(
          JSON.stringify({ runs: [runWithStages({ depth: "completed" })] }),
          { status: 200 },
        );
      }
      return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    renderReports();

    await waitFor(() => expect(screen.getByText("Depth Backend")).toBeInTheDocument(), { timeout: 5000 });
    const requested = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(requested).not.toContain("/api/runs/null/manifest");
  });
});

/**
 * The Accuracy section must be driven by the metric-validation endpoint, and a
 * run with NO measurement artifact must never render an accuracy claim.
 *
 * Regression: the endpoint used to answer every existing project with a 200
 * envelope whose `accuracy_summary.relative_reconstruction` was hardcoded to
 * "available", so this page rendered "Available (reprojection / cross-view /
 * dense→mesh diagnostics)" for a run that never executed, and the honest
 * "no report exists" state below became unreachable.
 */
describe("Reports — Accuracy section", () => {
  it("renders the measured report and reads it from the metric-validation endpoint", async () => {
    const fetchMock = mockManifestFetch(
      { run_id: "run_test", status: "COMPLETED", stages: {} },
      {
        status: 200,
        body: {
          run_id: "run_test",
          certification_status: "CERTIFIED ≤1m (STRATA engineering criterion)",
          certification_reason: "3D RMSE 0.412 m ≤ 1.0 m and 3D P95 0.880 m ≤ 1.0 m",
          accuracy_summary: {
            // The token the backend actually emits (both builders now agree on
            // "measured"; one of them used to say "available", which this page
            // did not match, so a measured run rendered "Unavailable").
            relative_reconstruction: "measured",
            metric_scale: "validated (trajectory scale error 1.20%)",
            absolute: "CERTIFIED ≤1m (STRATA engineering criterion)",
          },
        },
      },
    );
    renderReports();

    await waitFor(
      () => expect(screen.getByText(/CERTIFIED ≤1m/)).toBeInTheDocument(),
      { timeout: 5000 },
    );
    expect(screen.getByText(/3D RMSE 0.412 m/)).toBeInTheDocument();
    expect(screen.getByText(/trajectory scale error 1.20%/)).toBeInTheDocument();
    // A measured report must not render its own relative quality as missing.
    expect(screen.getByText(/Available \(reprojection/)).toBeInTheDocument();
    expect(screen.queryByText(/^Unavailable$/)).not.toBeInTheDocument();
    // The empty state must not also be showing.
    expect(screen.queryByText(/No metric-validation report exists/)).not.toBeInTheDocument();
    // Wiring: the page must read accuracy through the endpoint, not a raw
    // artifact path (a swap back to getRunJsonArtifact would pass silently
    // before this assertion existed).
    const requested = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(requested.some((u) => u.includes("/api/reconstruction/metric-validation/run_test"))).toBe(true);
    expect(requested.some((u) => u.includes("validation/validation_report.json"))).toBe(false);
    // The export row offers only what works: PDF export never produced a file,
    // so the dead control is gone rather than sitting there disabled.
    expect(screen.getByText("Export Data (JSON)")).toBeInTheDocument();
    expect(screen.queryByText(/PDF/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/unavailable\)/)).not.toBeInTheDocument();
  });

  it("shows the honest empty state when the run has no accuracy report", async () => {
    mockManifestFetch({ run_id: "run_test", status: "COMPLETED", stages: {} });
    renderReports();

    await waitFor(
      () => expect(screen.getByText(/No metric-validation report exists for this run/)).toBeInTheDocument(),
      { timeout: 5000 },
    );
    expect(screen.getByText(/NOT CERTIFIED — NO INDEPENDENT REFERENCE/)).toBeInTheDocument();
    // The fabricated-claim regression: nothing may assert that relative
    // diagnostics are available when nothing was measured.
    expect(screen.queryByText(/Available \(reprojection/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Relative reconstruction quality/)).not.toBeInTheDocument();
  });

  it("renders the ground-truth DSM panel when the run has a reference grid", async () => {
    mockManifestFetch(
      { run_id: "run_test", status: "COMPLETED", stages: {} },
      undefined,
      {
        status: 200,
        body: {
          status: "ok",
          mae_m: 6.122,
          p95_m: 22.361,
          height_datum_offset_m: -493.96,
          coverage: 0.403,
        },
      },
    );
    renderReports();

    await waitFor(
      () => expect(screen.getByText(/vs reference DSM \(ground truth/)).toBeInTheDocument(),
      { timeout: 5000 },
    );
    expect(screen.getByText("6.12 m")).toBeInTheDocument();
    expect(screen.getByText(/Coverage of reference grid/)).toBeInTheDocument();
    expect(screen.getByText(/40\.3%/)).toBeInTheDocument();
    // The datum offset is labeled as a removed convention, never as error.
    expect(screen.getByText(/convention, removed/)).toBeInTheDocument();
  });

  it("renders no DSM claim when the run has no reference grid", async () => {
    const fetchMock = mockManifestFetch({ run_id: "run_test", status: "COMPLETED", stages: {} });
    renderReports();

    await waitFor(
      () => expect(screen.getByText(/No metric-validation report exists for this run/)).toBeInTheDocument(),
      { timeout: 5000 },
    );
    expect(screen.queryByText(/vs reference DSM/)).not.toBeInTheDocument();
    // Wiring: the DSM endpoint must actually be consulted (a silent removal
    // of the fetch would leave the panel absent for the wrong reason).
    const requested = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(requested.some((u) => u.includes("/api/reconstruction/dsm-accuracy/run_test"))).toBe(true);
  });
});
