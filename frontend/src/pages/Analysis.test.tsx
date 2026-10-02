import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import Analysis from "./Analysis";
import { RunsProvider } from "../hooks/useRuns";
import type { RunSummary } from "../lib/types";

/**
 * The Analysis page must render only what the /analysis engine measured:
 * area, elevation extremes, flight stats, the per-cell error grid and the
 * metric-validation checks. A run without a dense surface must show the
 * honest "no analysable reconstruction" state — never a fabricated figure.
 */

function runWith(stages: RunSummary["stages"]): RunSummary {
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

const MEASURED_ANALYSIS = {
  run_id: "run_test",
  generated_at: "2026-09-26T03:03:16Z",
  frame: {
    kind: "local_enu",
    metric: true,
    units: "meters",
    anchor_wgs84: { lat: 52.5163, lon: 13.4018, alt: 365.9 },
    alignment_scale: 0.99995,
    source: "georef/dense_model_enu.ply",
  },
  sources: { surface: "georef/dense_model_enu.ply", surface_points: 2227996, accuracy_map: "sparse_dense_error.ply" },
  site: {
    units: "meters",
    cell_size_m: 1,
    area_m2: 255473,
    area_ha: 25.55,
    area_km2: 0.255,
    area_method: "occupied cells (density-supported)",
    outline_area_m2: 452722,
    outline_area_km2: 0.4527,
    outline_method: "convex hull of occupied cells",
    occupied_cells: 255473,
    bbox: { min: [-60.8, 40.37], max: [769.83, 824.42], size: [830.63, 784.05] },
    centroid: { enu: [333.4, 418.67], wgs84: { lat: 52.5201, lon: 13.4067 } },
    outline_wgs84: [[52.521, 13.4132], [52.5224, 13.41277], [52.5236, 13.41022]],
    footprint_cells: null,
  },
  elevation: {
    units: "meters",
    surface: "georef/dense_model_enu.ply",
    lowest: { enu: [259.22, 72.09, -377.88], wgs84: { lat: 52.51699, lon: 13.40569 }, alt_m: -11.96 },
    highest: { enu: [513.1, 472.28, -239.14], wgs84: { lat: 52.52059, lon: 13.4094 }, alt_m: 96.13 },
    min: -12.0,
    max: 96.13,
    relief: 108.13,
    mean: 3.2,
    median: 2.8,
    robust_min: -8.0,
    robust_max: 90.0,
    robust_relief: 98.0,
    robust_percentiles: [1, 99],
    histogram: null,
    datum: "alt_m (WGS84 ellipsoid)",
  },
  density: { points: 2227996, points_per_m2: 8.72, mean_spacing_m: 0.564, coverage_percent: 47.66, occlusion_percent: 17.85 },
  flight: {
    available: true,
    points: 89,
    path_length_m: 1368.82,
    mean_speed_m_s: 15.55,
    max_speed_m_s: 71.54,
    altitude_std_m: 6.25,
    discontinuities: 4,
    drift_m: 39.67,
    gps_score: 78.09,
    grade: "Good",
    track_points: 89,
    track_wgs84: [[52.5163, 13.4018], [52.517, 13.4025], [52.5177, 13.4032]],
    track_up_min_m: 0,
    track_up_max_m: 10,
    agl_min_m: 120,
    agl_max_m: 160,
    agl_mean_m: 140,
    cameras: 89,
    focal_px_median: 1818,
    gsd_m_per_px: 0.08,
    gsd_method: "median focal vs median scene depth",
    anchor_wgs84: { lat: 52.5163, lon: 13.4018 },
  },
  consistency: {
    correspondences: 52869,
    median_m: 0.685,
    p95_m: 4.0979,
    within_3m_pct: 91.43,
    measure: "sparse↔dense nearest-neighbour agreement (internal consistency)",
    source: "dense_report.json",
  },
  accuracy_map: {
    available: true,
    cell_size_m: 10,
    min_points_per_cell: 3,
    origin_enu: [-60.8, 40.37],
    cells: [[2, 22, 1.706, 4, 2.85], [3, 21, 1.25, 22, 1.995], [4, 22, 2.2, 5, 2.898]],
  },
  available: true,
  notes: [],
  metric_validation: {
    validation_kind: "internal_consistency",
    certification_status: "NOT CERTIFIED — NO INDEPENDENT REFERENCE",
    certification_reason: "no independent LiDAR/checkpoint reference exists for this scene",
    internal_validation: {
      label: "Internal consistency — no independent reference available",
      verified: true,
      checks: [
        { name: "Sparse reprojection error (median)", measured: true, value: 0.2576, unit: "px", criterion: "≤ 2.0 px", pass: true },
        { name: "Cross-view depth agreement (median)", measured: true, value: 0.764, unit: "m", criterion: "≤ 2.0 m", pass: true },
      ],
    },
  },
};

function renderAnalysis() {
  return render(
    <MemoryRouter>
      <RunsProvider>
        <Analysis />
      </RunsProvider>
    </MemoryRouter>,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("Analysis — measured payload", () => {
  it("renders area, elevation extremes, flight, error grid and validation from the engine", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith("/api/runs")) {
        return new Response(JSON.stringify({ runs: [runWith({})] }), { status: 200 });
      }
      if (url.endsWith("/api/runs/run_test/analysis")) {
        return new Response(JSON.stringify(MEASURED_ANALYSIS), { status: 200 });
      }
      return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    localStorage.setItem("strata.selectedRun", "run_test");

    renderAnalysis();

    // Area numbers come from the payload (m² primary, km² secondary).
    await waitFor(() => expect(screen.getByText("255,473 m²")).toBeInTheDocument(), { timeout: 5000 });
    expect(screen.getByText("0.45 km²")).toBeInTheDocument(); // site covered (incl. gaps)
    // Elevation extremes
    expect(screen.getByText("96.13 m alt_m (WGS84 ellipsoid)")).toBeInTheDocument();
    expect(screen.getByText("-12 m alt_m (WGS84 ellipsoid)")).toBeInTheDocument();
    // Flight
    expect(screen.getByText("1,369 m")).toBeInTheDocument();
    expect(screen.getByText("Good (78.1)")).toBeInTheDocument();
    // Consistency
    expect(screen.getByText("0.69 / 4.1 m")).toBeInTheDocument();
    // Validation checks render measured values with pass/fail
    expect(screen.getByText(/0.258 px/)).toBeInTheDocument();
    expect(screen.getByText(/0.764 m/)).toBeInTheDocument();
    expect(screen.getByText("NOT CERTIFIED — NO INDEPENDENT REFERENCE")).toBeInTheDocument();
    // Map renders with OSM tiles when GPS exists (leaflet container mounts).
    await waitFor(() => expect(document.querySelector(".leaflet-container")).not.toBeNull(), { timeout: 5000 });
  });

  it("shows the honest empty state when the run has no dense surface", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith("/api/runs")) {
        return new Response(JSON.stringify({ runs: [runWith({})] }), { status: 200 });
      }
      if (url.endsWith("/api/runs/run_test/analysis")) {
        return new Response(
          JSON.stringify({
            run_id: "run_test",
            generated_at: "2026-09-26T03:03:16Z",
            frame: { kind: "reconstruction", metric: false, units: "reconstruction_units", anchor_wgs84: null, alignment_scale: null, source: "" },
            sources: { surface: "", surface_points: 0, accuracy_map: null },
            site: null,
            elevation: null,
            density: null,
            flight: null,
            consistency: null,
            accuracy_map: { available: false },
            available: false,
            notes: [],
            metric_validation: null,
          }),
          { status: 200 },
        );
      }
      return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    localStorage.setItem("strata.selectedRun", "run_test");

    renderAnalysis();

    expect(await screen.findByText(/no analysable reconstruction/i, {}, { timeout: 5000 })).toBeInTheDocument();
    // Nothing fabricated: no area, no map.
    expect(screen.queryByText(/m²/)).not.toBeInTheDocument();
    expect(document.querySelector(".leaflet-container")).toBeNull();
  });

  it("renders without a map when the run has no georeferenced outline", async () => {
    const payload = {
      ...MEASURED_ANALYSIS,
      site: { ...MEASURED_ANALYSIS.site, outline_wgs84: null, centroid: { enu: [1, 2], wgs84: null } },
      flight: { ...MEASURED_ANALYSIS.flight, track_wgs84: null },
      frame: { ...MEASURED_ANALYSIS.frame, anchor_wgs84: null },
    };
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith("/api/runs")) {
        return new Response(JSON.stringify({ runs: [runWith({})] }), { status: 200 });
      }
      if (url.endsWith("/api/runs/run_test/analysis")) {
        return new Response(JSON.stringify(payload), { status: 200 });
      }
      return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    localStorage.setItem("strata.selectedRun", "run_test");

    renderAnalysis();

    // Area still renders from real artifacts; the map section is absent
    // entirely because neither outline nor track is georeferenced.
    await waitFor(() => expect(screen.getByText("255,473 m²")).toBeInTheDocument(), { timeout: 5000 });
    expect(screen.queryByText(/Modelled area & flight track/)).not.toBeInTheDocument();
    expect(document.querySelector(".leaflet-container")).toBeNull();
  });
});
