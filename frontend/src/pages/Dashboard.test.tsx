import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import Dashboard from "./Dashboard";
import { RunsProvider } from "../hooks/useRuns";
import { mockFetchJson } from "../test/setup";

const REAL_CAPS = {
  compute: {
    cpu: { cores: 8, load_1m: 1.2, count: 8 },
    gpu: { available: false, reason: "cuda_unavailable" },
    device: "cpu",
  },
  memory: { process_rss_bytes: 734003200 },
  disk: { path: "/x", total_gb: 500, used_gb: 120, free_gb: 380 },
  tools: { colmap: false, ffprobe: false, ffmpeg: true },
};

const RUNS = {
  runs: [
    {
      run_id: "shitan_ms1_20260909_131251",
      dataset: "shitan",
      mission: "ms1",
      status: "PASS",
      stages: {},
      metrics: {},
      dependencies: {
        torch_version: "2.2.2",
        pycolmap_version: "3.12.5",
        ffmpeg_available: true,
        depth_anything_checkpoint: "models/weights/depth_anything_v2_vits.pth",
        cuda_available: false,
        mps_available: false,
      },
      limitations: [],
      artifacts: [],
      dense_points: 2626577,
      sparse_points: 2796,
      cameras: 10,
      gps_points: 10,
    },
  ],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Dashboard", () => {
  it("renders with the REAL capabilities response (no crash — audit regression)", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url === "/api/runs") return new Response(JSON.stringify(RUNS), { status: 200 });
      if (url === "/api/system/capabilities") return new Response(JSON.stringify(REAL_CAPS), { status: 200 });
      return new Response("{}", { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <MemoryRouter>
        <RunsProvider>
          <Dashboard />
        </RunsProvider>
      </MemoryRouter>,
    );

    expect(await screen.findByText("shitan")).toBeInTheDocument();
    expect(screen.getByText("Total Missions")).toBeInTheDocument();
    expect(screen.getByText("2.63M")).toBeInTheDocument(); // Dense Points from REAL data
    // System Status moved to Settings: the Dashboard is the mission overview
    // and reads runs, not the machine. Capability rows are asserted in
    // Settings.test.tsx.
    expect(screen.queryByText("System Status")).toBeNull();
  });

  it("shows an honest empty state when there are no runs", async () => {
    mockFetchJson({ runs: [] });
    render(
      <MemoryRouter>
        <RunsProvider>
          <Dashboard />
        </RunsProvider>
      </MemoryRouter>,
    );
    expect(await screen.findByText(/No missions found/)).toBeInTheDocument();
  });

  it("never displays fabricated dependency facts when no run deps exist (honesty regression)", async () => {
    // Runs carry an empty dependency record; capabilities must come only
    // from the real backend response. The old code injected a fabricated
    // default ({ torch_version: "2.x", pycolmap_version: "3.12.5" }) that
    // this test rejects.
    const fetchMock = vi.fn(async (url: string) => {
      if (url === "/api/runs") {
        return new Response(
          JSON.stringify({
            runs: [
              {
                run_id: "no_deps_run",
                dataset: "airport1/video.mp4",
                mission: "Mission no_deps_run",
                status: "completed",
                stages: {},
                metrics: {},
                dependencies: {},
                limitations: [],
                artifacts: [],
              },
            ],
          }),
          { status: 200 },
        );
      }
      if (url === "/api/system/capabilities") return new Response(JSON.stringify(REAL_CAPS), { status: 200 });
      return new Response("{}", { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <MemoryRouter>
        <RunsProvider>
          <Dashboard />
        </RunsProvider>
      </MemoryRouter>,
    );

    expect(await screen.findByText("airport1/video.mp4")).toBeInTheDocument();
    const body = document.body.textContent ?? "";
    expect(body).not.toContain("2.x");
    expect(body).not.toContain("3.12.5");
    expect(body).not.toContain("depth_anything_v2_vits.pth");
    expect(body).not.toContain("validated SfM engine");
  });
});