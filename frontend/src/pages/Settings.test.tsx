import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import Settings from "./Settings";
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

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Settings", () => {
  it("renders the real capabilities shape without crashing (audit regression)", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith("/api/system/capabilities")) return new Response(JSON.stringify(REAL_CAPS), { status: 200 });
      if (url.endsWith("/health")) return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
      return new Response(JSON.stringify({ runs: [] }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <MemoryRouter>
        <RunsProvider>
          <Settings />
        </RunsProvider>
      </MemoryRouter>,
    );

    // One System Status panel owns the capability rows, and this is where it
    // lives (it moved off the Dashboard).
    expect(await screen.findByText("System Status")).toBeInTheDocument();
    expect(screen.getAllByText("GPU Acceleration").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Ready").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/PyTorch CPU Engine/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Metric 3D Validation").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Not validated").length).toBeGreaterThan(0);
    // The COLMAP CLI is not a reported capability: its absence is not a gap.
    expect(screen.queryByText("COLMAP CLI")).toBeNull();
  });

  it("handles a missing capabilities endpoint gracefully", async () => {
    mockFetchJson({ runs: [] });
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({}), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    // /api/system/capabilities returns 200 with an unexpected shape — must not crash
    render(
      <MemoryRouter>
        <RunsProvider>
          <Settings />
        </RunsProvider>
      </MemoryRouter>,
    );
    expect(await screen.findByText("System Status")).toBeInTheDocument();
  });
});