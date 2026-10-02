/**
 * New Mission preview + analytics — rendered-page contract.
 *
 * The page must show a local preview of the selected footage and render
 * fitness insights ONLY from measured stats (browser decoder / server
 * metadata). When the browser cannot decode the file, the page says so and
 * points at the server's ffmpeg fallback instead of failing the mission.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import NewMission from "./NewMission";
import * as insights from "../lib/videoInsights";

const probeMock = vi.spyOn(insights, "probeVideo");
const motionMock = vi.spyOn(insights, "probeMotion");

// The measured healthy reference (airport1: 0.882 static at 1 s cadence,
// 0.337 across its 16.6 s sampling span — a translating camera).
const healthyMotion = {
  staticP50: 0.882,
  staticP90: 0.919,
  pairs: 24,
  sharpP50: 20,
  crossP50: 0.337,
  crossSpanS: 16.6,
  crossPairs: 12,
};
// The genuine duplicate-view signature: 1 s and whole-clip similarity agree.
const stationaryMotion = { ...healthyMotion, staticP50: 0.93, staticP90: 0.96, crossP50: 0.95 };

function renderPage() {
  return render(
    <MemoryRouter>
      <NewMission />
    </MemoryRouter>,
  );
}

function stubFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify({ videos: [] }), { status: 200 })),
  );
}

function aVideoFile(name = "drone_clip.mp4"): File {
  return new File([new Uint8Array(1024)], name, { type: "video/mp4" });
}

beforeEach(() => {
  stubFetch();
});

afterEach(() => {
  vi.unstubAllGlobals();
  probeMock.mockReset();
  motionMock.mockReset();
});

describe("NewMission preview & analytics", () => {
  it("shows a local preview and auto-named mission when a file is picked", async () => {
    probeMock.mockImplementation(() =>
      Promise.resolve({ duration_sec: 45, width: 3840, height: 2160, aspect: 16 / 9 }));
    motionMock.mockImplementation((() => Promise.resolve(healthyMotion)) as never);
    const user = userEvent.setup();
    renderPage();

    const input = screen.getByLabelText("Choose a drone video file");
    await user.upload(input, aVideoFile("My Survey.mp4"));

    const preview = await screen.findByLabelText("Selected footage preview");
    expect(preview.getAttribute("src")).toMatch(/^blob:/);
    expect(screen.getByDisplayValue("My Survey Mission")).toBeTruthy();
    expect(screen.getByText("Change file")).toBeTruthy();
  });

  it("renders fitness insights from measured stats", async () => {
    probeMock.mockImplementation(() =>
      Promise.resolve({ duration_sec: 45, width: 3840, height: 2160, aspect: 16 / 9 }));
    motionMock.mockImplementation((() => Promise.resolve(healthyMotion)) as never);
    const user = userEvent.setup();
    renderPage();

    const input = screen.getByLabelText("Choose a drone video file");
    await user.upload(input, aVideoFile());

    expect(await screen.findByText("Good motion budget")).toBeTruthy();
    expect(screen.getByText("Sharp source")).toBeTruthy();
    expect(screen.getByText("Upload")).toBeTruthy();
  });

  it("renders the readiness panel with per-component bars when motion measured", async () => {
    probeMock.mockImplementation(() =>
      Promise.resolve({ duration_sec: 45, width: 3840, height: 2160, aspect: 16 / 9 }));
    motionMock.mockImplementation((() => Promise.resolve(healthyMotion)) as never);
    const user = userEvent.setup();
    renderPage();

    const input = screen.getByLabelText("Choose a drone video file");
    await user.upload(input, aVideoFile());

    expect(await screen.findByText("Ready to reconstruct")).toBeTruthy();
    expect(screen.getByText("100")).toBeTruthy();
    expect(screen.getByText("View overlap")).toBeTruthy();
    // duration and resolution both max out at 30/30 each.
    expect(screen.getAllByText("30/30").length).toBe(2);
    expect(screen.getByText("40/40")).toBeTruthy();
    // The overlap verdict detail rides on the component tooltip.
    expect(screen.getByTitle(/new vantages/)).toBeTruthy();
  });

  it("warns before starting a mission whose measured signature matches failed runs", async () => {
    probeMock.mockImplementation(() =>
      Promise.resolve({ duration_sec: 45, width: 3840, height: 2160, aspect: 16 / 9 }));
    motionMock.mockImplementation((() => Promise.resolve(stationaryMotion)) as never);
    const user = userEvent.setup();
    renderPage();

    const input = screen.getByLabelText("Choose a drone video file");
    await user.upload(input, aVideoFile("hover_flight.mp4"));

    expect(await screen.findByText("Unlikely to reconstruct")).toBeTruthy();
    expect(screen.getByTitle(/duplicate|one view/i)).toBeTruthy();
    expect(screen.getByText("START ANYWAY — LIKELY TO FAIL")).toBeTruthy();
  });

  it("degrades honestly when the motion probe fails (score without overlap)", async () => {
    probeMock.mockImplementation(() =>
      Promise.resolve({ duration_sec: 45, width: 3840, height: 2160, aspect: 16 / 9 }));
    motionMock.mockImplementation(() => Promise.reject(new Error("seek failed")));
    const user = userEvent.setup();
    renderPage();

    const input = screen.getByLabelText("Choose a drone video file");
    await user.upload(input, aVideoFile());

    expect(await screen.findByText("Ready to reconstruct")).toBeTruthy();
    expect(screen.getByText(/overlap not measured/i)).toBeTruthy();
  });

  it("surfaces an honest message when the browser cannot decode the file", async () => {
    probeMock.mockImplementation(() => Promise.reject(new Error("browser cannot decode")));
    const user = userEvent.setup();
    renderPage();

    const input = screen.getByLabelText("Choose a drone video file");
    await user.upload(input, aVideoFile("exotic_codec.mp4"));

    expect(
      await screen.findByText(/browser cannot decode this file for preview/i),
    ).toBeTruthy();
  });

  it("shows the degraded-probe counts when some pairs are undecodable, not a fake full count", async () => {
    probeMock.mockImplementation(() =>
      Promise.resolve({ duration_sec: 45, width: 3840, height: 2160, aspect: 16 / 9 }));
    motionMock.mockImplementation((() =>
      Promise.resolve({ ...healthyMotion, pairs: 11, pairsAttempted: 24, pairsUnusable: 13 })) as never);
    const user = userEvent.setup();
    renderPage();

    const input = screen.getByLabelText("Choose a drone video file");
    await user.upload(input, aVideoFile("partially_decodable.mp4"));

    expect(await screen.findByText(/11\s*\/\s*24 frame pairs \(13 undecodable\)/i)).toBeTruthy();
  });

  it("launches once for a careless triple click, and still allows a retry", async () => {
    // The disabled attribute is a render-time signal: three clicks dispatched
    // before React re-renders all pass it and used to start the same run three
    // times. One launch per intent, and a failed launch must not latch shut.
    let startCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL) => {
        const u = String(url);
        const json = (body: unknown, status = 200) =>
          new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
        if (u.includes("/start")) {
          startCalls += 1;
          return startCalls === 1 ? json({ detail: "queue unavailable" }, 500) : json({ run_id: "job123" });
        }
        return json({
          videos: [
            { name: "base.mp4", duration_sec: 10, size_bytes: 3_100_000, gps_sources: ["video.SRT"] },
          ],
        });
      }),
    );
    const user = userEvent.setup();
    renderPage();

    await user.click(screen.getByText("From data/ folder"));
    await user.click((await screen.findByText("base.mp4")).closest("button")!);

    const start = screen.getByText("START RECONSTRUCTION").closest("button")!;
    await act(async () => {
      start.click();
      start.click();
      start.click();
    });
    await waitFor(() => expect(startCalls).toBe(1));

    // The backend refusal is surfaced, and the latch is released for a retry.
    expect(await screen.findByText(/queue unavailable/)).toBeTruthy();
    await user.click(screen.getByText("START RECONSTRUCTION"));
    await waitFor(() => expect(startCalls).toBe(2));
  });

  it("selects a server-side video without starting it; the start bar launches it", async () => {
    // The page promises "review what STRATA can read from it, then start the
    // pipeline". A row click used to fire POST /api/data-videos/{name}/start
    // immediately, so merely reading the list launched a full reconstruction.
    const calls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL, init?: RequestInit) => {
        const u = String(url);
        calls.push(`${init?.method ?? "GET"} ${u}`);
        const body = u.includes("/start")
          ? { run_id: "job123" }
          : {
              videos: [
                { name: "base.mp4", duration_sec: 10, size_bytes: 3_100_000, gps_sources: ["video.SRT"] },
              ],
            };
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );
    const user = userEvent.setup();
    renderPage();

    await user.click(screen.getByText("From data/ folder"));
    const row = (await screen.findByText("base.mp4")).closest("button")!;
    await user.click(row);

    expect(calls.some((c) => c.includes("/start"))).toBe(false);
    expect(screen.getByText("SELECTED")).toBeTruthy();

    await user.click(screen.getByText("START RECONSTRUCTION"));
    expect(calls.some((c) => c === "POST /api/data-videos/base.mp4/start")).toBe(true);
  });

  it("refuses to start a video-only server dataset until a GPS log is attached", async () => {
    // GPS is required: a dataset the picker reports no GPS for cannot launch,
    // and attaching the log is what unlocks it.
    const calls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL, init?: RequestInit) => {
        const u = String(url);
        calls.push(`${init?.method ?? "GET"} ${u}`);
        const body = u.includes("/start")
          ? { run_id: "job123" }
          : { videos: [{ name: "video_only.mp4", duration_sec: 10, size_bytes: 3_100_000, gps_sources: [] }] };
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );
    const user = userEvent.setup();
    renderPage();

    await user.click(screen.getByText("From data/ folder"));
    await user.click((await screen.findByText("video_only.mp4")).closest("button")!);

    const start = screen.getByText("START RECONSTRUCTION").closest("button") as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    expect(screen.getByText(/No GPS source/)).toBeTruthy();
    expect(calls.some((c) => c.includes("/start"))).toBe(false);

    const attach = screen.getByText(/Attach telemetry log/);
    const input = attach.querySelector("input")!;
    await user.upload(input, new File(["timestamp,lat,lon,alt\n"], "flight.csv", { type: "text/csv" }));

    expect((screen.getByText("START RECONSTRUCTION").closest("button") as HTMLButtonElement).disabled).toBe(false);
    await user.click(screen.getByText("START RECONSTRUCTION"));
    await waitFor(() => expect(calls.some((c) => c.includes("/start"))).toBe(true));
  });

  it("refuses to start an upload until a GPS log is attached", async () => {
    probeMock.mockImplementation(() =>
      Promise.resolve({ duration_sec: 45, width: 3840, height: 2160, aspect: 16 / 9 }));
    motionMock.mockImplementation((() => Promise.resolve(healthyMotion)) as never);
    const user = userEvent.setup();
    renderPage();

    await user.upload(screen.getByLabelText("Choose a drone video file"), aVideoFile());
    await screen.findByLabelText("Selected footage preview");

    const start = screen.getByText("START RECONSTRUCTION").closest("button") as HTMLButtonElement;
    expect(start.disabled).toBe(true);

    const attach = screen.getByText(/Attach telemetry log/);
    await user.upload(
      attach.querySelector("input")!,
      new File(["timestamp,lat,lon,alt\n"], "flight.csv", { type: "text/csv" }),
    );
    expect((screen.getByText("START RECONSTRUCTION").closest("button") as HTMLButtonElement).disabled).toBe(false);
  });

  it("never requires a LiDAR reference — it only says whether accuracy is measurable", async () => {
    // LiDAR is held-out ground truth, not an input: its absence must not gate a
    // run, it must only be stated (otherwise a dataset without a tile could
    // never be reconstructed at all).
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL) => {
        const u = String(url);
        const body = u.includes("/start")
          ? { run_id: "job123" }
          : {
              videos: [
                { name: "tiled.mp4", duration_sec: 10, size_bytes: 3_100_000, gps_sources: ["poses.csv"], lidar_sources: ["lidar.las"] },
                { name: "untiled.mp4", duration_sec: 10, size_bytes: 3_100_000, gps_sources: ["poses.csv"] },
              ],
            };
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );
    const user = userEvent.setup();
    renderPage();

    await user.click(screen.getByText("From data/ folder"));
    await user.click((await screen.findByText("tiled.mp4")).closest("button")!);
    expect(await screen.findByText(/lidar\.las on the server/)).toBeTruthy();
    const start = () =>
      screen.getByText("START RECONSTRUCTION").closest("button") as HTMLButtonElement;
    expect(start().disabled).toBe(false);

    await user.click(screen.getByText("untiled.mp4").closest("button")!);
    expect(await screen.findByText(/None found beside this video/)).toBeTruthy();
    expect(screen.getByText(/absolute accuracy stays NOT MEASURED/)).toBeTruthy();
    // still launchable — the tile is optional, its absence only removes a claim
    expect(start().disabled).toBe(false);
  });

  it("rejects a non-LiDAR file as a LiDAR reference, then accepts a .las", async () => {
    probeMock.mockImplementation(() =>
      Promise.resolve({ duration_sec: 45, width: 3840, height: 2160, aspect: 16 / 9 }));
    motionMock.mockImplementation((() => Promise.resolve(healthyMotion)) as never);
    const user = userEvent.setup();
    renderPage();

    await user.upload(screen.getByLabelText("Choose a drone video file"), aVideoFile());
    await screen.findByLabelText("Selected footage preview");

    const lidarInput = screen
      .getByText(/Attach LiDAR reference/)
      .querySelector("input")! as HTMLInputElement;
    // The file dialog itself offers only tiles; the handler re-checks because
    // drag-and-drop and programmatic sets bypass `accept` entirely (which is
    // also why userEvent.upload cannot deliver a .txt here).
    expect(lidarInput.getAttribute("accept")).toBe(".las,.laz,.npz");
    fireEvent.change(lidarInput, {
      target: { files: [new File(["not a tile"], "notes.txt", { type: "text/plain" })] },
    });
    await waitFor(() =>
      expect(document.querySelector(".nm-error")?.textContent).toMatch(
        /must be a \.las, \.laz, \.npz file/,
      ),
    );

    // re-query: the rejection re-rendered the card, so the old node is stale
    fireEvent.change(
      screen.getByText(/Attach LiDAR reference/).querySelector("input")!,
      { target: { files: [new File(["x"], "survey_tile.las")] } },
    );
    await waitFor(() => {
      expect(document.body.textContent).toMatch(/survey_tile\.las/);
      expect(document.body.textContent).toMatch(/Held out from reconstruction/);
    });
  });

  it("accepts a pre-baked .npz reference (returns or height grid)", async () => {
    probeMock.mockImplementation(() =>
      Promise.resolve({ duration_sec: 45, width: 3840, height: 2160, aspect: 16 / 9 }));
    motionMock.mockImplementation((() => Promise.resolve(healthyMotion)) as never);
    const user = userEvent.setup();
    renderPage();

    await user.upload(screen.getByLabelText("Choose a drone video file"), aVideoFile());
    await screen.findByLabelText("Selected footage preview");

    // The page does not inspect the archive: whether it carries `points` or a
    // height grid is the backend's call, and both are valid references.
    fireEvent.change(
      screen.getByText(/Attach LiDAR reference/).querySelector("input")!,
      { target: { files: [new File(["x"], "baked_reference.npz")] } },
    );
    await waitFor(() => {
      expect(document.body.textContent).toMatch(/baked_reference\.npz/);
      expect(document.body.textContent).toMatch(/Held out from reconstruction/);
    });
    expect(document.querySelector(".nm-error")).toBeNull();
  });

  it("keeps overlap unmeasured when every pair is undecodable instead of scoring garbage", async () => {
    probeMock.mockImplementation(() =>
      Promise.resolve({ duration_sec: 45, width: 3840, height: 2160, aspect: 16 / 9 }));
    motionMock.mockImplementation(() =>
      Promise.reject(new Error("Only 0/24 frame pairs decodable — the browser cannot decode this footage for probing.")));
    const user = userEvent.setup();
    renderPage();

    const input = screen.getByLabelText("Choose a drone video file");
    await user.upload(input, aVideoFile("black_frames.mp4"));

    expect(await screen.findByText(/overlap not measured/i)).toBeTruthy();
    expect(screen.queryByText(/Measured from/i)).toBeNull();
  });
});
