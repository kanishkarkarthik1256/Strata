import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import Missions from "./Missions";
import { RunsProvider } from "../hooks/useRuns";

/**
 * The Missions page search box filters the run table by run_id, dataset and
 * mission name; it composes with the status filter tabs, and its empty state
 * names the query instead of pretending nothing was searched.
 */

const RUNS = {
  runs: [
    {
      run_id: "airport_53_8f0e90",
      dataset: "video.mp4",
      mission: "airport_53",
      status: "completed",
      dense_points: 290306,
    },
    {
      run_id: "furnerhem_6_34be62",
      dataset: "video.mp4",
      mission: "furnerhem6",
      status: "completed",
      dense_points: 2227996,
    },
    {
      run_id: "flight_to_tower_7511dc",
      dataset: "video.MP4",
      mission: "flight_to_tower",
      status: "failed",
      dense_points: null,
    },
  ],
};

function renderPage() {
  return render(
    <MemoryRouter>
      <RunsProvider>
        <Missions />
      </RunsProvider>
    </MemoryRouter>,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Missions search", () => {
  it("renders all runs and a search box", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url === "/api/runs"
          ? new Response(JSON.stringify(RUNS), { status: 200 })
          : new Response("{}", { status: 200 }),
      ),
    );
    renderPage();
    expect(await screen.findByText("airport_53_8f0e90")).toBeTruthy();
    expect(screen.getByText("furnerhem_6_34be62")).toBeTruthy();
    expect(screen.getByLabelText("Search missions")).toBeTruthy();
  });

  it("filters by run_id substring", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url === "/api/runs"
          ? new Response(JSON.stringify(RUNS), { status: 200 })
          : new Response("{}", { status: 200 }),
      ),
    );
    const user = userEvent.setup();
    renderPage();
    await screen.findByText("airport_53_8f0e90");

    await user.type(screen.getByLabelText("Search missions"), "furnerhem");
    expect(screen.queryByText("airport_53_8f0e90")).toBeNull();
    expect(screen.getByText("furnerhem_6_34be62")).toBeTruthy();
  });

  it("filters by dataset and mission names, case-insensitively", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url === "/api/runs"
          ? new Response(JSON.stringify(RUNS), { status: 200 })
          : new Response("{}", { status: 200 }),
      ),
    );
    const user = userEvent.setup();
    renderPage();
    await screen.findByText("airport_53_8f0e90");

    // mission name, different case
    await user.type(screen.getByLabelText("Search missions"), "TOWER");
    expect(screen.queryByText("airport_53_8f0e90")).toBeNull();
    expect(screen.getByText("flight_to_tower_7511dc")).toBeTruthy();

    await user.clear(screen.getByLabelText("Search missions"));
    await user.type(screen.getByLabelText("Search missions"), "video.MP4");
    // dataset matches all three
    expect(screen.getByText("airport_53_8f0e90")).toBeTruthy();
    expect(screen.getByText("furnerhem_6_34be62")).toBeTruthy();
  });

  it("empty state names the query", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url === "/api/runs"
          ? new Response(JSON.stringify(RUNS), { status: 200 })
          : new Response("{}", { status: 200 }),
      ),
    );
    const user = userEvent.setup();
    renderPage();
    await screen.findByText("airport_53_8f0e90");

    await user.type(screen.getByLabelText("Search missions"), "zzz-no-match");
    expect(screen.getByText(/No runs found matching "zzz-no-match"/)).toBeTruthy();
  });
});
