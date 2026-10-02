import { afterEach, describe, expect, it, vi } from "vitest";
import { api, clearToken, setToken } from "./api";
import { ApiError, friendlyMessage } from "./errors";
import { mockFetchFailure, mockFetchJson, mockFetchNetworkError } from "../test/setup";

const RUN = "shitan_ms1_20260909_131251";

afterEach(() => {
  clearToken();
  vi.unstubAllGlobals();
  try {
    localStorage.clear();
  } catch {
    /* noop */
  }
});

describe("api client contracts", () => {
  it("attaches the bearer token when present", async () => {
    setToken("tok123");
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await api.getRuns();
    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("Authorization")).toBe("Bearer tok123");
  });

  it("maps 401 to ApiError with status 401", async () => {
    mockFetchFailure(401);
    const err = await api.getMissions().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(401);
  });

  it("maps 422 to ApiError with status 422", async () => {
    mockFetchFailure(422, [{ loc: ["query", "run_id"], msg: "Field required" }]);
    const err = await api.getRun(RUN).catch((e: unknown) => e);
    expect((err as ApiError).status).toBe(422);
  });

  it("upload errors keep the backend diagnostic in detail so friendlyMessage shows it", async () => {
    // Regression: the XHR path dropped the third ApiError argument, so the
    // backend's precise message (e.g. "Telemetry CSV missing required
    // column(s): …") was discarded and users saw a generic 400 message.
    const RealXHR = globalThis.XMLHttpRequest;
    const BODY = JSON.stringify({
      error: "Telemetry CSV missing required column(s): timestamp, latitude, longitude, altitude",
      status_code: 400,
      detail: null,
    });
    vi.stubGlobal(
      "XMLHttpRequest",
      class FakeXHR {
        status = 0;
        responseText = "";
        upload = {};
        onload: (() => void) | null = null;
        open() {}
        setRequestHeader() {}
        send() {
          this.status = 400;
          this.responseText = BODY;
          this.onload?.();
        }
      },
    );
    const err = await api.uploadVideo(new File(["v"], "video.mp4")).catch((e: unknown) => e);
    globalThis.XMLHttpRequest = RealXHR;
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(400);
    expect((err as ApiError).message).toContain("Telemetry CSV missing required column(s)");
    expect((err as ApiError).detail).toContain("Telemetry CSV missing required column(s)");
    expect(friendlyMessage(err)).toContain("Telemetry CSV missing required column(s)");
  });

  it("maps 404 to ApiError with status 404", async () => {
    mockFetchFailure(404);
    const err = await api.getRunManifest(RUN).catch((e: unknown) => e);
    expect((err as ApiError).status).toBe(404);
  });

  it("maps network failure to ApiError with status null", async () => {
    mockFetchNetworkError();
    const err = await api.getRuns().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBeNull();
  });

  it("builds artifact URLs with the run id and relative artifact path", () => {
    expect(api.artifactUrl(RUN, "dense/dense_model.ply")).toBe(
      `/api/runs/${RUN}/artifact/dense/dense_model.ply`,
    );
  });
});