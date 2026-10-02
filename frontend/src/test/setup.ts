import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

/** Install a mock fetch returning the given JSON body with the given status. */
export function mockFetchJson(body: unknown, status = 200): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    ),
  );
}

/** Install a mock fetch that rejects (network failure). */
export function mockFetchNetworkError(): void {
  vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
}

export function mockFetchFailure(status: number, detail?: unknown): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      new Response(JSON.stringify({ detail: detail ?? "err" }), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    ),
  );
}