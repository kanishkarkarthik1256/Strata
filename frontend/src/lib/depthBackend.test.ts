import { describe, expect, it } from "vitest";
import { DEPTH_BACKEND_LABELS, depthBackendLabel, depthBackendNote } from "./depthBackend";

/**
 * The shared depth-backend label owner: Processing and Reports must render
 * identical labels from run data, with honest fallbacks. A stereo run's
 * label must never contain "Depth Anything"; a depth_anything label must
 * always carry the "not metric" qualifier.
 */
describe("depthBackend — single label owner", () => {
  it("maps depth_anything to the SfM-aligned, not-metric label", () => {
    expect(depthBackendLabel("depth_anything")).toBe("Depth Anything V2 (SfM-aligned, not metric)");
    expect(depthBackendLabel("depth_anything")).toContain("not metric");
  });

  it("maps stereo to the SGBM label with no Depth Anything mention", () => {
    const label = depthBackendLabel("stereo");
    expect(label).toBe("Stereo SGBM (SfM-scaled)");
    expect(label).not.toContain("Depth Anything");
  });

  it("falls back to unknown for missing or non-string values", () => {
    expect(depthBackendLabel(undefined)).toBe("unknown");
    expect(depthBackendLabel(null)).toBe("unknown");
    expect(depthBackendLabel("")).toBe("unknown");
    expect(depthBackendLabel(42)).toBe("unknown");
  });

  it("passes through an unrecognized backend string verbatim (honest, not a guess)", () => {
    expect(depthBackendLabel("future_backend")).toBe("future_backend");
  });

  it("every mapped label is distinct and the map has no Depth-Anything-on-stereo entry", () => {
    const values = Object.values(DEPTH_BACKEND_LABELS);
    expect(new Set(values).size).toBe(values.length);
    for (const [key, label] of Object.entries(DEPTH_BACKEND_LABELS)) {
      if (key !== "depth_anything") expect(label).not.toContain("Depth Anything");
    }
  });

  it("notes exist only for known backends", () => {
    expect(depthBackendNote("depth_anything")).toContain("1/Z = a·D_raw + b");
    expect(depthBackendNote("stereo")).toContain("Stereo SGBM");
    expect(depthBackendNote(undefined)).toBeNull();
    expect(depthBackendNote("mystery")).toBeNull();
  });
});
