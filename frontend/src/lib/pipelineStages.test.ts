/**
 * Stage vocabulary contract — the Processing page renders these strings
 * verbatim, so nothing here may claim an outcome.
 *
 * Live defect this pins: georef's fallback read "skipped without telemetry",
 * so Video_Mission_9ab1aa — georef status `completed`, ENU artifacts on disk,
 * 116/116 telemetry frames matched — displayed "skipped" in the stage list
 * directly above its own "GPS telemetry: Available" panel.
 */

import { describe, expect, it } from "vitest";
import { STAGES, overallProgress } from "./pipelineStages";
import type { PipelineStatusResponse } from "./types";

describe("STAGES", () => {
  it("keeps the backend's stage ids", () => {
    expect(STAGES.map((s) => s.id)).toEqual([
      "frames",
      "sparse",
      "depth",
      "dense",
      "georef",
    ]);
  });

  it("names an engine, never a status", () => {
    const statusWords = /\b(skipped|failed|pending|completed|not run|unavailable)\b/i;
    for (const stage of STAGES) {
      expect(stage.fallback, `${stage.id} fallback claims a status`).not.toMatch(statusWords);
    }
  });
});

describe("overallProgress", () => {
  const status = (stages: Record<string, { status: string }>) =>
    ({ stages } as unknown as PipelineStatusResponse);

  it("counts only known stages and never exceeds the vocabulary", () => {
    expect(overallProgress(null)).toBeNull();
    expect(overallProgress(status({}))).toBeNull();
    expect(
      overallProgress(status({ frames: { status: "completed" }, sparse: { status: "completed" } })),
    ).toBeCloseTo(2 / STAGES.length);
  });

  it("treats a skipped stage as done rather than leaving progress stuck", () => {
    const all = Object.fromEntries(
      STAGES.map((s) => [s.id, { status: s.id === "georef" ? "skipped" : "completed" }]),
    );
    expect(overallProgress(status(all))).toBe(1);
  });
});
