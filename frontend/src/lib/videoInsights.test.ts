/**
 * Client-side footage fitness assessment — the New Mission page must render
 * only what these rules measure, and every threshold must trace to the
 * pipeline's real behavior (frame extraction budget, 518 px inference width).
 */

import { describe, expect, it } from "vitest";
import {
  candidateFrameCount,
  durationInsight,
  overlapInsight,
  percentile,
  readinessScore,
  resolutionInsight,
  sharpnessInsight,
  uploadEstimate,
  type ClientVideoStats,
  type MotionProbe,
} from "./videoInsights";

const stats = (over: Partial<ClientVideoStats> = {}): ClientVideoStats => ({
  duration_sec: 45,
  width: 1920,
  height: 1080,
  aspect: 16 / 9,
  ...over,
});

const probe = (over: Partial<MotionProbe> = {}): MotionProbe => ({
  // Default mirrors the MEASURED healthy reference (airport1, through this
  // same norm-based metric at 1 s cadence: 0.882 static, 0.337 over its
  // 16.6 s sampling span). The earlier 0.62 "healthy anchor" was never
  // measured and put every real clip in the failure band.
  staticP50: 0.882,
  staticP90: 0.919,
  pairs: 24,
  sharpP50: 20,
  crossP50: 0.337,
  crossSpanS: 16.6,
  crossPairs: 12,
  ...over,
});

describe("durationInsight (frame budget ≈ 1 fps candidates)", () => {
  it("rates long footage good and reports the candidate budget", () => {
    const ins = durationInsight(60);
    expect(ins.level).toBe("good");
    expect(ins.detail).toContain("60 candidate frames");
  });

  it("rates 10-30 s footage fair", () => {
    expect(durationInsight(15).level).toBe("fair");
  });

  it("rates under 10 s poor", () => {
    const ins = durationInsight(5);
    expect(ins.level).toBe("poor");
    expect(ins.detail).toMatch(/10 s/);
  });
});

describe("resolutionInsight (depth model infers at 518 px width)", () => {
  it("rates 1080p+ good", () => {
    expect(resolutionInsight(1920, 1080).level).toBe("good");
    expect(resolutionInsight(3840, 2160).level).toBe("good");
  });

  it("rates 720p fair and names the 518 px limit", () => {
    const ins = resolutionInsight(1280, 720);
    expect(ins.level).toBe("fair");
    expect(ins.detail).toContain("518");
  });

  it("rates sub-720p poor", () => {
    expect(resolutionInsight(640, 480).level).toBe("poor");
  });
});

describe("candidateFrameCount", () => {
  it("matches the ~1 fps extraction budget", () => {
    expect(candidateFrameCount(30)).toBe(30);
    expect(candidateFrameCount(0)).toBe(0);
  });
});

describe("uploadEstimate", () => {
  it("scales with file size at ~4 MB/s", () => {
    expect(uploadEstimate(4 * 1024 * 1024)).toBe("a few seconds");
    expect(uploadEstimate(400 * 1024 * 1024)).toMatch(/about \d+ s|min/);
    expect(uploadEstimate(3 * 1024 * 1024 * 1024)).toMatch(/min/);
  });
});

describe("overlapInsight (bands anchored to this pipeline's measured footage)", () => {
  it("rates the measured healthy reference (airport1: 0.882) good", () => {
    const ins = overlapInsight(probe());
    expect(ins.level).toBe("good");
    expect(ins.detail).toContain("88%");
    expect(ins.detail).toContain("34%"); // the across-clip drop, named
  });

  it("rates every other clip this pipeline reconstructed good too", () => {
    // Measured: london 0.730, DJI_0501 0.768, base 0.891 (all reconstructed).
    for (const s of [0.73, 0.768, 0.891]) {
      expect(overlapInsight(probe({ staticP50: s, staticP90: s + 0.02 })).level).toBe("good");
    }
  });

  it("rates both measured failure anchors (0.22 shattered, 0.23 gate-refused) poor", () => {
    expect(overlapInsight(probe({ staticP50: 0.22, staticP90: 0.26 })).level).toBe("poor");
    expect(overlapInsight(probe({ staticP50: 0.23, staticP90: 0.26 })).level).toBe("poor");
  });

  it("rates the unmeasured in-between band (0.35–0.5) fair", () => {
    expect(overlapInsight(probe({ staticP50: 0.4, staticP90: 0.5 })).level).toBe("fair");
  });

  it("flags a camera that never changes view (both gaps identical)", () => {
    // The duplicate-view failure: 1 s and whole-span similarity agree.
    const ins = overlapInsight(probe({ staticP50: 0.93, staticP90: 0.96, crossP50: 0.95 }));
    expect(ins.level).toBe("poor");
    expect(ins.detail).toMatch(/duplicate|one view/i);
  });

  it("does NOT blame a translating camera for high 1 s similarity", () => {
    // The exact regression: 0.882 static at 1 s over footage that is not
    // stationary — the old bands called this "mostly stationary" + veto.
    const ins = overlapInsight(probe({ staticP50: 0.93, staticP90: 0.96, crossP50: 0.42 }));
    expect(ins.level).toBe("good");
  });

  it("says coverage is unconfirmed when the clip is too short to measure it", () => {
    const ins = overlapInsight(probe({ staticP50: 0.8, staticP90: 0.85, crossP50: null, crossSpanS: 0 }));
    expect(ins.level).toBe("fair");
    expect(ins.label).toMatch(/unconfirmed/i);
  });

  it("calls near-identical short-clip frames what they are", () => {
    const ins = overlapInsight(probe({ staticP50: 0.99, staticP90: 0.995, crossP50: null, crossSpanS: 0 }));
    expect(ins.level).toBe("fair");
    expect(ins.label).toMatch(/near-identical/i);
  });

  it("is honest when the probe never ran", () => {
    expect(overlapInsight(null).level).toBe("fair");
    expect(overlapInsight(null).label).toMatch(/not measured/i);
  });
});

describe("sharpnessInsight (gradient-energy noise floor)", () => {
  it("only surfaces a real outlier", () => {
    expect(sharpnessInsight(probe())).toBeNull();
    const ins = sharpnessInsight(probe({ sharpP50: 2.1 }));
    expect(ins?.level).toBe("poor");
    expect(ins?.detail).toMatch(/noise floor/);
  });
});

describe("readinessScore", () => {
  it("scores healthy footage ready", () => {
    const r = readinessScore(stats(), probe());
    expect(r.score).toBe(100);
    expect(r.verdict).toBe("ready");
    expect(r.components.map((c) => c.key)).toEqual(["duration", "resolution", "overlap"]);
  });

  it("scores the measured shattered signature unlikely", () => {
    // estrel: healthy duration/resolution but overlap in the measured
    // failure band — the poor-overlap veto applies regardless of score.
    const r = readinessScore(stats(), probe({ staticP50: 0.22, staticP90: 0.26 }));
    expect(r.score).toBe(60);
    expect(r.verdict).toBe("unlikely");
  });

  it("scores a camera that never changes view unlikely", () => {
    const r = readinessScore(stats(), probe({ staticP50: 0.93, staticP90: 0.96, crossP50: 0.95 }));
    expect(r.verdict).toBe("unlikely");
  });

  it("does not collapse to one score for every video (the reported bug)", () => {
    // Before: any 1080p+ clip >= 30 s with static >= 0.65 measured exactly
    // 60 / unlikely. Now the score responds to duration, resolution and the
    // real overlap signal, across the measured fleet.
    const scores = [
      readinessScore(stats(), probe()).score,                       // airport1: 100
      readinessScore(stats({ duration_sec: 12 }), probe()).score,    // short: 85
      readinessScore(stats({ width: 1280, height: 720 }), probe()).score, // 85
      readinessScore(stats({ duration_sec: 5, width: 640, height: 360 }), probe()).score, // 40
      readinessScore(stats(), probe({ staticP50: 0.22, staticP90: 0.26 })).score, // 60
    ];
    expect(new Set(scores).size).toBeGreaterThan(1);
    expect(scores[0]).toBe(100);
    expect(scores[1]).toBe(85);
    expect(scores[2]).toBe(85);
    expect(scores[3]).toBe(40);
    expect(scores[4]).toBe(60);
  });

  it("normalizes over measured components only", () => {
    // No motion probe: 60/60 from duration+resolution must still be 100.
    const r = readinessScore(stats(), null);
    expect(r.score).toBe(100);
    const comp = r.components.find((c) => c.key === "overlap");
    expect(comp?.level).toBe("unmeasured");
  });

  it("punishes the too-short + fast-motion combination", () => {
    const r = readinessScore(stats({ duration_sec: 5 }), probe({ staticP50: 0.05 }));
    expect(r.verdict).toBe("unlikely");
  });

  it("keeps a short clip with confirmed coverage out of the failure bucket", () => {
    // A 12 s clip cannot be "unlikely" just for being short when its motion
    // measured healthy — the earlier constant-60 behaviour did exactly that.
    const r = readinessScore(stats({ duration_sec: 12 }), probe({ staticP50: 0.73, crossP50: null, crossSpanS: 0 }));
    expect(r.verdict).not.toBe("unlikely");
  });
});

describe("percentile", () => {
  it("clamps to the measured range", () => {
    expect(percentile([1, 2, 3], 0)).toBe(1);
    expect(percentile([1, 2, 3], 1)).toBe(3);
    expect(percentile([1, 2, 3], 0.5)).toBe(2);
    expect(percentile([], 0.5)).toBeNaN();
  });
});
