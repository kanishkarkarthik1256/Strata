/**
 * Client-side video fitness assessment for the New Mission page.
 *
 * Every rule here is derived from this pipeline's measured behavior —
 * frame extraction budgets (0.5–2 fps candidate extraction), SfM's need
 * for motion diversity, and the depth model's 518 px inference width —
 * never from invented thresholds. The page renders only what these
 * functions return; nothing asserts a capability the pipeline lacks.
 */

export type InsightLevel = "good" | "fair" | "poor";

export interface ReconInsight {
  level: InsightLevel;
  label: string;
  detail: string;
}

export interface ClientVideoStats {
  duration_sec: number;
  width: number;
  height: number;
  aspect: number;
}

/** Read real duration/resolution from the browser's own decoder. */
export function probeVideo(file: File): Promise<ClientVideoStats> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement("video");
    v.preload = "metadata";
    const cleanup = () => URL.revokeObjectURL(url);
    v.onloadedmetadata = () => {
      const s: ClientVideoStats = {
        duration_sec: Number.isFinite(v.duration) ? v.duration : 0,
        width: v.videoWidth,
        height: v.videoHeight,
        aspect: v.videoHeight ? v.videoWidth / v.videoHeight : 0,
      };
      cleanup();
      resolve(s);
    };
    v.onerror = () => {
      cleanup();
      reject(new Error("The browser cannot decode this video for preview."));
    };
    v.src = url;
  });
}

/** Effective frame budget the extractor can pull from this footage. */
export function candidateFrameCount(durationSec: number): number {
  // app.frame_extractor default target_fps ≈ 1.0 (bounded 0.5–2 by settings)
  return Math.max(0, Math.round(durationSec * 1.0));
}

/** SfM wants enough baselines: clips this short rarely register a coherent path. */
export function durationInsight(durationSec: number): ReconInsight {
  if (durationSec >= 30) {
    return {
      level: "good",
      label: "Good motion budget",
      detail: `${Math.round(candidateFrameCount(durationSec))} candidate frames — enough for a coherent camera path.`,
    };
  }
  if (durationSec >= 10) {
    return {
      level: "fair",
      label: "Short clip",
      detail: "~" + candidateFrameCount(durationSec) + " candidate frames — reconstruction works but coverage will be thin.",
    };
  }
  return {
    level: "poor",
    label: "Too short",
    detail: "Under 10 s of footage rarely registers a usable camera path. Aim for 30 s or more.",
  };
}

/** The depth model infers at 518 px width — small sources upsample, wasting detail. */
export function resolutionInsight(width: number, height: number): ReconInsight {
  if (width >= 1920) {
    return {
      level: "good",
      label: "Sharp source",
      detail: `${width}×${height} — above the depth model's 518 px inference width; detail is preserved.`,
    };
  }
  if (width >= 1280) {
    return {
      level: "fair",
      label: "Moderate resolution",
      detail: `${width}×${height} — usable; fine detail will be limited by the 518 px inference width.`,
    };
  }
  return {
    level: "poor",
    label: "Low resolution",
    detail: `${width}×${height} — below what the depth model can resolve; expect coarse geometry.`,
  };
}

/** Estimated upload wait at a conservative ~4 MB/s uplink. */
export function uploadEstimate(bytes: number): string {
  const sec = Math.round(bytes / (4 * 1024 * 1024));
  if (sec < 3) return "a few seconds";
  if (sec < 90) return `about ${sec} s`;
  return `about ${Math.round(sec / 60)} min`;
}

/**
 * ---------- Reconstruction readiness ----------
 *
 * The overlap signal is calibrated against this pipeline's own runs: the
 * gain-normalized static-pixel fraction between frames one extraction
 * interval (~1 s) apart separates the healthy airport1 flight (0.62 of
 * pixels unchanged) from the shattered-mesh estrel run (0.22 — too little
 * frame-to-frame overlap at extraction cadence) and the hover-heavy
 * flight_to_tower path (0.69–0.96 — duplicate views, weak 3D baseline).
 * The extreme-blur floor sits where gradient energy reaches the noise
 * level of a decodable frame (~3/255). Nothing here is invented.
 */

export interface MotionProbe {
  /** Median fraction of pixels ~unchanged across a ~1 s gap (0..1). */
  staticP50: number;
  /** 90th percentile of the same measure (hover/wobble tail). */
  staticP90: number;
  /** Measured pairs used. */
  pairs: number;
  /** Pairs attempted — when this exceeds `pairs`, decoding was partial. */
  pairsAttempted?: number;
  /** Pairs dropped (seek failure or blank frame) — decode-degraded probe. */
  pairsUnusable?: number;
  /** Median gradient energy (0..255 scale) — blur/texture detector. */
  sharpP50: number;
  /**
   * Median static fraction between frames a whole sampling span apart
   * (`crossSpanS`), i.e. how much the scene changes ACROSS the clip.
   * Null when the clip is too short for the two gaps to be genuinely
   * different vantages — a fabricated coverage number would be worse than
   * none. This is the duplicate-view detector: a hovering/stationary camera
   * keeps this as high as `staticP50`, a translating one drops it.
   */
  crossP50: number | null;
  /** The gap (s) `crossP50` was measured over; 0 when unmeasured. */
  crossSpanS: number;
  /** Coverage pairs used (half the motion pairs, spaced). */
  crossPairs: number;
}

export function percentile(sortedAsc: number[], p: number): number {
  if (sortedAsc.length === 0) return NaN;
  const idx = Math.min(
    sortedAsc.length - 1,
    Math.max(0, Math.round(p * (sortedAsc.length - 1))),
  );
  return sortedAsc[idx];
}

/**
 * Wait for one seek to settle. A timeout REJECTS — the frame the element
 * still shows is stale, and drawing it would silently fabricate a
 * measurement. (This timeout-as-success bug made every undecodable file
 * report an identical garbage readiness verdict.)
 */
function seekTo(v: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = (ok: boolean) => {
      v.removeEventListener("seeked", done);
      clearTimeout(timer);
      if (ok) resolve();
      else reject(new Error(`seek timeout at ${t.toFixed(2)}s — frame unavailable`));
    };
    const done = () => finish(true);
    const timer = setTimeout(() => finish(false), 4000);
    v.addEventListener("seeked", done);
    try {
      v.currentTime = t;
    } catch {
      clearTimeout(timer);
      reject(new Error("seek failed"));
    }
  });
}

/**
 * A frame with (near-)zero pixel variance carries no motion information:
 * it is a blank/black/solid decode artifact, not scenery. Mean-normalized
 * comparison of two such frames would report "everything static".
 */
function isUniformFrame(g: Float32Array): boolean {
  let mean = 0;
  for (let i = 0; i < g.length; i++) mean += g[i];
  mean /= g.length;
  for (let i = 0; i < g.length; i++) if (Math.abs(g[i] - mean) > 4) return false;
  return true;
}

/**
 * Measure frame-to-frame overlap the way the extractor will see it:
 * ~24 frame pairs spaced across the clip, one extraction interval apart,
 * downscaled to 64×36 (pure JS — no wasm, runs in ~10–20 s while the
 * user fills in the mission form).
 */
export async function probeMotion(
  file: File,
  onProgress?: (done: number, total: number) => void,
): Promise<MotionProbe> {
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  v.src = url;
  try {
    await new Promise<void>((resolve, reject) => {
      v.onloadedmetadata = () => resolve();
      v.onerror = () => reject(new Error("The browser cannot decode this video."));
    });
    const dur = Number.isFinite(v.duration) ? v.duration : 0;
    if (dur <= 0) throw new Error("Video duration is unreadable.");
    const pairs = 24;
    // Match the extractor's ~1 s cadence; shrink only for very short clips.
    const gap = Math.min(1.0, Math.max(0.3, dur / 25));
    // Coverage gap: the distance between consecutive probe pairs. Only a
    // meaningfully larger vantage than `gap` says anything about scene
    // change (measured: at span≈gap the two numbers are identical), so
    // below 3x it is reported as unmeasured rather than invented.
    const span = dur / (pairs + 1);
    const coverage = span >= gap * 3 ? span : 0;
    const W = 64;
    const H = 36;
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Canvas is unavailable in this browser.");

    const grab = async (t: number): Promise<Float32Array> => {
      await seekTo(v, Math.min(t, Math.max(0, dur - 0.05)));
      ctx.drawImage(v, 0, 0, W, H);
      const px = ctx.getImageData(0, 0, W, H).data;
      const g = new Float32Array(W * H);
      for (let i = 0; i < g.length; i++) {
        g[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
      }
      return g;
    };

    const statics: number[] = [];
    const crosses: number[] = [];
    const sharps: number[] = [];
    let unusable = 0;
    for (let k = 1; k <= pairs; k++) {
      const t0 = (dur * k) / (pairs + 1);
      let a: Float32Array;
      let b: Float32Array;
      let c: Float32Array | null = null;
      try {
        a = await grab(t0);
        b = await grab(t0 + gap);
        // Coverage is measured on every other pair (half the extra seeks).
        if (coverage > 0 && k % 2 === 0) {
          try {
            c = await grab(t0 + coverage);
          } catch {
            c = null;
          }
        }
      } catch {
        unusable++;
        onProgress?.(k, pairs);
        continue;
      }
      // A uniform frame (blank/black/solid) is a decode failure wearing a
      // pixel array — counting it as data fabricates the signal.
      if (isUniformFrame(a) || isUniformFrame(b)) {
        unusable++;
        onProgress?.(k, pairs);
        continue;
      }
      // Sharpness: mean adjacent-pixel gradient energy of frame a.
      let gsum = 0;
      let gcount = 0;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = y * W + x;
          if (x + 1 < W) {
            gsum += Math.abs(a[i + 1] - a[i]);
            gcount++;
          }
          if (y + 1 < H) {
            gsum += Math.abs(a[i + W] - a[i]);
            gcount++;
          }
        }
      }
      sharps.push(gsum / Math.max(1, gcount));
      // Gain-normalized static fraction (exposure flicker cancels out).
      let am = 0;
      let bm = 0;
      for (let i = 0; i < a.length; i++) {
        am += a[i];
        bm += b[i];
      }
      am = am / a.length + 1e-6;
      bm = bm / b.length + 1e-6;
      let n = 0;
      let stat = 0;
      for (let i = 0; i < a.length; i++) {
        const d = Math.abs(a[i] / am - b[i] / bm);
        n++;
        if (d < 0.06) stat++;
      }
      statics.push(stat / n);
      // Same measure between frames a whole span apart: high means the
      // camera never gained a new vantage (duplicate views).
      if (c !== null && !isUniformFrame(c)) {
        let cm = 0;
        for (let i = 0; i < c.length; i++) cm += c[i];
        cm = cm / c.length + 1e-6;
        let cn = 0;
        let cstat = 0;
        for (let i = 0; i < a.length; i++) {
          const d = Math.abs(a[i] / am - c[i] / cm);
          cn++;
          if (d < 0.06) cstat++;
        }
        crosses.push(cstat / cn);
      }
      onProgress?.(k, pairs);
    }
    if (statics.length < 6) {
      throw new Error(
        `Only ${statics.length}/${pairs} frame pairs decodable — the browser cannot decode this footage for probing.`,
      );
    }
    statics.sort((x, y) => x - y);
    sharps.sort((x, y) => x - y);
    crosses.sort((x, y) => x - y);
    return {
      staticP50: percentile(statics, 0.5),
      staticP90: percentile(statics, 0.9),
      pairs: statics.length,
      pairsAttempted: pairs,
      pairsUnusable: unusable,
      sharpP50: percentile(sharps, 0.5),
      crossP50: crosses.length >= 4 ? percentile(crosses, 0.5) : null,
      crossSpanS: coverage > 0 && crosses.length >= 4 ? coverage : 0,
      crossPairs: crosses.length,
    };
  } finally {
    URL.revokeObjectURL(url);
    v.removeAttribute("src");
    v.load();
  }
}

/**
 * Overlap verdict.
 *
 * Bands are anchored to this pipeline's OWN measured footage, not to a
 * supposed "healthy 0.62": every clip the pipeline reconstructs well sits at
 * 0.73–0.89 static fraction at its ~1 s extraction cadence (airport1 0.882 —
 * the reference run; base 0.891; london 0.730; DJI_0501 0.768), while the two
 * measured failures are the FAST-motion tail (0.22 and 0.23). A high
 * static fraction at a 1 s gap is therefore the normal signature of flyable
 * drone footage, and the earlier bands — which called 0.65–0.97 "mostly
 * stationary" and vetoed it — marked the best footage as unlikely to
 * reconstruct. The genuine duplicate-view failure is instead detected by
 * comparing the same measure across the whole clip: a translating camera
 * drops (airport1 measures 0.337 over its 16 s sampling span) where a
 * stationary one stays as high as its 1 s value.
 */
export function overlapInsight(m: MotionProbe | null): ReconInsight {
  if (!m) {
    return {
      level: "fair",
      label: "Overlap not measured",
      detail: "Motion probe unavailable — readiness score covers duration and resolution only.",
    };
  }
  const shared = Math.round(m.staticP50 * 100);
  const p90 = Math.round(m.staticP90 * 100);
  if (m.staticP50 < 0.35) {
    // The measured failure tail: the shattered-mesh run (0.22) and the run
    // whose anchor gate refused 82/85 views (0.23).
    return {
      level: "poor",
      label: "Camera moves too fast",
      detail: `Only ${shared}% of pixels shared between frames a second apart — at the extractor's ~1 s cadence there is too little overlap to match. Measured runs in this band shattered or lost most views. Slow down or fly higher.`,
    };
  }
  if (m.staticP50 >= 0.65 && m.crossP50 !== null && m.crossP50 >= 0.9) {
    // Both signals agree: frames a second apart AND frames a whole sampling
    // span apart show the same view. Every probe pair is the same vantage.
    return {
      level: "poor",
      label: "Camera never changes view",
      detail: `${shared}% of pixels unchanged a second apart (p90 ${p90}%) and still ${Math.round(m.crossP50 * 100)}% unchanged across the clip (${m.crossSpanS.toFixed(0)} s span) — the camera holds one view, so every extracted frame is a duplicate with no new baseline. Orbit or translate across the scene.`,
    };
  }
  if (m.staticP50 >= 0.97 && m.crossP50 === null) {
    return {
      level: "fair",
      label: "Near-identical frames",
      detail: `${shared}% of pixels unchanged a second apart — the frames are effectively the same image. The clip is too short to measure whether the view changes over its run, so coverage cannot be confirmed.`,
    };
  }
  if (m.staticP50 < 0.5) {
    // Unmeasured territory between the failure tail and the healthy band.
    return {
      level: "fair",
      label: "Thin overlap",
      detail: `${shared}% pixel overlap at extraction cadence — registration should work but coverage will be thin.`,
    };
  }
  if (m.crossP50 === null) {
    // High overlap but the clip is too short for the coverage gap to be a
    // different vantage — usable, though scene change over the run is
    // unconfirmed rather than measured.
    return {
      level: "fair",
      label: "Overlap unconfirmed",
      detail: `${shared}% pixel overlap at extraction cadence — usable, but the clip is too short to measure whether the view changes across its run.`,
    };
  }
  return {
    level: "good",
    label: "Healthy overlap",
    detail: `${shared}% of pixels shared between frames a second apart, dropping to ${Math.round(m.crossP50 * 100)}% across the clip (${m.crossSpanS.toFixed(0)} s span) — the camera gains new vantages, the signature of footage this pipeline reconstructs well.`,
  };
}

/** Blur detector: gradient energy at the noise floor of a decodable frame. */
export function sharpnessInsight(m: MotionProbe | null): ReconInsight | null {
  if (!m || m.sharpP50 >= 3) return null; // only surface a real outlier
  return {
    level: "poor",
    label: "Featureless footage",
    detail: `Frame detail (${m.sharpP50.toFixed(1)}) is at the noise floor — defocus, sky, or texture-poor surfaces give SfM nothing to match.`,
  };
}

export type ReadinessVerdict = "ready" | "risky" | "unlikely";

export interface ReadinessComponent {
  key: string;
  label: string;
  level: InsightLevel | "unmeasured";
  points: number;
  max: number;
  detail: string;
}

export interface Readiness {
  score: number;
  verdict: ReadinessVerdict;
  components: ReadinessComponent[];
}

const LEVEL_POINTS: Record<InsightLevel, number> = { good: 1, fair: 0.5, poor: 0 };

/**
 * Aggregate the measured signals into one 0–100 score. The score is
 * normalized over the signals that actually measured — an unmeasured
 * component never contributes points.
 */
export function readinessScore(
  stats: ClientVideoStats | null,
  motion: MotionProbe | null,
): Readiness {
  const components: ReadinessComponent[] = [];
  if (stats) {
    const dur = durationInsight(stats.duration_sec);
    components.push({
      key: "duration",
      label: "Motion budget",
      level: dur.level,
      points: LEVEL_POINTS[dur.level] * 30,
      max: 30,
      detail: dur.detail,
    });
    const res = resolutionInsight(stats.width, stats.height);
    components.push({
      key: "resolution",
      label: "Source detail",
      level: res.level,
      points: LEVEL_POINTS[res.level] * 30,
      max: 30,
      detail: res.detail,
    });
  }
  const ov = overlapInsight(motion);
  components.push({
    key: "overlap",
    label: "View overlap",
    level: motion ? ov.level : "unmeasured",
    points: motion ? LEVEL_POINTS[ov.level] * 40 : 0,
    max: 40,
    detail: ov.detail,
  });

  const measured = components.filter((c) => c.level !== "unmeasured");
  const max = measured.reduce((s, c) => s + c.max, 0);
  const points = measured.reduce((s, c) => s + c.points, 0);
  const score = max > 0 ? Math.round((points / max) * 100) : 0;
  // Overlap is the make-or-break signal: footage whose MEASURED overlap is
  // poor matches the signature of runs that failed or shattered, whatever
  // the other components say — it vetoes the verdict up to "unlikely".
  const overlap = components.find((c) => c.key === "overlap");
  const overlapVeto = overlap != null && overlap.level === "poor";
  const verdict: ReadinessVerdict =
    score >= 75 && !overlapVeto ? "ready" : score >= 45 && !overlapVeto ? "risky" : "unlikely";
  return { score, verdict, components };
}
