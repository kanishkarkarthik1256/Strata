import { useCallback, useEffect, useRef, useState } from "react";
import {
  durationInsight,
  probeVideo,
  probeMotion,
  readinessScore,
  resolutionInsight,
  sharpnessInsight,
  type ClientVideoStats,
  type MotionProbe,
  type Readiness,
  type ReconInsight,
} from "../lib/videoInsights";

/** What the browser measured about the picked file, already interpreted. */
export interface FootageProbeResult {
  /** Object URL for the local preview; revoked when the file changes. */
  previewUrl: string | null;
  stats: ClientVideoStats | null;
  /** Honest message when the browser could not decode the file. */
  previewError: string | null;
  motion: MotionProbe | null;
  motionProgress: { done: number; total: number } | null;
  /** Duration/resolution insights derived from `stats`. */
  insights: ReconInsight[];
  /** Sharpness insight derived from `motion`; null until measured. */
  blurInsight: ReconInsight | null;
  /** Reconstruction-readiness score; null until the file is measurable. */
  readiness: Readiness | null;
  /** Measure a file (null clears everything). */
  probe: (file: File | null) => void;
  clear: () => void;
}

/**
 * Everything measured about the picked file without uploading it: the local
 * preview URL, the browser decoder's stats, and the overlap/sharpness probe.
 *
 * The page owns *which* file is selected; this hook owns what measuring it
 * produced. A probe runs off the render path and never blocks the form.
 */
export function useFootageProbe(): FootageProbeResult {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [stats, setStats] = useState<ClientVideoStats | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [motion, setMotion] = useState<MotionProbe | null>(null);
  const [motionProgress, setMotionProgress] = useState<{ done: number; total: number } | null>(null);

  // Only the newest probe may publish results: picking a second file while
  // the first is still measuring must never leave file A's measured numbers
  // displayed against file B.
  const generation = useRef(0);
  const urlRef = useRef<string | null>(null);

  const revoke = useCallback(() => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
  }, []);

  const clear = useCallback(() => {
    generation.current += 1;
    revoke();
    setPreviewUrl(null);
    setStats(null);
    setPreviewError(null);
    setMotion(null);
    setMotionProgress(null);
  }, [revoke]);

  const probe = useCallback(
    (file: File | null) => {
      if (!file) {
        clear();
        return;
      }
      generation.current += 1;
      const mine = generation.current;
      revoke();
      setStats(null);
      setPreviewError(null);
      setMotion(null);
      setMotionProgress(null);
      const url = URL.createObjectURL(file);
      urlRef.current = url;
      setPreviewUrl(url);

      probeVideo(file)
        .then((s) => {
          if (generation.current === mine) setStats(s);
        })
        .catch(() => {
          if (generation.current === mine) {
            setPreviewError(
              "The browser cannot decode this file for preview — it may still upload and reconstruct fine (the server decodes via ffmpeg).",
            );
          }
        });

      // Overlap/blur probe over the same local file — runs while the user
      // fills the mission form; never blocks preview or upload.
      probeMotion(file, (done, total) => {
        if (generation.current === mine) setMotionProgress({ done, total });
      })
        .then((m) => {
          if (generation.current === mine) setMotion(m);
        })
        .catch(() => {
          if (generation.current === mine) setMotion(null);
        })
        .finally(() => {
          if (generation.current === mine) setMotionProgress(null);
        });
    },
    [clear, revoke],
  );

  // The preview URL is a browser resource, not page state: release it when
  // the page goes away.
  useEffect(() => () => revoke(), [revoke]);

  return {
    previewUrl,
    stats,
    previewError,
    motion,
    motionProgress,
    insights: stats
      ? [durationInsight(stats.duration_sec), resolutionInsight(stats.width, stats.height)]
      : [],
    blurInsight: sharpnessInsight(motion),
    readiness: stats ? readinessScore(stats, motion) : null,
    probe,
    clear,
  };
}
