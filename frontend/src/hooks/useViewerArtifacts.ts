import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { friendlyMessage } from "../lib/errors";
import type { PosesJson, RunAnalysis, RunSummary, ViewerAlignment } from "../lib/types";

export interface SparseDenseStats {
  correspondences: number | null;
  median_m: number | null;
  p95_m: number | null;
  rmse_m: number | null;
  within_3m_pct?: number | null;
  screened?: {
    correspondences: number | null;
    dropped: number | null;
    median_m: number | null;
    p95_m: number | null;
    rmse_m: number | null;
    note?: string;
  } | null;
}

export interface ViewerArtifacts {
  run: RunSummary | null;
  poses: PosesJson | null;
  /** Rigid presentation transform (ground → +Y, corridor → +X); null = identity. */
  alignment: ViewerAlignment | null;
  meshUrl: string | null;
  denseUrl: string | null;
  sparseUrl: string | null;
  /** sparse+dense merged in one world frame (combined_model.ply), when present. */
  combinedUrl: string | null;
  meshFaces: number | null;
  meshVertices: number | null;
  points: number | null;
  cameras: number | null;
  gps: number | null;
  reprojectionErrorPx: number | null;
  sparseDense: SparseDenseStats | null;
  /** Measured dense-cloud spacing (m) from the run's analysis; null = unknown. */
  meanSpacingM: number | null;
  loading: boolean;
  error: string | null;
}

/**
 * Resolve the real artifact URLs and metadata for a run. Point/camera/GPS/mesh
 * counts come from the run's actual PLY headers and poses.json — never
 * hardcoded.
 */
export function useViewerArtifacts(runId: string | null): ViewerArtifacts {
  const [run, setRun] = useState<RunSummary | null>(null);
  const [poses, setPoses] = useState<PosesJson | null>(null);
  const [alignment, setAlignment] = useState<ViewerAlignment | null>(null);
  /** Measured dense-cloud point spacing (m) from the run's analysis — the
   *  viewer sizes point sprites from it so zoomed-in clouds stay solid. */
  const [meanSpacingM, setMeanSpacingM] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) {
      setRun(null);
      setPoses(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getRun(runId)
      .then((detail) => {
        if (cancelled) return [null, null, null] as [PosesJson | null, ViewerAlignment | null, RunAnalysis | null];
        setRun(detail);
        // Poses/alignment may be absent for runs without camera localization — that's ok.
        return Promise.all([
          api.getPoses(runId).catch(() => null),
          api.getViewerAlignment(runId).catch(() => null),
          api.getRunAnalysis(runId).catch(() => null),
        ]);
      })
      .then(([p, a, analysis]) => {
        if (cancelled) return;
        setPoses(p ?? null);
        setAlignment(a ?? null);
        setMeanSpacingM(analysis?.density?.mean_spacing_m ?? null);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(friendlyMessage(err));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  // Prefer the viewer LOD (decimated GLB) — the full mesh.ply is an
  // analysis artifact and must never be shipped to the browser.
  const glbArtifact = (run?.artifacts ?? []).find((a) => a.path === "mesh/mesh_viewer.glb");
  const meshArtifact = (run?.artifacts ?? []).find(
    (a) =>
      a.path === "mesh/mesh.ply" ||
      a.path === "mesh/mesh_full.ply" ||
      a.path === "mesh/base_mesh.ply" ||
      a.path === "diagnostics/mesh.ply",
  );
  const hasMesh = Boolean(run?.mesh_faces) || Boolean(meshArtifact) || Boolean(glbArtifact);

  const hasDense = Boolean(run?.dense_points) || (run?.artifacts ?? []).some(
    (a) => a.path === "dense/dense_model.ply",
  );
  const hasSparse =
    Boolean(run?.sparse_points) ||
    (run?.artifacts ?? []).some(
      (a) => a.path === "sparse/sparse_model.ply" || a.path === "sparse_model.ply",
    );
  const hasCombined = (run?.artifacts ?? []).some((a) => a.path === "combined_model.ply");

  return {
    run,
    poses,
    alignment,
    meshUrl: runId && hasMesh ? api.artifactUrl(runId, glbArtifact?.path ?? meshArtifact?.path ?? "mesh/mesh.ply") : null,
    denseUrl: runId && hasDense ? api.artifactUrl(runId, "dense/dense_model.ply") : null,
    sparseUrl:
      runId && hasSparse
        ? api.artifactUrl(runId, (run?.artifacts ?? []).some((a) => a.path === "sparse/sparse_model.ply")
          ? "sparse/sparse_model.ply"
          : "sparse_model.ply")
        : null,
    combinedUrl: runId && hasCombined ? api.artifactUrl(runId, "combined_model.ply") : null,
    meshFaces: run?.mesh_faces ?? null,
    meshVertices: run?.mesh_vertices ?? null,
    points: run?.dense_points ?? run?.sparse_points ?? null,
    cameras: run?.cameras ?? null,
    gps: run?.gps_points ?? null,
    reprojectionErrorPx: run?.mean_reprojection_error_px ?? null,
    sparseDense: run?.sparse_dense ?? null,
    meanSpacingM,
    loading,
    error,
  };
}
