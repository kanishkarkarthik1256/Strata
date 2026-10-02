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

export default function ViewerStats({
  points,
  pointsLabel = "Points",
  meshFaces,
  cameras,
  gps,
  reprojectionErrorPx,
  sparseDense,
  runId,
}: {
  points: number | null;
  /** What the live count actually is: point samples, or the shipped GLB's
   *  vertices (a textured LOD splits vertices once per UV seam, so its vertex
   *  count is not a point count). */
  pointsLabel?: string;
  meshFaces?: number | null;
  cameras: number | null;
  gps: number | null;
  reprojectionErrorPx: number | null;
  sparseDense?: SparseDenseStats | null;
  runId: string | null;
}) {
  const fmt = (n: number | null): string =>
    n === null ? "Not available" : n >= 1_000_000 ? `${(n / 1_000_000).toFixed(2)}M` : n.toLocaleString();
  const fmtM = (n: number | null | undefined): string =>
    n === null || n === undefined ? "Not available" : `${n.toFixed(2)} m`;
  const sd = sparseDense ?? null;
  return (
    <div className="viewer-stats">
      {meshFaces ? <div className="viewer-stat">Mesh Faces: {fmt(meshFaces)}</div> : null}
      <div className="viewer-stat">{pointsLabel}: {fmt(points)}</div>
      <div className="viewer-stat">Cameras: {cameras ?? "Not available"}</div>
      <div className="viewer-stat">GPS: {gps ?? "Not available"}</div>
      <div className="viewer-stat">Reproj: {reprojectionErrorPx !== null ? `${reprojectionErrorPx.toFixed(2)} px` : "Not available"}</div>
      {sd ? (
        <>
          <div className="viewer-stat">Sparse↔Dense NN: {fmtM(sd.median_m)} median / {fmtM(sd.p95_m)} p95</div>
          {sd.screened ? (
            <div className="viewer-stat" title={sd.screened.note ?? ""}>
              Consistent scaffold: {fmtM(sd.screened.median_m)} median / {fmtM(sd.screened.p95_m)} p95 ({sd.screened.correspondences ?? "—"} pts, {sd.screened.dropped ?? 0} beyond depth budget)
            </div>
          ) : null}
        </>
      ) : null}
      {runId ? <div className="viewer-stat run">Run: {runId}</div> : null}
    </div>
  );
}