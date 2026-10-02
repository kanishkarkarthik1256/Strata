"""Ground-truth DSM accuracy — the absolute-accuracy measure for runs whose
dataset ships a reference digital surface model.

Honesty contract (the audit-bench discipline, with the registration lesson
this thread measured the hard way):

- The DSM tile's bounds are **tile-local meters** (the dataset's own export
  convention; UTM anchors live in dataset meta and do NOT place the run's
  frame — the fabricated-telemetry anchor is ~24 km from the tile's UTM
  offset, so anchor-based placement was measured as pure noise: real-scene
  MAE ≈ permutation-null MAE). Horizontal placement is therefore
  **measured, not assumed**: the service co-registers the reconstruction to
  the tile by minimizing the robust spread (MAD) of the height residual
  over a coarse-to-fine 2-D translation search — the 2-D analogue of Nuth
  & Kääb DEM co-registration — and reports the measured offset next to
  the score.
- MAE is reported **after removing the global height offset** between the
  reconstruction frame and the DSM height datum; that offset is reported
  separately — it is a datum convention, not an accuracy claim.
- Coverage (the fraction of sampled reconstruction points inside the DSM
  footprint) is reported next to MAE — a MAE on 20% coverage means
  something different from one on 90%.
- A sanity gate refuses to score when the same geometry, at the same
  measured offset, scores comparably against a **rolled-null DSM** (the
  tile circularly shifted by a random offset larger than the terrain
  correlation length — same marginal heights, same autocorrelation,
  decorrelated placement). Geometry that matches decorrelated terrain as
  well as the real tile has no measurable registration; scoring it would
  manufacture accuracy. Featureless geometry is refused for the same
  reason: a plane registers equally everywhere.
- Runs without a registered reference DSM get ``no_reference`` — the UI
  renders nothing for them rather than an unmeasured claim.

The reference is the run's own **held-out LiDAR input** (LAS/LAZ):
:func:`app.services.lidar.resolve_reference` finds it in the run workspace and
:func:`app.services.lidar.load_grid` turns it into the height grid sampled
below. There is no per-run registry of known-good files — a run either
carries a reference or it reports ``no_reference``, so a stale entry can never
attach one dataset's ground truth to another dataset's reconstruction.

The run's local frame is whatever frame its telemetry placed it in; the
co-registration below measures the placement rather than assuming the tile
and the reconstruction already share one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services import lidar


def _load_dsm(path: Path) -> tuple[np.ndarray, tuple, float]:
    """The reference height grid: a pre-baked ``.npz`` or a raw LAS/LAZ tile."""
    grid = lidar.load_grid(path)
    return grid.height, grid.bounds, grid.gsd


def _sample(
    H: np.ndarray, bounds: tuple, gsd: float, tx: np.ndarray, ty: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear DSM heights under tile-local points; mask of in-footprint."""
    x0, y0, x1, y1 = bounds
    h, w = H.shape
    col = (tx - x0) / gsd
    row = (y1 - ty) / gsd
    ok = (col >= 0) & (col < w - 1) & (row >= 0) & (row < h - 1)
    if not ok.any():
        return np.empty(0), ok
    c0, r0 = col[ok].astype(int), row[ok].astype(int)
    dc, dr = col[ok] - c0, row[ok] - r0
    z = (
        H[r0, c0] * (1 - dc) * (1 - dr)
        + H[r0, c0 + 1] * dc * (1 - dr)
        + H[r0 + 1, c0] * (1 - dc) * dr
        + H[r0 + 1, c0 + 1] * dc * dr
    )
    return z, ok


def _mad(dz: np.ndarray) -> float:
    return float(np.median(np.abs(dz - np.median(dz))))


def coregister(
    H: np.ndarray,
    bounds: tuple,
    gsd: float,
    points: np.ndarray,
    max_points: int = 20_000,
    coarse_m: float = 100.0,
    window_m: float = 3000.0,
    min_coverage_frac: float = 0.6,
) -> tuple[float, float, np.ndarray, float]:
    """Measure horizontal placement: the (dx, dy) translation of *points*
    that minimizes the MAD of the height residual against the tile.

    Coarse-to-fine (100 m over ±3000 m, then 20 m and 5 m local refine).
    Offsets that drop most of the footprint out of the tile are excluded
    from competition: a sliver overlap can fit anything (measured on this
    very dataset — two runs of the same scene "registered" 2.5 km apart
    when slivers were allowed). Only offsets keeping ≥
    ``min_coverage_frac`` × the best achievable coverage compete; the
    returned coverage is the winner's.

    Returns (dx, dy, sampled points used, coverage at the winner).
    """
    rng = np.random.default_rng(0)
    if len(points) > max_points:
        points = points[rng.choice(len(points), max_points, replace=False)]
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    def eval_at(dx: float, dy: float) -> tuple[float, float] | None:
        z, ok = _sample(H, bounds, gsd, X + dx, Y + dy)
        cov = float(ok.mean())
        if ok.sum() < 500:
            return None
        return _mad(Z[ok] - z), cov

    # Coarse sweep first — establishes the best achievable coverage.
    coarse: list[tuple[float, float, float, float]] = []  # mad, cov, dx, dy
    for dx in np.arange(-window_m, window_m + coarse_m / 2, coarse_m):
        for dy in np.arange(-window_m, window_m + coarse_m / 2, coarse_m):
            r = eval_at(dx, dy)
            if r is not None:
                coarse.append((r[0], r[1], float(dx), float(dy)))
    if not coarse:
        return 0.0, 0.0, points, 0.0
    max_cov = max(c[1] for c in coarse)
    floor = min_coverage_frac * max_cov
    eligible = [c for c in coarse if c[1] >= floor]
    best = min(eligible, key=lambda c: c[0])
    best = (best[0], best[1], best[2], best[3])
    for step, span in ((20.0, 120.0), (5.0, 25.0)):
        cur = (float("inf"), best[1], best[2], best[3])
        for dx in np.arange(best[2] - span, best[2] + span + step / 2, step):
            for dy in np.arange(best[3] - span, best[3] + span + step / 2, step):
                r = eval_at(dx, dy)
                if r is not None and r[1] >= floor and r[0] < cur[0]:
                    cur = (r[0], r[1], float(dx), float(dy))
        if not np.isfinite(cur[0]):
            break
        best = cur
    return best[2], best[3], points, best[1]


def dsm_mae(
    points: np.ndarray,
    *,
    dsm_path: Path | None = None,
    H_ext: tuple[np.ndarray, tuple, float] | None = None,
    offset_m: tuple[float, float] = (0.0, 0.0),
    max_points: int = 20_000,
    remove_offset: bool = True,
) -> dict:
    """Score reconstruction points against the tile at a given placement.

    Returns status/mae_m/p95_m/height_datum_offset_m/coverage. Pass
    ``H_ext`` (as returned by :func:`_load_dsm`, or a modified grid for
    null gating) to score against a pre-loaded grid.
    """
    H, bounds, gsd = H_ext if H_ext is not None else _load_dsm(dsm_path)
    rng = np.random.default_rng(0)
    if len(points) > max_points:
        points = points[rng.choice(len(points), max_points, replace=False)]
    dx, dy = offset_m
    z, ok = _sample(H, bounds, gsd, points[:, 0] + dx, points[:, 1] + dy)
    cov = float(ok.mean())
    if ok.sum() < 500:
        return {"status": "insufficient_coverage", "coverage": round(cov, 4)}
    dz = points[ok, 2] - z
    off = float(np.median(dz))
    resid = np.abs(dz - (off if remove_offset else 0.0))
    return {
        "status": "ok",
        "mae_m": round(float(resid.mean()), 3),
        "p95_m": round(float(np.percentile(resid, 95)), 3),
        "height_datum_offset_m": round(off, 2),
        "coverage": round(cov, 4),
    }


def run_dsm_accuracy(ws: Path, run_id: str | None = None) -> dict:
    """Full per-run measurement: held-out LiDAR → mesh vertices →
    co-registration → score at the measured placement → rolled-null gate."""
    reference = lidar.resolve_reference(ws)
    if reference is None:
        return {"status": "no_reference"}
    mesh_ply = ws / "mesh" / "mesh.ply"
    if not mesh_ply.exists():
        return {"status": "no_mesh"}
    try:
        import open3d as o3d

        V = np.asarray(o3d.io.read_triangle_mesh(str(mesh_ply)).vertices)
    except Exception as exc:  # unreadable mesh — honest failure, no score
        return {"status": "mesh_unreadable", "error": str(exc)[:200]}
    if len(V) < 500:
        return {"status": "no_mesh"}

    H, bounds, gsd = _load_dsm(reference)
    dx, dy, sample, cov = coregister(H, bounds, gsd, V)
    real = dsm_mae(sample, H_ext=(H, bounds, gsd), offset_m=(dx, dy))
    if real["status"] != "ok":
        return {**real, "reference": str(reference)}

    out = {
        **real,
        "measured_offset_m": [round(dx, 1), round(dy, 1)],
        "registration_coverage": round(cov, 4),
        "reference": str(reference),
        "reference_kind": "held-out LiDAR (rasterised height grid)",
    }

    # Sanity gate: same geometry, same offset, against rolled-null DSMs —
    # the tile circularly shifted by random offsets of a quarter to half a
    # tile (decorrelated placement, preserved marginal heights and
    # autocorrelation). Five rolls; the MEDIAN null score decides, so no
    # single lucky roll can pass or fail the gate.
    nulls: list[float] = []
    rng = np.random.default_rng(123)
    for k in range(5):
        dr = int(rng.integers(H.shape[0] // 4, H.shape[0] // 2)) * int(rng.choice([-1, 1]))
        dc = int(rng.integers(H.shape[1] // 4, H.shape[1] // 2)) * int(rng.choice([-1, 1]))
        H_null = np.roll(H, (dr, dc), axis=(0, 1))
        null = dsm_mae(sample, H_ext=(H_null, bounds, gsd), offset_m=(dx, dy))
        if null["status"] == "ok":
            nulls.append(null["mae_m"])
    if nulls:
        gate_mae = float(np.median(nulls))
        if real["mae_m"] > gate_mae * 0.8:
            out["status"] = "alignment_gate_failed"
            out["gate_mae_m"] = round(gate_mae, 3)
    return out
