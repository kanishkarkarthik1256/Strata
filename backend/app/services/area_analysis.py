"""Site analysis — how much area a run actually represents, and where it is.

Everything here is measured from the run's own artifacts; nothing is inferred
from the request, and nothing is invented when a measurement is impossible.
The output is a single JSON payload (persisted at
``analysis/area_analysis.json`` and recomputed when its inputs change) that the
Analysis page renders directly.

Frame honesty is the first concern. A number is only called metres when the
run has a metric frame: either the georef stage's local ENU outputs, or the
``georef/alignment.json`` transform that produced them. Otherwise the same
geometry is reported in *reconstruction units* with ``metric: false`` and the
UI must not label it in metres — an estimated-scale SfM cloud is internally
consistent but has no absolute scale.

Three area numbers are reported because they answer different questions:

* ``area_m2``          cells the reconstruction actually occupies (density
                       supported) — "how much ground did we model";
* ``outline_area_m2``  the convex hull of those cells — "how big is the site
                       the model sits on", including the gaps between passes;
* ``cells_per_m2`` etc. density facts so the two can be read sensibly.

Accuracy is reported as what it is: INTERNAL consistency (sparse↔dense
disagreement per cell), never absolute accuracy, which requires an independent
reference this pipeline does not have.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from app.logging_config import get_logger
from app.services.georeferencing import enu_to_wgs84

log = get_logger("drone_recon.services.area_analysis")

_ARTIFACT_REL = Path("analysis") / "area_analysis.json"

#: Occupancy grid: a cell counts as modelled when it holds at least this many
#: points. One point per cell would let isolated stragglers inflate the area.
_MIN_POINTS_PER_CELL = 3
#: Default site grid resolution (metres). Fine enough for a 1 km site to keep
#: its shape, coarse enough that the grid stays small.
_DEFAULT_CELL_M = 1.0
#: Error-grid cells target this many correspondence points each; the size is
#: then rounded up to a "nice" value so the heat map is readable.
_TARGET_POINTS_PER_ERROR_CELL = 4
#: A cell's median is only reported with this many correspondences behind it.
#: One or two stray sparse points would otherwise turn an isolated outlier into
#: a "measured" cell (observed: a 1-point cell medians at 348 m) and dominate a
#: map whose real p50 is under a metre.
_MIN_POINTS_PER_ERROR_CELL = 3
_NICE_CELLS = (2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0)
#: Candidates in order of preference; the first readable wins. A metric-ENU
#: artifact needs no transform, so it is always preferred over one that does.
_SURFACE_CANDIDATES = (
    "georef/dense_model_enu.ply",
    "georef/combined_model_enu.ply",
    "dense/dense_model.ply",
    "mesh/mesh_full.ply",
    "mesh/mesh.ply",
    "sparse_model.ply",
)

#: Input files whose mtime+size invalidate the cached payload.
_FINGERPRINT_INPUTS = (
    "georef/dense_model_enu.ply",
    "georef/combined_model_enu.ply",
    "georef/alignment.json",
    "georef/gps_report.json",
    "georef/gps_track.csv",
    "dense/dense_model.ply",
    "mesh/mesh_full.ply",
    "mesh/mesh.ply",
    "sparse_model.ply",
    "sparse_dense_error.ply",
    "dense_report.json",
    "poses.json",
    "validation/validation_report.json",
)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def _fingerprint(workspace: Path) -> dict:
    out: dict = {}
    for rel in _FINGERPRINT_INPUTS:
        path = workspace / rel
        try:
            st = path.stat()
            out[rel] = [int(st.st_size), int(st.st_mtime)]
        except OSError:
            out[rel] = None
    return out


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _load_points(path: Path) -> np.ndarray | None:
    """Read a PLY point cloud as (N, 3) float64, or None when unreadable."""
    from app.services.pointcloud import read_ply

    try:
        cloud = read_ply(path)
        pts = np.asarray(cloud.xyz, dtype=np.float64)
        return pts if len(pts) else None
    except Exception as exc:
        log.warning("analysis_cloud_read_failed", path=str(path), error=str(exc))
        return None


def _frame_of(workspace: Path) -> tuple[np.ndarray | None, dict]:
    """The reconstruction→ENU transform and the CRS record, when available.

    ``georef/alignment.json`` carries the similarity the georef stage applied;
    it is the single source of truth for which frame a run's cloudy artifacts
    live in. The anchor comes from ``georef/gps_report.json`` (the CRS the
    ENU coordinates are expressed against).
    """
    alignment = _read_json(workspace / "georef" / "alignment.json") or {}
    crs = (_read_json(workspace / "georef" / "gps_report.json") or {}).get("crs") or {}
    anchor = crs.get("anchor_wgs84")
    matrix = alignment.get("matrix")
    M = None
    if isinstance(matrix, list) and len(matrix) == 4:
        M = np.asarray(matrix, dtype=np.float64)
    return M, {
        "anchor_wgs84": anchor,
        "alignment_scale": alignment.get("scale"),
        "alignment_applied_to": alignment.get("applied_to"),
    }


def _surface(workspace: Path, M: np.ndarray | None) -> tuple[np.ndarray | None, str, bool]:
    """(points, source rel-path, already_in_enu) for the best available surface."""
    for rel in _SURFACE_CANDIDATES:
        path = workspace / rel
        if not path.is_file():
            continue
        pts = _load_points(path)
        if pts is None or len(pts) < 500:
            continue
        in_enu = "georef/" in rel
        if not in_enu and M is not None:
            pts = (M @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
        return pts, rel, in_enu
    return None, "", False


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _occupancy(pts: np.ndarray, cell_m: float) -> dict:
    """XY occupancy grid: which cells hold enough points to count as modelled."""
    xy = pts[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    nx, ny = np.maximum(np.ceil((hi - lo) / cell_m).astype(int), 1)
    ix = np.clip(((xy[:, 0] - lo[0]) / cell_m).astype(np.int64), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - lo[1]) / cell_m).astype(np.int64), 0, ny - 1)
    counts = np.zeros(nx * ny, dtype=np.int64)
    np.add.at(counts, iy * nx + ix, 1)
    occupied = counts >= _MIN_POINTS_PER_CELL
    any_point = counts > 0
    cell_area = cell_m * cell_m
    return {
        "origin_enu": lo,
        "nx": int(nx),
        "ny": int(ny),
        "cell_size_m": float(cell_m),
        "counts": counts,
        "occupied": occupied,
        "area_m2": float(occupied.sum()) * cell_area,
        "any_point_area_m2": float(any_point.sum()) * cell_area,
        "occupied_cells": int(occupied.sum()),
        "extent_cells": int(nx * ny),
    }


def _coarse_occupancy(grid: dict, max_cells: int = 4000) -> dict:
    """The occupancy grid re-binned to a transport-friendly resolution.

    The convex hull is a coarse description of a flight's site (it spans the
    gaps between passes), so the map also needs the real modelled shape. This
    is that layer: one entry per occupied coarse cell.
    """
    cell_m = float(grid["cell_size_m"])
    nx, ny = grid["nx"], grid["ny"]
    bbox_area = (nx * cell_m) * (ny * cell_m)
    coarse = max(_nice_cell(bbox_area / max_cells), cell_m)
    factor = max(int(round(coarse / cell_m)), 1)
    counts = grid["counts"].reshape(ny, nx)
    # Pad so the reshape is exact, then sum each factor×factor block.
    pad_y = (-ny) % factor
    pad_x = (-nx) % factor
    if pad_y or pad_x:
        counts = np.pad(counts, ((0, pad_y), (0, pad_x)))
    blocks = counts.reshape(counts.shape[0] // factor, factor,
                            counts.shape[1] // factor, factor).sum(axis=(1, 3))
    by, bx = np.nonzero(blocks >= _MIN_POINTS_PER_CELL)
    return {
        "cell_size_m": float(cell_m * factor),
        "origin_enu": [round(float(v), 3) for v in grid["origin_enu"]],
        "cells": [[int(x), int(y), int(blocks[y, x])] for y, x in zip(by, bx)],
        "cell_fields": ["ix", "iy", "points"],
        "note": "cells holding >= min_points_per_cell points — the modelled footprint",
    }


def _hull_polygon(points_xy: np.ndarray) -> np.ndarray | None:
    """Convex hull outline (counter-clockwise) of a 2D point set."""
    if len(points_xy) < 3:
        return None
    try:
        hull = ConvexHull(points_xy)
    except (QhullError, ValueError):
        return None
    return points_xy[hull.vertices]


def _polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area of a closed 2D polygon (absolute)."""
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _histogram(values: np.ndarray, bins: int = 24) -> dict:
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bin_edges": [round(float(e), 3) for e in edges],
        "counts": [int(c) for c in counts],
        "bin_width": round(float(edges[1] - edges[0]), 3) if len(edges) > 1 else None,
    }


def _nice_cell(target_m2: float) -> float:
    for size in _NICE_CELLS:
        if size * size >= target_m2:
            return size
    return _NICE_CELLS[-1]


def _error_grid(corr_xy: np.ndarray, residual: np.ndarray | None,
                origin: np.ndarray, cell_m: float) -> dict:
    """Per-cell INTERNAL consistency: median sparse↔dense NN error.

    This is the spatial accuracy layer. It measures disagreement between two
    products of this same reconstruction (the sparse cloud and the fused
    dense surface), so it shows where the model is self-inconsistent — it is
    not an absolute accuracy map and is never labelled as one.
    """
    n = len(corr_xy)
    if n == 0:
        return {"available": False, "reason": "no sparse↔dense correspondences recorded"}
    residual = np.ones(n) if residual is None else np.asarray(residual, dtype=np.float64)
    # Cell size aimed at _TARGET_POINTS_PER_ERROR_CELL points per cell.
    span = corr_xy.max(axis=0) - corr_xy.min(axis=0)
    footprint = float(max(span[0], 1.0) * max(span[1], 1.0))
    cell_m = max(cell_m, _nice_cell(footprint * _TARGET_POINTS_PER_ERROR_CELL / n))
    ix = np.floor((corr_xy[:, 0] - origin[0]) / cell_m).astype(np.int64)
    iy = np.floor((corr_xy[:, 1] - origin[1]) / cell_m).astype(np.int64)
    keys, inverse = np.unique(np.stack([ix, iy], axis=1), axis=0, return_inverse=True)
    cells: list[list] = []
    dropped_low_support = 0
    for k, (cx, cy) in enumerate(keys):
        sel = inverse == k
        count = int(sel.sum())
        if count < _MIN_POINTS_PER_ERROR_CELL:
            dropped_low_support += 1
            continue
        vals = residual[sel]
        cells.append([int(cx), int(cy), round(float(np.median(vals)), 3),
                      count, round(float(np.percentile(vals, 95)), 3)])
    medians = np.array([c[2] for c in cells], dtype=np.float64)
    return {
        "available": True,
        "cell_size_m": float(cell_m),
        "min_points_per_cell": _MIN_POINTS_PER_ERROR_CELL,
        "origin_enu": [round(float(v), 3) for v in origin],
        "cells": cells,
        "measured_cells": len(cells),
        "dropped_low_support": dropped_low_support,
        "correspondences": int(n),
        "min_median_m": round(float(medians.min()), 3),
        "max_median_m": round(float(medians.max()), 3),
        "p50_median_m": round(float(np.median(medians)), 3),
        "p95_median_m": round(float(np.percentile(medians, 95)), 3),
        "p99_median_m": round(float(np.percentile(medians, 99)), 3),
        "measure": "per-cell median sparse↔dense NN disagreement",
        "note": "INTERNAL consistency between the sparse cloud and the fused dense "
                "surface — not absolute accuracy, which needs an independent reference",
        "cell_fields": ["ix", "iy", "median_m", "points", "p95_m"],
    }


def _flight(workspace: Path, anchor: dict | None, ground_up: float | None) -> dict:
    """Flight facts from the georef record (single owner of track analysis)."""
    report = _read_json(workspace / "georef" / "gps_report.json") or {}
    quality = report.get("gps_quality") or {}
    out: dict = {"available": bool(quality) or (workspace / "georef" / "gps_track.csv").is_file()}
    for key in ("points", "path_length_m", "mean_speed_m_s", "max_speed_m_s",
                "altitude_std_m", "discontinuities", "drift_m", "gps_score", "grade"):
        if quality.get(key) is not None:
            out[key] = quality[key]
    track_path = workspace / "georef" / "gps_track.csv"
    if track_path.is_file():
        try:
            import csv

            with track_path.open() as fh:
                rows = list(csv.DictReader(fh))
            lat = np.array([float(r["lat"]) for r in rows])
            lon = np.array([float(r["lon"]) for r in rows])
            up = np.array([float(r["up_m"]) for r in rows])
            out["track_points"] = len(rows)
            # Decimate for transport; the map only needs the shape.
            step = max(1, len(rows) // 400)
            out["track_wgs84"] = [[round(float(a), 7), round(float(b), 7)]
                                  for a, b in zip(lat[::step], lon[::step])]
            out["track_up_min_m"] = round(float(up.min()), 3)
            out["track_up_max_m"] = round(float(up.max()), 3)
            if ground_up is not None:
                agl = up - ground_up
                out["agl_min_m"] = round(float(agl.min()), 3)
                out["agl_max_m"] = round(float(agl.max()), 3)
                out["agl_mean_m"] = round(float(agl.mean()), 3)
        except (OSError, ValueError, KeyError) as exc:
            out["track_error"] = str(exc)
    # Ground sampling distance from the actual geometry: scene depth per pixel
    # is AGL / focal_px, so no camera datasheet is needed.
    poses = _read_json(workspace / "poses.json") or {}
    frames = poses.get("frames") or []
    focals = [f.get("K", [[None]])[0][0] for f in frames if isinstance(f, dict)]
    focals = [float(f) for f in focals if isinstance(f, (int, float)) and f > 1]
    if focals:
        out["cameras"] = len(frames)
        out["focal_px_median"] = round(float(np.median(focals)), 2)
        if out.get("agl_mean_m"):
            out["gsd_m_per_px"] = round(float(out["agl_mean_m"]) / float(np.median(focals)), 4)
            out["gsd_method"] = "mean AGL / median focal length (px), measured per run"
    out["anchor_wgs84"] = anchor
    return out


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------


def analyze(workspace: Path, run_id: str, *, cell_m: float = _DEFAULT_CELL_M) -> dict:
    """Measure the site a run represents. Pure read — never writes artifacts."""
    M, frame = _frame_of(workspace)
    anchor = frame.get("anchor_wgs84")
    pts, source, in_enu = _surface(workspace, M)
    notes: list[str] = []

    metric = bool(anchor) and (in_enu or M is not None)
    units = "meters" if metric else "reconstruction_units"
    payload: dict = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frame": {
            "kind": "local_enu" if metric else "reconstruction",
            "metric": metric,
            "units": units,
            "anchor_wgs84": anchor if metric else None,
            "alignment_scale": frame.get("alignment_scale"),
            "source": source or None,
        },
        "sources": {},
    }
    if pts is None:
        payload["available"] = False
        payload["reason"] = (
            "no readable surface artifact (looked for "
            + ", ".join(_SURFACE_CANDIDATES) + ")"
        )
        return payload

    xyz = pts
    payload["sources"]["surface"] = source
    payload["sources"]["surface_points"] = int(len(xyz))
    if not metric:
        notes.append(
            "No metric frame for this run (no georef anchor/alignment), so distances are "
            "reported in reconstruction units — the reconstruction's absolute scale is "
            "estimated, not measured."
        )

    # ---- footprint / area -------------------------------------------------
    grid = _occupancy(xyz, cell_m)
    origin = grid["origin_enu"]
    xs = origin[0] + (np.arange(grid["nx"]) + 0.5) * cell_m
    ys = origin[1] + (np.arange(grid["ny"]) + 0.5) * cell_m
    occ_mask = grid["occupied"].reshape(grid["ny"], grid["nx"])
    oy, ox = np.nonzero(occ_mask)
    cell_centres = np.column_stack([xs[ox], ys[oy]]) if len(ox) else np.zeros((0, 2))
    polygon = _hull_polygon(cell_centres)
    hull_area = _polygon_area(polygon) if polygon is not None else None
    centroid_enu = cell_centres.mean(axis=0) if len(cell_centres) else xyz[:, :2].mean(axis=0)

    wgs_outline = None
    centroid_wgs84 = None
    if metric and polygon is not None:
        ll = enu_to_wgs84(polygon[:, 0], polygon[:, 1], np.full(len(polygon), xyz[:, 2].min()),
                          float(anchor["lat"]), float(anchor["lon"]), float(anchor.get("alt") or 0.0))
        wgs_outline = [[round(float(a), 7), round(float(b), 7)] for a, b in zip(ll[:, 0], ll[:, 1])]
        cll = enu_to_wgs84([centroid_enu[0]], [centroid_enu[1]], [float(xyz[:, 2].min())],
                           float(anchor["lat"]), float(anchor["lon"]), float(anchor.get("alt") or 0.0))
        centroid_wgs84 = {"lat": round(float(cll[0, 0]), 7), "lon": round(float(cll[0, 1]), 7)}

    bbox_min = xyz[:, :2].min(axis=0)
    bbox_max = xyz[:, :2].max(axis=0)
    payload["site"] = {
        "units": units,
        "cell_size_m": float(cell_m),
        "min_points_per_cell": _MIN_POINTS_PER_CELL,
        "area_m2": round(grid["area_m2"], 1),
        "area_ha": round(grid["area_m2"] / 10_000.0, 4),
        "area_km2": round(grid["area_m2"] / 1_000_000.0, 6),
        "area_method": "occupied cells (density-supported) — the ground actually modelled",
        "outline_area_m2": round(hull_area, 1) if hull_area else None,
        "outline_area_km2": round(hull_area / 1_000_000.0, 6) if hull_area else None,
        "outline_method": "convex hull of occupied cells — the site the model covers, "
                          "including unmodelled gaps",
        "occupied_cells": grid["occupied_cells"],
        "extent_cells": grid["extent_cells"],
        "any_point_area_m2": round(grid["any_point_area_m2"], 1),
        "bbox": {
            "min": [round(float(v), 2) for v in bbox_min],
            "max": [round(float(v), 2) for v in bbox_max],
            "size": [round(float(bbox_max[0] - bbox_min[0]), 2),
                     round(float(bbox_max[1] - bbox_min[1]), 2)],
        },
        "centroid": {"enu": [round(float(v), 2) for v in centroid_enu], "wgs84": centroid_wgs84},
        "outline_enu": ([[round(float(a), 2), round(float(b), 2)] for a, b in polygon]
                        if polygon is not None else None),
        "outline_wgs84": wgs_outline,
        "footprint_cells": _coarse_occupancy(grid),
    }

    # ---- elevation --------------------------------------------------------
    up = xyz[:, 2]
    lo_i, hi_i = int(np.argmin(up)), int(np.argmax(up))
    p_lo = float(np.percentile(up, 0.5))
    p_hi = float(np.percentile(up, 99.5))

    def _at(i: int) -> dict:
        row = xyz[i]
        out = {"enu": [round(float(v), 2) for v in row]}
        if metric:
            ll = enu_to_wgs84([row[0]], [row[1]], [row[2]], float(anchor["lat"]),
                              float(anchor["lon"]), float(anchor.get("alt") or 0.0))
            out["wgs84"] = {"lat": round(float(ll[0, 0]), 7), "lon": round(float(ll[0, 1]), 7)}
            out["alt_m"] = round(float(ll[0, 2]), 2)
        return out

    payload["elevation"] = {
        "units": units,
        "surface": source,
        "lowest": _at(lo_i),
        "highest": _at(hi_i),
        "min": round(float(up.min()), 2),
        "max": round(float(up.max()), 2),
        "relief": round(float(up.max() - up.min()), 2),
        "mean": round(float(up.mean()), 2),
        "median": round(float(np.median(up)), 2),
        # Extremes of a fused cloud can be single-point outliers; the robust
        # pair is what "highest/lowest structure" actually means.
        "robust_min": round(p_lo, 2),
        "robust_max": round(p_hi, 2),
        "robust_relief": round(p_hi - p_lo, 2),
        "robust_percentiles": "0.5 / 99.5",
        "histogram": _histogram(up),
        "datum": ("ENU up relative to the first GPS fix" if metric
                  else "reconstruction up axis (no metric datum)"),
    }

    # ---- density + flight -------------------------------------------------
    dense_report = _read_json(workspace / "dense_report.json") or {}
    quality = dense_report.get("quality") or {}
    payload["density"] = {
        "points": int(len(xyz)),
        "points_per_m2": round(float(len(xyz)) / grid["area_m2"], 3) if grid["area_m2"] else None,
        "mean_spacing_m": quality.get("mean_spacing"),
        "coverage_percent": quality.get("coverage_percent"),
        "occlusion_percent": quality.get("occlusion_percent"),
        "sources": ["surface cloud", "dense_report.json"],
    }
    payload["flight"] = _flight(workspace, anchor if metric else None, float(p_lo) if metric else None)

    # ---- internal consistency (the accuracy layer) ------------------------
    consist = (dense_report.get("stages") or {}).get("sparse_dense_consistency") or {}
    payload["consistency"] = {
        "correspondences": consist.get("correspondences"),
        "median_m": consist.get("median_m"),
        "p95_m": consist.get("p95_m"),
        "within_3m_pct": consist.get("within_3m_pct"),
        "screened_median_m": consist.get("screened_median_m"),
        "screened_p95_m": consist.get("screened_p95_m"),
        "measure": "sparse↔dense nearest-neighbour agreement (internal consistency)",
        "source": "dense_report.json → stages.sparse_dense_consistency",
    }
    err_path = workspace / "sparse_dense_error.ply"
    if err_path.is_file():
        from app.services.pointcloud import read_ply

        try:
            corr = read_ply(err_path)
            corr_xyz = np.asarray(corr.xyz, dtype=np.float64)
            if not in_enu and M is not None:
                corr_xyz = (M @ np.hstack([corr_xyz, np.ones((len(corr_xyz), 1))]).T).T[:, :3]
            payload["accuracy_map"] = _error_grid(
                corr_xyz[:, :2], corr.residual, origin, cell_m)
            payload["sources"]["accuracy_map"] = "sparse_dense_error.ply (residual = NN error, m)"
        except Exception as exc:
            payload["accuracy_map"] = {"available": False, "reason": str(exc)}
    else:
        payload["accuracy_map"] = {"available": False, "reason": "sparse_dense_error.ply absent"}

    # ---- metric validation (read through the single owner, embedded for the
    #      page so one fetch carries the full analysis) ---------------------
    from app.services.metric_validation import load_report

    report = load_report(workspace)
    payload["metric_validation"] = (
        {
            "validation_kind": report.get("validation_kind"),
            "certification_status": report.get("certification_status"),
            "certification_reason": report.get("certification_reason"),
            "internal_validation": report.get("internal_validation"),
        }
        if report
        else None
    )

    payload["available"] = True
    payload["notes"] = notes
    payload["generated_from"] = "run artifacts (read-only)"
    log.info("area_analysis_complete", run_id=run_id, area_m2=payload["site"]["area_m2"],
             metric=metric, points=int(len(xyz)))
    return payload


# ---------------------------------------------------------------------------
# cached entry point
# ---------------------------------------------------------------------------


def load_or_compute(workspace: Path, run_id: str, *, force: bool = False) -> dict:
    """Serve the cached analysis when its inputs are unchanged, else recompute.

    The cache is keyed on an explicit fingerprint of every input file rather
    than on existence, so a re-run of any stage invalidates it.
    """
    path = workspace / _ARTIFACT_REL
    fingerprint = _fingerprint(workspace)
    if not force:
        cached = _read_json(path)
        if cached and cached.get("input_fingerprint") == fingerprint:
            return cached
    started = time.perf_counter()
    payload = analyze(workspace, run_id)
    payload["input_fingerprint"] = fingerprint
    payload["compute_ms"] = round((time.perf_counter() - started) * 1000, 2)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    except OSError as exc:  # a cache miss is not a failure
        log.warning("area_analysis_cache_write_failed", error=str(exc))
    return payload
