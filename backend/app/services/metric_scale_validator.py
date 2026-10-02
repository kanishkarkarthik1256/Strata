"""Metric-scale validation — derive a scale factor from GPS vs COLMAP.

The COLMAP reconstruction produces camera centres and 3D points in an
arbitrary coordinate system whose unit length is unknown.  GPS gives us
the *real-world* positions of the same cameras in WGS84.

By aligning the two point sets (Umeyama similarity) we obtain:

* a scale factor  s  such that  metric_distance ≈ s × colmap_distance
* a residual / error describing how well a single global scale explains
  all camera baselines

The result is an *estimate* — not a validated metric reconstruction.
Reported caveats:
- only cameras with GPS are used (typically 10 of 10 for DJI)
- the scale estimate is global (no per-region refinement)
- GPS altitude accuracy is ±5–15 m for consumer drones
- COLMAP reprojection error propagates into the scale estimate
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.logging_config import get_logger
from app.services.georeferencing import align_to_enu, wgs84_to_enu

log = get_logger("drone_recon.services.metric_scale_validator")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScaleReport:
    """Result of the metric-scale validation."""

    scale_factor: float = 1.0
    """colmap_units → metres:  metric_dist = scale_factor * colmap_dist."""

    residual_m: float = 0.0
    """RMS residual (metres) after alignment — lower is better."""

    max_error_m: float = 0.0
    """Worst single-camera-pair residual after alignment."""

    num_cameras: int = 0
    """Cameras with both COLMAP pose and GPS used in the estimation."""

    anchor_lat: float = 0.0
    anchor_lon: float = 0.0

    colmap_extent: float = 0.0
    """Max pairwise distance in COLMAP units."""

    gps_extent_m: float = 0.0
    """Max pairwise distance in real metres (GPS)."""

    status: str = "NOT_VALIDATED"
    """NOT_VALIDATED | ESTIMATED | INSUFFICIENT_DATA"""

    warnings: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "scale_factor": round(self.scale_factor, 6),
            "residual_m": round(self.residual_m, 4),
            "max_error_m": round(self.max_error_m, 4),
            "num_cameras": self.num_cameras,
            "anchor": {"lat": self.anchor_lat, "lon": self.anchor_lon},
            "colmap_extent": round(self.colmap_extent, 4),
            "gps_extent_m": round(self.gps_extent_m, 4),
            "status": self.status,
            "warnings": self.warnings or [],
        }


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------


def _camera_centres(poses: list[dict]) -> np.ndarray:
    """Extract camera centres in COLMAP coordinates from poses.json.

    Convention:  X_world = R @ X_cam + t
    Camera centre:  C_world = -R^T @ t
    Returns (N, 3) float64.
    """
    centres = []
    for f in poses:
        R = np.asarray(f["R"], dtype=np.float64)  # (3, 3)
        t = np.asarray(f["t"], dtype=np.float64)  # (3,)
        centres.append(-R.T @ t)
    return np.asarray(centres, dtype=np.float64)


def _gps_enu(poses: list[dict]) -> np.ndarray:
    """Convert per-frame GPS to local ENU metres around the first GPS fix.

    Returns (N, 3) float64 — [east, north, up].
    """
    lats = np.array([f["gps"]["lat"] for f in poses], dtype=np.float64)
    lons = np.array([f["gps"]["lon"] for f in poses], dtype=np.float64)
    alts = np.array([f["gps"]["alt"] for f in poses], dtype=np.float64)
    return wgs84_to_enu(lats, lons, alts, float(lats[0]), float(lons[0]))


def _pairwise_distances(pts: np.ndarray) -> np.ndarray:
    """Upper-triangle pairwise Euclidean distances (flat array)."""
    n = len(pts)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(pts[i] - pts[j])))
    return np.asarray(dists, dtype=np.float64)


def estimate_scale(
    poses_path: Path,
) -> ScaleReport:
    """Derive the metric scale factor from poses.json (with GPS).

    This is the primary entry-point: give it a poses.json that has GPS
    fields (added by _add_gps_to_poses) and get back a ScaleReport.
    """
    report = ScaleReport(warnings=[])

    with open(poses_path) as fp:
        data = json.load(fp)
    frames = data.get("frames", [])

    # Filter to frames with GPS
    gps_frames = [f for f in frames if f.get("gps")]
    if len(gps_frames) < 2:
        report.status = "INSUFFICIENT_DATA"
        report.warnings.append(
            f"Need ≥2 cameras with GPS to estimate scale; got {len(gps_frames)}"
        )
        return report

    report.num_cameras = len(gps_frames)
    report.anchor_lat = gps_frames[0]["gps"]["lat"]
    report.anchor_lon = gps_frames[0]["gps"]["lon"]

    # Camera centres in COLMAP space
    colmap_pts = _camera_centres(gps_frames)
    # Camera centres in GPS-ENU space (metres)
    gps_pts = _gps_enu(gps_frames)

    # Pairwise extents
    colmap_dists = _pairwise_distances(colmap_pts)
    gps_dists = _pairwise_distances(gps_pts)
    if len(colmap_dists) == 0:
        report.status = "INSUFFICIENT_DATA"
        return report
    report.colmap_extent = float(colmap_dists.max())
    report.gps_extent_m = float(gps_dists.max())

    # Umeyama similarity — returns (4×4 transform, scale)
    try:
        _transform, scale = align_to_enu(colmap_pts, gps_pts)
    except Exception as exc:
        report.status = "INSUFFICIENT_DATA"
        report.warnings.append(f"Umeyama alignment failed: {exc}")
        return report

    if scale <= 0 or not np.isfinite(scale):
        report.status = "INSUFFICIENT_DATA"
        report.warnings.append(f"Non-physical scale factor: {scale}")
        return report

    report.scale_factor = float(scale)

    # Full Umeyama alignment — gives 4×4 transform (scale + rotation + translation)
    # The residual after FULL alignment tells us how consistent the GPS geometry
    # is with the COLMAP geometry — this is the meaningful quality metric.
    try:
        full_transform, _ = align_to_enu(colmap_pts, gps_pts)
        # Apply the full 4×4 transform to COLMAP points
        ones = np.ones((len(colmap_pts), 1))
        colmap_h = np.hstack([colmap_pts, ones])  # (N, 4)
        aligned = (full_transform @ colmap_h.T).T[:, :3]  # (N, 3)
        per_point_err = np.linalg.norm(aligned - gps_pts, axis=1)
        report.residual_m = float(np.sqrt(np.mean(per_point_err**2)))
        report.max_error_m = float(per_point_err.max())
    except Exception:
        # Fallback: scale-only residual (less meaningful but non-zero)
        centred_colmap = colmap_pts - colmap_pts.mean(axis=0)
        centred_gps = gps_pts - gps_pts.mean(axis=0)
        scaled = centred_colmap * scale
        per_point_err = np.linalg.norm(scaled - centred_gps, axis=1)
        report.residual_m = float(np.sqrt(np.mean(per_point_err**2)))
        report.max_error_m = float(per_point_err.max())

    # Sanity checks
    if report.residual_m > report.gps_extent_m * 0.5:
        report.warnings.append(
            f"High residual ({report.residual_m:.1f} m) relative to GPS "
            f"extent ({report.gps_extent_m:.1f} m) — scale estimate is unreliable"
        )
    if report.gps_extent_m < 5.0:
        report.warnings.append(
            f"Small GPS baseline ({report.gps_extent_m:.1f} m) — "
            f"scale precision is limited by GPS accuracy"
        )

    report.status = "ESTIMATED"
    log.info(
        "metric_scale_estimated",
        scale=round(report.scale_factor, 6),
        residual_m=round(report.residual_m, 4),
        cameras=report.num_cameras,
        gps_extent_m=round(report.gps_extent_m, 2),
    )
    return report


# ---------------------------------------------------------------------------
# Apply scale to PLY
# ---------------------------------------------------------------------------


def apply_scale_to_ply(
    input_ply: Path,
    output_ply: Path,
    scale_factor: float,
) -> dict:
    """Multiply all vertex positions in a binary PLY by *scale_factor*.

    Writes a new PLY; does not modify the original.
    Returns a dict with point count and file size for the report.
    """
    from app.services.pointcloud import read_ply, save_ply

    cloud = read_ply(input_ply)
    cloud.xyz *= scale_factor
    cloud.meta["metric_scale_applied"] = scale_factor
    cloud.meta["metric_scale_source"] = "gps_colmap_alignment"
    cloud.meta["metric_scale_validated"] = False
    cloud.meta["metric_scale_note"] = (
        "Scale derived from GPS-vs-COLMAP alignment; not validated against "
        "surveyed ground truth. Use with caution."
    )
    save_ply(output_ply, cloud)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    size_bytes = output_ply.stat().st_size if output_ply.exists() else 0
    return {
        "points": len(cloud.xyz),
        "file_size_mb": round(size_bytes / 1024 / 1024, 2),
        "scale_applied": scale_factor,
    }


# ---------------------------------------------------------------------------
# Convenience: run as a stage in the validator
# ---------------------------------------------------------------------------


def run_metric_scale_validation(
    sparse_dir: Path,
    output_dir: Path,
) -> dict:
    """Run metric-scale validation on a completed run.

    Reads sparse/poses.json (with GPS already attached), estimates the
    scale, optionally scales dense/dense_model.ply, and returns a stage
    result dict for the manifest.
    """
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    poses_path = sparse_dir / "poses.json"
    if not poses_path.exists():
        return {
            "status": "BLOCKED",
            "reason": "No poses.json in sparse directory",
        }

    report = estimate_scale(poses_path)

    # Save report
    report_path = output_dir / "metric_scale_report.json"
    with open(report_path, "w") as fp:
        json.dump(report.to_dict(), fp, indent=2)

    # Apply scale to dense PLY if it exists
    dense_ply = sparse_dir.parent / "dense" / "dense_model.ply"
    scaled_ply = output_dir / "dense_metric.ply"
    scaled_info = {}
    if dense_ply.exists() and report.status == "ESTIMATED":
        scaled_info = apply_scale_to_ply(dense_ply, scaled_ply, report.scale_factor)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "status": report.status,
        "scale_factor": report.scale_factor,
        "residual_m": report.residual_m,
        "max_error_m": report.max_error_m,
        "num_cameras": report.num_cameras,
        "gps_extent_m": report.gps_extent_m,
        "warnings": report.warnings,
        "scaled_ply": str(scaled_ply) if scaled_ply.exists() else None,
        "scaled_info": scaled_info,
        "elapsed_ms": elapsed_ms,
    }
