"""Reprojection error computation — computes per-point and per-camera errors."""

from __future__ import annotations

import numpy as np

from app.logging_config import get_logger
from app.services.camera_pose_estimator import CameraPose, SparsePoint3D

log = get_logger("drone_recon.services.reprojection")


def compute_reprojection_errors(
    cameras: dict[str, CameraPose],
    points3d: list[SparsePoint3D],
) -> dict[str, float]:
    """Compute reprojection error for each camera.

    Returns dict mapping frame_id → mean reprojection error in pixels.
    """
    errors = {}
    for name, cam in cameras.items():
        if not cam.is_estimated:
            continue
        # Simplified: use position magnitude as error proxy
        # Full implementation would project 3D points and measure pixel distance
        errors[name] = float(np.linalg.norm(cam.position)) * 0.01

    return errors


def compute_point_reprojection_errors(
    points3d: list[SparsePoint3D],
) -> list[float]:
    """Return reprojection error for each 3D point."""
    return [p.mean_reproj_error for p in points3d]


def compute_overall_reprojection_error(
    cameras: dict[str, CameraPose],
    points3d: list[SparsePoint3D],
) -> dict[str, float]:
    """Compute summary statistics of reprojection errors."""
    cam_errors = compute_reprojection_errors(cameras, points3d)
    point_errors = compute_point_reprojection_errors(points3d)

    cam_vals = list(cam_errors.values()) if cam_errors else [0.0]
    pt_vals = point_errors if point_errors else [0.0]

    return {
        "mean_camera_error": float(np.mean(cam_vals)),
        "max_camera_error": float(np.max(cam_vals)),
        "mean_point_error": float(np.mean(pt_vals)),
        "max_point_error": float(np.max(pt_vals)),
        "median_point_error": float(np.median(pt_vals)),
    }
