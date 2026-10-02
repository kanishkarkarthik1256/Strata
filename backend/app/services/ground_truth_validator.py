"""Independent ground-truth metric evaluators (measurement only).

Each ``evaluate_*`` function measures ONE accuracy quantity against held-out
reference data (checkpoints, known distances, reference surfaces, depth).

This module deliberately does NOT aggregate, certify, or persist anything:
aggregation, the capability ladder, the ≤1 m criterion and the canonical
artifact all belong to :mod:`app.services.metric_validation` (single owner).
The former aggregator here (``build_metric_accuracy_report``) and its
``metric_validation.json`` / ``.md`` writer were retired for exactly that
reason — they constituted a second, contradictory accuracy authority.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from app.schemas.ground_truth import (
    GCPPoint,
    GCPRole,
    GroundTruthData,
    KnownDistance,
    CameraPositionMetrics,
    CheckPointMetrics,
    DepthAccuracyMetrics,
    DistanceAccuracyMetrics,
    SurfaceDistanceMetrics,
    ReprojectionMetrics,
)


def validate_gps_coordinates(lat: float, lon: float, alt: Optional[float] = None) -> bool:
    """Verify lat/lon/alt are non-null, finite, non-zero, and within valid WGS84 ranges."""
    if lat is None or lon is None:
        return False
    try:
        lat, lon = float(lat), float(lon)
    except (ValueError, TypeError):
        return False
        
    if not math.isfinite(lat) or not math.isfinite(lon):
        return False
    if abs(lat) < 1e-6 and abs(lon) < 1e-6:
        return False  # Reject (0,0) fallback
    if abs(lat) > 90.0 or abs(lon) > 180.0:
        return False
    if alt is not None:
        try:
            alt = float(alt)
            if not math.isfinite(alt):
                return False
        except (ValueError, TypeError):
            return False
    return True


def separate_control_and_check_points(gt: GroundTruthData) -> Tuple[List[GCPPoint], List[GCPPoint]]:
    """Separate control points (for alignment) and check points (for validation)."""
    ctrl = [p for p in gt.control_points if p.role == GCPRole.CONTROL]
    ctrl_ids = {p.id for p in ctrl}
    check = [p for p in gt.check_points if p.role == GCPRole.CHECK and p.id not in ctrl_ids]
    return ctrl, check


def evaluate_camera_positions(
    estimated: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
    total_cameras: int = 0
) -> CameraPositionMetrics:
    """Calculate RMSE and errors between estimated camera centers and GT camera centers."""
    common_keys = sorted(set(estimated.keys()) & set(gt.keys()))
    if not common_keys:
        return CameraPositionMetrics(
            total_cameras=total_cameras,
            registered_cameras=0,
            registration_rate_percent=0.0,
            horizontal_rmse_m=0.0,
            vertical_rmse_m=0.0,
            rmse_3d_m=0.0,
            max_error_m=0.0,
        )
        
    e_pts = np.array([estimated[k] for k in common_keys])
    g_pts = np.array([gt[k] for k in common_keys])
    
    diffs = e_pts - g_pts
    horiz_diffs = diffs[:, :2]
    vert_diffs = diffs[:, 2]
    
    horiz_rmse = float(np.sqrt(np.mean(horiz_diffs ** 2)))
    vert_rmse = float(np.sqrt(np.mean(vert_diffs ** 2)))
    rmse_3d = float(np.sqrt(np.mean(diffs ** 2)))
    max_err = float(np.max(np.linalg.norm(diffs, axis=1)))
    
    reg_rate = (len(common_keys) / max(1, total_cameras)) * 100.0 if total_cameras > 0 else 100.0
    
    return CameraPositionMetrics(
        total_cameras=total_cameras or len(common_keys),
        registered_cameras=len(common_keys),
        registration_rate_percent=round(reg_rate, 2),
        horizontal_rmse_m=round(horiz_rmse, 4),
        vertical_rmse_m=round(vert_rmse, 4),
        rmse_3d_m=round(rmse_3d, 4),
        max_error_m=round(max_err, 4),
    )


def evaluate_check_points(
    estimated_points: Dict[str, np.ndarray],
    check_points: List[GCPPoint]
) -> CheckPointMetrics:
    """Calculate 3D error metrics on independent check points."""
    valid_cps = [cp for cp in check_points if cp.id in estimated_points]
    if not valid_cps:
        return CheckPointMetrics(
            num_check_points=0,
            horizontal_rmse_m=0.0,
            vertical_rmse_m=0.0,
            rmse_3d_m=0.0,
            max_error_m=0.0,
        )
        
    e_pts = np.array([estimated_points[cp.id] for cp in valid_cps])
    g_pts = []
    for cp in valid_cps:
        if getattr(cp, 'x', None) is not None:
            g_pts.append([cp.x, cp.y, cp.z])
        else:
            g_pts.append([cp.longitude, cp.latitude, cp.altitude])
    g_pts = np.array(g_pts)
    
    diffs = e_pts - g_pts
    horiz_rmse = float(np.sqrt(np.mean(np.sum(diffs[:, :2] ** 2, axis=1))))
    vert_rmse = float(np.sqrt(np.mean(diffs[:, 2] ** 2)))
    rmse_3d = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))
    max_err = float(np.max(np.linalg.norm(diffs, axis=1)))
    
    return CheckPointMetrics(
        num_check_points=len(valid_cps),
        horizontal_rmse_m=round(horiz_rmse, 4),
        vertical_rmse_m=round(vert_rmse, 4),
        rmse_3d_m=round(rmse_3d, 4),
        max_error_m=round(max_err, 4),
    )


def evaluate_distances(
    estimated_points: Dict[str, np.ndarray],
    known_distances: List[KnownDistance]
) -> DistanceAccuracyMetrics:
    """Evaluate distance errors between pairs of reference points."""
    if not known_distances:
        return DistanceAccuracyMetrics(
            num_distances=0,
            mean_absolute_error_m=0.0,
            median_absolute_error_m=0.0,
            rmse_m=0.0,
            mean_percentage_error=0.0,
            max_error_m=0.0,
        )
        
    errors = []
    pct_errors = []
    
    for kd in known_distances:
        # Distance calculated from kd.point_a and kd.point_b
        pa = np.array(kd.point_a)
        pb = np.array(kd.point_b)
        est_d = float(np.linalg.norm(pa - pb))
        gt_d = float(kd.distance_m)
        err = abs(est_d - gt_d)
        pct = (err / max(1e-6, gt_d)) * 100.0
        errors.append(err)
        pct_errors.append(pct)
        
    errors = np.array(errors)
    pct_errors = np.array(pct_errors)
    
    return DistanceAccuracyMetrics(
        num_distances=len(known_distances),
        mean_absolute_error_m=round(float(np.mean(errors)), 4),
        median_absolute_error_m=round(float(np.median(errors)), 4),
        rmse_m=round(float(np.sqrt(np.mean(errors ** 2))), 4),
        mean_percentage_error=round(float(np.mean(pct_errors)), 4),
        max_error_m=round(float(np.max(errors)), 4),
    )


def evaluate_surface_distance(
    reconstructed_cloud: np.ndarray,
    reference_cloud: np.ndarray
) -> SurfaceDistanceMetrics:
    """Compute surface distance percentiles (P50, P90, P95, P99) via nearest neighbor."""
    if len(reconstructed_cloud) == 0 or len(reference_cloud) == 0:
        return SurfaceDistanceMetrics(
            num_samples=0,
            mean_m=0.0,
            p50_m=0.0,
            p90_m=0.0,
            p95_m=0.0,
            p99_m=0.0,
        )
        
    from scipy.spatial import cKDTree
    tree = cKDTree(reference_cloud)
    dists, _ = tree.query(reconstructed_cloud)
    
    return SurfaceDistanceMetrics(
        num_samples=len(reconstructed_cloud),
        mean_m=round(float(np.mean(dists)), 4),
        p50_m=round(float(np.percentile(dists, 50)), 4),
        p90_m=round(float(np.percentile(dists, 90)), 4),
        p95_m=round(float(np.percentile(dists, 95)), 4),
        p99_m=round(float(np.percentile(dists, 99)), 4),
    )


def evaluate_depth(
    pred_depth: np.ndarray,
    ref_depth: np.ndarray,
    is_metric: bool = False,
    model_name: str = "Depth Anything V2"
) -> DepthAccuracyMetrics:
    """Evaluate depth predictions against reference depth maps."""
    valid = (pred_depth > 0) & (ref_depth > 0) & np.isfinite(pred_depth) & np.isfinite(ref_depth)
    if not valid.any():
        return DepthAccuracyMetrics(
            is_metric=is_metric,
            model_type=f"{'Metric' if is_metric else 'Relative depth'} model ({model_name})",
            mae_m=0.0,
            rmse_m=0.0,
        )
        
    p_val = pred_depth[valid]
    r_val = ref_depth[valid]
    
    if not is_metric:
        scale = float(np.median(r_val / p_val))
        p_val = p_val * scale
        
    diffs = np.abs(p_val - r_val)
    mae = float(np.mean(diffs))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    
    return DepthAccuracyMetrics(
        is_metric=is_metric,
        model_type=f"{'Metric' if is_metric else 'Relative depth'} model ({model_name})",
        mae_m=round(mae, 4),
        rmse_m=round(rmse, 4),
    )


def evaluate_reprojection(residuals: np.ndarray) -> ReprojectionMetrics:
    """Evaluate 2D reprojection error metrics."""
    if len(residuals) == 0:
        return ReprojectionMetrics(num_points=0, mean_error_px=0.0, median_error_px=0.0, p95_error_px=0.0)
    return ReprojectionMetrics(
        num_points=len(residuals),
        mean_error_px=round(float(np.mean(residuals)), 4),
        median_error_px=round(float(np.median(residuals)), 4),
        p95_error_px=round(float(np.percentile(residuals, 95)), 4),
    )


def calculate_scale_error(reconstructed_distances: np.ndarray, gt_distances: np.ndarray) -> Dict[str, float]:
    """Calculate percentage scale error metrics against independent ground truth distances."""
    if len(reconstructed_distances) == 0 or len(gt_distances) == 0:
        raise ValueError("Empty distance arrays provided for scale error calculation")
        
    abs_errors = np.abs(reconstructed_distances - gt_distances)
    pct_errors = (abs_errors / gt_distances) * 100.0
    rmse_m = float(np.sqrt(np.mean(abs_errors ** 2)))
    
    return {
        "mean_error_percent": float(np.mean(pct_errors)),
        "median_error_percent": float(np.median(pct_errors)),
        "rmse_meters": rmse_m,
        "p90_error_percent": float(np.percentile(pct_errors, 90)),
        "p95_error_percent": float(np.percentile(pct_errors, 95)),
        "max_error_percent": float(np.max(pct_errors)),
        "mean_distance_error_m": float(np.mean(abs_errors)),
        "median_distance_error_m": float(np.median(abs_errors)),
        "p95_distance_error_m": float(np.percentile(abs_errors, 95)),
        "p99_distance_error_m": float(np.percentile(abs_errors, 99))
    }


def calculate_absolute_trajectory_error(reconstructed_centers: np.ndarray, gt_centers: np.ndarray) -> Dict[str, float]:
    """Compute Absolute Trajectory Error (ATE) via optimal Sim3 alignment."""
    if len(reconstructed_centers) != len(gt_centers) or len(gt_centers) < 3:
        raise ValueError("Need at least 3 matching trajectory camera centers")
        
    mu_recon = reconstructed_centers.mean(axis=0)
    mu_gt = gt_centers.mean(axis=0)
    
    x_c = reconstructed_centers - mu_recon
    y_c = gt_centers - mu_gt
    
    H = x_c.T @ y_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    var_x = np.sum(x_c ** 2) / len(x_c)
    scale = np.sum(S) / (len(x_c) * var_x) if var_x > 0 else 1.0
    t = mu_gt - scale * (mu_recon @ R.T)
    
    aligned_recon = scale * (reconstructed_centers @ R.T) + t
    pos_errors = np.linalg.norm(aligned_recon - gt_centers, axis=1)
    
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(pos_errors ** 2))),
        "median_position_error_m": float(np.median(pos_errors)),
        "p95_position_error_m": float(np.percentile(pos_errors, 95)),
        "max_position_error_m": float(np.max(pos_errors)),
        "estimated_scale_factor": float(scale)
    }


