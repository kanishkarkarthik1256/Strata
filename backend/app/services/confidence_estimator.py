"""Confidence estimation — per-camera and per-point confidence scoring.

Generates confidence values (0.0–1.0) based on feature count, match quality,
reprojection error, and track length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.logging_config import get_logger
from app.services.camera_pose_estimator import ReconstructionResult

log = get_logger("drone_recon.services.confidence_estimator")


@dataclass
class CameraConfidence:
    """Confidence score for a single camera."""
    frame_id: str
    confidence: float  # 0.0-1.0
    level: str  # "high", "medium", "low"
    feature_count: int = 0
    match_count: int = 0
    reproj_error: float = 0.0


@dataclass
class PointConfidence:
    """Confidence score for a single 3D point."""
    point_id: int
    confidence: float
    level: str
    track_length: int = 0
    reproj_error: float = 0.0


@dataclass
class ConfidenceMap:
    """Full confidence estimation for a reconstruction."""
    camera_confidences: list[CameraConfidence]
    point_confidences: list[PointConfidence]
    mean_camera_confidence: float = 0.0
    mean_point_confidence: float = 0.0


def estimate_confidence(
    result: ReconstructionResult,
    feature_counts: dict[str, int] | None = None,
    match_counts: dict[str, int] | None = None,
    reproj_errors: dict[str, float] | None = None,
) -> ConfidenceMap:
    """Estimate confidence for all cameras and points."""
    camera_confs = []
    point_confs = []

    # Camera confidence
    for name, cam in result.cameras.items():
        feat_count = (feature_counts or {}).get(name, 0)
        match_count = (match_counts or {}).get(name, 0)
        reproj = (reproj_errors or {}).get(name, 0.0)

        conf = _camera_confidence(feat_count, match_count, reproj)
        camera_confs.append(CameraConfidence(
            frame_id=name,
            confidence=conf,
            level=_conf_to_level(conf),
            feature_count=feat_count,
            match_count=match_count,
            reproj_error=reproj,
        ))

    # Point confidence
    for pt in result.points3d:
        conf = _point_confidence(pt.track_length, pt.mean_reproj_error)
        point_confs.append(PointConfidence(
            point_id=pt.point_id,
            confidence=conf,
            level=_conf_to_level(conf),
            track_length=pt.track_length,
            reproj_error=pt.mean_reproj_error,
        ))

    mean_cam = float(np.mean([c.confidence for c in camera_confs])) if camera_confs else 0.0
    mean_pt = float(np.mean([p.confidence for p in point_confs])) if point_confs else 0.0

    log.info(
        "confidence_estimated",
        cameras=len(camera_confs),
        points=len(point_confs),
        mean_camera=round(mean_cam, 4),
        mean_point=round(mean_pt, 4),
    )

    return ConfidenceMap(
        camera_confidences=camera_confs,
        point_confidences=point_confs,
        mean_camera_confidence=round(mean_cam, 4),
        mean_point_confidence=round(mean_pt, 4),
    )


def _camera_confidence(feature_count: int, match_count: int, reproj_error: float) -> float:
    """Compute camera confidence from multiple signals."""
    # Feature score: 2000+ features = 1.0
    feat_score = min(feature_count / 2000.0, 1.0)

    # Match score: 500+ matches = 1.0
    match_score = min(match_count / 500.0, 1.0)

    # Reprojection score: <1px = 1.0, >5px = 0.0
    reproj_score = max(0.0, min(1.0, 1.0 - (reproj_error - 0.5) / 4.5))

    return 0.3 * feat_score + 0.3 * match_score + 0.4 * reproj_score


def _point_confidence(track_length: int, reproj_error: float) -> float:
    """Compute point confidence from track length and reprojection error."""
    # Track score: 5+ observations = 1.0
    track_score = min(track_length / 5.0, 1.0)

    # Reprojection score
    reproj_score = max(0.0, min(1.0, 1.0 - reproj_error / 5.0))

    return 0.5 * track_score + 0.5 * reproj_score


def _conf_to_level(conf: float) -> str:
    if conf >= 0.7:
        return "high"
    if conf >= 0.4:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Dense point cloud confidence (Phase 6)
# ---------------------------------------------------------------------------

# Four-class confidence for dense points (Module: AI Point Confidence).
CONFIDENCE_CLASSES = ("high", "medium", "low", "very_low")

# Discrete class colours for visualisation / exports: green → yellow → red.
CLASS_COLORS = np.array(
    [
        [46, 204, 113],  # high
        [241, 196, 15],  # medium
        [230, 126, 34],  # low
        [192, 57, 43],  # very_low
    ],
    dtype=np.uint8,
)


def dense_level_from_conf(conf: np.ndarray) -> np.ndarray:
    """Map confidence values to the four dense classes (high/medium/low/very_low)."""
    out = np.full(np.asarray(conf).shape, "very_low", dtype=object)
    out[conf >= 0.7] = "high"
    out[(conf >= 0.45) & (conf < 0.7)] = "medium"
    out[(conf >= 0.25) & (conf < 0.45)] = "low"
    return out


def dense_point_confidence(
    observations: np.ndarray,
    residual: np.ndarray | None = None,
    voxel_size: float = 1.0,
) -> np.ndarray:
    """Per-point confidence (0..1) for a fused dense cloud.

    Combines how many views agree on a point (observation count) with how
    tightly they agree (residual vs the fusion voxel size):

    * ``obs_score = 1 - exp(-observations / 3)``  (saturates near ~10 views)
    * ``residual_score = exp(-residual / voxel_size)``
    * ``confidence = 0.6*obs_score + 0.4*residual_score``
    """
    obs = np.asarray(observations, dtype=np.float64)
    obs_score = 1.0 - np.exp(-obs / 3.0)
    if residual is not None and len(residual):
        resid = np.asarray(residual, dtype=np.float64)
        residual_score = np.exp(-resid / max(float(voxel_size), 1e-9))
    else:
        residual_score = np.ones_like(obs)
    conf = 0.6 * obs_score + 0.4 * residual_score
    return np.clip(conf, 0.0, 1.0)


def confidence_colors(conf: np.ndarray) -> np.ndarray:
    """Map confidence to discrete class colours for visualisation/exports."""
    levels = dense_level_from_conf(np.asarray(conf, dtype=np.float64))
    colors = np.zeros((len(levels), 3), dtype=np.uint8)
    for i, level in enumerate(CONFIDENCE_CLASSES):
        colors[levels == level] = CLASS_COLORS[i]
    return colors
