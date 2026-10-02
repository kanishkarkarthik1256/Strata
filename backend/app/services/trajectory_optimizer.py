"""Camera trajectory generation and optimization.

Generates smooth camera paths from estimated poses and computes
heading, altitude, and speed profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from app.logging_config import get_logger
from app.services.camera_pose_estimator import CameraPose

log = get_logger("drone_recon.services.trajectory_optimizer")


@dataclass
class TrajectoryPoint:
    """A single point along the camera trajectory."""
    frame_id: str
    position: np.ndarray  # (3,)
    heading_deg: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0
    timestamp_sec: float = 0.0


@dataclass
class Trajectory:
    """Full camera trajectory."""
    points: list[TrajectoryPoint]
    total_length: float = 0.0  # meters
    mean_speed: float = 0.0  # m/s
    smoothness: float = 0.0  # 0-1, higher = smoother


def generate_trajectory(cameras: dict[str, CameraPose]) -> Trajectory:
    """Generate a trajectory from estimated camera poses.

    Sorts by frame_id (assumed temporal order), computes heading,
    altitude, speed, and smoothness.
    """
    if not cameras:
        return Trajectory(points=[])

    # Sort cameras by frame_id (temporal order)
    sorted_cams = sorted(cameras.items(), key=lambda x: x[0])
    points = []

    for i, (name, cam) in enumerate(sorted_cams):
        pos = cam.position
        heading = _compute_heading(cam.rotation) if cam.is_estimated else 0.0
        altitude = float(pos[2]) if len(pos) > 2 else 0.0

        speed = 0.0
        if i > 0:
            prev_pos = sorted_cams[i - 1][1].position
            speed = float(np.linalg.norm(pos - prev_pos))

        points.append(TrajectoryPoint(
            frame_id=name,
            position=pos.copy(),
            heading_deg=heading,
            altitude=altitude,
            speed=speed,
            timestamp_sec=float(i),
        ))

    # Compute trajectory metrics
    total_length = sum(p.speed for p in points)
    mean_speed = total_length / max(len(points) - 1, 1)
    smoothness = _compute_smoothness([p.position for p in points])

    log.info(
        "trajectory_generated",
        points=len(points),
        total_length=round(total_length, 2),
        smoothness=round(smoothness, 4),
    )

    return Trajectory(
        points=points,
        total_length=round(total_length, 2),
        mean_speed=round(mean_speed, 4),
        smoothness=round(smoothness, 4),
    )


def smooth_trajectory(trajectory: Trajectory, smoothing_factor: float = 0.5) -> Trajectory:
    """Apply cubic spline smoothing to the trajectory."""
    if len(trajectory.points) < 4:
        return trajectory

    positions = np.array([p.position for p in trajectory.points])
    t = np.arange(len(positions))

    # Interpolate each axis
    smoothed = np.zeros_like(positions)
    for axis in range(3):
        cs = CubicSpline(t, positions[:, axis])
        smoothed[:, axis] = cs(t) * smoothing_factor + positions[:, axis] * (1 - smoothing_factor)

    # Update points
    new_points = []
    for i, pt in enumerate(trajectory.points):
        new_points.append(TrajectoryPoint(
            frame_id=pt.frame_id,
            position=smoothed[i],
            heading_deg=pt.heading_deg,
            altitude=float(smoothed[i][2]),
            speed=pt.speed,
            timestamp_sec=pt.timestamp_sec,
        ))

    return Trajectory(
        points=new_points,
        total_length=trajectory.total_length,
        mean_speed=trajectory.mean_speed,
        smoothness=_compute_smoothness([p.position for p in new_points]),
    )


def _compute_heading(rotation: np.ndarray) -> float:
    """Extract heading angle (yaw) from rotation matrix in degrees."""
    heading = np.degrees(np.arctan2(rotation[0, 2], rotation[2, 2]))
    return float(heading)


def _compute_smoothness(positions: list[np.ndarray]) -> float:
    """Compute trajectory smoothness as inverse of mean curvature.

    Returns 0.0 (jerky) to 1.0 (perfectly smooth).
    """
    if len(positions) < 3:
        return 1.0

    positions = np.array(positions)
    # Compute second differences (acceleration proxy)
    diffs = np.diff(positions, axis=0)
    if len(diffs) < 2:
        return 1.0

    accelerations = np.diff(diffs, axis=0)
    mean_accel = np.mean(np.linalg.norm(accelerations, axis=1))

    # Normalize: 0 acceleration = smooth, high = jerky
    return float(max(0.0, min(1.0, 1.0 - mean_accel / 10.0)))
