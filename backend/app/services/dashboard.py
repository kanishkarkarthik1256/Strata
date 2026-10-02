"""Reconstruction dashboard — aggregated metrics for the REST API."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.camera_pose_estimator import ReconstructionResult
from app.services.confidence_estimator import ConfidenceMap
from app.services.mission_analyzer import MissionAnalysis
from app.services.trajectory_optimizer import Trajectory


@dataclass
class DashboardData:
    """All metrics needed for the reconstruction dashboard."""
    mission_score: float = 0.0
    grade: str = "Poor"
    registered_cameras: int = 0
    failed_cameras: int = 0
    total_cameras: int = 0
    sparse_points: int = 0
    average_reprojection_error: float = 0.0
    camera_registration_rate: float = 0.0
    average_feature_count: float = 0.0
    average_match_quality: float = 0.0
    trajectory_smoothness: float = 0.0
    # Selected frames that registered (percent) — a capture-quality ratio, kept
    # distinct from the dense cloud's 2.5D footprint coverage.
    camera_registration_percent: float = 0.0
    mean_camera_confidence: float = 0.0
    mean_point_confidence: float = 0.0
    suggestions: list[str] = field(default_factory=list)
    status: str = "idle"
    pipeline_stages: list[dict] = field(default_factory=list)


def build_dashboard(
    result: ReconstructionResult,
    analysis: MissionAnalysis,
    confidence: ConfidenceMap,
    trajectory: Trajectory | None = None,
    total_frames: int = 0,
    status: str = "completed",
) -> DashboardData:
    """Build complete dashboard data from all analysis results."""
    return DashboardData(
        mission_score=analysis.mission_score,
        grade=analysis.grade,
        registered_cameras=result.num_registered,
        failed_cameras=total_frames - result.num_registered,
        total_cameras=total_frames,
        sparse_points=result.num_points,
        average_reprojection_error=result.mean_reproj_error,
        camera_registration_rate=analysis.camera_registration_rate,
        average_feature_count=analysis.average_feature_count,
        average_match_quality=analysis.average_match_quality,
        trajectory_smoothness=trajectory.smoothness if trajectory else 0.0,
        camera_registration_percent=analysis.camera_registration_percent,
        mean_camera_confidence=confidence.mean_camera_confidence,
        mean_point_confidence=confidence.mean_point_confidence,
        suggestions=analysis.suggestions,
        status=status,
    )
