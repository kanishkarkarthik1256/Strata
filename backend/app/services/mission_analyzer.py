"""Mission quality analyzer — evaluates reconstruction quality in real time.

Computes a comprehensive Mission Score (0-100) and provides actionable
suggestions for improving data capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.logging_config import get_logger
from app.services.camera_pose_estimator import CameraPose, ReconstructionResult
from app.services.trajectory_optimizer import Trajectory

log = get_logger("drone_recon.services.mission_analyzer")


@dataclass
class MissionAnalysis:
    """Full mission quality analysis."""
    mission_score: float = 0.0  # 0-100
    grade: str = "Poor"  # Excellent/Good/Fair/Poor
    camera_registration_rate: float = 0.0
    average_feature_count: float = 0.0
    average_match_quality: float = 0.0
    average_reprojection_error: float = 0.0
    sparse_density: float = 0.0
    # How many selected frames registered (percent). This is a CAPTURE-quality
    # ratio, NOT surface coverage — the previous name ``coverage_percent``
    # collided with the dense stage's genuine 2.5D footprint coverage
    # (app.services.point_statistics), which measures a completely different
    # quantity under the same key in other payloads.
    camera_registration_percent: float = 0.0
    trajectory_smoothness: float = 0.0
    suggestions: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def analyze_mission(
    result: ReconstructionResult,
    trajectory: Trajectory | None = None,
    total_frames: int = 0,
    selected_frames: int = 0,
    feature_counts: dict[str, int] | None = None,
    match_qualities: list[float] | None = None,
) -> MissionAnalysis:
    """Compute comprehensive mission quality analysis."""
    analysis = MissionAnalysis()

    # Camera registration rate
    if total_frames > 0:
        analysis.camera_registration_rate = result.num_registered / total_frames
    elif selected_frames > 0:
        analysis.camera_registration_rate = result.num_registered / selected_frames

    # Feature count
    if feature_counts:
        analysis.average_feature_count = float(np.mean(list(feature_counts.values())))

    # Match quality
    if match_qualities:
        analysis.average_match_quality = float(np.mean(match_qualities))

    # Reprojection error
    analysis.average_reprojection_error = result.mean_reproj_error

    # Sparse density (points per registered camera)
    if result.num_registered > 0:
        analysis.sparse_density = result.num_points / result.num_registered

    # Registration rate, expressed as percent (see the field's docstring: not
    # a surface-coverage measure).
    analysis.camera_registration_percent = analysis.camera_registration_rate * 100

    # Trajectory smoothness
    if trajectory:
        analysis.trajectory_smoothness = trajectory.smoothness

    # Compute mission score
    analysis.mission_score = _compute_score(analysis)
    analysis.grade = _score_to_grade(analysis.mission_score)

    # Generate suggestions
    analysis.suggestions = _generate_suggestions(analysis, result)

    # Store raw metrics
    analysis.metrics = {
        "registered_cameras": result.num_registered,
        "total_points": result.num_points,
        "mean_reproj_error": result.mean_reproj_error,
        "registration_rate": analysis.camera_registration_rate,
        "sparse_density": analysis.sparse_density,
        "camera_registration_percent": analysis.camera_registration_percent,
        "smoothness": analysis.trajectory_smoothness,
    }

    log.info(
        "mission_analyzed",
        score=round(analysis.mission_score, 1),
        grade=analysis.grade,
        registered=result.num_registered,
        points=result.num_points,
    )

    return analysis


def _compute_score(analysis: MissionAnalysis) -> float:
    """Weighted mission score (0-100)."""
    weights = {
        "registration": 0.25,
        "reprojection": 0.20,
        "density": 0.15,
        "coverage": 0.20,
        "smoothness": 0.10,
        "features": 0.10,
    }

    # Registration rate → 0-100
    reg_score = min(analysis.camera_registration_rate * 100, 100)

    # Reprojection error → 0-100 (lower is better, <1px = 100, >5px = 0)
    reproj_score = max(0, 100 - (analysis.average_reprojection_error - 1.0) * 25)

    # Density → 0-100 (target: 1000+ points per camera)
    density_score = min(analysis.sparse_density / 10.0, 1.0) * 100

    # Coverage → already 0-100
    coverage_score = analysis.camera_registration_percent

    # Smoothness → 0-100
    smooth_score = analysis.trajectory_smoothness * 100

    # Features → 0-100 (target: 2000+ features per image)
    feat_score = min(analysis.average_feature_count / 20.0, 1.0) * 100

    total = (
        weights["registration"] * reg_score +
        weights["reprojection"] * reproj_score +
        weights["density"] * density_score +
        weights["coverage"] * coverage_score +
        weights["smoothness"] * smooth_score +
        weights["features"] * feat_score
    )

    return float(np.clip(total, 0, 100))


def _score_to_grade(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"


def _generate_suggestions(analysis: MissionAnalysis, result: ReconstructionResult) -> list[str]:
    """Generate actionable improvement suggestions."""
    suggestions = []

    if analysis.camera_registration_rate < 0.8:
        suggestions.append("Increase image overlap — some cameras failed to register")

    if analysis.average_reprojection_error > 2.0:
        suggestions.append("Reduce flight speed to decrease motion blur")

    if analysis.sparse_density < 500:
        suggestions.append("Fly lower or slower for denser feature coverage")

    if analysis.trajectory_smoothness < 0.7:
        suggestions.append("Fly smoother trajectories — jerky motion reduces match quality")

    if analysis.average_feature_count < 500:
        suggestions.append("Fly over more textured terrain — weak texture limits features")

    if not suggestions:
        suggestions.append("Mission quality is good — no immediate improvements needed")

    return suggestions
