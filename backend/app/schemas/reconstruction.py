"""Pydantic schemas for the reconstruction API."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReconstructionStatus(str, Enum):
    IDLE = "idle"
    EXTRACTING = "extracting_features"
    MATCHING = "matching_features"
    VERIFYING = "geometric_verification"
    ESTIMATING = "pose_estimation"
    OPTIMIZING = "bundle_adjustment"
    ANALYZING = "analysis"
    COMPLETED = "completed"
    FAILED = "failed"


class ReconstructionStartResponse(BaseModel):
    """Response when reconstruction is started."""
    job_id: str
    status: str
    message: str


class ReconstructionStatusResponse(BaseModel):
    """Full reconstruction status."""
    job_id: str
    status: str
    mission_score: Optional[float] = None
    grade: Optional[str] = None
    registered_cameras: Optional[int] = None
    total_cameras: Optional[int] = None
    sparse_points: Optional[int] = None
    average_reprojection_error: Optional[float] = None
    # Registered-camera percentage. Deliberately NOT named coverage_percent:
    # surface coverage is measured on the dense cloud (footprint occupancy)
    # and reported by the dense quality report.
    camera_registration_percent: Optional[float] = None
    suggestions: list[str] = []


class TrajectoryPoint(BaseModel):
    """A single trajectory point."""
    frame_id: str
    position: list[float]
    heading_deg: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0


class TrajectoryResponse(BaseModel):
    """Camera trajectory response."""
    job_id: str
    points: list[TrajectoryPoint]
    total_length: float = 0.0
    mean_speed: float = 0.0
    smoothness: float = 0.0


class FeatureInfo(BaseModel):
    """Feature extraction info for a frame."""
    frame_id: str
    num_keypoints: int
    extraction_time_ms: float
    backend: str


class FeaturesResponse(BaseModel):
    """Feature extraction results for a job."""
    job_id: str
    features: list[FeatureInfo]
    total_keypoints: int = 0
    average_keypoints: float = 0.0
    backend: str = "sift"


class MatchInfo(BaseModel):
    """Match result for a pair."""
    frame_a: str
    frame_b: str
    num_matches: int
    confidence: float
    backend: str


class MatchesResponse(BaseModel):
    """Matching results for a job."""
    job_id: str
    matches: list[MatchInfo]
    total_pairs: int = 0
    average_confidence: float = 0.0


class DashboardResponse(BaseModel):
    """Full dashboard data."""
    job_id: str
    mission_score: float = 0.0
    grade: str = "Poor"
    registered_cameras: int = 0
    failed_cameras: int = 0
    total_cameras: int = 0
    sparse_points: int = 0
    average_reprojection_error: float = 0.0
    camera_registration_rate: float = 0.0
    trajectory_smoothness: float = 0.0
    camera_registration_percent: float = 0.0
    mean_camera_confidence: float = 0.0
    mean_point_confidence: float = 0.0
    suggestions: list[str] = []


class ConfidenceInfo(BaseModel):
    """Confidence for a single entity."""
    id: str
    confidence: float
    level: str


class ConfidenceResponse(BaseModel):
    """Confidence estimation for a job."""
    job_id: str
    camera_confidences: list[ConfidenceInfo]
    point_confidences: list[ConfidenceInfo]
    mean_camera_confidence: float = 0.0
    mean_point_confidence: float = 0.0


class ReportResponse(BaseModel):
    """Full reconstruction report."""
    job_id: str
    status: str
    pipeline_time_ms: float = 0.0
    reconstruction: dict = {}
    trajectory: dict = {}
    analysis: dict = {}
    confidence: dict = {}
    dashboard: dict = {}
