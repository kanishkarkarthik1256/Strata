"""Pydantic schemas for the dense reconstruction API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DenseStartRequest(BaseModel):
    """Tunables for one dense run. Every field has a settings default."""

    voxel_size: Optional[float] = Field(default=None, gt=0, description="Fusion/merge voxel size in meters")
    sor_k: Optional[int] = Field(default=None, ge=3, description="Statistical outlier removal neighbours")
    sor_std_ratio: Optional[float] = Field(default=None, gt=0)
    ror_radius_m: Optional[float] = Field(default=None, gt=0, description="Radius outlier removal radius")
    ror_min_neighbors: Optional[int] = Field(default=None, ge=1)
    normal_k: Optional[int] = Field(default=None, ge=3, description="PCA normal neighbours")
    min_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    min_depth_m: Optional[float] = Field(default=None, gt=0)
    max_depth_m: Optional[float] = Field(default=None, gt=0)


class DenseStartResponse(BaseModel):
    """Response when dense reconstruction completes."""

    job_id: str
    status: str
    message: str


class DenseStatusResponse(BaseModel):
    """Status of the dense run for a job."""

    job_id: str
    status: str
    point_count: Optional[int] = None
    dense_score: Optional[float] = None
    grade: Optional[str] = None
    error: Optional[str] = None


class DenseStatisticsResponse(BaseModel):
    """Quality statistics for the dense model."""

    job_id: str
    quality: dict = {}
    measurements: dict = {}
    intelligence: dict = {}


class DenseConfidenceResponse(BaseModel):
    """Confidence summary for the dense model."""

    job_id: str
    mean_confidence: float = 0.0
    class_counts: dict = {}
    weak_regions: list[dict] = []
