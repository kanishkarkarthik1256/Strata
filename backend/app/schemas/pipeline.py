"""Pydantic schemas for the autonomous pipeline API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class StereoOptions(BaseModel):
    """SGBM stereo depth tunables (override settings defaults)."""

    min_disparity: Optional[int] = None
    num_disparities: Optional[int] = Field(default=None, ge=16)
    block_size: Optional[int] = None
    uniqueness_ratio: Optional[int] = None


class RefineOptions(BaseModel):
    """Optional depth refinement filters (all default off)."""

    median: Optional[bool] = None
    median_k: Optional[int] = None
    bilateral: Optional[bool] = None
    edge_preserving: Optional[bool] = None
    hole_fill: Optional[bool] = None


class GpsFix(BaseModel):
    """A WGS84 fix (project-level telemetry fallback)."""

    lat: float
    lon: float
    alt: float = 0.0


class PipelineStartRequest(BaseModel):
    """Tunables for one autonomous run. Stages already complete are reused."""

    extraction_mode: Literal["every_frame", "every_n", "target_fps", "interval"] = "every_n"
    every_n: Optional[int] = Field(default=None, ge=1)
    target_fps: Optional[float] = Field(default=None, gt=0)
    top_percent: Optional[float] = Field(default=None, gt=0, le=1)
    quality_threshold: Optional[float] = None
    depth_backend: Literal["auto", "depth_anything", "colmap", "stereo"] = "auto"
    frame_stride: int = Field(default=1, ge=1)
    max_depth_views: int = Field(default=200, ge=1)
    force: list[Literal["frames", "sparse", "depth", "dense"]] = []
    stereo: Optional[StereoOptions] = None
    refine: Optional[RefineOptions] = None
    gps: Optional[GpsFix] = None
    telemetry_csv: Optional[str] = Field(
        default=None,
        description=(
            "Workspace-relative external telemetry CSV. Any reasonable UAV format is auto-detected (column aliases, delimiters, units, header position); canonical 'timestamp,latitude,longitude,altitude' remains supported. "
            "Omit for video-only reconstruction."
        ),
    )


class PipelineStartResponse(BaseModel):
    """Result of an autonomous pipeline run."""

    job_id: str
    status: str
    run_time_ms: float = 0.0
    error: str = ""
    stages: dict = {}
    message: str = ""


class PipelineStatusResponse(BaseModel):
    """Current pipeline state for a job (live from the workspace artifacts)."""

    job_id: str
    status: str
    run_time_ms: float = 0.0
    error: str = ""
    stages: dict = {}
    profile: dict = {}
    resume: dict = {}


class PipelineCancelResponse(BaseModel):
    """Acknowledges a cancellation request."""

    job_id: str
    cancelled: bool = True
