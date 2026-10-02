"""Pydantic schemas for the frame extraction API."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ExtractionMode(str, Enum):
    EVERY_FRAME = "every_frame"
    EVERY_N = "every_n"
    TARGET_FPS = "target_fps"
    INTERVAL = "interval"


class ExtractionRequest(BaseModel):
    """Configuration for a frame extraction run."""
    extraction_mode: ExtractionMode = Field(
        default=ExtractionMode.TARGET_FPS,
        description="Frame extraction strategy",
    )
    target_fps: Optional[float] = Field(None, gt=0, description="Target FPS for extraction")
    every_n: Optional[int] = Field(None, gt=0, description="Extract every Nth frame")
    interval_sec: Optional[float] = Field(None, gt=0, description="Seconds between extractions")
    quality_threshold: Optional[float] = Field(None, ge=0, le=1, description="Minimum composite score to keep")
    top_percent: Optional[float] = Field(None, gt=0, le=1, description="Keep top N percent of frames")


class FrameInfo(BaseModel):
    """Metadata for a single extracted frame."""
    index: int
    timestamp_sec: float
    filename: str
    kept: bool
    rejection_reason: Optional[str] = None
    scores: dict[str, float] = {}


class ExtractionResponse(BaseModel):
    """Response after starting or querying extraction."""
    job_id: str
    status: str
    message: str
    selected_count: Optional[int] = None
    rejected_count: Optional[int] = None
    total_candidates: Optional[int] = None


class ExtractionStatusResponse(BaseModel):
    """Full status of a frame extraction job."""
    job_id: str
    status: str
    extraction_mode: Optional[str] = None
    selected_count: Optional[int] = None
    rejected_count: Optional[int] = None
    total_candidates: Optional[int] = None
    frames: list[FrameInfo] = []


class FrameDetailResponse(BaseModel):
    """Detail for a single frame."""
    frame_id: str
    project_id: str
    index: int
    timestamp_sec: float
    file_path: str
    kept: bool
    blur_score: Optional[float] = None
    sharpness_score: Optional[float] = None
    exposure_score: Optional[float] = None
    motion_score: Optional[float] = None
    composite_score: Optional[float] = None
