"""Pydantic schemas for the upload / ingestion API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    VALIDATED = "validated"
    EXTRACTING = "extracting"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Video metadata (extracted from the uploaded file)
# ---------------------------------------------------------------------------


class VideoMetadata(BaseModel):
    """Metadata extracted from an uploaded drone video."""

    filename: str = Field(..., description="Original filename")
    duration_sec: float = Field(..., ge=0, description="Video duration in seconds")
    fps: float = Field(..., gt=0, description="Frames per second")
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")
    codec: str = Field(..., description="Video codec name (e.g. h264, hevc)")
    frame_count: int = Field(..., ge=0, description="Total number of frames")
    bitrate_kbps: float = Field(..., ge=0, description="Average bitrate in kbps")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    creation_time: Optional[datetime] = Field(None, description="Creation timestamp from metadata")

    # GPS (if present in video metadata)
    gps_lat: Optional[float] = Field(None, description="Latitude in degrees")
    gps_lon: Optional[float] = Field(None, description="Longitude in degrees")
    gps_alt: Optional[float] = Field(None, description="Altitude in metres")

    # Camera
    camera_make: Optional[str] = Field(None, description="Camera manufacturer")
    camera_model: Optional[str] = Field(None, description="Camera model name")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """Returned immediately after a video is accepted for upload."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Human-readable status message")
    metadata: Optional[VideoMetadata] = Field(
        None, description="Server-measured video properties (post-validation)"
    )


class JobStatusResponse(BaseModel):
    """Full status of an upload job, including metadata when available."""

    job_id: str
    status: JobStatus
    filename: Optional[str] = None
    metadata: Optional[VideoMetadata] = None
    workspace_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DeleteResponse(BaseModel):
    """Returned after a job is successfully deleted."""

    job_id: str
    status: str = "deleted"
    message: str
