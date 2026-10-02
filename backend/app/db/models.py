"""SQLAlchemy ORM models for the DroneRecon database.

Core tables (Phases 1–9)
------------------------
projects   – one row per uploaded drone video / reconstruction job
jobs       – tracks each pipeline stage execution within a project
frames     – per-frame metadata (quality scores, GPS, pose, file path)
models_3d  – output 3D model metadata (format, path, bounding box)
exports    – download / export history

Enterprise tables (Phase 10) — additive only, existing tables untouched
---------------------------------------------------------------------
users          – accounts with a role (admin|operator|analyst|viewer)
auth_sessions  – bearer sessions (token stored as SHA-256)
missions       – mission lifecycle record wrapping a project workspace
mission_events – append-only audit trail of mission state changes
queue_jobs     – durable in-process job queue driving the orchestrator
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Project(Base):
    """Represents a single upload → reconstruction pipeline run."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    video_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    video_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|processing|completed|failed|cancelled
    current_stage: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    stage_progress: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Video metadata
    video_duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    video_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    video_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    video_fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # GPS (from video metadata, if available)
    gps_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gps_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gps_alt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    jobs: Mapped[list["Job"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    frames: Mapped[list["Frame"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    models: Mapped[list["Model3D"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    exports: Mapped[list["Export"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Job(Base):
    """Tracks execution of a single pipeline stage."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|running|completed|failed|skipped
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    log: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    project: Mapped["Project"] = relationship(back_populates="jobs")


class Frame(Base):
    """Metadata for a single extracted frame."""

    __tablename__ = "frames"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    index: Mapped[int] = mapped_column(Integer, nullable=False)  # frame order
    timestamp_sec: Mapped[float] = mapped_column(Float, nullable=False)  # time in video
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    # Quality scores
    blur_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exposure_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    motion_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    overlap_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    composite_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kept: Mapped[bool] = mapped_column(default=True)

    # GPS per frame
    gps_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gps_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gps_alt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Camera pose (from COLMAP)
    pose_qw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose_qx: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose_qy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose_qz: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose_tx: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose_ty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose_tz: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Depth map
    depth_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    project: Mapped["Project"] = relationship(back_populates="frames")


class Model3D(Base):
    """Metadata for an output 3D model."""

    __tablename__ = "models_3d"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    format: Mapped[str] = mapped_column(String(16), nullable=False)  # obj|ply|gltf|glb|las
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    vertex_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    face_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    point_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Bounding box
    bbox_min_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_min_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_min_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_max_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_max_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_max_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    project: Mapped["Project"] = relationship(back_populates="models")


class Export(Base):
    """Tracks export / download history."""

    __tablename__ = "exports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    format: Mapped[str] = mapped_column(String(16), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    project: Mapped["Project"] = relationship(back_populates="exports")


# ===========================================================================
# Phase 10 — enterprise platform tables (additive)
# ===========================================================================


class User(Base):
    """An authenticated account with a platform role."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(16), default="viewer", nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    sessions: Mapped[list["AuthSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class AuthSession(Base):
    """A bearer session. The raw token is shown once at login; the DB stores
    only its SHA-256 hash so a leaked database cannot be replayed directly."""

    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    user: Mapped["User"] = relationship(back_populates="sessions")


# Mission lifecycle states (uppercase strings stored verbatim).
MISSION_CREATED = "CREATED"
MISSION_UPLOADING = "UPLOADING"
MISSION_VALIDATING = "VALIDATING"
MISSION_QUEUED = "QUEUED"
MISSION_PROCESSING = "PROCESSING"
MISSION_PAUSED = "PAUSED"
MISSION_RESUMING = "RESUMING"
MISSION_COMPLETED = "COMPLETED"
MISSION_FAILED = "FAILED"
MISSION_CANCELLED = "CANCELLED"

MISSION_STATES = (
    MISSION_CREATED, MISSION_UPLOADING, MISSION_VALIDATING, MISSION_QUEUED,
    MISSION_PROCESSING, MISSION_PAUSED, MISSION_RESUMING, MISSION_COMPLETED,
    MISSION_FAILED, MISSION_CANCELLED,
)


class Mission(Base):
    """Enterprise mission record wrapping an existing project workspace.

    ``project_id`` is nullable so a mission can be created before its input is
    attached; the pipeline runs against the referenced project (job id) using
    the existing orchestrator, and the project's workspace is the storage
    namespace for every phase artifact.
    """

    __tablename__ = "missions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    owner_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    project_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("projects.id"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=MISSION_CREATED, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    current_stage: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    stage_progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    config_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    owner: Mapped[Optional["User"]] = relationship()
    events: Mapped[list["MissionEvent"]] = relationship(
        back_populates="mission", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["QueueJob"]] = relationship(
        back_populates="mission", cascade="all, delete-orphan"
    )


class MissionEvent(Base):
    """Append-only audit entry for one mission state change."""

    __tablename__ = "mission_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    mission_id: Mapped[str] = mapped_column(ForeignKey("missions.id"), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(320), default="system", nullable=False)
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    from_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    to_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    mission: Mapped["Mission"] = relationship(back_populates="events")


class QueueJob(Base):
    """A durable unit of work executed by the in-process queue worker.

    ``kind`` selects an executor (default ``pipeline`` = run the existing
    orchestrator for ``project_id``). Rows survive restarts: jobs left in
    ``queued``/``running`` by a crash are re-adopted on startup.
    """

    __tablename__ = "queue_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    mission_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("missions.id"), nullable=True, index=True
    )
    project_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(64), default="pipeline", nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued", nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=2, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    mission: Mapped[Optional["Mission"]] = relationship(back_populates="jobs")
