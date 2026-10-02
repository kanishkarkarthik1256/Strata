"""Pydantic schemas for the Phase 10 enterprise platform API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MissionCreateRequest(BaseModel):
    """Create a mission around an existing project workspace."""

    name: str = Field(min_length=1, max_length=256)
    project_id: str = Field(min_length=1, max_length=64)
    priority: int = Field(default=5, ge=0, le=10)
    description: Optional[str] = None


class MissionStartRequest(BaseModel):
    """Configuration for a mission run enqueued to the queue worker."""

    plugins: list[str] = Field(default_factory=list)  # e.g. mesh/intel chain names
    force: list[str] = Field(default_factory=list)  # force re-run of these stages
    priority: Optional[int] = Field(default=None, ge=0, le=10)
    max_attempts: Optional[int] = Field(default=None, ge=1, le=10)


class MissionResponse(BaseModel):
    id: str
    project_id: Optional[str]
    name: str
    status: str
    priority: int
    current_stage: Optional[str]
    stage_progress: float
    error: Optional[str]
    owner: Optional[dict]
    created_at: Optional[str]
    updated_at: Optional[str]
    completed_at: Optional[str]


class MissionEventResponse(BaseModel):
    id: str
    actor: str
    event: str
    from_status: Optional[str]
    to_status: Optional[str]
    detail: Optional[dict]
    created_at: Optional[str]
