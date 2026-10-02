"""Pydantic schemas for the digital-twin (Phase 7) API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TwinBuildRequest(BaseModel):
    """Kick off the mesh → twin stage chain for a job.

    ``plugins`` selects which auto-discovered pipeline stages to run (order
    matters; empty = the full MESH_CHAIN). Stages whose artifact already
    exists are reused, so builds resume cleanly.
    """

    plugins: list[str] = Field(
        default_factory=list,
        description="Stage names to run; empty = full mesh → twin chain",
    )
    force: list[str] = Field(default_factory=list, description="Re-run these stages")


class TwinBuildResponse(BaseModel):
    """Result of a digital-twin build run."""

    job_id: str
    status: str
    run_time_ms: float = 0.0
    error: str = ""
    stages: dict = {}
    message: str = ""


class TwinStatusResponse(BaseModel):
    """Current digital-twin state for a job."""

    job_id: str
    status: str
    run_time_ms: float = 0.0
    error: str = ""
    stages: dict = {}
    resume: dict = {}


class CopilotRequest(BaseModel):
    """A natural-language question about the scene."""

    query: str = Field(..., min_length=1, max_length=500)
