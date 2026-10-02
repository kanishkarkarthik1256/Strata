"""Pydantic schemas for the geospatial-intelligence (Phase 8) API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IntelRunRequest(BaseModel):
    """Run the intelligence chain over an existing digital twin.

    ``plugins`` selects which auto-discovered stages to run (order matters;
    empty = the full INTEL_CHAIN). Completed stage artifacts are reused, so
    runs resume cleanly.
    """

    plugins: list[str] = Field(
        default_factory=list,
        description="Stage names to run; empty = full intelligence chain",
    )
    force: list[str] = Field(default_factory=list, description="Re-run these stages")


class IntelRunResponse(BaseModel):
    job_id: str
    status: str
    run_time_ms: float = 0.0
    error: str = ""
    stages: dict = {}
    message: str = ""


class IntelStatusResponse(BaseModel):
    job_id: str
    status: str
    run_time_ms: float = 0.0
    error: str = ""
    stages: dict = {}
    resume: dict = {}


class IntelQueryRequest(BaseModel):
    """A natural-language question grounded in this mission's data."""

    query: str = Field(..., min_length=1, max_length=500)


class ChangeCompareRequest(BaseModel):
    """Compare this job's current reconstruction against a baseline job."""

    baseline_job: str = Field(..., min_length=1, description="The earlier mission's job id")
    voxel_m: float = Field(default=0.5, ge=0.2, le=10.0, description="Voxel size for the occupancy diff")
