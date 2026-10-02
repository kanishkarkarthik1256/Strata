"""Pydantic schemas for the mission-planning (Phase 9) API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SceneModel(BaseModel):
    """Simulation AOI + measured scene evidence (extent is required)."""

    extent_w_m: float = Field(..., gt=1.0, description="AOI width in metres")
    extent_h_m: float = Field(..., gt=1.0, description="AOI height in metres")
    mean_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    blind_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_scene_height_m: Optional[float] = Field(default=None, ge=0.0)
    targets: list = Field(default_factory=list)
    restricted: list = Field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items()}


class SimulationRequest(BaseModel):
    plan: Optional[dict] = Field(default=None, description="Overrides on top of defaults")
    scene: SceneModel


class WhatIfRequest(BaseModel):
    scenario: dict = Field(..., description="Plan parameter deltas vs the baseline")
    plan: Optional[dict] = Field(default=None)
    scene: SceneModel


class OptimizeRequest(BaseModel):
    scene: SceneModel
    plan: Optional[dict] = Field(default=None, description="Optional fixed parameters")


class CopilotRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    job_id: Optional[str] = Field(default=None, description="Mission context to answer over")


class HistoryAppendRequest(BaseModel):
    record: dict[str, Any] = Field(..., description="Measured mission record to store")
