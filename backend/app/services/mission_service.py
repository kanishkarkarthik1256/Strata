"""Mission lifecycle (Phase 10).

A mission is the enterprise wrapper around a project workspace. This module
owns the *state machine*: every transition is validated against
:data:`ALLOWED_TRANSITIONS` and recorded as an append-only
:class:`MissionEvent` audit row. Actual compute is delegated to the job
queue (see :mod:`app.services.job_queue`), which drives the existing
pipeline orchestrator.

States follow the Phase 10 contract exactly:
CREATED → UPLOADING → VALIDATING → QUEUED → PROCESSING → COMPLETED / FAILED
                                                                ↓ pause
                                                              PAUSED → RESUMING → QUEUED
                                                                ↓ cancel
                                                              CANCELLED
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import (
    MISSION_CANCELLED,
    MISSION_COMPLETED,
    MISSION_CREATED,
    MISSION_FAILED,
    MISSION_PAUSED,
    MISSION_PROCESSING,
    MISSION_QUEUED,
    MISSION_RESUMING,
    MISSION_UPLOADING,
    MISSION_VALIDATING,
    Mission,
    MissionEvent,
    User,
)
from app.logging_config import get_logger

log = get_logger("drone_recon.services.mission")

#: mission lifecycle name → allowed targets.
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    MISSION_CREATED: {MISSION_UPLOADING, MISSION_QUEUED, MISSION_CANCELLED},
    MISSION_UPLOADING: {MISSION_VALIDATING, MISSION_QUEUED, MISSION_FAILED, MISSION_CANCELLED},
    MISSION_VALIDATING: {MISSION_QUEUED, MISSION_FAILED, MISSION_CANCELLED},
    MISSION_QUEUED: {MISSION_PROCESSING, MISSION_PAUSED, MISSION_CANCELLED},
    # QUEUED from PROCESSING is a retry/recovery re-queue (worker-driven).
    MISSION_PROCESSING: {MISSION_COMPLETED, MISSION_FAILED, MISSION_PAUSED,
                          MISSION_CANCELLED, MISSION_QUEUED},
    MISSION_PAUSED: {MISSION_RESUMING, MISSION_CANCELLED},
    MISSION_RESUMING: {MISSION_QUEUED, MISSION_FAILED, MISSION_CANCELLED},
    MISSION_COMPLETED: {MISSION_QUEUED, MISSION_CANCELLED},  # allow re-flight
    MISSION_FAILED: {MISSION_QUEUED, MISSION_CANCELLED},  # allow retry
    MISSION_CANCELLED: {MISSION_QUEUED},  # re-flight allowed after cancellation
}

#: Role required to mutate a mission (start/pause/resume/cancel/delete).
MIN_MUTATION_ROLE = "operator"


def mission_public(mission: Mission, with_owner: bool = True) -> dict:
    data = {
        "id": mission.id,
        "project_id": mission.project_id,
        "name": mission.name,
        "status": mission.status,
        "priority": mission.priority,
        "current_stage": mission.current_stage,
        "stage_progress": mission.stage_progress,
        "error": mission.error,
        "created_at": mission.created_at.isoformat() if mission.created_at else None,
        "updated_at": mission.updated_at.isoformat() if mission.updated_at else None,
        "completed_at": mission.completed_at.isoformat() if mission.completed_at else None,
    }
    if with_owner:
        data["owner"] = (
            {"id": mission.owner.id, "email": mission.owner.email, "role": mission.owner.role}
            if mission.owner
            else None
        )
    return data


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


async def get_mission(db: AsyncSession, mission_id: str) -> Mission:
    mission = (
        await db.execute(
            select(Mission).options(selectinload(Mission.owner))
            .where(Mission.id == mission_id)
        )
    ).scalar_one_or_none()
    if mission is None:
        raise HTTPException(status_code=404, detail=f"Mission not found: {mission_id}")
    return mission


# ---------------------------------------------------------------------------
# Transitions (single owner of mission state)
# ---------------------------------------------------------------------------


async def transition(
    db: AsyncSession,
    mission: Mission,
    to_status: str,
    actor: Optional[User] = None,
    event: Optional[str] = None,
    detail: Optional[dict] = None,
    actor_name: Optional[str] = None,
) -> Mission:
    """Validate + apply one lifecycle transition, appending an audit event.

    All callers must go through this function so mission state changes have
    exactly one owner and every change is audited. ``actor`` is a :class:`User`
    when the transition comes from an authenticated request; the background
    worker passes ``actor=None`` plus ``actor_name="worker"`` (or "system").
    """
    allowed = ALLOWED_TRANSITIONS.get(mission.status, set())
    if to_status not in allowed:
        raise HTTPException(
            status_code=409,
            detail=f"Invalid transition {mission.status} → {to_status} "
                   f"(allowed: {sorted(allowed)})",
        )
    from_status = mission.status
    mission.status = to_status
    if to_status in (MISSION_COMPLETED, MISSION_FAILED, MISSION_CANCELLED):
        mission.completed_at = datetime.now(timezone.utc)
    if to_status == MISSION_FAILED:
        mission.error = (detail or {}).get("error", "") or mission.error
    await _audit(
        db, mission, actor,
        event=event or f"to_{to_status.lower()}",
        from_status=from_status, to_status=to_status, detail=detail,
        actor_name=actor_name,
    )
    who = actor_name or (actor.email if actor else "system")
    log.info("mission_transition", mission_id=mission.id,
             to_status=to_status, actor=who)
    return mission


async def _audit(
    db: AsyncSession,
    mission: Mission,
    actor: Optional[User],
    event: str,
    from_status: Optional[str],
    to_status: Optional[str],
    detail: Optional[dict],
    actor_name: Optional[str] = None,
) -> MissionEvent:
    record = MissionEvent(
        mission_id=mission.id,
        actor=actor_name or (actor.email if actor else "system"),
        event=event,
        from_status=from_status or mission.status,
        to_status=to_status,
        detail=json.dumps(detail) if detail else None,
    )
    db.add(record)
    await db.flush()
    return record


# ---------------------------------------------------------------------------
# Convenience mutations
# ---------------------------------------------------------------------------


async def create_mission(
    db: AsyncSession,
    owner: Optional[User],
    name: str,
    project_id: Optional[str] = None,
    priority: int = 5,
    config: Optional[dict] = None,
) -> Mission:
    mission = Mission(
        owner_id=owner.id if owner else None,
        name=name,
        project_id=project_id,
        status=MISSION_CREATED,
        priority=int(priority),
        config_json=json.dumps(config or {}),
    )
    if owner is not None:
        mission.owner = owner  # eagerly populate so mission_public() stays async-safe
    db.add(mission)
    await db.flush()
    await _audit(db, mission, owner, event="created",
                 from_status=None, to_status=MISSION_CREATED, detail=None)
    log.info("mission_created", mission_id=mission.id, project_id=project_id,
             owner=owner.email if owner else None)
    return mission


async def list_missions(
    db: AsyncSession, owner_id: Optional[str] = None, status: Optional[str] = None, limit: int = 100
) -> list[Mission]:
    query = (
        select(Mission).options(selectinload(Mission.owner))
        .order_by(Mission.created_at.desc()).limit(limit)
    )
    if owner_id:
        query = query.where(Mission.owner_id == owner_id)
    if status:
        query = query.where(Mission.status == status)
    return list((await db.execute(query)).scalars().all())


async def mission_events(db: AsyncSession, mission_id: str, limit: int = 200) -> list[dict]:
    rows = (
        await db.execute(
            select(MissionEvent)
            .where(MissionEvent.mission_id == mission_id)
            .order_by(MissionEvent.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()
    return [
        {
            "id": e.id,
            "actor": e.actor,
            "event": e.event,
            "from_status": e.from_status,
            "to_status": e.to_status,
            "detail": json.loads(e.detail) if e.detail else None,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in rows
    ]
