"""Mission lifecycle API (Phase 10).

GET    /api/missions                — list missions
POST   /api/missions                — create around an existing project
GET    /api/missions/{id}           — mission detail
POST   /api/missions/{id}/start     — enqueue the mission on the job queue
POST   /api/missions/{id}/pause     — pause (stop at next stage boundary)
POST   /api/missions/{id}/resume    — resume a paused mission from its artifacts
POST   /api/missions/{id}/cancel    — cancel (terminal)
GET    /api/missions/{id}/events    — audit trail
GET    /api/missions/{id}/logs      — structured run logs (streaming history)
GET    /api/missions/{id}/artifacts — categorized workspace artifacts
POST   /api/missions/{id}/cleanup   — purge intermediate artifacts
DELETE /api/missions/{id}           — delete mission (+ workspace)
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.db.models import (
    MISSION_CANCELLED,
    MISSION_COMPLETED,
    MISSION_CREATED,
    MISSION_PAUSED,
    MISSION_PROCESSING,
    MISSION_QUEUED,
    MISSION_RESUMING,
    Project,
    User,
)
from app.exceptions import ProjectNotFoundError
from app.logging_config import get_logger
from app.schemas.platform import (
    MissionCreateRequest,
    MissionResponse,
    MissionStartRequest,
)
from app.services import job_queue, mission_service
from app.services.auth_service import (
    get_current_user,
    get_optional_user,
    require_any_role,
)
from app.services.metrics import metrics
from app.services.storage_manager import catalog, cleanup_intermediates

log = get_logger("drone_recon.routes.missions")

router = APIRouter(prefix="/api/missions", tags=["missions"])

#: Roles that may mutate missions (start/pause/resume/cancel/delete).
MUTATE = require_any_role("operator")


async def _get_project_or_404(db: AsyncSession, project_id: str) -> Project:
    project = (
        await db.execute(select(Project).where(Project.id == project_id))
    ).scalar_one_or_none()
    if project is None:
        raise ProjectNotFoundError(project_id)
    return project


@router.get("")
async def list_missions(
    status: str = Query(default=None),
    mine: bool = Query(default=False),
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict:
    missions = await mission_service.list_missions(
        db,
        owner_id=user.id if (mine and user) else None,
        status=status.upper() if status else None,
    )
    return {
        "missions": [mission_service.mission_public(m) for m in missions],
        "count": len(missions),
    }


@router.post("", status_code=201, response_model=MissionResponse)
async def create_mission(
    req: MissionCreateRequest,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(MUTATE),
) -> dict:
    """Create a mission record around an existing uploaded project."""
    await _get_project_or_404(db, req.project_id)
    mission = await mission_service.create_mission(
        db,
        owner=user,
        name=req.name,
        project_id=req.project_id,
        priority=req.priority,
        config={"description": req.description, "queue": {}},
    )
    metrics.inc("drone_mission_events_total", {"event": "created"})
    return mission_service.mission_public(mission)


@router.get("/{mission_id}", response_model=MissionResponse)
async def get_mission(
    mission_id: str,
    db: AsyncSession = Depends(get_session),
    _user: Optional[User] = Depends(get_optional_user),
) -> dict:
    mission = await mission_service.get_mission(db, mission_id)
    return mission_service.mission_public(mission)


@router.post("/{mission_id}/start")
async def start_mission(
    mission_id: str,
    req: MissionStartRequest,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(MUTATE),
) -> dict:
    """Validate project state, transition to QUEUED, enqueue a run."""
    mission = await mission_service.get_mission(db, mission_id)
    if not mission.project_id:
        raise HTTPException(status_code=400, detail="Mission has no bound project")
    if mission.status in (MISSION_PROCESSING, MISSION_QUEUED, MISSION_PAUSED, MISSION_RESUMING):
        raise HTTPException(status_code=409,
                            detail=f"Mission is already {mission.status}")

    from app.services.streaming_engine import engine

    engine.clear_cancel(mission.project_id)

    # CREATED / COMPLETED / FAILED / CANCELLED → QUEUED (restart or first run).
    event = "restarted" if mission.status != MISSION_CREATED else "queued"
    await mission_service.transition(db, mission, MISSION_QUEUED, user, event=event)
    job = await job_queue.enqueue(
        db,
        project_id=mission.project_id,
        mission_id=mission.id,
        kind="pipeline",
        payload={"plugins": req.plugins, "force": req.force},
        priority=req.priority if req.priority is not None else mission.priority,
    )
    metrics.inc("drone_mission_events_total", {"event": "started"})
    return {
        "mission_id": mission.id,
        "job_id": job.id,
        "status": mission.status,
        "message": f"Mission queued (job {job.id}) — the worker will drive the "
                   f"pipeline and stream progress on /api/dense/stream/{mission.project_id}",
    }


@router.post("/{mission_id}/pause")
async def pause_mission(
    mission_id: str,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(MUTATE),
) -> dict:
    mission = await mission_service.get_mission(db, mission_id)
    if mission.status not in (MISSION_PROCESSING, MISSION_QUEUED):
        raise HTTPException(status_code=409,
                            detail=f"Cannot pause a mission in {mission.status}")
    await mission_service.transition(db, mission, MISSION_PAUSED, user, event="paused")
    await job_queue.pause_job(db, mission)
    return {"mission_id": mission.id, "status": mission.status,
            "message": "Pause requested — pipeline stops at the next stage boundary"}


@router.post("/{mission_id}/resume")
async def resume_mission(
    mission_id: str,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(MUTATE),
) -> dict:
    mission = await mission_service.get_mission(db, mission_id)
    if mission.status != MISSION_PAUSED:
        raise HTTPException(status_code=409, detail="Only a paused mission can be resumed")
    # PAUSED → RESUMING → QUEUED; only then enqueue the fresh run so the
    # worker never claims a job while the mission is still RESUMING.
    await mission_service.transition(db, mission, MISSION_RESUMING, user, event="resuming")
    await mission_service.transition(db, mission, MISSION_QUEUED, user, event="queued")
    job = await job_queue.resume_job(db, mission)
    return {"mission_id": mission.id, "job_id": job.id, "status": mission.status}


@router.post("/{mission_id}/cancel")
async def cancel_mission(
    mission_id: str,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(MUTATE),
) -> dict:
    mission = await mission_service.get_mission(db, mission_id)
    if mission.status in (MISSION_COMPLETED, MISSION_CANCELLED):
        raise HTTPException(status_code=409, detail=f"Mission already {mission.status}")
    await mission_service.transition(db, mission, MISSION_CANCELLED, user, event="cancelled")
    await job_queue.cancel_job(db, mission)
    return {"mission_id": mission.id, "status": mission.status, "message": "Mission cancelled"}


@router.get("/{mission_id}/stream")
async def mission_stream(mission_id: str, db: AsyncSession = Depends(get_session)):
    """SSE stream of live pipeline progress for a mission's project.

    Replays recorded events for completed runs, then tails live events while
    the queue worker drives the pipeline. Mirrors the existing dense-stream
    event source but scoped to the mission record.
    """
    import json as _json

    from fastapi.responses import StreamingResponse

    mission = await mission_service.get_mission(db, mission_id)
    if not mission.project_id:
        raise HTTPException(status_code=400, detail="Mission has no bound project")

    from app.services.streaming_engine import engine

    async def event_source():
        async for event in engine.iter_events(mission.project_id):
            yield f"event: {event['type']}\ndata: {_json.dumps(event)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{mission_id}/events")
async def mission_events_route(
    mission_id: str,
    db: AsyncSession = Depends(get_session),
    _user: Optional[User] = Depends(get_optional_user),
) -> dict:
    await mission_service.get_mission(db, mission_id)
    return {"mission_id": mission_id, "events": await mission_service.mission_events(db, mission_id)}


@router.get("/{mission_id}/logs")
async def mission_logs(
    mission_id: str,
    db: AsyncSession = Depends(get_session),
    _user: Optional[User] = Depends(get_optional_user),
) -> dict:
    """Structured run history: pipeline/queue events from the streaming engine."""
    mission = await mission_service.get_mission(db, mission_id)
    from app.services.streaming_engine import engine

    events = engine.replay(mission.project_id) if mission.project_id else []
    report = {}
    if mission.project_id:
        from app.routes.digital_twin import _load_pipeline_report

        report = _load_pipeline_report(mission.project_id)
    return {
        "mission_id": mission.id,
        "project_id": mission.project_id,
        "events": events[-500:],
        "pipeline_report": report,
    }


@router.get("/{mission_id}/artifacts")
async def mission_artifacts(
    mission_id: str,
    db: AsyncSession = Depends(get_session),
    _user: Optional[User] = Depends(get_optional_user),
) -> dict:
    """Categorized artifact listing for the mission workspace."""
    mission = await mission_service.get_mission(db, mission_id)
    if not mission.project_id:
        return {"mission_id": mission.id, "project_id": None, "artifacts": [], "summary": {}}
    from app.services.storage_manager import summary

    return {
        "mission_id": mission.id,
        "project_id": mission.project_id,
        "artifacts": [_entry_dict(e) for e in catalog(mission.project_id)],
        "summary": summary(mission.project_id),
    }


def _entry_dict(entry) -> dict:
    return {
        "rel_path": entry.rel_path,
        "category": entry.category,
        "size_bytes": entry.size_bytes,
        "is_dir": entry.is_dir,
        "children": [_entry_dict(c) for c in entry.children] if entry.children else [],
    }


@router.post("/{mission_id}/cleanup")
async def mission_cleanup(
    mission_id: str,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(MUTATE),
    force: bool = Query(default=False),
) -> dict:
    """Delete re-generable intermediate artifacts older than the retention
    window (or all, with force) — never raw inputs or final outputs."""
    mission = await mission_service.get_mission(db, mission_id)
    if not mission.project_id:
        return {"mission_id": mission.id, "removed": [], "freed_bytes": 0}
    result = cleanup_intermediates(mission.project_id, force=force)
    log.info("mission_cleanup", mission_id=mission.id, actor=user.email,
             removed=len(result["removed"]))
    return {"mission_id": mission.id, **result}


@router.delete("/{mission_id}", status_code=200)
async def delete_mission(
    mission_id: str,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(MUTATE),
    purge_workspace: bool = Query(default=True),
) -> dict:
    """Delete the mission (and optionally its workspace). Admin/operator only."""
    if user.role.lower() not in ("admin", "operator"):
        raise HTTPException(status_code=403, detail="Requires admin or operator")
    mission = await mission_service.get_mission(db, mission_id)
    project_id = mission.project_id
    if purge_workspace and project_id:
        from app.services.storage_manager import purge_workspace as _purge

        _purge(project_id)
    await db.delete(mission)  # cascade: events + jobs
    await db.flush()
    log.info("mission_deleted", mission_id=mission_id, actor=user.email,
             purge_workspace=purge_workspace)
    return {"mission_id": mission_id, "deleted": True, "purged_workspace": purge_workspace}
