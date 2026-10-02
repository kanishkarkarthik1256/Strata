"""In-process durable job queue (Phase 10).

The queue persists jobs in the ``queue_jobs`` table and executes them with a
small worker loop (started from the app lifespan). Priorities are honoured
(higher number first; FIFO within a priority), jobs retry up to their
``max_attempts``, and a crash leaves the row in ``running`` — the worker
re-adopts such jobs on startup (``recover_interrupted``) and re-runs them,
which resumes the underlying pipeline from its last completed stage because
the orchestrator is artifact-resumable.

Executors are looked up by ``kind`` in :data:`EXECUTORS`. The only
production executor is ``pipeline``: it drives the existing
:func:`run_autonomous_pipeline` (frame extraction → reconstruction →
twin/intel/planning chains) inside a worker thread, publishing progress over
the shared streaming engine. Cancellation/pause are expressed through that
engine's per-job cancel flag, which the orchestrator already polls between
stages.

The worker is a single asyncio loop bounded by ``platform.queue_max_workers``
concurrent threads — resource-aware without adding infrastructure.

Mission state ownership: the queue never writes ``missions.status`` or
``mission_events`` directly. Every mission transition — including the
worker-driven ones (claim → PROCESSING, retry, terminal outcomes) — goes
through :func:`app.services.mission_service.transition`, so the state machine
and the audit trail have exactly one owner.

Worker/settlement protocol: the claim (job ``running`` + mission PROCESSING)
is committed *before* the executor is invoked, so a long pipeline run does
not hold the SQLite write lock and the pause/cancel/resume routes can commit
their own row flips concurrently. When the executor returns, the worker
re-reads the job row: pause marks it ``paused``, cancel marks it
``cancelled``, and a resume closes it as ``completed`` and enqueues a fresh
job — so any of those states means the executor result is *stale* and the
mission must not be rewritten. Only a row still ``running`` lets the result
drive the mission transition.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.models import (
    MISSION_CANCELLED,
    MISSION_COMPLETED,
    MISSION_FAILED,
    MISSION_PROCESSING,
    MISSION_QUEUED,
    Mission,
    Project,
    QueueJob,
)
from app.logging_config import get_logger

log = get_logger("drone_recon.services.job_queue")

Executor = Callable[[QueueJob, AsyncSession], Awaitable[dict]]

if TYPE_CHECKING:
    from app.services.pipeline_orchestrator import PipelineRequest

#: kind → async executor. Production registers ``pipeline`` at import time;
#: tests register lightweight executors to exercise queue mechanics.
EXECUTORS: dict[str, Executor] = {}


#: payload keys handled specially when building a PipelineRequest (never
#: passed through as dataclass fields — they'd collide with the explicit
#: defaults below).
_PAYLOAD_NON_FIELD_KEYS = frozenset({"plugins", "force"})


def _request_from_payload(payload: dict) -> "PipelineRequest":
    """Build the per-run PipelineRequest from a queue payload.

    Any PipelineRequest field the caller set is honoured verbatim; ``plugins``
    and ``force`` get list coercion; everything else keeps its default.
    Nested option dataclasses (stereo/refine) arrive as JSON dicts from the
    REST route and are rebuilt here.
    """
    from app.services.pipeline_orchestrator import PipelineRequest

    fields = PipelineRequest.__dataclass_fields__
    kwargs = {
        k: v
        for k, v in payload.items()
        if k in fields and k not in _PAYLOAD_NON_FIELD_KEYS
    }
    if isinstance(kwargs.get("stereo"), dict):
        from app.services.depth_generator import StereoParams

        kwargs["stereo"] = StereoParams(**kwargs["stereo"])
    if isinstance(kwargs.get("refine"), dict):
        from app.services.depth_refinement import RefineParams

        kwargs["refine"] = RefineParams(**kwargs["refine"])
    return PipelineRequest(
        **kwargs,
        plugins=list(payload.get("plugins", [])),
        force=list(payload.get("force", [])),
    )


async def _pipeline_executor(job: QueueJob, db: AsyncSession) -> dict:
    """Run the existing autonomous pipeline for the job's project."""
    import asyncio as _aio

    from app.services.pipeline_orchestrator import run_autonomous_pipeline
    from app.services.streaming_engine import engine

    payload = json.loads(job.payload_json or "{}")
    engine.clear_cancel(job.project_id or "")
    engine.publish(job.project_id or "", "queue_job_started",
                   {"job_id": job.id, "kind": job.kind})
    request = _request_from_payload(payload)
    return await _aio.to_thread(run_autonomous_pipeline, job.project_id, request)


def register_default_executors() -> None:
    """Register production executors (idempotent)."""
    EXECUTORS.setdefault("pipeline", _pipeline_executor)


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------


async def enqueue(
    db: AsyncSession,
    *,
    project_id: str,
    kind: str = "pipeline",
    payload: Optional[dict] = None,
    priority: int = 5,
    mission_id: Optional[str] = None,
    max_attempts: Optional[int] = None,
) -> QueueJob:
    register_default_executors()
    if kind not in EXECUTORS:
        raise ValueError(f"No executor registered for kind '{kind}'")
    job = QueueJob(
        mission_id=mission_id,
        project_id=project_id,
        kind=kind,
        status="queued",
        priority=priority,
        payload_json=json.dumps(payload or {}),
        max_attempts=max_attempts if max_attempts is not None
        else settings.platform.queue_max_attempts,
    )
    db.add(job)
    await db.flush()
    log.info("job_enqueued", job_id=job.id, project_id=project_id,
             kind=kind, priority=priority, mission_id=mission_id)
    return job


async def recover_interrupted(db: AsyncSession) -> int:
    """Re-adopt jobs a crashed process left in ``running``.

    Missions stuck in PROCESSING are moved back to QUEUED so re-running the
    job resumes from the last completed stage (artifact-based resume).
    Returns the number of recovered jobs.
    """
    count = 0
    rows = (
        await db.execute(select(QueueJob).where(QueueJob.status == "running"))
    ).scalars().all()
    for job in rows:
        job.status = "queued"
        count += 1
        if job.mission_id:
            mission = await _mission_of(db, job)
            if mission is not None and mission.status == MISSION_PROCESSING:
                await _transition_mission(db, job, MISSION_QUEUED, event="recovered")
    await db.flush()
    if count:
        log.warning("recovered_interrupted_jobs", count=count)
    return count


async def _next_queued(db: AsyncSession) -> Optional[QueueJob]:
    return (
        await db.execute(
            select(QueueJob)
            .where(QueueJob.status == "queued")
            .order_by(QueueJob.priority.desc(), QueueJob.created_at.asc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def _mission_of(db: AsyncSession, job: QueueJob) -> Optional[Mission]:
    if not job.mission_id:
        return None
    return (
        await db.execute(select(Mission).where(Mission.id == job.mission_id))
    ).scalar_one_or_none()


async def _transition_mission(
    db: AsyncSession,
    job: QueueJob,
    to_status: str,
    *,
    event: str,
    error: Optional[str] = None,
) -> None:
    """Apply a worker-driven mission transition via the single owner.

    Invalid transitions (the operator moved the mission elsewhere while the
    job ran) are logged and skipped — never raised into the worker loop.
    """
    from app.services.mission_service import transition as mission_transition

    mission = await _mission_of(db, job)
    if mission is None:
        return
    detail = {"job_id": job.id}
    if error:
        detail["error"] = error
    try:
        await mission_transition(db, mission, to_status, actor_name="worker",
                                 event=event, detail=detail)
    except Exception:
        log.warning("worker_transition_skipped", job_id=job.id,
                    mission_id=mission.id, to_status=to_status,
                    current_status=mission.status)


async def _refresh_or_none(db: AsyncSession, job: QueueJob) -> Optional[QueueJob]:
    """Re-read the job row from the DB. Returns None if the row is gone
    (e.g. the mission was deleted while its executor was running)."""
    try:
        await db.refresh(job)
    except Exception:  # pragma: no cover - row deleted concurrently
        return None
    return job


async def _settle_executor_result(
    db: AsyncSession, job: QueueJob, report: dict
) -> None:
    """Apply an executor report to the job row and — only when this job is
    still the mission's live run — to the mission.

    The operator's pause/cancel/resume already flipped the job row while the
    executor ran: those states make the result stale, so the mission row is
    left exactly where the routes put it.
    """
    if await _refresh_or_none(db, job) is None:
        return
    now = datetime.now(timezone.utc)

    if job.status != "running":
        # Paused (awaiting resume), cancelled, or closed as completed by a
        # resume that enqueued a newer job — the mission is already where the
        # operator put it.
        log.info("stale_executor_result_ignored", job_id=job.id,
                 row_status=job.status, result=report.get("status"))
        return

    mission = await _mission_of(db, job)
    current = mission.status if mission is not None else None
    status = report.get("status", "completed")

    # Keep the project row in step with the terminal outcome (the sync REST
    # path used to do this inline; the queue is the executor now).
    project: Project | None = None
    if job.project_id:
        project = (
            await db.execute(select(Project).where(Project.id == job.project_id))
        ).scalar_one_or_none()
    if project is not None:
        if status == "completed":
            project.status = "pipeline_completed"
            project.error_message = None
        elif status == "cancelled":
            project.status = "cancelled"
        else:
            project.status = "failed"
            project.error_message = report.get("error") or "pipeline reported failure"
        failed = [n for n, s in (report.get("stages") or {}).items()
                  if isinstance(s, dict) and s.get("status") == "failed"]
        project.current_stage = failed[0] if failed else project.current_stage

    if status == "cancelled":
        if current == MISSION_PROCESSING:
            job.status = "cancelled"
            job.finished_at = now
            await _transition_mission(db, job, MISSION_CANCELLED, event="cancelled")
        else:
            # Row still running but the mission moved on underneath (rare) —
            # never cancel a mission that is no longer PROCESSING.
            job.status = "cancelled" if current == MISSION_CANCELLED else "completed"
            job.finished_at = now
        return

    if status == "failed":
        job.error = report.get("error") or "pipeline reported failure"
        if current == MISSION_PROCESSING:
            job.status = "failed"
            job.finished_at = now
            await _transition_mission(db, job, MISSION_FAILED, event="failed",
                                      error=job.error)
        else:
            job.status = "failed"
            job.finished_at = now
        return

    # completed
    if current == MISSION_PROCESSING:
        job.status = "completed"
        job.finished_at = now
        await _transition_mission(db, job, MISSION_COMPLETED, event="completed")
    else:
        job.status = "completed"
        job.finished_at = now


async def process_next(db: AsyncSession) -> Optional[QueueJob]:
    """Run the highest-priority queued job to completion (used by worker + tests).

    Handles: running state, retry on failure, mission status propagation.
    Returns the processed job or None when the queue is empty.
    """
    register_default_executors()
    job = await _next_queued(db)
    if job is None:
        return None
    executor = EXECUTORS.get(job.kind)
    if executor is None:
        job.status = "failed"
        job.error = f"No executor registered for kind '{job.kind}'"
        job.finished_at = datetime.now(timezone.utc)
        await _transition_mission(db, job, MISSION_FAILED, event="failed",
                                  error=job.error)
        await db.flush()
        return job

    job.status = "running"
    job.started_at = datetime.now(timezone.utc)
    job.attempts += 1
    await _transition_mission(db, job, MISSION_PROCESSING, event="processing")
    await db.commit()  # release the write lock before a long executor run

    try:
        report = await executor(job, db)
    except Exception as exc:  # executor raised — retry or fail
        log.warning("job_executor_error", job_id=job.id, error=str(exc))
        if await _refresh_or_none(db, job) is None:
            return job
        if job.status != "running":
            log.info("executor_error_ignored_stale", job_id=job.id,
                     row_status=job.status)
            return job
        job.error = str(exc)
        mission = await _mission_of(db, job)
        current = mission.status if mission is not None else None
        if job.attempts < job.max_attempts and current == MISSION_PROCESSING:
            job.status = "queued"  # retry (same priority/kind)
            job.started_at = None
            await _transition_mission(db, job, MISSION_QUEUED, event="retry")
        else:
            job.status = "failed"
            job.finished_at = datetime.now(timezone.utc)
            if current == MISSION_PROCESSING:
                await _transition_mission(db, job, MISSION_FAILED, event="failed",
                                          error=str(exc))
        await db.flush()
        return job

    await _settle_executor_result(db, job, report)
    await db.flush()
    try:
        from app.services.metrics import metrics

        metrics.inc("drone_queue_jobs_total", {"kind": job.kind, "result": job.status})
    except Exception:  # pragma: no cover - metrics must never break the worker
        pass
    return job


async def pause_job(db: AsyncSession, mission: Mission) -> None:
    """Pause a mission: stop the current run at its next stage boundary.

    The orchestrator is *not* killed mid-stage — the streaming engine cancel
    flag is set and the pipeline returns ``cancelled`` between stages; the
    mission then sits PAUSED and may be resumed (re-enqueue re-runs from the
    last completed stage because the orchestrator is artifact-resumable).

    The PAUSED mission transition is applied by the caller through
    :func:`mission_service.transition` (single owner); this only requests the
    engine cancel and marks the affected job rows paused.
    """
    from app.services.streaming_engine import engine

    if mission.project_id:
        engine.request_cancel(mission.project_id)
    jobs = (
        await db.execute(
            select(QueueJob).where(QueueJob.mission_id == mission.id,
                                   QueueJob.status.in_(("queued", "running")))
        )
    ).scalars().all()
    for job in jobs:
        job.status = "paused"
    await db.flush()


async def cancel_job(db: AsyncSession, mission: Mission) -> None:
    """Cancel a mission (terminal). The CANCELLED transition is applied by the
    caller through :func:`mission_service.transition`; this requests the
    engine cancel and marks job rows cancelled so no queued job is picked up.
    """
    from app.services.streaming_engine import engine

    if mission.project_id:
        engine.request_cancel(mission.project_id)
    jobs = (
        await db.execute(
            select(QueueJob).where(QueueJob.mission_id == mission.id,
                                   QueueJob.status.in_(("queued", "running", "paused")))
        )
    ).scalars().all()
    for job in jobs:
        job.status = "cancelled"
        job.finished_at = datetime.now(timezone.utc)
    await db.flush()


async def resume_job(db: AsyncSession, mission: Mission) -> QueueJob:
    """Resume a paused mission: clear cancel, re-enqueue a fresh run job.

    Because the orchestrator skips stages whose artifacts already exist, the
    re-run continues from where the pause stopped instead of restarting.
    The mission transitions (PAUSED → RESUMING → QUEUED) are applied by the
    caller through :func:`mission_service.transition`; this closes out the
    paused job row and enqueues the fresh run. The new job inherits the kind
    + payload of the paused job so an echo/test run resumes as the same kind
    rather than falling back to ``pipeline``.
    """
    from app.services.streaming_engine import engine

    if mission.project_id:
        engine.clear_cancel(mission.project_id)
    jobs = (
        await db.execute(
            select(QueueJob).where(QueueJob.mission_id == mission.id,
                                   QueueJob.status == "paused")
        )
    ).scalars().all()
    kind = "pipeline"
    payload: dict = {}
    for job in jobs:
        if job.kind and job.kind not in ("pipeline",):
            kind = job.kind
        if job.payload_json:
            try:
                payload = json.loads(job.payload_json)
            except ValueError:
                payload = {}
        job.status = "completed"
        job.finished_at = datetime.now(timezone.utc)
    await db.flush()
    return await enqueue(
        db,
        project_id=mission.project_id or "",
        mission_id=mission.id,
        kind=kind,
        payload=payload,
        priority=mission.priority,
    )


# ---------------------------------------------------------------------------
# Background worker (lifespan)
# ---------------------------------------------------------------------------


async def worker_loop(stop: asyncio.Event) -> None:
    """Poll the queue forever, executing at most ``queue_max_workers`` jobs
    concurrently. Call ``stop.set()`` to shut down cleanly."""
    from app.db.engine import get_session
    from app.services.metrics import metrics

    log.info("queue_worker_started",
             max_workers=settings.platform.queue_max_workers,
             poll_seconds=settings.platform.queue_poll_seconds)

    async def _process_one() -> None:
        """Run one queued job inside a session."""
        async for db in get_session():
            try:
                await process_next(db)
            except Exception:  # pragma: no cover - never kill the loop on one job
                log.exception("queue_process_error")

    # Startup recovery: re-adopt jobs a crashed process left running. Runs
    # once (not per cycle) so a concurrent worker's live job is never stolen.
    try:
        async for db in get_session():
            await recover_interrupted(db)
    except Exception:  # pragma: no cover - best effort
        log.exception("queue_recover_failed")

    workers: set[asyncio.Task] = set()
    try:
        while not stop.is_set():
            # Top up to max_workers concurrent sessions; each drains one job.
            while len(workers) < settings.platform.queue_max_workers and not stop.is_set():
                task = asyncio.create_task(_process_one())
                workers.add(task)
                task.add_done_callback(workers.discard)
            metrics.set_gauge("drone_queue_worker_alive", 1)
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.platform.queue_poll_seconds)
            except asyncio.TimeoutError:
                pass
    finally:
        metrics.set_gauge("drone_queue_worker_alive", 0)
        for task in list(workers):
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        log.info("queue_worker_stopped")
