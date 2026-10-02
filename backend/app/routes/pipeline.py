"""Autonomous pipeline API routes.

POST /api/pipeline/start/{job_id}   — enqueue the full chain (frames → sparse
                                      → depth → dense → georef) on the durable
                                      job queue, resuming from whatever already
                                      exists in the workspace. Returns
                                      immediately; the queue worker executes
                                      the pipeline.
GET  /api/pipeline/status/{job_id}  — stage state, timeline profile, resume map
POST /api/pipeline/cancel/{job_id}  — request cancellation (checked between
                                      stages and depth frames)

Live progress for the run streams over the existing SSE endpoint
``GET /api/dense/stream/{job_id}`` (events are job-scoped on the same engine).
"""

from __future__ import annotations

import dataclasses
import json

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.db.models import Project
from app.exceptions import ProjectNotFoundError
from app.logging_config import get_logger
from app.schemas.pipeline import (
    PipelineCancelResponse,
    PipelineStartRequest,
    PipelineStartResponse,
    PipelineStatusResponse,
    RefineOptions,
)
from app.services import job_queue
from app.services.depth_generator import StereoParams
from app.services.depth_refinement import RefineParams
from app.services.pipeline_orchestrator import STAGES, PipelineRequest
from app.services.streaming_engine import engine

log = get_logger("drone_recon.routes.pipeline")

router = APIRouter(tags=["pipeline"])


@router.post("/api/pipeline/start/{job_id}", response_model=PipelineStartResponse)
async def start_pipeline(
    job_id: str,
    req: PipelineStartRequest,
    db: AsyncSession = Depends(get_session),
) -> PipelineStartResponse:
    """Enqueue (or re-enqueue) the autonomous pipeline for a project.

    Long reconstruction must never run inside the HTTP request: browsers cut
    the connection after ~1–5 minutes and the user sees a network failure even
    though the backend is still healthy. The durable job queue owns execution
    (the same executor the data-video missions use); this endpoint validates
    the request, persists the run parameters, and returns immediately with
    ``status="queued"``.
    """
    project = await _get_project(job_id, db)
    engine.clear_cancel(job_id)
    engine.publish(job_id, "pipeline_started", {"requested_stages": req.force})

    # Resolve settings-backed defaults here (validated request → service
    # request), then hand the worker a JSON-safe dict of the same values.
    service_request = _to_service_request(req)
    if not service_request.force:
        # Retry semantics: re-running a FAILED run must re-run the failed
        # stage and its downstream, or cached-but-stale artifacts make the
        # retry a no-op loop. The cached skip decision lives in the
        # orchestrator; this only seeds ``force`` from the run's own record.
        prior = _load_report(job_id)
        if prior and prior.get("status") == "failed":
            retry_force = _retry_force_stages(prior)
            if retry_force:
                service_request.force = retry_force
                log.info("retry_force_resolved", job_id=job_id, force=retry_force,
                         note="failed run retried without explicit force — re-running the failed stage and downstream")
    await job_queue.enqueue(
        db,
        project_id=job_id,
        kind="pipeline",
        payload=json.loads(json.dumps(dataclasses.asdict(service_request))),
        priority=5,
    )
    await db.commit()

    return PipelineStartResponse(
        job_id=job_id,
        status="queued",
        message=(
            "Pipeline queued — track progress at /api/pipeline/status/{job_id} "
            "(the queue worker runs reconstruction; the browser connection is "
            "no longer held open)"
        ).format(job_id=job_id),
    )


@router.get("/api/pipeline/status/{job_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> PipelineStatusResponse:
    """Current pipeline state: stage timeline, profile, and resume map."""
    await _get_project(job_id, db)
    report = _load_report(job_id)
    live = _live_status(job_id)

    # Live events win in two cases: no finished-run report yet, or a run
    # started again after a finished one (e.g. retry after failure) — the
    # stale report file must not mask the in-progress rerun.
    report_is_stale = bool(live) and float(live.get("_event_ts", 0.0)) > float(
        report.get("_written_at", 0.0)
    )
    if not report or report_is_stale:
        if live:
            live.pop("_event_ts", None)
            return PipelineStatusResponse(job_id=job_id, resume={}, **live)
        # A never-run job still reports what is already available (resume map).
        from app.config.settings import settings
        from app.services.pipeline_orchestrator import ARTIFACT_CHECKS, STAGES

        workspace = settings.storage.project_dir(job_id)
        return PipelineStatusResponse(
            job_id=job_id,
            status="not_run",
            stages={},
            profile={},
            resume={s: bool(ARTIFACT_CHECKS.get(s, lambda _w: False)(workspace)) for s in STAGES},
        )
    report = _strip_internal(report)
    return PipelineStatusResponse(
        job_id=job_id,
        status=report.get("status", "unknown"),
        run_time_ms=report.get("run_time_ms", 0.0),
        error=report.get("error", ""),
        stages=report.get("stages", {}),
        profile=report.get("profile", {}),
        resume=report.get("resume", {}),
    )


def _strip_internal(report: dict) -> dict:
    """Drop bookkeeping keys that must not leak into API payloads."""
    return {k: v for k, v in report.items() if not k.startswith("_")}


@router.post("/api/pipeline/cancel/{job_id}", response_model=PipelineCancelResponse)
async def cancel_pipeline(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> PipelineCancelResponse:
    """Request cancellation of a running pipeline."""
    await _get_project(job_id, db)
    engine.request_cancel(job_id)
    return PipelineCancelResponse(job_id=job_id, cancelled=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _retry_force_stages(report: dict) -> list[str]:
    """Stages a RETRY of a failed run must re-run: the failed stage itself
    plus everything downstream.

    Without this, Retry with an empty ``force`` list skips every stage whose
    artifacts exist — including the one that just failed — and the pipeline
    re-fails identically: a no-op retry loop (sunset_06cfea re-failed dense
    on a stale 1-map depth dir on every retry). Upstream COMPLETED stages
    keep their cache: a retry must not re-run sparse when frames+sparse are
    healthy and dense failed. Completed stages are not re-run even when the
    fingerprint would allow reuse of only part of their output — the run
    either reproduces its good stages from cache or re-runs them by force,
    never half-and-half.
    """
    failed = next(
        (
            name
            for name, st in report.get("stages", {}).items()
            if isinstance(st, dict) and st.get("status") == "failed"
        ),
        None,
    )
    if failed not in STAGES:
        return []
    return list(STAGES[STAGES.index(failed):])


def _to_service_request(req: PipelineStartRequest) -> PipelineRequest:
    stereo = None
    if req.stereo is not None:
        stereo = StereoParams(
            min_disparity=req.stereo.min_disparity
            if req.stereo.min_disparity is not None
            else 0,
            num_disparities=req.stereo.num_disparities or 96,
            block_size=req.stereo.block_size or 5,
            uniqueness_ratio=req.stereo.uniqueness_ratio or 10,
        )
    refine = None
    if req.refine is not None:
        refine = _refine_options(req.refine)
    gps = {"lat": req.gps.lat, "lon": req.gps.lon, "alt": req.gps.alt} if req.gps else None
    return PipelineRequest(
        extraction_mode=req.extraction_mode,
        every_n=req.every_n,
        target_fps=req.target_fps,
        top_percent=req.top_percent,
        quality_threshold=req.quality_threshold,
        depth_backend=req.depth_backend,
        frame_stride=req.frame_stride,
        max_depth_views=req.max_depth_views,
        force=list(req.force),
        stereo=stereo,
        refine=refine,
        gps=gps,
        telemetry_csv=req.telemetry_csv,
    )


def _refine_options(opts: RefineOptions) -> RefineParams:
    params = RefineParams.from_settings()
    for name in ("median", "bilateral", "edge_preserving", "hole_fill"):
        value = getattr(opts, name)
        if value is not None:
            setattr(params, name, value)
    if opts.median_k is not None:
        params.median_k = opts.median_k
    return params


async def _get_project(job_id: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == job_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ProjectNotFoundError(job_id)
    return project


def _load_report(job_id: str) -> dict:
    import json

    from app.config.settings import settings

    path = settings.storage.project_dir(job_id) / "pipeline_report.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                data["_written_at"] = path.stat().st_mtime
            return data
    return {}


def _live_status(job_id: str) -> dict | None:
    """Derive a mid-run status from the streaming engine's event history.

    ``pipeline_report.json`` is only written when the run finishes, so while
    the pipeline is executing the polling UI would otherwise see ``not_run``
    for the entire run. The last ``stage:<name>`` event per stage carries the
    real status and progress fraction — displayed verbatim, never invented.
    Returns None when the history holds nothing for this job.
    """

    from app.services.streaming_engine import engine

    latest: dict[str, dict] = {}
    order: list[str] = []
    last_event_ts = 0.0
    failure_error = ""
    for ev in engine.replay(job_id):
        last_event_ts = max(last_event_ts, float(ev.get("timestamp", 0.0) or 0.0))
        etype = ev.get("type", "")
        if not etype.startswith("stage:"):
            continue
        name = etype.split(":", 1)[1]
        if name not in latest:
            order.append(name)
        latest[name] = ev.get("payload") or {}
        if latest[name].get("status") == "failed":
            failure_error = str(latest[name].get("error", ""))

    if not latest:
        return None

    running = [n for n, p in latest.items() if p.get("status") == "running"]
    failed = [n for n, p in latest.items() if p.get("status") == "failed"]
    cancelled = [n for n, p in latest.items() if p.get("status") == "cancelled"]
    if failed:
        status = "failed"
    elif cancelled:
        status = "cancelled"
    elif running:
        status = "running"
    elif all(p.get("status") in {"completed", "skipped"} for p in latest.values()):
        status = "completed"
    else:
        # Stages seen but none running yet: either mid-restart of a stage or done.
        status = "running"

    return {
        "status": status,
        "error": failure_error if failed else "",
        "stages": {n: latest[n] for n in order},
        "profile": {},
        "run_time_ms": 0.0,
        "_event_ts": last_event_ts,
    }
