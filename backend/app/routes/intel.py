"""Geospatial-intelligence (Phase 8) API routes.

POST /api/intel/run/{job_id}          — run the intelligence chain (INTEL_CHAIN)
GET  /api/intel/status/{job_id}       — stage state + resume map
GET  /api/intel/artifact/{job_id}/{name}  — raw intel artifact (report, risk, …)
GET  /api/intel/dashboard/{job_id}    — dashboard payload JSON
GET  /api/intel/dashboard/page/{job_id}   — standalone HTML dashboard
GET  /api/intel/report/{job_id}       — disaster report (json|md|html)
POST /api/intel/copilot/{job_id}      — grounded natural-language answers (RAG)
POST /api/intel/change/{job_id}       — compare vs a baseline mission
"""

from __future__ import annotations

import asyncio
import json
import re

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.engine import get_session
from app.exceptions import BadRequestError, ProcessingError
from app.logging_config import get_logger

#: shared route helpers live with the digital-twin routes (same job model)
from app.routes.digital_twin import _get_project, _load_pipeline_report
from app.schemas.intel import (
    ChangeCompareRequest,
    IntelQueryRequest,
    IntelRunRequest,
    IntelRunResponse,
    IntelStatusResponse,
)
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.pipeline_stage import INTEL_CHAIN, discover, get_stage

log = get_logger("drone_recon.routes.intel")

router = APIRouter(tags=["intel"])

_ARTIFACT_NAME = re.compile(r"^[a-z_]+$")
_REPORT_FORMATS = {"json": "disaster_report.json", "md": "disaster_report.md",
                   "html": "disaster_report.html"}


@router.post("/api/intel/run/{job_id}", response_model=IntelRunResponse)
async def run_intel(
    job_id: str,
    req: IntelRunRequest,
    db: AsyncSession = Depends(get_session),
) -> IntelRunResponse:
    """Run (or resume) the intelligence chain for a project's digital twin."""
    project = await _get_project(job_id, db)
    discover()
    plugins = list(req.plugins) or list(INTEL_CHAIN)
    try:
        for p in plugins:
            get_stage(p)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc

    from app.services.streaming_engine import engine

    engine.clear_cancel(job_id)
    engine.publish(job_id, "intel_run_started", {"plugins": plugins})
    pipeline_req = PipelineRequest(force=list(req.force), plugins=plugins)
    try:
        report = await asyncio.to_thread(run_autonomous_pipeline, job_id, pipeline_req)
    except Exception as exc:  # pipeline wraps stage failures already
        raise ProcessingError("intel", detail=str(exc)) from exc

    status = report.get("status", "unknown")
    project.status = {"completed": "intel_completed", "cancelled": "cancelled"}.get(status, "failed")
    if status == "failed":
        project.error_message = report.get("error", "")
    failed = [n for n, s in report.get("stages", {}).items() if s.get("status") == "failed"]
    project.current_stage = failed[0] if failed else "intelligence"
    await db.flush()

    message = {
        "completed": "Intelligence complete — environment, damage, risk, "
                     "recommendations, RAG index and reports are ready",
        "cancelled": "Run cancelled — rerun to resume from the last completed stage",
        "failed": f"Run failed at '{failed[0] if failed else '?'}': "
                  f"{report.get('error', '')[:200]}",
    }.get(status, "Run finished")
    return IntelRunResponse(
        job_id=job_id, status=status, run_time_ms=report.get("run_time_ms", 0.0),
        error=report.get("error", ""), stages=report.get("stages", {}), message=message,
    )


@router.get("/api/intel/status/{job_id}", response_model=IntelStatusResponse)
async def intel_status(job_id: str, db: AsyncSession = Depends(get_session)) -> IntelStatusResponse:
    await _get_project(job_id, db)
    report = _load_pipeline_report(job_id)
    if not report:
        return IntelStatusResponse(job_id=job_id, status="not_run")
    return IntelStatusResponse(
        job_id=job_id, status=report.get("status", "unknown"),
        run_time_ms=report.get("run_time_ms", 0.0), error=report.get("error", ""),
        stages=report.get("stages", {}), resume=report.get("resume", {}),
    )


@router.get("/api/intel/artifact/{job_id}/{name}")
async def intel_artifact(
    job_id: str, name: str, db: AsyncSession = Depends(get_session),
) -> dict:
    """Return one raw intel artifact (environment, damage, risk, alerts, …)."""
    await _get_project(job_id, db)
    if not _ARTIFACT_NAME.match(name):
        raise BadRequestError("artifact name must be [a-z_]+")
    path = settings.storage.project_dir(job_id) / "intel" / f"{name}.json"
    if not path.exists():
        raise BadRequestError(f"no intel artifact '{name}' — run the intelligence chain first")
    return json.loads(path.read_text())


@router.get("/api/intel/dashboard/{job_id}")
async def intel_dashboard(job_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    await _get_project(job_id, db)
    path = settings.storage.project_dir(job_id) / "intel" / "dashboard.json"
    if not path.exists():
        raise BadRequestError("no dashboard payload — run the intelligence chain first")
    return json.loads(path.read_text())


@router.get("/api/intel/dashboard/page/{job_id}")
async def intel_dashboard_page(job_id: str, db: AsyncSession = Depends(get_session)):
    await _get_project(job_id, db)
    path = settings.storage.project_dir(job_id) / "intel" / "dashboard.html"
    if not path.exists():
        raise BadRequestError("no dashboard page — run the intelligence chain first")
    return FileResponse(path, media_type="text/html", filename="dashboard.html")


@router.get("/api/intel/report/{job_id}")
async def intel_report(
    job_id: str,
    format: str = Query(default="json", pattern="^(json|md|html)$"),
    db: AsyncSession = Depends(get_session),
):
    """Download the disaster report in the requested format."""
    await _get_project(job_id, db)
    path = settings.storage.project_dir(job_id) / "intel" / _REPORT_FORMATS[format]
    if not path.exists():
        raise BadRequestError("no report — run the intelligence chain first")
    media = {"json": "application/json", "md": "text/markdown", "html": "text/html"}[format]
    return FileResponse(path, media_type=media, filename=path.name)


@router.post("/api/intel/copilot/{job_id}")
async def intel_copilot(
    job_id: str, req: IntelQueryRequest, db: AsyncSession = Depends(get_session),
) -> dict:
    """Grounded natural-language answer from this mission's RAG index."""
    await _get_project(job_id, db)
    from app.services.rag_engine import query_workspace

    return query_workspace(settings.storage.project_dir(job_id), req.query)


@router.post("/api/intel/change/{job_id}")
async def intel_change(
    job_id: str, req: ChangeCompareRequest, db: AsyncSession = Depends(get_session),
) -> dict:
    """Configure and run change detection against a baseline mission."""
    await _get_project(job_id, db)
    await _get_project(req.baseline_job, db)
    workspace = settings.storage.project_dir(job_id)
    intel_dir = workspace / "intel"
    intel_dir.mkdir(parents=True, exist_ok=True)
    (intel_dir / "change_config.json").write_text(json.dumps({
        "baseline_job": req.baseline_job, "voxel_m": req.voxel_m,
    }))
    discover()
    from app.services.streaming_engine import engine

    engine.clear_cancel(job_id)
    engine.publish(job_id, "change_detection_started", {"baseline": req.baseline_job})
    # Change feeds the RAG index and reports, so rebuild them after the diff or
    # the artifacts on disk would keep the pre-change state.
    downstream = ["rag_index", "report_generation"]
    pipeline_req = PipelineRequest(plugins=["change_detection"] + downstream,
                                   force=["change_detection"] + downstream)
    report = await asyncio.to_thread(run_autonomous_pipeline, job_id, pipeline_req)
    stage = (report.get("stages", {}) or {}).get("change_detection", {})
    return {
        "job_id": job_id,
        "baseline_job": req.baseline_job,
        "status": stage.get("status", report.get("status", "unknown")),
        "detail": stage.get("detail", {}),
        "error": stage.get("error") or report.get("error", ""),
        "rebuild": {name: (report.get("stages", {}) or {}).get(name, {})
                     .get("status", "unknown") for name in downstream},
    }
