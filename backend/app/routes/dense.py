"""Dense reconstruction API routes.

POST /api/dense/start/{job_id}       — run dense reconstruction (fusion → digital twin)
GET  /api/dense/status/{job_id}      — run status
GET  /api/dense/statistics/{job_id}  — quality statistics + measurements + intelligence
GET  /api/dense/stream/{job_id}      — SSE stream of live pipeline events
GET  /api/dense/confidence/{job_id}  — per-class confidence summary
GET  /api/dense/download/{job_id}    — export the dense model (?format=ply|xyz|pcd|las)
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.engine import get_session
from app.db.models import Export, Model3D, Project
from app.exceptions import BadRequestError, ProjectNotFoundError
from app.logging_config import get_logger
from app.schemas.dense import (
    DenseConfidenceResponse,
    DenseStartRequest,
    DenseStartResponse,
    DenseStatisticsResponse,
    DenseStatusResponse,
)
from app.services.pointcloud import CONTENT_TYPES, EXPORT_EXTENSIONS
from app.services.streaming_engine import engine

log = get_logger("drone_recon.routes.dense")

router = APIRouter(tags=["dense"])


@router.post("/api/dense/start/{job_id}", response_model=DenseStartResponse)
async def start_dense(
    job_id: str,
    req: DenseStartRequest,
    db: AsyncSession = Depends(get_session),
) -> DenseStartResponse:
    """Run the dense reconstruction pipeline for an uploaded project."""
    project = await _get_project(job_id, db)
    params = _params_from_request(req)

    engine.publish(job_id, "started", {"params": params})

    from app.services.dense_reconstruction import DenseParams, run_dense_reconstruction

    def _run() -> dict:
        p = DenseParams(
            voxel_size=params["voxel_size"],
            sor_k=params["sor_k"],
            sor_std_ratio=params["sor_std_ratio"],
            ror_radius_m=params["ror_radius_m"],
            ror_min_neighbors=params["ror_min_neighbors"],
            normal_k=params["normal_k"],
            min_confidence=params["min_confidence"],
            max_points_per_view=params["max_points_per_view"],
            max_depth_m=params["max_depth_m"],
            min_depth_m=params["min_depth_m"],
        )
        return run_dense_reconstruction(job_id, p)

    try:
        report = await asyncio.to_thread(_run)
    except Exception as exc:
        from app.exceptions import ProcessingError

        raise ProcessingError("dense_reconstruction", detail=str(exc)) from exc

    # Persist status + model metadata on success.
    project.status = "dense_completed"
    project.current_stage = "dense_reconstruction"
    quality = report.get("quality", {})
    measurements = report.get("twin", {}).get("measurements", {})
    model = Model3D(
        project_id=job_id,
        format="ply",
        file_path=str(settings.storage.project_dir(job_id) / "dense" / "dense_model.ply"),
        file_size_bytes=0,
        point_count=quality.get("point_count", 0),
        bbox_min_x=measurements.get("bbox_min", [None, None, None])[0],
        bbox_min_y=measurements.get("bbox_min", [None, None, None])[1],
        bbox_min_z=measurements.get("bbox_min", [None, None, None])[2],
        bbox_max_x=measurements.get("bbox_max", [None, None, None])[0],
        bbox_max_y=measurements.get("bbox_max", [None, None, None])[1],
        bbox_max_z=measurements.get("bbox_max", [None, None, None])[2],
    )
    db.add(model)
    await db.flush()

    return DenseStartResponse(
        job_id=job_id,
        status=report["status"],
        message=(
            f"Dense reconstruction complete: {quality.get('point_count', 0)} points, "
            f"score {quality.get('dense_score', 0):.1f} ({quality.get('grade', 'Poor')})"
        ),
    )


@router.get("/api/dense/status/{job_id}", response_model=DenseStatusResponse)
async def get_dense_status(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> DenseStatusResponse:
    """Get dense reconstruction status."""
    await _get_project(job_id, db)
    report = await _load_report(job_id)
    if not report:
        return DenseStatusResponse(job_id=job_id, status="not_run")
    return DenseStatusResponse(
        job_id=job_id,
        status=report.get("status", "unknown"),
        point_count=report.get("quality", {}).get("point_count"),
        dense_score=report.get("quality", {}).get("dense_score"),
        grade=report.get("quality", {}).get("grade"),
        error=report.get("error"),
    )


@router.get("/api/dense/statistics/{job_id}", response_model=DenseStatisticsResponse)
async def get_dense_statistics(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> DenseStatisticsResponse:
    """Get dense model quality statistics, measurements, and intelligence."""
    await _get_project(job_id, db)
    report = await _load_report(job_id)
    return DenseStatisticsResponse(
        job_id=job_id,
        quality=report.get("quality", {}),
        measurements=report.get("twin", {}).get("measurements", {}),
        intelligence=report.get("twin", {}).get("intelligence", {}),
    )


@router.get("/api/dense/confidence/{job_id}", response_model=DenseConfidenceResponse)
async def get_dense_confidence(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> DenseConfidenceResponse:
    """Get per-class confidence summary for the dense model."""
    await _get_project(job_id, db)
    report = await _load_report(job_id)
    twin = report.get("twin", {})
    return DenseConfidenceResponse(
        job_id=job_id,
        mean_confidence=twin.get("measurements", {}).get("mean_confidence", 0.0),
        class_counts=twin.get("confidence", {}),
        weak_regions=twin.get("intelligence", {}).get("weak_regions", []),
    )


@router.get("/api/dense/stream/{job_id}")
async def stream_dense(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Server-sent events: live pipeline progress for *job_id*.

    Replays the recorded event history for completed runs, then tails live
    events while a run is in progress.
    """
    await _get_project(job_id, db)

    async def event_source():
        async for event in engine.iter_events(job_id):
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/dense/download/{job_id}")
async def download_dense(
    job_id: str,
    format: str = Query(default="ply", pattern="^(ply|xyz|pcd|las)$"),
    db: AsyncSession = Depends(get_session),
):
    """Download the dense model in the requested format."""
    await _get_project(job_id, db)
    try:
        from app.services.dense_reconstruction import export_dense_format

        path = export_dense_format(job_id, format)
    except FileNotFoundError as exc:
        raise BadRequestError(str(exc)) from exc
    except RuntimeError as exc:  # e.g. missing optional laspy
        raise BadRequestError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc

    # Record the download in the exports history.
    existing = await db.execute(
        select(Export).where(Export.project_id == job_id, Export.format == format)
    )
    if existing.scalar_one_or_none() is None:
        db.add(
            Export(
                project_id=job_id,
                format=format,
                file_path=str(path),
                file_size_bytes=path.stat().st_size,
            )
        )
        await db.flush()

    media = CONTENT_TYPES.get(format, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=f"dense_model{EXPORT_EXTENSIONS[format]}")


def _params_from_request(req: DenseStartRequest) -> dict:
    d = settings.dense
    merged = {
        "voxel_size": req.voxel_size if req.voxel_size is not None else d.voxel_size,
        "sor_k": req.sor_k if req.sor_k is not None else d.sor_k,
        "sor_std_ratio": req.sor_std_ratio if req.sor_std_ratio is not None else d.sor_std_ratio,
        "ror_radius_m": req.ror_radius_m if req.ror_radius_m is not None else d.ror_radius_m,
        "ror_min_neighbors": req.ror_min_neighbors if req.ror_min_neighbors is not None else d.ror_min_neighbors,
        "normal_k": req.normal_k if req.normal_k is not None else d.normal_k,
        "min_confidence": req.min_confidence if req.min_confidence is not None else d.min_confidence,
        "min_depth_m": req.min_depth_m if req.min_depth_m is not None else d.min_depth_m,
        "max_depth_m": req.max_depth_m if req.max_depth_m is not None else d.max_depth_m,
        "max_points_per_view": d.max_points_per_view,
    }
    if merged["min_depth_m"] >= merged["max_depth_m"]:
        raise BadRequestError("min_depth_m must be smaller than max_depth_m")
    return merged


async def _get_project(job_id: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == job_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ProjectNotFoundError(job_id)
    return project


async def _load_report(job_id: str) -> dict:
    from app.services.dense_reconstruction import load_dense_report

    report = load_dense_report(job_id)
    return report or {}
