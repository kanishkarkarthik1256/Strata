"""Reconstruction API routes.

POST /api/reconstruction/start/{job_id}       — start reconstruction
GET  /api/reconstruction/status/{job_id}      — get status
GET  /api/reconstruction/trajectory/{job_id}  — get trajectory
GET  /api/reconstruction/features/{job_id}    — get feature info
GET  /api/reconstruction/matches/{job_id}     — get match info
GET  /api/reconstruction/dashboard/{job_id}   — get dashboard
GET  /api/reconstruction/confidence/{job_id}  — get confidence
GET  /api/reconstruction/report/{job_id}      — get full report
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.engine import get_session
from app.db.models import Frame, Project
from app.logging_config import get_logger
from app.schemas.reconstruction import (
    ConfidenceInfo,
    ConfidenceResponse,
    DashboardResponse,
    FeatureInfo,
    FeaturesResponse,
    MatchInfo,
    MatchesResponse,
    ReconstructionStartResponse,
    ReconstructionStatusResponse,
    ReportResponse,
    TrajectoryPoint,
    TrajectoryResponse,
)
from app.services.image_files import list_image_files

log = get_logger("drone_recon.routes.reconstruction")

router = APIRouter(tags=["reconstruction"])


@router.post("/api/reconstruction/start/{job_id}", response_model=ReconstructionStartResponse)
async def start_reconstruction(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> ReconstructionStartResponse:
    """Start sparse reconstruction on selected frames."""
    project = await _get_project(job_id, db)

    # Find selected frames directory
    output_dir = settings.storage.project_dir(job_id)
    selected_dir = output_dir / "selected"

    if not selected_dir.exists():
        # Fall back to frames dir
        selected_dir = output_dir / "frames"

    if not selected_dir.exists():
        from app.exceptions import BadRequestError
        raise BadRequestError("No frames found. Run frame extraction first.")

    # Count frames
    frame_files = list_image_files(selected_dir)
    total_frames = await _count_total_frames(job_id, db)
    selected_count = len(frame_files)

    # Run reconstruction in thread pool
    from app.services.sparse_reconstruction import run_sparse_reconstruction
    report = await asyncio.to_thread(
        run_sparse_reconstruction,
        selected_dir,
        output_dir,
        project_id=job_id,
        total_frames=total_frames,
        selected_frames=selected_count,
    )

    # Update project status
    project.status = "reconstructed"
    project.current_stage = "sparse_reconstruction"
    await db.flush()

    return ReconstructionStartResponse(
        job_id=job_id,
        status="completed",
        message=f"Reconstruction complete: {report['reconstruction']['num_cameras']} cameras, {report['reconstruction']['num_points']} points",
    )


@router.get("/api/reconstruction/status/{job_id}", response_model=ReconstructionStatusResponse)
async def get_reconstruction_status(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> ReconstructionStatusResponse:
    """Get reconstruction status."""
    project = await _get_project(job_id, db)
    report = await _load_report(job_id)

    return ReconstructionStatusResponse(
        job_id=job_id,
        status=project.status,
        mission_score=report.get("analysis", {}).get("mission_score"),
        grade=report.get("analysis", {}).get("grade"),
        registered_cameras=report.get("reconstruction", {}).get("num_cameras"),
        total_cameras=report.get("dashboard", {}).get("total_cameras"),
        sparse_points=report.get("reconstruction", {}).get("num_points"),
        average_reprojection_error=report.get("reconstruction", {}).get("mean_reproj_error"),
        camera_registration_percent=report.get("dashboard", {}).get("camera_registration_percent"),
        suggestions=report.get("analysis", {}).get("suggestions", []),
    )


@router.get("/api/reconstruction/trajectory/{job_id}", response_model=TrajectoryResponse)
async def get_trajectory(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> TrajectoryResponse:
    """Get camera trajectory."""
    await _get_project(job_id, db)
    report = await _load_report(job_id)
    traj = report.get("trajectory", {})

    return TrajectoryResponse(
        job_id=job_id,
        points=[],  # Full trajectory from reconstruction data
        total_length=traj.get("total_length", 0.0),
        mean_speed=0.0,
        smoothness=traj.get("smoothness", 0.0),
    )


@router.get("/api/reconstruction/features/{job_id}", response_model=FeaturesResponse)
async def get_features(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> FeaturesResponse:
    """Get feature extraction results."""
    await _get_project(job_id, db)
    report = await _load_report(job_id)
    stages = report.get("stages", {})
    fe = stages.get("feature_extraction", {})

    return FeaturesResponse(
        job_id=job_id,
        features=[],
        total_keypoints=0,
        average_keypoints=0.0,
        backend=fe.get("backend", "sift"),
    )


@router.get("/api/reconstruction/matches/{job_id}", response_model=MatchesResponse)
async def get_matches(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> MatchesResponse:
    """Get feature matching results."""
    await _get_project(job_id, db)
    report = await _load_report(job_id)
    stages = report.get("stages", {})
    fm = stages.get("feature_matching", {})

    return MatchesResponse(
        job_id=job_id,
        matches=[],
        total_pairs=fm.get("matches", 0),
        average_confidence=0.0,
    )


@router.get("/api/reconstruction/dashboard/{job_id}", response_model=DashboardResponse)
async def get_dashboard(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> DashboardResponse:
    """Get reconstruction dashboard data."""
    await _get_project(job_id, db)
    report = await _load_report(job_id)
    dash = report.get("dashboard", {})
    analysis = report.get("analysis", {})
    recon = report.get("reconstruction", {})

    return DashboardResponse(
        job_id=job_id,
        mission_score=dash.get("mission_score", analysis.get("mission_score", 0.0)),
        grade=dash.get("grade", analysis.get("grade", "Poor")),
        registered_cameras=dash.get("registered_cameras", recon.get("num_cameras", 0)),
        total_cameras=dash.get("total_cameras", 0),
        sparse_points=dash.get("sparse_points", recon.get("num_points", 0)),
        average_reprojection_error=recon.get("mean_reproj_error", 0.0),
        camera_registration_percent=dash.get("camera_registration_percent", 0.0),
        suggestions=analysis.get("suggestions", []),
    )


@router.get("/api/reconstruction/confidence/{job_id}", response_model=ConfidenceResponse)
async def get_confidence(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> ConfidenceResponse:
    """Get confidence estimation."""
    await _get_project(job_id, db)
    report = await _load_report(job_id)
    conf = report.get("confidence", {})

    return ConfidenceResponse(
        job_id=job_id,
        camera_confidences=[],
        point_confidences=[],
        mean_camera_confidence=conf.get("mean_camera", 0.0),
        mean_point_confidence=conf.get("mean_point", 0.0),
    )


@router.get("/api/reconstruction/report/{job_id}", response_model=ReportResponse)
async def get_report(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> ReportResponse:
    """Get full reconstruction report."""
    await _get_project(job_id, db)
    report = await _load_report(job_id)

    return ReportResponse(
        job_id=job_id,
        status=report.get("status", "unknown"),
        pipeline_time_ms=report.get("pipeline_time_ms", 0.0),
        reconstruction=report.get("reconstruction", {}),
        trajectory=report.get("trajectory", {}),
        analysis=report.get("analysis", {}),
        confidence=report.get("confidence", {}),
        dashboard=report.get("dashboard", {}),
    )


@router.get("/api/reconstruction/metric-validation/{job_id}")
async def get_metric_validation(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> dict:
    """Get metric validation status and ground-truth accuracy report."""
    await _get_project(job_id, db)
    project_dir = settings.storage.project_dir(job_id)

    # ONE accuracy artifact, produced by ONE engine (metric_validation), served
    # verbatim. This route previously read a filename nothing ever wrote and
    # then fell back to a report that invented a ground-truth pose whenever
    # poses.json existed — reporting GPS_GEOREFERENCED for runs with no GPS
    # correspondence.
    #
    # Absence is reported as absence (404). It used to be answered with a 200
    # envelope, which made the page's honest "no report exists" state
    # unreachable and let a run that never measured anything render as if it
    # had. The body still explains why, in the engine's own report shape.
    from app.services.metric_validation import (
        ARTIFACT_RELATIVE_PATH,
        STATUS_NOT_MEASURED,
        load_report,
        not_certified_report,
    )

    report = load_report(project_dir)
    if report is not None:
        return report

    # The app's 404 handler preserves only string details, so the reason is
    # composed by the envelope's single owner and delivered as text (a dict
    # detail would be silently dropped and the user would get a generic
    # "resource not found" with no explanation).
    envelope = not_certified_report(
        run_id=job_id,
        reason=(
            f"no accuracy report exists for this run "
            f"({ARTIFACT_RELATIVE_PATH.as_posix()} absent) — absolute spatial accuracy: "
            f"{STATUS_NOT_MEASURED}; relative reconstruction quality is reported separately"
        ),
    )
    raise HTTPException(status_code=404, detail=envelope["certification_reason"])



@router.get("/api/reconstruction/dsm-accuracy/{job_id}")
async def get_dsm_accuracy(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> dict:
    """Ground-truth DSM accuracy for runs whose dataset ships a reference grid.

    404 with an honest reason when there is no reference for this run — the
    page renders nothing rather than an unmeasured claim.
    """
    await _get_project(job_id, db)
    from app.services.dsm_accuracy import run_dsm_accuracy

    result = run_dsm_accuracy(settings.storage.project_dir(job_id), job_id)
    if result.get("status") == "no_reference":
        raise HTTPException(
            status_code=404,
            detail=f"no reference DSM is registered for run {job_id} — ground-truth accuracy cannot be measured",
        )
    return result



async def _get_project(job_id: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == job_id))
    project = result.scalar_one_or_none()
    if project is None:
        from app.exceptions import ProjectNotFoundError
        raise ProjectNotFoundError(job_id)
    return project


async def _count_total_frames(job_id: str, db: AsyncSession) -> int:
    result = await db.execute(select(Frame).where(Frame.project_id == job_id))
    return len(result.scalars().all())


async def _load_report(job_id: str) -> dict:
    report_path = settings.storage.project_dir(job_id) / "reconstruction_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {}
