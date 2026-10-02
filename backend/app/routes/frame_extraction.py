"""Frame extraction API routes.

POST /api/frame-extraction/{job_id} — start extraction
GET  /api/frame-extraction/{job_id} — get extraction status
GET  /api/frame/{frame_id}          — get single frame detail
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.engine import get_session
from app.db.models import Frame, Project
from app.logging_config import get_logger
from app.schemas.frame_extraction import (
    ExtractionMode,
    ExtractionRequest,
    ExtractionResponse,
    ExtractionStatusResponse,
    FrameDetailResponse,
    FrameInfo,
)
from app.services.frame_extractor import extract_frames

log = get_logger("drone_recon.routes.frame_extraction")

router = APIRouter(tags=["frame-extraction"])


@router.post("/api/frame-extraction/{job_id}", response_model=ExtractionResponse)
async def start_extraction(
    job_id: str,
    request: ExtractionRequest = ExtractionRequest(),
    db: AsyncSession = Depends(get_session),
) -> ExtractionResponse:
    """Start frame extraction on an uploaded video.

    Reads the video, scores all frames, rejects poor ones, and saves
    selected frames to the project workspace.
    """
    project = await _get_project(job_id, db)

    video_path = Path(project.video_path)
    if not video_path.exists():
        from app.exceptions import BadRequestError
        raise BadRequestError(f"Video file not found: {video_path}")

    output_dir = settings.storage.project_dir(job_id)

    # Run extraction (synchronous — runs in thread pool via FastAPI)
    import asyncio
    report = await asyncio.to_thread(
        extract_frames,
        video_path,
        output_dir,
        extraction_mode=request.extraction_mode.value,
        target_fps=request.target_fps,
        every_n=request.every_n,
        interval_sec=request.interval_sec,
        quality_threshold=request.quality_threshold,
        top_percent=request.top_percent,
    )

    # Persist frame records to DB
    frames_dir = output_dir / "frames"
    selected_dir = output_dir / "selected"

    for frame_data in report["frames"]:
        frame_record = Frame(
            project_id=job_id,
            index=frame_data["index"],
            timestamp_sec=frame_data["timestamp_sec"],
            file_path=str(frames_dir / frame_data["filename"]),
            blur_score=frame_data["scores"]["blur"],
            sharpness_score=frame_data["scores"]["sharpness"],
            exposure_score=frame_data["scores"]["exposure"],
            motion_score=frame_data["scores"]["motion"],
            composite_score=frame_data["scores"]["composite"],
            kept=frame_data["kept"],
        )
        db.add(frame_record)

    # Update project status
    project.status = "extracted"
    project.current_stage = "frame_extraction"
    await db.flush()

    log.info(
        "extraction_api_complete",
        job_id=job_id,
        selected=report["selected_count"],
        rejected=report["rejected_count"],
    )

    return ExtractionResponse(
        job_id=job_id,
        status="completed",
        message=f"Extracted {report['candidates_extracted']} frames, kept {report['selected_count']}",
        selected_count=report["selected_count"],
        rejected_count=report["rejected_count"],
        total_candidates=report["candidates_extracted"],
    )


@router.get("/api/frame-extraction/{job_id}", response_model=ExtractionStatusResponse)
async def get_extraction_status(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> ExtractionStatusResponse:
    """Get the status and frame list for a frame extraction job."""
    project = await _get_project(job_id, db)

    # Query frames from DB
    result = await db.execute(
        select(Frame).where(Frame.project_id == job_id).order_by(Frame.index)
    )
    frames = result.scalars().all()

    frame_infos = [
        FrameInfo(
            index=f.index,
            timestamp_sec=f.timestamp_sec,
            filename=Path(f.file_path).name,
            kept=f.kept,
            rejection_reason=_reason_from_scores(f),
            scores={
                "blur": f.blur_score or 0.0,
                "sharpness": f.sharpness_score or 0.0,
                "exposure": f.exposure_score or 0.0,
                "motion": f.motion_score or 0.0,
                "composite": f.composite_score or 0.0,
            },
        )
        for f in frames
    ]

    selected = sum(1 for f in frames if f.kept)
    rejected = len(frames) - selected

    return ExtractionStatusResponse(
        job_id=job_id,
        status=project.status,
        extraction_mode=project.current_stage,
        selected_count=selected,
        rejected_count=rejected,
        total_candidates=len(frames),
        frames=frame_infos,
    )


@router.get("/api/frame/{frame_id}", response_model=FrameDetailResponse)
async def get_frame_detail(
    frame_id: str,
    db: AsyncSession = Depends(get_session),
) -> FrameDetailResponse:
    """Get detailed information for a single frame."""
    result = await db.execute(select(Frame).where(Frame.id == frame_id))
    frame = result.scalar_one_or_none()
    if frame is None:
        from app.exceptions import ProjectNotFoundError
        raise ProjectNotFoundError(f"Frame {frame_id}")

    return FrameDetailResponse(
        frame_id=frame.id,
        project_id=frame.project_id,
        index=frame.index,
        timestamp_sec=frame.timestamp_sec,
        file_path=frame.file_path,
        kept=frame.kept,
        blur_score=frame.blur_score,
        sharpness_score=frame.sharpness_score,
        exposure_score=frame.exposure_score,
        motion_score=frame.motion_score,
        composite_score=frame.composite_score,
    )


async def _get_project(job_id: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == job_id))
    project = result.scalar_one_or_none()
    if project is None:
        from app.exceptions import ProjectNotFoundError
        raise ProjectNotFoundError(job_id)
    return project


def _reason_from_scores(frame: Frame) -> str | None:
    """Infer rejection reason from score patterns if not stored explicitly."""
    if frame.kept:
        return None
    if frame.blur_score is not None and frame.blur_score < 0.3:
        return "blurry"
    if frame.exposure_score is not None and frame.exposure_score < 0.1:
        return "poor_exposure"
    if frame.motion_score is not None and frame.motion_score > 0.8:
        return "motion_blur"
    return "low_quality"
