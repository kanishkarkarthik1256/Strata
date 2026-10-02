"""Upload API routes.

POST   /api/upload      — upload a drone video, returns job_id + status
GET    /api/upload/{id} — full job status with metadata
DELETE /api/upload/{id} — delete project and workspace
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Form, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import InvalidVideoError, NotFoundError
from app.services.telemetry import TelemetryError, format_telemetry_detection
from app.db.engine import get_session
from app.logging_config import get_logger
from app.schemas.upload import (
    DeleteResponse,
    JobStatus,
    JobStatusResponse,
    UploadResponse,
    VideoMetadata,
)
from app.services.upload_service import (
    delete_project,
    get_project,
    process_upload,
    save_lidar_upload,
    save_telemetry_upload,
)

log = get_logger("drone_recon.routes.upload")

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_video(
    file: UploadFile = File(...,
        description="Drone video file (mp4, mov, avi, mkv)"),
    mission_name: Optional[str] = Form(
        None,
        description=(
            "Optional mission name — seeds a human-readable project id, so "
            "the run workspace folder under data/storage/ is named after the "
            "mission (e.g. Site_Alpha_Survey_a1b2c3) instead of a random hex id."
        ),
    ),
    telemetry: Optional[UploadFile] = File(
        None,
        description=(
            "Optional external telemetry CSV — schema auto-detected "
            "(any reasonable UAV telemetry format). "
            "Not required — video-only reconstruction is fully supported."
        ),
    ),
    lidar: Optional[UploadFile] = File(
        None,
        description=(
            "Optional LiDAR reference (LAS/LAZ) for absolute accuracy. "
            "Held out: the pipeline never reads it, it is only compared "
            "against the finished model."
        ),
    ),
    db: AsyncSession = Depends(get_session),
) -> UploadResponse:
    """Upload a drone video with an optional external telemetry file.

    The video is validated, metadata is extracted, and a project record is
    created. A missing telemetry file is expected for most manual uploads —
    the run is classified VIDEO_ONLY, never an ingestion failure. A supplied
    telemetry file is structure-validated immediately; malformed CSVs are
    rejected with a clear error.
    """
    if not file.filename:
        raise InvalidVideoError("No filename provided")

    project_id, metadata = await process_upload(file, db, mission_name=mission_name)

    schema_report: dict | None = None
    srt_provenance: dict | None = None
    if telemetry is not None and telemetry.filename:
        try:
            saved = await save_telemetry_upload(telemetry, project_id)
            schema_report = saved.get("schema_report")
            srt_provenance = saved.get("srt_provenance")
        except (InvalidVideoError, TelemetryError) as exc:
            # No detectable GPS columns: rejected with the detection
            # diagnostics instead of the old hard-coded header error.
            await delete_project(project_id, db)
            raise InvalidVideoError(
                "Telemetry file unusable: " + str(exc),
            ) from None
        except Exception as exc:
            # Video already persisted; do not leave a half-registered
            # project behind on unexpected failures (e.g. disk).
            await delete_project(project_id, db)
            raise RuntimeError(
                "Telemetry save failed: " + str(exc)[:100],
            ) from exc

    lidar_note = ""
    if lidar is not None and lidar.filename:
        try:
            saved_lidar = await save_lidar_upload(lidar, project_id)
        except InvalidVideoError as exc:
            # An unreadable tile is rejected with its own reason instead of
            # producing a run whose ground truth silently is not there.
            await delete_project(project_id, db)
            raise InvalidVideoError(str(exc)) from None
        except Exception as exc:
            await delete_project(project_id, db)
            raise RuntimeError(
                "LiDAR reference save failed: " + str(exc)[:100],
            ) from exc
        # Report what the file actually was: a pre-baked height grid carries no
        # returns to count, and claiming a count for it would be invented.
        if saved_lidar.get("points") is None:
            lidar_note = (
                f" — LiDAR reference held out for accuracy"
                f" (pre-baked height grid, {saved_lidar['grid_gsd_m']} m cells)"
            )
        else:
            lidar_note = (
                f" — LiDAR reference held out for accuracy"
                f" ({saved_lidar['points']:,} returns, {saved_lidar['grid_gsd_m']} m grid)"
            )

    srt_note = ""
    if srt_provenance is not None:
        # DJI SRT flight log converted at upload. Two families reach here:
        # gimbal-bearing logs become flight_poses.csv (telemetry-assisted SfM)
        # and report frame/sync numbers; legacy position-only logs become
        # telemetry.csv and report SAMPLES with no sync offset. Report whichever
        # was actually measured — an unresolved "?" told the user nothing.
        samples = srt_provenance.get("frames") or srt_provenance.get("samples")
        sync = srt_provenance.get("sync_offset_sec")
        detail = f"{samples} samples" if samples else "converted"
        if sync is not None:
            detail += f", sync offset {sync} s"
        artifact = srt_provenance.get("artifact")
        if artifact:
            detail += f", written to {artifact}"
        srt_note = f" — DJI SRT flight log converted ({detail})"
    message = (
        f"Video '{metadata.filename}' uploaded and validated successfully"
        + (" with external telemetry"
           if telemetry is not None and telemetry.filename else "")
    )
    if schema_report is not None:
        message += " — " + format_telemetry_detection(schema_report)
    message += srt_note + lidar_note

    return UploadResponse(
        job_id=project_id,
        status=JobStatus.UPLOADED,
        message=message,
        metadata=metadata,
    )


@router.get("/{project_id}", response_model=JobStatusResponse)
async def get_job_status(
    project_id: str,
    db: AsyncSession = Depends(get_session),
) -> JobStatusResponse:
    """Get full status of an upload job, including extracted metadata."""
    project = await get_project(project_id, db)

    metadata = None
    if project.video_duration_sec is not None:
        metadata = VideoMetadata(
            filename=project.video_filename,
            duration_sec=project.video_duration_sec,
            fps=project.video_fps or 0.0,
            width=project.video_width or 0,
            height=project.video_height or 0,
            codec="",
            frame_count=0,
            bitrate_kbps=0.0,
            file_size_bytes=0,
            gps_lat=project.gps_lat,
            gps_lon=project.gps_lon,
            gps_alt=project.gps_alt,
        )

    return JobStatusResponse(
        job_id=project.id,
        status=JobStatus(project.status),
        filename=project.video_filename,
        metadata=metadata,
        workspace_path=project.video_path,
        error=project.error_message,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.delete("/{project_id}", response_model=DeleteResponse)
async def delete_job(
    project_id: str,
    db: AsyncSession = Depends(get_session),
) -> DeleteResponse:
    """Delete a previously uploaded project and its workspace."""
    await delete_project(project_id, db)
    return DeleteResponse(
        job_id=project_id,
        status="deleted",
        message="Project deleted successfully",
    )