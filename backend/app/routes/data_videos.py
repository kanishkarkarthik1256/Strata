"""Data-folder video routes — mission from any video already in ``data/``.

POST /api/data-videos           — list available videos with header metadata
POST /api/data-videos/{name}/start — create the project + enqueue the pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.exceptions import InvalidVideoError
from app.logging_config import get_logger
from app.services import data_video_service

log = get_logger("drone_recon.routes.data_videos")

router = APIRouter(prefix="/api/data-videos", tags=["data-videos"])


@router.get("")
async def list_videos() -> dict[str, Any]:
    """Videos available in the repository ``data/`` folder."""
    videos = data_video_service.list_data_videos()
    return {"videos": videos, "count": len(videos)}


@router.post("/{name:path}/start")
async def start_mission(
    name: str,
    payload: Optional[dict[str, Any]] = None,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Start a mission from ``data/<name>`` on the durable job queue."""
    try:
        return await data_video_service.start_data_video_mission(
            db, video_name=name, pipeline_payload=payload
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidVideoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{name:path}/start-with-telemetry")
async def start_mission_with_telemetry(
    name: str,
    telemetry: UploadFile = File(
        ..., description="External telemetry CSV — schema auto-detected (any reasonable UAV telemetry format)"
    ),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Start a data-folder mission with an external telemetry CSV (multipart).

    Kept as a separate multipart endpoint so the plain JSON ``/start``
    contract is untouched. Malformed telemetry is rejected with 422; a
    missing telemetry file is simply the other endpoint.
    """
    import tempfile

    tmp_telemetry: Path | None = None
    try:
        suffix = Path(telemetry.filename or "t.csv").suffix or ".csv"
        if suffix.lower() not in (".csv", ".txt", ".srt"):
            raise InvalidVideoError(
                f"Unsupported telemetry format '{suffix}'. Supported: .csv, .srt"
            )
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
            while chunk := telemetry.file.read(1024 * 1024):
                tmp.write(chunk)
            tmp_telemetry = Path(tmp.name)
        if suffix.lower() == ".srt":
            # DJI SRT flight log: convert to the canonical artifact its
            # format supports — flight_poses.csv (per-frame metric pose +
            # orientation, bracketed family) or telemetry.csv (position-only
            # legacy GPS-function family, fed to the placement path). The
            # mission starter copies it into the run workspace where the
            # sparse stage consumes it automatically.
            from app.services.dji_srt_telemetry import parse_srt, srt_has_orientation, srt_to_flight_poses, srt_to_telemetry_csv
            from app.services.metadata_extraction import extract_metadata
            from app.services.video_validation import validate_all

            video_path = data_video_service.resolve_data_video(name)
            validate_all(video_path)
            metadata = extract_metadata(video_path)
            try:
                frames = parse_srt(tmp_telemetry)
                with_orientation = srt_has_orientation(frames)
            except ValueError as exc:
                raise InvalidVideoError(f"SRT telemetry unusable: {exc}") from exc
            with tempfile.TemporaryDirectory() as conv_dir:
                converted = Path(conv_dir) / (
                    "flight_poses.csv" if with_orientation else "telemetry.csv"
                )
                try:
                    video_gps = (
                        (metadata.gps_lat, metadata.gps_lon)
                        if metadata.gps_lat is not None and metadata.gps_lon is not None
                        else None
                    )
                    if with_orientation:
                        prov = srt_to_flight_poses(
                            tmp_telemetry, converted, video_fps=metadata.fps,
                        )
                    else:
                        prov = srt_to_telemetry_csv(
                            tmp_telemetry, converted, video_gps=video_gps
                        )
                except (ValueError, OSError) as exc:
                    raise InvalidVideoError(f"SRT telemetry unusable: {exc}") from exc
                return await data_video_service.start_data_video_mission(
                    db, video_name=name, pipeline_payload={
                        "srt_flight_log": True, "srt_provenance": prov,
                    },
                    flight_poses_path=converted if with_orientation else None,
                    telemetry_path=None if with_orientation else converted,
                )
        return await data_video_service.start_data_video_mission(
            db, video_name=name, pipeline_payload=None, telemetry_path=tmp_telemetry
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidVideoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_telemetry is not None:
            tmp_telemetry.unlink(missing_ok=True)
