"""Upload job management service.

Orchestrates the full upload lifecycle:
1. Stream video to disk with size enforcement (safe, deterministic filename)
2. Validate the video
3. Extract metadata
4. Persist project record to SQLite
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.models import Project
from app.exceptions import FileTooLargeError, InvalidVideoError, ProjectNotFoundError
from app.logging_config import get_logger
from app.schemas.upload import VideoMetadata
from app.services import lidar
from app.services.metadata_extraction import extract_metadata
from app.services.video_validation import validate_all

log = get_logger("drone_recon.services.upload")

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB read chunks
_UNSAFE_FILENAME = re.compile(r"[^a-zA-Z0-9_.-]")


def safe_telemetry_filename(raw: str) -> str:
    """Reduce a client filename to a safe basename with a ``.csv``-family
    extension. Rejects anything else — telemetry is CSV (extensible later to
    IMU/RTK formats, which will get their own validators)."""
    raw_str = (raw or "").replace("\\", "/")
    name = Path(raw_str).name
    cleaned = _UNSAFE_FILENAME.sub("_", name).strip(".")
    if not cleaned:
        raise InvalidVideoError("Telemetry file has no usable filename")
    ext = cleaned.lower().rsplit(".", 1)[-1]
    # SRT is DJI's native per-frame flight-log format (FrameCnt/GPS/gimbal
    # blocks) — accepted and converted to the canonical flight_poses.csv at
    # upload; see _convert_srt_telemetry.
    if ext not in ("csv", "txt", "srt"):
        raise InvalidVideoError(
            f"Unsupported telemetry format '.{ext}'. Supported: .csv, .srt"
        )
    return cleaned


def safe_lidar_filename(raw: str) -> str:
    """Reduce a client filename to a safe basename with a LiDAR extension.

    Nothing else is accepted: a reference that cannot be read is not a
    reference, and storing one would leave a run that looks measured against
    ground truth it never had. ``.las``/``.laz`` carry returns; ``.npz`` is
    the pre-baked form and may carry returns, a height grid, or both.
    """
    raw_str = (raw or "").replace("\\", "/")
    name = Path(raw_str).name
    cleaned = _UNSAFE_FILENAME.sub("_", name).strip(".")
    if not cleaned or "." not in cleaned:
        raise InvalidVideoError("LiDAR file has no usable filename")
    ext = "." + cleaned.lower().rsplit(".", 1)[-1]
    if ext not in lidar.LIDAR_EXTENSIONS:
        raise InvalidVideoError(
            f"Unsupported LiDAR format '{ext}'. Supported: "
            f"{', '.join(lidar.LIDAR_EXTENSIONS)}"
        )
    return cleaned


def safe_filename(raw: str) -> str:
    """Reduce a client filename to a safe basename (no separators/traversal).

    Raises ``InvalidVideoError`` when the result is empty or has no supported
    video extension — the file is rejected *before* anything touches disk.
    """
    raw_str = (raw or "").replace("\\", "/")
    name = Path(raw_str).name  # strips any ../ or absolute components
    cleaned = _UNSAFE_FILENAME.sub("_", name).strip(".")
    if not cleaned:
        raise InvalidVideoError("Uploaded file has no usable filename")
    ext = cleaned.lower().rsplit(".", 1)[-1]
    if ext not in settings.processing.supported_video_formats:
        raise InvalidVideoError(
            f"Unsupported video format '.{ext}'. "
            f"Supported: {', '.join(settings.processing.supported_video_formats)}"
        )
    return cleaned


async def save_upload(file: UploadFile, project_id: str) -> Path:
    """Stream the uploaded file to disk with a safe, deterministic filename.

    The extension allowlist is checked up front (before any bytes are
    written) and the stored name never contains path separators, so a
    crafted ``filename`` cannot escape the project workspace. Returns the
    path to the saved file.
    """
    safe = safe_filename(file.filename or "")
    workspace = settings.storage.project_dir(project_id)
    dest = workspace / safe

    max_bytes = settings.processing.max_upload_size_mb * 1024 * 1024
    written = 0

    import anyio

    def _sync_write():
        nonlocal written
        with open(dest, "wb") as f:
            while True:
                chunk = file.file.read(CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    dest.unlink(missing_ok=True)
                    raise FileTooLargeError(settings.processing.max_upload_size_mb)
                f.write(chunk)

    await anyio.to_thread.run_sync(_sync_write)
    log.info("upload_saved", project_id=project_id, filename=safe, size_bytes=written)
    return dest


async def save_telemetry_upload(file: UploadFile, project_id: str) -> dict:
    """Stream an external telemetry CSV into the project workspace.

    Stored canonically as ``telemetry.csv`` regardless of the client filename
    (the pipeline references it by that fixed name). The file is parsed
    *before* acceptance — a file with no detectable GPS columns is rejected
    with a 422-class error rather than discovered mid-pipeline. Returns the
    saved path plus the schema-detection report for the upload response
    (``TELEMETRY DETECTED`` UX, spec §17).
    """
    from app.services.telemetry import (
        TelemetryError,
        load_telemetry_with_schema,
    )

    safe = safe_telemetry_filename(file.filename or "")
    is_srt = safe.lower().endswith(".srt")
    workspace = settings.storage.project_dir(project_id)
    workspace.mkdir(parents=True, exist_ok=True)
    # SRT is converted to flight_poses.csv (the telemetry-assisted sparse
    # path's canonical input), not telemetry.csv: the CSV schema detector
    # has no meaning for SRT subtitle blocks, and flight_poses.csv carries
    # the richer per-frame metric pose the sparse stage consumes directly.
    dest = workspace / "telemetry_upload.srt" if is_srt else workspace / "telemetry.csv"

    import anyio

    written = 0
    max_bytes = 32 * 1024 * 1024  # telemetry is tiny; 32 MB is generous

    def _sync_write():
        nonlocal written
        with open(dest, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise FileTooLargeError(
                        f"Telemetry file exceeds {max_bytes // (1024 * 1024)} MB limit"
                    )
                out.write(chunk)

    await anyio.to_thread.run_sync(_sync_write)

    if is_srt:
        # DJI SRT flight log -> canonical flight_poses.csv. The video is
        # already persisted at this point (process_upload ran first), so the
        # real fps is available for the sync-offset estimate. A corrupt or
        # non-DJI log raises TelemetryError -> the route deletes the project
        # and answers 422, exactly like unusable CSV telemetry.
        video_exts = set(settings.processing.supported_video_formats) | {"webm", "m4v"}
        video_file = next(
            (f for f in workspace.iterdir()
             if f.is_file() and f.suffix.lower().lstrip(".") in video_exts),
            None,
        )
        if video_file is None:
            dest.unlink(missing_ok=True)
            raise TelemetryError("SRT telemetry uploaded without a video file")
        try:
            prov = await anyio.to_thread.run_sync(
                lambda: _convert_srt_telemetry(dest, video_file, workspace)
            )
        except TelemetryError:
            log.warning("srt_telemetry_rejected", project_id=project_id)
            raise
        log.info("telemetry_saved", project_id=project_id, bytes=written,
                 format="srt")
        return {"path": dest, "schema_report": None, "srt_provenance": prov}

    try:
        # Parse validates detectability; samples are re-loaded by the pipeline.
        _samples, schema_report = load_telemetry_with_schema(dest)
    except TelemetryError:
        dest.unlink(missing_ok=True)
        log.warning("telemetry_rejected", project_id=project_id)
        raise
    log.info("telemetry_saved", project_id=project_id, bytes=written)
    return {"path": dest, "schema_report": schema_report}


async def save_lidar_upload(file: UploadFile, project_id: str) -> dict:
    """Stream an optional LiDAR reference into the run's reference directory.

    Stored canonically as ``reference/lidar.<ext>`` — the path
    :func:`app.services.lidar.resolve_reference` resolves, and one the
    reconstruction stage never reads. The tile is *parsed* before acceptance
    and rejected with its own error when it cannot be read, the same discipline
    the telemetry upload follows, so a broken tile fails at upload rather than
    surfacing later as a run that mysteriously has no ground truth.

    Returns the saved path plus what the tile actually contains (returns,
    rasterised cell size, bounds) for the upload response.
    """
    safe = safe_lidar_filename(file.filename or "")
    ext = "." + safe.lower().rsplit(".", 1)[-1]
    workspace = settings.storage.project_dir(project_id)
    dest = workspace / lidar.REFERENCE_DIRNAME / f"{lidar.REFERENCE_STEM}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    import anyio

    written = 0
    max_bytes = settings.processing.max_upload_size_mb * 1024 * 1024

    def _sync_write():
        nonlocal written
        with open(dest, "wb") as out:
            while chunk := file.file.read(CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    raise FileTooLargeError(settings.processing.max_upload_size_mb)
                out.write(chunk)

    await anyio.to_thread.run_sync(_sync_write)

    def _inspect() -> tuple[int | None, lidar.HeightGrid]:
        """What the file actually contains, read before it is accepted."""
        if dest.suffix.lower() == lidar.NPZ_EXTENSION:
            files = lidar.npz_keys(dest)
            if "points" not in files and {"height", "bounds", "gsd"} <= files:
                # A pre-baked height grid IS a usable reference — the accuracy
                # path loads it as-is (:func:`lidar.load_grid`); it simply has
                # no returns to count or rasterise.
                return None, lidar.load_grid(dest)
        points = lidar.read_points(dest)
        return len(points.xyz), lidar.rasterise(points)

    try:
        count, grid = await anyio.to_thread.run_sync(_inspect)
    except lidar.LidarError as exc:
        dest.unlink(missing_ok=True)
        log.warning("lidar_rejected", project_id=project_id)
        raise InvalidVideoError(f"LiDAR reference unusable: {exc}") from exc

    log.info(
        "lidar_saved",
        project_id=project_id,
        bytes=written,
        points=count,
        gsd_m=round(grid.gsd, 3),
    )
    return {
        "path": dest,
        "points": count,
        "grid_gsd_m": round(grid.gsd, 3),
        "grid_bounds_m": [round(float(b), 2) for b in grid.bounds],
    }


def _convert_srt_telemetry(srt_path: Path, video_path: Path, workspace: Path) -> dict:
    """Convert an uploaded DJI SRT log into its canonical telemetry artifact.

    The format decides the target: gimbal-bearing logs (bracketed family)
    convert to ``flight_poses.csv`` (telemetry-assisted triangulation);
    legacy position-only logs (GPS-function family) convert to
    ``telemetry.csv`` (measured-similarity placement + georef). Runs after
    video metadata extraction (needs the real fps for the sync offset).
    Raises ``TelemetryError`` for non-DJI/corrupt logs so the route can
    reject the upload with the usual 422 semantics.
    """
    from app.services.dji_srt_telemetry import convert_srt_for_run
    from app.services.metadata_extraction import extract_metadata
    from app.services.telemetry import TelemetryError

    metadata = extract_metadata(video_path)
    # The video's own fix (ISO 6709 atom / tags) is the independent evidence
    # that settles a position-only log's coordinate order. Absent -> the
    # converter keeps the log's written order and says so in provenance.
    video_gps = (
        (metadata.gps_lat, metadata.gps_lon)
        if metadata.gps_lat is not None and metadata.gps_lon is not None
        else None
    )
    try:
        prov = convert_srt_for_run(
            srt_path, workspace, video_fps=metadata.fps, video_gps=video_gps,
        )
    except (ValueError, OSError) as exc:
        srt_path.unlink(missing_ok=True)
        (workspace / "flight_poses.csv").unlink(missing_ok=True)
        (workspace / "telemetry.csv").unlink(missing_ok=True)
        raise TelemetryError(f"SRT telemetry unusable: {exc}") from exc
    (workspace / "srt_telemetry_provenance.json").write_text(
        json.dumps(prov, indent=2, default=str)
    )
    log.info(
        "srt_telemetry_converted", project_id=workspace.name,
        frames=prov.get("frames"), sync_offset_sec=prov.get("sync_offset_sec"),
    )
    return prov


def derive_project_id(mission_name: str | None) -> str:
    """Derive a human-readable, collision-safe project id from a mission name.

    The id doubles as the on-disk workspace folder name under
    ``data/storage/``, so "Site Alpha Survey" becomes
    ``data/storage/Site_Alpha_Survey_a1b2c3/``. Format:
    ``<sanitized-name, max 25 chars>_<6 hex chars>``, capped at the DB's
    ``String(32)`` primary key. Empty/absent names fall back to the legacy
    ``uuid4().hex``.
    """
    if not mission_name or not mission_name.strip():
        return uuid.uuid4().hex
    slug = _UNSAFE_FILENAME.sub("_", mission_name).strip("._-")
    if not slug:
        return uuid.uuid4().hex
    return f"{slug[:25]}_{uuid.uuid4().hex[:6]}"


async def process_upload(
    file: UploadFile,
    db: AsyncSession,
    mission_name: str | None = None,
) -> tuple[str, VideoMetadata]:
    """Full upload flow: save, validate, extract metadata, persist.

    ``mission_name`` (optional) seeds a readable project id / workspace
    folder name. Returns (project_id, metadata).
    """
    project_id = derive_project_id(mission_name)
    log.info(
        "upload_started",
        project_id=project_id,
        filename=file.filename,
        mission_name=mission_name or None,
    )

    # Save to disk (safe name enforced inside), then content-validate.
    # A genuinely corrupt file raises — but is NEVER deleted here: the user
    # keeps the bytes so they can re-encode/inspect instead of losing the
    # upload silently. Decodable files (via the ffmpeg fallback) always pass.
    filepath = await save_upload(file, project_id)

    import anyio

    metadata = await anyio.to_thread.run_sync(
        lambda: (validate_all(filepath), extract_metadata(filepath))[1]
    )

    # Persist to database — display name follows the mission name when the
    # user provided one, so the runs list matches the folder they see.
    project = Project(
        id=project_id,
        name=mission_name.strip() if mission_name and mission_name.strip() else filepath.name,
        video_filename=filepath.name,
        video_path=str(filepath),
        status="uploaded",
        video_duration_sec=metadata.duration_sec,
        video_width=metadata.width,
        video_height=metadata.height,
        video_fps=metadata.fps,
        gps_lat=metadata.gps_lat,
        gps_lon=metadata.gps_lon,
        gps_alt=metadata.gps_alt,
    )
    db.add(project)
    await db.flush()

    # Durable run-input provenance sidecar (kind: uploaded).
    from app.services.provenance import write_source_record

    write_source_record(
        settings.storage.project_dir(project_id),
        kind="uploaded",
        original_filename=file.filename or filepath.name,
        stored_path=filepath,
        duration_sec=metadata.duration_sec,
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
    )

    log.info(
        "upload_complete",
        project_id=project_id,
        filename=metadata.filename,
        duration=metadata.duration_sec,
        resolution=f"{metadata.width}x{metadata.height}",
    )

    return project_id, metadata


async def get_project(project_id: str, db: AsyncSession) -> Project:
    """Fetch a project by ID, or raise ProjectNotFoundError."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ProjectNotFoundError(project_id)
    return project


async def delete_project(project_id: str, db: AsyncSession) -> None:
    """Delete a project and its workspace directory."""
    project = await get_project(project_id, db)

    # Delete workspace directory
    workspace = settings.storage.base_dir / project_id
    if workspace.exists():
        import shutil
        shutil.rmtree(workspace)
        log.info("workspace_deleted", project_id=project_id, path=str(workspace))

    # Delete database record (cascade deletes child records)
    await db.delete(project)
    await db.flush()

    log.info("project_deleted", project_id=project_id)
