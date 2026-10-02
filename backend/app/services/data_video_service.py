"""By-name video mission service.

Lets any video already present in the repository ``data/`` folder be turned
into a mission by name (``base.mp4``, ``london.mp4``, …) without a manual
upload. Unlike :mod:`canonical_demo_service` (which runs the pipeline inline
and blocks the HTTP request), this service creates the project record and
enqueues the real pipeline on the durable job queue, returning immediately —
progress streams through the same pipeline status/SSE surface as uploaded
missions.
"""

from __future__ import annotations

import datetime
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.models import Project
from app.logging_config import get_logger
from app.services import job_queue, lidar
from app.services.metadata_extraction import extract_metadata
from app.services.video_validation import ensure_cv2_readable, validate_all

log = get_logger("drone_recon.services.data_video")

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

#: Infix of the cv2-decode transcode sibling :func:`ensure_cv2_readable`
#: caches beside a video (``<stem>_cv2dec.mp4``). A sidecar is derived
#: footage for one OpenCV build's benefit — never a mission source — so the
#: picker must not offer it and the resolver must not accept it by name.
TRANSCODE_SIDECAR_INFIX = "_cv2dec"

#: Files a dataset folder may carry beside its video as its GPS/pose source.
#: The start path (``start_data_video_mission``) consumes exactly these names
#: (converting the SRT / copying the CSVs into the run); the picker reports
#: them through :func:`dataset_gps_sources` so the UI can tell whether a run
#: has a GPS source BEFORE it starts one.
GPS_SIBLINGS = ("poses.csv", "movingdrone_telemetry.csv", "video.SRT")


def dataset_gps_sources(video_path: Path) -> list[str]:
    """GPS/pose files sitting beside *video_path* that the pipeline consumes.

    Empty means the dataset is video-only: reconstruction can still run, but
    it cannot be georeferenced or metrically validated, which is why the
    mission form refuses to start without one.
    """
    folder = video_path.parent
    return [name for name in GPS_SIBLINGS if (folder / name).is_file()]


#: File names a dataset ships its held-out LiDAR reference under.
LIDAR_SIBLINGS = tuple(f"lidar{ext}" for ext in lidar.LIDAR_EXTENSIONS)


def dataset_lidar_source(video_path: Path) -> Path | None:
    """The held-out LiDAR reference sitting beside *video_path*, if there is one.

    Optional by design: a LiDAR tile is ground truth for the accuracy
    comparison, never an input to reconstruction, so a dataset without one
    reconstructs exactly as before and simply reports no absolute accuracy.

    Canonical names first, then a single ``.las``/``.laz`` in the same folder
    (survey tiles usually keep the vendor's own filename). Two or more
    candidates are left alone rather than guessed at — picking one at random
    would attach the wrong ground truth to a run.
    """
    folder = video_path.parent
    for name in LIDAR_SIBLINGS:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    found = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in lidar.LIDAR_EXTENSIONS
    )
    return found[0] if len(found) == 1 else None


def dataset_lidar_sources(video_path: Path) -> list[str]:
    """Names of the LiDAR references beside *video_path* (usually 0 or 1)."""
    source = dataset_lidar_source(video_path)
    return [source.name] if source is not None else []


def data_dir() -> Path:
    """Repository ``data/`` folder (``__file__``-anchored; cwd-independent)."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "data"


def _run_store_roots() -> set[Path]:
    """Every directory that holds run workspaces, never footage.

    Two roots, because the store path is cwd-derived while the data folder is
    ``__file__``-anchored: a backend started from ``backend/`` writes to
    ``backend/data/storage``, and one started from the repository root writes
    to ``<data>/storage``. Both are the pipeline's own output; neither is
    footage. Read as paths (never through ``settings.storage.base_dir``, whose
    getter creates directories) so listing stays side-effect free.
    """
    return {
        Path(settings.storage.base_path).resolve(),
        (data_dir() / "storage").resolve(),
    }


def _is_video_name(name: str) -> bool:
    """A dataset video by name: allowed container, never a transcode sidecar.

    String-based on purpose. The walk calls this once per entry and the
    reference tree has folders holding ~80k images; building a ``Path`` per
    candidate name cost more than reading the directory itself. Equivalent to
    ``Path(name).suffix`` / ``.stem`` for a bare filename — note a leading dot
    is a hidden file, not an extension (``".mp4"`` is not a video).
    """
    dot = name.rfind(".")
    if dot <= 0:
        return False
    if name[dot:].lower() not in ALLOWED_EXTENSIONS:
        return False
    return TRANSCODE_SIDECAR_INFIX not in name[:dot]


def _iter_video_files(root: Path):
    """Yield dataset videos up to two levels deep (``test/<dataset>/video.MP4``).

    The run store is skipped: every mission the pipeline has ever run leaves a
    copy of its source video (and its frames) under ``data/storage``, so
    walking it offered the user seven duplicate ``london.mp4`` entries and
    orphaned scratch clips beside the real datasets. What the picker shows is
    footage the user put in ``data/`` — the run store is output, not input.

    Entries are typed from the directory record (``os.scandir`` → ``d_type``)
    rather than ``Path.is_file()``. The picker walks folders such as
    ``AGZ/MAV Images`` that hold ~80k image files and no footage at all, and a
    stat per entry made one listing cost 2.4 s on the reference tree. The
    directory read is work the walk already did; only the per-entry syscalls
    are gone.
    """
    skip = _run_store_roots()

    def walk(dir_path: Path, depth: int):
        # ``depth`` is how many more directory levels may be entered: a folder
        # at depth 0 is still listed (videos beside it are yielded) but its own
        # subfolders are not.
        with os.scandir(dir_path) as entries:
            # Keep only what the walk can act on (subfolders, video-named
            # files) BEFORE sorting: the biggest folder on the reference tree
            # holds ~80k images, and sorting entries that can never be yielded
            # cost more than the directory read itself. Relative order is
            # unchanged — dropped entries were never yielded.
            candidates = []
            for entry in entries:
                if entry.is_dir():
                    if depth < 1 or entry.name.startswith("."):
                        continue
                    candidates.append(entry)
                elif _is_video_name(entry.name):
                    candidates.append(entry)
            for entry in sorted(candidates, key=lambda e: e.name):
                if entry.is_dir():
                    child = Path(entry.path)
                    if child.resolve() in skip:
                        continue
                    yield from walk(child, depth - 1)
                else:
                    yield Path(entry.path)

    if root.is_dir():
        yield from walk(root, 2)


def list_data_videos() -> list[dict[str, Any]]:
    """Videos available in ``data/`` with cheap header metadata.

    Includes videos nested up to two levels deep (``test/Flight_to_tower/video.MP4``) —
    dataset sequences live in their own folders next to their flight metadata.
    """
    videos: list[dict[str, Any]] = []
    root = data_dir()
    for path in _iter_video_files(root):
        entry: dict[str, Any] = {
            "name": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "gps_sources": dataset_gps_sources(path),
            "lidar_sources": dataset_lidar_sources(path),
        }
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                entry["duration_sec"] = round(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, cap.get(cv2.CAP_PROP_FPS)), 2
                )
                entry["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                entry["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                entry["fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)
            cap.release()
        except Exception:  # metadata is best-effort; the file itself stays usable
            log.warning("data_video_metadata_failed", video=path.name)
        videos.append(entry)
    return videos


def resolve_data_video(name: str) -> Path:
    """Resolve *name* to a video inside ``data/`` — no traversal, real file.

    One or two levels of nesting are allowed (``airport1/video.mp4``,
    ``test/Flight_to_tower/video.MP4``): dataset folders sit beside their
    flight metadata. ``..`` and absolute escapes stay rejected.
    """
    if "\\" in name or name in (".", "..") or name.startswith("/"):
        raise FileNotFoundError(f"invalid video name: {name!r}")
    parts = name.split("/")
    if len(parts) > 3 or any(p in ("", ".", "..") for p in parts):
        raise FileNotFoundError(f"invalid video name: {name!r}")
    path = (data_dir() / name).resolve()
    data_root = data_dir().resolve()
    if (
        data_root not in path.parents
        or path.suffix.lower() not in ALLOWED_EXTENSIONS
        or TRANSCODE_SIDECAR_INFIX in path.stem
    ):
        raise FileNotFoundError(f"invalid video name: {name!r}")
    # Same rule the picker lists by: a run store copy is output, not footage.
    if any(root in path.parents for root in _run_store_roots()):
        raise FileNotFoundError(
            f"'{name}' is inside a run store, not a data/ dataset; start the "
            f"dataset instead"
        )
    if not path.is_file():
        available = ", ".join(v["name"] for v in list_data_videos()) or "none"
        raise FileNotFoundError(f"video '{name}' not found in data/ (available: {available})")
    return path


async def start_data_video_mission(
    db: AsyncSession,
    *,
    video_name: str,
    pipeline_payload: dict[str, Any] | None = None,
    telemetry_path: Path | None = None,
    flight_poses_path: Path | None = None,
) -> dict[str, Any]:
    """Create a project for *video_name* and enqueue the pipeline on the queue.

    ``telemetry_path`` optionally associates an external telemetry CSV with
    the run (copied into the workspace as ``telemetry.csv``; the pipeline
    payload carries ``telemetry_csv`` so the georef stage finds it).

    Returns immediately with the ``run_id``; progress is observable through the
    pipeline status endpoint and streaming events while the queue worker runs.
    """
    from app.services.telemetry import (
        TelemetryError,
        load_telemetry_with_schema,
    )

    video_path = resolve_data_video(video_name)
    validate_all(video_path)
    metadata = extract_metadata(video_path)

    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"data_run_{timestamp_str}_{uuid.uuid4().hex[:6]}"

    workspace = settings.storage.project_dir(run_id)
    workspace.mkdir(parents=True, exist_ok=True)
    # Copy the DECODE-READY bytes, not the raw source. ``validate_all`` above
    # already ran ``ensure_cv2_readable`` on the data/ source, transcoding a
    # cv2-unreadable encoding (H.264/HEVC on a FFMPEG-less OpenCV build) ONCE
    # to a cached sibling. Copying the raw original re-planted the problem
    # inside the run: the frames stage re-ran the same multi-minute transcode
    # inside the pipeline, silently, before its first progress tick (measured:
    # ~8 minutes of dead "Frame Extraction pending" per mission). The video is
    # only ever decoded — never modified — so a frame-exact re-encode is a
    # faithful run input; provenance hashes the bytes that produced the run.
    decode_ready = ensure_cv2_readable(video_path)
    copied_video = workspace / decode_ready.name
    shutil.copy2(decode_ready, copied_video)

    payload = dict(pipeline_payload or {})

    # Held-out LiDAR reference (optional). Copied into the run's own
    # reference/ directory for the same reason the video is copied: the run has
    # to be measurable on its own, without the dataset folder still being in
    # place. Nothing in reconstruction reads this directory — the accuracy
    # comparison reads it after the model exists.
    lidar_source = dataset_lidar_source(video_path)
    if lidar_source is not None:
        reference_dir = workspace / lidar.REFERENCE_DIRNAME
        reference_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            lidar_source,
            reference_dir / f"{lidar.REFERENCE_STEM}{lidar_source.suffix.lower()}",
        )

    if flight_poses_path is not None:
        # Pre-converted canonical flight poses (e.g. from an uploaded DJI
        # SRT log) — copied into the workspace where the sparse stage's
        # telemetry-assisted path consumes it automatically.
        shutil.copy2(flight_poses_path, workspace / "flight_poses.csv")
    if telemetry_path is not None:
        # Copy first, then validate THE COPY: artifacts (telemetry_schema.json
        # etc.) land in the run workspace, never beside an uploaded temp file
        # or a dataset source. Validation happens before enqueue — undetectable
        # telemetry is still a 422, just with the diagnostics already stored.
        dest = workspace / "telemetry.csv"
        shutil.copy2(telemetry_path, dest)
        try:
            load_telemetry_with_schema(dest)  # validate + write schema artifacts
        except TelemetryError as exc:
            from app.exceptions import InvalidVideoError

            dest.unlink(missing_ok=True)
            for art in ("telemetry_normalized.csv", "telemetry_schema.json",
                        "telemetry_quality.json"):
                (workspace / art).unlink(missing_ok=True)
            raise InvalidVideoError(f"Telemetry file unusable: {exc}") from exc
        payload.setdefault("telemetry_csv", "telemetry.csv")

    # Dataset folders ship their own flight metadata. A sibling poses.csv
    # (MovingDrone native: frame_id,x,y,z,qw..qz,lat,lon,alt,fov,rpy) is the
    # rich source — metric camera centres + orientations per video frame — and
    # is copied verbatim into the workspace for the sparse stage to consume.
    # A canonical-schema CSV (timestamp,lat,lon,alt) is the GPS-only fallback.
    dataset_name: str | None = None
    dataset_rel: str | None = None
    if video_path.parent != data_dir().resolve():
        dataset_name = video_path.parent.name
        dataset_rel = video_path.relative_to(data_dir()).as_posix()
        native = video_path.parent / "poses.csv"
        if native.is_file() and not (workspace / "flight_poses.csv").exists():
            shutil.copy2(native, workspace / "flight_poses.csv")
        canonical = video_path.parent / "movingdrone_telemetry.csv"
        if canonical.is_file() and "telemetry_csv" not in payload and not (workspace / "telemetry.csv").exists():
            shutil.copy2(canonical, workspace / "telemetry.csv")
            payload["telemetry_csv"] = "telemetry.csv"
        intr = video_path.parent / "intrinsics.json"
        if intr.is_file() and not (workspace / "intrinsics.json").exists():
            shutil.copy2(intr, workspace / "intrinsics.json")
        # COLMAP text calibration (JB3D style): convert to intrinsics.json,
        # scaling focal/principal point from the calibrated resolution to the
        # actual video resolution (same factor — uniform decode, no crop).
        cams = video_path.parent / "cameras.txt"
        if cams.is_file() and not (workspace / "intrinsics.json").exists() \
                and not intr.is_file():
            try:
                from app.services.colmap_text import camera_txt_to_intrinsics

                parsed = camera_txt_to_intrinsics(cams, metadata.width, metadata.height)
                if parsed is not None:
                    import json as _json

                    (workspace / "intrinsics.json").write_text(_json.dumps(parsed))
                    payload["calibration_provenance"] = {
                        "source": "cameras.txt", "model": parsed["model"],
                        "calibrated_resolution": [parsed["orig_width"], parsed["orig_height"]],
                    }
            except Exception as exc:  # calibration is optional; never block the run
                log.warning("cameras_txt_conversion_failed", error=str(exc))
        # DJI SRT telemetry (any format family): convert to the canonical
        # artifact the log's measured content supports — flight_poses.csv
        # for gimbal-bearing logs, telemetry.csv for legacy position-only
        # logs (the sparse placement path + georef consume the latter).
        if not (workspace / "flight_poses.csv").exists() and not (workspace / "telemetry.csv").exists() \
                and "telemetry_csv" not in payload:
            srt = video_path.parent / "video.SRT"
            if srt.is_file():
                from app.services.dji_srt_telemetry import convert_srt_for_run

                try:
                    video_gps = (
                        (metadata.gps_lat, metadata.gps_lon)
                        if metadata.gps_lat is not None and metadata.gps_lon is not None
                        else None
                    )
                    prov = convert_srt_for_run(
                        srt, workspace, video_fps=metadata.fps, video_gps=video_gps
                    )
                    payload["telemetry_provenance"] = {"source": "DJI SRT", **prov}
                except Exception as exc:  # telemetry is optional; never block the run
                    log.warning("srt_telemetry_conversion_failed", error=str(exc))

    project = Project(
        id=run_id,
        name=f"{video_path.name} (data mission {run_id})",
        video_filename=decode_ready.name,
        video_path=str(copied_video),
        video_duration_sec=metadata.duration_sec,
        video_fps=metadata.fps,
        video_width=metadata.width,
        video_height=metadata.height,
        status="uploaded",
        current_stage="ingestion",
    )
    db.add(project)
    await db.flush()

    # Durable run-input provenance: what exact file, from where, with what
    # fingerprint. Written once at creation; the manifest promotes it.
    from app.services.provenance import write_source_record

    write_source_record(
        workspace,
        kind="data_video",
        # The identity gate compares this name against the workspace video's
        # actual filename, so it must name the decode-ready copy the pipeline
        # will really consume (differs from the source only when transcoded).
        original_filename=decode_ready.name,
        stored_path=copied_video,
        duration_sec=metadata.duration_sec,
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
        dataset_name=dataset_name,
        sequence_id=dataset_name,
        dataset_relative_path=f"data/{video_path.relative_to(data_dir()).as_posix()}",
        extras={
            k: payload[k]
            for k in ("telemetry_provenance", "calibration_provenance")
            if k in payload
        },
    )

    payload.setdefault("force", ["frames", "sparse", "depth", "dense", "georef"])
    await job_queue.enqueue(db, project_id=run_id, kind="pipeline", payload=payload)
    await db.commit()

    log.info("data_video_mission_started", run_id=run_id, video=video_path.name)
    return {
        "run_id": run_id,
        "video": video_path.name,
        "status": "queued",
        "message": f"Mission queued for {video_path.name} — follow progress on /api/pipeline/status/{run_id}",
    }
