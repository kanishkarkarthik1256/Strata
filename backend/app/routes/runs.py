"""Run discovery + artifact serving API (Phase 11.1).

GET  /api/runs                      — discover Phase 9.5 runs (manifest-driven)
GET  /api/runs/{run_id}             — run detail (summary + manifest + counts)
GET  /api/runs/{run_id}/manifest    — raw manifest JSON
GET  /api/runs/{run_id}/artifact/{name}  — validated artifact file (PLY/JSON/…)

Read-only and unauthenticated: these endpoints expose locally generated run
artifacts (the Phase 9.5 convention), consistent with the pre-Phase-10
surface that remains open under ``auth_mode=disabled``. Every artifact path
is validated by :func:`app.services.run_service.resolve_artifact`.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.exceptions import BadRequestError, NotFoundError
from app.logging_config import get_logger
from app.services import run_service

log = get_logger("drone_recon.routes.runs")

router = APIRouter(prefix="/api/runs", tags=["runs"])

_MEDIA_TYPES = {
    ".ply": "application/octet-stream",
    ".obj": "text/plain",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".json": "application/json",
    ".npy": "application/octet-stream",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
}


@router.get("")
async def list_runs() -> dict:
    import anyio

    runs = await anyio.to_thread.run_sync(run_service.list_runs)
    return {"runs": runs}


@router.get("/{run_id}")
async def get_run(run_id: str) -> dict:
    import anyio

    try:
        return await anyio.to_thread.run_sync(run_service.get_run, run_id)
    except run_service.RunNotFoundError:
        raise NotFoundError(f"run not found: {run_id}") from None


@router.get("/{run_id}/viewer-alignment")
async def get_viewer_alignment(run_id: str) -> dict:
    """Rigid presentation transform aligning the scene with the viewer grid.

    Cached as viewer_alignment.json in the run directory; recomputed only when
    any upstream geometry/pose artifact is newer than the cached file. The
    transform is presentation-only: authoritative artifacts are never modified.
    """
    import anyio

    try:
        return await anyio.to_thread.run_sync(
            run_service.get_viewer_alignment, run_id
        )
    except run_service.RunNotFoundError:
        raise NotFoundError(f"run not found: {run_id}") from None


@router.get("/{run_id}/analysis")
async def run_analysis(run_id: str, refresh: bool = False) -> dict:
    """Measured site analysis for one run: area, footprint, elevation, accuracy map.

    Every number comes from the run's own artifacts (see
    :mod:`app.services.area_analysis`); nothing is inferred from the request.
    The payload is cached in the run directory against a fingerprint of its
    inputs, so it is recomputed only when an input actually changes.
    ``refresh=true`` forces a recompute.
    """
    import anyio

    try:
        run = await anyio.to_thread.run_sync(run_service.run_dir, run_id)
    except run_service.RunNotFoundError:
        raise NotFoundError(f"run not found: {run_id}") from None

    from app.services.area_analysis import load_or_compute

    def _compute() -> dict:
        return load_or_compute(run, run_id, force=refresh)

    return await anyio.to_thread.run_sync(_compute)


@router.get("/{run_id}/manifest")
async def run_manifest(run_id: str) -> dict:
    import anyio

    try:
        run = await anyio.to_thread.run_sync(run_service.get_run, run_id)
    except run_service.RunNotFoundError:
        raise NotFoundError(f"run not found: {run_id}") from None
    return run["manifest"]


@router.get("/{run_id}/artifact/{name:path}")
async def run_artifact(run_id: str, name: str):
    """Serve one validated artifact file from the run directory."""
    import anyio

    try:
        path = await anyio.to_thread.run_sync(run_service.resolve_artifact, run_id, name)
    except run_service.InvalidArtifactError as exc:
        raise BadRequestError(str(exc)) from None
    except run_service.RunNotFoundError:
        raise NotFoundError(f"run not found: {run_id}") from None
    media = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)