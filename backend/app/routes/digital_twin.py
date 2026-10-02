"""Digital-twin (Phase 7) API routes.

POST /api/digital-twin/build/{job_id}   — run the mesh → twin stage chain
GET  /api/digital-twin/status/{job_id}  — stage state + resume map
GET  /api/digital-twin/scene/{job_id}   — scene index manifest
GET  /api/digital-twin/mesh/{job_id}    — download the mesh (?format=ply|obj)
GET  /api/digital-twin/texture/{job_id} — texture atlas PNG
GET  /api/digital-twin/semantics/{job_id}
GET  /api/digital-twin/objects/{job_id} — twin objects (+ measurements)
GET  /api/digital-twin/confidence/{job_id}
GET  /api/digital-twin/lods/{job_id}
GET  /api/digital-twin/geojson/{job_id}
POST /api/digital-twin/copilot/{job_id} — natural-language scene queries
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.engine import get_session
from app.db.models import Model3D, Project
from app.exceptions import BadRequestError, ProcessingError, ProjectNotFoundError
from app.logging_config import get_logger
from app.schemas.digital_twin import (
    CopilotRequest,
    TwinBuildRequest,
    TwinBuildResponse,
    TwinStatusResponse,
)
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.pipeline_stage import MESH_CHAIN, discover
from app.services.scene_index import build_scene_index

log = get_logger("drone_recon.routes.digital_twin")

router = APIRouter(tags=["digital-twin"])

_MESH_FORMATS = {"ply": ".ply", "obj": ".obj"}


@router.post("/api/digital-twin/build/{job_id}", response_model=TwinBuildResponse)
async def build_twin(
    job_id: str,
    req: TwinBuildRequest,
    db: AsyncSession = Depends(get_session),
) -> TwinBuildResponse:
    """Run (or resume) the digital-twin chain for a project."""
    project = await _get_project(job_id, db)
    discover()
    plugins = list(req.plugins) or list(MESH_CHAIN)
    try:
        from app.services.pipeline_stage import get_stage

        for p in plugins:
            get_stage(p)
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc

    from app.services.streaming_engine import engine

    engine.clear_cancel(job_id)
    engine.publish(job_id, "twin_build_started", {"plugins": plugins})
    pipeline_req = PipelineRequest(force=list(req.force), plugins=plugins)

    try:
        report = await asyncio.to_thread(run_autonomous_pipeline, job_id, pipeline_req)
    except Exception as exc:  # pipeline already wraps stages; unexpected errors here
        raise ProcessingError("digital_twin", detail=str(exc)) from exc

    status = report.get("status", "unknown")
    project.status = {"completed": "twin_completed", "cancelled": "cancelled"}.get(status, "failed")
    if status == "failed":
        project.error_message = report.get("error", "")
    failed = [n for n, s in report.get("stages", {}).items() if s.get("status") == "failed"]
    project.current_stage = failed[0] if failed else "digital_twin"
    await db.flush()
    await _record_models(job_id, db, plugins, status)

    message = {
        "completed": "Digital twin complete — mesh, semantics, objects and LODs are ready",
        "cancelled": "Build cancelled — rerun to resume from the last completed stage",
        "failed": f"Build failed at '{failed[0] if failed else '?'}': "
                  f"{report.get('error', '')[:200]}",
    }.get(status, "Build finished")
    return TwinBuildResponse(
        job_id=job_id,
        status=status,
        run_time_ms=report.get("run_time_ms", 0.0),
        error=report.get("error", ""),
        stages=report.get("stages", {}),
        message=message,
    )


@router.get("/api/digital-twin/status/{job_id}", response_model=TwinStatusResponse)
async def twin_status(
    job_id: str,
    db: AsyncSession = Depends(get_session),
) -> TwinStatusResponse:
    """Digital-twin pipeline state (from the workspace artifacts)."""
    await _get_project(job_id, db)
    report = _load_pipeline_report(job_id)
    if not report:
        return TwinStatusResponse(job_id=job_id, status="not_run")
    return TwinStatusResponse(
        job_id=job_id,
        status=report.get("status", "unknown"),
        run_time_ms=report.get("run_time_ms", 0.0),
        error=report.get("error", ""),
        stages=report.get("stages", {}),
        resume=report.get("resume", {}),
    )


@router.get("/api/digital-twin/scene/{job_id}")
async def twin_scene(job_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    """The aggregated scene manifest (mesh, semantics, twin, confidence…)."""
    await _get_project(job_id, db)
    return build_scene_index(job_id, settings.storage.project_dir(job_id))


@router.get("/api/digital-twin/mesh/{job_id}")
async def twin_mesh(
    job_id: str,
    format: str = Query(default="ply", pattern="^(ply|obj)$"),
    db: AsyncSession = Depends(get_session),
):
    """Download the final (repaired) mesh as PLY or OBJ."""
    await _get_project(job_id, db)
    workspace = settings.storage.project_dir(job_id)
    ply = workspace / "mesh" / "repaired_mesh.ply"
    if not ply.exists():
        raise BadRequestError("no mesh for this job — run the digital-twin build first")
    path = ply
    if format == "obj":
        obj = workspace / "mesh" / "repaired_mesh.obj"
        if not obj.exists():
            from app.services.mesh import TriangleMesh

            TriangleMesh.read_ply(ply).save_obj(obj)
        path = obj
    media = "application/octet-stream" if format == "ply" else "text/plain"
    return FileResponse(path, media_type=media, filename=f"mesh{_MESH_FORMATS[format]}")


@router.get("/api/digital-twin/texture/{job_id}")
async def twin_texture(job_id: str, db: AsyncSession = Depends(get_session)):
    """The texture atlas PNG (if the texture stage produced one)."""
    await _get_project(job_id, db)
    path = settings.storage.project_dir(job_id) / "texture" / "texture_atlas.png"
    if not path.exists():
        raise BadRequestError("no texture atlas for this job")
    return FileResponse(path, media_type="image/png", filename="texture_atlas.png")


@router.get("/api/digital-twin/semantics/{job_id}")
async def twin_semantics(job_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    await _get_project(job_id, db)
    report_path = settings.storage.project_dir(job_id) / "semantic" / "semantic_report.json"
    if not report_path.exists():
        raise BadRequestError("no semantic labels — run the digital-twin build first")
    return json.loads(report_path.read_text())


@router.get("/api/digital-twin/objects/{job_id}")
async def twin_objects(job_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    """Twin objects with measurements and the scene graph."""
    await _get_project(job_id, db)
    workspace = settings.storage.project_dir(job_id)
    twin_path = workspace / "twin" / "twin.json"
    if not twin_path.exists():
        raise BadRequestError("no digital twin — run the digital-twin build first")
    twin = json.loads(twin_path.read_text())
    graph_path = workspace / "twin" / "scene_graph.json"
    return {"objects": twin.get("objects", []), "classes": twin.get("classes", {}),
            "scene_graph": json.loads(graph_path.read_text()) if graph_path.exists() else {}}


@router.get("/api/digital-twin/confidence/{job_id}")
async def twin_confidence(job_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    await _get_project(job_id, db)
    conf_path = settings.storage.project_dir(job_id) / "confidence" / "confidence_report.json"
    if not conf_path.exists():
        raise BadRequestError("no confidence overlay — run the digital-twin build first")
    data = json.loads(conf_path.read_text())
    overlay = settings.storage.project_dir(job_id) / "confidence" / "mesh_confidence.ply"
    return {**data, "overlay_ply": str(overlay) if overlay.exists() else None}


@router.get("/api/digital-twin/lods/{job_id}")
async def twin_lods(job_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    await _get_project(job_id, db)
    manifest = settings.storage.project_dir(job_id) / "mesh" / "lod" / "manifest.json"
    if not manifest.exists():
        raise BadRequestError("no LODs — run the digital-twin build first")
    return {"lods": json.loads(manifest.read_text())}


@router.get("/api/digital-twin/geojson/{job_id}")
async def twin_geojson(job_id: str, db: AsyncSession = Depends(get_session)):
    """Semantic objects as GeoJSON (ENU-aligned when GPS was available)."""
    await _get_project(job_id, db)
    path = settings.storage.project_dir(job_id) / "georef" / "objects.geojson"
    if not path.exists():
        raise BadRequestError(
            "no georeferenced objects — run the build with GPS telemetry "
            "(outputs are local until then)")
    return FileResponse(path, media_type="application/geo+json", filename="objects.geojson")


@router.post("/api/digital-twin/copilot/{job_id}")
async def twin_copilot(
    job_id: str,
    req: CopilotRequest,
    db: AsyncSession = Depends(get_session),
) -> dict:
    """Answer a natural-language question from the twin's measurements."""
    await _get_project(job_id, db)
    from app.services.copilot import answer

    workspace = settings.storage.project_dir(job_id)
    twin_path = workspace / "twin" / "twin.json"
    scene: dict = {}
    if twin_path.exists():
        scene = json.loads(twin_path.read_text())
    elif (workspace / "twin" / "scene_index.json").exists():
        scene = json.loads((workspace / "twin" / "scene_index.json").read_text())
    return answer(scene, req.query)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _get_project(job_id: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == job_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise ProjectNotFoundError(job_id)
    return project


def _load_pipeline_report(job_id: str) -> dict:
    path = settings.storage.project_dir(job_id) / "pipeline_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


async def _record_models(job_id: str, db: AsyncSession, plugins: list[str], status: str) -> None:
    """Persist Model3D rows for the mesh artifacts a completed build produced."""
    if status != "completed":
        return
    workspace = settings.storage.project_dir(job_id)
    candidates = []
    if (workspace / "mesh" / "repaired_mesh.ply").exists():
        candidates.append(("ply", workspace / "mesh" / "repaired_mesh.ply"))
    if (workspace / "mesh" / "lod" / "lod0.ply").exists():
        candidates.append(("ply", workspace / "mesh" / "lod" / "lod0.ply"))
    for fmt, path in candidates:
        existing = await db.execute(
            select(Model3D).where(Model3D.project_id == job_id, Model3D.file_path == str(path))
        )
        if existing.scalar_one_or_none() is not None:
            continue
        db.add(Model3D(
            project_id=job_id, format=fmt, file_path=str(path),
            file_size_bytes=path.stat().st_size,
            vertex_count=_read_ply_count(path),
        ))
    await db.flush()


def _read_ply_count(path: Path) -> int:
    try:
        header = path.read_bytes().split(b"end_header")[0].decode("ascii")
        for line in header.splitlines():
            if line.startswith("element vertex "):
                return int(line.split()[-1])
    except (OSError, ValueError):
        pass
    return None
