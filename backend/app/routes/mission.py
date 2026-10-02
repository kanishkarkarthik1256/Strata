"""Mission planning (Phase 9) API routes.

POST /api/mission/simulate            — simulate one plan over a scene
POST /api/mission/what-if             — scenario deltas vs baseline
POST /api/mission/optimize            — Pareto sweep + labelled plans
POST /api/mission/analyze/{job_id}    — run the PLANNING_CHAIN on a mission
GET  /api/mission/coverage/{job_id}   — measured coverage/blind rasters
GET  /api/mission/plan/{job_id}       — stored plan (json|md|geojson|kml|html)
GET  /api/mission/recommendations/{job_id}
GET  /api/mission/battery             — battery estimate from query params
POST /api/mission/copilot             — source-labelled planning answers
GET  /api/mission/history             — mission history records
POST /api/mission/history             — append a measured record
GET  /api/mission/similar             — similar historical missions
GET  /api/mission/learning            — learning summary (data-gated)
GET  /api/mission/replay/{mission_id} — stored record for replay rendering
POST /api/mission/validate/{job_id}   — prediction-vs-actual validation
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from app.config.settings import settings
from app.exceptions import BadRequestError
from app.logging_config import get_logger
from app.schemas.mission import (
    CopilotRequest,
    HistoryAppendRequest,
    OptimizeRequest,
    SimulationRequest,
    WhatIfRequest,
)
from app.services import mission_history as mh
from app.services.mission_history import load_history
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.pipeline_stage import PLANNING_CHAIN, discover

log = get_logger("drone_recon.routes.mission")

router = APIRouter(tags=["mission"])


@router.post("/api/mission/simulate")
async def simulate(req: SimulationRequest) -> dict:
    from app.services.mission_simulator import plan_defaults
    from app.services.mission_simulator import simulate as run_sim

    scene = req.scene.to_dict()
    plan = {**plan_defaults(), **(req.plan or {})}
    return run_sim(plan, scene)


@router.post("/api/mission/what-if")
async def what_if(req: WhatIfRequest) -> dict:
    from app.services.mission_simulator import compare, plan_defaults
    from app.services.mission_simulator import simulate as run_sim

    scene = req.scene.to_dict()
    base_plan = {**plan_defaults(), **(req.plan or {})}
    change = req.scenario
    alt = change.get("altitude_m")
    scenario_plan = dict(base_plan)
    if alt is not None:
        scenario_plan["altitude_m"] = max(base_plan["altitude_m"] + float(alt), 5.0)
    for key in ("speed_m_s", "forward_overlap", "side_overlap"):
        if key in change:
            scenario_plan[key] = min(float(change[key]), 100.0 if key == "speed_m_s" else 0.95)
    return compare(run_sim(base_plan, scene), run_sim(scenario_plan, scene))


@router.post("/api/mission/optimize")
async def optimize(req: OptimizeRequest) -> dict:
    from app.services.path_optimizer import optimize as run_opt

    return run_opt(req.scene.to_dict())


@router.get("/api/mission/battery")
async def battery(duration_min: float = Query(default=10.0, gt=0.0),
                  altitude_m: Optional[float] = None,
                  speed_m_s: Optional[float] = None,
                  capacity_wh: Optional[float] = None) -> dict:
    from app.services.battery_model import battery_estimate
    from app.services.mission_simulator import plan_defaults

    plan = plan_defaults()
    if altitude_m is not None:
        plan["altitude_m"] = altitude_m
    if speed_m_s is not None:
        plan["speed_m_s"] = speed_m_s
    if capacity_wh is not None:
        plan["battery_capacity_wh"] = capacity_wh
    return battery_estimate(plan, duration_min / 60.0)


@router.post("/api/mission/analyze/{job_id}")
async def analyze(job_id: str) -> dict:
    """Run coverage prediction + mission planning + learning for a job."""

    workspace = settings.storage.project_dir(job_id)
    if not (workspace / "mesh" / "repaired_mesh.ply").exists():
        raise BadRequestError("no repaired mesh for this job — run the digital-twin "
                              "build before planning a follow-up mission")
    discover()
    from app.services.streaming_engine import engine

    engine.clear_cancel(job_id)
    engine.publish(job_id, "mission_planning_started", {"plugins": list(PLANNING_CHAIN)})
    report = await asyncio.to_thread(
        run_autonomous_pipeline, job_id,
        PipelineRequest(plugins=list(PLANNING_CHAIN), force=list(PLANNING_CHAIN)))
    stages = report.get("stages", {})
    return {
        "job_id": job_id,
        "status": report.get("status"),
        "stages": {n: {k: stages.get(n, {}).get(k)
                       for k in ("status", "detail", "error") if k in stages.get(n, {})}
                   for n in PLANNING_CHAIN},
        "error": report.get("error", ""),
    }


@router.get("/api/mission/coverage/{job_id}")
async def coverage(job_id: str) -> dict:
    workspace = settings.storage.project_dir(job_id)
    path = workspace / "intel" / "coverage_prediction.json"
    if not path.exists():
        # compute on demand from the twin artifacts when available
        from app.services.coverage_predictor import _load, build_coverage
        try:
            mesh, labels, conf = _load(workspace)
        except (OSError, ValueError) as exc:
            raise BadRequestError("no mesh/coverage for this job — run the twin build "
                                  "or mission analyze") from exc
        report = build_coverage(mesh, labels, conf)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    return json.loads(path.read_text())


@router.get("/api/mission/plan/{job_id}")
async def mission_plan(job_id: str,
                       format: str = Query(default="json",
                                           pattern="^(json|md|geojson|kml|html)$")) -> dict:
    workspace = settings.storage.project_dir(job_id)
    path = workspace / "intel" / "mission_plan.json"
    if not path.exists():
        raise BadRequestError("no mission plan — run /api/mission/analyze first")
    if format == "json":
        return json.loads(path.read_text())
    doc = json.loads(path.read_text())
    if format == "md":
        from app.services.mission_report import render_plan_markdown
        return {"plan_markdown": render_plan_markdown(doc)}
    # html is the markdown rendered as HTML
    from app.services.mission_report import render_plan_markdown
    from app.services.report_generator import _md_to_html
    if format == "html":
        return {"plan_html": _md_to_html(render_plan_markdown(doc))}
    # geojson / kml export files are written by the planning stage
    filename = f"mission_plan.{format}"
    export_path = workspace / "intel" / filename
    if not export_path.exists():
        raise BadRequestError(f"no {format} export — rerun mission analyze")
    return {"export": str(export_path), "download_path": f"/api/mission/plan_file/{job_id}?format={format}"}


@router.get("/api/mission/plan_file/{job_id}")
async def plan_file(job_id: str,
                    format: str = Query(pattern="^(geojson|kml)$")) -> FileResponse:
    workspace = settings.storage.project_dir(job_id)
    path = workspace / "intel" / f"mission_plan.{format}"
    if not path.exists():
        raise BadRequestError("no plan export — run /api/mission/analyze first")
    media = {"geojson": "application/geo+json", "kml": "application/vnd.google-earth.kml+xml"}[format]
    return FileResponse(path, media_type=media, filename=path.name)


@router.get("/api/mission/recommendations/{job_id}")
async def recommendations(job_id: str) -> dict:
    workspace = settings.storage.project_dir(job_id)
    path = workspace / "intel" / "mission_plan.json"
    if not path.exists():
        raise BadRequestError("no mission plan — run /api/mission/analyze first")
    doc = json.loads(path.read_text())
    return {"job_id": job_id, "chosen_plan": doc.get("chosen_plan"),
            "recommendations": doc.get("recommendations", [])}


@router.post("/api/mission/copilot")
async def copilot(req: CopilotRequest) -> dict:
    from app.services.flight_copilot import answer
    from app.services.mission_planner import build_scene

    ctx: dict = {"history": load_history()}
    if req.job_id:
        workspace = settings.storage.project_dir(req.job_id)
        if not (workspace / "mesh" / "repaired_mesh.ply").exists():
            raise BadRequestError("job has no reconstructed mesh")
        ctx["workspace"] = workspace
        ctx["scene"] = build_scene(workspace)
        plan_path = workspace / "intel" / "mission_plan.json"
        if plan_path.exists():
            ctx["plan"] = json.loads(plan_path.read_text())
        from app.services.mission_history import learning_summary
        ctx["learning"] = learning_summary(load_history())
    return answer(req.query, ctx)


@router.get("/api/mission/history")
async def history(limit: int = Query(default=20, ge=1, le=500)) -> dict:
    return {"history": mh.recent(load_history(), limit), "count": len(load_history())}


@router.post("/api/mission/history")
async def history_append(req: HistoryAppendRequest) -> dict:
    rec = req.record
    if not rec.get("mission_id"):
        raise BadRequestError("record requires a mission_id")
    return {"stored": True, "mission_id": mh.append_record(rec).get("mission_id")}


@router.get("/api/mission/similar")
async def similar(mission_id: Optional[str] = Query(default=None),
                  altitude_m: Optional[float] = None, speed_m_s: Optional[float] = None,
                  forward_overlap: Optional[float] = None,
                  env_score: Optional[float] = None) -> dict:
    records = load_history()
    query: dict = {"mission_id": mission_id or "query"}
    if altitude_m is not None:
        query["altitude_m"] = altitude_m
    if speed_m_s is not None:
        query["speed_m_s"] = speed_m_s
    if forward_overlap is not None:
        query["forward_overlap"] = forward_overlap
    if env_score is not None:
        query["env_score"] = env_score
    # No numeric query → compare against the most recent mission record.
    if not any(k in query for k in ("altitude_m", "speed_m_s", "forward_overlap", "env_score")) \
            and records:
        query = dict(records[-1])
        query["mission_id"] = mission_id or "latest"
    return mh.similar_missions(query, records)


@router.get("/api/mission/learning")
async def learning() -> dict:
    return mh.learning_summary(load_history())


@router.get("/api/mission/replay/{mission_id}")
async def replay(mission_id: str) -> dict:
    records = load_history()
    rec = next((r for r in records if r.get("mission_id") == mission_id), None)
    if rec is None:
        raise BadRequestError(f"no history record for mission '{mission_id}'")
    return {"mission": rec,
            "note": "replay renders the stored record (measured outcomes + plan "
                    "params); per-frame telemetry playback requires a telemetry "
                    "recording that this deployment does not capture"}


@router.post("/api/mission/validate/{job_id}")
async def validate(job_id: str) -> dict:
    workspace = settings.storage.project_dir(job_id)
    record = mh.record_from_workspace(workspace)
    row = mh.validate_mission(record)
    if row.get("status") == "no_prediction_recorded":
        return {"job_id": job_id, "status": row["status"],
                "message": row["message"]}
    return {"job_id": job_id, "validation": row}
