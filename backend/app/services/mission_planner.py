"""Mission planner — follow-up plan design from measured scene evidence.

Assembles one ``intel/mission_plan.json`` from:

* measured evidence — scene extent + max height (mesh), confidence overlay,
  blind-spot coverage prediction, environment report, GPS quality,
* a baseline simulation of the default parameters plus three what-if
  scenarios (lower altitude / slower speed / higher overlap),
* a multi-objective Pareto sweep with labelled archetypes (path optimizer),
* a recommended plan and structured recommendations, each carrying
  recommendation / reason / evidence / expected benefit / cost / confidence
  and a ``source`` label (``measured`` / ``simulation_estimate`` /
  ``historical`` / ``user_provided``).

The chosen plan is also exported as GeoJSON/KML waypoint geometry so the
plan is directly viewable in a GIS client.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.mission_simulator import compare, footprint, plan_defaults, simulate
from app.services.path_optimizer import optimize
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register
from app.services.risk_aware_planner import weather_severity

log = get_logger("drone_recon.services.mission_planner")

_SCENARIOS = {
    "lower_altitude_20m": {"altitude_m": -20.0, "reason": "finer GSD and detail"},
    "slower_speed": {"speed_m_s": 3.0, "reason": "less motion blur"},
    "higher_overlap": {"forward_overlap": 0.1, "side_overlap": 0.1,
                       "reason": "more multi-view redundancy"},
}


def build_scene(workspace: Path) -> dict:
    """Measured scene context for planning from the twin artifacts."""
    mesh = TriangleMesh.read_ply(workspace / "mesh" / "repaired_mesh.ply")
    lo, hi = mesh.vertices[:, :2].min(axis=0), mesh.vertices[:, :2].max(axis=0)
    scene = {"extent_w_m": round(float(hi[0] - lo[0]), 1),
             "extent_h_m": round(float(hi[1] - lo[1]), 1),
             "max_scene_height_m": round(float(mesh.vertices[:, 2].max()), 2),
             "mean_confidence": None, "blind_share": None, "targets": [],
             "restricted": []}

    def _read(*parts: str) -> dict | None:
        p = workspace.joinpath(*parts)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    conf = _read("confidence", "confidence_report.json")
    if conf:
        scene["mean_confidence"] = float(conf.get("mean_confidence", 0.0) or 0.0)
        scene["blind_share"] = float(conf.get("low_share", 0.0) or 0.0)
        scene["targets"] = list(conf.get("weak_regions", []))[:8]
    cover = _read("intel", "coverage_prediction.json")
    if cover and scene["blind_share"] is None:
        scene["blind_share"] = cover.get("blind_share")
        if not scene["targets"]:
            scene["targets"] = list(cover.get("weak_regions", []))[:8]
    restricted = _read("intel", "restricted.json")
    if restricted:
        scene["restricted"] = restricted.get("zones", [])
    return scene


def _env_context(workspace: Path) -> dict | None:
    p = workspace / "intel" / "environment_report.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def recommend_from(deltas: list[dict], chosen: dict, scene: dict,
                   env: dict | None) -> list[dict]:
    """Structured planning recommendations from simulation + measured data."""
    recs: list[dict] = []

    def add(recommendation: str, reason: str, evidence: dict, benefit: str,
            cost: str, confidence: float, source: str) -> None:
        recs.append({"recommendation": recommendation, "reason": reason,
                     "evidence": evidence, "expected_benefit": benefit,
                     "expected_cost": cost, "confidence": round(confidence, 2),
                     "source": source})

    for d in deltas:
        tag = d.get("tag", "")
        q = d.get("compare", {}).get("deltas", {}).get("quality_index", {})
        g = d.get("compare", {}).get("deltas", {}).get("gsd", {})
        if tag == "lower_altitude_20m" and q.get("delta_pct", 0) and \
                float(q["delta_pct"]) >= 10:
            add("fly 20 m lower for this region",
                f"finer GSD improves the quality index by ≈ {q['delta_pct']:.0f}% "
                f"(GSD {g.get('delta_pct', 0):.0f}%)",
                {"source": "simulation_estimate", "quality_delta_pct": q["delta_pct"],
                 "gsd_delta_pct": g.get("delta_pct")},
                "higher reconstruction quality/detail",
                f"battery +{d.get('battery_pct', '?')}%, duration "
                f"+{d.get('duration_pct', '?')}%",
                0.8, "simulation_estimate")
    # speed / overlap scenarios folded into the sweep result message
    recs.append({
        "recommendation": f"use the '{chosen.get('label')}' follow-up plan",
        "reason": "Pareto-best trade-off of quality, coverage, battery, time and risk "
                  "from the simulated sweep",
        "evidence": {"source": "simulation_estimate",
                     "metrics": {k: chosen.get("metrics", {}).get(k) for k in
                                 ("quality_index", "coverage_est", "gain_est",
                                  "duration_min", "battery_wh", "risk_score")}},
        "expected_benefit": "improved reconstruction of measured weak regions",
        "expected_cost": "mission flight time and battery consumption per the metrics",
        "confidence": 0.7, "source": "simulation_estimate"})
    for t in scene.get("targets", [])[:3]:
        add("revisit measured weak region",
            f"reconstruction confidence is low at {t.get('centroid')} "
            f"(≈{t.get('radius_m')} m radius)",
            {"source": "measured", "region": t.get("centroid"),
             "radius_m": t.get("radius_m")},
            "closes the measured coverage gap",
            "extra flight time for one target orbit",
            0.75, "measured")
    b = chosen.get("battery", {}) or {}
    if b.get("completion_prob", 1.0) < 1.0:
        add("battery reserve is insufficient for the full plan",
            f"estimated completion probability {b.get('completion_prob')}",
            {"source": "simulation_estimate",
             "consumed_wh": b.get("consumed_wh"),
             "remaining_pct": b.get("remaining_pct")},
            "re-route or swap battery before launch",
            "fewer frames per sortie",
            0.85, "simulation_estimate")
    w = weather_severity(env)
    if w > 0.4:
        add("expect degraded reconstruction in current conditions",
            f"environment severity {w:.2f} from the measured environment report",
            {"source": "measured", "severity": w,
             "dominant": (env or {}).get("dominant_condition")},
            "plan slower flight / higher overlap or wait for clear conditions",
            "longer mission",
            0.7, "measured")
    return recs


def path_waypoints(plan: dict, scene: dict) -> list[list[float]]:
    """2D waypoints approximating the plan pattern (for export / replay)."""
    ew, eh = float(scene.get("extent_w_m", 300.0)), float(scene.get("extent_h_m", 200.0))
    pattern = plan.get("pattern", "lawnmower")
    if pattern == "orbit":
        pts = [[0, 0], [ew, 0], [ew, eh], [0, eh], [0, 0]]
        return [[round(x, 1), round(y, 1)] for x, y in pts]
    fp = footprint(float(plan.get("altitude_m", 60.0)))
    spacing = max(fp["footprint_w_m"] * (1.0 - float(plan.get("side_overlap", 0.6))), 1.0)
    pts = []
    rows = int(ew / spacing) + 1
    for r in range(rows):
        x = float(r * spacing)
        pts.append([round(x, 1), 0.0])
        pts.append([round(x, 1), eh])
    return pts


def compose_plan(workspace: Path, env: dict | None = None,
                 scene: dict | None = None) -> dict:
    """Full planning bundle: baseline, what-if, sweep, chosen plan, recs."""
    scene = scene or build_scene(workspace)
    env = env if env is not None else _env_context(workspace)
    baseline_plan = plan_defaults()
    baseline = simulate(baseline_plan, scene)
    deltas = []
    for tag, change in _SCENARIOS.items():
        alt = change.get("altitude_m")
        plan = dict(baseline_plan)
        if alt is not None:
            plan["altitude_m"] = max(plan["altitude_m"] + alt, 5.0)
        if "speed_m_s" in change:
            plan["speed_m_s"] = max(plan["speed_m_s"] - change["speed_m_s"], 1.0)
        for key in ("forward_overlap", "side_overlap"):
            if key in change:
                plan[key] = min(plan[key] + change[key], 0.95)
        sim = simulate(plan, scene)
        cmp = compare(baseline, sim)
        battery_pct = cmp["deltas"].get("battery_consumed_wh", {}).get("delta_pct")
        duration_pct = cmp["deltas"].get("duration_min", {}).get("delta_pct")
        deltas.append({"tag": tag, "plan": plan, "compare": cmp,
                       "battery_pct": battery_pct, "duration_pct": duration_pct})
    sweep = optimize(scene, env=env)
    chosen_label = "BEST BALANCED"
    chosen = next((p for p in sweep.get("plans", []) if p["label"] == chosen_label),
                  (sweep.get("plans") or [None])[0])
    chosen = chosen or {"label": "NONE", "plan": dict(baseline_plan),
                        "metrics": {}, "battery": {}, "risk_level": "Low"}
    recommendations = recommend_from(deltas, chosen, scene, env)
    plan_doc = {
        "job": workspace.name,
        "scene": scene,
        "baseline": {"plan": baseline_plan,
                     "metrics": {k: baseline.get(k) for k in
                                 ("gsd", "frames", "coverage_est", "quality_index",
                                  "duration_min", "path_len_m")},
                     "battery": baseline["battery"]},
        "what_if": deltas,
        "sweep": {"candidates_evaluated": sweep.get("candidates_evaluated"),
                  "pareto_size": sweep.get("pareto_size"),
                  "method": sweep.get("method")},
        "candidate_plans": sweep.get("plans", []),
        "chosen_plan": chosen,
        "recommendations": recommendations,
        "estimates": {"label": "simulation estimates for planning comparison "
                               "— not measured outcomes", "measured_evidence": {
            "mean_confidence": scene.get("mean_confidence"),
            "blind_share": scene.get("blind_share"),
            "weak_regions": len(scene.get("targets", [])),
            "env_dominant": (env or {}).get("dominant_condition")}},
    }
    return plan_doc


@register
class MissionPlanningStage(PipelineStage):
    name = "mission_planning"
    description = "Follow-up mission plan: simulation, what-if, Pareto sweep, recommendations"
    artifact_rel = "intel/mission_plan.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run the digital-twin chain first")

    def execute(self) -> None:
        doc = compose_plan(self.workspace)
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "mission_plan.json").write_text(json.dumps(doc, indent=2))
        from app.services.mission_report import render_plan_markdown, write_plan_exports
        from app.services.report_generator import _md_to_html

        write_plan_exports(doc, intel_dir)
        (intel_dir / "mission_plan.html").write_text(_md_to_html(render_plan_markdown(doc)))
        chosen = doc["chosen_plan"]
        self._count = len(doc["recommendations"])
        self._detail = {"chosen": chosen.get("label"),
                        "recommendations": self._count,
                        "blind_share": doc["scene"].get("blind_share"),
                        "candidates": doc["sweep"]["candidates_evaluated"]}
        self._outputs = [{"kind": "data", "name": "mission_plan",
                          "path": str(intel_dir / "mission_plan.json")},
                         {"kind": "data", "name": "mission_plan_geojson",
                          "path": str(intel_dir / "mission_plan.geojson")},
                         {"kind": "data", "name": "mission_plan_kml",
                          "path": str(intel_dir / "mission_plan.kml")},
                         {"kind": "report", "name": "mission_plan_md",
                          "path": str(intel_dir / "mission_plan.md")},
                         {"kind": "report", "name": "mission_plan_html",
                          "path": str(intel_dir / "mission_plan.html")}]
        self.progress(1.0, {"chosen": chosen.get("label")})
