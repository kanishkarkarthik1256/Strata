"""Mission recommender — evidence-based follow-up flight recommendations.

Reads the run's measured quality signals (``intel/environment_report.json``,
``georef/gps_report.json``, ``confidence/confidence_report.json``) and turns
them into structured, actionable recommendations with reasons:

* re-fly regions — low-confidence clusters from the confidence overlay
  (each with centroid + radius, ready for a targeted second pass),
* altitude / overlap adjustments when GPS quality is poor or absent,
* sensor / timing suggestions when the environment report implicates
  weather or optics,
* processing suggestions carried over from the environment's adaptive
  strategies.

A ``mission_optimization_score`` (0-100) aggregates only the signals that
were actually measured — missing sources lower the *certainty*, never
fabricate a component. Outputs ``intel/mission_recommendations.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.mission_recommender")

_GRADE_CUTS = (("Excellent", 85.0), ("Good", 70.0), ("Fair", 50.0), ("Poor", 30.0))


def _grade(score: float) -> str:
    for name, cut in _GRADE_CUTS:
        if score >= cut:
            return name
    return "Unusable"


def _gps_score(gps: dict | None) -> float | None:
    if not gps:
        return None
    q = gps.get("gps_quality", {})
    val = q.get("gps_score")
    return float(val) if val is not None else None


def recommend_mission(mesh: TriangleMesh | None, env: dict | None = None,
                      gps: dict | None = None, conf: dict | None = None,
                      twin: dict | None = None) -> dict:
    """Deterministic recommendation + score from measured quality signals."""
    env_q = env.get("quality_score") if env else None
    env_cond = (env.get("dominant_condition") if env else None) or "clear"
    gps_q = _gps_score(gps)
    low_share = conf.get("low_share") if conf else None
    weak = conf.get("weak_regions", []) if conf else []
    twin = twin or {"objects": []}
    objects = twin.get("objects", [])
    structures = [o for o in objects if o.get("class") in ("roof", "wall", "structure")]

    recommendations: list[dict] = []
    components: dict[str, float] = {}

    # --- coverage: re-fly the weak clusters ---
    if low_share is not None:
        components["coverage"] = round(float((1.0 - low_share) * 100.0), 1)
        top = sorted(weak, key=lambda r: -r.get("vertices", 0))[:8]
        for i, r in enumerate(top, 1):
            recommendations.append({
                "type": "refly_region",
                "priority": "High" if i <= 3 else "Medium",
                "reason": f"reconstruction confidence cluster (≈{r.get('radius_m', 0):.1f} m "
                          f"radius, {r.get('vertices', 0)} vertices) — re-fly with more "
                          "overlap or lower altitude",
                "region": {"centroid": r.get("centroid"), "radius_m": r.get("radius_m")},
            })
        if not weak and low_share <= 0.02:
            recommendations.append({
                "type": "coverage",
                "priority": "Low",
                "reason": "coverage confidence is uniform — no targeted re-fly regions",
            })

    # --- GPS: altitude / overlap / hardware adjustments ---
    if gps_q is None:
        recommendations.append({
            "type": "gps",
            "priority": "High",
            "reason": "mission has no GPS quality record — georeferencing is absent; "
                      "enable RTK/PPK or a surveyed ground-control pass for GIS output",
        })
    elif gps_q < 70.0:
        recommendations.append({
            "type": "flight_parameter",
            "priority": "High" if gps_q < 40.0 else "Medium",
            "reason": f"GPS quality {gps_q:.0f}/100 — increase image overlap, fly lower "
                      "and slower, and prefer RTK/PPK corrections",
            "params": {"overlap_pct": ">85", "altitude": "lower", "gps_quality": gps_q},
        })
    else:
        recommendations.append({
            "type": "gps",
            "priority": "Low",
            "reason": f"GPS quality {gps_q:.0f}/100 is healthy — keep the current flight profile",
        })
    if gps_q is not None:
        components["gps"] = round(gps_q, 1)

    # --- environment: weather / optics / processing ---
    if env is not None:
        if env_q is not None:
            components["environment"] = round(float(env_q), 1)
        if env_q is not None and env_q < 45.0:
            recommendations.append({
                "type": "refly_conditions",
                "priority": "High",
                "reason": f"environmental quality {env_q:.0f}/100 dominated by "
                          f"'{env_cond}' — re-fly when conditions clear",
            })
        for strategy in env.get("adaptive_strategies", []):
            recommendations.append({
                "type": "processing" if strategy.get("target") != "hardware" else "sensor",
                "priority": "Medium",
                "reason": strategy.get("rationale", ""),
                "params": strategy.get("params", {}),
            })
    else:
        recommendations.append({
            "type": "environment",
            "priority": "Low",
            "reason": "no environment report — run environmental_intelligence on the frames",
        })

    # --- coverage of key objects ---
    if not structures and objects:
        recommendations.append({
            "type": "coverage",
            "priority": "Medium",
            "reason": "no standing structures detected — check that the AOI was fully "
                      "flown or that object extraction succeeded",
        })

    # --- mission optimization score over measured components ---
    weights = {"gps": 0.4, "environment": 0.3, "coverage": 0.3}
    have = {k: components[k] for k in weights if k in components}
    if have:
        wsum = sum(weights[k] for k in have)
        score = float(sum(v * weights[k] for k, v in have.items()) / wsum)
    else:
        score = 0.0
    note = ("score aggregates " + ", ".join(sorted(have)) + " only"
            if have else "no measurable quality signals present — score is 0")

    extent = (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)) if mesh is not None else None
    return {
        "mission_optimization_score": round(score, 1),
        "grade": _grade(score),
        "components": {k: {"score": v, "weight": weights[k]} for k, v in have.items()},
        "recommendations": recommendations,
        "profile": {
            "objects": len(objects), "structures": len(structures),
            "scene_extent_m": [round(float(v), 2) for v in extent] if extent is not None else None,
        },
        "note": note,
        "method": "weighted_quality_aggregation",
    }


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class MissionRecommendationStage(PipelineStage):
    name = "mission_recommendation"
    description = "Mission optimization score and evidence-based re-fly recommendations"
    artifact_rel = "intel/mission_recommendations.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run the digital-twin chain first")

    def execute(self) -> None:
        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "repaired_mesh.ply")
        env = _read_json(self.workspace / "intel" / "environment_report.json")
        gps = _read_json(self.workspace / "georef" / "gps_report.json")
        conf = _read_json(self.workspace / "confidence" / "confidence_report.json")
        twin = _read_json(self.workspace / "twin" / "twin.json")
        report = recommend_mission(mesh, env=env, gps=gps, conf=conf, twin=twin)
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "mission_recommendations.json").write_text(json.dumps(report, indent=2))
        self._count = len(report["recommendations"])
        self._detail = {"score": report["mission_optimization_score"],
                        "grade": report["grade"],
                        "recommendations": self._count}
        self._outputs = [{"kind": "data", "name": "mission_recommendations",
                          "path": str(intel_dir / "mission_recommendations.json")}]
        self.progress(1.0, {"score": report["mission_optimization_score"]})


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
