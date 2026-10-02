"""Follow-up path generation + multi-objective Pareto optimization.

Enumerates candidate plans (pattern × altitude × speed × overlap) over the
scene AOI and evaluates each with the mission simulator + risk planner, then
keeps the non-dominated set across five objectives:

``quality`` and ``coverage`` (maximize), ``duration``, ``battery`` and
``risk`` (minimize). From the Pareto front it extracts labelled archetypes
(BEST QUALITY / BEST BATTERY / BEST TIME / BEST SAFETY / BEST BALANCED) —
the comparison surface the API and UI present.

Coverage-gain of a candidate is a simulation estimate (the plan re-images
the AOI, so its gain ceiling is the measured blind share, discounted for
geometry); it never claims measured improvement.
"""

from __future__ import annotations

import itertools

from app.config.settings import settings
from app.services.mission_simulator import simulate
from app.services.risk_aware_planner import assess_plan_risk

_LABELS = ("BEST QUALITY", "BEST BATTERY", "BEST TIME", "BEST SAFETY", "BEST BALANCED")


def candidate_plans(scene: dict) -> list[dict]:
    p = settings.planning
    patterns = ["lawnmower", "grid", "orbit"]
    if scene.get("targets"):
        patterns.append("revisit")
    plans = []
    for pattern in patterns:
        for alt, speed, fwd in itertools.product(p.optimize_alts_m,
                                                 p.optimize_speeds_m_s,
                                                 p.optimize_overlaps):
            plans.append({"altitude_m": float(alt), "speed_m_s": float(speed),
                          "forward_overlap": float(fwd), "side_overlap": float(p.default_side_overlap),
                          "pattern": pattern})
    plans.sort(key=lambda x: (x["pattern"], x["altitude_m"], x["speed_m_s"],
                              x["forward_overlap"]))
    step = max(1, len(plans) // p.max_candidate_plans)
    return plans[::step] or plans[:1]


def evaluate_plan(plan: dict, scene: dict, env: dict | None = None) -> dict:
    sim = simulate(plan, scene)
    risk = assess_plan_risk(plan, sim, scene, env=env)
    gain = None
    if scene.get("blind_share") is not None:
        gain = round(float(scene["blind_share"]) *
                     min(1.0, float(sim.get("coverage_est", 0.0))), 4)
    metrics = {
        "quality_index": sim["quality_index"],
        "coverage_est": sim["coverage_est"],
        "gain_est": gain,
        "duration_min": sim["duration_min"],
        "battery_wh": sim["battery"]["consumed_wh"],
        "battery_completion": sim["battery"]["completion_prob"],
        "risk_score": risk["risk_score"],
    }
    return {"plan": plan, "sim": sim, "risk": risk, "metrics": metrics}


def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    rng = hi - lo or 1.0
    return [(v - lo) / rng for v in values]


def pareto_front(evals: list[dict]) -> tuple[list[dict], dict[str, float]]:
    """Non-dominated front over 5 objectives; returns (front, obj->direction)."""
    objectives = {"quality_index": +1.0, "coverage_est": +1.0,
                  "duration_min": -1.0, "battery_wh": -1.0, "risk_score": -1.0}
    vals = {k: [e["metrics"][k] for e in evals] for k in objectives}
    norm = {k: _normalize(v) for k, v in vals.items()}

    def dominates(a: int, b: int) -> bool:
        better, worse = False, False
        for k, direction in objectives.items():
            da = norm[k][a] * direction
            db = norm[k][b] * direction
            if da > db + 1e-9:
                better = True
            elif da < db - 1e-9:
                worse = True
        return better and not worse

    front = [i for i in range(len(evals))
             if not any(dominates(j, i) for j in range(len(evals)))]
    return [evals[i] for i in front], objectives


def optimize(scene: dict, env: dict | None = None) -> dict:
    """Run the sweep → Pareto front → labelled archetype plans."""
    candidates = candidate_plans(scene)
    evals = [evaluate_plan(p, scene, env=env) for p in candidates]
    front, direction = pareto_front(evals)
    if not front:
        return {"plans": [], "candidates_evaluated": len(candidates),
                "message": "no feasible plan found for the given constraints"}
    full = evals if len(front) < 3 else front
    indices = range(len(full))

    def pick(key: str) -> int:
        if key == "quality_index":
            return max(indices, key=lambda i: full[i]["metrics"][key])
        if key == "battery_wh":
            return min(indices, key=lambda i: full[i]["metrics"][key])
        if key == "duration_min":
            return min(indices, key=lambda i: full[i]["metrics"][key])
        if key == "risk_score":
            return min(indices, key=lambda i: full[i]["metrics"][key])
        # balanced: smallest sum of objective deficits from each best value
        best = {k: max(e["metrics"][k] * d for e in full)
                for k, d in direction.items()}
        def deficit(i: int) -> float:
            return sum(best[k] - full[i]["metrics"][k] * d
                       for k, d in direction.items())
        return min(indices, key=deficit)

    chosen = {}
    for label, key in zip(_LABELS, ("quality_index", "battery_wh", "duration_min",
                                    "risk_score", "balanced")):
        best_i = pick("balanced" if key == "balanced" else key)
        chosen[label] = full[best_i]

    plans = []
    for label, ev in chosen.items():
        m = ev["metrics"]
        sim = ev["sim"]
        plans.append({
            "label": label,
            "plan": ev["plan"],
            "metrics": {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                        for k, v in m.items()},
            "gsd_cm": sim["gsd"],
            "frames": sim["frames"],
            "battery": {"consumed_wh": sim["battery"]["consumed_wh"],
                        "remaining_pct": sim["battery"]["remaining_pct"],
                        "completion_prob": sim["battery"]["completion_prob"]},
            "risk_level": ev["risk"]["level"],
            "warnings": ev["risk"]["warnings"][:3],
        })
    return {"plans": plans, "candidates_evaluated": len(candidates),
            "pareto_size": len(front),
            "objectives": sorted(direction),
            "method": "multi_objective_pareto_sweep",
            "estimate": True,
            "note": "plan metrics are simulation estimates for comparison "
                    "— not measured outcomes"}
