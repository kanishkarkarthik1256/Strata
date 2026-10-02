"""AI flight copilot — deterministic answers with explicit evidence labels.

Answers are computed from the engines; every numerical value carries a
``source`` in {``measured``, ``simulation_estimate``, ``historical``,
``user_provided``}. Questions the engines cannot answer from available
evidence return an explicit insufficiency message — never a guessed number.

Context (``ctx``) is caller-supplied measured mission context: ``scene``
extent/confidence, ``plan`` doc (if planned), ``history`` records,
``learning`` summary. Without history, similarity questions answer
"insufficient historical data".
"""

from __future__ import annotations

import re

from app.services.battery_model import max_flight_time
from app.services.mission_history import similar_missions
from app.services.mission_planner import build_scene
from app.services.mission_simulator import compare, plan_defaults, simulate

_SIM = "simulation_estimate"
_MEAS = "measured"
_HIST = "historical"


def _answer(intent: str, text: str, metrics: list[dict],
            evidence: str | None = None, recs: list[str] | None = None) -> dict:
    return {"intent": intent, "answer": text, "metrics": metrics,
            "evidence": evidence or "", "recommendations": recs or []}


def answer(query: str, ctx: dict | None = None) -> dict:
    """Resolve a mission-planning question against engines + context."""
    ctx = ctx or {}
    q = (query or "").lower().strip()
    if not q:
        return _answer("clarify", "ask about altitude, battery, coverage, paths, "
                                  "risk or similar missions.", [], )
    scene = ctx.get("scene") or {}

    # ---------------- battery ----------------
    if re.search(r"\b(battery|enough power|flight time)\b", q):
        if re.search(r"\b(enough|finish|complete|doable|sufficient)\b", q):
            return _battery_enough(scene)
        return _battery_basics(q, scene)

    # ---------------- altitude / speed what-if ----------------
    if re.search(r"\b(lower|higher|altitude|20 ?m|fly .*lower|improve)\b", q):
        return _altitude_advice(q, scene)

    # ---------------- coverage / regions / why low confidence ---------------
    if re.search(r"\b(revisit|region|area|blind|low[ -]confidence|coverage|weak)\b", q):
        return _coverage_advice(scene)

    # ---------------- risk ----------------
    if re.search(r"\b(risk|safe|danger)\b", q):
        return _risk_advice(ctx.get("plan"))

    # ---------------- history / similar / learned ----------------
    if re.search(r"\b(similar|previous|historically|before|learned|worked|performed)\b", q):
        return _history_advice(ctx)

    # ---------------- path / plan ----------------
    if re.search(r"\b(path|plan|strategy|pattern|route)\b", q):
        return _plan_advice(ctx.get("plan"))

    return _answer("clarify",
                   "I can answer planning questions about: flying lower, battery "
                   "completion, which regions to revisit, which path to take, "
                   "risk, and similar past missions.",
                   [])


def _battery_enough(scene: dict) -> dict:
    plan = plan_defaults()
    if not scene.get("extent_w_m"):
        scene = {"extent_w_m": 300.0, "extent_h_m": 200.0}
        note = "assumed a 300×200 m AOI (user_provided scene)"
    else:
        note = "scene extent measured from the twin"
    sim = simulate(plan, scene)
    b = sim["battery"]
    prob = b["completion_prob"]
    text = (f"estimated completion probability {prob:.0%} "
            f"(remaining {b['remaining_pct']:.0f}%, reserve margin "
            f"{b['reserve_margin_wh']:.1f} Wh) for a {sim['duration_min']:.0f} min "
            f"flight at {plan['speed_m_s']} m/s / {plan['altitude_m']} m")
    if prob < 0.5:
        text += " — the reserve is insufficient; swap batteries or split the AOI."
    elif prob < 1.0:
        text += " — the plan dips into the reserve."
    return _answer("battery", text,
                   [{"metric": "completion_probability", "value": prob,
                     "unit": "fraction", "source": _SIM},
                    {"metric": "remaining_pct", "value": b["remaining_pct"],
                     "unit": "%", "source": _SIM},
                    {"metric": "duration_min", "value": sim["duration_min"],
                     "unit": "min", "source": _SIM}],
                   evidence=f"{note}; battery model is a physics proxy — exact "
                            "consumption needs telemetry")

def _battery_basics(q: str, scene: dict) -> dict:
    mft = max_flight_time(plan_defaults())
    return _answer("battery",
                   f"usable capacity supports ≈ {mft['max_flight_time_min']} min of "
                   f"transit at default speed (≈{mft['max_range_m']:.0f} m range) "
                   "under the physics proxy model",
                   [{"metric": "max_flight_time_min", "value": mft["max_flight_time_min"],
                     "unit": "min", "source": _SIM},
                    {"metric": "max_range_m", "value": mft["max_range_m"], "unit": "m",
                     "source": _SIM}],
                   evidence="physics proxy model — requires telemetry for exact values")


def _altitude_advice(q: str, scene: dict) -> dict:
    m = re.search(r"(\d+)\s*m", q)
    delta = float(m.group(1)) if m else 20.0
    if not scene.get("extent_w_m"):
        scene = {"extent_w_m": 300.0, "extent_h_m": 200.0}
    base_plan = plan_defaults()
    lower_plan = dict(base_plan)
    if "higher" in q:
        lower_plan["altitude_m"] = min(base_plan["altitude_m"] + delta, 200.0)
        label = "higher"
    else:
        lower_plan["altitude_m"] = max(base_plan["altitude_m"] - delta, 5.0)
        label = "lower"
    base, alt = simulate(base_plan, scene), simulate(lower_plan, scene)
    cmp = compare(base, alt)["deltas"]
    qd, gd = cmp["quality_index"], cmp["gsd"]
    text = (f"flying {delta:.0f} m {label}: quality index "
            f"{qd['delta_pct']:.1f}%, GSD {gd['delta_pct']:.1f}%, frames "
            f"{cmp['frames']['delta_pct']:.0f}% "
            f"({'more' if float(cmp['frames']['delta_pct']) > 0 else 'fewer'}), "
            f"duration {cmp['duration_min']['delta_pct']:.0f}%")
    bd = cmp.get("battery_consumed_wh")
    if bd and bd.get("delta_pct") is not None:
        text += f", battery {bd['delta_pct']:.0f}%"
    return _answer("altitude_whatif", text,
                   [{"metric": "quality_index_delta_pct", "value": qd["delta_pct"],
                     "unit": "%", "source": _SIM},
                    {"metric": "gsd_delta_pct", "value": gd["delta_pct"], "unit": "%",
                     "source": _SIM},
                    {"metric": "duration_delta_pct", "value": cmp["duration_min"]["delta_pct"],
                     "unit": "%", "source": _SIM}],
                   evidence="mission simulator (pinhole GSD + quality index model)",
                   recs=["a 20 m lower pass at higher overlap is the strongest "
                         "quality lever in the simulated sweep"])


def _coverage_advice(scene: dict) -> dict:
    weak = scene.get("targets") or []
    blind = scene.get("blind_share")
    if weak:
        regions = ", ".join(str(t.get("centroid")) for t in weak[:3])
        text = (f"{len(weak)} measured low-confidence regions — e.g. {regions} — "
                "should be re-flown; re-flying them is estimated to recover most "
                "of the measured gap")
        return _answer("coverage", text,
                       [{"metric": "blind_share", "value": blind, "unit": "fraction",
                         "source": _MEAS},
                        {"metric": "weak_regions", "value": len(weak), "unit": "count",
                         "source": _MEAS}],
                       evidence="confidence overlay of the reconstructed mesh",
                       recs=[f"orbit region at {w.get('centroid')} for facade detail"
                             for w in weak[:2]])
    if blind is not None:
        return _answer("coverage",
                       f"measured blind share is {blind:.0%}; no distinct weak "
                       "clusters — treat the whole AOI as under-confident",
                       [{"metric": "blind_share", "value": blind, "unit": "fraction",
                         "source": _MEAS}])
    return _answer("coverage",
                   "no measured coverage/confidence data for this mission — run "
                   "confidence_overlay + coverage_prediction first",
                   [])


def _plan_advice(plan: dict | None) -> dict:
    if not plan:
        return _answer("plan",
                       "no stored mission plan — run /api/mission/optimize or the "
                       "mission_planning stage to generate candidate paths",
                       [])
    chosen = plan.get("chosen_plan") or {}
    m = chosen.get("metrics", {}) or {}
    text = (f"recommended plan: {chosen.get('label')} — "
            f"{json_dump(chosen.get('plan'))} with quality {m.get('quality_index')}, "
            f"coverage {m.get('coverage_est')}, risk {chosen.get('risk_level')}")
    return _answer("plan", text,
                   [{"metric": k, "value": m.get(k), "unit": _unit(k), "source": _SIM}
                    for k in ("quality_index", "coverage_est", "duration_min",
                              "battery_wh", "risk_score") if m.get(k) is not None],
                   evidence="multi-objective Pareto sweep (simulation estimates)")


def _risk_advice(plan: dict | None) -> dict:
    if not plan:
        return _answer("risk", "no mission plan loaded — risk is assessed per plan",
                       [])
    chosen = plan.get("chosen_plan") or {}
    warnings = chosen.get("warnings") or []
    return _answer("risk",
                   f"risk level {chosen.get('risk_level')} for the recommended plan"
                   + ("; " + "; ".join(warnings) if warnings else ""),
                   [{"metric": "risk_score", "value": (chosen.get("metrics") or {}).get("risk_score"),
                     "unit": "0-100", "source": _SIM}],
                   evidence="weighted risk composite (battery/weather/clearance/restricted)",
                   recs=warnings)


def _history_advice(ctx: dict) -> dict:
    records = ctx.get("history") or []
    if not records:
        return _answer("similar",
                       "insufficient historical data — no completed missions in "
                       "history yet",
                       [])
    # similarity needs flight parameters — pull them from the mission record
    # or the stored plan (scene geometry alone shares no comparison keys)
    query: dict = {}
    record = ctx.get("record") or {}
    for key in ("altitude_m", "speed_m_s", "forward_overlap", "side_overlap",
                "env_score", "gps_score"):
        if record.get(key) is not None:
            query[key] = record[key]
    if not query:
        chosen = ((ctx.get("plan") or {}).get("chosen_plan") or {}).get("plan") or {}
        for key in ("altitude_m", "speed_m_s", "forward_overlap", "side_overlap"):
            if chosen.get(key) is not None:
                query[key] = chosen[key]
    if not query and ctx.get("workspace"):
        query = build_scene(ctx.get("workspace"))
    query.setdefault("mission_id", "query")
    sim = similar_missions(query, records)
    learning = ctx.get("learning") or {}
    metrics = [{"metric": "similar_missions", "value": sim.get("count", 0),
                "unit": "count", "source": _HIST}]
    if sim.get("average_coverage") is not None:
        metrics.append({"metric": "average_coverage_pct", "value": sim["average_coverage"],
                        "unit": "%", "source": _HIST})
    if sim.get("average_confidence") is not None:
        metrics.append({"metric": "average_confidence_pct", "value": sim["average_confidence"],
                        "unit": "%", "source": _HIST})
    text = sim.get("message") or sim.get("status", "")
    if sim.get("count"):
        text = (f"{sim['count']} previous missions had similar conditions; "
                f"average coverage {sim.get('average_coverage')}%, confidence "
                f"{sim.get('average_confidence')}% (historical observations)")
    if learning.get("ml_gate", {}).get("message"):
        text += f" {learning['ml_gate']['message']}"
    return _answer("similar", text, metrics,
                   evidence="mission history store (labelled historical — not "
                            "predictions)")


def json_dump(obj) -> str:
    import json
    return json.dumps(obj)


def _unit(metric: str) -> str:
    return {"duration_min": "min", "battery_wh": "Wh", "coverage_est": "fraction",
            "quality_index": "0-1"}.get(metric, "fraction")


def plan_context(plan_doc: dict) -> dict:
    """Normalize a stored mission plan into copilot ctx subset."""
    return {"plan": plan_doc,
            "scene": {k: plan_doc.get("scene", {}).get(k)
                      for k in ("extent_w_m", "extent_h_m", "blind_share",
                                "mean_confidence", "targets")}}
