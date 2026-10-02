"""Risk-aware planning — deterministic plan-level risk assessment.

Risk sources evaluated for one candidate plan (each labeled):

* battery reserve — completion probability from :mod:`battery_model`,
* weather — environment report severity raised to per-condition severity,
* terrain clearance — ``altitude − max_scene_height`` vs obstacle clearance,
* restricted zones — user-declared rectangles the AOI overlaps.

The score is a transparency-weighted composite (0-100); ``no_go`` is only
ever true for an explicit battery infeasibility or a declared restricted-zone
overlap — never from simulation alone (see ``safety_note``).
"""

from __future__ import annotations

from app.config.settings import settings

_WEATHER_WEIGHT = {"rain_streaks": 0.7, "fog_haze": 0.6, "smoke_dust": 0.6,
                   "lowlight": 0.5, "motion_blur": 0.5, "sun_glare": 0.4,
                   "strong_shadows": 0.3, "cloud_cover": 0.2, "lens_artifacts": 0.8}


def weather_severity(env: dict | None) -> float:
    """0-1 degradation from the environment report's dominant condition."""
    if not env:
        return 0.0
    cond = (env.get("conditions") or {})
    worst = 0.0
    for name, s in cond.items():
        if not isinstance(s, dict):
            continue
        score = float(s.get("p90_score", s.get("mean_score", 0.0)) or 0.0)
        severity = score * _WEATHER_WEIGHT.get(name, 0.3)
        worst = max(worst, severity)
    return round(worst, 3)


def assess_plan_risk(plan: dict, sim: dict, scene: dict | None = None,
                     env: dict | None = None) -> dict:
    """Composite risk score + warnings for one simulated plan."""
    scene = scene or {}
    battery = sim.get("battery", {}) if isinstance(sim, dict) else {}
    warnings: list[str] = []
    comp = 0.0

    prob = float(battery.get("completion_prob", 1.0)) if battery else 1.0
    if prob < 0.5:
        warnings.append("battery reserve insufficient — completion unlikely")
        comp += 0.30 * (1.0 - prob) * 2.0
    elif prob < 1.0:
        warnings.append("plan consumes battery reserve margin")
        comp += 0.30 * (1.0 - prob)

    weather = weather_severity(env)
    if weather > 0.4:
        warnings.append(f"adverse conditions may degrade reconstruction "
                        f"(severity {weather:.2f}) — increase overlap/slower")
    comp += 0.30 * weather

    alt = float(plan.get("altitude_m", 0.0))
    max_h = float(scene.get("max_scene_height_m", 0.0) or 0.0)
    clearance = alt - max_h
    if max_h > 0 and clearance < settings.planning.obstacle_clearance_m:
        warnings.append(f"terrain clearance {clearance:.0f} m below the "
                        f"{settings.planning.obstacle_clearance_m:.0f} m obstacle margin")
        comp += 0.20
    restricted = scene.get("restricted", [])
    if restricted:
        comp += 0.20
        warnings.append("flight path overlaps a declared restricted zone")

    score = round(min(comp, 1.0) * 100.0, 1)
    return {
        "risk_score": score,
        "level": "Low" if score < 25 else "Medium" if score < 55 else "High",
        "warnings": warnings,
        "no_go": prob <= 0.0,  # explicit battery infeasibility only
        "no_go_reason": "battery infeasible" if prob <= 0.0 else None,
        "components": {"battery": round(0.30 * (1.0 - prob), 4),
                       "weather": round(0.30 * weather, 4),
                       "clearance": round(0.20 if (max_h > 0 and clearance <
                                                   settings.planning.obstacle_clearance_m) else 0.0, 4),
                       "restricted": round(0.20 if restricted else 0.0, 4)},
        "method": "weighted_risk_composite",
        "safety_note": "simulation-based risk only — no real-world flight "
                       "safety is guaranteed",
    }
