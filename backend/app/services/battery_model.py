"""Configurable battery estimation model.

Consumption is a documented physics proxy, NOT telemetry:

* hover power  ``P_hover = k_hover · mass``  (calibrated W/kg),
* transit power ``P_cruise = 0.75·P_hover + k_drag·speed³ + P_avionics``
  (induced-power scaling plus a parasitic drag term — good to ~10-20% for a
  multirotor in calm air, worse with wind),
* energy ``E = P_cruise·t_transit + P_hover·t_hover`` against usable capacity
  ``C·(1 − reserve_pct)``.

Completion probability and margins derive from the reserve shortfall. Every
number is labeled a model estimate; exact consumption requires flight
telemetry.
"""

from __future__ import annotations

from app.config.settings import settings


def _power_model(plan: dict) -> dict:
    p = settings.planning
    mass = float(plan.get("mass_kg", p.drone_mass_kg))
    speed = max(float(plan.get("speed_m_s", p.default_speed_m_s)), 0.0)
    hover = float(plan.get("hover_power_w_per_kg", p.hover_power_w_per_kg)) * mass
    cruise = 0.75 * hover + float(plan.get("drag_coeff", p.drag_coeff)) * speed**3 \
        + float(plan.get("avionics_w", p.avionics_w))
    return {"hover_w": round(hover, 1), "cruise_w": round(cruise, 1), "speed_m_s": speed}


def battery_estimate(plan: dict, duration_h: float, hover_h: float = 0.0) -> dict:
    """Consumption / remaining / reserve / completion-probability estimates."""
    p = settings.planning
    cap_wh = float(plan.get("battery_capacity_wh", p.battery_capacity_wh))
    reserve_pct = float(plan.get("battery_reserve_pct", p.battery_reserve_pct))
    reserve_wh = cap_wh * reserve_pct / 100.0
    pm = _power_model(plan)
    transit_h = max(float(duration_h) - hover_h, 0.0)
    # W·h = Wh directly (no 1/1000 — that factor would mislabel Wh as kWh)
    used = pm["cruise_w"] * transit_h + pm["hover_w"] * hover_h
    remaining = cap_wh - used
    usable_below_reserve = cap_wh * (1.0 - reserve_pct / 100.0)
    # completion probability: how far into the reserve the plan would push
    if used <= 0:
        prob = 1.0
    elif used >= cap_wh:
        prob = 0.0
    else:
        excess = used - usable_below_reserve
        prob = float(max(0.0, min(1.0, 1.0 - excess / max(reserve_wh, 1e-6))))
    return {
        "consumed_wh": round(used, 2),
        "consumed_pct": round(used / cap_wh * 100.0, 1) if cap_wh else 0.0,
        "remaining_wh": round(max(remaining, 0.0), 2),
        "remaining_pct": round(max(remaining / cap_wh * 100.0, 0.0), 1) if cap_wh else 0.0,
        "reserve_wh": round(reserve_wh, 2),
        "reserve_margin_wh": round(remaining - reserve_wh, 2),
        "completion_prob": round(prob, 3),
        "power": pm,
        "params": {"capacity_wh": cap_wh, "mass_kg": plan.get("mass_kg", settings.planning.drone_mass_kg)},
        "method": "physics_proxy_model",
        "estimate": True,
        "note": "model estimate — exact consumption requires flight telemetry",
    }


def max_flight_time(plan: dict) -> dict:
    """Largest transit time within usable capacity (reserve kept)."""
    pm = _power_model(plan)
    cap = float(plan.get("battery_capacity_wh", settings.planning.battery_capacity_wh))
    reserve_pct = float(plan.get("battery_reserve_pct", settings.planning.battery_reserve_pct))
    usable = cap * (1.0 - reserve_pct / 100.0)
    hours = usable / pm["cruise_w"] if pm["cruise_w"] else 0.0
    return {"max_flight_time_min": round(hours * 60.0, 1),
            "usable_wh": round(usable, 2),
            "max_range_m": round(hours * pm["speed_m_s"] * 3600.0, 0)
            if pm["speed_m_s"] else 0,
            "method": "physics_proxy_model", "estimate": True}
