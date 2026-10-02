"""Mission simulator — physically/model-derived planning estimates.

Every quantity this module produces is an *estimate* (photogrammetry and
propulsion proxies), never a measured value. The simulator is deliberately
self-contained so it can run with zero historical data; scene evidence
(measured mean confidence, blind-spot share) is optional input that refines
the quality estimate.

Model notes (exposed as ``method`` labels in every result):

* GSD / footprint: pinhole projection  ``footprint = (sensor/focal)·altitude``,
  ``gsd = footprint / pixels`` (nadir).
* frames / path: lawnmower passes at ``pass_spacing = footprint_w·(1-side)``
  and ``line_step = footprint_h·(1-forward)``; orbit/revisit paths fly a
  closed loop at the along-track step.
* battery: see :mod:`app.services.battery_model`.
* relative quality index: anchored to ``planning.quality_ref_altitude_m``
  GSD, plus overlap and speed (motion blur) terms, rescaled by measured scene
  confidence when available. It is a *comparative* index, not an absolute
  reconstruction-accuracy promise.
"""

from __future__ import annotations

from app.config.settings import settings
from app.services.battery_model import battery_estimate

_CAM = None


def camera_defaults() -> dict:
    p = settings.planning
    return {"width_px": p.camera_width_px, "height_px": p.camera_height_px,
            "sensor_width_mm": p.sensor_width_mm, "sensor_height_mm": p.sensor_height_mm,
            "focal_mm": p.focal_mm}


def footprint(altitude_m: float, camera: dict | None = None) -> dict:
    """Nadir image footprint + GSD from altitude and the camera model."""
    cam = dict(camera or camera_defaults())
    sw = cam["sensor_width_mm"] / cam["focal_mm"]
    sh = cam["sensor_height_mm"] / cam["focal_mm"]
    fw = float(sw * altitude_m)
    fh = float(sh * altitude_m)
    return {
        "footprint_w_m": round(fw, 2),
        "footprint_h_m": round(fh, 2),
        "gsd_cm": round(100.0 * fw / cam["width_px"], 2),
        "fov_h_deg": round(2.0 * float(__import__("math").atan(sw / 2.0)) * 180.0
                           / 3.141592653589793, 1),
        "method": "pinhole_projection_nadir",
    }


def plan_defaults() -> dict:
    p = settings.planning
    return {"altitude_m": p.default_altitude_m, "speed_m_s": p.default_speed_m_s,
            "forward_overlap": p.default_forward_overlap,
            "side_overlap": p.default_side_overlap, "pattern": p.default_pattern}


def path_geometry(plan: dict, extent_w_m: float, extent_h_m: float,
                  scene_targets: list[dict] | None = None) -> dict:
    """Path length / frames for a pattern over a rectangular AOI.

    ``scene_targets`` (centroids + radius of revisit zones) are used by the
    ``revisit`` pattern; orbit flies the AOI perimeter loop.
    """
    alt = float(plan["altitude_m"])
    fp = footprint(alt)
    fw, fh = fp["footprint_w_m"], fp["footprint_h_m"]
    fwd, side = float(plan.get("forward_overlap", 0.7)), float(plan.get("side_overlap", 0.6))
    pattern = plan.get("pattern", "lawnmower")
    step = max(fh * (1.0 - fwd), 0.5)
    spacing = max(fw * (1.0 - side), 0.5)

    if pattern == "orbit":
        path_len = 2.0 * (extent_w_m + extent_h_m)
        frames = max(int(path_len / step) + 1, 4)
        return {"pattern": pattern, "path_len_m": round(path_len, 1),
                "frames": frames, "line_step_m": round(step, 2),
                "passes": 1, "method": "orbit_loop_estimate"}
    if pattern == "revisit":
        targets = scene_targets or []
        path_len = sum(2.0 * 3.14159 * max(float(t.get("radius_m", 15.0)), 10.0)
                       for t in targets)
        frames = max(int(path_len / step) + 1, len(targets))
        return {"pattern": pattern, "path_len_m": round(path_len, 1),
                "frames": frames, "line_step_m": round(step, 2),
                "passes": len(targets), "targets": len(targets),
                "method": "target_orbit_estimate"}

    # lawnmower / grid: parallel passes along the longer axis
    passes = max(int(extent_w_m / spacing) + 1, 1)
    if pattern == "grid":  # both directions
        passes *= 2
    path_len = float(passes * extent_h_m) + extent_w_m
    line = max(extent_h_m / step, 0.0)
    frames = max(int(line) + 1, 1) * passes
    return {"pattern": pattern, "path_len_m": round(path_len, 1),
            "frames": int(frames), "passes": passes,
            "line_step_m": round(step, 2), "pass_spacing_m": round(spacing, 2),
            "method": "lawnmower_geometry_estimate"}


def relative_quality_index(plan: dict, measured_conf: float | None = None,
                           blind_share: float | None = None) -> dict:
    """Comparative reconstruction-quality index (0-1) from plan parameters.

    Lower altitude (finer GSD), higher overlap and slower speed (less motion
    blur) all raise the index; measured scene confidence rescales it when the
    caller has real evidence. Method label is explicit: this is a model
    estimate for *comparing plans*, not an accuracy guarantee.
    """
    alt = max(float(plan["altitude_m"]), 1.0)
    ref = settings.planning.quality_ref_altitude_m
    gsd_term = min(1.0, (ref / alt) ** 0.5)
    fwd = float(plan.get("forward_overlap", 0.7))
    side = float(plan.get("side_overlap", 0.6))
    overlap_term = 0.5 + 0.5 * ((fwd + side) / 1.6)
    v = float(plan.get("speed_m_s", 8.0))
    speed_term = 1.0 if v <= 5.0 else max(0.55, 1.25 - v * 0.05)
    raw = float(np_clip(gsd_term * overlap_term * speed_term, 0.0, 1.0))
    if measured_conf is not None:
        raw *= float(np_clip(measured_conf, 0.0, 1.0)) ** 0.5
    if blind_share is not None:
        raw *= 1.0 - 0.6 * float(np_clip(blind_share, 0.0, 1.0))
    return {"value": round(float(np_clip(raw, 0.0, 1.0)), 3),
            "range": [round(max(raw - 0.08, 0.0), 3), round(min(raw + 0.08, 1.0), 3)],
            "method": "gsd_overlap_speed_model_estimate"}


def np_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def simulate(plan: dict, scene: dict | None = None) -> dict:
    """Full estimate bundle for one plan over one scene AOI.

    ``scene``: ``extent_w_m``, ``extent_h_m`` (required), optional measured
    ``mean_confidence``, ``blind_share`` and ``targets``.
    """
    scene = scene or {}
    if not scene.get("extent_w_m") or not scene.get("extent_h_m"):
        raise ValueError("scene extent_w_m and extent_h_m are required to simulate coverage")
    ew, eh = float(scene["extent_w_m"]), float(scene["extent_h_m"])
    plan = {**plan_defaults(), **plan}
    fp = footprint(plan["altitude_m"])
    geo = path_geometry(plan, ew, eh, scene.get("targets"))
    dist = geo["path_len_m"]
    frames = geo["frames"]
    speed = max(float(plan["speed_m_s"]), 0.5)
    hover_h = 0.0
    dur_h = dist / (speed * 3600.0) + hover_h
    bat = battery_estimate(plan, dur_h, hover_h)
    pixels_mp = float(camera_defaults()["width_px"] * camera_defaults()["height_px"]) / 1e6
    q = relative_quality_index(plan, scene.get("mean_confidence"),
                               scene.get("blind_share"))
    area_per_frame_m2 = fp["footprint_w_m"] * fp["footprint_h_m"]
    nominal_area = area_per_frame_m2 * frames
    coverage_est = min(1.0, nominal_area / max(ew * eh, 1e-9)) if ew * eh > 0 else 0.0
    return {
        "plan": {k: plan[k] for k in ("altitude_m", "speed_m_s", "forward_overlap",
                                      "side_overlap", "pattern")},
        "gsd": fp["gsd_cm"], "footprint_w_m": fp["footprint_w_m"],
        "footprint_h_m": fp["footprint_h_m"],
        "frames": frames, "path_len_m": geo["path_len_m"],
        "coverage_est": round(float(np_clip(coverage_est, 0.0, 1.0)), 3),
        "coverage_range": [round(max(coverage_est - 0.03, 0.0), 3),
                           round(min(coverage_est + 0.03, 1.0), 3)],
        "quality_index": q["value"], "quality_range": q["range"],
        "duration_min": round(dur_h * 60.0, 1),
        "battery": bat,
        "processing_load_gpix": round(frames * pixels_mp * 1.08 / 1000.0, 2),
        "method": {"gsd": fp["method"], "geometry": geo["method"],
                   "battery": bat["method"], "quality": q["method"],
                   "coverage": "nominal_footprint_overlap_estimate"},
        "estimate": True,
        "estimate_note": "all values are model estimates for planning "
                         "comparison — not measured telemetry",
    }


def compare(base: dict, scenario: dict) -> dict:
    """Percentage deltas between two simulation bundles, plan comparison.

    Fields that a metric cannot support honestly (e.g. no ``quality_index``
    because of missing scene evidence) are reported as ``None`` with an
    ``insufficient_evidence`` flag rather than invented.
    """
    keys = ("coverage_est", "quality_index", "frames", "duration_min",
            "path_len_m", "gsd")
    diffs = {}
    for key in keys:
        b = base.get(key)
        s = scenario.get(key)
        if isinstance(b, (int, float)) and isinstance(s, (int, float)) and b:
            diffs[key] = {"base": round(b, 3), "scenario": round(s, 3),
                          "delta_pct": round((s - b) / abs(b) * 100.0, 1)}
        else:
            diffs[key] = {"base": b, "scenario": s, "delta_pct": None,
                          "note": "insufficient evidence for a reliable comparison"}
    bbat = base.get("battery", {})
    sbat = scenario.get("battery", {})
    for k in ("consumed_wh", "remaining_pct", "completion_prob"):
        if k in bbat and k in sbat and isinstance(bbat[k], (int, float)) \
                and isinstance(sbat[k], (int, float)):
            bv, sv = bbat[k], sbat[k]
            diffs[f"battery_{k}"] = {"base": bv, "scenario": sv,
                                     "delta_pct": round((sv - bv) / abs(bv) * 100.0, 1)
                                     if bv else None}
    return {"baseline": {"plan": base.get("plan"), "gsd": base.get("gsd"),
                         "coverage_est": base.get("coverage_est"),
                         "quality_index": base.get("quality_index"),
                         "duration_min": base.get("duration_min"),
                         "frames": base.get("frames"),
                         "battery_consumed_wh": bbat.get("consumed_wh")},
            "scenario": {"plan": scenario.get("plan"), "gsd": scenario.get("gsd"),
                         "coverage_est": scenario.get("coverage_est"),
                         "quality_index": scenario.get("quality_index"),
                         "duration_min": scenario.get("duration_min"),
                         "frames": scenario.get("frames"),
                         "battery_consumed_wh": sbat.get("consumed_wh")},
            "deltas": diffs,
            "method": "what_if_simulation_compare",
            "estimate": True}
