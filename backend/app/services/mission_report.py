"""Mission plan report + geometry exports.

* ``export_plan_geometry`` writes GeoJSON (LineString) and KML of the chosen
  plan's waypoints,
* ``render_plan_markdown`` renders the plan document (baseline, what-if
  deltas, candidates, recommendations) with estimate labels intact.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.mission_planner import path_waypoints


def export_plan_geometry(doc: dict, out_dir: Path) -> None:
    """Write mission_plan.geojson + mission_plan.kml for the chosen plan."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = doc.get("chosen_plan") or {}
    plan = chosen.get("plan") or {}
    waypoints = path_waypoints(plan, doc.get("scene", {}))
    coords = [[x, y] for x, y in waypoints]
    gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"label": chosen.get("label"), "pattern": plan.get("pattern"),
                           "altitude_m": plan.get("altitude_m"),
                           "speed_m_s": plan.get("speed_m_s"),
                           "source": "simulation_estimate",
                           "mission": doc.get("job")},
            "geometry": {"type": "LineString", "coordinates": coords},
        }],
    }
    (out_dir / "mission_plan.geojson").write_text(json.dumps(gj, indent=2))
    kml_lines = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                 "<kml xmlns=\"http://www.opengis.net/kml/2.2\">",
                 "  <Document>",
                 "    <name>Mission plan</name>",
                 f"    <description>{chosen.get('label', 'plan')} "
                 f"(simulation estimate)</description>",
                 "    <Placemark><name>plan</name><LineString><coordinates>"]
    for x, y in coords:
        kml_lines.append(f"      {x},{y},0")
    kml_lines += ["    </coordinates></LineString></Placemark>",
                  "  </Document>", "</kml>"]
    (out_dir / "mission_plan.kml").write_text("\n".join(kml_lines))


def render_plan_markdown(doc: dict) -> str:
    scene = doc.get("scene", {})
    chosen = doc.get("chosen_plan", {})
    lines = [f"# Mission plan — {doc.get('job')}",
             f"_measured scene: extent {scene.get('extent_w_m')}×"
             f"{scene.get('extent_h_m')} m, blind share {scene.get('blind_share')}, "
             f"mean confidence {scene.get('mean_confidence')}_", ""]
    base = doc.get("baseline", {})
    lines += ["## Baseline simulation",
              f"- plan: alt {base.get('plan', {}).get('altitude_m')} m @ "
              f"{base.get('plan', {}).get('speed_m_s')} m/s, pattern "
              f"{base.get('plan', {}).get('pattern')}",
              f"- GSD {base.get('metrics', {}).get('gsd')} cm · "
              f"{base.get('metrics', {}).get('frames')} frames · "
              f"duration ≈ {base.get('metrics', {}).get('duration_min')} min",
              "- _all values are simulation estimates_", ""]
    lines += ["## What-if deltas vs baseline"]
    for d in doc.get("what_if", []):
        deltas = d.get("compare", {}).get("deltas", {})
        q = deltas.get("quality_index", {}).get("delta_pct")
        bat = d.get("battery_pct")
        dur = d.get("duration_pct")
        g = deltas.get("gsd", {}).get("delta_pct")
        lines.append(f"- **{d.get('tag')}**: quality {q if q is not None else 'n/a'}%, "
                     f"GSD {g if g is not None else 'n/a'}%, battery "
                     f"{bat if bat is not None else 'n/a'}%, time "
                     f"{dur if dur is not None else 'n/a'}%")
    lines += ["", "## Recommended plan",
              f"- **{chosen.get('label')}**: {json.dumps(chosen.get('plan'))}",
              f"- metrics: {json.dumps(chosen.get('metrics', {}))}"]
    lines += ["", "## Recommendations"]
    for r in doc.get("recommendations", []):
        lines.append(f"- [{r.get('source')}] {r.get('recommendation')} — "
                     f"{r.get('reason')} (confidence {r.get('confidence')})")
    return "\n".join(lines) + "\n"


def write_plan_exports(doc: dict, intel_dir: Path) -> dict:
    """All plan exports; returns artifact metadata."""
    export_plan_geometry(doc, intel_dir)
    md = render_plan_markdown(doc)
    (intel_dir / "mission_plan.md").write_text(md)
    return {"geojson": str(intel_dir / "mission_plan.geojson"),
            "kml": str(intel_dir / "mission_plan.kml"),
            "md": str(intel_dir / "mission_plan.md")}
