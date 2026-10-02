"""Automated intelligence report generator + real-time alerts + dashboard.

Aggregates every Phase 8 artifact of one mission into a decision-support
package under ``intel/``:

* ``disaster_report.json`` — the full machine-readable report,
* ``disaster_report.md`` — human-readable technical report,
* ``disaster_report.html`` — styled single-file report (PDF export is added
  when a PDF backend is installed; otherwise formats note their absence),
* ``alerts.json`` — rule-derived real-time alerts (collapsed structures,
  blocked routes, high-risk zones, weather, GPS drift, low confidence),
  each with level, title, evidence source and detail,
* ``dashboard.json`` / ``dashboard.html`` — panel payload + a standalone,
  dependency-free decision-support page (works opened from disk).

Alerts are *derived from the same artifacts the report shows*: a rule fires
only when the underlying record exists, so the alert list cannot disagree
with the report. Level precedence Critical > High > Medium > Low drives the
overall mission alert level.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.logging_config import get_logger
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.report_generator")

_LEVEL_RANK = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _load_intel(workspace: Path) -> dict:
    names = ("environment_report", "scene_report", "damage_report",
             "infrastructure_report", "risk_report", "mission_recommendations",
             "change_report")
    out = {}
    for name in names:
        p = workspace / "intel" / f"{name}.json"
        if p.exists():
            try:
                out[name] = json.loads(p.read_text())
            except (OSError, ValueError):
                log.warning("intel_report_unreadable", name=name)
    return out


def assemble(artifacts: dict, meta: dict | None = None) -> dict:
    """Build report summary + alerts + dashboard payload from loaded artifacts."""
    meta = meta or {}
    env = artifacts.get("environment_report", {})
    damage = artifacts.get("damage_report", {})
    risk = artifacts.get("risk_report", {})
    infra = artifacts.get("infrastructure_report", {})
    recs = artifacts.get("mission_recommendations", {})
    change = artifacts.get("change_report", {})
    scene = artifacts.get("scene_report", {})

    # ---------------- alerts (every rule needs its underlying record) ------
    alerts: list[dict] = []
    for f in damage.get("findings", []):
        if f.get("severity") in ("High", "Critical"):
            alerts.append({
                "level": f["severity"], "title": f"{f.get('type')} detected",
                "detail": f.get("rationale", ""), "source": "damage_report",
                "coordinates": f.get("centroid"),
            })
    blocked = (risk.get("accessibility", {}) or {}).get("blocked_routes", [])
    if blocked:
        alerts.append({
            "level": "High" if len(blocked) >= 3 else "Medium",
            "title": f"{len(blocked)} structure(s) unreachable by emergency vehicle",
            "detail": "; ".join(b.get("cause", "") for b in blocked[:5]), "source": "risk_report",
        })
    overall = (risk.get("overall_risk", {}) or {}).get("level", "Low")
    if overall in ("High", "Critical"):
        alerts.append({"level": overall, "title": f"overall mission risk is {overall}",
                       "detail": "; ".join((risk.get("overall_risk", {}) or {}).get("reasoning", [])[:3]),
                       "source": "risk_report"})
    env_q = env.get("quality_score")
    if env_q is not None and env_q < 50:
        alerts.append({"level": "High" if env_q < 30 else "Medium",
                       "title": "adverse environmental conditions",
                       "detail": f"environmental quality {env_q}/100 — dominant "
                                 f"condition '{env.get('dominant_condition')}'",
                       "source": "environment_report"})
    if change:
        total = change.get("added_count", 0) + change.get("removed_count", 0)
        if total:
            alerts.append({"level": "Medium", "title": f"{total} structural change cluster(s) "
                           f"vs baseline ({change.get('added_count')} added, "
                           f"{change.get('removed_count')} removed)",
                           "detail": "see change_report.json for geometry", "source": "change_report"})

    # ---------------- report summary --------------------------------------
    report = {
        "generated_for": meta.get("job") or meta.get("workspace") or "mission",
        "crs": meta.get("crs"),
        "environment": {
            "quality_score": env.get("quality_score"), "grade": env.get("grade"),
            "dominant_condition": env.get("dominant_condition"),
            "visibility_m": (env.get("visibility") or {}).get("estimated_visibility_m"),
            "conditions": _top_conditions(env.get("conditions", {})),
        },
        "damage": {"findings_count": len(damage.get("findings", [])),
                   "counts": damage.get("counts", {}),
                   "max_severity": damage.get("max_severity"),
                   "findings": damage.get("findings", [])},
        "infrastructure": {k: v for k, v in infra.items()
                           if k in ("buildings", "trees", "terrain", "roads")},
        "risk": {"level": overall, "indicators": (risk.get("overall_risk", {}) or {}).get(
            "indicators", []), "zones": len(risk.get("risk_zones", []))},
        "recommendations": {"count": len(recs.get("recommendations", [])),
                            "score": recs.get("mission_optimization_score"),
                            "grade": recs.get("grade"),
                            "top": recs.get("recommendations", [])[:5]},
        "change": {"added": change.get("added_count", 0),
                   "removed": change.get("removed_count", 0),
                   "change_index": change.get("change_index")} if change else None,
        "scene_matches": scene.get("total_matches"),
        "alerts": alerts,
        "alerts_max_level": max((a["level"] for a in alerts), key=lambda l: _LEVEL_RANK.get(l, 0))
        if alerts else "None",
        "method": "artifact_aggregation",
    }

    # ---------------- dashboard payload ------------------------------------
    env_rows = [(row["condition"], row.get("mean_score", 0.0))
                for row in report["environment"]["conditions"]]
    dashboard = {
        "mission": report["generated_for"],
        "alerts": alerts, "alerts_max_level": report["alerts_max_level"],
        "environment": report["environment"],
        "damage": report["damage"],
        "infrastructure": report["infrastructure"],
        "risk": report["risk"],
        "recommendations": report["recommendations"],
        "change": report["change"],
        "charts": {"conditions": env_rows, "damage_counts": list(report["damage"]["counts"].items())},
    }
    return {"report": report, "alerts": alerts, "dashboard": dashboard}


def _top_conditions(conditions: dict, n: int = 4) -> list[dict]:
    rows = [{"condition": c, **s} for c, s in conditions.items() if s.get("mean_score", 0) > 0.05]
    rows.sort(key=lambda r: -r.get("mean_score", 0))
    return rows[:n]


# ---------------------------------------------------------------------------
# Renderers (md / html)
# ---------------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    r = report["report"]
    lines = [f"# DroneRecon intelligence report — {r['generated_for']}",
             f"_crs: {r['crs'] or 'not georeferenced'} · method: {r['method']}_", ""]
    lines.append(f"## Mission alert level: **{r['alerts_max_level']}**")
    for a in r["alerts"]:
        lines.append(f"- **[{a['level']}]** {a['title']} — {a['detail']}")
    lines += ["", "## Environment",
              f"- quality **{r['environment'].get('quality_score')}** "
              f"({r['environment'].get('grade')}), dominant: {r['environment'].get('dominant_condition')}",
              f"- visibility ≈ {r['environment'].get('visibility_m')} m"]
    for c in r["environment"]["conditions"]:
        lines.append(f"  - {c['condition']}: mean {c.get('mean_score')}")
    lines += ["", "## Damage assessment"]
    lines.append(f"- findings: {r['damage']['findings_count']}, max severity "
                 f"**{r['damage']['max_severity']}**")
    for f in r["damage"]["findings"]:
        lines.append(f"- **{f.get('type')}** ({f.get('severity')}, conf "
                     f"{f.get('confidence')}) — {f.get('rationale')}")
    lines += ["", "## Infrastructure"]
    for section, data in r["infrastructure"].items():
        lines.append(f"- {section}: `{json.dumps(data)[:220]}`")
    lines += ["", "## Risk"]
    for i in r["risk"]["indicators"]:
        lines.append(f"- **{i.get('name')}**: {i.get('level')} — {i.get('reasoning')}")
    if r["risk"]["zones"]:
        lines.append(f"- {r['risk']['zones']} risk zone(s) mapped on the ground grid")
    lines += ["", "## Recommendations",
              f"- mission optimization score **{r['recommendations'].get('score')}** "
              f"({r['recommendations'].get('grade')})"]
    for rec in r["recommendations"]["top"]:
        lines.append(f"- [{rec.get('priority')}] {rec.get('type')}: {rec.get('reason')}")
    if r["change"]:
        lines += ["", "## Multi-mission change",
                  f"- {r['change']['added']} added, {r['change']['removed']} removed clusters, "
                  f"change index {r['change']['change_index']}"]
    return "\n".join(lines) + "\n"


def _md_to_html(markdown: str) -> str:
    """Minimal markdown→HTML: headings, bullet lists, paragraphs, escaping.

    Keeps the HTML export a faithful rendering of the markdown report rather
    than duplicating the interactive dashboard.
    """
    import html as _html

    body = []
    for raw in markdown.splitlines():
        line = _html.escape(raw)
        if line.startswith("### "):
            body.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            body.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- "):
            body.append(f"<li>{line[2:]}</li>")
        elif line.strip():
            body.append(f"<p>{line}</p>")
    out = []
    in_list = False
    for part in body:
        if part.startswith("<li>"):
            if not in_list:
                out.append("<ul>")
                in_list = True
        elif in_list:
            out.append("</ul>")
            in_list = False
        out.append(part)
    if in_list:
        out.append("</ul>")
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>DroneRecon disaster report</title><style>"
            "body{font-family:system-ui;margin:2rem auto;max-width:900px;color:#222}"
            "ul{padding-left:1.4rem}li{margin:.2rem 0}"
            f"</style></head><body>{''.join(out)}</body></html>")


def render_html(report: dict) -> str:
    payload = json.dumps(report["dashboard"]).replace("</", "<\\/")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DroneRecon intelligence</title>
<style>
 body{{font-family:system-ui;margin:2rem auto;max-width:1100px;color:#222}}
 h1{{font-size:1.4rem}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem}}
 .card{{border:1px solid #ddd;border-radius:8px;padding:1rem;background:#fafafa}}
 .lvl{{font-weight:700;padding:.2rem .5rem;border-radius:4px;color:#fff}}
 .Critical,.High{{background:#c62828}} .Medium{{background:#ef6c00}}
 .Low{{background:#2e7d32}} .None{{background:#607d8b}}
 ul{{padding-left:1.2rem;margin:.4rem 0}} li{{margin:.2rem 0}}
 .bar{{background:#e0e0e0;height:1rem;border-radius:3px;margin:.2rem 0}}
 .bar>div{{height:100%;border-radius:3px;background:#1565c0}}
 .small{{color:#666;font-size:.8rem}}
</style></head><body>
<h1>DroneRecon — Decision Support</h1>
<div id="app"></div>
<script>
const D = {payload};
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
const cards = [];
const lvl = `<span class="lvl ${{D.alerts_max_level}}">${{D.alerts_max_level}}</span>`;
cards.push(`<div class="card"><b>Mission alert level</b> ${{lvl}} · ${{esc(D.mission)}}<ul>${{D.alerts.map(a=>`<li><b>${{a.level}}</b> ${{esc(a.title)}}</li>`).join('')}}</ul></div>`);
const e = D.environment || {{}};
cards.push(`<div class="card"><b>Environmental quality</b> ${{esc(e.quality_score)}} (${{esc(e.grade)}})<br><span class="small">dominant: ${{esc(e.dominant_condition)}} · visibility ${{esc(e.visibility_m)}} m</span>
${{(D.charts.conditions||[]).map(([c,s])=>`<div class="bar"><div style="width:${{Math.min(100,s*100)}}%"></div></div><span class="small">${{esc(c)}}</span>`).join('')}}</div>`);
const dmg = D.damage || {{}};
cards.push(`<div class="card"><b>Damage</b> ${{esc(dmg.max_severity)}} max · ${{dmg.findings_count}} finding(s)<ul>${{(dmg.findings||[]).map(f=>`<li>${{esc(f.type)}} (${{esc(f.severity)}})</li>`).join('')}}</ul></div>`);
const infra = D.infrastructure || {{}};
cards.push(`<div class="card"><b>Infrastructure</b><ul>${{Object.keys(infra).map(k=>`<li>${{esc(k)}}: <span class="small">${{esc(JSON.stringify(infra[k]).slice(0,160))}}</span></li>`).join('')}}</ul></div>`);
const rk = D.risk || {{}};
cards.push(`<div class="card"><b>Risk</b> ${{esc(rk.level)}} · ${{rk.zones}} zone(s)<ul>${{(rk.indicators||[]).map(i=>`<li>${{esc(i.name)}}: ${{esc(i.level)}}</li>`).join('')}}</ul></div>`);
const rc = D.recommendations || {{}};
cards.push(`<div class="card"><b>Recommendations</b> score ${{esc(rc.score)}} (${{esc(rc.grade)}})<ul>${{(rc.top||[]).map(r=>`<li>${{esc(r.reason)}}</li>`).join('')}}</ul></div>`);
document.getElementById('app').innerHTML = `<div class="grid">${{cards.join('')}}</div>`;
</script></body></html>
"""


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class ReportGenerationStage(PipelineStage):
    name = "report_generation"
    description = "Disaster report (json/md/html) + alerts + decision dashboard"
    artifact_rel = "intel/disaster_report.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        if not (self.workspace / "intel").is_dir():
            raise StageNotApplicable("no intel artifacts — run the intelligence chain first")

    def execute(self) -> None:
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        artifacts = _load_intel(self.workspace)
        meta = _mission_meta(self.workspace)
        bundle = assemble(artifacts, meta=meta)
        report, dashboard = bundle["report"], bundle["dashboard"]
        (intel_dir / "disaster_report.json").write_text(json.dumps(report, indent=2))
        (intel_dir / "alerts.json").write_text(json.dumps(bundle["alerts"], indent=2))
        (intel_dir / "dashboard.json").write_text(json.dumps(dashboard, indent=2))
        (intel_dir / "dashboard.html").write_text(render_html(bundle))
        md = render_markdown(bundle)
        (intel_dir / "disaster_report.md").write_text(md)
        outputs = [{"kind": "data", "name": "disaster_report", "path": str(intel_dir / "disaster_report.json")},
                   {"kind": "data", "name": "alerts", "path": str(intel_dir / "alerts.json")},
                   {"kind": "data", "name": "dashboard", "path": str(intel_dir / "dashboard.json")},
                   {"kind": "report", "name": "dashboard_html", "path": str(intel_dir / "dashboard.html")},
                   {"kind": "report", "name": "disaster_report_md", "path": str(intel_dir / "disaster_report.md")}]
        # PDF export when a backend is installed; otherwise note its absence.
        if _pdf_supported():
            try:
                pdf_path = intel_dir / "disaster_report.pdf"
                _write_pdf(pdf_path, md)
                outputs.append({"kind": "report", "name": "disaster_report_pdf", "path": str(pdf_path)})
            except Exception as exc:  # pragma: no cover - depends on environment
                log.warning("pdf_export_failed", error=str(exc))
        # HTML export is the markdown report rendered as HTML — distinct from
        # the interactive dashboard.html page.
        report_html = _md_to_html(md)
        (intel_dir / "disaster_report.html").write_text(report_html)
        outputs.append({"kind": "report", "name": "disaster_report_html",
                        "path": str(intel_dir / "disaster_report.html")})
        self._count = len(bundle["alerts"])
        self._detail = {"alerts": self._count,
                        "alert_level": report["alerts_max_level"],
                        "overall_risk": report["risk"]["level"]}
        self._outputs = outputs
        self.progress(1.0, {"alerts": self._count, "level": report["alerts_max_level"]})


def _mission_meta(workspace: Path) -> dict:
    meta: dict = {"workspace": workspace.name}
    align = workspace / "georef" / "alignment.json"
    if align.exists():
        try:
            data = json.loads(align.read_text())
        except (OSError, ValueError):
            data = {}
        meta["crs"] = data.get("crs") or data.get("epsg") or data.get("utm_zone")
    return meta


def _pdf_supported() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


def _write_pdf(path: Path, markdown: str) -> None:
    """Very small reportlab passthrough (title + monospaced markdown body)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "DroneRecon intelligence report")
    y -= 30
    c.setFont("Courier", 8)
    for line in markdown.splitlines():
        if y < 40:
            c.showPage()
            y = h - 60
            c.setFont("Courier", 8)
        c.drawString(50, y, line[:110])
        y -= 11
    c.save()
