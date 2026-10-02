"""Tests for Phase 8 — geospatial intelligence: environmental detection,
damage assessment, risk, mission recommendations, RAG answers, reports/alerts,
the intelligence stage chain and its REST API."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.models import Project
from app.services.mesh import TriangleMesh
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.pipeline_stage import INTEL_CHAIN, MESH_CHAIN
from tests.test_digital_twin import _grid_mesh, _seed_dense_workspace

_CLASSES = list(settings.semantic.classes)


def _cid(name: str) -> int:
    assert name in _CLASSES, name
    return _CLASSES.index(name)


# ---------------------------------------------------------------------------
# Environmental intelligence (real image statistics)
# ---------------------------------------------------------------------------


class TestEnvironmentDetector:
    def _img(self, value: int = 128) -> np.ndarray:
        return np.full((48, 48, 3), value, dtype=np.uint8)

    def test_fog_needs_veil_not_just_low_contrast(self):
        from app.services.environment_detector import detect_fog_haze

        clear = detect_fog_haze(self._img(150))
        veiled = detect_fog_haze(np.full((48, 48, 3), 205, dtype=np.uint8))  # near-white haze
        assert veiled["score"] > clear["score"] + 0.3
        assert veiled["evidence"]["dark_channel"] > clear["evidence"]["dark_channel"]

    def test_lowlight_and_blur_detectors(self):
        from app.services.environment_detector import detect_lowlight, detect_motion_blur

        dark = detect_lowlight(self._img(20))
        assert dark["score"] > 0.5
        assert detect_lowlight(self._img(200))["score"] < 0.3

        sharp = np.random.default_rng(1).integers(0, 255, (48, 48, 3)).astype(np.uint8)
        soft = cv2.GaussianBlur(sharp, (15, 15), 6)
        assert detect_motion_blur(soft)["score"] > detect_motion_blur(sharp)["score"] + 0.2

    def test_sun_glare_saturated_region(self):
        from app.services.environment_detector import detect_sun_glare

        glare = self._img(120)
        glare[8:20, 8:30] = 250  # compact saturated bloom
        res = detect_sun_glare(glare)
        assert res["score"] > 0.3 and res["evidence"]["hot_fraction"] > 0

    def test_aggregate_score_and_strategies(self):
        from app.services.environment_detector import (
            adaptive_strategies,
            analyze_frame,
            score_environment,
        )

        results = [analyze_frame(np.full((48, 48, 3), 205, dtype=np.uint8)) for _ in range(4)]
        env = score_environment(results)
        assert env["quality_score"] < 70  # persistent haze degrades quality
        assert "fog_haze" in env["conditions"]
        strategies = adaptive_strategies(env)
        assert any(s["condition"] == "fog_haze" for s in strategies)
        assert all("params" in s and "rationale" in s for s in strategies)


# ---------------------------------------------------------------------------
# Damage assessment (geometric evidence rules)
# ---------------------------------------------------------------------------


def _labelled_mesh(bottom_labels: str, top_labels: str | None = None,
                   top_patches: bool = False) -> tuple[TriangleMesh, np.ndarray]:
    """Ground plane (+ optional elevated top plane, optionally broken into
    3x3-cell patches separated by 1-cell gaps — the debris signature)."""
    g = 20
    base = _grid_mesh(g)
    n_base = base.n
    verts = list(map(list, base.vertices))
    faces = list(map(list, base.faces))
    rgb = list(map(list, base.colors))
    if top_labels is not None:
        base_i = len(verts)
        top = base.vertices + np.array([0.0, 0.0, 1.0])
        verts += list(map(list, top))
        rgb += list(map(list, base.colors))
        top_faces = np.asarray(base.faces) + base_i
        if top_patches:
            tc = base.face_centroids()
            keep = (tc[:, 0] % 4 < 3) & (tc[:, 1] % 4 < 3)
            top_faces = top_faces[keep]
        faces += list(map(list, top_faces))
    mesh = TriangleMesh(vertices=np.asarray(verts, dtype=float),
                        faces=np.asarray(faces, dtype=np.int64),
                        colors=np.asarray(rgb, dtype=np.uint8))
    labels = np.full(mesh.m, _cid(bottom_labels), dtype=np.int64)
    if top_labels is not None:
        labels[n_base:] = _cid(top_labels)
    return mesh, labels


class TestDamageAssessment:
    def test_flood_from_water_surface(self):
        from app.services.damage_detector import assess_flood

        mesh, labels = _labelled_mesh("ground")
        # water band along the left edge (a connected region > flood_min_area)
        cents = mesh.face_centroids()
        water = cents[:, 0] < 4.0
        labels[water] = _cid("water")
        areas = mesh.face_areas()
        flood = assess_flood(mesh, labels, _CLASSES, cents, areas)
        assert flood is not None and flood["type"] == "flood"
        assert flood["affected_area_m2"] > 15
        assert flood["severity"] in ("Medium", "High", "Critical")

    def test_debris_fragmented_elevated_blocks(self):
        from app.services.damage_detector import assess_debris

        mesh, labels = _labelled_mesh("ground", top_labels="structure", top_patches=True)
        # debris needs 4+ m² fragments: top plane now floats at z=1 disconnected
        # from the ground plane only where they do not touch.
        cents, areas = mesh.face_centroids(), mesh.face_areas()
        gn = np.array([0.0, 0.0, 1.0])
        debris = assess_debris(mesh, labels, _CLASSES, cents, areas, gn, 0.0)
        assert debris is not None and debris["type"] == "debris_field"
        assert debris["fragment_count"] >= 12  # extensive fragmentation → High
        assert debris["severity"] == "High"

    def test_collapse_needs_debris_over_structure(self):
        from app.services.damage_detector import assess_collapse

        debris = {"type": "debris_field", "severity": "High", "confidence": 0.7,
                  "affected_area_m2": 40.0, "scene_share": 0.1,
                  "centroid": [5.0, 5.0, 0.6], "rationale": "fragmentation"}
        twin = {"objects": [{"uuid": "s1", "class": "roof", "confidence": 0.9,
                             "bbox_min": [0.0, 0.0, 0.0], "bbox_max": [12.0, 12.0, 2.0]}]}
        hit = assess_collapse(debris, twin)
        assert hit is not None and hit["type"] == "collapse_candidate"
        assert hit["confidence"] > debris["confidence"]
        assert "baseline" in hit["note"]
        far = dict(twin)
        far["objects"] = [dict(twin["objects"][0], bbox_min=[50.0, 50.0, 0],
                               bbox_max=[60.0, 60.0, 2.0])]
        assert assess_collapse(debris, far) is None


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------


def _structure(uuid: str = "s1", at=(10.0, 10.0)) -> dict:
    return {"uuid": uuid, "class": "roof", "confidence": 0.9,
            "bbox_min": [at[0] - 1, at[1] - 1, 0.0], "bbox_max": [at[0] + 1, at[1] + 1, 2.0],
            "centroid": [at[0], at[1], 1.0]}


class TestRiskAssessment:
    def test_clean_scene_low_risk(self):
        from app.services.risk_assessment import assess_risk

        mesh = _grid_mesh(24)
        report = assess_risk(mesh, {"findings": []}, {"objects": [_structure()]})
        assert report["overall_risk"]["level"] == "Low"
        assert report["risk_zones"] == []
        assert report["accessibility"]["grade"] == "Good"

    def test_flood_zone_blocks_structure_and_raises_level(self):
        from app.services.risk_assessment import assess_risk

        mesh = _grid_mesh(24)
        damage = {"findings": [{"type": "flood", "severity": "High",
                                "centroid": [10.0, 10.0, 0.0], "affected_area_m2": 900.0}]}
        report = assess_risk(mesh, damage, {"objects": [_structure()]})
        assert report["flood_risk"] is not None
        assert report["flood_risk"]["level"] in ("High", "Critical")
        assert report["risk_zones"] and report["risk_zones"][0]["cause"] == "flood"
        blocked = report["accessibility"]["blocked_routes"]
        assert len(blocked) == 1 and blocked[0]["cause"] == "flooded_access"
        assert report["overall_risk"]["level"] in ("High", "Critical")
        assert report["overall_risk"]["reasoning"]  # explainable

    def test_steep_terrain_landslide_indicator(self):
        from app.services.risk_assessment import assess_risk

        g = 24
        mesh = _grid_mesh(g)
        verts = mesh.vertices.copy()
        # A steep ramp rising along x (occupancy on the full plane).
        verts[:, 2] = verts[:, 0] * 0.6  # tan(31°) ≈ 0.6 → above the 25° slope gate
        steep = TriangleMesh(vertices=verts, faces=mesh.faces,
                             colors=mesh.colors, confidence=mesh.confidence)
        report = assess_risk(steep, {"findings": []}, {"objects": []})
        terrain = report["terrain"]
        assert terrain["mean_slope_deg"] > settings.intel.risk_slope_deg
        levels = [i["name"] for i in report["overall_risk"]["indicators"]]
        assert "landslide_risk" in levels


# ---------------------------------------------------------------------------
# Mission recommender
# ---------------------------------------------------------------------------


class TestMissionRecommender:
    def test_no_signals_scores_zero_and_asks_for_gps(self):
        from app.services.mission_recommender import recommend_mission

        out = recommend_mission(_grid_mesh(12), env=None, gps=None, conf=None, twin=None)
        assert out["mission_optimization_score"] == 0.0
        kinds = {r["type"] for r in out["recommendations"]}
        assert "gps" in kinds

    def test_healthy_signals_score_high(self):
        from app.services.mission_recommender import recommend_mission

        out = recommend_mission(
            _grid_mesh(12),
            env={"quality_score": 90.0, "dominant_condition": "clear",
                 "adaptive_strategies": []},
            gps={"gps_quality": {"gps_score": 85.0}},
            conf={"low_share": 0.02, "weak_regions": []},
            twin={"objects": [_structure()]},
        )
        assert out["mission_optimization_score"] >= 80
        assert out["grade"] == "Excellent"
        assert {"gps", "environment", "coverage"} == set(out["components"])
        assert any(r["type"] == "coverage" for r in out["recommendations"])

    def test_weak_cluster_suggests_refly_region(self):
        from app.services.mission_recommender import recommend_mission

        out = recommend_mission(
            _grid_mesh(12), env=None, gps={"gps_quality": {"gps_score": 90.0}},
            conf={"low_share": 0.4,
                  "weak_regions": [{"centroid": [3.0, 3.0, 0.0], "radius_m": 2.0,
                                    "vertices": 500}]},
            twin=None)
        refly = [r for r in out["recommendations"] if r["type"] == "refly_region"]
        assert len(refly) == 1
        assert refly[0]["region"]["centroid"] == [3.0, 3.0, 0.0]

    def test_measured_signals_retained_when_others_missing(self):
        """An absent signal must not discard a measured one (regression: the
        score dropped GPS when no env report existed, and coverage when no GPS
        report existed)."""
        from app.services.mission_recommender import recommend_mission

        mesh = _grid_mesh(12)
        # coverage + environment measured, GPS absent → both kept
        out = recommend_mission(
            mesh, env={"quality_score": 90.0, "dominant_condition": "clear",
                       "adaptive_strategies": []}, gps=None,
            conf={"low_share": 0.05, "weak_regions": []}, twin=None)
        assert set(out["components"]) == {"environment", "coverage"}
        assert 0 < out["mission_optimization_score"] < 100
        # GPS measured, environment absent → GPS kept
        out = recommend_mission(mesh, env=None,
                                gps={"gps_quality": {"gps_score": 85.0}},
                                conf=None, twin=None)
        assert set(out["components"]) == {"gps"}
        assert out["mission_optimization_score"] == 85.0
        # env report without a quality score is not treated as a zero
        out = recommend_mission(mesh, env={"adaptive_strategies": []}, gps=None,
                                conf=None, twin=None)
        assert "environment" not in out["components"]


# ---------------------------------------------------------------------------
# RAG engine (grounded answers)
# ---------------------------------------------------------------------------


class TestRagEngine:
    def _workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "inteljob"
        (ws / "twin").mkdir(parents=True)
        (ws / "intel").mkdir(parents=True)
        (ws / "twin" / "twin.json").write_text(json.dumps({"objects": [
            {"uuid": "b1", "class": "roof", "height_m": 3.0, "surface_area_m2": 40.0,
             "volume_m3": None, "confidence": 0.92, "centroid": [2.0, 2.0, 2.5]},
            {"uuid": "b2", "class": "wall", "height_m": 3.0, "surface_area_m2": 20.0,
             "volume_m3": None, "confidence": 0.6, "centroid": [2.0, 2.0, 1.5]},
            {"uuid": "r1", "class": "ground", "height_m": 0.0, "surface_area_m2": 500.0,
             "volume_m3": None, "confidence": 0.9, "centroid": [2.0, 2.0, 0.0]},
        ]}))
        (ws / "intel" / "damage_report.json").write_text(json.dumps({"findings": [
            {"type": "collapse_candidate", "severity": "High", "confidence": 0.8,
             "rationale": "debris overlaps a roof footprint"}],
            "counts": {"High": 1}}))
        return ws

    def _index(self, tmp_path):
        from app.services.rag_engine import build_index

        return build_index(self._workspace(tmp_path))

    def test_count_and_locate_grounded(self, tmp_path):
        from app.services.rag_engine import answer_question

        index = self._index(tmp_path)
        assert index["record_count"] >= 4
        out = answer_question(index, "how many buildings")
        assert out["intent"] == "count"
        assert out["results"][0]["count"] == 2  # roof + wall objects
        out = answer_question(index, "where is the roof")
        assert out["intent"] == "locate" and len(out["results"]) == 1
        assert out["results"][0]["uuid"] == "b1"

    def test_damage_and_no_hallucination(self, tmp_path):
        from app.services.rag_engine import answer_question

        index = self._index(tmp_path)
        out = answer_question(index, "how many damaged buildings")
        assert out["intent"] == "count_damage" and out["results"]
        out = answer_question(index, "predict tomorrow's weather")
        assert out["intent"] in ("no_match", "retrieval")
        assert out["grounded"] is True


# ---------------------------------------------------------------------------
# Report generator + alerts
# ---------------------------------------------------------------------------


class TestReportGenerator:
    def _artifacts(self):
        return {
            "environment_report": {"quality_score": 40.0, "grade": "Poor",
                                   "dominant_condition": "fog_haze",
                                   "visibility": {"estimated_visibility_m": 800.0},
                                   "conditions": {"fog_haze": {"mean_score": 0.7}}},
            "damage_report": {"findings": [{"type": "flood", "severity": "High",
                                            "rationale": "large water surface",
                                            "centroid": [1.0, 1.0, 0.0],
                                            "affected_area_m2": 900.0}],
                              "counts": {"High": 1}, "max_severity": "High"},
            "risk_report": {"overall_risk": {"level": "High", "indicators": [
                {"name": "flood_risk", "level": "High", "reasoning": "flood ≈ 900 m²"}]},
                "accessibility": {"blocked_routes": [{"uuid": "s1", "class": "roof",
                                                      "cause": "flooded_access"}]},
                "risk_zones": [{"cause": "flood", "area_m2": 100.0}]},
            "infrastructure_report": {"buildings": {"count": 2, "height_m": {"max": 3.0}}},
            "mission_recommendations": {"recommendations": [
                {"type": "refly_region", "priority": "High", "reason": "low confidence at X"}],
                "mission_optimization_score": 70.0, "grade": "Good"},
        }

    def test_alerts_follow_evidence(self):
        from app.services.report_generator import assemble

        bundle = assemble(self._artifacts())
        alerts = bundle["alerts"]
        titles = [a["title"] for a in alerts]
        assert any("flood" in t for t in titles)
        assert any("unreachable" in t for t in titles)
        assert any("adverse" in t for t in titles)
        assert bundle["report"]["alerts_max_level"] == "High"

    def test_markdown_and_html_render(self):
        from app.services.report_generator import assemble, render_html, render_markdown

        bundle = assemble(self._artifacts())
        md = render_markdown(bundle)
        assert "# DroneRecon" in md and "## Damage assessment" in md
        html = render_html(bundle)
        assert "Decision Support" in html and "damage" in html.lower()
        assert "</script>" in html and "<script>" in html  # self-contained page

    def test_report_html_is_not_the_dashboard(self):
        """disaster_report.html renders the markdown report; the interactive
        dashboard is a separate artifact (regression: both were identical)."""
        from app.services.report_generator import _md_to_html, assemble, render_markdown

        bundle = assemble(self._artifacts())
        report_html = _md_to_html(render_markdown(bundle))
        assert "Decision Support" not in report_html  # that's the dashboard page
        assert "Mission alert level" in report_html
        assert "Damage assessment" in report_html
        assert "<h1>" in report_html and "<ul>" in report_html


# ---------------------------------------------------------------------------
# Intelligence chain over a seeded digital twin
# ---------------------------------------------------------------------------


class TestIntelChain:
    def test_full_intel_chain(self, tmp_path: Path):
        job_id = "inteljob0001"
        workspace = _seed_dense_workspace(tmp_path, job_id)
        # first produce the digital twin the intel stages consume
        twin = run_autonomous_pipeline(job_id, PipelineRequest(plugins=list(MESH_CHAIN)))
        assert twin["status"] == "completed", twin.get("error")
        report = run_autonomous_pipeline(job_id, PipelineRequest(plugins=list(INTEL_CHAIN)))
        assert report["status"] == "completed", report.get("error")
        stages = report["stages"]
        for name in ("environmental_intelligence", "scene_intelligence",
                     "damage_assessment", "infrastructure_analysis", "risk_assessment",
                     "mission_recommendation", "rag_index", "report_generation"):
            assert stages[name]["status"] == "completed", (name, stages[name])
        # change detection needs a baseline job → clean skip, not a failure
        assert stages["change_detection"]["status"] == "skipped"
        for rel in ("intel/environment_report.json", "intel/damage_report.json",
                    "intel/risk_report.json", "intel/disaster_report.json",
                    "intel/alerts.json", "intel/rag_index.json",
                    "intel/dashboard.html", "intel/dashboard.json",
                    "intel/disaster_report.md"):
            assert (workspace / rel).exists(), rel

        # rerun: everything resumes from artifacts
        again = run_autonomous_pipeline(job_id, PipelineRequest(plugins=list(INTEL_CHAIN)))
        assert again["status"] == "completed"
        for name in INTEL_CHAIN:
            assert again["stages"][name]["status"] in ("skipped", "completed"), name


# ---------------------------------------------------------------------------
# Multi-mission change detection (ENU diff) + downstream rebuild
# ---------------------------------------------------------------------------


def _enu_cloud(ws: Path, x_ranges: list[tuple[float, float]]) -> None:
    from app.services.pointcloud import PointCloud, save_ply

    pts = [[x, y, 0.0] for x0, x1 in x_ranges for x in np.arange(x0, x1, 0.4)
           for y in np.arange(0, 4, 0.4)]
    d = ws / "georef"
    d.mkdir(parents=True, exist_ok=True)
    save_ply(d / "dense_model_enu.ply",
             PointCloud(xyz=np.asarray(pts, dtype=float), confidence=np.ones(len(pts))))


class TestChangeIntegration:
    def test_change_report_feeds_rag_and_report(self, tmp_path: Path):
        """Baseline vs current ENU clouds → change clusters, then the forced
        change→rag→report run keeps reports consistent with the new data."""
        settings.storage.base_path = str(tmp_path)
        job = "chgcur"
        workspace = _seed_dense_workspace(tmp_path, job)
        base_ws = settings.storage.project_dir("chgbase")
        # baseline: shared region + an extra west patch; current: shared +
        # a new east patch → 1 added + 1 removed cluster each ≥ 1 m³
        _enu_cloud(base_ws, [(0, 6), (-2, -1)])
        _enu_cloud(workspace, [(0, 6), (8, 10)])
        (workspace / "intel").mkdir(parents=True, exist_ok=True)
        (workspace / "intel" / "change_config.json").write_text(json.dumps(
            {"baseline_job": "chgbase", "voxel_m": 0.5}))

        report = run_autonomous_pipeline(
            job, PipelineRequest(plugins=["change_detection", "rag_index",
                                          "report_generation"],
                                 force=["change_detection", "rag_index",
                                        "report_generation"]))
        assert report["status"] == "completed", report.get("error")
        chg = json.loads((workspace / "intel" / "change_report.json").read_text())
        assert chg["added_count"] == 1 and chg["removed_count"] == 1
        assert (workspace / "intel" / "change_delta.ply").exists()
        # report and RAG index now carry the change data
        art = json.loads((workspace / "intel" / "disaster_report.json").read_text())
        assert art["change"]["added"] == 1 and art["change"]["removed"] == 1
        index = json.loads((workspace / "intel" / "rag_index.json").read_text())
        assert any(r["kind"] == "change" for r in index["records"])

    def test_change_without_enu_is_clean_skip(self, tmp_path: Path):
        """No georeferenced dense model → stage skips, pipeline completes."""
        settings.storage.base_path = str(tmp_path)
        job = "chgskp"
        workspace = _seed_dense_workspace(tmp_path, job)
        settings.storage.project_dir("nobase").mkdir(parents=True, exist_ok=True)
        (workspace / "intel").mkdir(parents=True, exist_ok=True)
        (workspace / "intel" / "change_config.json").write_text(json.dumps(
            {"baseline_job": "nobase", "voxel_m": 0.5}))
        report = run_autonomous_pipeline(job, PipelineRequest(plugins=["change_detection"],
                                                             force=["change_detection"]))
        assert report["status"] == "completed"
        assert report["stages"]["change_detection"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# REST integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intel_api(client, db_session: AsyncSession, tmp_path: Path):
    """POST /api/intel/run over a built twin; artifacts + copilot are served."""
    job_id = "intelapi0001"
    db_session.add(Project(id=job_id, name="site.avi", video_filename="site.avi",
                           video_path=str(tmp_path / "site.avi"), status="uploaded"))
    await db_session.flush()
    _seed_dense_workspace(tmp_path, job_id)
    twin = run_autonomous_pipeline(job_id, PipelineRequest(plugins=list(MESH_CHAIN)))
    assert twin["status"] == "completed", twin.get("error")

    resp = await client.post(f"/api/intel/run/{job_id}", json={"plugins": []})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"

    dash = await client.get(f"/api/intel/dashboard/{job_id}")
    assert dash.status_code == 200
    assert "alerts" in dash.json()

    risk = await client.get(f"/api/intel/artifact/{job_id}/risk_report")
    assert risk.status_code == 200 and "overall_risk" in risk.json()

    page = await client.get(f"/api/intel/dashboard/page/{job_id}")
    assert page.status_code == 200 and page.content.startswith(b"<!doctype html>")

    report = await client.get(f"/api/intel/report/{job_id}?format=md")
    assert report.status_code == 200 and report.content.startswith(b"# DroneRecon")

    cop = await client.post(f"/api/intel/copilot/{job_id}",
                            json={"query": "summary of the mission"})
    assert cop.status_code == 200
    body = cop.json()
    assert body["grounded"] is True

    artifact = await client.get(f"/api/intel/artifact/{job_id}/Risk_Report")
    assert artifact.status_code == 400  # name is restricted to [a-z_]+

    status = await client.get(f"/api/intel/status/{job_id}")
    assert status.status_code == 200 and status.json()["status"] == "completed"

    missing = await client.post("/api/intel/run/ghost", json={"plugins": []})
    assert missing.status_code == 404
