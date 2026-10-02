"""Tests for Phase 9 — mission planning: physical simulation, what-if
comparison, battery model, risk planner, path optimization, mission history /
learning gates, the planning chain over a built twin, and the REST API.

Every assertion exercises a deterministic model whose outputs carry explicit
``method`` / ``estimate`` labels and ``source`` (measured | simulation_estimate |
historical) — nothing is asserted to be a learned or ground-truth value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.models import Project
from app.services.mission_history import (
    append_record,
    learning_summary,
    load_history,
    similar_missions,
    validate_mission,
)
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.pipeline_stage import INTEL_CHAIN, MESH_CHAIN, PLANNING_CHAIN
from tests.test_digital_twin import _grid_mesh, _seed_dense_workspace


def _scene(w_m: float = 300.0, h_m: float = 200.0, **extra) -> dict:
    scene = {"extent_w_m": w_m, "extent_h_m": h_m}
    scene.update(extra)
    return scene


# ---------------------------------------------------------------------------
# Mission simulator (pinhole GSD / geometry / quality-index model)
# ---------------------------------------------------------------------------


class TestSimulator:
    def test_footprint_gsd_math(self):
        from app.services.mission_simulator import footprint

        f60 = footprint(60.0)
        f120 = footprint(120.0)
        assert f60["gsd_cm"] == pytest.approx(1.64, abs=0.01)
        # double altitude → double footprint width, double GSD
        assert f120["footprint_w_m"] == pytest.approx(2.0 * f60["footprint_w_m"])
        assert f120["gsd_cm"] == pytest.approx(2.0 * f60["gsd_cm"], abs=0.02)
        assert f60["method"] == "pinhole_projection_nadir"

    def test_simulate_requires_extent(self):
        from app.services.mission_simulator import plan_defaults, simulate

        with pytest.raises(ValueError):
            simulate(plan_defaults(), scene={})

    def test_lower_altitude_trades_cost_for_quality(self):
        from app.services.mission_simulator import plan_defaults, simulate

        base = simulate(plan_defaults(), _scene())
        low = simulate({**plan_defaults(), "altitude_m": 40.0}, _scene())
        assert low["gsd"] < base["gsd"]
        assert low["quality_index"] > base["quality_index"]
        assert low["frames"] > base["frames"]
        assert low["duration_min"] >= base["duration_min"]
        assert low["estimate"] is True
        assert low["method"]["quality"].startswith("gsd_overlap_speed")

    def test_compare_deltas_and_honest_insufficiency(self):
        from app.services.mission_simulator import compare, plan_defaults, simulate

        base = simulate(plan_defaults(), _scene())
        low = simulate({**plan_defaults(), "altitude_m": 40.0}, _scene())
        cmp = compare(base, low)
        assert cmp["deltas"]["gsd"]["delta_pct"] < 0
        assert cmp["deltas"]["quality_index"]["delta_pct"] > 0
        assert cmp["deltas"]["frames"]["delta_pct"] > 0
        assert cmp["method"] == "what_if_simulation_compare"

        # an empty scenario cannot be compared → no invented numbers
        empty = compare(base, {})
        assert empty["deltas"]["quality_index"]["delta_pct"] is None
        assert "insufficient evidence" in empty["deltas"]["quality_index"]["note"]

    def test_quality_index_respects_measured_evidence(self):
        from app.services.mission_simulator import relative_quality_index

        hi = relative_quality_index({"altitude_m": 40.0, "speed_m_s": 5.0})
        lo = relative_quality_index({"altitude_m": 80.0, "speed_m_s": 12.0})
        assert hi["value"] > lo["value"]
        penalized = relative_quality_index({"altitude_m": 40.0, "speed_m_s": 5.0},
                                           measured_conf=0.3)
        assert penalized["value"] < hi["value"] - 0.05
        assert 0.0 <= penalized["value"] <= 1.0
        assert "range" in penalized and len(penalized["range"]) == 2


# ---------------------------------------------------------------------------
# Battery model (physics proxy — Wh, not kWh)
# ---------------------------------------------------------------------------


class TestBattery:
    def test_consumption_is_wh_not_kwh(self):
        from app.services.battery_model import battery_estimate
        from app.services.mission_simulator import plan_defaults

        b = battery_estimate(plan_defaults(), 10.0 / 60.0)
        # ~148 W cruise × 0.167 h ≈ 25 Wh on a 60 Wh pack
        assert 10.0 < b["consumed_wh"] < 40.0
        assert 40.0 < b["remaining_pct"] < 80.0
        assert b["completion_prob"] == pytest.approx(1.0)
        assert b["estimate"] is True and b["method"] == "physics_proxy_model"

    def test_longer_flights_and_speed_consume_more(self):
        from app.services.battery_model import battery_estimate
        from app.services.mission_simulator import plan_defaults

        plan = plan_defaults()
        b1 = battery_estimate(plan, 0.1)
        b2 = battery_estimate(plan, 1.0)
        assert b2["consumed_wh"] > b1["consumed_wh"] + 50.0
        # parasitic drag grows with speed³ → a faster plan burns more per hour
        fast = battery_estimate({**plan, "speed_m_s": 12.0}, 0.1)
        assert fast["consumed_wh"] > b1["consumed_wh"] + 5.0
        assert fast["power"]["cruise_w"] > b1["power"]["cruise_w"]

    def test_infeasible_plan_reports_zero_completion(self):
        from app.services.battery_model import battery_estimate
        from app.services.mission_simulator import plan_defaults

        b = battery_estimate(plan_defaults(), 8.0)
        assert b["completion_prob"] == 0.0
        assert b["reserve_margin_wh"] < 0.0

    def test_max_flight_time_keeps_reserve(self):
        from app.services.battery_model import max_flight_time
        from app.services.mission_simulator import plan_defaults

        mft = max_flight_time(plan_defaults())
        # usable = 60 Wh × 0.8 = 48 Wh at ~148 W → ~19 min, ~9.4 km
        assert 10.0 < mft["max_flight_time_min"] < 40.0
        assert 4000.0 < mft["max_range_m"] < 15000.0
        assert mft["estimate"] is True


# ---------------------------------------------------------------------------
# Risk-aware planner
# ---------------------------------------------------------------------------


class TestRisk:
    def test_weather_severity_from_environment_report(self):
        from app.services.risk_aware_planner import weather_severity

        env = {"conditions": {"fog_haze": {"p90_score": 0.9}}}
        assert weather_severity(env) == pytest.approx(0.54, abs=0.001)
        assert weather_severity(None) == 0.0
        assert weather_severity({}) == 0.0

    def test_clean_plan_is_low_risk(self):
        from app.services.mission_simulator import plan_defaults, simulate
        from app.services.risk_aware_planner import assess_plan_risk

        sim = simulate(plan_defaults(), _scene())
        risk = assess_plan_risk(plan_defaults(), sim, _scene())
        assert risk["risk_score"] == 0.0 and risk["level"] == "Low"
        assert risk["no_go"] is False
        assert risk["safety_note"]

    def test_battery_infeasible_is_the_only_no_go(self):
        from app.services.battery_model import battery_estimate
        from app.services.mission_simulator import plan_defaults
        from app.services.risk_aware_planner import assess_plan_risk

        plan = plan_defaults()
        dead = {"battery": battery_estimate(plan, 8.0)}  # completion 0.0
        risk = assess_plan_risk(plan, dead, _scene(max_scene_height_m=0.0))
        assert risk["no_go"] is True
        assert "insufficient" in risk["warnings"][0]

        # restricted + clearance hazards raise the score but never set no_go
        sim = {"battery": battery_estimate(plan, 0.05)}
        scene = _scene(max_scene_height_m=120.0, restricted=[{"x0": 0, "x1": 50}])
        risky = assess_plan_risk(plan, sim, scene)
        assert risky["no_go"] is False
        assert risky["risk_score"] >= 40.0
        assert any("restricted" in w for w in risky["warnings"])
        assert any("clearance" in w for w in risky["warnings"])


# ---------------------------------------------------------------------------
# Path optimization (multi-objective Pareto + labelled archetypes)
# ---------------------------------------------------------------------------


class TestOptimizer:
    def test_candidates_include_revisit_when_targets_exist(self):
        from app.services.path_optimizer import candidate_plans

        plain = candidate_plans(_scene())
        with_t = candidate_plans(_scene(targets=[{"centroid": [5, 5], "radius_m": 12.0}]))
        assert all(p["pattern"] != "revisit" for p in plain)
        assert any(p["pattern"] == "revisit" for p in with_t)

    def test_optimize_returns_labelled_archetypes(self):
        from app.services.path_optimizer import optimize

        out = optimize(_scene(blind_share=0.25, targets=[{"centroid": [5, 5],
                                                          "radius_m": 12.0}]))
        assert out["estimate"] is True
        labels = [p["label"] for p in out["plans"]]
        for want in ("BEST QUALITY", "BEST BATTERY", "BEST TIME", "BEST SAFETY",
                     "BEST BALANCED"):
            assert want in labels
        for p in out["plans"]:
            m = p["metrics"]
            assert 0.0 <= m["quality_index"] <= 1.0
            assert 0.0 <= m["coverage_est"] <= 1.0
            assert m["duration_min"] > 0.0
            assert "battery" in p and "risk_level" in p
        assert out["candidates_evaluated"] > 0

    def test_pareto_front_is_non_dominated(self):
        from app.services.path_optimizer import pareto_front

        evals = [
            {"metrics": {"quality_index": 0.9, "coverage_est": 0.8, "duration_min": 5.0,
                         "battery_wh": 20.0, "risk_score": 10.0}},
            {"metrics": {"quality_index": 0.5, "coverage_est": 0.4, "duration_min": 9.0,
                         "battery_wh": 30.0, "risk_score": 40.0}},
            {"metrics": {"quality_index": 0.85, "coverage_est": 0.75, "duration_min": 6.0,
                         "battery_wh": 22.0, "risk_score": 25.0}},
        ]
        front, direction = pareto_front(evals)
        assert len(front) == 1 and front[0]["metrics"]["quality_index"] == 0.9
        assert set(direction) == {"quality_index", "coverage_est", "duration_min",
                                  "battery_wh", "risk_score"}


# ---------------------------------------------------------------------------
# Mission history: store, features, similarity, learning gates, validation
# ---------------------------------------------------------------------------


def _rec(mission_id: str, **kw) -> dict:
    base = {"mission_id": mission_id, "timestamp": "2026-01-01T00:00:00Z",
            "altitude_m": 60.0, "speed_m_s": 8.0, "forward_overlap": 0.7,
            "side_overlap": 0.6, "env_score": 80.0, "gps_score": 85.0,
            "coverage": 90.0, "confidence": 80.0, "battery_consumed_pct": 45.0,
            "processing_time_s": 120.0, "frames": 100, "mission_success": True}
    base.update(kw)
    return base


class TestHistory:
    def test_append_dedupe_and_load(self, tmp_path: Path):
        base = tmp_path / "hist"
        append_record(_rec("m1"), base)
        append_record(_rec("m2"), base)
        append_record(_rec("m1", coverage=95.0), base)  # overwrite in place
        records = load_history(base)
        assert [r["mission_id"] for r in records] == ["m2", "m1"]
        assert records[-1]["coverage"] == 95.0

    def test_features_keep_missing_values(self):
        from app.services.mission_history import build_features

        feats = build_features([_rec("a"), _rec("b", altitude_m=None, env_score=None)])
        assert feats[0]["altitude_m"] == 60.0
        assert feats[1]["altitude_m"] is None and feats[1]["env_score"] is None
        assert feats[1]["mission_id"] == "b"

    def test_similar_missions_returns_nearest_historical(self):
        records = [
            _rec("far", altitude_m=120.0, speed_m_s=12.0, coverage=70.0, confidence=60.0),
            _rec("near", altitude_m=55.0, speed_m_s=7.0, coverage=94.0, confidence=91.0),
            _rec("also_near", altitude_m=62.0, speed_m_s=8.0, coverage=91.0, confidence=88.0),
        ]
        query = {"mission_id": "q", "altitude_m": 58.0, "speed_m_s": 7.5,
                 "forward_overlap": 0.7, "side_overlap": 0.6, "env_score": 80.0}
        out = similar_missions(query, records, k=2)
        assert out["status"] == "historical" and out["count"] == 2
        assert out["neighbours"][0]["source"] == "historical"
        assert out["label"].startswith("historical")
        assert out["average_coverage"] is not None

    def test_similar_empty_history(self, tmp_path: Path):
        out = similar_missions({"mission_id": "q"}, [])
        assert out["status"] == "insufficient_data"
        assert "no previous missions" in out["message"]

    def test_learning_gate_insufficient(self):
        summary = learning_summary([_rec("m1")])
        assert summary["status"] == "insufficient_for_ml"
        assert summary["ml_gate"]["message"] and "Insufficient" in summary["ml_gate"]["message"]
        assert summary["best_performer"]["label"].startswith("historical")

    def test_learning_gate_opens_with_enough_records(self):
        records = [_rec(f"m{i}") for i in range(settings.planning.min_history_for_ml)]
        summary = learning_summary(records)
        assert summary["status"] == "ready"
        assert summary["ml_gate"]["message"] is None
        assert summary["missions"] == settings.planning.min_history_for_ml

    def test_no_history(self):
        summary = learning_summary([])
        assert summary["status"] == "no_history"

    def test_validate_prediction_error_in_percent(self):
        rec = _rec("m1", coverage=85.0, confidence=78.0,
                   plan_predicted={"coverage_est": 90.0, "quality_index": 84.0,
                                   "duration_min": 4.4})
        row = validate_mission(rec)
        assert row["prediction_error"] == {"coverage_est": -5.0, "quality_index": -6.0}
        assert row["label"] == "prediction-vs-actual validation"

    def test_validate_no_prediction(self):
        row = validate_mission(_rec("m1"))
        assert row["status"] == "no_prediction_recorded"

    def test_validate_duration_only_with_measured_duration(self):
        rec = _rec("m1", plan_predicted={"duration_min": 5.0},
                   duration_measured_min=6.0)
        row = validate_mission(rec)
        assert row["measured"]["duration_min"] == 6.0
        assert row["prediction_error"]["duration_min"] == pytest.approx(1.0)
        # processing_time_s (compute time) is never used as measured duration
        rec2 = _rec("m2", plan_predicted={"duration_min": 5.0})
        assert "duration_min" not in validate_mission(rec2)["measured"]

    def test_record_from_workspace_uses_percent_units(self, tmp_path: Path):
        from app.services.mission_history import record_from_workspace

        ws = tmp_path / "ws"
        (ws / "confidence").mkdir(parents=True)
        (ws / "confidence" / "confidence_report.json").write_text(json.dumps(
            {"low_share": 0.2, "mean_confidence": 0.7}))
        (ws / "intel").mkdir(parents=True)
        (ws / "intel" / "mission_plan.json").write_text(json.dumps(
            {"chosen_plan": {"plan": {"altitude_m": 60.0, "speed_m_s": 8.0,
                                      "forward_overlap": 0.7},
                             "metrics": {"coverage_est": 0.95, "quality_index": 0.8,
                                         "duration_min": 4.4}}}))
        rec = record_from_workspace(ws)
        assert rec["coverage"] == 80.0 and rec["confidence"] == 70.0
        assert rec["plan_predicted"] == {"coverage_est": 95.0, "quality_index": 80.0,
                                         "duration_min": 4.4}
        assert rec["measured"] is True
        row = validate_mission(rec)
        assert row["prediction_error"]["coverage_est"] == -15.0
        assert abs(row["prediction_error"]["quality_index"] + 10.0) < 0.1


# ---------------------------------------------------------------------------
# Coverage predictor (measured rasters over the twin mesh)
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_coverage_rasters_from_mesh_and_confidence(self):
        from app.services.coverage_predictor import build_coverage

        mesh = _grid_mesh(12)
        conf = np.ones(mesh.n)
        out = build_coverage(mesh, labels=None, conf=conf)
        assert out["source"] == "measured"
        assert out["blind_cells"] == 0 and out["blind_share"] == 0.0
        assert len(out["observed"]) == out["grid"]["cells"]
        assert out["confidence"] is not None

        # a low-confidence corner produces measured blind cells + clusters
        low = conf.copy()
        low[: int(mesh.n) // 4] = 0.1
        out2 = build_coverage(mesh, labels=None, conf=low)
        assert out2["blind_cells"] > 0 and out2["blind_share"] > 0.0
        assert any(w["vertices"] >= 8 for w in out2["weak_regions"])

    def test_class_shares_from_semantic_labels(self):
        from app.services.coverage_predictor import build_coverage

        mesh = _grid_mesh(8)
        labels = np.zeros(mesh.m, dtype=np.int64)
        out = build_coverage(mesh, labels=labels, conf=None)
        assert out["confidence"] is None and out["weak_regions"] == []
        assert abs(sum(out["class_shares"].values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Mission planner recommendations + report exports
# ---------------------------------------------------------------------------


class TestPlanner:
    def test_recommendations_carry_full_structure(self):
        from app.services.mission_planner import recommend_from

        chosen = {"label": "BEST BALANCED", "plan": {"altitude_m": 50.0},
                  "metrics": {"quality_index": 0.6, "coverage_est": 0.9,
                              "duration_min": 6.0, "battery_wh": 12.0,
                              "risk_score": 10.0}, "battery": {"completion_prob": 1.0}}
        recs = recommend_from([], chosen, _scene(targets=[{"centroid": [3, 3],
                                                           "radius_m": 10.0}]), env=None)
        assert recs, "the chosen-plan recommendation should always be present"
        for r in recs:
            for field in ("recommendation", "reason", "evidence", "expected_benefit",
                          "expected_cost", "confidence", "source"):
                assert field in r, (field, r)
            assert r["source"] in ("measured", "simulation_estimate", "historical",
                                   "user_provided")
        assert any(r["source"] == "measured" for r in recs)  # weak-region revisit

    def test_what_if_tag_rules_recommendation_in(self):
        from app.services.mission_planner import recommend_from

        chosen = {"label": "BEST BALANCED", "plan": {}, "metrics": {}, "battery": {}}
        delta = {"tag": "lower_altitude_20m", "battery_pct": 9.0, "duration_pct": 13.0,
                 "compare": {"deltas": {"quality_index": {"delta_pct": 12.0},
                                        "gsd": {"delta_pct": -33.0}}}}
        recs = recommend_from([delta], chosen, _scene(), env=None)
        assert any("fly 20 m lower" in r["recommendation"] for r in recs)
        assert any(r["evidence"]["source"] == "simulation_estimate" for r in recs)

    def test_plan_exports_geojson_kml_md(self, tmp_path: Path):
        from app.services.mission_planner import path_waypoints
        from app.services.mission_report import (
            export_plan_geometry,
            render_plan_markdown,
            write_plan_exports,
        )

        doc = {"job": "job1",
               "scene": {"extent_w_m": 300.0, "extent_h_m": 200.0,
                         "blind_share": 0.2, "mean_confidence": 0.7},
               "baseline": {"plan": {"altitude_m": 60, "speed_m_s": 8,
                                     "pattern": "lawnmower"},
                            "metrics": {"gsd": 1.64, "frames": 108, "duration_min": 4.4}},
               "what_if": [], "recommendations": [],
               "chosen_plan": {"label": "BEST QUALITY",
                               "plan": {"pattern": "lawnmower", "altitude_m": 40.0,
                                        "speed_m_s": 5.0},
                               "metrics": {"quality_index": 0.8, "coverage_est": 0.95}}}
        pts = path_waypoints(doc["chosen_plan"]["plan"], doc["scene"])
        assert len(pts) >= 4 and all(len(p) == 2 for p in pts)

        out = tmp_path / "exports"
        export_plan_geometry(doc, out)
        gj = json.loads((out / "mission_plan.geojson").read_text())
        coords = gj["features"][0]["geometry"]["coordinates"]
        assert gj["features"][0]["properties"]["source"] == "simulation_estimate"
        assert len(coords) == len(pts)
        kml = (out / "mission_plan.kml").read_text()
        assert kml.startswith("<?xml") and "<LineString>" in kml
        md = render_plan_markdown(doc)
        assert "Mission plan" in md and "BEST QUALITY" in md
        written = write_plan_exports(doc, out)
        assert set(written) == {"geojson", "kml", "md"}


# ---------------------------------------------------------------------------
# Flight copilot (source-labelled answers)
# ---------------------------------------------------------------------------


class TestCopilot:
    def test_battery_question_simulated_source(self):
        from app.services.flight_copilot import answer

        out = answer("will my battery be enough for this mission")
        assert out["intent"] == "battery"
        assert all(m["source"] == "simulation_estimate" for m in out["metrics"])
        assert "physics proxy" in out["evidence"] and "telemetry" in out["evidence"]

    def test_altitude_whatif_numbers_from_simulator(self):
        from app.services.flight_copilot import answer

        out = answer("how much improvement if I fly 20 m lower", ctx={
            "scene": _scene()})
        assert out["intent"] == "altitude_whatif"
        q = next(m for m in out["metrics"] if m["metric"] == "quality_index_delta_pct")
        assert q["value"] > 0 and q["source"] == "simulation_estimate"

    def test_coverage_advice_uses_measured_scene(self):
        from app.services.flight_copilot import answer

        scene = _scene(targets=[{"centroid": [3, 3, 1], "radius_m": 10.0}],
                       blind_share=0.3)
        out = answer("which regions should I revisit", ctx={"scene": scene})
        assert out["intent"] == "coverage"
        assert any(m["source"] == "measured" for m in out["metrics"])

    def test_history_question_without_data_is_honest(self):
        from app.services.flight_copilot import answer

        out = answer("what happened on previous similar missions", ctx={"history": []})
        assert out["intent"] == "similar"
        assert "insufficient historical data" in out["answer"]
        assert out["metrics"] == []

    def test_history_question_with_records_uses_historical_label(self):
        from app.services.flight_copilot import answer

        records = [_rec("m1", altitude_m=50.0, env_score=75.0, coverage=94.0,
                        confidence=91.0)]
        # the similarity query must come from the mission record / plan
        # parameters, not bare scene geometry (regression: scene extent shares
        # no comparison keys with history, so every answer said "no data")
        ctx_record = _rec("current", altitude_m=52.0, speed_m_s=8.0,
                          forward_overlap=0.7, side_overlap=0.6,
                          env_score=75.0, gps_score=85.0)
        out = answer("how did similar missions perform", ctx={
            "history": records, "record": ctx_record, "scene": _scene()})
        assert out["intent"] == "similar"
        assert any(m["source"] == "historical" for m in out["metrics"])
        assert "historical" in out["evidence"]
        assert "1 previous missions" in out["answer"]
        assert "94" in out["answer"]  # historical average coverage

        # a stored plan alone is enough context for similarity
        plan_only = answer("how did similar missions perform", ctx={
            "history": records,
            "plan": {"chosen_plan": {"plan": {"altitude_m": 50.0,
                                                 "speed_m_s": 8.0}}}})
        assert plan_only["intent"] == "similar"
        assert "historical observations" in plan_only["answer"]

    def test_strategy_history_dispatch(self):
        from app.services.flight_copilot import answer

        # spec question: strategy + history keywords must reach the history
        # engine (which reports insufficient data here), not the plan branch
        out = answer("which strategy performed best historically", ctx={"history": []})
        assert out["intent"] == "similar"
        assert "insufficient" in out["answer"]

    def test_risk_and_plan_and_clarify(self):
        from app.services.flight_copilot import answer

        assert answer("is the plan safe")["intent"] == "risk"
        assert answer("which path should I take")["intent"] == "plan"
        assert answer("what is the weather tomorrow")["intent"] == "clarify"


# ---------------------------------------------------------------------------
# Planning chain over a seeded digital twin (full pipeline e2e)
# ---------------------------------------------------------------------------


class TestPlanningChain:
    def test_full_planning_chain(self, tmp_path: Path):
        settings.storage.base_path = str(tmp_path)
        job_id = "planjob0001"
        workspace = _seed_dense_workspace(tmp_path, job_id)
        twin = run_autonomous_pipeline(job_id, PipelineRequest(plugins=list(MESH_CHAIN)))
        assert twin["status"] == "completed", twin.get("error")
        intel = run_autonomous_pipeline(job_id, PipelineRequest(plugins=list(INTEL_CHAIN)))
        assert intel["status"] == "completed", intel.get("error")

        plan = run_autonomous_pipeline(job_id, PipelineRequest(plugins=list(PLANNING_CHAIN)))
        assert plan["status"] == "completed", plan.get("error")
        for name in PLANNING_CHAIN:
            assert plan["stages"][name]["status"] == "completed", (name, plan["stages"][name])

        # artifacts exist and are internally consistent
        plan_doc = json.loads((workspace / "intel" / "mission_plan.json").read_text())
        assert plan_doc["chosen_plan"]["label"].startswith("BEST ")
        assert len(plan_doc["candidate_plans"]) >= 5
        assert plan_doc["what_if"] and plan_doc["recommendations"]
        assert (workspace / "intel" / "mission_plan.geojson").exists()
        assert (workspace / "intel" / "mission_plan.html").exists()

        learning = json.loads((workspace / "intel" / "learning_report.json").read_text())
        assert learning["history_total"] >= 1
        assert learning["learning"]["label"]
        assert learning["learning"]["ml_gate"]["message"]

        validation = json.loads((workspace / "intel" / "validation.json").read_text())
        # plan predictions existed → a real comparison row with sane scale
        errs = validation.get("prediction_error", {})
        assert abs(errs.get("quality_index", 0.0)) < 100.0

        coverage = json.loads((workspace / "intel" / "coverage_prediction.json").read_text())
        assert coverage["source"] == "measured"
        assert 0.0 <= coverage["blind_share"] <= 1.0

        # rerun → everything resumes from artifacts (no recompute)
        again = run_autonomous_pipeline(job_id, PipelineRequest(plugins=list(PLANNING_CHAIN)))
        assert again["status"] == "completed"
        for name in PLANNING_CHAIN:
            assert again["stages"][name]["status"] == "skipped", name


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mission_api(client, db_session: AsyncSession, tmp_path: Path):
    settings.storage.base_path = str(tmp_path)
    job_id = "missionapi0001"
    db_session.add(Project(id=job_id, name="site.avi", video_filename="site.avi",
                           video_path=str(tmp_path / "site.avi"), status="uploaded"))
    await db_session.flush()
    _seed_dense_workspace(tmp_path, job_id)

    # pure planning endpoints do not need a twin
    scene = {"extent_w_m": 300.0, "extent_h_m": 200.0}
    sim = await client.post("/api/mission/simulate", json={"scene": scene})
    assert sim.status_code == 200, sim.text
    body = sim.json()
    assert body["gsd"] > 0 and body["frames"] > 0 and body["estimate"] is True

    wi = await client.post("/api/mission/what-if",
                           json={"scene": scene, "scenario": {"altitude_m": -20.0}})
    assert wi.status_code == 200
    assert wi.json()["deltas"]["gsd"]["delta_pct"] < 0

    opt = await client.post("/api/mission/optimize", json={"scene": scene})
    assert opt.status_code == 200
    assert any(p["label"] == "BEST BALANCED" for p in opt.json()["plans"])

    bat = await client.get("/api/mission/battery", params={"duration_min": 10.0})
    assert bat.status_code == 200
    assert 0 < bat.json()["consumed_wh"] < 60

    # build the twin, then run the planning chain through the API
    twin = run_autonomous_pipeline(job_id, PipelineRequest(plugins=list(MESH_CHAIN)))
    assert twin["status"] == "completed", twin.get("error")

    an = await client.post(f"/api/mission/analyze/{job_id}")
    assert an.status_code == 200, an.text
    assert an.json()["status"] == "completed", an.json()

    plan = await client.get(f"/api/mission/plan/{job_id}")
    assert plan.status_code == 200 and "chosen_plan" in plan.json()
    md = await client.get(f"/api/mission/plan/{job_id}?format=md")
    assert md.status_code == 200 and "Mission plan" in md.json()["plan_markdown"]
    kml = await client.get(f"/api/mission/plan_file/{job_id}?format=kml")
    assert kml.status_code == 200

    recs = await client.get(f"/api/mission/recommendations/{job_id}")
    assert recs.status_code == 200 and recs.json()["recommendations"]

    cop = await client.post("/api/mission/copilot",
                            json={"query": "should I fly lower", "job_id": job_id})
    assert cop.status_code == 200
    assert cop.json()["intent"] == "altitude_whatif"

    # the analyze run appended one measured history record
    hist = await client.get("/api/mission/history")
    assert hist.status_code == 200 and hist.json()["count"] == 1

    learning = await client.get("/api/mission/learning")
    assert learning.status_code == 200
    assert learning.json()["ml_gate"]["message"]  # still under the ML gate

    simi = await client.get("/api/mission/similar")
    assert simi.status_code == 200 and simi.json()["status"] == "historical"

    replay = await client.get(f"/api/mission/replay/{job_id}")
    assert replay.status_code == 200 and replay.json()["mission"]["mission_id"] == job_id
    missing = await client.get("/api/mission/replay/ghost")
    assert missing.status_code == 400

    val = await client.post(f"/api/mission/validate/{job_id}")
    assert val.status_code == 200
    assert "prediction_error" in val.json()["validation"]

    appended = await client.post("/api/mission/history",
                                 json={"record": {"mission_id": "extra", "coverage": 88.0}})
    assert appended.status_code == 200 and appended.json()["stored"] is True
    assert (await client.get("/api/mission/history")).json()["count"] == 2

    bad = await client.post("/api/mission/simulate",
                            json={"scene": {"extent_w_m": 300.0}})
    assert bad.status_code == 422  # missing extent_h_m
