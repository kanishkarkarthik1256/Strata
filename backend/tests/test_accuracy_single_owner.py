"""Single-owner contract for the accuracy stack.

The audit found the accuracy story spread across four modules with three
artifact names and two contradictory capability vocabularies, and an API
endpoint that fabricated a ``GPS_GEOREFERENCED`` claim whenever a
``poses.json`` existed. These tests pin the consolidated contract:

* ONE artifact name, produced by ONE writer (``metric_validation``);
* ONE capability derivation, which never claims more than its evidence;
* accuracy is read through the API, never assembled from raw files;
* ``coverage_percent`` means exactly one thing (dense footprint occupancy),
  so a capture-quality ratio cannot be mistaken for surface coverage.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.services import metric_validation as mv

BACKEND = Path(__file__).resolve().parents[1]


def test_one_artifact_name_and_round_trip(tmp_path: Path):
    """persist_report/load_report own the canonical artifact."""
    assert mv.ARTIFACT_RELATIVE_PATH == Path("validation") / "validation_report.json"
    assert mv.load_report(tmp_path) is None  # absent, not a fabricated default

    report = mv.not_certified_report("run_1", "no reference for this scene")
    written = mv.persist_report(tmp_path, report)

    assert written == tmp_path / "validation" / "validation_report.json"
    assert written.is_file()
    loaded = mv.load_report(tmp_path)
    assert loaded is not None
    assert loaded["certification_status"] == report["certification_status"]


def test_only_metric_validation_writes_the_artifact():
    """Structural guard: no other module may write the accuracy artifact.

    A second writer is how the duplicate schemas and the ``.md``/``.json``
    pair appeared in the first place.
    """
    offenders: list[str] = []
    write_calls = ("write_text", "json.dump", "open(")
    for path in list((BACKEND / "app").rglob("*.py")) + list((BACKEND / "scripts").glob("*.py")):
        if path.name == "metric_validation.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "validation_report.json" in line and any(c in line for c in write_calls):
                offenders.append(f"{path.relative_to(BACKEND)}:{lineno}")
    assert offenders == [], f"accuracy artifact written outside its owner: {offenders}"


def test_retired_accuracy_authorities_are_gone():
    """The duplicate aggregator, its dead writer and the orphan modules must
    not come back — each was a second authority on accuracy."""
    validator_src = (BACKEND / "app" / "services" / "ground_truth_validator.py").read_text()
    assert "build_metric_accuracy_report" not in validator_src.replace(
        "``build_metric_accuracy_report``", ""
    ), "the retired aggregator returned"
    assert "_export_reports" not in validator_src
    assert 'val_dir / "metric_validation.json"' not in validator_src

    assert not (BACKEND / "app" / "services" / "synthetic_benchmark.py").exists()
    assert not (BACKEND / "app" / "services" / "movingdrone_adapter.py").exists()

    schemas = (BACKEND / "app" / "schemas" / "ground_truth.py").read_text()
    assert "class MetricAccuracyReport" not in schemas


def test_endpoint_has_no_fabricating_fallback():
    """The metric-validation route must serve the artifact or the honest
    envelope — never synthesize a capability claim from a file's existence."""
    src = (BACKEND / "app" / "routes" / "reconstruction.py").read_text()
    assert "gt_poses" not in src, "a fabricated ground-truth pose returned to the route"
    assert "not_certified_report" in src
    assert "metric_accuracy_report.json" not in src


def test_envelope_asserts_nothing():
    """The no-artifact envelope carries no measurement, provenance or claim.

    Regression: it used to hardcode ``relative_reconstruction: "available"``,
    which made a run that never executed render as though it had reprojection /
    cross-view / dense→mesh diagnostics.
    """
    env = mv.not_certified_report("run_2", "insufficient GPS correspondence")
    assert env["certification_status"] == mv.STATUS_NOT_MEASURED
    assert env["accuracy_certified"] is False
    assert env["capability_state"] == "NOT_MEASURED"
    # Every measured section is explicitly absent...
    for section in ("relative_consistency", "registration", "checkpoint_validation",
                    "scale_validation", "reference_comparison", "error_decomposition",
                    "reference_type", "reference_source", "figures"):
        assert env[section] is None, section
    # ...and nothing claims relative quality.
    assert env["accuracy_summary"]["relative_reconstruction"] == "not measured"
    assert env["accuracy_summary"]["metric_scale"] == "not measured"
    assert "available" not in env["accuracy_summary"].values()
    # No provenance claim either.
    assert "gps_camera_alignment" not in json.dumps(env)
    assert "NOT VALIDATED" not in json.dumps(env)


def test_accuracy_summary_vocabulary_is_one_token_per_fact():
    """Every builder spells the relative-quality fact the same way.

    Regression (live): the reference-path builder emitted
    ``"available"/"unavailable"`` while the internal-consistency builder and
    the no-artifact envelope emit ``"measured"/"not measured"``. The Reports
    page matches that token, so Video_Mission_9ab1aa — 0.32 px mean
    reprojection with cross-view and dense→mesh diagnostics on disk — rendered
    "Relative reconstruction quality: Unavailable".
    """
    src = (BACKEND / "app" / "services" / "metric_validation.py").read_text()
    assignments = [l.strip() for l in src.splitlines() if '"relative_reconstruction":' in l]
    assert len(assignments) >= 3, assignments
    for line in assignments:
        assert '"measured" if' in line or '"not measured"' in line, line


def test_report_and_envelope_share_one_shape():
    """The two must be built by the same code path, so shapes cannot drift."""
    skeleton = mv.build_report("r", scene="s", dataset_split="development")
    envelope = mv.not_certified_report("r", "why")
    assert set(envelope) == set(skeleton)

    # And a real, measured report exposes the same keys as the skeleton.
    measured = mv.validate_run(Path(tempfile.mkdtemp()), "s", None, None, "development")
    assert set(measured) == set(skeleton), set(measured) ^ set(skeleton)


async def test_endpoint_404s_when_no_accuracy_report_exists(client, db_session, tmp_path):
    """Absence must be reported as absence.

    A 200 envelope made the page's honest "no report exists" state unreachable
    and let an unmeasured run render as measured. The body still explains why,
    in the engine's own report shape.
    """
    from app.config.settings import settings
    from app.db.models import Project

    job_id = "noacc0001"
    db_session.add(Project(id=job_id, name="s.avi", video_filename="s.avi",
                           video_path=str(tmp_path / "s.avi"), status="uploaded"))
    await db_session.flush()
    settings.storage.base_path = str(tmp_path)
    (tmp_path / job_id).mkdir(parents=True, exist_ok=True)

    resp = await client.get(f"/api/reconstruction/metric-validation/{job_id}")
    assert resp.status_code == 404, resp.text
    # The explanation must survive the app's error contract (string detail),
    # not degrade into a generic "resource not found".
    detail = resp.json()["detail"]
    assert isinstance(detail, str) and detail
    assert "no accuracy report exists" in detail
    assert mv.STATUS_NOT_MEASURED in detail

    # With the canonical artifact present the same route serves it verbatim.
    mv.persist_report(tmp_path / job_id, {
        "run_id": job_id,
        "certification_status": "CERTIFIED ≤1m (STRATA engineering criterion)",
        "certification_reason": "3D RMSE 0.4 m / P95 0.7 m",
        "accuracy_summary": {"relative_reconstruction": "available", "absolute": "CERTIFIED"},
    })
    resp2 = await client.get(f"/api/reconstruction/metric-validation/{job_id}")
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["certification_status"].startswith("CERTIFIED")


def test_capability_ladder_requires_evidence():
    """Each level needs its own proof; georeferencing alone cannot skip rungs."""
    # Nothing measured is NOT the same claim as a measured relative model.
    assert mv.capability_state(mv.AccuracyFacts())[0] == "NOT_MEASURED"
    assert mv.capability_state(mv.AccuracyFacts(has_relative_measurement=True))[0] == "RELATIVE_ONLY"
    assert mv.capability_state(mv.AccuracyFacts(scale_factor=1.05))[0] == "SCALED"
    assert (
        mv.capability_state(mv.AccuracyFacts(scale_factor=1.05, has_georeferencing=True))[0]
        == "GPS_GEOREFERENCED"
    )
    # A reference that has not validated anything is not validation.
    assert (
        mv.capability_state(
            mv.AccuracyFacts(has_reference=True, reference_validated=False,
                             has_georeferencing=True)
        )[0]
        == "GPS_GEOREFERENCED"
    )
    assert (
        mv.capability_state(
            mv.AccuracyFacts(has_reference=True, reference_validated=True)
        )[0]
        == "METRIC_VALIDATED"
    )


def test_coverage_percent_is_not_reused_for_registration_rate():
    """Regression: ``coverage_percent`` meant two different quantities in two
    payloads (registration rate vs 2.5D footprint occupancy). The mission /
    dashboard field is now named for what it measures."""
    from app.services.mission_analyzer import MissionAnalysis
    from app.services.dashboard import DashboardData

    assert hasattr(MissionAnalysis(), "camera_registration_percent")
    assert not hasattr(MissionAnalysis(), "coverage_percent")
    assert hasattr(DashboardData(), "camera_registration_percent")
    assert not hasattr(DashboardData(), "coverage_percent")

    # The genuine footprint coverage survives in the dense quality report only.
    from app.services.point_statistics import DenseQuality

    assert hasattr(DenseQuality(), "coverage_percent")


def test_region_sampling_filters_are_honest_guards(tmp_path: Path):
    """The engine must never describe an artifact it did not read."""
    (tmp_path / "validation").mkdir(parents=True)
    (tmp_path / "validation" / "validation_report.json").write_text("{ not json")
    assert mv.load_report(tmp_path) is None
