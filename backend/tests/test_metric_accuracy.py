"""Automated unit and adversarial test suite for Phase 13 metric accuracy & ground-truth validation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import json

from app.schemas.ground_truth import (
    GCPPoint,
    GCPRole,
    GroundTruthData,
    KnownDistance,
)
from app.services.georeferencing import (
    align_to_enu,
    filter_valid_gps_points,
    geodetic_to_ecef,
    validate_gps_point,
    wgs84_to_enu,
)
from app.services.ground_truth_validator import (
    evaluate_camera_positions,
    evaluate_check_points,
    evaluate_depth,
    evaluate_distances,
    evaluate_reprojection,
    evaluate_surface_distance,
    separate_control_and_check_points,
    validate_gps_coordinates,
)


def test_wgs84_coordinate_transforms():
    """Verify WGS84 -> ECEF -> ENU mathematical accuracy."""
    lat0, lon0, alt0 = 37.7749, -122.4194, 10.0
    lat1, lon1, alt1 = 37.7750, -122.4194, 10.0

    ecef = geodetic_to_ecef(np.array([lat0, lat1]), np.array([lon0, lon1]), np.array([alt0, alt1]))
    assert ecef.shape == (2, 3)

    enu = wgs84_to_enu(np.array([lat0, lat1]), np.array([lon0, lon1]), np.array([alt0, alt1]), lat0, lon0, alt0)
    assert abs(enu[0, 0]) < 1e-3 and abs(enu[0, 1]) < 1e-3  # Anchor at origin (0,0,0)
    assert enu[1, 1] > 10.0  # Moving north increases North coordinate in ENU
    assert abs(enu[1, 0]) < 1.0  # Minimal east offset


def test_gps_coordinate_validation_and_zero_rejection():
    """Adversarial test for missing GPS, (0,0) coordinates, NaN, and Inf."""
    assert validate_gps_coordinates(37.7749, -122.4194, 10.0) is True
    assert validate_gps_coordinates(0.0, 0.0, 0.0) is False  # Reject (0,0) fallback
    assert validate_gps_coordinates(float("nan"), -122.4194) is False
    assert validate_gps_coordinates(37.7749, float("inf")) is False
    assert validate_gps_coordinates(95.0, 0.0) is False  # Out of range lat
    assert validate_gps_coordinates(0.0, 190.0) is False  # Out of range lon

    pts = [
        {"lat": 37.7749, "lon": -122.4194, "alt": 10.0},
        {"lat": 0.0, "lon": 0.0, "alt": 0.0},  # Invalid
        {"lat": 37.7750, "lon": -122.4194, "alt": 12.0},
    ]
    valid = filter_valid_gps_points(pts)
    assert len(valid) == 2
    assert valid[0]["lat"] == 37.7749
    assert valid[1]["lat"] == 37.7750


def test_control_vs_check_point_separation():
    """Verify Control Points and Check Points are strictly isolated."""
    cp1 = GCPPoint(id="P1", latitude=37.7, longitude=-122.4, altitude=0.0, role=GCPRole.CONTROL)
    cp2 = GCPPoint(id="P2", latitude=37.8, longitude=-122.4, altitude=0.0, role=GCPRole.CHECK)
    cp_dup = GCPPoint(id="P1", latitude=37.7, longitude=-122.4, altitude=0.0, role=GCPRole.CHECK)

    gt = GroundTruthData(control_points=[cp1], check_points=[cp2, cp_dup])
    ctrl, check = separate_control_and_check_points(gt)

    assert len(ctrl) == 1
    assert len(check) == 1
    assert check[0].id == "P2"  # P1 check point dropped due to overlap with control


def test_evaluate_camera_positions():
    """Verify camera position RMSE calculations."""
    est = {"c1": np.array([0.0, 0.0, 0.0]), "c2": np.array([10.0, 0.0, 0.0])}
    gt = {"c1": np.array([0.0, 0.0, 0.0]), "c2": np.array([10.0, 0.0, 1.0])}

    metrics = evaluate_camera_positions(est, gt, total_cameras=2)
    assert metrics.registered_cameras == 2
    assert metrics.registration_rate_percent == 100.0
    assert metrics.vertical_rmse_m == 0.7071  # sqrt((0^2 + 1^2)/2)
    assert metrics.horizontal_rmse_m == 0.0


def test_evaluate_check_points():
    """Verify independent check point error calculations."""
    est = {"CHK_01": np.array([10.0, 10.0, 5.0])}
    cps = [GCPPoint(id="CHK_01", latitude=10.0, longitude=10.0, altitude=4.0, role=GCPRole.CHECK)]

    metrics = evaluate_check_points(est, cps)
    assert metrics.num_check_points == 1
    assert metrics.vertical_rmse_m == 1.0
    assert metrics.horizontal_rmse_m == 0.0
    assert metrics.rmse_3d_m == 1.0


def test_evaluate_distances():
    """Verify relative distance error and percentage calculation."""
    est = {"a": np.array([0.0, 0.0, 0.0]), "b": np.array([10.0, 0.0, 0.0])}
    kd = KnownDistance(id="d1", point_a=[0.0, 0.0, 0.0], point_b=[9.5, 0.0, 0.0], distance_m=10.0)

    metrics = evaluate_distances(est, [kd])
    assert metrics.num_distances == 1
    assert metrics.mean_absolute_error_m == 0.5
    assert metrics.mean_percentage_error == 5.0


def test_evaluate_surface_distance():
    """Verify point cloud surface distance percentiles (P50, P90, P95, P99)."""
    recon = np.random.normal(0, 0.1, size=(500, 3))
    ref = np.zeros((500, 3))

    metrics = evaluate_surface_distance(recon, ref)
    assert metrics.num_samples == 500
    assert metrics.p50_m > 0
    assert metrics.p95_m >= metrics.p90_m
    assert metrics.p99_m >= metrics.p95_m


def test_evaluate_depth_relative_labeling():
    """Verify Depth Anything V2 outputs are correctly annotated as relative depth."""
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    ref = np.array([[2.0, 4.0], [6.0, 8.0]])

    metrics = evaluate_depth(pred, ref, is_metric=False, model_name="Depth Anything V2")
    assert "Relative depth model" in metrics.model_type
    assert metrics.is_metric is False
    assert metrics.mae_m == 0.0  # Perfectly aligned after relative scale estimation


def test_no_data_claim_asserts_nothing():
    """Regression: the metric-validation endpoint used to fabricate a
    ground-truth pose merely because ``poses.json`` existed, which promoted a
    relative-only run to GPS_GEOREFERENCED with scale_source
    "gps_camera_alignment" and scale_uncertainty 0.0. With no measurements the
    engine must assert nothing beyond a relative reconstruction.
    """
    from app.services.metric_validation import AccuracyFacts, capability_state

    # Nothing measured at all must NOT read as a measured relative model.
    state, reason = capability_state(AccuracyFacts())
    assert state == "NOT_MEASURED"

    state, reason = capability_state(AccuracyFacts(has_relative_measurement=True))
    assert state == "RELATIVE_ONLY"
    assert "not validated" in reason.lower()

    # And the served envelope for a run with no artifact carries no figures.
    from app.services.metric_validation import not_certified_report

    envelope = not_certified_report("run_x", "no artifact")
    assert envelope["accuracy_certified"] is False
    assert envelope["certification_status"].startswith("NOT MEASURED")
    assert envelope["reference_comparison"] is None
    assert envelope["scale_validation"] is None
    assert envelope["accuracy_summary"]["relative_reconstruction"] == "not measured"
    assert "gps" not in json.dumps(envelope).lower()


def test_capability_state_machine():
    """The capability ladder is a pure function of measured facts, and each
    level requires its own evidence (single owner: metric_validation)."""
    from app.services.metric_validation import AccuracyFacts, capability_state

    # 1. Relative only — measured, but nothing independent checked it
    state, reason = capability_state(
        AccuracyFacts(scale_factor=1.0, has_relative_measurement=True)
    )
    assert state == "RELATIVE_ONLY"
    assert "not validated" in reason.lower()

    # 0. Nothing measured at all is its own honest rung
    state, _ = capability_state(AccuracyFacts(scale_factor=1.0))
    assert state == "NOT_MEASURED"

    # 2. Scaled — metric scale calibrated, nothing independent checked
    state, reason = capability_state(AccuracyFacts(scale_factor=1.045))
    assert state == "SCALED"
    assert "not validated" in reason.lower()

    # 3. GPS georeferenced — telemetry correspondence exists, still unvalidated
    state, reason = capability_state(
        AccuracyFacts(scale_factor=1.045, has_georeferencing=True)
    )
    assert state == "GPS_GEOREFERENCED"
    assert "not validated" in reason.lower()

    # 4. Metric validated — an independent reference actually checked it
    state, _ = capability_state(
        AccuracyFacts(scale_factor=1.045, has_georeferencing=True, has_reference=True,
                      reference_validated=True)
    )
    assert state == "METRIC_VALIDATED"

    # Georeferencing alone cannot skip a level: a claim never exceeds evidence.
    state, _ = capability_state(AccuracyFacts(has_reference=True, reference_validated=False))
    assert state == "NOT_MEASURED"


def test_adversarial_unit_mismatch_and_swapped_coords():
    """Adversarial test for swapped lat/lon and unit mismatches."""
    # Swapped lat/lon test
    swapped_lat, swapped_lon = -122.4194, 37.7749  # Lat out of range [-90, 90]
    assert validate_gps_coordinates(swapped_lat, swapped_lon) is False

    # Empty ground truth test
    empty_gt = GroundTruthData()
    assert len(empty_gt.control_points) == 0
    assert len(empty_gt.check_points) == 0
    # An empty ground-truth set validates nothing (and raises no claim).
    from app.services.metric_validation import AccuracyFacts, capability_state

    state, _ = capability_state(AccuracyFacts(has_reference=False))
    assert state == "NOT_MEASURED"
