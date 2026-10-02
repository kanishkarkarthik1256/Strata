"""Tests for app.services.metric_validation (validation-only engine).

Covers: registration math, checkpoint statistics, bidirectional
accuracy-vs-coverage separation, error decomposition, and every
certification branch — especially the honest NOT CERTIFIED paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.metric_validation import (
    CERT_P95_M,
    CERT_RMSE_M,
    certify,
    checkpoint_metrics,
    cloud_to_reference,
    coregister_reference,
    error_decomposition,
    registration_from_telemetry,
    relative_consistency,
    threshold_percentages,
    umeyama,
)


def _rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


# ---------------------------------------------------------------- registration

def test_umeyama_recovers_rigid_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(0, 10, size=(500, 3))
    R, t = _rot((0.3, 1, 0.2), 37.0), np.array([5.0, -3.0, 11.0])
    dst = src @ R.T + t
    R_est, t_est, s = umeyama(src, dst, with_scale=True)
    assert abs(s - 1.0) < 1e-9
    assert np.allclose(R_est, R, atol=1e-9)
    assert np.allclose(t_est, t, atol=1e-9)


def test_umeyama_recovers_known_scale():
    """Regression: scale must be σ-weighted (mean-variance denominator), not off by n."""
    rng = np.random.default_rng(1)
    src = rng.normal(0, 10, size=(500, 3))
    s_true = 1.37
    dst = s_true * src + np.array([1.0, 2.0, 3.0])
    _, _, s_est = umeyama(src, dst, with_scale=True)
    assert abs(s_est - s_true) < 1e-6


def test_registration_insufficient_correspondences():
    out = registration_from_telemetry(np.zeros((2, 3)), np.zeros((2, 3)))
    assert out["sufficient"] is False
    assert out["matched_frames"] == 2


def test_registration_exact_correspondence_is_identity():
    rng = np.random.default_rng(2)
    P = rng.normal(0, 20, size=(40, 3))
    out = registration_from_telemetry(P, P.copy())
    assert out["sufficient"] is True
    assert out["scale_is_identity"] is True
    assert out["residual_median_m"] < 1e-6


# ---------------------------------------------------------------- checkpoints

def test_checkpoint_metrics_horizontal_vertical_split():
    recon = np.array([[0, 0, 0], [3, 0, 0], [0, 0, 4]])
    gt = np.zeros((3, 3))
    m = checkpoint_metrics(recon, gt)
    assert m["N"] == 3
    # engine rounds to 4 decimals for reporting
    assert m["horizontal_rmse"] == pytest.approx(np.sqrt(3), rel=1e-3)
    assert m["vertical_rmse"] == pytest.approx(4 / np.sqrt(3), rel=1e-3)
    assert m["median"] == pytest.approx(3.0)  # median of {0, 3, 4}
    assert m["within_1.0m_pct"] == pytest.approx(100 / 3, abs=0.01)  # only the coincident point


def test_checkpoint_metrics_length_mismatch_is_empty():
    assert checkpoint_metrics(np.zeros((3, 3)), np.zeros((2, 3)))["N"] == 0


def test_threshold_percentages_exact():
    d = np.array([0.1, 0.4, 0.9, 1.5, 3.0, 6.0])
    p = threshold_percentages(d)
    # engine rounds to 2 decimals for reporting
    assert p["within_0.25m_pct"] == pytest.approx(100 / 6, abs=0.01)
    assert p["within_1.0m_pct"] == 50.0
    assert p["within_5.0m_pct"] == pytest.approx(5 * 100 / 6, abs=0.01)


# ------------------------------------------------- accuracy vs coverage split

def test_cloud_to_reference_separates_accuracy_from_coverage():
    """Offset recon → accuracy error; missing half the reference → coverage gap.

    The offset must show in STRATA→reference (accuracy), while the missing
    surface shows ONLY in reference→STRATA (coverage) — never as accuracy.
    """
    rng = np.random.default_rng(3)
    # reference: dense plane z=0, x,y in [0,20]², plus an area the recon lacks
    ref_a = rng.uniform(0, 20, size=(20_000, 3)) * [1, 1, 0]
    ref_b = rng.uniform(20, 30, size=(20_000, 3)) * [1, 1, 0]  # uncovered area
    ref = np.vstack([ref_a, ref_b])

    recon = rng.uniform(0, 20, size=(10_000, 3)) * [1, 1, 0] + [0, 0, 0.5]
    out = cloud_to_reference(recon, ref, max_query=5_000, max_ref=30_000)
    acc = out["strata_to_reference_m"]
    cov = out["reference_to_strata_coverage_m"]

    assert acc["median"] == pytest.approx(0.5, abs=0.1)  # the true offset
    assert acc["rmse"] == pytest.approx(0.5, abs=0.15)
    # a third of the reference is >2 m from any recon point
    assert cov["coverage_within_1m_pct"] < 75.0
    # coverage direction must NOT be reported as accuracy anywhere
    assert "coverage" in out["reference_to_strata_coverage_m"]["note"].lower()


# ----------------------------------------------------------------- decomposition

def test_error_decomposition_recovers_signed_offset_and_horiz_error():
    """Vertical offsets ARE recoverable from NN deltas; horizontal shifts over a
    full uniform plane are NOT (a reference point exists directly beneath any
    shifted query, so the horizontal component snaps to ~0 — a known limit of
    NN decomposition, which is why global shift evidence lives in `registration`).
    """
    rng = np.random.default_rng(4)
    ref = rng.uniform(0, 50, size=(80_000, 3)) * [1, 1, 0]
    recon = rng.uniform(0, 50, size=(30_000, 3)) * [1, 1, 0] + [0.0, 0.0, 1.5]
    dec = error_decomposition(recon, ref, sample=8_000, max_ref=60_000)
    assert dec["vertical_signed_median_m"] == pytest.approx(1.5, abs=0.15)
    assert dec["vertical_abs_median_m"] == pytest.approx(1.5, abs=0.15)
    assert dec["horizontal_median_m"] < 0.25  # NN snapping, see docstring
    assert "recon_self_nn_median_m" in dec
    assert dec["recon_points_total"] == len(recon)
    # constant-height cloud → no usable height bands (degenerate bands skipped)
    assert dec["error_by_height_band"] == []


def test_error_decomposition_handles_reference_class():
    rng = np.random.default_rng(5)
    n = 60_000
    ref = rng.uniform(0, 40, size=(n, 3)) * [1, 1, 0]
    cls = np.where(np.arange(n) % 2 == 0, 2, 20)  # ground vs vegetation
    recon = rng.uniform(0, 40, size=(15_000, 3)) * [1, 1, 0]
    dec = error_decomposition(recon, ref, ref_class=cls, sample=6_000, max_ref=60_000)
    assert set(dec["error_by_reference_class"]) == {"2", "20"}


# ---------------------------------------------------------------- certification

def _acc(rmse, p95, q=5_000, cov=99.0):
    return {
        "strata_to_reference_m": {"rmse": rmse, "p95": p95, "query_points": q},
        "reference_coverage_of_recon_bbox_pct": cov,
    }


def test_certify_all_honest_paths():
    reg_ok = {"sufficient": True}
    reg_bad = {"sufficient": False, "reason": "too few matched frames"}

    # no reference at all
    s, r = certify({}, reg_ok, ref_available=False)
    assert s == "NOT CERTIFIED — NO INDEPENDENT REFERENCE"

    # reference but insufficient georeferencing
    s, r = certify(_acc(0.2, 0.4), reg_bad, ref_available=True)
    assert s == "NOT CERTIFIED — INSUFFICIENT GEOREFERENCING"
    assert "too few" in r

    # real reference, above thresholds → never a ≤1m claim
    s, r = certify(_acc(2.35, 3.68), reg_ok, ref_available=True)
    assert s == "VALIDATION AVAILABLE — ABOVE 1m"
    assert "2.35" in r and "3.68" in r

    # passes thresholds + covered → certified, labeled engineering criterion
    s, r = certify(_acc(0.4, 0.9), reg_ok, ref_available=True)
    assert s.startswith("CERTIFIED ≤1m")
    assert "engineering criterion" in r or "RMSE" in r

    # passes accuracy but reference barely covers the footprint → not certified
    s, r = certify(_acc(0.4, 0.9, cov=23.0), reg_ok, ref_available=True)
    assert s == "VALIDATION AVAILABLE — COVERAGE LIMITED"
    assert "23%" in r


def test_certify_thresholds_are_the_documented_engineering_rule():
    assert CERT_RMSE_M == 1.0 and CERT_P95_M == 1.0
    # RMSE pass + P95 fail → above 1m (both required)
    s, _ = certify(_acc(0.5, 2.0), {"sufficient": True}, ref_available=True)
    assert s == "VALIDATION AVAILABLE — ABOVE 1m"


# ---------------------------------------------------------------- relative (A)

def test_relative_consistency_never_claims_absolute(tmp_path):
    """A. relative block reports consistency metrics only — no accuracy fields."""
    import json

    (tmp_path / "reconstruction_report.json").write_text(json.dumps(
        {"mean_reprojection_error_px": 1.2, "bundle_adjustment": {"backend": "pycolmap"}}))
    out = relative_consistency(tmp_path)
    assert out["available"] is True
    assert "accuracy" not in out and "certified" not in out
    assert out["reprojection"]["median_px"] == 1.2


# ------------------------------------------- reference co-registration (§C)

def _bumpy_surface(rng, n, scale=1.0):
    """A non-periodic undulating surface — no self-similarity to alias onto."""
    xy = rng.uniform(0.0, 200.0, size=(n, 2))
    z = (1.3 * np.sin(xy[:, 0] * 0.021 + 0.7)
         + 0.9 * np.cos(xy[:, 1] * 0.034 - 1.1)
         + 0.4 * np.sin((xy[:, 0] + xy[:, 1]) * 0.013))
    return np.column_stack([xy, z * scale])


def test_coregister_recovers_a_large_frame_offset():
    """A held-out reference delivered in its own frame must be REGISTERED.

    Comparing raw coordinates across two frames measures the frame offset, not
    the reconstruction — 227 m of pure datum difference on the real runs, which
    is exactly why every LiDAR run reported "ABOVE 1m" and none ever certified.
    After a gated rigid registration the reported number is the surface error.
    """
    rng = np.random.default_rng(11)
    recon = _bumpy_surface(rng, 25_000)
    ref = _bumpy_surface(rng, 60_000) + np.array([-640.0, 910.0, 226.0])
    al = coregister_reference(recon, ref)
    assert al is not None, "registration returned nothing"
    assert al["gate_passed"], al["gate_reason"]
    assert al["applied"] is True
    # raw comparison is a kilometre of frame difference
    assert al["median_before_m"] > 500.0
    # after registration the SAME surfaces sit on top of each other
    assert al["median_after_m"] < 1.0, al
    # scale is never fitted to a reference
    assert al["scale"] == 1.0
    assert al["applied"] is True


def test_coregister_leaves_an_in_frame_reference_alone():
    """A reference already in the reconstruction's frame must not be moved.

    A fit that does not improve the comparison is discarded for the identity,
    so an in-frame tile is compared exactly as delivered (and never nudged
    into a better-looking number by the registration step).
    """
    rng = np.random.default_rng(12)
    recon = _bumpy_surface(rng, 25_000)
    ref = _bumpy_surface(rng, 40_000)
    al = coregister_reference(recon, ref)
    assert al is not None
    assert al["applied"] is False
    assert al["median_after_m"] == al["median_before_m"]
    assert al["gate_passed"], al["gate_reason"]


def test_coregister_never_manufactures_accuracy_for_an_unrelated_reference():
    """Registration must not turn a bad reference into a good number.

    Two unrelated cloud volumes cannot become the same surface, whatever
    transform is applied, so the reported median must stay at the clouds' own
    spacing — never a sub-metre figure that would certify a run the reference
    never actually validated.
    """
    rng = np.random.default_rng(13)
    recon = rng.uniform(0.0, 400.0, size=(25_000, 3))
    ref = rng.uniform(0.0, 400.0, size=(25_000, 3)) + 5000.0
    al = coregister_reference(recon, ref)
    assert al is not None
    assert al["median_after_m"] > 1.0, al
    assert al["scale"] == 1.0
