"""Per-view depth refusal must measure the BULK, not the residual tail.

The refusal gate asks one question: did the aligned map's depth gradient
collapse (so fusing it would ship a wrong gauge)? It answered with np.std,
which answers a different question — "did ANY pixel mismatch badly" — and a
per-view residual distribution is heavy-tailed by construction (occluded
surfaces, sky, featureless ground mismatch by tens of metres while the bulk
aligns to a few percent).

Measured failure this closes (London_Mission_ca3c5b, 60/60 views refused):
frame_000000 fitted its 112 m scene to 2.9 m median error (0.22x the anchor
budget) — better than runs that are accepted — while the std of the same
residuals sat 2.5-9.4x over the gate.

Contracts pinned here:
1. A view whose bulk is exactly aligned but which carries an occlusion tail
   (4% of landmarks on a surface 0.4x the distance) is ACCEPTED; the same
   fixture is shown to breach the old std rule, so the fixture discriminates.
2. A gradient-collapsed view (aligned Z compressed into ~5 m across a 180 m
   envelope — the flight_to_tower 37-50 signature, slope still positive so
   only this gate can catch it) is still REFUSED, by name.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.services import depth_generator as dg
from app.services.geometry import project_world_to_pixel

H, W = 480, 640
K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
R_C2W = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # looking down
CAM_CENTER = np.array([0.0, 0.0, 0.0])
FID = "frame_000000"

# Gauge of the synthetic model output: Z = 1/(a*D + b).
A_GAUGE, B_GAUGE = 0.0030, 0.0060


def _pose(frame_id: str = FID) -> dict:
    return {
        "frame_id": frame_id,
        "K": K.tolist(),
        "R": R_C2W.tolist(),
        "t": CAM_CENTER.tolist(),
    }


def _sample_cloud(n: int = 6000, seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse landmarks on a WIDE depth envelope (20-200 m, log-uniform).

    Returns (world_xyz, pixel_u, pixel_v) — the pixels are known exactly
    because the points are built by backprojecting them.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, W, n)
    v = rng.uniform(0, H, n)
    z = np.exp(rng.uniform(np.log(20.0), np.log(200.0), n))
    xn = (u - K[0, 2]) / K[0, 0]
    yn = (v - K[1, 2]) / K[1, 1]
    xc = np.stack([xn * z, yn * z, z], axis=1)
    return xc @ R_C2W.T + CAM_CENTER, u, v


def _raw_from_metric(z_map: np.ndarray) -> np.ndarray:
    return (1.0 / np.maximum(z_map, 1e-3) - B_GAUGE) / A_GAUGE


def _raw_map_at_pixels(u: np.ndarray, v: np.ndarray, z_map: np.ndarray) -> np.ndarray:
    raw = np.zeros((H, W), dtype=np.float32)
    raw[np.clip(v, 0, H - 1).astype(int), np.clip(u, 0, W - 1).astype(int)] = _raw_from_metric(z_map)
    return raw


def _occlusion_tail(n: int = 6000, frac: float = 0.04, factor: float = 0.4, seed: int = 11):
    """Exact bulk + a tail of landmarks whose surface sits at ``factor`` x depth."""
    xyz, u, v = _sample_cloud(n=n, seed=seed)
    z_true = np.linalg.norm(xyz - CAM_CENTER, axis=1)  # axis depth == norm here (t=origin)
    z_map = z_true.copy()
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(n, int(frac * n), replace=False)
    z_map[idx] = z_true[idx] * factor
    return xyz, u, v, z_true, z_map


def _collapsed_map(n: int = 6000, span_m: float = 5.0, seed: int = 11):
    """Aligned Z compressed into ``span_m`` across the whole envelope.

    Slope stays positive (the fit keeps a > 0), so the ``a <= 0`` guard does
    not catch it and the uncertainty gate is the only thing standing between
    this map and fusion.
    """
    xyz, u, v = _sample_cloud(n=n, seed=seed)
    z_true = np.linalg.norm(xyz - CAM_CENTER, axis=1)
    z_map = 137.0 + span_m * (z_true - z_true.min()) / (z_true.max() - z_true.min())
    return xyz, u, v, z_map


def _std_statistic(raw: np.ndarray, pose: dict, sparse: np.ndarray) -> float:
    """The retired rule's decision variable: std(residual) * z_center / 0.40.

    Reconstructed here so the tests can assert the fixture actually
    discriminates (>1 means the old rule refused this view).
    """
    _u, _v, z = project_world_to_pixel(
        sparse, np.asarray(pose["R"], float), np.asarray(pose["t"], float),
        np.abs(np.asarray(pose["K"], float)),
    )
    inb = (z > 0.2) & (_u >= 0) & (_u < W) & (_v >= 0) & (_v < H)
    d_s = raw[_v[inb].astype(int), _u[inb].astype(int)]
    z_s = z[inb]
    keep = d_s > 0.01
    d_s, z_s = d_s[keep], z_s[keep]
    a, b = dg._fit_inverse_depth_robust(d_s, z_s)
    resid = (1.0 / z_s) - (a * d_s + b)
    z_center = 0.5 * (float(np.percentile(z_s, 2)) + float(np.percentile(z_s, 98)))
    return float(np.std(resid)) * z_center / 0.40


class TestBulkOverTail:
    def test_exact_bulk_with_occlusion_tail_is_accepted(self):
        xyz, u, v, _z_true, z_map = _occlusion_tail()
        raw = _raw_map_at_pixels(u, v, z_map)
        state = dg.DepthAlignmentState()
        depth, aligned = dg._scale_depth_to_sparse(raw, _pose(), xyz, alignment_state=state)

        assert aligned is True
        assert state.views[FID]["accepted"] is True
        assert "gate" not in state.views[FID]
        # The map is metric where it matters: the aligned depth at the bulk
        # landmarks tracks the sparse geometry.
        assert state.views[FID]["err_after_m"] < 0.05 * 112.0

    def test_the_same_fixture_breaches_the_retired_std_rule(self):
        """Guard the guard: without this the acceptance above is vacuous."""
        xyz, u, v, _z_true, z_map = _occlusion_tail()
        raw = _raw_map_at_pixels(u, v, z_map)
        assert _std_statistic(raw, _pose(), xyz) > 1.0

    def test_accepted_fit_is_refined_against_the_bulk(self):
        """Acceptance is not indifference: the accepted fit still reports the
        in-sample landmark error it achieved on the bulk."""
        xyz, _u, _v, z_true, z_map = _occlusion_tail()
        raw = _raw_map_at_pixels(_u, _v, z_map)
        state = dg.DepthAlignmentState()
        _depth, aligned = dg._scale_depth_to_sparse(raw, _pose(), xyz, state)

        assert aligned is True
        rec = state.views[FID]
        assert rec["err_after_m"] is not None
        # Well inside the run's own depth-uncertainty budget for 112 m scenes.
        assert rec["err_after_m"] < 0.08 * float(np.median(z_true)) * 1.5

    def test_refusal_gate_reaches_the_persisted_report(self):
        """A refused view must be traceable after the run: the named gate lands
        in both the per-view record and the report's validation flags."""
        xyz, u, v, z_map = _collapsed_map()
        raw = _raw_map_at_pixels(u, v, z_map)
        state = dg.DepthAlignmentState()
        dg._scale_depth_to_sparse(raw, _pose(), xyz, state)

        report = state.to_report()
        rec = next(r for r in report["per_view"] if r["frame_id"] == FID)
        assert rec["gate"] == "fit_uncertainty_exceeds_budget"
        assert FID in report["validation_flags"]["fit_uncertainty_exceeds_budget"]
        assert FID in report["validation_flags"]["unstable_affine_fit"]


class TestCollapseStillRefused:
    def test_compressed_gradient_view_refused_by_name(self):
        xyz, u, v, z_map = _collapsed_map()
        raw = _raw_map_at_pixels(u, v, z_map)
        state = dg.DepthAlignmentState()
        _depth, aligned = dg._scale_depth_to_sparse(raw, _pose(), xyz, alignment_state=state)

        assert aligned is False, "a 5 m span across a 180 m envelope must not fuse"
        rec = state.views[FID]
        assert rec["accepted"] is False
        assert rec["gate"] == "fit_uncertainty_exceeds_budget"
        assert rec["fit_scale_z_m"] > rec["uncertainty_budget_m"]

    def test_depth_stage_failure_names_the_exclusion_reason(self, tmp_path: Path, monkeypatch):
        """Regression (London_Mission_ca3c5b): when nothing generated and no
        view FAILED — every view was excluded by name — the stage error said
        "dominant_failure=none recorded", hiding the entire diagnosis. The
        message must report the exclusion reasons it actually recorded."""
        from app.services import pipeline_orchestrator as orch
        from app.services.depth_generator import DepthSummary

        summary = DepthSummary(backend="depth_anything")
        summary.excluded = [f"frame_{i:06d}" for i in range(60)]
        summary.excluded_reasons = {fid: "fit_uncertainty_exceeds_budget" for fid in summary.excluded}
        monkeypatch.setattr(orch, "generate_view_depths", lambda *a, **k: summary)

        request = SimpleNamespace(
            depth_backend="depth_anything", frame_stride=1, max_depth_views=200,
            stereo=None, refine=None,
        )
        state = orch.StageState(name="depth")
        with pytest.raises(ValueError) as excinfo:
            orch._stage_depth("job", request, tmp_path, state, lambda *a, **k: None)

        message = str(excinfo.value)
        assert "excluded=60" in message
        assert "dominant_reason=[('fit_uncertainty_exceeds_budget', 60)]" in message
        assert "none recorded" not in message
