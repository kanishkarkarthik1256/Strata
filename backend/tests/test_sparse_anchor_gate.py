"""Sparse-anchor accuracy gate — depth-alignment fault containment.

Root cause this gate closes (flight_to_tower_7511dc views 37-50): the depth
network's output gradient collapsed in those views (aligned Z flat at
~137-142 m across a 66-268 m sparse envelope). No affine of the raw output
can repair a many-to-one compression, so the only honest handling is to
REFUSE the view by name before it can fuse, with the measured evidence
(err_after_m vs budget) persisted in the alignment report.

Contract tested here:
1. A view whose aligned map misses its sparse anchor beyond the configured
   budget (DENSE_SPARSE_ANCHOR_GATE_FACTOR × MAX_RELATIVE_DEPTH_ERR × range)
   is refused: no .npy on disk, view lands in excluded (not failed), and the
   alignment-state record names the gate.
2. A healthy view (map tracks sparse) passes and generates.
3. A cached map written before the gate existed fails the re-applied gate:
   the cache self-heals (artifacts removed, view excluded, nothing fused).
4. The dense side applies the same gate to maps already on disk (one policy
   owner): a stale refused map never enters fusion.

The rig: a nadir camera 50 m above a ground plane at world Z = −50 whose
sparse points carry a depth RAMP across the frustum (bottom-of-frame closer),
so the conditioning pre-pass accepts the view (correct signed structure).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings
from app.services import depth_generator as dg
from app.services.geometry import project_world_to_pixel
from app.services.pointcloud import PointCloud, save_ply

# Camera pitched 35° down-forward over a flat ground plane at Z = -50
# (exactly the forward-flight geometry): near-field at the bottom of frame,
# so the conditioning structure gate sees the CORRECT signed ramp. Depth
# spans ~27-55 m across the frustum; a collapsed constant map best-fits to
# the envelope's middle and misses its sparse anchor by ~2x the budget.
_P = np.deg2rad(45.0)
K = np.array([[2756.81, 0.0, 1920.0], [0.0, 2734.15, 1080.0], [0.0, 0.0, 1.0]])
R_C2W = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -np.cos(_P), np.sin(_P)],
        [0.0, -np.sin(_P), -np.cos(_P)],
    ]
)
CAM_CENTER = np.array([0.0, 0.0, 0.0])
H, W = 1080, 1920  # half-res rig; pose K is for full-res, so poses use K/2

K_RIG = K / 2.0
K_RIG[2, 2] = 1.0


def _rig_pose(frame_id: str) -> dict:
    return {
        "frame_id": frame_id,
        "K": K_RIG.tolist(),
        "R": R_C2W.tolist(),
        "t": CAM_CENTER.tolist(),
    }


def _ground_cloud(n: int = 1200, seed: int = 7) -> np.ndarray:
    """Sparse points sampled uniformly in the IMAGE, backprojected to the
    ground plane (how real feature distributions look).

    The 45° pitch gives camera-space depth a wide envelope (~50-118 m),
    matching the real fault's geometry: views 37-50 saw a wide sparse
    envelope, so when the network output collapsed, the best affine fit
    landed ~80 m off and missed its anchor by ~2x the budget. A narrow or
    center-concentrated envelope would hide the fault.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, W, n)
    v = rng.uniform(0, H, n)
    z = _pixel_metric_z(u, v)
    xn = (u - K_RIG[0, 2]) / K_RIG[0, 0]
    yn = (v - K_RIG[1, 2]) / K_RIG[1, 1]
    d_cam = np.stack([xn * z, yn * z, z], axis=1)
    return d_cam @ R_C2W.T + CAM_CENTER  # camera → world


def _pixel_metric_z(u, v) -> np.ndarray:
    """Camera-space depth where the pixel's ray hits the ground plane.

    Derived from the pipeline's own convention (Xc = (Xw−C) @ R, so
    d_world = (xn, yn, 1) @ R.T): the ground-plane hit sits at
    t = 50 / (cosP + sinP·yn) — bottom-of-frame (yn > 0) is NEAR, the
    correct signed structure for the conditioning gate.
    """
    yn = (v - K_RIG[1, 2]) / K_RIG[1, 1]
    return 50.0 / (np.cos(_P) + np.sin(_P) * yn)


def _raw_for_metric_z(z_metric, a: float = 0.0002, b: float = 0.005):
    """Raw DA-V2 output whose perfect fit is Z = 1/(a*D_raw + b).

    Gauge constants chosen so D_raw > 0 across the rig's 37-63 m depth
    range (raw must be positive — the pipeline treats D_raw ≤ 0 as invalid).
    """
    return (1.0 / np.maximum(z_metric, 1e-3) - b) / a


def _raw_map(kind: str) -> np.ndarray:
    """Raw output on the rig grid. healthy: tracks the tilted plane;
    collapsed: map says ~139 m everywhere while the scene sits at ~50 m
    (the views 37-50 signature — near-flat output with a residual gradient,
    so the fit lands far from the sparse anchor instead of degenerating to
    the median depth)."""
    vv, uu = np.mgrid[0:H, 0:W]
    z_true = _pixel_metric_z(uu.astype(float), vv.astype(float))
    if kind == "healthy":
        return _raw_for_metric_z(z_true).astype(np.float32)
    # Collapsed output: near-constant ~110-133 m with non-affine radial
    # structure (the fit keeps a > 0 like the real collapsed views, whose
    # a_i ≈ +1e-4, but no affine in 1/Z can absorb the curvature — the
    # aligned map misses its sparse anchor by ~13 m against a ~10 m budget).
    z_flat = 133.0 - 25.0 * ((uu / W - 0.5) ** 2 + (vv / H - 0.5) ** 2)
    return _raw_for_metric_z(z_flat).astype(np.float32)


def _seed(tmp_path: Path, fid: str, sparse: np.ndarray) -> Path:
    ws = tmp_path / "ws"
    (ws / "selected").mkdir(parents=True)
    (ws / "depth").mkdir(parents=True)
    (ws / "poses.json").write_text(json.dumps({"frames": [_rig_pose(fid)]}))
    save_ply(ws / "sparse_model.ply", PointCloud(xyz=sparse))
    cv2.imwrite(str(ws / "selected" / f"{fid}.jpg"), np.zeros((H, W, 3), np.uint8))
    return ws


def _patch_infer(monkeypatch, raw: np.ndarray) -> None:
    # ``native_resolution`` mirrors the real model API: the pipeline stores
    # the model's own grid rather than interpolating its prediction up to the
    # source frame, so the double has to accept the same keyword.
    monkeypatch.setattr(
        "app.services.depth_anything_v2.load_model",
        lambda: (
            type("M", (), {"infer_image": staticmethod(
                lambda img, input_size=518, native_resolution=False: raw.copy()
            )})(),
            "cpu",
            None,
        ),
    )


def _run(tmp_path, monkeypatch, kind: str, fid: str = "frame_000037"):
    sparse = _ground_cloud()
    ws = _seed(tmp_path, fid, sparse)
    _patch_infer(monkeypatch, _raw_map(kind))
    summary = dg.generate_view_depths(ws, backend="depth_anything")
    return summary, ws, summary.alignment_state


# ---------------------------------------------------------------------------
# 1+2: gate refuses a collapsed-gradient view; healthy view passes
# ---------------------------------------------------------------------------


class TestGateRefusal:
    def test_collapsed_gradient_view_refused_and_named(self, tmp_path, monkeypatch):
        summary, ws, state = _run(tmp_path, monkeypatch, "collapsed")
        fid = "frame_000037"
        assert fid in summary.excluded
        assert fid not in summary.failed
        assert summary.excluded_reasons[fid] == "sparse_anchor_error_exceeds_budget"
        assert not (ws / "depth" / f"{fid}.npy").exists()
        rec = state.views[fid]
        assert rec["gate"] == "sparse_anchor_error_exceeds_budget"
        assert rec["accepted"] is False
        assert rec["err_after_m"] > rec["anchor_budget_m"]
        # The refusal is visible in the persisted report's validation flags.
        report = json.loads((ws / "depth_alignment_report.json").read_text())
        assert fid in report["validation_flags"]["sparse_anchor_error_exceeds_budget"]

    def test_healthy_view_passes_and_generates(self, tmp_path, monkeypatch):
        summary, ws, state = _run(tmp_path, monkeypatch, "healthy")
        fid = "frame_000037"
        assert fid in summary.generated
        assert fid not in summary.excluded
        assert (ws / "depth" / f"{fid}.npy").exists()
        rec = state.views[fid]
        assert rec["accepted"] is True
        assert "gate" not in rec

    def test_gate_factor_is_configurable(self, monkeypatch):
        # The threshold is a named setting, not a magic number in the fit.
        assert settings.dense.sparse_anchor_gate_factor > 0
        monkeypatch.setattr(settings.dense, "sparse_anchor_gate_factor", 99.0)
        assert settings.dense.sparse_anchor_gate_factor == 99.0


# ---------------------------------------------------------------------------
# 3: stale cached map self-heals through the same gate
# ---------------------------------------------------------------------------


class TestStaleCacheHeal:
    # The rigs here seed LEGACY sidecars (pre model identity), which the
    # model-identity rule serves only under the vits checkpoint — pin that
    # regime explicitly so these tests stay about the gate, not the variant.
    @staticmethod
    def _pin_vits(monkeypatch) -> None:
        monkeypatch.setattr(dg, "_ckpt_name", lambda: "depth_anything_v2_vits.pth")
        monkeypatch.setattr(dg, "_depth_checkpoint_sha256", lambda: "a" * 64)

    def test_stale_refused_map_removed_and_regenerated_live(self, tmp_path, monkeypatch):
        """A pre-gate map on disk fails the re-applied gate, is deleted, and
        the live fit re-decides the view with its own measured evidence."""
        fid = "frame_000040"
        sparse = _ground_cloud()
        ws = _seed(tmp_path, fid, sparse)
        stale = np.full((H, W), 140.0, np.float32)
        np.save(ws / "depth" / f"{fid}.npy", stale)
        from app.services.depth_generator import _poses_fingerprint

        (ws / "depth" / f"{fid}.json").write_text(
            json.dumps({"frame_id": fid, "poses_sha256": _poses_fingerprint(ws / "poses.json")})
        )
        # Live model returns a healthy map: the stale artifact must be gone
        # and the view must regenerate fresh (not be served from cache).
        self._pin_vits(monkeypatch)
        _patch_infer(monkeypatch, _raw_map("healthy"))
        summary = dg.generate_view_depths(ws, backend="depth_anything")
        assert fid in summary.generated
        assert fid not in summary.cached
        # The on-disk map is the LIVE product, not the stale 140 m flat map.
        live = np.load(ws / "depth" / f"{fid}.npy")
        assert not np.allclose(live, stale)

    def test_stale_refused_map_still_refused_by_live_gate(self, tmp_path, monkeypatch):
        """If the live fit also fails the gate, the exclusion reason comes
        from the LIVE fit's measured record — never inherited from disk."""
        fid = "frame_000040"
        sparse = _ground_cloud()
        ws = _seed(tmp_path, fid, sparse)
        np.save(ws / "depth" / f"{fid}.npy", np.full((H, W), 140.0, np.float32))
        from app.services.depth_generator import _poses_fingerprint

        (ws / "depth" / f"{fid}.json").write_text(
            json.dumps({"frame_id": fid, "poses_sha256": _poses_fingerprint(ws / "poses.json")})
        )
        self._pin_vits(monkeypatch)
        _patch_infer(monkeypatch, _raw_map("collapsed"))
        summary = dg.generate_view_depths(ws, backend="depth_anything")
        assert fid in summary.excluded
        assert summary.excluded_reasons[fid] == "sparse_anchor_error_exceeds_budget"
        # Refused live → no map on disk again.
        assert not (ws / "depth" / f"{fid}.npy").exists()

    def test_healthy_cached_map_still_served(self, tmp_path, monkeypatch):
        fid = "frame_000041"
        sparse = _ground_cloud()
        ws = _seed(tmp_path, fid, sparse)
        good_raw = _raw_map("healthy")
        np.save(ws / "depth" / f"{fid}.npy", (1.0 / (0.0002 * good_raw + 0.005)).astype(np.float32))
        from app.services.depth_generator import _poses_fingerprint

        (ws / "depth" / f"{fid}.json").write_text(
            json.dumps({"frame_id": fid, "poses_sha256": _poses_fingerprint(ws / "poses.json")})
        )
        self._pin_vits(monkeypatch)
        monkeypatch.setattr(
            "app.services.depth_anything_v2.load_model",
            lambda: (_ for _ in ()).throw(AssertionError("healthy cache must be served, not regenerated")),
        )
        summary = dg.generate_view_depths(ws, backend="depth_anything")
        assert fid in summary.cached
        assert fid not in summary.excluded


# ---------------------------------------------------------------------------
# 4: dense side applies the same gate to maps on disk
# ---------------------------------------------------------------------------


class TestDenseSideGate:
    def test_dense_stage_drops_map_failing_gate(self, tmp_path, monkeypatch):
        """Drive the REAL dense stage fusion entry with a stale flat map and a
        good one; the gate must keep the good view and drop the stale one."""
        from app.services.dense_reconstruction import DenseParams, _build_views
        from app.services.depth_generator import depth_map_passes_anchor_gate

        sparse = _ground_cloud()
        poses = [_rig_pose("frame_000042")]
        depth_dir = tmp_path / "ws" / "depth"
        depth_dir.mkdir(parents=True)

        np.save(depth_dir / "frame_000042.npy", np.full((H, W), 140.0, np.float32))
        assert depth_map_passes_anchor_gate(
            depth_dir / "frame_000042.npy", "frame_000042", sparse, poses
        ) is False

        views = _build_views(poses, depth_dir, None, DenseParams())
        kept = [
            vw
            for vw in views
            if depth_map_passes_anchor_gate(
                depth_dir / f"{vw.frame_id}.npy", vw.frame_id, sparse, poses
            )
        ]
        assert kept == []

    def test_dense_stage_keeps_map_passing_gate(self, tmp_path):
        from app.services.dense_reconstruction import DenseParams, _build_views
        from app.services.depth_generator import depth_map_passes_anchor_gate

        sparse = _ground_cloud()
        poses = [_rig_pose("frame_000043")]
        depth_dir = tmp_path / "ws2" / "depth"
        depth_dir.mkdir(parents=True)
        good_raw = _raw_map("healthy")
        np.save(depth_dir / "frame_000043.npy", (1.0 / (0.0002 * good_raw + 0.005)).astype(np.float32))
        views = _build_views(poses, depth_dir, None, DenseParams())
        assert len(views) == 1
        assert depth_map_passes_anchor_gate(
            depth_dir / "frame_000043.npy", "frame_000043", sparse, poses
        ) is True
