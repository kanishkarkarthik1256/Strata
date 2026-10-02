"""Honesty tests for the depth diagnostics layer.

The 9,987 m stereo bug passed as "Criteria A-G" because the old report
hardcoded the model name and used tautological criteria. These tests pin the
fixed contract:

- provenance comes from the depth sidecars, never a hardcoded label — a
  stereo run must NOT claim Depth Anything;
- the learned model is loaded only for depth_anything runs;
- criteria can actually fail (a junk 9,987 m depth map must be rejected);
- ``multi_view_depth_consistency`` is a real measurement, not a copy of
  ``can_feed_tsdf``.

Reconstruction math is untouched: every test drives only the diagnostics
reader/evaluator over seeded workspaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config.settings import settings
from app.services.depth_diagnostics import run_depth_diagnostics


def _patched_storage(tmp_path: Path) -> None:
    settings.storage.base_path = str(tmp_path)


def _camera() -> list[list[float]]:
    return [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]


def _write_workspace(
    tmp_path: Path,
    job_id: str,
    *,
    backend: str,
    depth_median_m: float,
    n_frames: int = 3,
) -> Path:
    """A minimal but geometrically coherent workspace: cameras looking down
    at a ground plane ~``depth_median_m`` below, sparse points on that plane,
    and depth maps centered on ``depth_median_m`` (junk values simulate the
    SGBM failure mode while staying finite and dense)."""
    _patched_storage(tmp_path)
    workspace = settings.storage.project_dir(job_id)
    (workspace / "selected").mkdir(parents=True, exist_ok=True)
    (workspace / "depth").mkdir(parents=True, exist_ok=True)

    frames = []
    K = np.asarray(_camera(), dtype=np.float64)
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # looking down
    for i in range(n_frames):
        fid = f"frame_{i:06d}"
        img = np.full((480, 640, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(workspace / "selected" / f"{fid}.jpg"), img)
        frames.append({"frame_id": fid, "K": K.tolist(), "R": R.tolist(), "t": [0.3 * i, 0.0, 0.0]})
    (workspace / "poses.json").write_text(json.dumps({"frames": frames}))

    # Sparse ground-plane points at the TRUE scene depth (25 m below the
    # cameras) — independent of any junk depth maps added later. World
    # z = -25; with cameras at the origin every view sees ≈25 m axis depth.
    rng = np.random.default_rng(0)
    pts = rng.uniform([-8, -6, 24.0], [8, 6, 26.0], size=(200, 3))
    xyz = pts @ R  # camera -> world
    from app.services.pointcloud import save_ply, PointCloud

    save_ply(workspace / "sparse_model.ply", PointCloud(xyz=xyz, rgb=np.full((200, 3), 128, dtype=np.uint8)))

    depth = np.full((480, 640), depth_median_m, dtype=np.float32)
    metric = backend == "stereo"
    for f in frames:
        fid = f["frame_id"]
        np.save(workspace / "depth" / f"{fid}.npy", depth)
        (workspace / "depth" / f"{fid}.json").write_text(
            json.dumps(
                {
                    "frame_id": fid,
                    "backend": backend,
                    "model_version": "stereo-sgbm" if backend == "stereo" else "depth_anything_v2_vits.pth",
                    "metric": metric,
                    "valid_ratio": 1.0,
                }
            )
        )
    return workspace


def test_stereo_run_must_not_claim_depth_anything(tmp_path: Path):
    """Regression: provenance comes from sidecars; a stereo run is reported
    as stereo even when a Depth Anything checkpoint exists."""
    workspace = _write_workspace(tmp_path, "diag_stereo1", backend="stereo", depth_median_m=25.0)
    report = run_depth_diagnostics(workspace)
    assert report.backends == {"stereo": 3}
    assert "Stereo" in report.depth_model and "Depth Anything" not in report.depth_model
    # And the per-frame provenance too.
    assert all(f.backend == "stereo" for f in report.frames_diagnostics)
    # Model weights are irrelevant for a stereo run — not claimed as loaded.
    assert report.weights_loaded is False


def test_depth_anything_run_reports_its_model(tmp_path: Path, monkeypatch):
    workspace = _write_workspace(tmp_path, "diag_da1", backend="depth_anything", depth_median_m=25.0)
    ckpt = Path("models/weights/depth_anything_v2_vits.pth")
    monkeypatch.setattr("app.services.depth_diagnostics.find_checkpoint", lambda: ckpt)

    class _StubModel:
        def infer_image(self, img, input_size=518, native_resolution=False):
            # ``native_resolution`` mirrors the real model API: the pipeline
            # stores the model's own grid, so the double must accept it.
            del input_size, native_resolution
            return np.full(img.shape[:2], 0.5, dtype=np.float32)

    monkeypatch.setattr(
        "app.services.depth_diagnostics.load_model", lambda: (_StubModel(), "cpu", ckpt)
    )
    report = run_depth_diagnostics(workspace)
    assert report.backends == {"depth_anything": 3}
    assert "Depth Anything" in report.depth_model
    assert report.weights_loaded is True
    assert all(f.backend == "depth_anything" for f in report.frames_diagnostics)


def test_junk_depth_range_fails_criteria(tmp_path: Path):
    """The 9,987 m failure mode must now be rejected: depth is finite and
    dense, but wildly disagrees with the frame's SfM geometry (≈25 m)."""
    workspace = _write_workspace(tmp_path, "diag_junk1", backend="stereo", depth_median_m=9987.0)
    report = run_depth_diagnostics(workspace)
    # Provenance still honest about what ran...
    assert report.backends == {"stereo": 3}
    # ...but the junk range is caught against the SfM reference.
    assert report.acceptance_criteria["B_depth_range_vs_sfm_geometry"] is False
    assert report.can_feed_tsdf is False


def test_mixed_backends_fail_provenance(tmp_path: Path):
    """Half stereo, half depth_anything sidecars = incoherent provenance."""
    workspace = _write_workspace(tmp_path, "diag_mix1", backend="stereo", depth_median_m=25.0)
    # Flip one sidecar to depth_anything.
    sc = json.loads((workspace / "depth" / "frame_000001.json").read_text())
    sc.update({"backend": "depth_anything", "metric": False})
    (workspace / "depth" / "frame_000001.json").write_text(json.dumps(sc))
    report = run_depth_diagnostics(workspace)
    assert len(report.backends) == 2
    assert report.acceptance_criteria["E_backend_provenance_consistent"] is False
    assert report.can_feed_tsdf is False


def test_cross_view_report_is_a_measurement_not_a_copy(tmp_path: Path):
    """multi_view_depth_consistency carries pair measurements and a verdict —
    it is no longer verbatim can_feed_tsdf."""
    workspace = _write_workspace(tmp_path, "diag_xview1", backend="stereo", depth_median_m=25.0)
    report = run_depth_diagnostics(workspace)
    mv = report.multi_view_depth_consistency
    assert isinstance(mv, dict)
    assert mv["pairs"], "coherent seeded workspace must yield measurable pairs"
    for pair in mv["pairs"]:
        assert pair["n_compared"] > 0
        assert 0.0 <= pair["median_rel_err"] < 1e9
    assert mv["verdict"] in ("PASS", "FAIL", "NOT_EVALUATED")
    # Sanity: with coherent seeded data the measurement must pass.
    assert mv["verdict"] == "PASS"


def test_no_depth_maps_fails_closed(tmp_path: Path):
    workspace = _write_workspace(tmp_path, "diag_empty1", backend="stereo", depth_median_m=25.0)
    for p in (workspace / "depth").glob("*.npy"):
        p.unlink()
    report = run_depth_diagnostics(workspace)
    assert report.can_feed_tsdf is False
    assert report.acceptance_criteria["F_depth_artifacts_present"] is False


def test_excluded_leading_views_do_not_blank_the_audit(tmp_path: Path):
    """Regression (hover-heavy flight, flight_to_tower_7511dc): the first
    registered poses are conditioning-excluded and legitimately have no map.
    Sampling the first poses audited an EMPTY set, so A/B/C/F all failed for
    the wrong reason even though 42 good maps existed. The audit must sample
    the artifact set that is actually on disk."""
    workspace = _write_workspace(
        tmp_path, "diag_excl1", backend="stereo", depth_median_m=25.0, n_frames=6
    )
    depth = workspace / "depth"
    for i in range(4):
        fid = f"frame_{i:06d}"
        (depth / f"{fid}.npy").unlink()
        (depth / f"{fid}.json").unlink()

    report = run_depth_diagnostics(workspace, max_frames=5)

    assert report.acceptance_criteria["F_depth_artifacts_present"] is True
    assert [f.frame_id for f in report.frames_diagnostics] == ["frame_000004", "frame_000005"]
    # Provenance still comes from the real sidecars of the sampled maps.
    assert report.backends == {"stereo": 2}
    assert report.acceptance_criteria["A_coherent_scene_structure"] is True
    assert report.acceptance_criteria["B_depth_range_vs_sfm_geometry"] is True


def test_view_inconsistent_depth_fails_cross_view(tmp_path: Path):
    """Regression for the e2e-discovered failure mode: when one view's depth
    disagrees wildly with its neighbours (e.g. a per-view alignment fallback
    fitted junk for that view), the cross-view criterion must fail the run —
    the old audit waved this through and fused 106 m next to 3 m."""
    workspace = _write_workspace(tmp_path, "diag_xview2", backend="stereo", depth_median_m=25.0)
    # Poison one frame: same finite, dense, hygienic map — but 4× deeper.
    d = np.load(workspace / "depth" / "frame_000001.npy")
    np.save(workspace / "depth" / "frame_000001.npy", d * 4.0)
    report = run_depth_diagnostics(workspace)
    mv = report.multi_view_depth_consistency
    assert mv["verdict"] == "FAIL", mv
    assert report.acceptance_criteria["G_cross_view_consistency"] is False
    assert report.can_feed_tsdf is False


def _write_ramp_workspace(tmp_path: Path, job_id: str, n_frames: int = 3) -> Path:
    """Workspace whose depth maps are a vertical depth ramp: top rows far
    (60 m), bottom rows near (10 m), median 35 m — with sparse points spread
    across that SAME ramp so geometry and depth are perfectly aligned.

    ``_unproject_view`` emits points in row-major image order, so a test
    that samples the unprojected cloud's FIRST N points measures only the
    far top rows — exactly the sampling bias that failed criterion D on
    aligned aerial depth maps (run flight_to_tower_7511dc).
    """
    _patched_storage(tmp_path)
    workspace = settings.storage.project_dir(job_id)
    (workspace / "selected").mkdir(parents=True, exist_ok=True)
    (workspace / "depth").mkdir(parents=True, exist_ok=True)

    h, w = 240, 320
    fx = fy = 400.0
    cx, cy = w / 2.0, h / 2.0
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # looking down

    def ramp(v: np.ndarray | float) -> np.ndarray | float:
        return 60.0 - 50.0 * (np.asarray(v, dtype=np.float64) / (h - 1))

    frames = []
    for i in range(n_frames):
        fid = f"frame_{i:06d}"
        img = np.full((h, w, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(workspace / "selected" / f"{fid}.jpg"), img)
        frames.append({"frame_id": fid, "K": K.tolist(), "R": R.tolist(),
                       "t": [0.3 * i, 0.0, 0.0]})
        depth = np.tile(ramp(np.arange(h))[:, None].astype(np.float32), (1, w))
        np.save(workspace / "depth" / f"{fid}.npy", depth)
        (workspace / "depth" / f"{fid}.json").write_text(json.dumps(
            {"frame_id": fid, "backend": "stereo", "model_version": "stereo-sgbm",
             "metric": True, "valid_ratio": 1.0}))
    (workspace / "poses.json").write_text(json.dumps({"frames": frames}))

    # Sparse points ON the ramp surface, spanning the whole depth range.
    rng = np.random.default_rng(0)
    u = rng.uniform(0, w, size=200)
    v = rng.uniform(0, h, size=200)
    z = ramp(v)
    x_c = (u - cx) / fx * z
    y_c = (v - cy) / fy * z
    x_c = np.clip(x_c, -w, w)
    y_c = np.clip(y_c, -h, h)
    cam = np.stack([x_c, y_c, z], axis=1)
    xyz = cam @ R.T  # camera -> world (t = 0 for frame 0)
    from app.services.pointcloud import save_ply, PointCloud

    save_ply(workspace / "sparse_model.ply",
             PointCloud(xyz=xyz, rgb=np.full((200, 3), 128, dtype=np.uint8)))
    return workspace


def test_criterion_d_samples_whole_frame_not_top_rows(tmp_path: Path):
    """Regression (run flight_to_tower_7511dc): the D criterion measured the
    unprojected cloud's FIRST 5000 points — the top image rows, i.e. the far
    field of aerial footage — and failed perfectly aligned depth maps with a
    1.66x "scale mismatch" while every other criterion passed. The sample
    must span the whole frame: on this ramp fixture the far-field prefix
    measures ~59.5 m against a 35 m SfM median (ratio 1.70 → the old code
    fails D), while the whole-frame median is ~35 m (ratio ~1.0 → PASS)."""
    workspace = _write_ramp_workspace(tmp_path, "diag_ramp1")
    report = run_depth_diagnostics(workspace)

    sf = report.single_frame_cloud
    assert sf is not None
    # The honest whole-frame measurement agrees with the SfM geometry.
    ratio = sf.cloud_depth_median_m / sf.sfm_depth_median_m
    assert abs(ratio - 1.0) <= 0.5, f"ratio={ratio:.3f}"
    assert sf.status == "PASS"
    assert report.acceptance_criteria["D_sensible_single_frame_unprojection"] is True
    # And aligned maps must not be blocked from fusion by criterion D alone.
    assert report.can_feed_tsdf is True


def test_criterion_d_prefix_sample_would_fail_this_fixture(tmp_path: Path):
    """Guard the guard: demonstrate the fixture actually discriminates —
    sampling only the first N unprojected points (the old behaviour) yields
    a far-field median that breaches the 1.5x tolerance on this workspace.
    If this assertion ever fails, the fixture no longer exercises the bias
    and the regression above is vacuous."""
    workspace = _write_ramp_workspace(tmp_path, "diag_ramp2")
    import sys

    sys.path.insert(0, str(workspace.parents[1]))
    d = np.load(workspace / "depth" / "frame_000000.npy").astype(np.float64)
    poses = json.loads((workspace / "poses.json").read_text())["frames"]
    from app.services.depth_diagnostics import _sfm_camera_depths
    from app.services.pointcloud import read_ply

    sparse_xyz = read_ply(workspace / "sparse_model.ply").xyz
    sfm_median = float(np.median(_sfm_camera_depths(poses[0], sparse_xyz, d.shape)))

    # Row-major prefix of the valid map = top rows only.
    flat = d.ravel()
    prefix_median = float(np.median(flat[flat > 0][:5000]))
    assert prefix_median / sfm_median > 1.5, (
        f"fixture no longer discriminates: prefix ratio {prefix_median / sfm_median:.3f}"
    )


def test_sidecar_less_depth_maps_report_unknown_provenance(tmp_path: Path):
    """Externally produced depth maps (no sidecars) are audited on their own
    merits: provenance is reported as unknown — never guessed as Depth
    Anything — and must not claim the model was loaded."""
    workspace = _write_workspace(tmp_path, "diag_nosc1", backend="stereo", depth_median_m=25.0)
    for p in (workspace / "depth").glob("*.json"):
        p.unlink()
    report = run_depth_diagnostics(workspace)
    assert report.backends == {}
    assert "unknown" in report.depth_model
    assert report.weights_loaded is False
    # Provenance is unknown, not incoherent: the geometry checks still run.
    assert report.acceptance_criteria["E_backend_provenance_consistent"] is True
    assert report.can_feed_tsdf is True
