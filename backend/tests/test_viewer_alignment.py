"""Viewer alignment regression tests.

The alignment is a presentation-only rigid transform: the authoritative
reconstruction artifacts must never be modified, distances must be preserved
exactly, and the aligned ground must be horizontal in the viewer frame.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.services import viewer_alignment as va


def _rotation_about(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@pytest.fixture()
def synthetic_run(tmp_path: Path) -> Path:
    """A tiny run: tilted ground plane + cameras whose up vector agrees."""
    rng = np.random.default_rng(42)
    # world where "up" is tilted 30 deg from +Z toward +X
    Rtilt = _rotation_about([0, 1, 0], np.deg2rad(30))
    up_world = Rtilt @ np.array([0, 0, 1.0])

    n_ground = 4000
    ground = rng.uniform(-10, 10, size=(n_ground, 2))
    ground_pts = np.column_stack([ground, np.full(n_ground, 0.0) + rng.normal(0, 0.02, n_ground)])
    ground_pts = ground_pts @ Rtilt.T + np.array([0, 0, 2.0])

    # a few elevated blobs (buildings)
    blob = rng.uniform(-2, 2, size=(500, 3)) + np.array([0, 0, 6.0])
    blob = blob @ Rtilt.T + np.array([0, 0, 2.0])
    pts = np.vstack([ground_pts, blob])

    # corridor along world X: cameras above the ground, up ≈ up_world
    cameras = []
    frames = []
    fwd_ref = up_world + 0.25 * (Rtilt @ np.array([1.0, 0, 0]))  # oblique, not exactly -up
    fwd_ref /= np.linalg.norm(fwd_ref)
    for i in range(20):
        c = np.array([-8 + i * 0.8, 0.3 * np.sin(i), 8.0]) @ Rtilt.T + np.array([0, 0, 2.0])
        cameras.append(c)
        # build a camera rotation whose up column maps to up_world
        fwd = fwd_ref
        right = np.cross(up_world, fwd)
        right /= np.linalg.norm(right)
        upc = np.cross(fwd, right)
        R_c2w = np.column_stack([right, upc, fwd])  # X_world = R X_cam + C
        frames.append(
            {
                "frame_id": f"frame_{i:06d}",
                "K": np.eye(3).tolist(),
                "R": R_c2w.tolist(),
                "t": c.tolist(),
            }
        )

    run = tmp_path / "run_test"
    run.mkdir()
    (run / "poses.json").write_text(json.dumps({"frames": frames}))

    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(run / "dense_model.ply"), pc)
    dense_dir = run / "dense"
    dense_dir.mkdir()
    o3d.io.write_point_cloud(str(dense_dir / "dense_model.ply"), pc)
    return run


def test_rigid_transform_preserves_distances(synthetic_run: Path):
    payload = va.compute_alignment(synthetic_run)
    R = np.asarray(payload["R_view"])
    t = np.asarray(payload["t_view"])
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    assert np.max(np.abs(R @ R.T - np.eye(3))) < 1e-9
    assert payload["scale"] == 1.0

    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(2000, 3))
    A = X @ R.T + t
    d0 = np.linalg.norm(X[1:] - X[:-1], axis=1)
    d1 = np.linalg.norm(A[1:] - A[:-1], axis=1)
    assert np.max(np.abs(d1 - d0) / np.maximum(d0, 1e-12)) < 1e-9


def test_ground_is_horizontal_after_transform(synthetic_run: Path):
    payload = va.compute_alignment(synthetic_run)
    R = np.asarray(payload["R_view"])
    t = np.asarray(payload["t_view"])
    # the estimated ground normal must map to ±Y (we orient it toward +Y)
    n = np.asarray(payload["ground_normal_world"])
    n_view = R @ n
    assert abs(n_view[0]) < 1e-6 and abs(n_view[2]) < 1e-6 and n_view[1] > 0.99


def test_corridor_aligns_with_view_x(synthetic_run: Path):
    payload = va.compute_alignment(synthetic_run)
    R = np.asarray(payload["R_view"])
    h = np.asarray(payload["horizontal_direction_world"])
    h_view = R @ h
    assert h_view[0] > 0.99 and abs(h_view[1]) < 1e-6 and abs(h_view[2]) < 1e-6
    assert payload["confidence"] in {"medium", "high"}


def test_alignment_json_written_and_cached(synthetic_run: Path):
    payload = va.compute_alignment(synthetic_run)
    path = synthetic_run / "viewer_alignment.json"
    assert path.is_file()
    doc = json.loads(path.read_text())
    assert doc["R_view"] == payload["R_view"]
    # second call serves the cache (no recompute)
    again = va.compute_alignment(synthetic_run)
    assert again["created_at"] == payload["created_at"]


def _nadir_run(run_dir: Path, up_world: np.ndarray, n_cams: int = 12, georef: bool = False) -> np.ndarray:
    """A nadir survey over a ground plane whose true vertical is ``up_world``.

    The cameras look straight down (optical axis = -up), so the OLD prior
    (mean of -Y_cam, the top-of-frame direction) is a HORIZONTAL ground
    direction ~90 deg away from the truth — the measured bug on real nadir
    missions. Returns the truth so callers can compare against it.
    """
    rng = np.random.default_rng(3)
    up = up_world / np.linalg.norm(up_world)
    axis = -up  # camera looks down
    right = np.cross(up, np.array([1.0, 1.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(axis, right)
    R_c2w = np.column_stack([right, down, axis])  # columns: X,Y,Z axes in world

    xy = rng.uniform(-40, 40, size=(6000, 2))
    ground = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])
    # build the ground in a frame where +Z is the true up, then rotate it
    Rz = _frame_with_up(up)
    pts = ground @ Rz.T + up * 0.0
    blob = up * -3.0 + rng.uniform(-6, 6, size=(400, 3)) @ Rz.T
    pts = np.vstack([pts, blob])

    frames = []
    for i in range(n_cams):
        c = pts.mean(0) + up * 60.0 + right * (i - n_cams / 2) * 3.0
        frames.append({"frame_id": f"frame_{i:06d}", "R": R_c2w.tolist(), "t": c.tolist()})
    (run_dir / "poses.json").write_text(json.dumps({"frames": frames}))

    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    dense = run_dir / "dense"
    dense.mkdir(exist_ok=True)
    o3d.io.write_point_cloud(str(dense / "dense_model.ply"), pc)

    if georef:
        # georef/alignment.json maps the artifact frame to ENU; ENU up is `up`.
        e1 = np.cross(np.array([0.0, 0.0, 1.0]), up)
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(np.array([1.0, 0.0, 0.0]), up)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2] = e1, e2, up
        g = run_dir / "georef"
        g.mkdir(exist_ok=True)
        (g / "alignment.json").write_text(json.dumps({"scale": 1.0, "matrix": M.tolist()}))
    return up


def _frame_with_up(up: np.ndarray) -> np.ndarray:
    """Rotation whose third column is ``up`` (so +Z maps to the true vertical)."""
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), up)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(np.array([1.0, 0.0, 0.0]), up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return np.column_stack([e1, e2, up])


def test_enu_vertical_levels_a_nadir_run_whose_camera_prior_is_horizontal(tmp_path: Path):
    """The regression the real missions hit.

    For nadir imagery the fleet's -Y_cam is a HORIZONTAL direction, so the old
    up prior was ~90 deg from the vertical and the viewer stood the model on
    its edge. With a telemetry ENU frame available, the vertical is exact.
    """
    run = tmp_path / "run_enu"
    run.mkdir()
    true_up = np.array([0.2, 0.1, 0.97])
    _nadir_run(run, true_up, georef=True)
    payload = va.compute_alignment(run)

    assert payload["leveling"]["up_prior_source"] == "enu_vertical"
    assert payload["leveling"]["confidence"] == "high"
    n = np.asarray(payload["ground_normal_world"])
    angle = np.degrees(np.arccos(np.clip(abs(n @ (true_up / np.linalg.norm(true_up))), -1, 1)))
    assert angle < 5.0, f"ground normal {angle:.1f} deg from the true vertical"

    # ...and prove the old prior would have been wrong by a lot on this rig
    frames = json.loads((run / "poses.json").read_text())["frames"]
    old_prior = -np.asarray([f["R"] for f in frames])[:, :, 1].mean(0)
    old_prior /= np.linalg.norm(old_prior)
    old_angle = np.degrees(
        np.arccos(np.clip(abs(old_prior @ (true_up / np.linalg.norm(true_up))), -1, 1))
    )
    assert old_angle > 45.0


def test_enu_vertical_survives_a_scaled_georef_transform(tmp_path: Path):
    """A run placed with a measured scale stores a scaled rotation, not a rigid one.

    Rejecting it (|det - 1| > 1e-3) silently dropped the ENU vertical on
    flight_to_tower_7511dc (det 0.9667) and fell back to a camera assumption
    that was 109 deg from the truth.
    """
    run = tmp_path / "run_scaled_enu"
    run.mkdir()
    true_up = np.array([0.1, 0.2, 0.97])
    _nadir_run(run, true_up, georef=True)
    doc = json.loads((run / "georef" / "alignment.json").read_text())
    m = np.asarray(doc["matrix"], dtype=np.float64)
    m[:3, :3] *= 0.9888  # the uniform scale a measured placement carries
    (run / "georef" / "alignment.json").write_text(json.dumps({"scale": 0.9888, "matrix": m.tolist()}))

    payload = va.compute_alignment(run)
    assert payload["leveling"]["up_prior_source"] == "enu_vertical"
    n = np.asarray(payload["ground_normal_world"])
    angle = np.degrees(np.arccos(np.clip(abs(n @ (true_up / np.linalg.norm(true_up))), -1, 1)))
    assert angle < 5.0


def test_nadir_run_without_telemetry_uses_the_surface_the_fleet_looks_at(tmp_path: Path):
    run = tmp_path / "run_nadir"
    run.mkdir()
    true_up = np.array([0.15, -0.25, 0.95])
    _nadir_run(run, true_up, georef=False)
    payload = va.compute_alignment(run)

    assert payload["leveling"]["up_prior_source"] == "camera_facing_surface"
    n = np.asarray(payload["ground_normal_world"])
    angle = np.degrees(np.arccos(np.clip(abs(n @ (true_up / np.linalg.norm(true_up))), -1, 1)))
    assert angle < 15.0, f"ground normal {angle:.1f} deg from the true vertical"


def test_estimated_plane_normal_sign_follows_the_prior(tmp_path: Path):
    """The SVD normal's sign is arbitrary; the prior fixes it.

    Before canonicalisation a perfectly level band could be rejected with a
    reported agreement of ~-1 purely because the eigendecomposition chose the
    opposite sign (measured: agreement -0.8032 on furnerhem).
    """
    run = tmp_path / "run_sign"
    run.mkdir()
    _nadir_run(run, np.array([0.0, 0.0, 1.0]))
    pts = va._load_points(run)
    prior = np.array([0.0, 0.0, -1.0])  # deliberately anti-parallel
    n, evidence = va._estimate_ground(pts, prior, {})
    assert n is not None and evidence["method"] == "geometry_plane_fit"
    assert float(n @ prior) > 0.99  # sign taken from the prior, never negative
    assert evidence["camera_up_agreement"] > 0


def test_stale_alignment_version_is_recomputed(tmp_path: Path):
    """A cached payload from the old (wrong-up) schema must not be served."""
    run = tmp_path / "run_stale"
    run.mkdir()
    _nadir_run(run, np.array([0.0, 0.0, 1.0]), georef=True)
    stale = {
        "version": 1,
        "run_id": run.name,
        "R_view": [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        "t_view": [0.0, 0.0, 0.0],
    }
    (run / "viewer_alignment.json").write_text(json.dumps(stale))
    payload = va.compute_alignment(run)
    assert payload["version"] == va._ALIGNMENT_VERSION
    assert payload["R_view"] != stale["R_view"]


def test_alignment_recomputes_when_the_geometry_is_rebuilt(tmp_path: Path):
    """A cached transform must not outlive the geometry it was fitted to.

    A re-run (or a healed sparse stage) changes poses/sparse/dense; serving the
    previous run's R_view would silently leave the rebuilt model unlevelled.
    """
    run = tmp_path / "run_inputs"
    run.mkdir()
    true_up = np.array([0.0, 0.0, 1.0])
    _nadir_run(run, true_up, georef=True)
    first = va.compute_alignment(run)
    import time as _time

    _time.sleep(0.01)
    again = va.compute_alignment(run)
    assert again["created_at"] == first["created_at"]  # unchanged inputs → cache hit

    # rebuild the reconstruction artifacts (a healed/regenerated run)
    time_slept = _time.time()
    while _time.time() - time_slept < 0.02:
        pass
    _nadir_run(run, true_up, georef=True)
    assert va.compute_alignment(run)["created_at"] != first["created_at"]


def test_identity_fallback_for_empty_run(tmp_path: Path):
    run = tmp_path / "run_empty"
    run.mkdir()
    payload = va.compute_alignment(run)
    assert payload["method"] == "identity_fallback"
    assert payload["R_view"] == np.eye(3).tolist()
    assert payload["scale"] == 1.0
    assert "fallback_reason" in payload


def test_authoritative_artifacts_unmodified(synthetic_run: Path):
    dense = synthetic_run / "dense" / "dense_model.ply"
    before = dense.read_bytes()
    poses = synthetic_run / "poses.json"
    poses_before = poses.read_bytes()
    va.compute_alignment(synthetic_run)
    assert dense.read_bytes() == before
    assert poses.read_bytes() == poses_before
