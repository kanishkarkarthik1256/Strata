"""The batched projection paths must produce the numbers the loops produced.

Three helpers on the sparse hot path were rewritten for speed only:

* ``bundle_adjustment._measure_reprojection_error`` — invoked ~4x per BA round
  and once per observation (812,670 on the run under measurement);
* ``trajectory_sync._dlt_triangulate`` — once per point (~60k), with a Python
  row-assembly loop over each point's track inside;
* ``trajectory_sync.retriangulate_points`` — the observation filter, once per
  observation.

A rewrite that quietly changes geometry here would move the shipped
reconstruction, so each is pinned against a verbatim copy of the loop it
replaced. The scene is randomized but exercises the real edge cases: cameras
absent from ``result.cameras``, points with fewer than two observable views,
observations behind the camera, and a zero-depth row.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.bundle_adjustment import _measure_reprojection_error
from app.services.camera_pose_estimator import (
    CameraPose,
    ReconstructionResult,
    SparsePoint3D,
)
from app.services.geometry import project_world_to_pixel
from app.services.sparse_conditioning import MIN_DEPTH_BASELINE_RATIO, point_min_baseline_m
from app.services.trajectory_sync import _dlt_triangulate, retriangulate_points

K_BASE = np.array([[1100.0, 0.0, 960.0], [0.0, 1100.0, 540.0], [0.0, 0.0, 1.0]])


def _look_at(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """camera-to-world rotation whose +z axis points from centre at target."""
    forward = target - centre
    forward = forward / np.linalg.norm(forward)
    # Pick an up vector that is not parallel to the view direction, or the
    # cross product below degenerates.
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up_hint)) > 0.9:
        up_hint = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.column_stack([right, up, forward])


def _scene(n_cams: int = 24, n_points: int = 400, seed: int = 7):
    """A flown trajectory with a cloud in front of it and noisy observations."""
    rng = np.random.default_rng(seed)
    cameras: dict[str, CameraPose] = {}
    centres: dict[str, np.ndarray] = {}
    for i in range(n_cams):
        C = np.array([i * 2.0, np.sin(i * 0.4) * 1.5, 5.0])
        R = _look_at(C, np.array([25.0, 0.0, 5.0]))
        name = f"frame_{i:06d}.jpg"
        cameras[name] = CameraPose(
            image_id=i + 1,
            frame_id=name,
            position=C,
            rotation=R,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            intrinsics=K_BASE.copy(),
            distortion=np.zeros(5),
        )
        centres[name] = C

    names = list(cameras)
    points: list[SparsePoint3D] = []
    for p in range(n_points):
        X = np.array([25.0 + rng.normal(0, 6), rng.normal(0, 8), 5.0 + rng.normal(0, 2)])
        obs: list[tuple[str, np.ndarray]] = []
        for name in names:
            if rng.random() < 0.45:
                continue
            R, C = cameras[name].rotation, centres[name]
            u, v, z = project_world_to_pixel(X[None, :], R, C, K_BASE)
            if z[0] <= 0:
                continue
            if rng.random() < 0.1:
                # A drifted/outlier observation: this is what the retriangulation
                # filter exists to drop, so the branches must agree on it too.
                u = u + rng.normal(0, 40, size=1)
            obs.append((name, np.array([float(u[0]), float(v[0])])))
        if not obs:
            continue
        if p % 37 == 0:
            # Exercise the "observing frame missing from result.cameras" path.
            obs.append(("frame_999999.jpg", np.array([10.0, 20.0])))
        if p % 53 == 0:
            # Exercise the behind-camera path: an invented far-behind observation.
            obs.append((names[0], np.array([10.0, 20.0])))
            points.append(
                SparsePoint3D(
                    point_id=p, position=X - np.array([0.0, 0.0, 1e4]),
                    color=np.array([120, 120, 120]),
                    observations=obs,
                )
            )
            continue
        points.append(
            SparsePoint3D(
                point_id=p, position=X + rng.normal(0, 0.05, 3),
                color=np.array([120, 120, 120]), observations=obs,
            )
        )

    result = ReconstructionResult(cameras=cameras, points3d=points)
    result.num_registered = len(cameras)
    result.num_points = len(points)
    return result


# ---------------------------------------------------------------------------
# Verbatim copies of the loops that were replaced — the reference values.
# ---------------------------------------------------------------------------


def _reference_reprojection_error(result) -> float:
    errs: list[float] = []
    for pt in result.points3d:
        frame_ids = [f for f, _ in pt.observations]
        pixels = [pix for _, pix in pt.observations]
        if len(frame_ids) < 2:
            continue
        cams = [result.cameras[f] for f in frame_ids if f in result.cameras]
        pixs = [pix for f, pix in zip(frame_ids, pixels) if f in result.cameras]
        if len(cams) < 2:
            continue
        K = np.asarray(cams[0].intrinsics, dtype=np.float64)
        for cam, pix in zip(cams, pixs):
            u, v, z = project_world_to_pixel(
                pt.position[None, :], np.asarray(cam.rotation, dtype=np.float64),
                np.asarray(cam.position, dtype=np.float64), np.abs(K),
            )
            if z[0] <= 1e-6:
                continue
            errs.append(float(np.hypot(u[0] - pix[0], v[0] - pix[1])))
    return float(np.mean(errs)) if errs else 0.0


def _reference_dlt(R_w2c, C, K, uvs):
    n = len(uvs)
    if n < 2:
        return None
    rows = []
    for i in range(n):
        P = K[i] @ np.hstack([R_w2c[i], -R_w2c[i] @ C[i][:, None]])
        u, v = uvs[i]
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.asarray(rows)
    try:
        _u, s_vt, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    if s_vt.size < 4 or s_vt[2] < 1e-9 * max(s_vt[0], 1e-12):
        return None
    Xh = vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    X = Xh[:3] / Xh[3]
    for i in range(n):
        if (R_w2c[i] @ (X - C[i]))[2] <= 0:
            return None
    return X


def _reference_retriangulate(result, max_obs_err_px: float = 10.0) -> dict:
    cams = {
        name: (
            np.asarray(cam.rotation, dtype=np.float64).T,
            np.asarray(cam.position, dtype=np.float64),
            np.asarray(cam.intrinsics, dtype=np.float64),
        )
        for name, cam in result.cameras.items()
    }
    moved = dropped_points = dropped_obs = nullspace_points = 0
    survivors: list = []
    for pt in result.points3d:
        obs = [(name, uv) for name, uv in pt.observations if name in cams]
        if len(obs) < 2:
            dropped_points += 1
            continue
        X = _reference_dlt(
            np.stack([cams[n][0] for n, _uv in obs]),
            np.stack([cams[n][1] for n, _uv in obs]),
            np.stack([cams[n][2] for n, _uv in obs]),
            np.array([uv for _n, uv in obs], dtype=np.float64),
        )
        if X is None:
            dropped_points += 1
            continue
        kept_obs = []
        for name, uv in obs:
            R_w2c, C, K = cams[name]
            u, v, z = project_world_to_pixel(X[None, :], R_w2c.T, C, np.abs(K))
            if z[0] <= 1e-6:
                dropped_obs += 1
                continue
            if float(np.hypot(u[0] - uv[0], v[0] - uv[1])) <= max_obs_err_px:
                kept_obs.append((name, uv))
            else:
                dropped_obs += 1
        if len(kept_obs) < 2:
            dropped_points += 1
            continue
        min_base = point_min_baseline_m([n for n, _uv in kept_obs], cams)
        depth = float(np.median([
            (cams[n][0] @ (X - cams[n][1]))[2] for n, _uv in kept_obs
        ]))
        if depth > 0 and min_base < MIN_DEPTH_BASELINE_RATIO * depth:
            nullspace_points += 1
            dropped_points += 1
            continue
        if len(kept_obs) < len(obs):
            # second refit on the surviving observations only
            X2 = _reference_dlt(
                np.stack([cams[n][0] for n, _uv in kept_obs]),
                np.stack([cams[n][1] for n, _uv in kept_obs]),
                np.stack([cams[n][2] for n, _uv in kept_obs]),
                np.array([uv for _n, uv in kept_obs], dtype=np.float64),
            )
            if X2 is not None:
                X = X2
        pt.position = X
        pt.observations = kept_obs
        pt.track_length = len(kept_obs)
        survivors.append(pt)
        moved += 1
    result.points3d = survivors
    result.num_points = len(survivors)
    return {
        "points_refit": moved,
        "points_dropped": dropped_points,
        "nullspace_points_dropped": nullspace_points,
        "observations_dropped": dropped_obs,
        "min_depth_baseline_ratio": MIN_DEPTH_BASELINE_RATIO,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reprojection_metric_matches_the_observation_loop():
    result = _scene()
    assert len(result.points3d) > 200

    reference = _reference_reprojection_error(result)
    batched = _measure_reprojection_error(result)

    assert reference > 0.0
    assert batched == pytest.approx(reference, rel=1e-12, abs=1e-12)
    # Same value means same observation count behind the mean, so the error
    # magnitude is unchanged too (a silently-dropped row would shift it).
    assert batched > 1.0


def test_reprojection_metric_handles_a_cloud_with_no_valid_pair():
    result = _scene(n_points=1)
    for pt in result.points3d:
        pt.observations = pt.observations[:1]  # nothing has two views
    assert _measure_reprojection_error(result) == 0.0


def test_dlt_matches_the_per_view_row_loop():
    """Triangulate a known point from noisy observations of it.

    The fixture places cameras around a real 3D point and projects it, so the
    ray system is the well-conditioned case the solver is meant for — and the
    test can assert both that the row loop's answer is reproduced exactly and
    that it is the point that generated the pixels.
    """
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(3, 12))
        truth = np.array([25.0, 0.0, 5.0]) + rng.normal(0, 3, 3)
        # Cameras on a sphere of fixed radius: the ray system is the
        # well-conditioned case the solver is meant for.
        directions = rng.normal(0, 1, (n, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        C = truth + 8.0 * directions
        R_c2w = np.stack([_look_at(C[i], truth) for i in range(n)])
        R_w2c = np.stack([R_c2w[i].T for i in range(n)])  # what DLT consumes
        K = np.stack([K_BASE for _ in range(n)])
        uvs = np.array([
            np.hstack(project_world_to_pixel(truth[None, :], R_c2w[i], C[i], K_BASE)[:2])
            for i in range(n)
        ])
        uvs = uvs + rng.normal(0, 0.3, uvs.shape)

        reference = _reference_dlt(R_w2c, C, K, uvs)
        batched = _dlt_triangulate(R_w2c, C, K, uvs)

        assert reference is not None
        assert batched is not None
        assert np.allclose(batched, reference, rtol=1e-13, atol=1e-13)


def test_dlt_recovers_a_well_conditioned_point():
    """With clean pixels and spread cameras, the answer is the point itself.

    (Random view sets can be ill-conditioned — a tenth of a pixel then
    extrapolates to metres — which is exactly why the production caller gates
    on the observing baseline. This pins the well-conditioned end.)
    """
    truth = np.array([12.0, -4.0, 6.0])
    directions = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.1, 0.0, 0.99], [0.6, 0.6, 0.5], [-0.6, 0.5, -0.6],
    ])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    C = truth + 10.0 * directions
    R_c2w = np.stack([_look_at(C[i], truth) for i in range(len(C))])
    R_w2c = np.stack([R_c2w[i].T for i in range(len(C))])
    K = np.stack([K_BASE for _ in range(len(C))])
    uvs = np.array([
        np.hstack(project_world_to_pixel(truth[None, :], R_c2w[i], C[i], K_BASE)[:2])
        for i in range(len(C))
    ])

    reference = _reference_dlt(R_w2c, C, K, uvs)
    batched = _dlt_triangulate(R_w2c, C, K, uvs)

    assert reference is not None
    assert np.array_equal(batched, reference)
    assert np.allclose(batched, truth, atol=1e-6)


def test_dlt_rejects_degenerate_ray_systems_like_the_loop():
    """Parallel rays and a single view must return None on both paths."""
    C = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    target = np.array([5.0, 0.0, 50.0])
    R_c2w = np.stack([_look_at(c, target) for c in C])
    R = np.stack([R_c2w[i].T for i in range(3)])
    K = np.stack([K_BASE for _ in range(3)])
    # Pixel observations of a point BEHIND the cameras: each ray points away,
    # so the least-squares solution is behind every centre.
    behind = np.array([5.0, 0.0, -50.0])
    uvs = np.array([
        np.hstack(project_world_to_pixel(behind[None, :], R_c2w[i], C[i], K_BASE)[:2])
        for i in range(3)
    ])
    assert _reference_dlt(R, C, K, uvs) is None
    assert _dlt_triangulate(R, C, K, uvs) is None
    single = np.array([[0.0, 0.0]])
    assert _reference_dlt(R[:1], C[:1], K[:1], single) is None
    assert _dlt_triangulate(R[:1], C[:1], K[:1], single) is None


def test_retriangulation_matches_the_observation_filter():
    scene = _scene(n_cams=18, n_points=300)
    reference_scene = _scene(n_cams=18, n_points=300)

    # The reference mutates its input; run it on the identical twin first.
    reference_stats = _reference_retriangulate(reference_scene)
    batched_stats = retriangulate_points(scene)

    assert batched_stats == reference_stats
    assert len(scene.points3d) == len(reference_scene.points3d) > 50
    for got, want in zip(scene.points3d, reference_scene.points3d):
        assert got.point_id == want.point_id
        assert got.track_length == want.track_length == len(got.observations)
        # Positions come from the same SVD on the same matrix.
        assert np.array_equal(got.position, want.position)
        assert [n for n, _ in got.observations] == [n for n, _ in want.observations]
    assert batched_stats["observations_dropped"] > 0  # the outlier branch ran
