"""Criterion D must compare LIKE populations.

The audit's single-frame unprojection test checks that an unprojected depth
map agrees in scale with the SfM geometry. It measured the whole-frame cloud
median against the sparse-FEATURE median — two different populations by
construction: the map covers the entire frame (sky, background, featureless
ground), the sparse cloud only texture.

Measured failure this closes (sunset_06cfea, dense blocked on D): the map
agrees with the sparse geometry to 1.001x AT the landmark pixels
(9.58 vs 9.57 m) while the whole-frame ratio reads 1.67x, so a correctly
scaled map could never pass.

Contracts pinned here:
1. A map whose scale is right where both measurements exist passes D even
   when its far background makes the whole-frame ratio breach 1.5x — and the
   fixture is shown to discriminate (the whole-frame ratio is recorded and
   asserted > 1.5).
2. A globally mis-scaled map still fails D, with the ratio recorded.
3. With no overlapping landmarks the scale check is not evaluable: recorded
   as null, never guessed as satisfied by a number.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.config.settings import settings
from app.services.depth_diagnostics import run_depth_diagnostics
from app.services.geometry import project_world_to_pixel
from app.services.pointcloud import PointCloud, save_ply

H, W = 480, 640
K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
R_C2W = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # looking down
NEAR_M, BACKGROUND_M = 25.0, 80.0


def _camera_axis_points(n: int = 300, seed: int = 4, z: float = NEAR_M) -> np.ndarray:
    """Sparse landmarks on a near surface, spread across the central frame."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.25 * W, 0.75 * W, n)
    v = rng.uniform(0.25 * H, 0.75 * H, n)
    xn = (u - K[0, 2]) / K[0, 0]
    yn = (v - K[1, 2]) / K[1, 1]
    xc = np.stack([xn * z, yn * z, np.full(n, z)], axis=1)
    return xc @ R_C2W.T  # cameras at the origin, so world = xc @ R^T


def _workspace(tmp_path: Path, job_id: str, depth_scale: float = 1.0, offset_points: bool = False) -> Path:
    settings.storage.base_path = str(tmp_path)
    ws = settings.storage.project_dir(job_id)
    (ws / "selected").mkdir(parents=True, exist_ok=True)
    (ws / "depth").mkdir(parents=True, exist_ok=True)

    frames = [{"frame_id": f"frame_{i:06d}", "K": K.tolist(), "R": R_C2W.tolist(),
               "t": [0.0, 0.0, 0.0]} for i in range(3)]
    (ws / "poses.json").write_text(json.dumps({"frames": frames}))
    for f in frames:
        cv2.imwrite(str(ws / "selected" / f"{f['frame_id']}.jpg"), np.full((H, W, 3), 120, np.uint8))

    if offset_points:
        # Landmarks far off-axis: nothing projects into this frame, so the
        # scale check has no common support to measure.
        xyz = _camera_axis_points(seed=9) + np.array([500.0, 500.0, 400.0])
    else:
        xyz = _camera_axis_points()
    save_ply(ws / "sparse_model.ply", PointCloud(xyz=xyz))

    # Depth map: the near surface where the landmarks are, background
    # everywhere else — the far-field population a real frame carries and the
    # sparse cloud cannot sample.
    depth = np.full((H, W), BACKGROUND_M * depth_scale, dtype=np.float32)
    if not offset_points:
        pose = frames[0]
        u, v, _z = project_world_to_pixel(
            xyz, np.asarray(pose["R"], float), np.asarray(pose["t"], float), K
        )
        pu = np.clip(u, 0, W - 1).astype(int)
        pv = np.clip(v, 0, H - 1).astype(int)
        depth[pv, pu] = NEAR_M * depth_scale
    for f in frames:
        fid = f["frame_id"]
        np.save(ws / "depth" / f"{fid}.npy", depth)
        (ws / "depth" / f"{fid}.json").write_text(json.dumps({
            "frame_id": fid, "backend": "stereo", "model_version": "stereo-sgbm",
            "metric": True, "valid_ratio": 1.0,
        }))
    return ws


def test_population_difference_no_longer_fails_a_correctly_scaled_map(tmp_path: Path):
    ws = _workspace(tmp_path, "diag_pop1")
    report = run_depth_diagnostics(ws)
    sf = report.single_frame_cloud
    assert sf is not None
    assert report.acceptance_criteria["D_sensible_single_frame_unprojection"] is True
    # The map's scale IS right where both populations exist.
    assert sf.landmark_scale_ratio is not None
    assert abs(sf.landmark_scale_ratio - 1.0) <= 0.05, sf.landmark_scale_ratio
    # ... while the whole-frame ratio is exactly the mismatch that used to
    # decide the verdict. If this ever drops below 1.5 the fixture stops
    # exercising the population bias and the test above goes vacuous.
    assert sf.cloud_depth_median_m / sf.sfm_depth_median_m > 1.5


def test_mis_scaled_map_still_fails(tmp_path: Path):
    ws = _workspace(tmp_path, "diag_pop2", depth_scale=1.6)
    report = run_depth_diagnostics(ws)
    sf = report.single_frame_cloud
    assert sf is not None
    assert sf.landmark_scale_ratio is not None
    assert abs(sf.landmark_scale_ratio - 1.6) <= 0.05, sf.landmark_scale_ratio
    assert sf.status == "FAIL"
    assert report.acceptance_criteria["D_sensible_single_frame_unprojection"] is False


def test_scale_check_is_not_evaluable_without_common_support(tmp_path: Path):
    ws = _workspace(tmp_path, "diag_pop3", offset_points=True)
    report = run_depth_diagnostics(ws)
    sf = report.single_frame_cloud
    assert sf is not None
    # No overlapping landmarks: the ratio is null (not evaluable), and D
    # rests on the cloud's geometric plausibility alone — as before.
    assert sf.landmark_scale_ratio is None
    assert sf.sfm_depth_median_m is None
    assert report.acceptance_criteria["D_sensible_single_frame_unprojection"] is True
    payload = sf.to_dict()
    assert payload["landmark_scale_ratio"] is None
    assert payload["landmark_depth_median_m"] is None
