"""Projection into a stored depth map must use THAT map's own intrinsics.

Depth maps are stored on the model's grid (518x924 for a 4K frame), not the
frame's. Every consumer that projects a world point into a map therefore has
to resolve the map grid's K through ``read_depth_geometry``. Two diagnostics
kept using the FRAME's K against the small grid, and because the frame K is
~4x larger in pixels, every projection landed outside the map except the
points falling in the frame's top-left corner.

Measured on the DJI clip when the maps went native: landmarks compared
dropped 37k -> 1.2k, the reference "sparse depth" became that corner's far
field (104 m -> 708 m), cross-view median relative error went 0.4-2% ->
218-322%, and criteria D and G failed the whole stage on a map that was
correct. These tests build exactly that geometry — a varying depth field so a
wrong pixel is a wrong depth — and pin the difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.depth_diagnostics import _cross_view_consistency, _sfm_camera_depths_px
from app.services.depth_generator import build_depth_geometry, read_depth_geometry

FRAME = (2160, 3840)
NATIVE = (518, 924)
FX_FRAME, FY_FRAME = 2700.0, 2700.0
CX_FRAME, CY_FRAME = 1920.0, 1080.0
SX = NATIVE[1] / FRAME[1]
SY = NATIVE[0] / FRAME[0]


def _pose(frame_id: str, centre: tuple[float, float, float]) -> dict:
    return {
        "frame_id": frame_id,
        "K": [[FX_FRAME, 0.0, CX_FRAME], [0.0, FY_FRAME, CY_FRAME], [0.0, 0.0, 1.0]],
        "R": np.eye(3).tolist(),
        "t": list(centre),
    }


def _sloped_depth() -> np.ndarray:
    """A depth field that varies with x, so a wrong pixel reads a wrong depth."""
    _, w = NATIVE
    u = np.arange(w, dtype=np.float32)[None, :]
    return np.broadcast_to(20.0 + 0.05 * u, NATIVE).astype(np.float32).copy()


def _write_map(depth_dir: Path, pose: dict, depth: np.ndarray) -> Path:
    depth_dir.mkdir(parents=True, exist_ok=True)
    npy = depth_dir / f"{pose['frame_id']}.npy"
    np.save(npy, depth)
    npy.with_suffix(".json").write_text(json.dumps(
        {"frame_id": pose["frame_id"], "geometry": build_depth_geometry(pose, depth.shape, FRAME)}
    ))
    return npy


def test_frame_intrinsics_against_a_native_map_see_only_the_top_left_corner():
    """The exact failure: the frame's K discards all but the frame's corner."""
    pose = _pose("frame_000000", (0.0, 0.0, 0.0))
    rng = np.random.default_rng(0)
    # Sparse points spread across the whole frame, at the map's own depths.
    u_f = np.linspace(0, FRAME[1] - 1, 4000)
    v_f = np.linspace(0, FRAME[0] - 1, 4000)
    z = 20.0 + 0.05 * (u_f * SX)
    x = (u_f - CX_FRAME) / FX_FRAME * z
    y = (v_f - CY_FRAME) / FY_FRAME * z
    sparse = np.stack([x, y, z], axis=1) + rng.normal(0, 1e-6, (4000, 3))

    _, u_bad, v_bad = _sfm_camera_depths_px(pose, sparse, NATIVE)
    K_map, _, _ = read_depth_geometry(
        _write_map(Path(_tmp()), pose, _sloped_depth()), pose
    )
    _, u_good, v_good = _sfm_camera_depths_px(pose, sparse, NATIVE, K=K_map)

    assert len(u_good) > 0.9 * len(sparse), (
        f"map-grid K kept only {len(u_good)}/{len(sparse)} landmarks"
    )
    # With the frame's K only points falling inside the map's rectangle
    # survive, so the ceiling is exactly the map's width/height fraction of
    # the frame — not a taste threshold. (On the real DJI cloud, whose
    # landmarks cluster near the principal point, that measured 3%: 37k
    # compared dropped to 1.2k.)
    assert len(u_bad) <= len(sparse) * SX * 1.05
    assert len(u_bad) < len(u_good) / 3
    assert u_bad.max() <= NATIVE[1] and v_bad.max() <= NATIVE[0]


def _tmp() -> str:
    import tempfile

    return tempfile.mkdtemp()


def test_resolved_geometry_is_the_maps_grid_not_the_frames():
    pose = _pose("frame_000000", (0.0, 0.0, 0.0))
    K_map, sx, sy = read_depth_geometry(
        _write_map(Path(_tmp()), pose, _sloped_depth()), pose
    )
    assert sx == pytest.approx(SX, rel=1e-6)
    assert sy == pytest.approx(SY, rel=1e-6)
    assert K_map[0, 0] < FX_FRAME and K_map[1, 1] < FY_FRAME


def test_cross_view_uses_the_maps_own_grid_for_both_frames(tmp_path):
    """Frame B's projection is where the wrong K hid — pin the verdict.

    On the buggy code this pair reports a median relative error above 2.0
    (the grid mismatch read as depth disagreement) and fails the gate on a
    scene whose two maps are identical.
    """
    pose_a = _pose("frame_000000", (0.0, 0.0, 0.0))
    pose_b = _pose("frame_000001", (1.0, 0.0, 0.0))
    depth = _sloped_depth()
    depth_dir = tmp_path / "depth"
    _write_map(depth_dir, pose_a, depth)
    _write_map(depth_dir, pose_b, depth)

    out = _cross_view_consistency(
        tmp_path, [pose_a, pose_b], depth_dir,
        sparse_xyz=np.zeros((0, 3)), max_depth_m=200.0,
    )

    assert out["pairs"], "no pair was compared — the fixture lost its overlap"
    pair = out["pairs"][0]
    assert pair["n_compared"] >= 100
    # Identical maps: any real disagreement is the projection's fault.
    assert pair["median_rel_err"] < 0.10, pair
    assert out["verdict"] == "PASS", out
