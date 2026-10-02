"""Texture pipeline vs native-resolution depth maps — the grid contract.

The 2026-09-28 Video_Mission_9ab1aa regression: depth maps are stored at the
depth model's own grid (~518 rows) while the texture projector projected with
frame-grid pixels (1080 rows). Indexing the map with frame pixels crashed
("index 1011 is out of bounds for axis 0 with size 518") and the whole viewer
GLB silently degraded to vertex colors. The occlusion test must project into
the map's OWN grid via ``K_depth``; the RGB order must survive to the encoded
atlas (views loaded as RGB, atlas encoded BGR-safe).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.mesh import TriangleMesh
from app.services.texture_blender import _load_views
from app.services.texture_projector import TextureView, select_best_views


def _K_frame() -> np.ndarray:
    # Frame grid 96x96, f = 60.
    return np.array([[60.0, 0, 47.5], [0, 60.0, 47.5], [0, 0, 1]])


def _look_down() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _flat_mesh() -> TriangleMesh:
    v = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    colors = np.full((4, 3), 127, dtype=np.uint8)
    return TriangleMesh(vertices=v, faces=f, colors=colors)


def _seed_workspace(tmp_path: Path) -> Path:
    """Image at the FRAME grid, depth map stored at a COARSER native grid."""
    workspace = tmp_path / "run_native"
    (workspace / "selected").mkdir(parents=True)
    (workspace / "depth").mkdir()

    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # cv2 writes BGR: channel 0 = blue
    cv2.imwrite(str(workspace / "selected" / "frame_000000.jpg"), img)

    pose = {"frame_id": "frame_000000", "K": _K_frame().tolist(),
            "R": _look_down().tolist(), "t": [2.0, 2.0, 10.0]}
    (workspace / "poses.json").write_text(json.dumps({"frames": [pose]}))

    # Depth map at half resolution (48x48) with matching K (fx/2). Values
    # sit slightly BEHIND the ground quad (z=10) so the quad passes the
    # occlusion test (depth <= z_map * 1.02 + 0.05).
    K48 = _K_frame().copy()
    K48[0, :] *= 0.5
    K48[1, :] *= 0.5
    depth = np.full((48, 48), 11.0, dtype=np.float64)
    np.save(workspace / "depth" / "frame_000000.npy", depth)
    (workspace / "depth" / "frame_000000.json").write_text(json.dumps({
        "frame_id": "frame_000000",
        "geometry": {
            "schema_version": 2,
            "store_width": 48, "store_height": 48,
            "frame_width": 96, "frame_height": 96,
            "scale_x": 0.5, "scale_y": 0.5,
            "K_store": K48.tolist(),
        },
    }))
    return workspace


def test_occlusion_uses_depth_grid_not_frame_grid(tmp_path):
    """The legacy code indexed the 48-row map with 95-row pixels — out of
    bounds the moment the map is coarser than the image. With K_depth the
    same geometry must pass the occlusion test and assign the view."""
    workspace = _seed_workspace(tmp_path)
    views, missing = _load_views(
        workspace / "poses.json", workspace / "selected", workspace / "depth")
    assert not missing
    assert views[0].depth.shape == (48, 48)
    assert views[0].K_depth is not None
    assert np.isclose(views[0].K_depth[0, 0], 30.0)  # frame fx 60 x 0.5

    mesh = _flat_mesh()
    projection = select_best_views(mesh, views, max_camera_dist=100.0)
    assert projection.visible_fraction == 1.0, (
        "both faces of the visible ground quad must survive occlusion — "
        "a frame-grid occlusion read crashes or falsely rejects here"
    )


def test_legacy_frame_grid_maps_still_occlude(tmp_path):
    """A depth map stored at the frame's resolution (no scale) keeps working:
    K_depth resolves to the frame K and the occlusion test behaves as before."""
    workspace = _seed_workspace(tmp_path)
    # Overwrite with a frame-resolution map and REMOVE the geometry sidecar.
    depth = np.full((96, 96), 11.0, dtype=np.float64)
    np.save(workspace / "depth" / "frame_000000.npy", depth)
    (workspace / "depth" / "frame_000000.json").unlink()

    views, _ = _load_views(
        workspace / "poses.json", workspace / "selected", workspace / "depth")
    assert views[0].K_depth is not None  # falls back to frame K
    assert np.isclose(views[0].K_depth[0, 0], 60.0)
    projection = select_best_views(_flat_mesh(), views, max_camera_dist=100.0)
    assert projection.visible_fraction == 1.0


def test_atlas_pixels_come_out_rgb_not_swapped(tmp_path):
    """The image is blue (B=200 in BGR). A correctly ordered atlas tile must
    be blue in RGB (low R, low G, high B) — the pre-fix pipeline sampled
    cv2's BGR buffer and produced red tiles (red/blue swap)."""
    workspace = _seed_workspace(tmp_path)
    views, _ = _load_views(
        workspace / "poses.json", workspace / "selected", workspace / "depth")
    img = views[0].image
    assert img.shape[:2] == (96, 96)
    # TextureView.image is documented RGB: blue image → high B channel.
    assert img[..., 2].mean() > 150, "view image must be RGB (blue in channel 2)"
    assert img[..., 0].mean() < 30, "view image must be RGB (red channel near zero)"

    projection = select_best_views(_flat_mesh(), views, max_camera_dist=100.0)
    assert projection.visible_fraction == 1.0
    # The baked atlas inherits the sampled view pixels in RGB order.
    from app.services.uv_mapper import build_atlas_layout
    from app.services.texture_blender import build_texture_atlas
    layout = build_atlas_layout(_flat_mesh().m, atlas_width=64)
    atlas, stats = build_texture_atlas(_flat_mesh(), views, layout)
    assert stats.get("atlas_written")
    painted = atlas[atlas.sum(axis=2) > 0]
    assert len(painted) > 0
    assert painted[:, 2].mean() > 150, "atlas must carry the blue pixels in RGB order"
    assert painted[:, 0].mean() < 30, "atlas must NOT be channel-swapped"
