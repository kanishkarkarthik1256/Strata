"""Stored-depth-map geometry contract.

Depth Anything V2 infers at ~518 px; a 4K frame is interpolated up to 8.3 M
pixels, which is ~17x more pixels than the model produced. The pipeline now
stores the model's OWN grid (``native_resolution=True``) and records that
grid's geometry beside the map, so:

* a consumer projecting a world point into the map uses the intrinsics of
  THAT grid (``K_store``), never the frame's, and
* colour, which lives at frame resolution, is sampled by mapping map pixels
  back up with ``scale_x/scale_y``.

Every map written before this contract (and any map whose recorded geometry
disagrees with its own intrinsics) must resolve to the frame geometry, so no
already-generated run needs regenerating.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.depth_generator import (
    DEPTH_GEOMETRY_SCHEMA_VERSION,
    build_depth_geometry,
    read_depth_geometry,
)

FRAME = (2160, 3840)
NATIVE = (518, 924)


def _pose(cx: float | None = None, cy: float | None = None) -> dict:
    return {
        "frame_id": "frame_000000",
        "K": [[2700.0, 0.0, cx if cx is not None else 1920.0],
              [0.0, 2700.0, cy if cy is not None else 1080.0],
              [0.0, 0.0, 1.0]],
        "R": np.eye(3).tolist(),
        "t": [0.0, 0.0, 0.0],
    }


def _write(tmp_path: Path, depth: np.ndarray, meta: dict | None) -> Path:
    npy = tmp_path / "frame_000000.npy"
    np.save(npy, depth)
    if meta is not None:
        npy.with_suffix(".json").write_text(json.dumps(meta))
    return npy


def test_geometry_records_the_stored_grid_and_scales_intrinsics():
    geom = build_depth_geometry(_pose(), NATIVE, FRAME)
    assert geom["schema_version"] == DEPTH_GEOMETRY_SCHEMA_VERSION
    assert (geom["store_height"], geom["store_width"]) == NATIVE
    assert (geom["frame_height"], geom["frame_width"]) == FRAME

    sx, sy = geom["scale_x"], geom["scale_y"]
    assert sx == pytest.approx(924 / 3840)
    assert sy == pytest.approx(518 / 2160)

    K_store = np.asarray(geom["K_store"])
    K_frame = np.asarray(_pose()["K"])
    assert K_store[0, 0] == pytest.approx(K_frame[0, 0] * sx)
    assert K_store[0, 2] == pytest.approx(K_frame[0, 2] * sx)
    assert K_store[1, 1] == pytest.approx(K_frame[1, 1] * sy)
    assert K_store[1, 2] == pytest.approx(K_frame[1, 2] * sy)


def test_geometry_round_trips_through_the_sidecar(tmp_path):
    depth = np.ones(NATIVE, dtype=np.float32)
    geom = build_depth_geometry(_pose(), NATIVE, FRAME)
    npy = _write(tmp_path, depth, {"geometry": geom})

    K, sx, sy = read_depth_geometry(npy, _pose())
    assert K == pytest.approx(np.asarray(geom["K_store"]))
    assert (sx, sy) == pytest.approx((geom["scale_x"], geom["scale_y"]))


def test_projection_into_the_stored_grid_lands_inside_it(tmp_path):
    """The whole point of the contract: a point projected with K_store lands
    in the stored map's own pixel range, and with the frame K it does not."""
    from app.services.geometry import project_world_to_pixel

    pose = _pose()
    K_frame = np.abs(np.asarray(pose["K"]))
    npy = _write(
        tmp_path,
        np.ones(NATIVE, dtype=np.float32),
        {"geometry": build_depth_geometry(pose, NATIVE, FRAME)},
    )
    K_store, _, _ = read_depth_geometry(npy, pose)
    # A ray through the frame's top-left-ish pixel region.
    pts = np.array([[0.0, 0.0, 100.0], [10.0, 5.0, 100.0]])
    u_f, v_f, _ = project_world_to_pixel(pts, np.eye(3), np.zeros(3), K_frame)
    u_s, v_s, _ = project_world_to_pixel(pts, np.eye(3), np.zeros(3), K_store)

    assert (u_s < NATIVE[1]).all() and (v_s < NATIVE[0]).all()
    assert (u_f > NATIVE[1]).any() or (v_f > NATIVE[0]).any()
    # The stored coordinates are the frame coordinates scaled down.
    assert u_s == pytest.approx(u_f * (NATIVE[1] / FRAME[1]))
    assert v_s == pytest.approx(v_f * (NATIVE[0] / FRAME[0]))


def test_legacy_map_without_geometry_resolves_to_the_frame(tmp_path):
    """Maps written before the contract must keep working unchanged."""
    pose = _pose()
    npy = _write(tmp_path, np.ones(FRAME, dtype=np.float32), {"backend": "depth_anything"})
    K, sx, sy = read_depth_geometry(npy, pose)
    assert K == pytest.approx(np.abs(np.asarray(pose["K"])))
    assert (sx, sy) == (1.0, 1.0)


def test_missing_sidecar_resolves_to_the_frame(tmp_path):
    pose = _pose()
    npy = _write(tmp_path, np.ones(FRAME, dtype=np.float32), None)
    K, sx, sy = read_depth_geometry(npy, pose)
    assert K == pytest.approx(np.abs(np.asarray(pose["K"])))
    assert (sx, sy) == (1.0, 1.0)


def test_inconsistent_geometry_is_refused(tmp_path):
    """A sidecar describing a grid the intrinsics do not agree with is not
    trustworthy: fall back rather than project with a made-up K."""
    pose = _pose()
    geom = build_depth_geometry(pose, NATIVE, FRAME)
    geom["K_store"][0][0] *= 3.0  # now disagrees with scale_x
    npy = _write(tmp_path, np.ones(NATIVE, dtype=np.float32), {"geometry": geom})
    K, sx, sy = read_depth_geometry(npy, pose)
    assert K == pytest.approx(np.abs(np.asarray(pose["K"])))
    assert (sx, sy) == (1.0, 1.0)


def test_corrupt_sidecar_resolves_to_the_frame(tmp_path):
    pose = _pose()
    npy = tmp_path / "frame_000000.npy"
    np.save(npy, np.ones(NATIVE, dtype=np.float32))
    npy.with_suffix(".json").write_text("{not json")
    K, sx, sy = read_depth_geometry(npy, pose)
    assert K == pytest.approx(np.abs(np.asarray(pose["K"])))
    assert (sx, sy) == (1.0, 1.0)


def test_missing_frame_size_yields_no_geometry():
    """Without the frame size the scale is unknowable; record nothing rather
    than a wrong scale (the map then resolves as frame-resolution)."""
    assert build_depth_geometry(_pose(), NATIVE, None) is None
