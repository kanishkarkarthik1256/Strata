"""The batched COLMAP-text export must be byte-identical to the scan it replaced.

``_export_colmap_text`` used to build each image's observation list by scanning
all points for every camera, and each point's track by scanning all cameras —
O(C·P·obs) (~7M dict lookups per export on the run under measurement). It now
does one pass over the points and one inversion pass, which is only legitimate
if the emitted files are unchanged: they are handed straight to pycolmap, so a
reordered track silently scrambles the imported geometry (measured once before:
mean reprojection 0.31 → 706 px).

These tests therefore hold the new implementation against a verbatim copy of
the old scan and assert the files are *identical*, on a scene that exercises
the real edge cases: observations listed out of camera order, a point that
observes a frame with no camera, a point with no track, and a camera with no
observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.bundle_adjustment import _export_colmap_text
from app.services.camera_pose_estimator import (
    CameraPose,
    ReconstructionResult,
    SparsePoint3D,
)


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _scene() -> ReconstructionResult:
    rng = np.random.default_rng(7)
    names = [f"frame_{i:06d}" for i in range(5)]
    result = ReconstructionResult()
    for i, name in enumerate(names):
        K = np.array([[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]])
        R = _rot_z(5.0 * i)
        result.cameras[name] = CameraPose(
            image_id=i + 1,
            frame_id=name,
            position=np.array([float(i), 0.5 * i, 30.0]),
            rotation=R,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            intrinsics=K,
            distortion=np.zeros(5),
        )
    result.num_registered = len(names)

    # Point 0 observes three frames, deliberately listed out of camera order.
    obs = [
        (names[3], np.array([10.0, 20.0])),
        (names[0], np.array([1.5, 2.5])),
        (names[2], np.array([7.25, 8.125])),
    ]
    result.points3d.append(
        SparsePoint3D(
            point_id=1,
            position=np.array([1.0, 2.0, 3.0]),
            color=np.array([255, 128, 0]),
            track_length=3,
            mean_reproj_error=0.31,
            observations=obs,
        )
    )
    # Point 1 observes a frame that is not a camera (must be skipped), plus two
    # real cameras that also see point 0 — so ordering within a camera matters.
    result.points3d.append(
        SparsePoint3D(
            point_id=2,
            position=np.array([-4.0, 0.5, 12.0]),
            color=np.array([0, 0, 255]),
            track_length=2,
            mean_reproj_error=0.87,
            observations=[
                (names[0], np.array([300.0, 400.0])),
                ("frame_not_a_camera", np.array([1.0, 1.0])),
                (names[1], np.array([50.0, 60.0])),
            ],
        )
    )
    # Point 2 has no observations at all (imported points can arrive trackless).
    result.points3d.append(
        SparsePoint3D(
            point_id=3,
            position=np.array([9.0, 9.0, 9.0]),
            color=np.array([10, 20, 30]),
            observations=[],
        )
    )
    # Point 3 is a long track that touches every camera, in scrambled order.
    order = rng.permutation(len(names))
    result.points3d.append(
        SparsePoint3D(
            point_id=4,
            position=np.array([0.25, -1.0, 22.0]),
            color=np.array([7, 200, 90]),
            track_length=len(names),
            mean_reproj_error=0.55,
            observations=[
                (names[int(i)], np.array([float(i) * 11.0, float(i) * 13.0]))
                for i in order
            ],
        )
    )
    result.num_points = len(result.points3d)
    return result


def _old_export_reference(result: ReconstructionResult, root: Path) -> None:
    """The per-camera scan `_export_colmap_text` replaced, verbatim.

    Kept in the test (not the app) so the batched rewrite has something to be
    equal to: cameras.txt and the pose headers come from the shared writer, but
    the observation lists and point tracks are rebuilt the old way.
    """
    from app.services.bundle_adjustment import _rot_to_qvec

    sparse = root / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    cams = list(result.cameras.items())
    name_to_colid = {name: cid for cid, (name, _) in enumerate(cams, start=1)}

    # OLD: for every camera, scan every point and every observation.
    img_lines = []
    obs_by_cam: dict[int, list[tuple[int, np.ndarray]]] = {}
    for cam_id, (name, cam) in enumerate(cams, start=1):
        obs: list[tuple[int, np.ndarray]] = []
        for pt_idx, pt in enumerate(result.points3d):
            for fid, pix in pt.observations:
                if name_to_colid.get(fid) == cam_id:
                    obs.append((pt_idx + 1, np.asarray(pix, dtype=np.float64)))
        obs_by_cam[cam_id] = obs
        R_w2c = np.asarray(cam.rotation, dtype=np.float64).T
        C = np.asarray(cam.position, dtype=np.float64)
        t = -R_w2c @ C
        q = _rot_to_qvec(R_w2c)
        header = (
            f"{cam_id} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
            f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {cam_id} {name}"
        )
        body = " ".join(f"{pix[0]:.4f} {pix[1]:.4f} {pid}" for pid, pix in obs)
        img_lines.append(header + "\n" + body)
    (sparse / "images.txt").write_text("\n".join(img_lines) + "\n")

    # OLD: for every point, scan all cameras for its track elements.
    pt_lines = []
    for pt_idx, pt in enumerate(result.points3d):
        elems: list[tuple[int, int]] = []
        for cam_id in range(1, len(cams) + 1):
            for i, (pid, _pix) in enumerate(obs_by_cam[cam_id]):
                if pid == pt_idx + 1:
                    elems.append((cam_id, i))
        if not elems:
            continue
        r = int(np.clip(pt.color[0], 0, 255))
        g = int(np.clip(pt.color[1], 0, 255))
        b = int(np.clip(pt.color[2], 0, 255))
        track = " ".join(f"{cid} {i}" for cid, i in elems)
        pt_lines.append(
            f"{pt_idx + 1} {pt.position[0]:.8f} {pt.position[1]:.8f} {pt.position[2]:.8f} "
            f"{r} {g} {b} {pt.mean_reproj_error:.6f} {track}"
        )
    (sparse / "points3D.txt").write_text("\n".join(pt_lines) + "\n")


def test_batched_export_is_byte_identical_to_the_old_scan(tmp_path: Path) -> None:
    new_root = tmp_path / "new"
    old_root = tmp_path / "old"
    result = _scene()

    _export_colmap_text(result, new_root)
    _old_export_reference(result, old_root)

    # The rewrite touched the observation-list builder (images.txt) and the
    # track builder (points3D.txt); cameras.txt comes from unchanged code, so
    # the equality claim covers exactly those two files.
    for name in ("images.txt", "points3D.txt"):
        new_bytes = (new_root / "sparse" / name).read_bytes()
        old_bytes = (old_root / "sparse" / name).read_bytes()
        assert new_bytes == old_bytes, f"{name} differs from the pre-rewrite output"

    camera_lines = (new_root / "sparse" / "cameras.txt").read_text().strip().splitlines()
    assert len(camera_lines) == len(result.cameras)
    assert all(line.split()[1] == "PINHOLE" for line in camera_lines)


def test_export_skips_observations_from_unknown_frames(tmp_path: Path) -> None:
    result = _scene()
    _export_colmap_text(result, tmp_path)

    images = (tmp_path / "sparse" / "images.txt").read_text().splitlines()
    points = (tmp_path / "sparse" / "points3D.txt").read_text().splitlines()

    # 5 cameras → 10 lines (pose header + points2D body per camera).
    assert len(images) == 10
    # Point 2 observes "frame_not_a_camera" — that observation must not appear
    # anywhere, and the point itself still exports with its two real tracks.
    assert "frame_not_a_camera" not in "\n".join(images)
    assert not any(line.startswith("3 ") for line in points)  # trackless point


def test_every_track_element_resolves_to_its_own_observation(tmp_path: Path) -> None:
    """Semantic check: track (image_id, idx) must address that pixel."""
    result = _scene()
    _export_colmap_text(result, tmp_path)

    images = (tmp_path / "sparse" / "images.txt").read_text().splitlines()
    bodies: dict[int, list[tuple[float, float, int]]] = {}
    for i in range(0, len(images), 2):
        header = images[i].split()
        cam_id = int(header[0])
        body = images[i + 1].split()
        triplets = [
            (float(body[j]), float(body[j + 1]), int(body[j + 2]))
            for j in range(0, len(body), 3)
        ]
        bodies[cam_id] = triplets

    points = (tmp_path / "sparse" / "points3D.txt").read_text().splitlines()
    assert points, "expected at least one exported point"
    checked = 0
    for line in points:
        parts = line.split()
        pid = int(parts[0])
        elems = parts[8:]
        assert len(elems) % 2 == 0
        for j in range(0, len(elems), 2):
            cam_id, idx = int(elems[j]), int(elems[j + 1])
            assert 0 <= idx < len(bodies[cam_id])
            assert bodies[cam_id][idx][2] == pid, (
                f"track of point {pid} points at camera {cam_id} slot {idx}, "
                f"which holds point {bodies[cam_id][idx][2]}"
            )
            checked += 1
    # Point 0: 3 obs, point 1: 2 obs, point 3: 5 obs = 10 track elements.
    assert checked == 10
    assert len(points) == 3  # the trackless point emits no line


def test_export_is_idempotent_and_orders_tracks_by_camera(tmp_path: Path) -> None:
    result = _scene()
    _export_colmap_text(result, tmp_path)
    first = (tmp_path / "sparse" / "points3D.txt").read_bytes()
    _export_colmap_text(result, tmp_path)
    assert (tmp_path / "sparse" / "points3D.txt").read_bytes() == first

    for line in first.decode().splitlines():
        elems = line.split()[8:]
        cams = [int(elems[j]) for j in range(0, len(elems), 2)]
        assert cams == sorted(cams)
