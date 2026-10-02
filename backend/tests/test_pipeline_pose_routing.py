from pathlib import Path

import cv2
import numpy as np

from app.services.pipeline_orchestrator import _has_metric_pose_columns
from app.services.sparse_reconstruction import _has_pose_columns, _pose_only_result
from app.services.depth_anything_v2 import resolve_device


def test_pose_rich_telemetry_is_detected(tmp_path: Path):
    path = tmp_path / "telemetry.csv"
    path.write_text("frame_id,x,y,z,qw,qx,qy,qz,latitude\n0,1,2,3,1,0,0,0,0\n")
    assert _has_metric_pose_columns(path)


def test_position_only_telemetry_is_not_used_as_pose_input(tmp_path: Path):
    path = tmp_path / "telemetry.csv"
    path.write_text("frame_id,latitude,longitude,altitude\n0,1,2,3\n")
    assert not _has_metric_pose_columns(path)


def test_requested_cuda_falls_back_cleanly_on_cpu_host():
    assert resolve_device("cuda") in {"cuda", "cpu"}
    assert resolve_device("not-a-device") == "cpu"


def test_pose_only_fast_path_keeps_all_telemetry_cameras(tmp_path: Path):
    path = tmp_path / "telemetry.csv"
    path.write_text("frame_id,x,y,z,qw,qx,qy,qz\n0,1,2,3,1,0,0,0\n")
    assert _has_pose_columns(path)

    result = _pose_only_result(
        {"frame_000000": (np.array([1.0, 2.0, 3.0]), np.eye(3))},
        np.eye(3),
        [tmp_path / "frame_000000.png"],
    )
    assert result.backend == "telemetry_triangulation"
    assert result.num_registered == 1
    assert result.num_points == 0


def test_rpy_inconsistency_does_not_reject_telemetry(tmp_path: Path):
    """RPY/quaternion discrepancy should NOT block telemetry-assisted poses.

    The quaternion is canonical and directly measured by the flight controller.
    A 124° RPY/quaternion disagreement means the Euler convention adapter is
    wrong — NOT that the quaternion is wrong. The telemetry path must still
    proceed with the valid quaternion poses.
    """
    from app.services.sparse_reconstruction import (
        _cross_validate_rpy,
        _MAX_TELEMETRY_ATTITUDE_DISAGREEMENT_DEG,
        _telemetry_assisted_poses,
    )

    # Build a flight_poses CSV with quaternions that disagree with the RPY
    # fields (simulating the Video_Mission_13c5b7 scenario: 124° discrepancy)
    csv_path = tmp_path / "flight_poses.csv"
    lines = ["frame_id,x,y,z,qw,qx,qy,qz,latitude,longitude,altitude,fov_vertical,roll,pitch,yaw"]
    for i in range(10):
        # Valid quaternion (identity-ish) but RPY pointing ~180° away
        lines.append(
            f"{i},{float(i)},0.0,50.0,"
            f"0.063,-0.115,0.870,-0.475,"  # quaternion (valid attitude)
            f"52.37,13.51,50.0,26.9,"
            f"-122.7,0.02,-164.9"          # RPY (inconsistent convention)
        )
    csv_path.write_text("\n".join(lines) + "\n")

    # Verify the cross-validation DOES detect a large discrepancy
    import csv as csv_mod
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv_mod.DictReader(fh))
    check = _cross_validate_rpy(rows)
    assert check is not None
    assert check["median_angle_diff_deg"] > _MAX_TELEMETRY_ATTITUDE_DISAGREEMENT_DEG

    # Create minimal frame files
    frames_dir = tmp_path / "selected"
    frames_dir.mkdir()
    for i in range(10):
        # Create tiny 4x4 grey images (just needs to be readable by cv2)
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

    frame_files = sorted(frames_dir.glob("*.jpg"))

    # Call _telemetry_assisted_poses — it should NOT return None
    result, info, gps_priors = _telemetry_assisted_poses(
        selected_dir=frames_dir,
        features={},        # no features: goes to pose_only_result
        verified=[],        # no verified pairs
        frame_files=frame_files,
        flight_poses_csv=csv_path,
        intrinsics_path=None,
    )

    # The RPY discrepancy must be recorded in provenance
    assert "rpy_quaternion_crosscheck" in info
    assert info["rpy_quaternion_crosscheck"]["median_angle_diff_deg"] > 90

    # But the result must NOT be None — telemetry poses should be accepted
    assert result is not None, (
        "RPY/quaternion discrepancy should not reject telemetry — "
        "quaternion is canonical"
    )
    assert result.num_registered == 10
    assert result.backend == "telemetry_triangulation"

    # GPS priors should still be passed through
    assert gps_priors is not None
    assert len(gps_priors) == 10

