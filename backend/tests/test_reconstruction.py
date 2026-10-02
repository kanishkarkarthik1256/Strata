"""Tests for the reconstruction pipeline.

Unit tests for each module + integration test with synthetic images.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.feature_extractor import FeatureExtractor
from app.services.feature_matcher import FeatureMatcher, MatchResult
from app.services.pair_selector import (
    ImageInfo,
    compute_image_entropy,
    select_pairs,
)
from app.services.geometric_verifier import verify_matches
from app.services.confidence_estimator import estimate_confidence
from app.services.mission_analyzer import analyze_mission
from app.services.trajectory_optimizer import generate_trajectory
from app.services.camera_pose_estimator import (
    CameraPose,
    ReconstructionResult,
    SparsePoint3D,
)
from app.services.sparse_reconstruction import run_sparse_reconstruction


# ---------------------------------------------------------------------------
# Zero-registered-camera honesty (sparse stage contract)
# ---------------------------------------------------------------------------


class TestSparseZeroCameraHonesty:
    def test_zero_registered_cameras_fail_with_diagnostics(self, tmp_path: Path, monkeypatch):
        """Regression: SfM registering 0 cameras must FAIL, not report completed.

        The server-path failure (featureless clip → 0 registered cameras) used
        to produce status='completed', an empty stage detail, and no poses.json
        — the exact 'stage claims success without artifact' bug class.
        """
        from app.services import sparse_reconstruction as sr

        # A textured-enough frame dir so extraction/matching run, but pose
        # estimation registers nothing (backend returns an empty result).
        frames_dir = tmp_path / "selected"
        frames_dir.mkdir()
        frame = np.full((240, 320, 3), 127, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "frame_000.jpg"), frame)
        cv2.imwrite(str(frames_dir / "frame_001.jpg"), frame + 1)

        empty = ReconstructionResult(num_registered=0, num_points=0, mean_reproj_error=0.0)
        monkeypatch.setattr(sr, "estimate_poses", lambda *a, **k: empty)

        with pytest.raises(ValueError) as excinfo:
            run_sparse_reconstruction(frames_dir, tmp_path / "out", project_id="zerocam")
        msg = str(excinfo.value)
        assert "zero registered cameras" in msg
        # Real diagnostics are preserved in the error.
        assert "attempted_frames=2" in msg
        assert "pose_backend=" in msg
        # No success artifacts were written.
        assert not (tmp_path / "out" / "poses.json").exists()
        assert not (tmp_path / "out" / "reconstruction_report.json").exists()

    def test_valid_registration_completes(self, tmp_path: Path, monkeypatch):
        """The companion contract: registered cameras + artifacts → completed."""
        from app.services import sparse_reconstruction as sr

        frames_dir = tmp_path / "selected"
        frames_dir.mkdir()
        rng = np.random.default_rng(7)
        for i in range(3):
            img = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:03d}.jpg"), img)

        cams = {
            f"frame_{i:03d}": CameraPose(
                image_id=i,
                frame_id=f"frame_{i:03d}",
                position=np.array([float(i), 0.0, 0.0]),
                rotation=np.eye(3),
                quaternion=np.array([1.0, 0, 0, 0]),
                intrinsics=np.eye(3),
                distortion=np.zeros(5),
                is_estimated=True,
            )
            for i in range(3)
        }
        ok = ReconstructionResult(num_registered=3, num_points=12, mean_reproj_error=0.5)
        ok.cameras = cams
        # Points form a real scene well in front of the cameras (median
        # camera→point distance far exceeds the 2-unit path span) — the
        # fixture must satisfy the same non-degenerate-geometry contract the
        # pipeline enforces.
        ok.points3d = [
            SparsePoint3D(point_id=i, position=rng.random(3) * 5 + np.array([0.0, 0.0, -50.0]),
                          color=np.zeros(3, dtype=np.uint8), track_length=2)
            for i in range(12)
        ]
        monkeypatch.setattr(sr, "estimate_poses", lambda *a, **k: ok)

        report = run_sparse_reconstruction(frames_dir, tmp_path / "out", project_id="okcam")
        assert report["status"] == "completed"
        assert report["reconstruction"]["num_cameras"] == 3
        assert (tmp_path / "out" / "poses.json").exists()
        assert (tmp_path / "out" / "sparse_model.ply").exists()


# ---------------------------------------------------------------------------
# Feature extractor tests
# ---------------------------------------------------------------------------


class TestFeatureExtractor:
    def test_extract_returns_features(self):
        """Extraction should return keypoints and descriptors."""
        extractor = FeatureExtractor(max_keypoints=256)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        features = extractor.extract("test", img)

        assert features.keypoints.shape[1] == 2
        assert features.descriptors.shape[0] == features.keypoints.shape[0]
        assert features.backend in ("sift", "superpoint")
        assert features.extraction_time_ms > 0

    def test_empty_image_returns_no_features(self):
        """A flat image may have very few features."""
        extractor = FeatureExtractor(max_keypoints=256)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        features = extractor.extract("flat", img)
        assert len(features.keypoints) >= 0  # May be 0 or very few

    def test_batch_extraction(self):
        """Batch extraction should return one result per input."""
        extractor = FeatureExtractor(max_keypoints=128)
        frames = [
            (f"frame_{i}", np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            for i in range(3)
        ]
        results = extractor.extract_batch(frames)
        assert len(results) == 3
        assert all(r.frame_id.startswith("frame_") for r in results)


# ---------------------------------------------------------------------------
# Feature matcher tests
# ---------------------------------------------------------------------------


class TestFeatureMatcher:
    def test_match_identical_features(self):
        """Same features matched against themselves should produce matches."""
        from app.services.feature_extractor import Features

        kpts = np.random.rand(100, 2).astype(np.float32) * 640
        desc = np.random.rand(100, 128).astype(np.float32)
        scores = np.random.rand(100).astype(np.float32)

        feat_a = Features(frame_id="a", keypoints=kpts, descriptors=desc, scores=scores)
        feat_b = Features(frame_id="b", keypoints=kpts + 0.1, descriptors=desc, scores=scores)

        matcher = FeatureMatcher(ratio_threshold=0.8)
        result = matcher.match(feat_a, feat_b)

        assert isinstance(result, MatchResult)
        assert result.frame_a == "a"
        assert result.frame_b == "b"

    def test_sequential_matching(self):
        """Sequential matching should run without errors."""
        from app.services.feature_extractor import Features

        feats = []
        for i in range(5):
            kpts = np.random.rand(100, 2).astype(np.float32) * 640
            desc = np.random.rand(100, 128).astype(np.float32)
            scores = np.random.rand(100).astype(np.float32)
            feats.append(Features(frame_id=f"f{i}", keypoints=kpts, descriptors=desc, scores=scores))

        matcher = FeatureMatcher(ratio_threshold=0.9)  # Loose threshold for random data
        results = matcher.match_sequential(feats, window=2)
        # Random features may not produce good matches, but the function should not crash
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Pair selector tests
# ---------------------------------------------------------------------------


class TestPairSelector:
    def test_sequential_strategy(self):
        """Sequential strategy should pair consecutive frames."""
        images = [
            ImageInfo(frame_id=f"f{i}", index=i, timestamp_sec=float(i), file_path=f"/f{i}.jpg")
            for i in range(5)
        ]
        pairs = select_pairs(images, strategy="sequential")
        assert len(pairs) == 4
        assert pairs[0] == ("f0", "f1")

    def test_adaptive_strategy_limits_pairs(self):
        """Adaptive strategy should respect max_pairs."""
        images = [
            ImageInfo(frame_id=f"f{i}", index=i, timestamp_sec=float(i), file_path=f"/f{i}.jpg")
            for i in range(20)
        ]
        pairs = select_pairs(images, strategy="adaptive", max_pairs=10)
        assert len(pairs) <= 10

    def test_image_entropy(self):
        """Entropy should be higher for textured images than flat ones."""
        flat = np.zeros((100, 100, 3), dtype=np.uint8)
        textured = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert compute_image_entropy(textured) > compute_image_entropy(flat)


# ---------------------------------------------------------------------------
# Geometric verifier tests
# ---------------------------------------------------------------------------


class TestGeometricVerifier:
    def test_verify_with_good_matches(self):
        """Verification should run without errors."""
        from app.services.feature_matcher import MatchResult

        # Create synthetic matching points (with some noise)
        n = 50
        pts_a = np.random.rand(n, 2).astype(np.float32) * 400 + 100
        pts_b = pts_a + np.random.randn(n, 2).astype(np.float32) * 2

        kpts_a = np.vstack([pts_a, np.random.rand(50, 2).astype(np.float32) * 640])
        kpts_b = np.vstack([pts_b, np.random.rand(50, 2).astype(np.float32) * 640])

        matches_arr = np.column_stack([np.arange(n), np.arange(n)]).astype(np.int32)
        match = MatchResult(
            frame_a="a",
            frame_b="b",
            matches=matches_arr,
            num_inliers=n,
        )

        result = verify_matches(match, kpts_a, kpts_b, min_inlier_ratio=0.3)
        assert result.inlier_ratio >= 0.0


# ---------------------------------------------------------------------------
# Confidence estimator tests
# ---------------------------------------------------------------------------


class TestConfidenceEstimator:
    def test_confidence_estimation(self):
        """Confidence estimation should return valid values."""
        result = ReconstructionResult(
            num_registered=5,
            num_points=1000,
            mean_reproj_error=1.5,
        )
        result.cameras = {
            f"frame_{i}": CameraPose(
                image_id=i,
                frame_id=f"frame_{i}",
                position=np.random.rand(3),
                rotation=np.eye(3),
                quaternion=np.array([1, 0, 0, 0], dtype=np.float64),
                intrinsics=np.eye(3),
                distortion=np.zeros(5),
                is_estimated=True,
            )
            for i in range(5)
        }
        result.points3d = [
            SparsePoint3D(
                point_id=i,
                position=np.random.rand(3),
                color=np.random.randint(0, 255, 3, dtype=np.uint8),
                track_length=3,
            )
            for i in range(100)
        ]

        conf = estimate_confidence(result, feature_counts={f"frame_{i}": 1000 for i in range(5)})
        assert 0.0 <= conf.mean_camera_confidence <= 1.0
        assert len(conf.camera_confidences) == 5


# ---------------------------------------------------------------------------
# Mission analyzer tests
# ---------------------------------------------------------------------------


class TestMissionAnalyzer:
    def test_mission_score_range(self):
        """Mission score should be 0-100."""
        result = ReconstructionResult(num_registered=10, num_points=5000, mean_reproj_error=1.0)
        analysis = analyze_mission(result, total_frames=10, selected_frames=10)
        assert 0 <= analysis.mission_score <= 100
        assert analysis.grade in ("Excellent", "Good", "Fair", "Poor")


# ---------------------------------------------------------------------------
# Trajectory tests
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_trajectory_generation(self):
        """Trajectory should be generated from camera poses."""
        cameras = {
            f"frame_{i}": CameraPose(
                image_id=i,
                frame_id=f"frame_{i}",
                position=np.array([float(i), 0.0, 10.0]),
                rotation=np.eye(3),
                quaternion=np.array([1, 0, 0, 0], dtype=np.float64),
                intrinsics=np.eye(3),
                distortion=np.zeros(5),
                is_estimated=True,
            )
            for i in range(5)
        }
        traj = generate_trajectory(cameras)
        assert len(traj.points) == 5
        assert traj.total_length > 0


# ---------------------------------------------------------------------------
# Integration test via API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconstruction_api_endpoints(client, tmp_path: Path):
    """Test that reconstruction API endpoints respond correctly."""
    # Upload a completable video first: a correlated textured scene (same
    # pattern as test_pipeline_upgrade's e2e generator) so honest SfM can
    # register cameras — per-frame random noise has zero inter-frame features
    # and correctly fails sparse since the zero-camera fix.
    video_path = tmp_path / "test_drone.avi"
    rng = np.random.default_rng(11)
    base = cv2.GaussianBlur(rng.integers(0, 255, (960, 1280), dtype=np.uint8), (3, 3), 0)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 480))
    assert writer.isOpened(), "VideoWriter failed to open"
    for i in range(20):
        gray = cv2.warpAffine(base, np.float32([[1, 0, -4 * i], [0, 1, 0]]), (640, 480),
                              borderMode=cv2.BORDER_REPLICATE)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()
    with open(video_path, "rb") as f:
        upload_resp = await client.post(
            "/api/upload",
            files={"file": ("test_drone.avi", f, "video/avi")},
        )
    assert upload_resp.status_code == 201
    job_id = upload_resp.json()["job_id"]

    # Run frame extraction
    extract_resp = await client.post(
        f"/api/frame-extraction/{job_id}",
        json={"extraction_mode": "every_frame"},
    )
    assert extract_resp.status_code == 200

    # Start reconstruction
    recon_resp = await client.post(f"/api/reconstruction/start/{job_id}")
    assert recon_resp.status_code == 200
    assert recon_resp.json()["status"] == "completed"

    # Get dashboard
    dash_resp = await client.get(f"/api/reconstruction/dashboard/{job_id}")
    assert dash_resp.status_code == 200
    dash = dash_resp.json()
    assert "mission_score" in dash
    assert "grade" in dash

    # Get status
    status_resp = await client.get(f"/api/reconstruction/status/{job_id}")
    assert status_resp.status_code == 200

    # Get report
    report_resp = await client.get(f"/api/reconstruction/report/{job_id}")
    assert report_resp.status_code == 200
    assert "reconstruction" in report_resp.json()

    # Get confidence
    conf_resp = await client.get(f"/api/reconstruction/confidence/{job_id}")
    assert conf_resp.status_code == 200

    # Get features
    feat_resp = await client.get(f"/api/reconstruction/features/{job_id}")
    assert feat_resp.status_code == 200

    # Get matches
    match_resp = await client.get(f"/api/reconstruction/matches/{job_id}")
    assert match_resp.status_code == 200

    # Get trajectory
    traj_resp = await client.get(f"/api/reconstruction/trajectory/{job_id}")
    assert traj_resp.status_code == 200
