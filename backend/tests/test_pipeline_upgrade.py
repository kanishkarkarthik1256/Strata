"""Tests for the Phase 6 autonomous-pipeline upgrade.

Covers the real geometry code end to end: rectified-stereo per-view depth
generation (metric, tested against a synthetic ground plane), optional depth
refinement, WGS84/ENU georeferencing + GPS quality analysis, sparse-output
persistence, and the orchestrator (resume-by-artifact, depth cache,
cancellation, stage timeline) driven through its REST entry points.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.models import Project, QueueJob
from app.services.camera_pose_estimator import CameraPose, ReconstructionResult, SparsePoint3D
from app.services.depth_generator import StereoParams, generate_view_depths
from app.services.depth_refinement import RefineParams, refine_depth
from app.services.georeferencing import (
    align_to_enu,
    analyze_gps_track,
    crs_metadata,
    geodetic_to_ecef,
    wgs84_to_enu,
)
from app.services import job_queue
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.sparse_reconstruction import _write_poses_json, _write_sparse_ply
from app.services.streaming_engine import engine

# ---------------------------------------------------------------------------
# Synthetic nadir plane-scene helpers (metric, known poses)
# ---------------------------------------------------------------------------

_LOOK_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
_Z = 10.0
_SIZE = 96
_F = 60.0


def _camera_matrix() -> np.ndarray:
    return np.array([[_F, 0, (_SIZE - 1) / 2], [0, _F, (_SIZE - 1) / 2], [0, 0, 1]], dtype=np.float64)


def _scene_images(offsets=(0.0, 0.6, 1.2)) -> list[np.ndarray]:
    """Nadir views of one shared textured plane (real parallax between views)."""
    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(rng.integers(0, 255, (_SIZE * 3, _SIZE * 3), dtype=np.uint8), (3, 3), 0)
    imgs = []
    for ox in offsets:
        # Camera translated +ox sees world content shifted left by ox*f/z px.
        dx = -ox * _F / _Z
        imgs.append(cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, 0]]), (_SIZE, _SIZE),
                                   borderMode=cv2.BORDER_REPLICATE))
    return imgs


def _poses_and_images(workspace: Path, offsets=(0.0, 0.6, 1.2)) -> list[dict]:
    img_dir = workspace / "selected"
    img_dir.mkdir(parents=True, exist_ok=True)
    poses = []
    for i, (ox, img) in enumerate(zip(offsets, _scene_images(offsets))):
        frame_id = f"frame_{i:06d}"
        cv2.imwrite(str(img_dir / f"{frame_id}.jpg"), img)
        poses.append({
            "frame_id": frame_id,
            "K": _camera_matrix().tolist(),
            "R": _LOOK_DOWN.tolist(),
            "t": [ox, 0.0, _Z],
        })
    (workspace / "poses.json").write_text(json.dumps({"frames": poses}))
    return poses


def _seed_job(tmp_path: Path, job_id: str) -> Path:
    """A job workspace with selected frames + sparse poses but no depth yet."""
    monkeypatch_storage(tmp_path)
    workspace = settings.storage.project_dir(job_id)
    _poses_and_images(workspace)
    return workspace


def monkeypatch_storage(tmp_path: Path) -> None:
    settings.storage.base_path = str(tmp_path)


def _stereo() -> StereoParams:
    return StereoParams(min_disparity=0, num_disparities=32, block_size=5)


def test_auto_backend_prefers_monocular_depth_anything(monkeypatch):
    """Regression: auto must match the monocular-video physics.

    Consecutive frames of forward-moving monocular video are NOT a calibrated
    stereo pair — horizontal disparity is near zero and SGBM depth explodes to
    subpixel-noise distances (observed up to ~10 km on real drone footage).
    When the Depth Anything V2 checkpoint is available, auto must select it
    (relative depth aligned to SfM by the global affine fit); only when the
    model is genuinely unavailable does auto fall back to stereo.
    """
    import app.services.depth_generator as dg

    monkeypatch.setattr(dg, "_backend_available", lambda name: name == "depth_anything")
    assert dg._choose_backend("auto") == "depth_anything"

    # Checkpoint absent → stereo is the only remaining metric-ish source.
    monkeypatch.setattr(dg, "_backend_available", lambda name: False)
    assert dg._choose_backend("auto") == "stereo"


def test_stereo_backend_does_not_fallback_to_depth_anything(monkeypatch, tmp_path):
    """Regression: a failed stereo pass must not be masked by a learned-depth fallback.

    The metric-safe depth policy should fail explicitly when stereo cannot
    produce a view. It must not silently substitute a relative-depth branch
    that cannot prove an exact model.
    """
    import app.services.depth_generator as dg

    workspace = _seed_job(tmp_path, "depthjob_no_fallback")

    monkeypatch.setattr(dg, "_backend_available", lambda name: name == "depth_anything")
    monkeypatch.setattr(dg, "_stereo_view_depth", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("stereo failed")))
    depth_anything_called = {"called": False}

    def fake_depth_anything(*args, **kwargs):
        depth_anything_called["called"] = True
        return np.ones((1, 1), dtype=np.float32), 0.0, False

    monkeypatch.setattr(dg, "_depth_anything_view_depth", fake_depth_anything)

    summary = dg.generate_view_depths(workspace, backend="stereo", stereo=_stereo())

    assert summary.backend == "stereo"
    assert summary.failed == ["frame_000000", "frame_000001", "frame_000002"]
    assert depth_anything_called["called"] is False


# ---------------------------------------------------------------------------
# Per-view depth generation
# ---------------------------------------------------------------------------


class TestDepthGeneration:
    def test_stereo_depth_has_no_missing_value_sentinel(self, tmp_path: Path):
        """SGBM's unmatched-pixel marker must never reach the stored map.

        With handleMissingValues=True, pixels outside SGBM's disparity range
        (including the min_disp-1 invalid marker in the negative-disparity
        branch) reproject to the (0,0,10000) substitute — a 10 km depth that
        the depth diagnostics audit rightly rejects. Regression for the e2e
        flake where 2 of 9 maps carried ~15k sentinel pixels each.
        """
        workspace = _seed_job(tmp_path, "depthjob_sentinel")
        summary = generate_view_depths(workspace, backend="stereo", stereo=_stereo())
        assert summary.generated
        for fid in summary.generated:
            depth = np.load(workspace / "depth" / f"{fid}.npy")
            assert int((depth >= 9999.0).sum()) == 0, fid
            assert float(depth.max()) < 1e4, fid

    def test_stereo_depth_accuracy(self, tmp_path: Path):
        workspace = _seed_job(tmp_path, "depthjob1")
        depth_dir = workspace / "depth"
        summary = generate_view_depths(workspace, backend="stereo", stereo=_stereo())
        assert summary.backend == "stereo"
        # Every view with a forward neighbour gets depth; the trailing view of
        # the open trajectory legitimately has none (SGBM needs positive
        # disparity on this build).
        assert set(summary.generated) == {"frame_000000", "frame_000001"}
        assert not summary.failed
        for fid in summary.generated:
            depth = np.load(depth_dir / f"{fid}.npy")
            valid = depth > 0
            assert valid.mean() > 0.2, fid
            rel_err = np.abs(depth[valid] - _Z) / _Z
            assert float(np.median(rel_err)) < 0.25, fid
            assert float(np.percentile(rel_err, 90)) < 0.4, fid
            # Sidecar metadata was written.
            meta = json.loads((depth_dir / f"{fid}.json").read_text())
            assert meta["backend"] == "stereo"
            assert meta["inference_time_ms"] > 0
            assert 0 <= meta["confidence"] <= 1
            # Artifact of record: float32 .npy + JSON sidecar. The legacy
            # 16-bit mm .png had no consumer anywhere and was removed.
            npy = depth_dir / f"{fid}.npy"
            assert npy.exists() and npy.stat().st_size > 0

    def test_depth_cache_reuse(self, tmp_path: Path):
        workspace = _seed_job(tmp_path, "depthjob2")
        first = generate_view_depths(workspace, backend="stereo", stereo=_stereo())
        second = generate_view_depths(workspace, backend="stereo", stereo=_stereo())  # noqa: F841
        assert second.count_generated == 0
        assert set(second.cached) == set(first.generated)

    def test_missing_source_image_is_named_not_opaque(self, tmp_path: Path):
        """Regression (user-facing "Criteria A-G" refusal): a registered pose
        whose image is not on disk used to die inside ``infer_image(None)``
        with an opaque TypeError ~0.1 s in, and the stage reported a bare
        frame id in ``failed``. The reason must be explicit and the stage
        must keep generating the views it CAN produce."""
        workspace = _seed_job(tmp_path, "depthjob_missing_img")
        (workspace / "selected" / "frame_000000.jpg").unlink()

        summary = generate_view_depths(workspace, backend="stereo", stereo=_stereo())

        assert "frame_000000" in summary.failed
        reason = summary.failure_reasons["frame_000000"]
        assert "ViewSourceMissing" in reason
        assert "source image not found" in reason
        # A missing frame is a failure, never a silent exclusion.
        assert "frame_000000" not in summary.excluded
        assert summary.to_dict()["failure_reasons"] == summary.failure_reasons
        # The remaining views still produce maps.
        assert summary.count_generated >= 1

    def test_frames_split_across_selected_and_frames_dirs(self, tmp_path: Path):
        """A re-selected keyframe budget can leave ``selected/`` and
        ``frames/`` out of step, so a registered pose's image may live in the
        other directory. Resolving against both keeps the view mappable."""
        workspace = _seed_job(tmp_path, "depthjob_split_dirs")
        frames_dir = workspace / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        # Move only the LAST frame's image to the secondary directory.
        src = workspace / "selected" / "frame_000002.jpg"
        (frames_dir / src.name).write_bytes(src.read_bytes())
        src.unlink()

        summary = generate_view_depths(workspace, backend="stereo", stereo=_stereo())

        assert "frame_000000" not in summary.failed
        assert summary.count_generated >= 1
        assert set(summary.failed) <= {"frame_000002"} or not summary.failed

    def test_unknown_backend_raises(self, tmp_path: Path):
        workspace = _seed_job(tmp_path, "depthjob3")
        from app.exceptions import COLMAPError, ModelNotAvailableError
        from app.services.depth_anything_v2 import find_checkpoint

        with pytest.raises(COLMAPError):
            generate_view_depths(workspace, backend="colmap")
        if find_checkpoint() is None:
            # No checkpoint installed: the explicit request must raise.
            with pytest.raises(ModelNotAvailableError):
                generate_view_depths(workspace, backend="depth_anything")
        else:
            # A real checkpoint is installed: the explicit request must run.
            summary = generate_view_depths(workspace, backend="depth_anything", max_views=1)
            assert summary.backend == "depth_anything"


class TestMissingImageRegeneration:
    """Registered poses whose source image is missing must become mappable
    again by regenerating exactly those frames from the source video — never
    re-extracting the video, never touching frames that exist — and views
    whose regeneration is impossible keep their honest named failure."""

    def test_missing_frames_regenerate_and_receive_depth_maps(self, tmp_path):
        """A deleted frame is healed from the video before inference, then
        flows into depth generation like any other view."""
        import cv2 as _cv2

        workspace = _seed_job(tmp_path, "depthjob_regen")
        offsets = (0.0, 0.6, 1.2)
        # A synthetic video whose decode index i holds the SAME image the
        # seed wrote for candidate i — seek_to_index regenerates bit-identical
        # input, so the stereo depth result must be unchanged for the healed view.
        video_path = workspace / "video.avi"
        w = _cv2.VideoWriter(str(video_path), _cv2.VideoWriter_fourcc(*"MJPG"), 30, (_SIZE, _SIZE))
        for img in _scene_images(offsets):
            w.write(_cv2.cvtColor(img, _cv2.COLOR_GRAY2BGR))
        w.release()
        (workspace / "quality_report.json").write_text(json.dumps({
            "video_path": str(video_path),
            "frames": [
                {"filename": f"frame_{i:06d}.jpg", "frame_num": i}
                for i in range(len(offsets))
            ],
        }))
        # One view's source image vanished (pruned/renamed selected/).
        (workspace / "selected" / "frame_000001.jpg").unlink()

        summary = generate_view_depths(workspace, backend="stereo", stereo=_stereo())

        # The missing view was healed and mapped, not failed.
        assert "frame_000001" not in summary.failed
        assert "frame_000001" in summary.regenerated
        assert "frame_000001" in summary.generated
        # The healing is reported on the payload the Processing page renders.
        payload = summary.to_dict()
        assert payload["regenerated"] == summary.regenerated
        assert payload["regeneration_failures"] is None
        # Frames that existed were never touched by the healing.
        assert set(summary.regenerated) == {"frame_000001"}
        # The healed map is a real depth map of the synthetic scene.
        depth = np.load(workspace / "depth" / "frame_000001.npy")
        assert (depth > 0).mean() > 0.2

    def test_regeneration_impossible_failure_still_named(self, tmp_path):
        """No video in the workspace → the view stays a named failure, and
        the reason says regeneration was attempted and impossible."""
        workspace = _seed_job(tmp_path, "depthjob_regen_impossible")
        (workspace / "selected" / "frame_000000.jpg").unlink()

        summary = generate_view_depths(workspace, backend="stereo", stereo=_stereo())

        assert "frame_000000" in summary.failed
        reason = summary.failure_reasons["frame_000000"]
        assert "ViewSourceMissing" in reason
        assert "source image not found" in reason
        # The attempt itself is on the record, with the per-frame reason.
        assert summary.regenerated == []
        assert "frame_000000" in summary.regeneration_failures
        assert "regeneration impossible" in summary.regeneration_failures["frame_000000"]
        assert summary.to_dict()["regeneration_failures"]["frame_000000"].startswith("no source video")

    def test_no_op_when_all_images_present(self, tmp_path):
        """Every image on disk: no video read, no healing fields reported."""
        workspace = _seed_job(tmp_path, "depthjob_regen_noop")

        summary = generate_view_depths(workspace, backend="stereo", stereo=_stereo())

        assert summary.regenerated == []
        assert summary.regeneration_failures == {}
        payload = summary.to_dict()
        assert payload["regenerated"] is None
        assert payload["regeneration_failures"] is None
        assert summary.count_generated >= 1
        assert not summary.failed

    def test_existing_frames_are_never_overwritten_by_healing(self, tmp_path):
        """The heal writes only absent files: an existing frame's bytes are
        identical before and after the stage runs."""
        import cv2 as _cv2

        workspace = _seed_job(tmp_path, "depthjob_regen_noover")
        video_path = workspace / "video.avi"
        w = _cv2.VideoWriter(str(video_path), _cv2.VideoWriter_fourcc(*"MJPG"), 30, (_SIZE, _SIZE))
        for img in _scene_images((0.0, 0.6, 1.2)):
            w.write(np.full((_SIZE, _SIZE, 3), 255, np.uint8))  # different content
        w.release()
        (workspace / "quality_report.json").write_text(json.dumps({
            "video_path": str(video_path),
            "frames": [
                {"filename": f"frame_{i:06d}.jpg", "frame_num": i}
                for i in range(3)
            ],
        }))
        before = {
            p.name: p.read_bytes()
            for p in sorted((workspace / "selected").glob("*.jpg"))
        }
        generate_view_depths(workspace, backend="stereo", stereo=_stereo())

        after = {
            p.name: p.read_bytes()
            for p in sorted((workspace / "selected").glob("*.jpg"))
        }
        assert before == after


# ---------------------------------------------------------------------------
# Depth refinement
# ---------------------------------------------------------------------------


class TestDepthRefinement:
    def test_filters_reduce_noise_and_fill_holes(self):
        depth = np.full((64, 64), 10.0, dtype=np.float32)
        rng = np.random.default_rng(0)
        noisy = depth + rng.normal(0, 0.5, depth.shape).astype(np.float32)
        noisy[::6, :] = 0.0  # stripe holes
        params = RefineParams(median=True, median_k=5, hole_fill=True, min_depth=1.0, max_depth=50.0)
        refined = refine_depth(noisy, params)
        valid_before = noisy > 0
        assert float(np.abs(refined[valid_before] - 10.0).mean()) < float(np.abs(noisy[valid_before] - 10.0).mean())
        assert float((refined <= 0).mean()) < float((noisy <= 0).mean())  # holes filled
        assert refined.max() <= 50.0

    def test_disabled_filters_are_noop(self):
        depth = np.full((32, 32), 5.0, dtype=np.float32)
        assert np.array_equal(refine_depth(depth, RefineParams()), depth)

    def test_no_invention_where_no_data(self):
        depth = np.zeros((64, 64), dtype=np.float32)
        depth[20:44, 20:44] = 10.0
        refined = refine_depth(depth, RefineParams(hole_fill=True, median=True, min_depth=1.0, max_depth=50.0))
        # The blank border has no valid neighbours within reach of the filters.
        assert float(refined[:10, :].max()) == 0.0


# ---------------------------------------------------------------------------
# Georeferencing
# ---------------------------------------------------------------------------


class TestGeoreferencing:
    def test_wgs84_to_ecef_origin(self):
        ecef = geodetic_to_ecef(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert np.allclose(ecef[0], [6378137.0, 0.0, 0.0], atol=0.01)

    def test_enu_east_step_at_equator(self):
        enu = wgs84_to_enu(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1e-4]),
            np.array([0.0, 0.0, 0.0]),
            0.0, 0.0, 0.0,
        )
        assert abs(enu[2, 0] - 11.13) < 0.02  # ~111.3 m per degree at equator
        assert abs(enu[2, 1]) < 1e-6

    def test_umeyama_recovers_similarity(self):
        src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.7, 1.2]])
        scale, rot = 2.0, np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        tgt = src @ (scale * rot).T + np.array([10.0, -5.0, 3.0])
        m, est_scale = align_to_enu(src, tgt)
        transformed = np.hstack([src, np.ones((len(src), 1))]) @ m.T
        assert abs(est_scale - scale) < 1e-6
        assert np.allclose(transformed[:, :3], tgt, atol=1e-6)

    def test_gps_quality_detects_jumps(self):
        clean = analyze_gps_track(
            np.zeros(10), np.linspace(0, 5e-4, 10), np.full(10, 100.0)
        )
        jumpy = analyze_gps_track(
            np.zeros(11),
            np.array([0.0, 5e-5, 1e-4, 0.5, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 4.5e-4, 5e-4]),
            np.full(11, 100.0),
        )
        assert clean.gps_score > 95
        assert jumpy.gps_score < clean.gps_score
        assert jumpy.discontinuities >= 2
        assert jumpy.suggestions  # actionable text present
        assert jumpy.grade in ("Excellent", "Good", "Fair", "Poor")
        assert set(crs_metadata(47.0, 8.0)) == {"horizontal_crs", "anchor_wgs84", "units", "note"}


# ---------------------------------------------------------------------------
# Sparse output persistence (dense input contract)
# ---------------------------------------------------------------------------


class TestSparsePersistence:
    def test_poses_and_ply_written(self, tmp_path: Path):
        result = ReconstructionResult(num_registered=1, num_points=2)
        result.cameras = {
            "frame_000000": CameraPose(
                image_id=0, frame_id="frame_000000",
                position=np.array([1.0, 2.0, 3.0]),
                rotation=np.eye(3), quaternion=np.array([1.0, 0, 0, 0]),
                intrinsics=np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]),
                distortion=np.zeros(5), is_estimated=True,
            )
        }
        result.points3d = [
            SparsePoint3D(point_id=i, position=np.array([i, i, i], dtype=np.float64),
                          color=np.array([10, 20, 30], dtype=np.uint8), track_length=3)
            for i in range(3)
        ]
        _write_poses_json(result, tmp_path)
        _write_sparse_ply(result, tmp_path)

        poses = json.loads((tmp_path / "poses.json").read_text())["frames"]
        assert len(poses) == 1
        assert poses[0]["frame_id"] == "frame_000000"
        assert len(poses[0]["K"]) == 3 and len(poses[0]["t"]) == 3

        from app.services.pointcloud import read_ply

        cloud = read_ply(tmp_path / "sparse_model.ply")
        assert cloud.n == 3
        assert np.allclose(cloud.xyz, [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        assert cloud.rgb is not None


# ---------------------------------------------------------------------------
# Autonomous orchestrator
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_full_chain_from_seeded_workspace(self, tmp_path: Path):
        workspace = _seed_job(tmp_path, "orchjob1")
        req = PipelineRequest(depth_backend="auto", stereo=_stereo())
        report = run_autonomous_pipeline("orchjob1", req)

        assert report["status"] == "completed", report.get("error")
        stages = report["stages"]
        # frames/sparse already done -> skipped; the rest actually executed.
        assert stages["frames"]["status"] == "skipped"
        assert stages["sparse"]["status"] == "skipped"
        assert stages["depth"]["status"] == "completed"
        assert stages["dense"]["status"] == "completed"
        assert stages["georef"]["status"] == "completed"
        assert stages["depth"]["count"] in (2, 3)  # 2 if stereo (trailing view skipped), 3 if depth_anything
        assert stages["dense"]["count"] > 100
        assert "no GPS" in stages["georef"]["detail"]["note"]
        assert report["profile"]["bottleneck_stage"] in ("depth", "dense", "georef")
        assert (workspace / "dense_report.json").exists()
        assert (workspace / "dense" / "dense_model.ply").exists()

    def test_rerun_resumes_and_skips(self, tmp_path: Path):
        _seed_job(tmp_path, "orchjob2")
        req = PipelineRequest(depth_backend="auto", stereo=_stereo())
        first = run_autonomous_pipeline("orchjob2", req)
        assert first["status"] == "completed"

        second = run_autonomous_pipeline("orchjob2", PipelineRequest(depth_backend="auto", stereo=_stereo()))
        assert second["status"] == "completed"
        statuses = [s["status"] for s in second["stages"].values()]
        # Four cached stages are skipped; georef has no artifact and reruns.
        assert statuses == ["skipped", "skipped", "skipped", "skipped", "completed"]
        assert list(second["profile"]["stage_durations_ms"]) == ["georef"]

    def test_cancelled_before_run(self, tmp_path: Path):
        _seed_job(tmp_path, "orchjob3")
        engine.request_cancel("orchjob3")
        try:
            report = run_autonomous_pipeline("orchjob3", PipelineRequest(depth_backend="auto", stereo=_stereo()))
            assert report["status"] == "cancelled"
            events = [e["type"] for e in engine.replay("orchjob3")]
            assert "cancel_requested" in events
        finally:
            engine.clear_cancel("orchjob3")


# ---------------------------------------------------------------------------
# End-to-end from a real video file (no seeded artifacts)
# ---------------------------------------------------------------------------


def _write_synthetic_video(path: Path, frames: int = 20) -> None:
    """A tiny nadir mission: one textured plane, camera sliding sideways.

    Pure translation keeps SfM solvable and gives real stereo parallax for the
    depth stage, while the random texture guarantees feature support and passes
    the entropy-based quality filter. MJPG/AVI works without FFmpeg.
    """
    rng = np.random.default_rng(11)
    base = cv2.GaussianBlur(rng.integers(0, 255, (960, 1280), dtype=np.uint8), (3, 3), 0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 480))
    assert writer.isOpened(), "VideoWriter failed to open"
    for i in range(frames):
        gray = cv2.warpAffine(base, np.float32([[1, 0, -4 * i], [0, 1, 0]]), (640, 480),
                              borderMode=cv2.BORDER_REPLICATE)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()


class TestEndToEndFromVideo:
    """Full video → artifacts chain: frames, sparse, depth, dense, georef.

    Unlike TestOrchestrator (which seeds poses.json and skips the front of the
    pipeline), this drives the real stages from a synthetic video so a broken
    stage — e.g. georef reporting success with an empty output directory —
    fails the suite instead of shipping.
    """

    def test_every_stage_produces_its_artifact(self, tmp_path: Path):
        monkeypatch_storage(tmp_path)
        job_id = "e2e_video_1"
        workspace = settings.storage.project_dir(job_id)
        workspace.mkdir(parents=True, exist_ok=True)
        _write_synthetic_video(workspace / "synthetic_mission.avi")

        report = run_autonomous_pipeline(
            job_id,
            # Stereo is requested explicitly: this synthetic video's COLMAP
            # reconstruction is degenerate (near-constant sparse depths), for
            # which the auto depth_anything path's global inverse-depth fit
            # is rightly rejected and the per-view fallback produces
            # view-inconsistent depth that the dense-stage diagnostics audit
            # now honestly refuses to fuse (see test_depth_diagnostics_honesty).
            # The chain-completeness contract is backend-agnostic.
            PipelineRequest(depth_backend="stereo", stereo=_stereo(),
                            extraction_mode="every_n", every_n=2),
        )

        assert report["status"] == "completed", report.get("error")
        stages = report["stages"]
        for name in ("frames", "sparse", "depth", "dense", "georef"):
            assert stages[name]["status"] == "completed", (name, stages[name])

        # frames — entropy-selected JPGs on disk
        selected = sorted((workspace / "selected").glob("*.jpg"))
        assert len(selected) >= 3, "quality filter rejected the whole synthetic scene"

        # sparse — one pose per selected frame + a non-empty point cloud
        poses = json.loads((workspace / "poses.json").read_text())["frames"]
        assert len(poses) == stages["sparse"]["count"] >= 2
        assert (workspace / "sparse_model.ply").stat().st_size > 0

        # depth — at least one real depth map
        depth_maps = sorted((workspace / "depth").glob("*.npy"))
        assert depth_maps and all(p.stat().st_size > 0 for p in depth_maps)

        # depth provenance — the served stage detail records the checkpoint/
        # parameter tag actually written into the per-frame sidecars, never a
        # name inferred from the backend id.
        depth_detail = stages["depth"]["detail"]
        sidecar = json.loads((workspace / "depth" / f"{depth_maps[0].stem}.json").read_text())
        assert depth_detail.get("checkpoint") == sidecar.get("model_version")
        assert depth_detail.get("checkpoint"), "stage detail must carry the actual model tag"

        # dense — fused cloud artifact exists and is non-trivial
        dense_ply = workspace / "dense" / "dense_model.ply"
        assert dense_ply.stat().st_size > 0
        assert stages["dense"]["count"] > 0

        # georef — synthetic video has no GPS: honest skip, never an empty dir
        assert "no GPS" in stages["georef"]["detail"]["note"]
        georef_dir = workspace / "georef"
        assert not georef_dir.exists() or any(georef_dir.iterdir()), \
            "georef reported success but wrote no artifacts"

        # the whole "stage succeeded but wrote nothing" bug class. (rejected/
        # is exempt: it's empty exactly when every frame passes quality.)
        stage_output_dirs = ["depth", "dense", "georef", "mesh"]
        for name in stage_output_dirs:
            d = workspace / name
            if d.is_dir():
                assert any(d.iterdir()), f"empty stage directory: {name}"

        # persisted reporting artifacts the API layer serves
        assert (workspace / "pipeline_report.json").exists()
        manifest = json.loads((workspace / "manifest.json").read_text())
        assert manifest["run_id"] == job_id
        # Provenance: runs whose workspace has no source.json (this e2e seeds
        # the video directly) must say so honestly, never guess.
        assert manifest["source"] == {"kind": "unknown"}

    def test_manifest_carries_source_record(self, tmp_path: Path):
        """A workspace with a provenance sidecar promotes it to the manifest."""
        from app.services.provenance import write_source_record

        monkeypatch_storage(tmp_path)
        job_id = "e2e_prov_1"
        workspace = settings.storage.project_dir(job_id)
        workspace.mkdir(parents=True, exist_ok=True)
        _write_synthetic_video(workspace / "synthetic_mission.avi")
        write_source_record(
            workspace,
            kind="synthetic_test",
            original_filename="synthetic_mission.avi",
            stored_path=workspace / "synthetic_mission.avi",
        )

        report = run_autonomous_pipeline(
            job_id,
            PipelineRequest(depth_backend="stereo", stereo=_stereo(),
                            extraction_mode="every_n", every_n=2),
        )
        # The manifest (and its source record) is written on EVERY terminal
        # status — completed or honestly failed — because _finish always runs.
        assert report["status"] in ("completed", "failed", "cancelled")
        manifest = json.loads((workspace / "manifest.json").read_text())
        assert manifest["run_id"] == job_id
        src = manifest["source"]
        assert src["kind"] == "synthetic_test"
        assert src["original_filename"] == "synthetic_mission.avi"
        assert len(src["sha256"]) == 64
        assert src["file_size_bytes"] > 0

    def test_server_path_sparse_sampling_fails_honestly(self, tmp_path: Path):
        """Regression for the server-ingest-vs-in-process divergence.

        The REST start path defaults to every_n=10 while the in-process e2e
        uses every_n=2. On a 20-frame synthetic clip the server parameters
        select 2 frames, which genuinely cannot register in SfM. The bug was
        that this used to be reported as sparse status='completed' with count
        0 — it must FAIL with the zero-camera diagnostic instead. (In-process
        run of the exact server parameters; no ingest divergence exists —
        same bytes, same stages, only the sampling density differs.)
        """
        monkeypatch_storage(tmp_path)
        job_id = "e2e_server_params"
        workspace = settings.storage.project_dir(job_id)
        workspace.mkdir(parents=True, exist_ok=True)
        _write_synthetic_video(workspace / "synthetic_mission.avi")

        report = run_autonomous_pipeline(
            job_id,
            PipelineRequest(depth_backend="stereo", stereo=_stereo(),
                            extraction_mode="every_n", every_n=10),
        )
        assert report["status"] == "failed"
        assert report["stages"]["sparse"]["status"] == "failed"
        assert "zero registered cameras" in report.get("error", "")
        # No downstream stage claimed success after the failed stage.
        assert (workspace / "poses.json").exists() is False

    def test_manifest_telemetry_mode_from_sparse_report(self, tmp_path: Path):
        """A cache-hit sparse stage must not re-label telemetry placement.

        When sparse was telemetry-placed and georef never ran, the pipeline
        report's sparse detail is only {reason, cache_hit} — the manifest must
        fall back to the sparse stage's on-disk localization report instead of
        claiming VIDEO_ONLY for a metrically placed reconstruction.
        """
        from app.services.pipeline_orchestrator import _sparse_localization_mode

        monkeypatch_storage(tmp_path)
        job_id = "e2e_sparse_mode"
        workspace = settings.storage.project_dir(job_id)
        workspace.mkdir(parents=True, exist_ok=True)

        # Cache-hit sparse detail + on-disk telemetry-placed report.
        report = {"stages": {"sparse": {"detail": {
            "reason": "artifact_present", "cache_hit": True}}}}
        (workspace / "sparse_rerun_report.json").write_text(json.dumps({
            "localization": {"mode": "telemetry_assisted"}}))
        assert _sparse_localization_mode(report, workspace) == "telemetry_assisted"

        # No evidence anywhere -> None -> manifest falls through to georef's
        # telemetry_mode (honest VIDEO_ONLY for a genuinely video-only run).
        bare = {"stages": {"sparse": {"detail": {"cache_hit": True}}}}
        assert _sparse_localization_mode(bare, workspace.parent) is None

        # Corrupt on-disk report -> skipped, not a crash.
        (workspace / "sparse_rerun_report.json").write_text("{not json")
        assert _sparse_localization_mode(bare, workspace) is None

        # Direct detail (fresh sparse run) still wins over the on-disk file.
        fresh = {"stages": {"sparse": {"detail": {
            "localization": {"mode": "video_only"}}}}}
        assert _sparse_localization_mode(fresh, workspace) == "video_only"


# ---------------------------------------------------------------------------
# REST integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_api_flow(client, db_session: AsyncSession, tmp_path: Path, monkeypatch):
    """POST /api/pipeline/start enqueues on the durable queue and returns
    immediately; the queue worker then runs the real chain."""
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path))
    job_id = "pipejob0001"
    db_session.add(Project(id=job_id, name="plane.avi", video_filename="plane.avi",
                           video_path=str(tmp_path / "plane.avi"), status="uploaded"))
    await db_session.flush()
    _poses_and_images(settings.storage.project_dir(job_id))

    resp = await client.post(
        "/api/pipeline/start/pipejob0001",
        json={"depth_backend": "auto", "stereo": {"num_disparities": 32, "block_size": 5}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "queued", body
    assert "queued" in body["message"].lower()

    # Exactly one durable job carries the run parameters verbatim.
    rows = (await db_session.execute(
        select(QueueJob).where(QueueJob.project_id == job_id)
    )).scalars().all()
    assert len(rows) == 1
    payload = json.loads(rows[0].payload_json)
    assert payload["depth_backend"] == "auto"
    assert payload["stereo"]["num_disparities"] == 32

    # The queue worker runs the pipeline to completion.
    processed = await job_queue.process_next(db_session)
    assert processed is not None and processed.status == "completed", processed.error

    status = await client.get("/api/pipeline/status/pipejob0001")
    assert status.status_code == 200
    s = status.json()
    assert s["status"] == "completed"
    assert s["resume"]["dense"] is True
    assert s["profile"]["stage_count"] >= 1

    # DB project was updated by the queue settlement.
    row = (await db_session.execute(select(Project).where(Project.id == job_id))).scalar_one()
    assert row.status == "pipeline_completed"

    # Dense model + report artifacts exist and the dense download works.
    workspace = settings.storage.project_dir(job_id)
    assert (workspace / "pipeline_report.json").exists()
    assert (workspace / "dense" / "dense_model.ply").exists()
    dl = await client.get("/api/dense/download/pipejob0001?format=ply")
    assert dl.status_code == 200 and dl.content.startswith(b"ply")


@pytest.mark.asyncio
async def test_pipeline_start_returns_before_executor_finishes(
    client, db_session: AsyncSession, tmp_path: Path, monkeypatch
):
    """Regression: start returns immediately even when the executor is slow.

    The original synchronous route ran the whole pipeline inside the HTTP
    request; browsers cut the connection and reported "Cannot connect to the
    local STRATA engine" mid-reconstruction. The response must arrive before
    the executor completes, whatever the executor's duration.
    """
    import asyncio as _aio
    from app.services import job_queue as jq

    async def slow_executor(job, db):
        await _aio.sleep(30)  # far longer than any HTTP round trip
        return {"status": "completed"}

    monkeypatch.setattr(jq, "EXECUTORS", {"pipeline": slow_executor})
    job_id = "slowjob0001"
    db_session.add(Project(id=job_id, name="slow.avi", video_filename="slow.avi",
                           video_path=str(tmp_path / "slow.avi"), status="uploaded"))
    await db_session.flush()

    import time
    t0 = time.monotonic()
    resp = await client.post(
        f"/api/pipeline/start/{job_id}", json={"depth_backend": "auto"}
    )
    elapsed = time.monotonic() - t0
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
    assert elapsed < 5, f"start took {elapsed:.1f}s — it is running the pipeline inline"

    # The queued job exists and is still pending (executor not yet run).
    row = (await db_session.execute(
        select(QueueJob).where(QueueJob.project_id == job_id)
    )).scalar_one()
    assert row.status == "queued"

    # Start emits only the enqueue event; stage events come from the worker
    # once the queue executes the job (asserted in test_pipeline_api_flow).
    events = [e["type"] for e in engine.replay(job_id)]
    assert "pipeline_started" in events
    assert "stage:depth" not in events

    missing = await client.get("/api/pipeline/status/doesnotexist")
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# Live status derivation (GET /api/pipeline/status during a run)
# ---------------------------------------------------------------------------


async def _publish_live_history(job_id: str) -> None:
    """Publish a realistic mid-run event history through the real engine."""
    engine.publish(job_id, "pipeline_started", {"requested_stages": []})
    engine.publish(job_id, "stage:frames", {"status": "running", "progress": 0.0})
    engine.publish(job_id, "stage:frames", {"status": "completed", "progress": 1.0, "count": 14, "duration_ms": 5210.4})
    engine.publish(job_id, "stage:sparse", {"status": "running", "progress": 0.0})
    engine.publish(job_id, "stage:sparse", {"status": "completed", "progress": 1.0, "count": 2930, "duration_ms": 67483.1})
    engine.publish(job_id, "stage:depth", {"status": "running", "progress": 0.4})


@pytest.mark.asyncio
async def test_status_reports_live_stages_mid_run(client, db_session: AsyncSession, tmp_path: Path, monkeypatch):
    """During a run (no report file yet) the status endpoint reports the live
    stage events — real statuses and the real progress fraction — instead of
    a bare ``not_run``."""
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path))
    job_id = "livejob0001"
    db_session.add(Project(id=job_id, name="v.avi", video_filename="v.avi",
                           video_path=str(tmp_path / "v.avi"), status="uploaded"))
    await db_session.flush()

    await _publish_live_history(job_id)

    resp = await client.get(f"/api/pipeline/status/{job_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "running"
    assert body["stages"]["frames"]["status"] == "completed"
    assert body["stages"]["sparse"]["count"] == 2930
    assert body["stages"]["depth"]["status"] == "running"
    assert body["stages"]["depth"]["progress"] == 0.4  # the real backend fraction


@pytest.mark.asyncio
async def test_status_live_wins_over_stale_failed_report(client, db_session: AsyncSession, tmp_path: Path, monkeypatch):
    """A retry started after a failed run publishes newer events than the old
    report file — the live state must win, or the UI would show FAILED for the
    whole second run."""
    import time

    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path))
    job_id = "retryjob001"
    done_id = "retryjob002"  # finished run: report exists, no live events
    db_session.add(Project(id=job_id, name="v.avi", video_filename="v.avi",
                           video_path=str(tmp_path / "v.avi"), status="failed"))
    db_session.add(Project(id=done_id, name="w.avi", video_filename="w.avi",
                           video_path=str(tmp_path / "w.avi"), status="pipeline_completed"))
    await db_session.flush()

    # A finished (failed) run's report, written before the retry's events.
    workspace = settings.storage.project_dir(job_id)
    workspace.mkdir(parents=True, exist_ok=True)
    report = {
        "job_id": job_id,
        "status": "failed",
        "error": "Depth maps failed diagnostic quality audit (Criteria A-G)",
        "stages": {"depth": {"name": "depth", "status": "failed", "duration_ms": 1.0,
                             "count": 0, "error": "audit", "detail": {}}},
        "profile": {},
        "resume": {},
    }
    report_path = workspace / "pipeline_report.json"
    report_path.write_text(json.dumps(report))

    # A finished (successful) run's report for the no-live-events job.
    done_ws = settings.storage.project_dir(done_id)
    done_ws.mkdir(parents=True, exist_ok=True)
    done_report = dict(report, job_id=done_id, status="completed", error="")
    (done_ws / "pipeline_report.json").write_text(json.dumps(done_report))

    await _publish_live_history(job_id)  # retry in progress (timestamps now > report mtime)
    assert time.time() > report_path.stat().st_mtime

    resp = await client.get(f"/api/pipeline/status/{job_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "running"  # not the stale "failed"
    assert body["stages"]["depth"]["status"] == "running"

    # A finished run with no newer events still serves its report.
    finished = await client.get(f"/api/pipeline/status/{done_id}")
    assert finished.status_code == 200
    assert finished.json()["status"] == "completed"


# ---------------------------------------------------------------------------
# placement provenance in the stage detail
# ---------------------------------------------------------------------------


def test_placement_summary_reports_a_refused_piecewise_correction():
    """A refused per-window placement must reach the stage detail.

    The sparse stage SUCCEEDS on the fallback, so without this block a
    degraded placement (measured: retriangulation kept 355 of 52,882 points)
    leaves no trace anywhere the user can see.
    """
    from app.services.pipeline_orchestrator import placement_summary

    report = {
        "trajectory_alignment": {
            "placement": {
                "placement_mode": "global_similarity",
                "cameras_matched": 89,
                "match_percent": 100.0,
                "applied": {"points_retained_fraction": 0.0067},
                "piecewise": {
                    "refused": {
                        "reason": "low_point_retention",
                        "points_before": 52882,
                        "points_after": 52882,
                        "points_retained_fraction": 0.0067,
                        "retained_floor": 0.3,
                        "observations_dropped": 569903,
                    }
                },
            }
        }
    }
    summary = placement_summary(report)
    assert summary["mode"] == "global_similarity"
    assert summary["refused"]["reason"] == "low_point_retention"
    assert summary["refused"]["observations_dropped"] == 569903
    assert summary["points_retained_fraction"] == 0.0067


def test_placement_summary_is_empty_but_shaped_for_a_video_only_run():
    from app.services.pipeline_orchestrator import placement_summary

    summary = placement_summary({})
    assert summary["mode"] is None
    assert summary["refused"] is None
    assert set(summary) == {
        "mode",
        "matched_cameras",
        "match_percent",
        "points_retained_fraction",
        "refused",
    }


# ---------------------------------------------------------------------------
# Retry semantics: a failed run's retry re-runs the failed stage + downstream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_of_failed_run_reruns_failed_stage_and_downstream(
    client, db_session: AsyncSession, tmp_path: Path, monkeypatch
):
    """Retry with no explicit force on a FAILED run must seed force with the
    failed stage + downstream, so stale cached artifacts cannot make the
    retry re-fail identically (sunset_06cfea looped: depth's 1-map dir passed
    its artifact check, dense re-failed the audit on every retry)."""
    from app.routes.pipeline import STAGES as ROUTE_STAGES

    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path))
    job_id = "retryjob0001"
    db_session.add(Project(id=job_id, name="v.avi", video_filename="v.avi",
                           video_path=str(tmp_path / "v.avi"), status="uploaded"))
    await db_session.flush()

    failed_report = {
        "status": "failed",
        "stages": {
            "frames": {"status": "completed"},
            "sparse": {"status": "completed"},
            "depth": {"status": "completed"},  # the degenerate cached one
            "dense": {"status": "failed", "error": "audit refused"},
            "georef": {"status": "pending"},
        },
    }
    (settings.storage.project_dir(job_id) / "pipeline_report.json").write_text(
        json.dumps(failed_report)
    )

    resp = await client.post(f"/api/pipeline/start/{job_id}", json={})
    assert resp.status_code == 200, resp.text

    row = (await db_session.execute(
        select(QueueJob).where(QueueJob.project_id == job_id)
    )).scalars().one()
    payload = json.loads(row.payload_json)
    idx = list(ROUTE_STAGES).index
    assert payload["force"] == [s for s in ROUTE_STAGES if idx(s) >= idx("dense")], payload["force"]


@pytest.mark.asyncio
async def test_retry_of_completed_run_does_not_inject_force(
    client, db_session: AsyncSession, tmp_path: Path, monkeypatch
):
    """A run that COMPLETED must keep the plain resume semantics — force is
    only seeded for failed runs, never for a fresh start or a healthy rerun."""
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path))
    job_id = "donework0001"
    db_session.add(Project(id=job_id, name="v.avi", video_filename="v.avi",
                           video_path=str(tmp_path / "v.avi"), status="uploaded"))
    await db_session.flush()
    (settings.storage.project_dir(job_id) / "pipeline_report.json").write_text(
        json.dumps({"status": "completed", "stages": {}})
    )

    resp = await client.post(f"/api/pipeline/start/{job_id}", json={})
    assert resp.status_code == 200, resp.text
    row = (await db_session.execute(
        select(QueueJob).where(QueueJob.project_id == job_id)
    )).scalars().one()
    assert json.loads(row.payload_json)["force"] == []
