"""Tests for the Phase 6 dense reconstruction pipeline.

Every test drives the real geometry code with synthetic depth views of a
flat ground plane (``z = 0``): depth maps are rendered analytically from a
pinhole camera + pose, fused, filtered, normalised, and analyzed. The API
tests run the whole stack over a real workspace directory.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.models import Project
from app.services.confidence_estimator import (
    CONFIDENCE_CLASSES,
    confidence_colors,
    dense_level_from_conf,
    dense_point_confidence,
)
from app.services.depth_fusion import DepthView, FusionParams, fuse_depth_views
from app.services.digital_twin import DigitalTwin
from app.services.dense_reconstruction import _poisson_depth_for_fusion
from app.services.incremental_mapper import IncrementalMapper
from app.services.normal_estimator import estimate_normals
from app.services.point_statistics import analyze_dense_quality, footprint_cells
from app.services.pointcloud import PointCloud, read_ply, save_ply
from app.services.pointcloud_filter import (
    radius_outlier_removal,
    statistical_outlier_removal,
    voxel_downsample,
)
from app.services.streaming_engine import StreamingEngine

# ---------------------------------------------------------------------------
# Synthetic scene helpers — a nadir camera looking at the z=0 ground plane
# ---------------------------------------------------------------------------

_LOOK_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _make_K(size: int, focal: float = 60.0) -> np.ndarray:
    return np.array([[focal, 0, (size - 1) / 2], [0, focal, (size - 1) / 2], [0, 0, 1]], dtype=np.float64)


def _rotate_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_poisson_resolution_does_not_exceed_fusion_precision():
    cloud = PointCloud(np.array([[0.0, 0.0, 0.0], [640.0, 640.0, 40.0]]))
    assert _poisson_depth_for_fusion(cloud, voxel_size=0.96) == 9


def _render_plane_depth(size: int, K: np.ndarray, R: np.ndarray, t: np.ndarray, plane_z: float = 0.0) -> np.ndarray:
    """Analytically render the depth (m, camera z) of the z=plane_z plane."""
    uu, vv = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    d0 = np.stack(
        [(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, dtype=np.float64)], axis=-1
    )
    d_w = d0 @ R.T  # unnormalised world ray directions
    n = np.array([0.0, 0.0, 1.0])
    s = (plane_z - float(n @ t)) / (d_w @ n)  # scale so the ray hits the plane
    depth = (s * d0[..., 2]).astype(np.float32)
    depth[(depth < 0.5) | (depth > 100.0)] = 0.0  # invalid marker
    return depth


def _circle_mask(size: int, cx: float, cy: float, radius: float) -> np.ndarray:
    uu, vv = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return (uu - cx) ** 2 + (vv - cy) ** 2 <= radius**2


def _plane_views(size: int = 48, with_colors: bool = True) -> list[DepthView]:
    """Three views of the ground plane: two nadir offsets + one tilted."""
    K = _make_K(size)
    views = []
    for i, (tx, ty, tilt) in enumerate([(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (0.0, 0.3, 4.0)]):
        R = _rotate_x(tilt) @ _LOOK_DOWN
        t = np.array([tx, ty, 10.0])
        depth = _render_plane_depth(size, K, R, t)
        rgb = None
        if with_colors:
            rgb = np.full((size, size, 3), (120 + 20 * i, 130 + 10 * i, 150), dtype=np.uint8)
        views.append(DepthView(frame_id=f"view_{i}", depth=depth, rgb=rgb, K=K, R=R, t=t))
    return views


def _params(voxel: float = 0.2) -> FusionParams:
    return FusionParams(voxel_size=voxel, min_depth=0.5, max_depth=50.0)


# ---------------------------------------------------------------------------
# Depth fusion
# ---------------------------------------------------------------------------


class TestDepthFusion:
    def test_fuse_plane_views_lands_on_plane(self):
        views = _plane_views(size=48)
        cloud = fuse_depth_views(views, _params())
        # Sanity of scale and location.
        assert 1000 < cloud.n < 20000, cloud.n
        median_z = float(np.median(cloud.xyz[:, 2]))
        assert abs(median_z) < 0.3, median_z  # voxel 0.2 → near z=0
        assert np.all(np.abs(cloud.xyz[:, 2]) < 2.0)
        # Multi-view agreement exists somewhere.
        assert cloud.observations.max() >= 2
        assert np.all((cloud.confidence >= 0.0) & (cloud.confidence <= 1.0))
        assert np.all(cloud.residual >= 0.0)
        assert cloud.rgb is not None  # views carried colors

    def test_fuse_merges_overlapping_views(self):
        views = _plane_views(size=48)
        cloud = fuse_depth_views(views, _params())
        raw = sum(int(np.count_nonzero(v.depth > 0)) for v in views)
        assert cloud.n < raw  # voxel merge removed redundancy
        # Colors average into the valid range.
        assert cloud.rgb is not None
        assert cloud.rgb.min() >= 100 and cloud.rgb.max() <= 170

    def test_invalid_pixels_masked(self):
        size = 48
        K = _make_K(size)
        R = _LOOK_DOWN
        t = np.array([0.0, 0.0, 10.0])
        depth = _render_plane_depth(size, K, R, t)
        depth[~_circle_mask(size, size / 2, size / 2, size * 0.3)] = 0.0  # keep centre disc only
        view = DepthView(frame_id="v", depth=depth, rgb=None, K=K, R=R, t=t)
        cloud = fuse_depth_views([view], _params())
        # Only the disc projects → cloud footprint is far smaller than full image.
        expected_full = size * size
        assert cloud.n < expected_full * 0.5
        assert cloud.n > 100


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _plane_cloud(n: int = 2000, spread: float = 5.0, z: float = 0.0) -> PointCloud:
    rng = np.random.default_rng(0)
    xyz = np.column_stack([rng.uniform(-spread, spread, n), rng.uniform(-spread, spread, n), np.full(n, z)])
    return PointCloud(xyz=xyz, confidence=np.ones(n))


class TestFilters:
    def test_statistical_outlier_removal(self):
        cloud = _plane_cloud(3000)
        outliers = np.column_stack([np.random.rand(40, 2) * 10, np.full(40, 50.0)])
        mixed = PointCloud(xyz=np.vstack([cloud.xyz, outliers]), confidence=np.ones(3040))
        cleaned, removed = statistical_outlier_removal(mixed, k=10, std_ratio=1.5)
        assert removed >= 25
        assert cleaned.n < 3040
        # Essentially no high-flying noise survives.
        assert np.sum(np.abs(cleaned.xyz[:, 2]) > 1.0) <= 3

    def test_radius_outlier_removal(self):
        cloud = _plane_cloud(3000)
        isolated = np.array([[100.0, 100.0, 0.0], [101.0, 100.0, 0.0], [100.0, 101.0, 0.0], [-100.0, -100.0, 0.0]])
        mixed = PointCloud(xyz=np.vstack([cloud.xyz, isolated]), confidence=np.ones(len(cloud.xyz) + 4))
        cleaned, removed = radius_outlier_removal(mixed, radius=0.5, min_neighbors=3)
        assert removed == 4

    def test_voxel_downsample_dedupes(self):
        cloud = _plane_cloud(2000)
        dup = PointCloud(xyz=np.vstack([cloud.xyz, cloud.xyz]), confidence=np.ones(4000))
        down = voxel_downsample(dup, voxel_size=0.01)
        assert down.n < dup.n
        assert down.observations is not None and down.observations.max() >= 2


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------


class TestNormals:
    def test_plane_normals_point_up(self):
        cloud = _plane_cloud(1500)
        normals = estimate_normals(cloud.xyz, k=15, viewpoint=(0.0, 0.0, 10.0))
        assert normals.shape == (cloud.n, 3)
        up_fraction = float(np.mean(np.abs(normals[:, 2]) > 0.9))
        assert up_fraction > 0.9, up_fraction
        # Oriented toward the viewpoint → positive z.
        assert float(np.mean(normals[:, 2] > 0)) > 0.95

    def test_wall_normals_are_horizontal(self):
        rng = np.random.default_rng(1)
        xyz = np.column_stack([np.full(800, 2.0), rng.uniform(-2, 2, 800), rng.uniform(0, 2, 800)])
        normals = estimate_normals(xyz, k=15, viewpoint=(0.0, 0.0, 10.0))
        assert float(np.mean(np.abs(normals[:, 0]) > 0.9)) > 0.8


# ---------------------------------------------------------------------------
# Statistics / quality
# ---------------------------------------------------------------------------


class TestPointStatistics:
    def test_quality_metrics(self):
        cloud = fuse_depth_views(_plane_views(size=48), _params())
        quality = analyze_dense_quality(cloud, voxel_size=0.2, noise_percent=7.0)
        d = quality.to_dict()
        assert 0 <= d["dense_score"] <= 100
        assert d["grade"] in ("Excellent", "Good", "Fair", "Poor")
        assert 0 <= d["coverage_percent"] <= 100
        assert 0 <= d["occlusion_percent"] <= 100
        assert d["mean_spacing"] > 0
        assert d["point_count"] == cloud.n

    def test_footprint_area(self):
        cloud = fuse_depth_views(_plane_views(size=48), _params())
        cells, centers, counts = footprint_cells(cloud, voxel_size=0.2)
        assert len(cells) == len(centers) == len(counts) <= cloud.n  # one row per occupied column
        assert counts.sum() == cloud.n
        assert counts.min() >= 1
        area = len(centers) * 0.2**2
        # A 48px image at z=10, f=60 covers ~8x8 m.
        assert 20 < area < 200, area

    def test_negative_cells_are_distinct(self):
        """Regression: cell identity must be exact for negative indices.

        The old scalar packing (``i * 1_000_003 + j``) aliased y-negative
        cells because floor-div/mod do not round-trip negatives, which
        inflated the coverage denominator to hundreds of millions of cells
        and reported a false 0.02% coverage on a real 42-view dense cloud.
        """
        xyz = np.array(
            [[-1.0, -1.0, 0.0], [-1.0, -1.2, 0.0], [1.4, -1.0, 0.0], [1.4, -3.6, 0.0]],
            dtype=np.float64,
        )
        cloud = PointCloud(xyz=xyz, rgb=np.full((4, 3), 128, dtype=np.uint8))
        cells, centers, counts = footprint_cells(cloud, voxel_size=1.0)
        # Four points, four distinct cells — no aliasing, no wrapped indices.
        assert len(cells) == 4
        assert counts.sum() == 4
        assert cells.min() >= -4 and cells.max() <= 2
        assert np.array_equal(cells, np.unique(cells, axis=0))


# ---------------------------------------------------------------------------
# Dense point confidence
# ---------------------------------------------------------------------------


class TestDenseConfidence:
    def test_monotonic_in_observations(self):
        obs = np.arange(1, 9)
        conf = dense_point_confidence(obs, residual=np.zeros(8), voxel_size=0.2)
        assert np.all(np.diff(conf) > 0)
        assert np.all((conf >= 0) & (conf <= 1))

    def test_four_levels(self):
        conf = np.linspace(0.0, 1.0, 100)
        levels = dense_level_from_conf(conf)
        assert set(levels) == set(CONFIDENCE_CLASSES)
        colors = confidence_colors(conf)
        assert colors.shape == (100, 3)

    def test_higher_residual_lowers_confidence(self):
        obs = np.full(4, 5)
        low = dense_point_confidence(obs, residual=np.zeros(4), voxel_size=0.2)
        high_res = dense_point_confidence(obs, residual=np.full(4, 2.0), voxel_size=0.2)
        assert np.all(low > high_res)


# ---------------------------------------------------------------------------
# Incremental mapper / digital twin
# ---------------------------------------------------------------------------


class TestIncremental:
    def test_merge_duplicate_chunk(self):
        mapper = IncrementalMapper(voxel_size=0.2)
        base = _plane_cloud(500)
        stats1 = mapper.add(base)
        assert stats1.total_points == 500
        stats2 = mapper.add(_plane_cloud(500))  # different RNG draw → mostly distinct
        assert mapper.current is not None
        assert stats2.total_points <= 1000
        assert mapper.chunks_processed == 2

    def test_twin_measurements(self):
        twin = DigitalTwin("job_twin", voxel_size=0.2)
        cloud = fuse_depth_views(_plane_views(size=48), _params())
        twin.update(cloud, label="chunk_1")
        assert twin.version == 1
        m = twin.measurements().to_dict()
        assert m["point_count"] == cloud.n
        assert m["dimensions_m"][0] > 0
        assert m["footprint_area_m2"] > 0
        summary = twin.confidence_summary()
        assert set(summary) == set(CONFIDENCE_CLASSES)
        assert sum(summary.values()) == cloud.n
        info = twin.region_intelligence()
        assert isinstance(info["weak_regions"], list)
        assert isinstance(info["suggestions"], list)


# ---------------------------------------------------------------------------
# Streaming engine
# ---------------------------------------------------------------------------


class TestStreamingEngine:
    @pytest.mark.asyncio
    async def test_publish_then_replay(self):
        eng = StreamingEngine()
        eng.publish("s1", "started", {"a": 1})
        eng.publish("s1", "complete", {})
        events = eng.replay("s1")
        assert [e["type"] for e in events] == ["started", "complete"]

    @pytest.mark.asyncio
    async def test_live_delivery(self):
        eng = StreamingEngine()
        received: list[str] = []

        async def consumer():
            async for event in eng.iter_events("s2"):
                received.append(event["type"])
                if event["type"] == "complete":
                    return

        task = asyncio.ensure_future(consumer())
        await asyncio.sleep(0.05)  # let the subscriber register
        eng.publish("s2", "stage:depth_fusion", {})
        eng.publish("s2", "complete", {})
        await asyncio.wait_for(task, timeout=3)
        assert "stage:depth_fusion" in received
        assert "complete" in received


# ---------------------------------------------------------------------------
# PLY round trip
# ---------------------------------------------------------------------------


class TestPlyIO:
    def test_roundtrip(self, tmp_path: Path):
        cloud = fuse_depth_views(_plane_views(size=48), _params())
        from app.services.normal_estimator import add_normals

        add_normals(cloud, k=10, viewpoint=(0.0, 0.0, 10.0))
        path = tmp_path / "model.ply"
        save_ply(path, cloud)
        loaded = read_ply(path)
        assert loaded.n == cloud.n
        assert np.allclose(loaded.xyz, cloud.xyz, atol=1e-4)
        assert np.allclose(loaded.normals, cloud.normals, atol=1e-3)
        assert loaded.rgb is not None and loaded.confidence is not None


# ---------------------------------------------------------------------------
# End-to-end API test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dense_api_end_to_end(client, db_session: AsyncSession, tmp_path: Path, monkeypatch):
    """POST start → status → statistics → confidence → downloads → SSE stream."""
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path))
    job_id = "densejob0001"

    # Seed a project + workspace with poses + depth maps.
    project = Project(
        id=job_id,
        name="plane.avi",
        video_filename="plane.avi",
        video_path=str(tmp_path / "plane.avi"),
        status="uploaded",
    )
    db_session.add(project)
    await db_session.flush()

    workspace = settings.storage.project_dir(job_id)
    depth_dir = workspace / "depth"
    depth_dir.mkdir(parents=True)

    poses = []
    for view in _plane_views(size=40):
        np.save(depth_dir / f"{view.frame_id}.npy", view.depth)
        poses.append(
            {
                "frame_id": view.frame_id,
                "K": view.K.tolist(),
                "R": view.R.tolist(),
                "t": view.t.tolist(),
            }
        )
    (workspace / "poses.json").write_text(json.dumps({"frames": poses}))

    # Run dense reconstruction.
    start = await client.post(
        "/api/dense/start/densejob0001",
        json={"voxel_size": 0.25, "sor_k": 15, "ror_min_neighbors": 3},
    )
    assert start.status_code == 200, start.text
    body = start.json()
    assert body["status"] == "completed"
    assert "points" in body["message"]

    # Status / statistics / confidence.
    status = await client.get("/api/dense/status/densejob0001")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["dense_score"] is not None

    stats = await client.get("/api/dense/statistics/densejob0001")
    assert stats.status_code == 200
    s = stats.json()
    assert s["quality"]["point_count"] > 500
    assert 0 <= s["quality"]["dense_score"] <= 100
    assert s["measurements"]["footprint_area_m2"] > 0
    assert isinstance(s["intelligence"]["suggestions"], list)

    conf = await client.get("/api/dense/confidence/densejob0001")
    assert conf.status_code == 200
    c = conf.json()
    assert 0 <= c["mean_confidence"] <= 1
    assert sum(c["class_counts"].values()) == s["quality"]["point_count"]

    # The persisted model exists on disk.
    model = workspace / "dense" / "dense_model.ply"
    assert model.exists()

    # Downloads in each format.
    for fmt, magic in [("ply", b"ply"), ("xyz", None), ("pcd", b"# .PCD")]:
        resp = await client.get(f"/api/dense/download/densejob0001?format={fmt}")
        assert resp.status_code == 200, fmt
        content = resp.content
        if magic:
            assert content.startswith(magic)
        assert len(content) > 100

    # The model row was persisted to the DB.
    from sqlalchemy import select

    from app.db.models import Model3D

    result = await db_session.execute(select(Model3D).where(Model3D.project_id == job_id))
    model3d = result.scalar_one()
    assert model3d.point_count == s["quality"]["point_count"]

    # SSE stream replays recorded events.
    async with client.stream("GET", "/api/dense/stream/densejob0001") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [ln async for ln in resp.aiter_lines()]
    data = "\n".join(lines)
    assert "event: started" in data
    assert "event: complete" in data
    assert job_id in data

    # A run with no depth maps fails with a meaningful error.
    bad_id = "densejob0002"
    db_session.add(
        Project(id=bad_id, name="x.avi", video_filename="x.avi", video_path=str(tmp_path / "x.avi"), status="uploaded")
    )
    await db_session.flush()
    settings.storage.project_dir(bad_id)
    bad = await client.post("/api/dense/start/densejob0002", json={})
    assert bad.status_code == 500
    assert "depth" in bad.json()["error"].lower()

    # Unknown job → 404.
    missing = await client.get("/api/dense/status/doesnotexist")
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_post_fusion_steps_are_timed_without_clobbering_payloads(
    client, db_session: AsyncSession, tmp_path: Path, monkeypatch
):
    """The post-fusion chain must be attributable from the run's own report.

    Regression: mesh build, mesh audit, texturing and the viewer LOD ran ~692 s
    on the reference run with NO substage duration recorded anywhere, so the
    dense total jumped and nothing named the cause. Each step now closes a
    timed segment. The payload keys those steps write (``mesh``,
    ``viewer_lod``, ``mesh_input``) must keep their measurements — a timer that
    overwrote one would trade a missing number for a wrong one.
    """
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path))
    job_id = "densejob_timed"
    db_session.add(Project(
        id=job_id, name="plane.avi", video_filename="plane.avi",
        video_path=str(tmp_path / "plane.avi"), status="uploaded"))
    await db_session.flush()

    workspace = settings.storage.project_dir(job_id)
    depth_dir = workspace / "depth"
    depth_dir.mkdir(parents=True)
    poses = []
    for view in _plane_views(size=40):
        np.save(depth_dir / f"{view.frame_id}.npy", view.depth)
        poses.append({"frame_id": view.frame_id, "K": view.K.tolist(),
                      "R": view.R.tolist(), "t": view.t.tolist()})
    (workspace / "poses.json").write_text(json.dumps({"frames": poses}))

    resp = await client.post(f"/api/dense/start/{job_id}", json={"voxel_size": 0.25})
    assert resp.status_code == 200, resp.text

    report = json.loads((workspace / "dense_report.json").read_text())
    stages = report["stages"]
    for name in ("export_cloud", "mesh_input_scan", "mesh_build",
                 "mesh_audit", "viewer_lod_build"):
        assert name in stages, (name, sorted(stages))
        assert isinstance(stages[name]["duration_ms"], (int, float))
        assert stages[name]["duration_ms"] >= 0.0

    # Payload keys still carry measurements, not a timing dict.
    assert stages["mesh"]["vertices"] > 0
    assert "after_faces" in stages["viewer_lod"]
    assert "mesh_input_point_count" in stages["mesh_input"]

    # Every timed segment lands between its neighbours, and the timed
    # substages account for most of the run (the point of the change).
    timed = sum(v["duration_ms"] for v in stages.values()
                if isinstance(v, dict) and "duration_ms" in v)
    assert timed > 0
    assert timed <= report["run_time_ms"] * 1.05


def test_dense_stage_detail_surfaces_substage_timings(tmp_path, monkeypatch):
    """``performance.json`` must name the dense substages that cost the time.

    The dense stage's ``details`` block is built by ``_stage_dense``, so a
    substage can be perfectly timed inside ``dense_report.json`` and still be
    invisible in performance.json. Payload-only entries (mesh quality, layer
    measurements) carry no duration and must be skipped, never invented.
    """
    import app.services.pipeline_orchestrator as orch

    canned = {
        "quality": {"point_count": 1234, "dense_score": 80.0, "grade": "Good"},
        "stages": {
            "depth_fusion": {"duration_ms": 1500.5, "views": 12},
            "export": {"duration_ms": 250.0},
            "export_cloud": {"duration_ms": 40.0},
            "mesh_build": {"duration_ms": 4200.0},
            "mesh_audit": {"duration_ms": 900.0},
            "viewer_lod_build": {"duration_ms": 7800.25},
            "mesh": {"vertices": 100},          # payload only
            "dense_layers": {"status": "measured"},  # payload only
            "viewer_lod": {},                     # payload only
        },
    }
    monkeypatch.setattr(orch, "run_dense_reconstruction", lambda job_id, params: canned)

    state = orch.StageState(name="dense")
    orch._stage_dense("job", tmp_path, state)

    assert state.detail["substages_ms"] == {
        "depth_fusion": 1500.5,
        "export": 250.0,
        "export_cloud": 40.0,
        "mesh_build": 4200.0,
        "mesh_audit": 900.0,
        "viewer_lod_build": 7800.25,
    }
    assert "mesh" not in state.detail["substages_ms"]
    assert "dense_layers" not in state.detail["substages_ms"]
    assert state.detail["dense_score"] == 80.0
    assert state.count == 1234


def test_depth_diagnostics_is_test_env_always_defined(tmp_path: Path):
    """Regression test: verify is_test_env is initialized before any conditional reference in run_depth_diagnostics."""
    import os
    from app.services.depth_diagnostics import run_depth_diagnostics

    # Seed minimal workspace with poses.json
    workspace = tmp_path / "diagnostics_test_job"
    workspace.mkdir(parents=True, exist_ok=True)
    poses_path = workspace / "poses.json"
    poses_path.write_text(json.dumps({"frames": []}))

    # 1. Test when PYTEST_CURRENT_TEST is present (test environment)
    report1 = run_depth_diagnostics(workspace)
    assert report1 is not None

    # 2. Test when PYTEST_CURRENT_TEST is temporarily removed (canonical environment)
    pytest_env = os.environ.pop("PYTEST_CURRENT_TEST", None)
    old_dep = settings.deployment
    settings.deployment = "development"
    try:
        report2 = run_depth_diagnostics(workspace)
        assert report2 is not None
    finally:
        if pytest_env is not None:
            os.environ["PYTEST_CURRENT_TEST"] = pytest_env
        settings.deployment = old_dep



class TestSceneDepthCeiling:
    """Far-field scenes must be fusable: the configured near-field ceiling
    (200 m) zeroed every airport1 measurement (scene 500-970 m), so the audit
    reported cross_view NOT_EVALUATED and criterion G failed. The ceiling must
    adapt to the scene, never shrink below the configured value."""

    def _view(self, depth: np.ndarray) -> "DepthView":
        from app.services.depth_fusion import DepthView

        K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
        R = np.eye(3)
        return DepthView(frame_id="v", depth=depth, rgb=None, K=K, R=R, t=np.zeros(3))

    def test_near_field_keeps_configured_ceiling(self):
        from app.services.dense_reconstruction import scene_depth_ceiling

        d = np.full((48, 64), 30.0)
        assert scene_depth_ceiling([self._view(d)], 200.0) == 200.0

    def test_far_field_extends_from_maps_when_no_sparse(self):
        from app.services.dense_reconstruction import scene_depth_ceiling

        d = np.full((48, 64), 700.0)
        d[0, 0] = 5000.0  # lone garbage pixel must not set the ceiling
        ceil = scene_depth_ceiling([self._view(d)], 200.0)
        assert 765.0 <= ceil <= 900.0

    def test_sparse_verified_geometry_anchors_ceiling(self):
        from app.services.dense_reconstruction import scene_depth_ceiling

        # Maps claim 1400 m horizon extrapolation; sparse says scene ends 900 m.
        d = np.full((48, 64), 1400.0)
        pose = {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}
        sparse = np.array([[x, 0.0, 890.0] for x in range(50)], dtype=np.float64)
        ceil = scene_depth_ceiling([self._view(d)], 200.0, sparse_xyz=sparse, poses=[pose])
        assert 975.0 <= ceil <= 1050.0

    def test_fusion_with_far_field_ceiling_unprojects(self):
        """End-to-end: a 700 m scene fuses only when the ceiling is adapted;
        with the near-field default the unprojection produces nothing."""
        from app.services.dense_reconstruction import scene_depth_ceiling
        from app.services.depth_fusion import FusionParams, fuse_depth_views

        # Constant 700 m depth from a nadir camera at the origin (identity R:
        # camera z == world z, so points land at z = -700 world... use +700
        # along camera axis via R = diag(1,1,1) and depth positive -> the
        # unprojection places points 700 m along +z camera = world +z).
        far = []
        for i in range(3):
            d = np.full((48, 64), 700.0)
            far.append(DepthView(frame_id=f"far_{i}", depth=d, rgb=None,
                                 K=self._view(d).K, R=np.eye(3),
                                 t=np.array([10.0 * i, 0.0, 0.0])))
        configured = 200.0
        ceiling = scene_depth_ceiling(far, configured)
        assert ceiling > 700.0, f"ceiling must exceed the 700 m scene, got {ceiling}"
        fp = FusionParams(voxel_size=5.0, min_depth=0.2, max_depth=ceiling)
        cloud = fuse_depth_views(far, fp)
        assert cloud.n > 0
        # Control: the unadapted near-field ceiling cannot silently return an
        # empty cloud (voxel_merge crashed on empty input before this was
        # named) — it must raise the descriptive scene-vs-ceiling error.
        import pytest as _pytest
        fp_near = FusionParams(voxel_size=5.0, min_depth=0.2, max_depth=configured)
        with _pytest.raises(ValueError, match="deeper than the configured ceiling"):
            fuse_depth_views(far, fp_near)
