"""Tests for Phase 7 — mesh generation, optimization/repair, texturing,
semantic understanding, digital twin, georeferencing, LODs, copilot and the
plugin-stage orchestrator chain."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.models import Project
from app.services.copilot import answer
from app.services.geo_alignment import objects_to_geojson
from app.services.mesh import TriangleMesh
from app.services.mesh_generator import MeshNotPossible, MeshParams, generate_mesh
from app.services.mesh_optimizer import PRESET_FRACTIONS, DecimateParams, decimate
from app.services.mesh_repair import RepairParams, repair
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.pipeline_stage import MESH_CHAIN
from app.services.pointcloud import PointCloud, save_ply

# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------


def _grid_mesh(g: int = 16, z_amp: float = 0.0, colors=None, rng_seed: int = 0):
    """A triangulated height-field grid mesh (unit spacing, z=0 plane)."""
    xs, ys = np.meshgrid(np.arange(g, dtype=float), np.arange(g, dtype=float))
    rng = np.random.default_rng(rng_seed)
    z = z_amp * rng.standard_normal((g, g)) if z_amp else np.zeros((g, g))
    verts = np.stack([xs.ravel(), ys.ravel(), z.ravel()], 1)
    faces = []
    for i in range(g - 1):
        for j in range(g - 1):
            a = i * g + j
            b = a + 1
            c = a + g
            d = c + 1
            faces += [[a, b, c], [b, d, c]]
    rgb = np.full((g * g, 3), 150, dtype=np.uint8)
    if colors is not None:
        rgb[:] = colors
    return TriangleMesh(vertices=verts, faces=np.array(faces), colors=rgb,
                        confidence=np.ones(g * g))


def _cloud_from_mesh(mesh: TriangleMesh) -> PointCloud:
    return PointCloud(xyz=mesh.vertices, rgb=mesh.colors,
                      confidence=np.ones(mesh.n))


def _seed_dense_workspace(tmp_path: Path, job_id: str, g: int = 14) -> Path:
    """A workspace the autonomous chain can resume from: dense model + poses +
    frames + depth cache (so the core stages all artifact-skip)."""
    settings.storage.base_path = str(tmp_path)
    settings.mesh.atlas_width = 512
    workspace = settings.storage.project_dir(job_id)
    (workspace / "selected").mkdir(parents=True, exist_ok=True)
    for i in range(2):
        cv2.imwrite(str(workspace / "selected" / f"frame_{i:06d}.jpg"),
                    np.full((64, 64, 3), 120 + i * 30, dtype=np.uint8))
    frames = []
    for i in range(2):
        frames.append({"frame_id": f"frame_{i:06d}",
                       "K": [[60.0, 0, 31.5], [0, 60.0, 31.5], [0, 0, 1]],
                       "R": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                       "t": [0.0, 0.0, 10.0]})
    (workspace / "poses.json").write_text(json.dumps({"frames": frames}))
    depth_dir = workspace / "depth"
    depth_dir.mkdir(exist_ok=True)
    # EVERY registered view is mapped, as a completed depth stage leaves it.
    # Seeding a single map used to pass the artifact check and no longer does:
    # the check now floors usable maps at two, so a 1-map "complete" stage on
    # a 2-camera trajectory is treated as degenerate (it was exactly the state
    # that made Retry a no-op loop) and re-runs instead of skipping.
    for f in frames:
        np.save(depth_dir / f"{f['frame_id']}.npy", np.full((64, 64), 10.0, dtype=np.float32))

    mesh = _grid_mesh(g)
    cloud = _cloud_from_mesh(mesh)
    (workspace / "dense").mkdir(exist_ok=True)
    save_ply(workspace / "dense" / "dense_model.ply", cloud)
    (workspace / "dense_report.json").write_text(json.dumps(
        {"status": "completed", "quality": {"point_count": cloud.n}}))
    return workspace


def _add_panel(verts, faces, rgb, p00, p10, p11, p01, n, color):
    """A subdivided quad between the four corners p00..p01 (grid n x n)."""
    base = len(verts)
    for i in range(n + 1):
        for j in range(n + 1):
            a = i / n
            b = j / n
            pt = (p00 * (1 - a) + p10 * a) * (1 - b) + (p01 * (1 - a) + p11 * a) * b
            verts.append(pt.tolist())
            rgb.append(color)
    for i in range(n):
        for j in range(n):
            a = base + i * (n + 1) + j
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])


def _box_scene_mesh(n: int = 6) -> TriangleMesh:
    """Ground sheet + an elevated box (subdivided roof + walls, brownish)."""
    ground = _grid_mesh(14)
    verts = list(map(list, ground.vertices))
    faces = list(map(list, ground.faces))
    rgb = list(map(list, ground.colors))

    z = 2.0
    top = np.array([5.0, 5.0, z]), np.array([9.0, 5.0, z]), np.array([9.0, 9.0, z]), np.array([5.0, 9.0, z])
    base_c = np.array([5.0, 5.0, 0.05]), np.array([9.0, 5.0, 0.05]), np.array([9.0, 9.0, 0.05]), np.array([5.0, 9.0, 0.05])
    roof_c = (160, 120, 90)
    wall_c = (140, 110, 85)
    _add_panel(verts, faces, rgb, top[0], top[1], top[2], top[3], n, roof_c)
    _add_panel(verts, faces, rgb, top[0], top[1], base_c[1], base_c[0], n, wall_c)
    _add_panel(verts, faces, rgb, top[1], top[2], base_c[2], base_c[1], n, wall_c)
    _add_panel(verts, faces, rgb, top[2], top[3], base_c[3], base_c[2], n, wall_c)
    _add_panel(verts, faces, rgb, top[3], top[0], base_c[0], base_c[3], n, wall_c)
    return TriangleMesh(vertices=np.asarray(verts, dtype=float),
                        faces=np.asarray(faces, dtype=np.int64),
                        colors=np.asarray(rgb, dtype=np.uint8),
                        confidence=np.ones(len(verts)))


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------


class TestMeshGeneration:
    def test_surface_from_heightfield(self):
        mesh = _grid_mesh(20, z_amp=0.05)
        cloud = _cloud_from_mesh(mesh)
        out, stats = generate_mesh(cloud, MeshParams(method="surface"))
        assert out.m > 100 and stats["method"] == "surface"
        assert out.colors is not None and out.confidence is not None
        assert out.face_areas().min() > 0

    def test_alpha_shape_on_filled_volume(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(-1.0, 1.0, size=(600, 3))
        cloud = PointCloud(xyz=pts, confidence=np.ones(len(pts)))
        out, _stats = generate_mesh(cloud, MeshParams(method="alpha"))
        assert out.m > 50
        assert out.face_areas().min() > 0

    def test_auto_resolves_to_surface(self):
        mesh = _grid_mesh(10)
        out, stats = generate_mesh(_cloud_from_mesh(mesh), MeshParams(method="auto"))
        assert stats["method"] == "surface" and out.m > 0

    def test_degenerate_cloud_raises(self):
        cloud = PointCloud(xyz=np.array([[0.0, 0, 0], [1, 0, 0]]))
        with pytest.raises(MeshNotPossible):
            generate_mesh(cloud, MeshParams(method="surface"))
        tiny = PointCloud(xyz=np.zeros((5, 3)))
        with pytest.raises(MeshNotPossible):
            generate_mesh(tiny, MeshParams(method="surface", min_points=50))


class TestMeshOptimizer:
    def test_presets_reduce_faces_monotonically(self):
        mesh = _grid_mesh(24, z_amp=0.05)
        counts = {}
        for preset in ("ultra", "high", "medium", "low"):
            out, _ = decimate(mesh, DecimateParams(preset=preset))
            counts[preset] = out.m
            assert out.face_components()[0] == 1, f"{preset} split the surface"
        assert counts["low"] < counts["medium"] < counts["high"] < counts["ultra"]
        assert counts["ultra"] <= int(mesh.m * PRESET_FRACTIONS["ultra"]) + 1

    def test_target_fraction(self):
        mesh = _grid_mesh(20, z_amp=0.05)
        out, stats = decimate(mesh, DecimateParams(target_fraction=0.4))
        assert out.m <= mesh.m * 0.41
        assert stats["reached_target"]


class TestMeshRepair:
    def test_repairs_common_defects(self):
        mesh = _grid_mesh(10)
        verts = list(map(list, mesh.vertices))
        faces = list(map(list, mesh.faces))
        rgb = list(map(list, mesh.colors))
        # duplicate vertices (welded) + a degenerate face + a duplicate face
        dup = add_vertex(verts, rgb, mesh.vertices[3] + [0, 0, 0])
        faces.append([int(dup), 0, 1])
        faces.append([0, 1, 2])  # duplicate of an existing face? existing order differs
        # tiny detached component (one triangle far away)
        far = add_vertex(verts, rgb, [50.0, 50.0, 0.0])
        faces.append([int(far), int(far), int(far)])
        fixed = TriangleMesh(vertices=np.asarray(verts), faces=np.asarray(faces, dtype=np.int64),
                             colors=np.asarray(rgb, dtype=np.uint8))
        out, rep = repair(fixed, RepairParams(weld_precision=1e-6, min_component_faces=4))
        assert rep["welded_vertices"] >= 1
        assert rep["removed_degenerate_faces"] >= 1
        assert out.face_components()[0] == 1

    def test_interior_hole_filled_rim_kept(self):
        mesh = _grid_mesh(12)
        cc = mesh.face_centroids()
        # Remove the single middle cell (two triangles) — a 4-edge hole.
        keep = ~((np.abs(cc[:, 0] - 5.5) < 0.6) & (np.abs(cc[:, 1] - 5.5) < 0.6))
        holed = mesh.submesh(keep)
        out, rep = repair(holed, RepairParams(max_hole_edges=64))
        assert rep["holes_filled"] == 1
        assert rep["hole_edges_filled"] == 4
        assert out.m == mesh.m
        assert len(out.boundary_loops()) == 1  # outer rim preserved


def add_vertex(verts, rgb, xyz):
    verts.append(list(map(float, xyz)))
    rgb.append([150, 150, 150])
    return len(verts) - 1


# ---------------------------------------------------------------------------
# Texture projection + atlas
# ---------------------------------------------------------------------------


class TestTexturing:
    def _views_and_mesh(self):
        mesh = _grid_mesh(12)
        cloud = _cloud_from_mesh(mesh)
        out, _ = generate_mesh(cloud, MeshParams(method="surface", surface_max_edge_factor=2.0))
        size, f = 128, 80.0
        K = np.array([[f, 0, (size - 1) / 2], [0, f, (size - 1) / 2], [0, 0, 1]])
        # checkerboard pattern in image space
        yy, xx = np.mgrid[0:size, 0:size]
        img = np.where(((xx // 16 + yy // 16) % 2) == 0, 220, 40)
        img = np.stack([img] * 3, axis=-1).astype(np.uint8)
        # BGR order to mimic cv2 reads
        img = img[:, :, ::-1].copy()
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        t = np.array([5.5, 5.5, 10.0])  # nadir camera above the mesh centre
        from app.services.texture_projector import TextureView

        view = TextureView(frame_id="f0", image=img, K=K, R=R, t=t)
        return out, [view]

    def test_best_view_and_atlas(self):
        from app.services.texture_blender import build_texture_atlas
        from app.services.uv_mapper import build_atlas_layout

        mesh, views = self._views_and_mesh()
        layout = build_atlas_layout(mesh.m, atlas_width=1024)
        atlas, stats = build_texture_atlas(mesh, views, layout)
        assert stats["faces_textured"] > mesh.m * 0.8, stats
        assert stats["atlas_written"]
        assert atlas.shape[0] > 0 and np.var(atlas) > 0
        assert 0.6 <= min(stats["view_gains"].values()) <= max(stats["view_gains"].values()) <= 1.6


# ---------------------------------------------------------------------------
# Semantic + digital twin + confidence
# ---------------------------------------------------------------------------


class TestSemantics:
    def test_box_scene_labels(self):
        from app.services.semantic_segmentation import segment_mesh

        mesh = _box_scene_mesh()
        report = segment_mesh(mesh)
        names = report["class_names"]
        labels = report["labels"]
        # roof faces of the box (all at z≈2.0 with up normals)
        centroids = mesh.face_centroids()
        fn = mesh.face_normals()
        roof_mask = (centroids[:, 2] > 1.5) & (np.abs(fn[:, 2]) > 0.7)
        assert roof_mask.any()
        assert set(labels[roof_mask]) == {names.index("roof")}
        assert (labels == names.index("ground")).sum() > 0
        assert "unclassified" in names
        assert report["confidence"].min() >= 0.0
        total = sum(report["coverage"].values())
        assert total == mesh.m
        assert report["method"] == "classical_geometry_rgb"

    def test_ground_plane_fit(self):
        from app.services.semantic_segmentation import _fit_ground_plane

        verts = np.random.default_rng(1).normal(size=(300, 3))
        verts[:, 2] = 0.05 * verts[:, 2]  # near-flat ground
        plane = _fit_ground_plane(verts, tol=0.2)
        assert plane is not None
        normal, d = plane
        assert abs(normal[2]) > 0.9
        assert abs(d) < 0.2


class TestDigitalTwinEngine:
    def test_objects_and_scene_graph(self):
        from app.services.digital_twin_engine import build_scene_graph, extract_objects
        from app.services.semantic_segmentation import segment_mesh

        mesh = _box_scene_mesh()
        report = segment_mesh(mesh)
        objects = extract_objects(mesh, report["labels"], report["confidence"])
        classes = {o.class_name for o in objects}
        assert "roof" in classes and "wall" in classes
        for obj in objects:
            assert obj.height_m >= 0 and obj.face_count > 0
            assert obj.surface_area_m2 > 0
            assert 0 <= obj.confidence <= 1
            assert obj.uuid
        wall = [o for o in objects if o.class_name == "wall"][0]
        assert abs(wall.height_m - 1.95) < 0.2  # box base at 0.05 → roof at 2.0
        roof = [o for o in objects if o.class_name == "roof"][0]
        assert roof.surface_area_m2 > 10
        graph = build_scene_graph(mesh, objects)
        assert graph["node_count"] == len(objects)
        rel = {(e["source_class"], e["target_class"]) for e in graph["edges"]}
        assert ("roof", "wall") in rel or ("wall", "roof") in rel


class TestConfidenceOverlay:
    def test_overlay_range(self):
        from app.services.confidence_overlay import confidence_color, vertex_confidence

        mesh = _grid_mesh(12)
        conf, stats = vertex_confidence(mesh, semantic_conf=np.full(mesh.m, 0.8),
                                        gps_quality=90.0)
        assert conf.min() >= 0 and conf.max() <= 1
        colors = confidence_color(conf)
        assert colors.shape == (mesh.n, 3)
        assert stats["sources"] == ["semantic", "gps"]


class TestGeoAlignment:
    def test_geojson_footprint(self):
        objects = [{
            "uuid": "u1", "class": "roof", "confidence": 0.8,
            "bbox_min": [0.0, 0.0, 0.0], "bbox_max": [2.0, 3.0, 1.0],
            "centroid": [1.0, 1.5, 0.5], "height_m": 1.0,
            "surface_area_m2": 8.0, "footprint_area_m2": 6.0, "volume_m3": None,
        }]
        transform = np.eye(4)
        gj = objects_to_geojson(objects, transform, {"type": "enu"})
        assert gj["features"][0]["geometry"]["coordinates"][0][0] == [0.0, 0.0]


class TestLods:
    def test_lod_chain_decreases(self):
        from app.services.lod_generator import generate_lods

        mesh = _grid_mesh(24, z_amp=0.05)
        manifest, _ = generate_lods(mesh, ratios=[1.0, 0.5, 0.25])
        faces = [m["faces"] for m in manifest]
        assert faces[0] == mesh.m
        assert faces[0] > faces[1] > faces[2]


class TestCopilot:
    def _scene(self):
        return {"objects": [
            {"uuid": "a", "class": "roof", "confidence": 0.9, "height_m": 2.0,
             "surface_area_m2": 16.0, "volume_m3": None, "volume_closed": False,
             "bbox_min": [0, 0, 1.9], "bbox_max": [4, 4, 2.0],
             "centroid": [2, 2, 1.95]},
            {"uuid": "b", "class": "wall", "confidence": 0.55, "height_m": 2.0,
             "surface_area_m2": 12.0, "volume_m3": None, "volume_closed": False,
             "bbox_min": [0, 0, 0.05], "bbox_max": [4, 4, 2.0],
             "centroid": [2, 2, 1.0]},
        ]}

    def test_measure(self):
        res = answer(self._scene(), "measure the roof")
        assert res["intent"] == "measure" and len(res["objects"]) == 1

    def test_volume_requires_closed(self):
        res = answer(self._scene(), "estimate volume of the wall")
        assert "closed" in res["answer"] or "cannot" in res["answer"]

    def test_low_confidence(self):
        res = answer(self._scene(), "show objects with confidence below 60%")
        assert res["intent"] == "low_confidence"
        assert [o["uuid"] for o in res["objects"]] == ["b"]

    def test_unanswerable_clarifies(self):
        res = answer(self._scene(), "predict tomorrow's damage")
        assert res["intent"] == "clarify"


# ---------------------------------------------------------------------------
# Orchestrator plugin chain
# ---------------------------------------------------------------------------


class TestPluginChain:
    def test_full_chain_resume(self, tmp_path: Path):
        job_id = "twinjob0001"
        workspace = _seed_dense_workspace(tmp_path, job_id)
        req = PipelineRequest(plugins=list(MESH_CHAIN))
        report = run_autonomous_pipeline(job_id, req)
        assert report["status"] == "completed", report.get("error")
        stages = report["stages"]
        # core stages: artifacts present → skipped; georef completes w/ no GPS
        assert stages["frames"]["status"] == "skipped"
        assert stages["sparse"]["status"] == "skipped"
        assert stages["depth"]["status"] == "skipped"
        assert stages["dense"]["status"] == "skipped"
        # plugin stages actually produced artifacts
        assert stages["mesh_generation"]["status"] == "completed", stages["mesh_generation"]
        assert stages["mesh_repair"]["status"] == "completed"
        assert stages["semantic"]["status"] == "completed"
        assert stages["digital_twin"]["status"] == "completed"
        assert stages["confidence_overlay"]["status"] == "completed"
        assert stages["lod"]["status"] == "completed"
        assert stages["scene_index"]["status"] == "completed"
        # geo needs GPS alignment → clean skip, not a failure
        assert stages["geo_alignment"]["status"] == "skipped"
        for rel in ("mesh/base_mesh.ply", "mesh/optimized_mesh.ply",
                    "mesh/repaired_mesh.ply", "semantic/semantic_report.json",
                    "twin/twin.json", "confidence/confidence_report.json",
                    "mesh/lod/manifest.json", "twin/scene_index.json"):
            assert (workspace / rel).exists(), rel

        # rerun → every stage whose artifact exists is skipped
        report2 = run_autonomous_pipeline(job_id, PipelineRequest(plugins=list(MESH_CHAIN)))
        assert report2["status"] == "completed"
        for name in MESH_CHAIN:
            assert report2["stages"][name]["status"] in ("skipped", "completed"), name

    def test_unknown_plugin_fails_fast(self, tmp_path: Path):
        _seed_dense_workspace(tmp_path, "twinjob0002")
        with pytest.raises(ValueError):
            run_autonomous_pipeline("twinjob0002",
                                    PipelineRequest(plugins=["does_not_exist"]))

    def test_chain_without_dense_skips_cleanly(self, tmp_path: Path):
        # No dense artifact → core stages try to run and stop at frames (no video).
        settings.storage.base_path = str(tmp_path)
        workspace = settings.storage.project_dir("twinjob0003")
        workspace.mkdir(parents=True, exist_ok=True)
        report = run_autonomous_pipeline("twinjob0003", PipelineRequest(plugins=["mesh_generation"]))
        assert report["status"] == "failed"  # frames stage: no video present


# ---------------------------------------------------------------------------
# REST integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digital_twin_api(client, db_session: AsyncSession, tmp_path: Path):
    """POST /api/digital-twin/build runs the chain; artifacts are served."""
    job_id = "twini000001"
    db_session.add(Project(id=job_id, name="site.avi", video_filename="site.avi",
                           video_path=str(tmp_path / "site.avi"), status="uploaded"))
    await db_session.flush()
    _seed_dense_workspace(tmp_path, job_id)

    resp = await client.post(f"/api/digital-twin/build/{job_id}", json={"plugins": []})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed", body

    scene = await client.get(f"/api/digital-twin/scene/{job_id}")
    assert scene.status_code == 200
    sdata = scene.json()
    assert "repaired_mesh" in sdata["artifacts"]
    assert "twin" in sdata

    obj = await client.get(f"/api/digital-twin/objects/{job_id}")
    assert obj.status_code == 200

    mesh = await client.get(f"/api/digital-twin/mesh/{job_id}?format=ply")
    assert mesh.status_code == 200 and mesh.content.startswith(b"ply")

    objfmt = await client.get(f"/api/digital-twin/mesh/{job_id}?format=obj")
    assert objfmt.status_code == 200 and objfmt.content.startswith(b"v ")

    cop = await client.post(f"/api/digital-twin/copilot/{job_id}",
                            json={"query": "summary of the scene"})
    assert cop.status_code == 200 and cop.json()["intent"] == "summary"

    status = await client.get(f"/api/digital-twin/status/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"

    geojson = await client.get(f"/api/digital-twin/geojson/{job_id}")
    assert geojson.status_code == 400  # no GPS alignment on this job

    missing = await client.get("/api/digital-twin/status/nope")
    assert missing.status_code == 404

    # project row was updated
    row = (await db_session.execute(
        __import__("sqlalchemy").select(Project).where(Project.id == job_id))).scalar_one()
    assert row.status == "twin_completed"
