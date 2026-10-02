"""Regression tests for the mesh-stage curtain failure (run 1ae5b0).

History: the fallback surface mesher culled long edges only in the PCA
projection plane, so a tall structure sharing ground footprint with terrain
was stitched to it with near-vertical triangles (89.7% of faces had
near-horizontal normals; median 3D edge 8.66 m vs ~1.4 m point spacing).
These tests pin the true-3D, locally-scaled edge validation.
"""

import numpy as np

from app.services.mesh_generator import MeshNotPossible, _surface_mesh
from app.services.pointcloud import PointCloud


def _terrain_cloud(n: int = 6000, side: float = 60.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    xy = rng.uniform(-side / 2, side / 2, (n, 2))
    z = 0.05 * np.sin(xy[:, 0] / 6.0) + rng.normal(0, 0.05, n)  # gently undulating ground
    return np.column_stack([xy, z])


class TestSurfaceMeshCurtainRegression:
    def test_tall_structure_not_stitched_to_terrain(self):
        """Terrain + a 100 m tower sharing the same XY footprint must not
        produce a vertical-curtain population: near-vertical triangles are
        rejected by the true-3D edge budget, and surviving faces stay
        near-horizontal (a surface, not a curtain)."""
        xyz = _terrain_cloud()
        rng = np.random.default_rng(1)
        # Tower: 1500 points on the wall of a 3x3 m column rising 100 m,
        # footprint strictly inside the terrain footprint (the historical
        # failure geometry).
        hh = rng.uniform(0.2, 100.0, 1500)
        aa = rng.uniform(0, 2 * np.pi, 1500)
        tower = np.column_stack([1.5 + 1.5 * np.cos(aa), -2.0 + 1.5 * np.sin(aa), hh])
        cloud = PointCloud(xyz=np.vstack([xyz, tower]))
        mesh = _surface_mesh(cloud, max_edge_factor=4.0)

        p0, p1, p2 = mesh.vertices[mesh.faces[:, 0]], mesh.vertices[mesh.faces[:, 1]], mesh.vertices[mesh.faces[:, 2]]
        n = np.cross(p1 - p0, p2 - p0)
        nz = n[:, 2] / (np.linalg.norm(n, axis=1) + 1e-12)
        vertical_like = float(np.mean(np.abs(nz) <= 0.2))
        edges = np.concatenate([
            np.linalg.norm(p1 - p0, axis=1),
            np.linalg.norm(p2 - p1, axis=1),
            np.linalg.norm(p0 - p2, axis=1),
        ])
        # Curtains are faces SPANNING height between surfaces (terrain z≈0 to
        # tower z≈100). A legitimate wall triangle sits ON the tower and
        # spans ~1-2 m; a curtain spans tens of metres. Discriminate by face
        # height span, not by normal orientation (walls are legal).
        zspan = np.max(np.stack([p0[:, 2], p1[:, 2], p2[:, 2]], axis=1), axis=1) - \
            np.min(np.stack([p0[:, 2], p1[:, 2], p2[:, 2]], axis=1), axis=1)
        assert float(zspan.max()) < 10.0, f"curtain survived: face spans {zspan.max():.1f} m of height"
        assert np.percentile(edges, 95) < 12.0, f"p95 3D edge {np.percentile(edges, 95):.1f} m — bridge regression"
        # The mesh must still cover the tower itself (structure preserved,
        # not just deleted): tower-wall points must be near mesh vertices.
        from scipy.spatial import cKDTree
        d, _ = cKDTree(mesh.vertices).query(tower, k=1)
        assert np.median(d) < 0.5, "tower wall lost from the mesh"
        # Informational: some vertical-like faces are legitimate (the wall).
        assert vertical_like < 0.5

    def test_disconnected_regions_not_bridged(self):
        """Two terrain patches separated by an unobserved gap must stay
        separate — no invented triangles across the gap."""
        left = _terrain_cloud(3000, side=40.0) - np.array([40.0, 0.0, 0.0])
        right = _terrain_cloud(3000, side=40.0) + np.array([40.0, 0.0, 0.0])
        cloud = PointCloud(xyz=np.vstack([left, right]))
        mesh = _surface_mesh(cloud, max_edge_factor=4.0)
        # Any triangle spanning the gap would have a ~40 m edge; the local
        # spacing budget (~1 m) rejects them all.
        p0, p1, p2 = mesh.vertices[mesh.faces[:, 0]], mesh.vertices[mesh.faces[:, 1]], mesh.vertices[mesh.faces[:, 2]]
        edges = np.stack([
            np.linalg.norm(p1 - p0, axis=1),
            np.linalg.norm(p2 - p1, axis=1),
            np.linalg.norm(p0 - p2, axis=1),
        ], axis=1)
        assert float(edges.max()) < 20.0, "bridge across unobserved gap survived"


# ---------------------------------------------------------------------------
# Ball-pivot production mesher (Part 4): same failure modes, new method.
# ---------------------------------------------------------------------------

from app.services.mesh_generator import (  # noqa: E402
    MeshParams,
    generate_mesh,
    nn_spacing_stats,
    poisson_unsupported_reason,
)


def _oriented_cloud(xyz: np.ndarray, toward: np.ndarray) -> PointCloud:
    """Cloud with camera-oriented normals (as the fusion stage now emits)."""
    from app.services.pointcloud import PointCloud as PC

    v = toward - xyz
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return PC(xyz=xyz, normals=v)


class TestAutoMethodResolution:
    def test_auto_resolves_to_poisson(self):
        """'auto' is the production Poisson surface. Measured on the airport1
        run's own fused cloud: the Delaunay default turned it into 875,914 m^2
        of mesh over a 121,576 m^2 footprint with 23% of edges on a hole
        boundary and 17,040 components; Poisson gave 0.2% boundary edges and
        415 components at 0.351 m median vertex-to-cloud distance — it follows
        the data, it does not smooth it away. `surface` and `ball_pivot` stay
        selectable by name."""
        # A cloud sampled at the scale the Poisson octree resolves. A sparse
        # fixture (0.4 m sampling over 60 m) is legitimately diverted to
        # `surface` by the guard; that case is pinned separately below.
        cloud = _oriented_cloud(_terrain_cloud(6000, side=60.0),
                                toward=np.array([0.0, 0.0, 60.0]))
        mesh, stats = generate_mesh(cloud, MeshParams(method="auto", poisson_depth=7))
        assert stats["method"] == "poisson"
        assert mesh.m > 0
        # The surface still follows the measurements: every vertex sits well
        # inside the cloud's own sampling spacing (nothing is invented).
        from scipy.spatial import cKDTree

        d, _ = cKDTree(np.asarray(cloud.xyz)).query(np.asarray(mesh.vertices), k=1)
        sampling = nn_spacing_stats(np.asarray(cloud.xyz))["p50"]
        assert float(np.median(d)) < sampling

    def test_auto_falls_back_to_surface_without_normals(self):
        """Poisson needs oriented normals; 'auto' must never fail where the
        old Delaunay default succeeded, so a cloud that arrives without them
        resolves to `surface` instead of raising MeshNotPossible."""
        from app.services.pointcloud import PointCloud as PC

        cloud = PC(xyz=_terrain_cloud(800))
        assert cloud.normals is None
        mesh, stats = generate_mesh(cloud, MeshParams(method="auto"))
        assert stats["method"] == "surface"
        assert mesh.m > 0

    def test_auto_falls_back_when_the_octree_cannot_sample_the_extent(self):
        """Poisson's octree must stay within the cloud's own sampling scale.
        When it does not, Open3D returns no error — it dies natively
        (measured: the autonomous pipeline's seeded scene segfaulted the whole
        process out of create_from_point_cloud_poisson at depth 10), so
        `auto` resolves to `surface` there instead of calling it."""
        cloud = _oriented_cloud(_terrain_cloud(2000, side=1.0),
                                toward=np.array([0.0, 0.0, 0.6]))
        reason = poisson_unsupported_reason(cloud, MeshParams(method="auto"))
        assert reason is not None and "finer than" in reason
        mesh, stats = generate_mesh(cloud, MeshParams(method="auto"))
        assert stats["method"] == "surface"
        assert mesh.m > 0

    def test_poisson_keeps_clouds_at_their_own_sampling_scale(self):
        """The production-shaped case keeps Poisson. The airport1 run's fused
        cloud measures an octree cell of 0.4825 m against 0.4590 m sampling
        (ratio 1.05) at the configured depth 10, so the guard must not fire
        on clouds sampled at the scale the octree resolves."""
        cloud = _oriented_cloud(_terrain_cloud(6000, side=60.0),
                                toward=np.array([0.0, 0.0, 60.0]))
        assert poisson_unsupported_reason(
            cloud, MeshParams(method="auto", poisson_depth=4)) is None

    def test_ball_pivot_still_measures_its_radii(self):
        """Ball pivot stays available by name, with measured radii recorded."""
        cloud = _oriented_cloud(_terrain_cloud(800), toward=np.array([0.0, 0.0, 60.0]))
        mesh, stats = generate_mesh(cloud, MeshParams(method="ball_pivot"))
        assert stats["method"] == "ball_pivot"
        assert mesh.m > 0
        assert "radii_m" in mesh.meta and all(r > 0 for r in mesh.meta["radii_m"])

    def test_bpa_terrain_plus_tower_no_curtains(self):
        """BPA on terrain+tower must not stitch the two: ball-pivot only
        triangulates locally supported triplets, so faces spanning from
        terrain z≈0 to tower top z≈100 (the curtain signature) cannot exist."""
        xyz = _terrain_cloud(4000)
        rng = np.random.default_rng(2)
        hh = rng.uniform(0.2, 100.0, 1200)
        aa = rng.uniform(0, 2 * np.pi, 1200)
        tower = np.column_stack([1.5 + 1.5 * np.cos(aa), -2.0 + 1.5 * np.sin(aa), hh])
        pts = np.vstack([xyz, tower])
        # Orient each point toward its own observing sensor (terrain seen from
        # above, tower wall from the side) — exactly the fused-normal setup.
        normals = np.zeros_like(pts)
        normals[: len(xyz)] = np.array([0.0, 0.0, 1.0])
        wall_dir = tower[:, :2] - np.array([1.5, -2.0])
        wall_dir /= np.linalg.norm(wall_dir, axis=1, keepdims=True)
        normals[len(xyz):] = np.column_stack([wall_dir, np.zeros(len(tower))])
        cloud = PointCloud(xyz=pts, normals=normals)
        mesh, stats = generate_mesh(cloud, MeshParams(method="ball_pivot"))
        assert mesh.m > 0
        p0, p1, p2 = mesh.vertices[mesh.faces[:, 0]], mesh.vertices[mesh.faces[:, 1]], mesh.vertices[mesh.faces[:, 2]]
        zspan = np.max(np.stack([p0[:, 2], p1[:, 2], p2[:, 2]], axis=1), axis=1) - \
            np.min(np.stack([p0[:, 2], p1[:, 2], p2[:, 2]], axis=1), axis=1)
        # Curtains span tens of metres; legal wall/terrain faces span ~spacing.
        assert float(zspan.max()) < 10.0, f"BPA curtain: face spans {zspan.max():.1f} m"
        # Tower wall must still be meshed (structure preserved).
        from scipy.spatial import cKDTree
        d, _ = cKDTree(mesh.vertices).query(tower, k=1)
        assert np.median(d) < 1.0, "tower wall lost from BPA mesh"

    def test_bpa_disconnected_regions_stay_disconnected(self):
        """Unobserved gap between two terrain patches must not be bridged:
        BPA never spans empty space wider than its largest ball radius."""
        left = _terrain_cloud(3000, side=40.0) - np.array([40.0, 0.0, 0.0])
        right = _terrain_cloud(3000, side=40.0) + np.array([40.0, 0.0, 0.0])
        pts = np.vstack([left, right])
        normals = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
        cloud = PointCloud(xyz=pts, normals=normals)
        mesh, _ = generate_mesh(cloud, MeshParams(method="ball_pivot"))
        p0, p1, p2 = mesh.vertices[mesh.faces[:, 0]], mesh.vertices[mesh.faces[:, 1]], mesh.vertices[mesh.faces[:, 2]]
        edges = np.stack([
            np.linalg.norm(p1 - p0, axis=1),
            np.linalg.norm(p2 - p1, axis=1),
            np.linalg.norm(p0 - p2, axis=1),
        ], axis=1)
        assert float(edges.max()) < 20.0, "BPA bridged the unobserved gap"
