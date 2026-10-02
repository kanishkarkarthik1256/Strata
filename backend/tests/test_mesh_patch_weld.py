"""Patch-weld regression — the mesh must sit ON the measured cloud.

The legacy BPA patch weld (``1x`` smallest radius = 1x median spacing) merged
up to a third of a 2.23M-point cloud into blob centroids (furnerhem_6:
mesh→cloud deviation p95 0.264 m, 2.23M→1.46M vertices) — the visible
"mesh differs from the pointcloud" defect. The measured default is
``0.5x`` (below sampling noise, still stitches same-surface patches):
deviation p95 0.000 m, +30% vertices, ~20% faster. These tests pin that
contract on synthetic clouds; the sheet-gap cap keeps the ability to bind
below the weld when real stacked sheets exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.mesh_generator import MeshParams, SheetGapParams, _ball_pivot_mesh
from app.services.pointcloud import PointCloud


def _flat_sheet_cloud(n=6000, span=8.0, seed=7) -> PointCloud:
    rng = np.random.default_rng(seed)
    uv = rng.uniform(-span, span, (n, 2))
    z = 0.03 * np.sin(uv[:, 0] * 1.1) + 0.02 * np.cos(uv[:, 1] * 0.8)
    xyz = np.column_stack([uv, z])
    normals = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    return PointCloud(xyz=xyz, normals=normals)


def test_mesh_vertices_sit_on_cloud_points():
    """The measured contract: with the default weld, mesh vertices coincide
    with cloud points (median NN distance ~0), never blob centroids."""
    cloud = _flat_sheet_cloud()
    mesh = _ball_pivot_mesh(cloud, MeshParams(method="ball_pivot"))
    tree = _kdtree(cloud.xyz)
    dist, _ = tree.query(mesh.vertices, k=1)
    assert float(np.median(dist)) < 1e-6, (
        "weld must not move mesh vertices off the measured cloud"
    )
    # No vertex may sit between samples either: p95 stays below half spacing.
    spacing = float(np.median(_nn(cloud.xyz)))
    assert float(np.percentile(dist, 95)) < 0.5 * spacing


def test_weld_still_stitches_same_surface_patches():
    """0.5x weld keeps patch-seam stitching: a smooth sheet must come out
    largely connected (component count far below the per-patch count)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    cloud = _flat_sheet_cloud()
    mesh = _ball_pivot_mesh(cloud, MeshParams(method="ball_pivot"))
    nv = mesh.n
    rows = np.concatenate([mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]])
    cols = np.concatenate([mesh.faces[:, 1], mesh.faces[:, 2], mesh.faces[:, 0]])
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nv, nv)).tocsr()
    n_comp, _ = connected_components(adj, directed=False)
    # 6k points triangulate in many patches without a weld; stitched, a
    # smooth sheet collapses to a handful of components.
    assert n_comp <= 8, f"same-surface patches not stitched: {n_comp} components"


def test_weld_weaker_than_legacy_keeps_more_vertices():
    cloud = _flat_sheet_cloud()
    fixed = _ball_pivot_mesh(cloud, MeshParams(method="ball_pivot"))
    legacy = _ball_pivot_mesh(
        cloud, MeshParams(method="ball_pivot", patch_weld_factor=1.0))
    assert fixed.n >= legacy.n, "gentler weld must not weld away more of the cloud"
    # On this cloud the legacy weld visibly destroys geometry.
    assert fixed.n > legacy.n * 1.15


def test_sheet_gap_cap_can_still_bind_below_weld():
    cloud = _flat_sheet_cloud()
    # A tiny sheet gap forces the weld cap below the 0.5x default.
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.2, cap_weld=True,
                        reject_triangles=False)
    mesh = _ball_pivot_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg))
    spacing = float(np.median(_nn(cloud.xyz)))
    legacy_weld = spacing * 1.0
    assert (mesh.meta.get("patch_merge_m") or 0) < legacy_weld, (
        "sheet-gap cap must be able to bind below the measured default weld"
    )


def _spread_sheet_cloud(n=9000, spacing=0.5, span=40.0, seed=11) -> PointCloud:
    """Far-field-like sheet: sampling whose local NN spacing SPREADS.

    Uniform dropout of 25% of a grid widens the NN distribution the way the
    fused far-field clouds do (measured on Video_Mission_9ab1aa: p50 0.665 m
    vs p90 0.942 m), which is what a median-rooted ball ladder cannot span.
    """
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(
        np.arange(-span, span, spacing), np.arange(-span, span, spacing)), -1)
    uv = g.reshape(-1, 2)
    keep = rng.random(len(uv)) > 0.25
    uv = uv[keep] + rng.normal(0, 0.06, (int(keep.sum()), 2))
    z = 0.4 * np.sin(uv[:, 0] * 0.05) + 0.3 * np.cos(uv[:, 1] * 0.04)
    xyz = np.column_stack([uv, z])
    normals = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    return PointCloud(xyz=xyz, normals=normals)


def _largest_edge_component_pct(mesh) -> tuple[int, float]:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(mesh.faces, dtype=np.int64)
    nv = int(f.max()) + 1
    pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]])
    key = pairs.min(1).astype(np.int64) * nv + pairs.max(1)
    order = np.argsort(key, kind="stable")
    fe = np.tile(np.arange(len(f)), 3)[order]
    ks = key[order]
    same = ks[1:] == ks[:-1]
    n_comp, lab = connected_components(
        sp.coo_matrix((np.ones(int(same.sum())), (fe[:-1][same], fe[1:][same])),
                      shape=(len(f), len(f))).tocsr(), directed=False)
    return n_comp, float(np.bincount(lab).max() / len(f) * 100.0)


def test_radii_ladder_is_rooted_at_measured_p90():
    """The ladder basis is the rule this fix replaces: radii = factors × P90
    measured spacing, never the median (which left the smallest ball under
    the gaps that occur on the real far-field clouds)."""
    from app.services.mesh_generator import nn_spacing_stats

    cloud = _spread_sheet_cloud()
    stats = nn_spacing_stats(np.asarray(cloud.xyz))
    assert stats["p90"] > stats["p50"], "fixture must carry a spacing spread"
    mesh = _ball_pivot_mesh(cloud, MeshParams(method="ball_pivot"))
    factors = MeshParams().ball_pivot_radii_factors
    assert MeshParams().ball_pivot_radii_factors == [1.0, 1.5, 2.33]
    assert mesh.meta["radii_basis"].startswith("p90")
    assert mesh.meta["nn_spacing_stats_m"]["p90"] > mesh.meta["nn_spacing_m"]
    assert np.allclose(
        mesh.meta["radii_m"], [stats["p90"] * f for f in factors], atol=5e-3)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1]
         / "data/storage/Video_Mission_9ab1aa/dense/dense_model.ply").is_file(),
    reason="real fused cloud artifact not present",
)
def test_real_far_field_cloud_meshes_as_one_surface():
    """Measured shatter defect on the real run (Video_Mission_9ab1aa): with
    the median-rooted ladder the same cloud meshed to 1.72M faces in 108,470
    edge-components (largest 1.8%) — the broken mesh the viewer showed. The
    P90-rooted ladder must dominate it on the SAME cloud, where synthetic
    fixtures cannot reproduce the shatter at all."""
    import open3d as o3d

    from app.services.mesh_generator import nn_spacing_stats

    ply = (Path(__file__).resolve().parents[1]
           / "data/storage/Video_Mission_9ab1aa/dense/dense_model.ply")
    pcd = o3d.io.read_point_cloud(str(ply))
    xyz = np.asarray(pcd.points)
    stats = nn_spacing_stats(xyz)

    old = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([stats["p50"] * f for f in (1, 2, 4)]))
    old = old.merge_close_vertices(stats["p50"] * 0.5)
    old.remove_degenerate_triangles()
    _c, old_pct = _largest_edge_component_pct(
        type("M", (), {"faces": np.asarray(old.triangles)})())

    cloud = PointCloud(xyz=xyz, normals=np.asarray(pcd.normals))
    mesh = _ball_pivot_mesh(cloud, MeshParams(method="ball_pivot"))
    n_comp, pct = _largest_edge_component_pct(mesh)

    assert old_pct < 30.0, f"median-rooted ladder no longer shatters: {old_pct:.1f}%"
    assert pct > 60.0, (
        f"P90-rooted ladder must mesh one dominant surface: {pct:.1f}% "
        f"({n_comp} components)"
    )


def _kdtree(xyz):
    from scipy.spatial import cKDTree
    return cKDTree(np.asarray(xyz, dtype=np.float64))


def _nn(xyz):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(xyz, dtype=np.float64))
    d, _ = tree.query(np.asarray(xyz, dtype=np.float64), k=2, workers=-1)
    return d[:, 1]


def test_bpa_scale_collapse_retry_meshes_mixture_cloud():
    """The measured e2e_video_1 failure: a cloud whose NN-spacing is a mixture
    (dense twin subpopulation at 0.013 next to a sheet at 0.06) stalls BPA's
    first ball below the sheet's scale — the legacy path returned double-digit
    triangles and produced an empty mesh dir after welding. Radii from the
    upper-quartile NN retry must mesh the sheet; the normal median path must
    be untouched when it already works."""
    rng = np.random.default_rng(3)
    g = np.stack(np.meshgrid(np.linspace(-40, 40, 160), np.linspace(-30, 30, 120)), -1)
    sheet = g.reshape(-1, 2)
    # Depth-resampling twins: half the sheet duplicated at 1/4-sheet offset —
    # a dense subpopulation that drags the NN median below the sheet scale.
    n_t = len(sheet[::2])
    twins = sheet[::2] + np.column_stack(
        [np.full(n_t, 0.015), np.zeros(n_t)]
    )
    uv = np.vstack([sheet, twins])
    xyz = np.column_stack([uv, np.zeros(len(uv))])
    normals = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    cloud = PointCloud(xyz=xyz, normals=normals)

    mesh = _ball_pivot_mesh(cloud, MeshParams(method="ball_pivot"))
    # The retry must produce a real surface, not the stalled first pass.
    assert len(mesh.faces) > 10_000, f"mixture cloud still stalls: {len(mesh.faces)} triangles"
    # The retried radii carry the sheet's scale (upper quartile), and the
    # provenance shows it.
    radii = mesh.meta["radii_m"]
    assert radii[0] > 0.01, radii
    # Vertices still sit ON the cloud (the weld contract holds).
    tree = _kdtree(cloud.xyz)
    dist, _ = tree.query(mesh.vertices, k=1)
    assert float(np.median(dist)) < 1e-6


def test_normal_cloud_path_unchanged_by_retry_guard():
    """A clean single-scale cloud must never trigger the retry — the guard
    is a fallback, not a behavior change — and the first pass must use the
    measured P90 ladder basis (not a retry-derived one)."""
    from app.services.mesh_generator import nn_spacing_stats

    cloud = _flat_sheet_cloud()
    mesh = _ball_pivot_mesh(cloud, MeshParams(method="ball_pivot"))
    stats = nn_spacing_stats(np.asarray(cloud.xyz))
    radii = mesh.meta["radii_m"]
    factors = MeshParams().ball_pivot_radii_factors
    # First pass stands: radii are exactly factors x the measured P90 spacing.
    assert abs(radii[0] - stats["p90"] * factors[0]) < 1e-3, (
        f"radii moved off the measured P90 basis: {radii[0]} vs "
        f"{stats['p90'] * factors[0]}"
    )
    assert mesh.meta["radii_basis"].startswith("p90")


def test_island_pruning_drops_shatter_not_surface():
    """The measured shatter failure (estrel_c0cf6d: 9,820 components, largest
    90.7%): BPA emits thousands of tiny clusters around boundary noise. A
    cluster with few vertices AND tiny diameter is dropped; anything at the
    sampling scale survives. The provenance records what was pruned."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    cloud = _flat_sheet_cloud()
    mesh = _ball_pivot_mesh(cloud, MeshParams(method="ball_pivot"))
    assert mesh.meta.get("island_pruning", {}).get("enabled") is True

    nv = mesh.n
    rows = np.concatenate([mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]])
    cols = np.concatenate([mesh.faces[:, 1], mesh.faces[:, 2], mesh.faces[:, 0]])
    n_comp, _ = connected_components(
        sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nv, nv)).tocsr(),
        directed=False)
    # A smooth sheet: pruning must never shatter it — few components remain.
    assert n_comp <= 8, f"pruning broke a real surface: {n_comp} components"

    # Every surviving cluster must exceed at least one pruning bound.
    min_pts = MeshParams().island_min_points
    labels_arr = np.asarray(_labels_of(mesh))
    sizes = np.bincount(labels_arr)
    small = [c for c in range(len(sizes)) if sizes[c] <= min_pts]
    for c in small:
        pts = mesh.vertices[labels_arr == c]
        span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        cap = MeshParams().island_max_diameter_factor * float(mesh.meta["radii_m"][0])
        assert span > cap, "a tiny cluster survived pruning"


def _labels_of(mesh):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    nv = mesh.n
    f = np.asarray(mesh.faces)
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    _, labels = connected_components(
        sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nv, nv)).tocsr(),
        directed=False)
    return labels
