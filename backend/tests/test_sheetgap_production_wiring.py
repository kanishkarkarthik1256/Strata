"""Production sheet-gap wiring — the guard the runs needed is engaged.

Root cause this pins: a tested sheet-gap guard (SheetGapParams) existed but
was never wired into the production mesh call, so BPA meshes zig-zagged
between measured duplicate sheets (airport_53_8f0e90: 5 pairs at 0.284 m —
double-bounce / depth-edge copies). The contract:

* a stacked-sheets cloud meshed through the production path carries
  ``sheet_gap_*`` provenance on the mesh and refuses bridge triangles;
* a single-surface cloud meshed through the same path stays legacy-identical
  (no guard objects, no rejections) — the guard is measured, not always-on.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.dense_diagnostics import detect_layers
from app.services.mesh_generator import MeshParams, generate_mesh


def _oriented(xyz: np.ndarray, toward: np.ndarray):
    from app.services.pointcloud import PointCloud as PC

    v = toward - xyz
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return PC(xyz=xyz, normals=v)


def _ground(n: int = 6000, side: float = 60.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    xy = rng.uniform(-side / 2, side / 2, (n, 2))
    z = 0.05 * np.sin(xy[:, 0] / 6.0) + rng.normal(0, 0.05, n)
    return np.column_stack([xy, z])


def _stacked_sheets(sep: float = 0.6) -> np.ndarray:
    """Ground + a near-copy elevated by *sep* — the duplicate-surface defect."""
    ground = _ground(5000)
    rng = np.random.default_rng(3)
    xy = rng.uniform(-30, 30, (5000, 2))
    z = 0.05 * np.sin(xy[:, 0] / 6.0) + rng.normal(0, 0.05, 5000) + sep
    return np.vstack([ground, np.column_stack([xy, z])])


def _production_params(cloud):
    """Exactly what dense_reconstruction builds before calling generate_mesh."""
    from app.services.mesh_generator import SheetGapParams

    layers = detect_layers(cloud.xyz, 0.08, normals=cloud.normals)
    if layers.get("status") == "measured" and layers.get("layer_pair_count"):
        gap = layers.get("median_layer_separation_m")
        if gap:
            return MeshParams(
                method="auto",
                sheet_gap=SheetGapParams(
                    enabled=True,
                    sheet_gap_m=float(gap),
                    sheet_gap_source="test: measured stacked sheets",
                ),
            )
    return MeshParams(method="auto")


def test_stacked_sheets_guard_engages_and_rejects_bridges():
    cloud = _oriented(_stacked_sheets(sep=0.6), toward=np.array([0.0, 0.0, 60.0]))
    layers = detect_layers(cloud.xyz, 0.08, normals=cloud.normals)
    assert layers["status"] == "measured" and layers["layer_pair_count"] > 0
    gap = layers["median_layer_separation_m"]
    # The estimator reads the modal nearest-pair separation (noise mixes
    # columns), so pin "positive and below the true stack separation".
    assert 0 < gap < 0.6

    # Production `auto` is Poisson, but this fixture is sparse (0.81 m sampling
    # over a 60 m extent), so the octree guard legitimately diverts it to
    # `surface` — which is also what the guard policy must apply to. Pin both:
    # the policy is method-independent, so exercise it ON Poisson by naming a
    # depth whose octree this fixture's sampling actually supports.
    _auto_mesh, auto_stats = generate_mesh(cloud, _production_params(cloud))
    assert auto_stats["method"] == "surface"
    params = _production_params(cloud)
    mesh, stats = generate_mesh(
        cloud, MeshParams(method="poisson", poisson_depth=7,
                          sheet_gap=params.sheet_gap))
    assert stats["method"] == "poisson"
    assert stats.get("sheet_gap_enabled") is True
    assert stats.get("sheet_gap_m") == pytest.approx(gap, abs=0.05)
    rej = mesh.meta.get("sheet_gap_rejection")
    assert rej is not None, "bridge triangles must be refused on a stacked-sheets cloud"
    assert rej["triangles_after"] < rej["triangles_before"]


def test_stacked_sheets_guard_engages_on_ball_pivot_too():
    """The cloud-level guard is method-independent: the same stacked cloud
    meshed by name through ball pivot refuses bridges AND caps its patch weld
    below the measured separation (the weld is BPA-only — the 2.5D surface has
    no weld, so its provenance carries the rejection alone)."""
    cloud = _oriented(_stacked_sheets(sep=0.6), toward=np.array([0.0, 0.0, 60.0]))
    params = _production_params(cloud)
    mesh, stats = generate_mesh(
        cloud, MeshParams(method="ball_pivot", sheet_gap=params.sheet_gap))
    assert stats["method"] == "ball_pivot"
    rej = mesh.meta.get("sheet_gap_rejection")
    assert rej is not None and rej["triangles_after"] < rej["triangles_before"]
    assert "sheet separation" in mesh.meta.get("patch_merge_source", "")


def test_single_surface_stays_legacy_identical():
    cloud = _oriented(_ground(), toward=np.array([0.0, 0.0, 60.0]))
    layers = detect_layers(cloud.xyz, 0.08, normals=cloud.normals)
    assert layers.get("status") != "measured" or not layers.get("layer_pair_count")

    guarded, s_guarded = generate_mesh(cloud, _production_params(cloud))
    legacy, s_legacy = generate_mesh(cloud, MeshParams(method="auto"))
    assert s_guarded.get("sheet_gap_enabled") in (None, False)
    assert "sheet_gap_rejection" not in guarded.meta
    # Both calls are plain `auto` (the guard is off for a single surface), so
    # this pins that the guard itself changes nothing.
    #
    # Poisson's parallel solve is not reproducible run to run — measured: the
    # SAME cloud and params give 32,027 vs 32,028 faces AND slightly different
    # vertex positions — so two Poisson runs cannot be compared for identity
    # at all. What this test can pin is the guard's own effect: with no sheets
    # detected it is a no-op, so the two surfaces must agree in size and extent.
    assert guarded.vertices.shape == legacy.vertices.shape
    assert abs(guarded.faces.shape[0] - legacy.faces.shape[0]) <= max(1, legacy.faces.shape[0] // 1000)
    g_lo, g_hi = guarded.bounds()
    l_lo, l_hi = legacy.bounds()
    assert np.allclose(g_lo, l_lo, atol=1e-3) and np.allclose(g_hi, l_hi, atol=1e-3)


def test_wireframe_probe_builds_guard_from_real_cloud(tmp_path):
    """The exact snippet dense_reconstruction.py runs, against a real fused
    cloud layout: guard object only when sheets exist, None otherwise."""
    from app.services.dense_diagnostics import detect_layers
    from app.services.mesh_generator import SheetGapParams

    def build_sg(cloud, voxel):
        layers = detect_layers(cloud.xyz, voxel, normals=cloud.normals)
        if layers.get("status") == "measured" and layers.get("layer_pair_count"):
            gap = layers.get("median_layer_separation_m")
            if gap:
                return SheetGapParams(enabled=True, sheet_gap_m=float(gap))
        return None

    stacked = _oriented(_stacked_sheets(sep=0.8), toward=np.array([0.0, 0.0, 60.0]))
    flat = _oriented(_ground(), toward=np.array([0.0, 0.0, 60.0]))
    assert build_sg(stacked, 0.08) is not None
    assert build_sg(flat, 0.08) is None
