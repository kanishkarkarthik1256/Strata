"""Phase 2 mesh regression tests.

Covers the spacing field, the sheet-gap-capped radius schedule, the
adaptive BPA bridge rejection, the matched-scale layer evaluation and the
frozen-input verification. Synthetic geometry only — no run artifacts.
"""

import hashlib

import numpy as np
import pytest

from app.services.mesh import TriangleMesh
from app.services.mesh_phase2 import (
    Phase2Params,
    _adaptive_ball_pivot_mesh,
    _radii_schedule,
    assert_frozen,
    evaluate_mesh,
    spacing_field,
    spacing_statistics,
    verify_frozen_inputs,
)
from app.services.pointcloud import PointCloud


@pytest.fixture
def voxel():
    return 1.5


@pytest.fixture
def stacked_sheets_cloud():
    """Two parallel 1 m-grid sheets 4 m apart — a real double-layer case."""
    rng = np.random.default_rng(0)
    g = np.mgrid[0:60:1.0, 0:60:1.0]
    a = np.column_stack([g[0].ravel(), g[1].ravel(), np.full(g[0].size, 10.0)])
    b = np.column_stack([g[0].ravel(), g[1].ravel(), np.full(g[0].size, 14.0)])
    xyz = np.vstack([a, b]) + rng.normal(0, 0.05, (a.shape[0] * 2, 3))
    nrm = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    return PointCloud(xyz=xyz, normals=nrm)


def test_mean_nn_distance_matches_true_sampling(stacked_sheets_cloud):
    """The spacing field must reflect the REAL sampling grid (1 m), not the
    inflated subsample spacing that caused the Phase 2 root-cause bug."""
    sp, qi = spacing_field(np.asarray(stacked_sheets_cloud.xyz), k=6)
    st = spacing_statistics(sp)
    # Grid spacing 1.0 m + kNN-6 widening + noise → roughly 1.0–2.2 m.
    assert 0.9 < st["median_m"] < 2.5, st


def test_radii_schedule_caps_at_measured_sheet_gap(stacked_sheets_cloud):
    sp, _ = spacing_field(np.asarray(stacked_sheets_cloud.xyz), k=6)
    radii, info = _radii_schedule(sp, Phase2Params(), sheet_gap_m=2.28)
    assert max(radii) <= 2.28 + 1e-6
    assert info["bridge_cap_source"] == "min_measured_sheet_gap (dense layer detection)"


def test_radii_schedule_falls_back_to_spacing_cap(stacked_sheets_cloud):
    sp, _ = spacing_field(np.asarray(stacked_sheets_cloud.xyz), k=6)
    radii, info = _radii_schedule(sp, Phase2Params(), sheet_gap_m=None)
    st = spacing_statistics(sp)
    assert max(radii) <= 3.0 * st["p95_m"] + 1e-6
    assert "spacing_p95" in info["bridge_cap_source"]


def test_adaptive_bpa_keeps_both_sheets(stacked_sheets_cloud, voxel):
    mesh, stats = _adaptive_ball_pivot_mesh(
        stacked_sheets_cloud, Phase2Params(), sheet_gap_fallback_voxel=voxel)
    assert mesh.m > 100
    assert stats["triangles_kept"] > 0
    ev = evaluate_mesh(mesh, np.asarray(stacked_sheets_cloud.xyz), voxel)
    # Both sheets must be represented: dense→mesh distances stay sub-voxel.
    assert ev["dense_to_mesh_median"] <= voxel
    assert ev["mesh_support_percent"] == 100.0


def test_adaptive_bpa_never_bridges_an_8m_gap():
    rng = np.random.default_rng(0)
    g = np.mgrid[0:30:1.0, 0:30:1.0]
    h1 = np.column_stack([g[0].ravel(), g[1].ravel(), np.zeros(g[0].size)])
    h2 = np.column_stack([g[0].ravel() + 38.0, g[1].ravel(), np.zeros(g[0].size)])
    xyz = np.vstack([h1, h2]) + rng.normal(0, 0.05, (h1.shape[0] * 2, 3))
    nrm = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    mesh, _ = _adaptive_ball_pivot_mesh(
        PointCloud(xyz=xyz, normals=nrm), Phase2Params(), sheet_gap_fallback_voxel=1.5)
    rows = np.concatenate([mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]])
    cols = np.concatenate([mesh.faces[:, 1], mesh.faces[:, 2], mesh.faces[:, 0]])
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(mesh.n, mesh.n)).tocsr()
    n_comp, _ = connected_components(adj, directed=False)
    assert n_comp >= 2, "an 8 m observation gap must not be bridged"


def test_evaluate_mesh_layers_use_matched_scale(stacked_sheets_cloud, voxel):
    """A clean single-surface mesh measured at its own scale must read ~0%."""
    rng = np.random.default_rng(0)
    g = np.mgrid[0:40:1.0, 0:40:1.0]
    xyz = np.column_stack([g[0].ravel(), g[1].ravel(), np.zeros(g[0].size)])
    nrm = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    mesh, _ = _adaptive_ball_pivot_mesh(
        PointCloud(xyz=xyz, normals=nrm), Phase2Params(), sheet_gap_fallback_voxel=voxel)
    ev = evaluate_mesh(mesh, xyz, voxel)
    assert ev["layered_percent"] <= 5.0, ev


def test_evaluate_mesh_reports_all_contract_fields(stacked_sheets_cloud, voxel):
    mesh, _ = _adaptive_ball_pivot_mesh(
        stacked_sheets_cloud, Phase2Params(), sheet_gap_fallback_voxel=voxel)
    ev = evaluate_mesh(mesh, np.asarray(stacked_sheets_cloud.xyz), voxel)
    for key in (
        "vertices", "faces", "median_edge_length", "p90_edge_length",
        "median_triangle_area", "p90_triangle_area", "degenerate_faces",
        "non_manifold_edges", "boundary_edges", "components",
        "largest_component_percent", "dense_to_mesh_median",
        "dense_to_mesh_p95", "dense_unsupported_percent",
        "mesh_support_percent", "layered_percent", "layer_regions",
        "layer_measurement_scale",
    ):
        assert key in ev, key


def test_verify_frozen_inputs_and_assert(tmp_path):
    p2 = tmp_path / "poses.json"
    p2.write_text("{}")
    before = verify_frozen_inputs(tmp_path)
    assert before["artifacts"]["sparse_model.ply"]["status"] == "missing"
    assert before["artifacts"]["poses.json"]["sha256"] == hashlib.sha256(b"{}").hexdigest()

    after_ok = verify_frozen_inputs(tmp_path)
    assert assert_frozen(before, after_ok)["unchanged"]

    p2.write_text("{\"changed\": true}")
    after_bad = verify_frozen_inputs(tmp_path)
    res = assert_frozen(before, after_bad)
    assert not res["unchanged"]
    assert res["changed_artifacts"] == ["poses.json"]
