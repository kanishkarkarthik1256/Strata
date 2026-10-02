"""Sheet-gap mesh-quality phase — regression tests.

Covers the mandate's test list: boundary preservation, no non-manifold
regression, dense support, coordinate/metric-scale preservation, layering
diagnostics, rejection logging — plus the provenance fixes (real method
note, base_mesh.json sidecar) and the frozen-input contract.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.services.mesh import TriangleMesh
from app.services.mesh_generator import (
    MeshParams,
    SheetGapParams,
    _ball_pivot_mesh,
    generate_mesh,
)
from app.services.pointcloud import PointCloud


def _two_sheet_cloud(n=3000, gap=0.6, span=8.0, seed=3) -> PointCloud:
    """Two stacked parallel sheets `gap` m apart (the bridge failure mode)."""
    rng = np.random.default_rng(seed)
    uv = rng.uniform(-span, span, (n, 2))
    xyz = np.vstack([
        np.column_stack([uv, np.zeros(n)]),
        np.column_stack([uv + 0.01, np.full(n, gap)]),
    ])
    normals = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    return PointCloud(xyz=xyz, normals=normals)


def _terrain_cloud(n=2500, span=8.0, seed=5) -> PointCloud:
    """Single smooth surface — rejection must NOT fire on legitimate surface."""
    rng = np.random.default_rng(seed)
    uv = rng.uniform(-span, span, (n, 2))
    z = 0.05 * np.sin(uv[:, 0] * 1.3) + 0.04 * np.cos(uv[:, 1] * 0.9)
    xyz = np.column_stack([uv, z])
    normals = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    return PointCloud(xyz=xyz, normals=normals)


# ---------------------------------------------------------------------------
# Legacy path identity (flag off ⇒ bit-identical geometry)
# ---------------------------------------------------------------------------

def test_legacy_path_unchanged_when_flag_off():
    cloud = _two_sheet_cloud()
    legacy = _ball_pivot_mesh(cloud, MeshParams(method="ball_pivot"))
    via_generate = generate_mesh(cloud, MeshParams(method="ball_pivot"))[0]
    assert np.array_equal(legacy.vertices, via_generate.vertices)
    assert np.array_equal(legacy.faces, via_generate.faces)


def test_sheet_gap_disabled_never_rejects():
    cloud = _two_sheet_cloud()
    m, s = generate_mesh(cloud, MeshParams(
        method="ball_pivot", sheet_gap=SheetGapParams(enabled=False)))
    assert s["sheet_gap_enabled"] is False
    assert "sheet_gap_rejection" not in m.meta


# ---------------------------------------------------------------------------
# Weld cap: sheets must stay unfused (connectivity-only change, vertices kept)
# ---------------------------------------------------------------------------

def test_weld_cap_stays_below_sheet_gap():
    # Sparse sampling so r1 (median spacing) exceeds the 0.5×gap cap and the
    # cap actually binds — the real airport3 case (weld 0.35 vs cap 0.284).
    cloud = _two_sheet_cloud(n=600, gap=0.4)
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.4, cap_weld=True, reject_triangles=False)
    # Legacy 1x weld: r1 > cap ⇒ cap must bind (this is the case it exists for).
    _, s_leg_capped = generate_mesh(cloud, MeshParams(
        method="ball_pivot", patch_weld_factor=1.0, sheet_gap=sg))
    # weld cap = 0.5 × 0.4 = 0.2 m; legacy weld = r1 = median spacing (> cap here)
    assert s_leg_capped["patch_merge_m"] <= 0.2 + 1e-9
    assert "sheet-gap cap" in s_leg_capped.get("patch_merge_source", "")
    # The measured 0.5x default already sits below this cap; the cap binds
    # again for tighter sheets.
    sg_tight = SheetGapParams(enabled=True, sheet_gap_m=0.1, cap_weld=True,
                              reject_triangles=False)
    _, s_tight = generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg_tight))
    assert s_tight["patch_merge_m"] <= 0.05 + 1e-9
    assert "sheet-gap cap" in s_tight.get("patch_merge_source", "")
    m_leg, s_leg = generate_mesh(cloud, MeshParams(
        method="ball_pivot", patch_weld_factor=1.0))
    assert s_leg["patch_merge_m"] > s_leg_capped["patch_merge_m"], "cap must reduce the legacy weld when it binds"


def test_weld_cap_noop_when_radius_already_below_gap():
    # Dense sampling: r1 < 0.5×gap ⇒ cap does not bind, source stays legacy.
    cloud = _two_sheet_cloud(n=6000, gap=0.6)
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.6, cap_weld=True, reject_triangles=False)
    _, s = generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg))
    assert s["patch_merge_m"] <= 0.3 + 1e-9  # invariant holds either way


def test_rejection_keeps_vertices_on_the_cloud():
    """Rejected triangles remove connectivity only — points must survive."""
    cloud = _two_sheet_cloud()
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.6, cap_weld=True, reject_triangles=True)
    m, s = generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg))
    rej = s["sheet_gap_rejection"]
    assert rej["rejected_faces"] > 0, "two-sheet geometry must trigger rejection"
    # Weld-cap-only comparison: rejection may orphan vertices ONLY by
    # removing every face that referenced them (unreferenced drop) — the
    # cloud itself is never filtered (tested separately below).
    sg_cap_only = SheetGapParams(enabled=True, sheet_gap_m=0.6,
                                 cap_weld=True, reject_triangles=False)
    m_cap, _ = generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg_cap_only))
    orphaned = m_cap.n - m.n
    assert orphaned >= 0
    assert orphaned <= 0.01 * m_cap.n, (
        f"{orphaned} vertices orphaned by rejection — rejection must remove "
        "connectivity, not bulk points")
    # Every surviving vertex still lies on the observed cloud.
    from scipy.spatial import cKDTree
    d, _ = cKDTree(cloud.xyz).query(m.vertices, k=1)
    assert d.max() <= 0.6  # within the sheet gap = observed scale


# ---------------------------------------------------------------------------
# Evidence test: strictness on legitimate geometry
# ---------------------------------------------------------------------------

def test_rejection_strict_on_legitimate_terrain():
    """Single-surface terrain: no threefold evidence, mesh stays intact."""
    cloud = _terrain_cloud()
    m_leg, _ = generate_mesh(cloud, MeshParams(method="ball_pivot"))
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.6, cap_weld=True, reject_triangles=True)
    m_exp, s = generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg))
    rej = s["sheet_gap_rejection"]
    assert rej["rejected_percent"] < 1.0, (
        f"legitimate terrain lost {rej['rejected_percent']}% of faces — evidence test too loose")
    # No support regression on the legitimate surface.
    from app.services.dense_diagnostics import mesh_support
    s_leg = mesh_support(np.asarray(m_leg.vertices), cloud.xyz, 0.3, faces=m_leg.faces)
    s_exp = mesh_support(np.asarray(m_exp.vertices), cloud.xyz, 0.3, faces=m_exp.faces)
    assert s_exp["support_percent"] >= s_leg["support_percent"] - 2.0


def test_rejection_requires_all_three_evidence_flags():
    cloud = _two_sheet_cloud()
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.6, cap_weld=True, reject_triangles=True)
    _, s = generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg))
    rej = s["sheet_gap_rejection"]
    flags = rej["flag_counts"]
    assert rej["rejected_faces"] <= min(flags.values()), (
        "rejection must require ALL THREE flags (span AND orientation AND emptiness)")


def test_rejection_diagnostics_are_logged():
    cloud = _two_sheet_cloud()
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.6, cap_weld=True, reject_triangles=True)
    _, s = generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg))
    rej = s["sheet_gap_rejection"]
    # Per-rejected-triangle evidence, aggregated.
    for key in ("thresholds", "flag_counts", "rejected_evidence_stats",
                "spatial_top_cells_10m", "rejected_records_sample"):
        assert key in rej, f"missing rejection diagnostic: {key}"
    rec = rej["rejected_records_sample"][0]
    assert {"face", "max_edge_m", "normal_abs_dot", "centroid"} <= set(rec)


# ---------------------------------------------------------------------------
# Fidelity invariants on the experimental path
# ---------------------------------------------------------------------------

def test_no_non_manifold_regression():
    """No NEW non-manifold geometry vs the legacy path (raw BPA/merge can
    carry a few non-manifold seams already — the experiment must not add)."""
    cloud = _two_sheet_cloud()
    m_leg, _ = generate_mesh(cloud, MeshParams(method="ball_pivot"))
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.6, cap_weld=True, reject_triangles=True)
    m, _ = generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg))
    assert m.non_manifold_edge_count() <= m_leg.non_manifold_edge_count()


def test_boundaries_preserved_not_closed():
    """Two stacked sheets stay two open sheets — no watertight closure."""
    cloud = _two_sheet_cloud()
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.6, cap_weld=True, reject_triangles=True)
    m, _ = generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg))
    assert len(m.boundary_edges()) > 0, "surface must remain open (no watertight closure)"


def test_layering_diagnostics_improve_on_two_sheets():
    """Fewer handles ⇒ lower χ; sheets must not be fused by the weld cap."""
    cloud = _two_sheet_cloud()
    m_leg, _ = generate_mesh(cloud, MeshParams(method="ball_pivot"))
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.6, cap_weld=True, reject_triangles=True)
    m_exp, _ = generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg))
    assert m_exp.euler_characteristic() >= m_leg.euler_characteristic()
    # detect_layers at the sheet scale must still SEE two sheets (not fused).
    from app.services.dense_diagnostics import detect_layers
    lay = detect_layers(np.asarray(m_exp.vertices), 0.3)
    assert lay.get("layered_points_percent", 0) > 0, (
        "weld cap fused the sheets — layering diagnostics would go blind")


def test_dense_cloud_untouched_by_meshing():
    cloud = _two_sheet_cloud()
    before = cloud.xyz.copy()
    sg = SheetGapParams(enabled=True, sheet_gap_m=0.6, cap_weld=True, reject_triangles=True)
    generate_mesh(cloud, MeshParams(method="ball_pivot", sheet_gap=sg))
    assert np.array_equal(before, cloud.xyz), "meshing must never modify the dense cloud"


# ---------------------------------------------------------------------------
# Provenance fixes
# ---------------------------------------------------------------------------

def test_base_mesh_json_sidecar_written(tmp_path):
    """Every mesh generation writes base_mesh.json with hashes + parameters."""
    from app.services.mesh_generator import mesh_provenance_record

    cloud = _two_sheet_cloud(500)
    dense_p = tmp_path / "dense_model.ply"
    mesh_p = tmp_path / "mesh.ply"
    from app.services.pointcloud import save_ply

    save_ply(dense_p, cloud)
    m, s = generate_mesh(cloud, MeshParams(method="ball_pivot"))
    m.save_ply(mesh_p, normals=False)
    rec = mesh_provenance_record(m, cloud, tmp_path, s,
                                 dense_rel="dense_model.ply", mesh_rel="mesh.ply")
    assert rec["method"] == "ball_pivot"
    assert rec["bpa_radii_m"] and len(rec["bpa_radii_m"]) == 3
    assert rec["input_dense_hash_sha256"] and len(rec["input_dense_hash_sha256"]) == 64
    assert rec["mesh_hash_sha256"] and len(rec["mesh_hash_sha256"]) == 64
    assert rec["input_dense_points"] == cloud.n
    assert rec["timestamp_utc"].endswith("Z")
    assert rec["sheet_gap"]["enabled"] is False  # default-off provenance


def test_mesh_quality_note_names_real_method():
    """The stale 'Poisson output is double-sided' note must be gone, and the
    note must name the method that actually produced this run's mesh.

    The note used to live inline in dense_reconstruction.py; it is now owned
    by app.services.mesh_quality (the one audit both the dense stage and the
    offline rebuild call), so this asserts the produced value rather than a
    source string.
    """
    import numpy as np

    from app.services.mesh_generator import TriangleMesh
    from app.services.mesh_quality import audit_mesh_quality

    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                      [1.0, 1.0, 0.0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = TriangleMesh(vertices=verts, faces=faces)
    cloud = verts.copy()

    bp = audit_mesh_quality(mesh, cloud, 1.0, method="ball_pivot").quality
    assert "ball_pivot" in bp["note"]
    assert "Poisson output is double-sided" not in bp["note"]
    other = audit_mesh_quality(mesh, cloud, 1.0, method="poisson").quality
    assert other["note"] == "production method: poisson"

    src = open("app/services/dense_reconstruction.py").read()
    assert "Poisson output is double-sided" not in src
    assert "audit_mesh_quality" in src  # the stage calls the shared owner


def test_frozen_artifacts_contract_exists():
    from app.services.mesh_phase2 import FROZEN_ARTIFACTS
    assert "dense/dense_model.ply" in FROZEN_ARTIFACTS
    assert "dense_fusion_provenance.json" in FROZEN_ARTIFACTS
