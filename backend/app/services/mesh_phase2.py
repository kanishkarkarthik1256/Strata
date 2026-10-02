"""Phase 2 — high-resolution surface reconstruction (dense → mesh ONLY).

FREEZE CONTRACT: every function here treats the upstream artifacts (sparse
reconstruction, poses, BA, GPS/georef, tracks, depth maps, depth alignment,
``dense_model.ply`` and its provenance) as READ-ONLY inputs. Nothing in this
module writes to, re-runs, or retunes any upstream stage; the only writes are
Phase 2 mesh outputs (``mesh/phase2_*``, the promoted ``mesh_full.ply``) and
``mesh/phase2_mesh_report.json``. ``verify_frozen_inputs`` records sha256 +
counts of every upstream artifact so the report can prove the dense cloud the
mesh consumed is byte-identical before and after.

Why the old mesh under-resolves the dense cloud (audited 2026-09-20):
  1. ``merge_close_vertices(r1)`` welds BPA patch seams at the FULL smallest
     radius — with r1 = median spacing this cascades: dom3 welds 1.29 M
     input points down to 35 k vertices (≈37 points per surviving vertex).
  2. Radii {1×, 2×, 4×} × global median spacing: the 4× radius bridges
     gaps up to ~4 spacing units — exactly the scale where sibling surface
     sheets (separation ≥ 1.5 × fusion voxel) live on dom3/airport8.
  3. No post-triangulation validation: triangles spanning empty space
     (unsupported by nearby dense points) survive into the production mesh.

The Phase 2 adaptive method keeps Open3D BPA (observed-surface, no
watertight closure) but derives every threshold from the MEASURED local
spacing field of the authoritative dense cloud:
  * radii from spacing percentiles, capped at ``bridge_cap_factor × p95``
    so no radius can span the minimum real sheet separation;
  * patch-weld at r1/2 (below the local sampling noise) instead of r1;
  * every output triangle is evidence-checked: all three vertices within
    ``support_factor × local spacing`` of dense points, and max edge within
    the local spacing budget — bridging triangles are rejected and counted.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pointcloud import PointCloud

log = get_logger("drone_recon.services.mesh_phase2")

# Upstream artifacts the mesh stage consumes but must never modify.
FROZEN_ARTIFACTS = (
    "sparse_model.ply",
    "poses.json",
    "intrinsics.json",
    "reconstruction_report.json",   # BA + GPS provenance
    "depth_alignment_report.json",  # per-view affine parameters
    "dense/dense_model.ply",        # authoritative dense cloud
    "dense_fusion_provenance.json", # dense provenance index
    "dense_fusion_provenance.npz",  # its per-point arrays (binary)
)


# ---------------------------------------------------------------------------
# Freeze verification (mandate §1)
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_frozen_inputs(ws: Path) -> dict[str, Any]:
    """Hash + count every frozen upstream artifact (read-only)."""
    out: dict[str, Any] = {"workspace": str(ws), "artifacts": {}}
    for rel in FROZEN_ARTIFACTS:
        p = ws / rel
        if not p.is_file():
            out["artifacts"][rel] = {"status": "missing"}
            continue
        entry: dict[str, Any] = {
            "status": "present",
            "sha256": _sha256(p),
            "bytes": p.stat().st_size,
        }
        if rel.endswith(".ply"):
            try:
                from app.services.pointcloud import read_ply

                entry["points"] = int(read_ply(p).n)
            except Exception as exc:  # pragma: no cover
                entry["points_error"] = str(exc)
        out["artifacts"][rel] = entry
    return out


def assert_frozen(before: dict, after: dict) -> dict[str, Any]:
    """Compare two ``verify_frozen_inputs`` snapshots; never raises — reports."""
    diffs: list[str] = []
    for rel, b in before["artifacts"].items():
        a = after["artifacts"].get(rel, {})
        if b.get("status") != "present":
            continue
        if b.get("sha256") != a.get("sha256"):
            diffs.append(rel)
    return {
        "unchanged": not diffs,
        "changed_artifacts": diffs,
        "checked": [r for r, v in before["artifacts"].items() if v.get("status") == "present"],
        "label": "MEASURED (sha256 before vs after the mesh stage)",
    }


# ---------------------------------------------------------------------------
# Part 3 — local point-spacing field on the authoritative dense cloud
# ---------------------------------------------------------------------------

def spacing_field(
    xyz: np.ndarray,
    k: int = 6,
    sample_anchors: int = 100_000,
    max_query: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point kNN spacing + the query indices used.

    IMPORTANT: the KD-tree must be built over ALL points whenever the point
    count allows, otherwise queries far from the (subsampled) anchor set
    return inflated distances — measured 3.63 m vs the true 1.54 m median
    on dom3 when the tree held only 100 k of 1.29 M points. Anchoring is
    only used above ``max_query`` points, where we query a stride sample
    (the spacing FIELD is then anchored-interpolated for the rest).
    """
    from scipy.spatial import cKDTree

    n = len(xyz)
    if n < 2:
        return np.array([]), np.array([], dtype=np.int64)
    q_idx = np.arange(n, dtype=np.int64)
    if n > max_query:
        q_idx = np.sort(np.random.default_rng(0).choice(n, max_query, replace=False))
    pts_q = xyz[q_idx]
    # Tree over every point up to max_query — exact kNN for in-tree queries.
    anchors = xyz if n <= max_query else xyz[q_idx]
    tree = cKDTree(anchors)
    kq = min(k, len(anchors))
    dist, _ = tree.query(pts_q, k=kq + 1, workers=-1)
    d = np.where(dist > 0, dist, np.inf)  # drop self-matches
    d_sorted = np.sort(d, axis=1)[:, :kq]
    spacing = np.median(d_sorted, axis=1)
    return spacing, q_idx


def spacing_statistics(spacing: np.ndarray) -> dict[str, Any]:
    """Robust spacing/density statistics (mandate §3)."""
    s = spacing[np.isfinite(spacing) & (spacing > 0)]
    if len(s) == 0:
        return {"status": "insufficient_points"}
    # Robust outlier handling for min/max: 0.5–99.5 percentile window.
    lo, hi = np.percentile(s, [0.5, 99.5])
    robust = s[(s >= lo) & (s <= hi)]
    med = float(np.median(s))
    return {
        "status": "measured",
        "n_points_measured": int(len(s)),
        "median_m": round(med, 4),
        "p25_m": round(float(np.percentile(s, 25)), 4),
        "p75_m": round(float(np.percentile(s, 75)), 4),
        "p90_m": round(float(np.percentile(s, 90)), 4),
        "p95_m": round(float(np.percentile(s, 95)), 4),
        "min_m_robust": round(float(robust.min()), 4),
        "max_m_robust": round(float(robust.max()), 4),
        "mean_m": round(float(s.mean()), 4),
        "local_density_pts_per_m3_heuristic": round(float(1.0 / max(med, 1e-9) ** 3), 6),
        "labels": "MEASURED (kNN spacing; density is the inverse-cube heuristic)",
    }


# ---------------------------------------------------------------------------
# Part 4/6 — adaptive BPA with sheet-gap cap and bridge rejection
# ---------------------------------------------------------------------------

@dataclass
class Phase2Params:
    """Every threshold derives from measured spacing — nothing hard-coded
    in absolute metres. Recorded verbatim in the mesh report."""

    radii_factors: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    bridge_cap_factor: float = 3.0      # max radius ≤ factor × spacing p95
    patch_weld_divisor: float = 2.0     # weld eps = r1 / divisor (was r1)
    support_factor: float = 3.0         # triangle verts within factor × local spacing of dense
    max_edge_factor: float = 4.0        # triangle max edge ≤ factor × local spacing
    spacing_k: int = 6
    max_input_points: int = 3_000_000
    poisson_depth: int = 9              # diagnostic benchmark only


def _radii_schedule(
    sp: np.ndarray,
    params: Phase2Params,
    sheet_gap_m: float | None = None,
) -> tuple[list[float], dict]:
    """Radii from measured spacing, hard-capped at the smallest REAL surface
    separation when one was measured.

    ``sheet_gap_m`` = min real sheet separation from the dense cloud's own
    layer detection (detect_layers min separation observed in layered cells,
    floored at 1.5 × fusion voxel — the fusion merge scale). A ball larger
    than the nearest other surface rolls across the gap and stitches two
    sheets — the exact dom3 CASE_B mechanism (measured sheet gaps 2.3–4 m;
    the uncapped 3× spacing radius was 10.9 m, i.e. bridge-everything).
    """
    st = spacing_statistics(sp)
    med, p95 = st["median_m"], st["p95_m"]
    cap = params.bridge_cap_factor * p95
    cap_source = "bridge_cap_factor_x_spacing_p95"
    if sheet_gap_m is not None and sheet_gap_m > 0:
        if sheet_gap_m < cap:
            cap_source = "min_measured_sheet_gap (dense layer detection)"
        cap = min(cap, sheet_gap_m)
    radii = sorted(set(round(min(f * med, cap), 4) for f in params.radii_factors))
    info = {
        "spacing_median_m": med,
        "spacing_p95_m": p95,
        "sheet_gap_cap_m": round(float(sheet_gap_m), 4) if sheet_gap_m else None,
        "bridge_cap_m": round(cap, 4),
        "bridge_cap_source": cap_source,
        "radii_m": radii,
        "capped": [f * med > cap for f in params.radii_factors],
    }
    return radii, info


def _local_spacing_at(xyz: np.ndarray, spacing: np.ndarray, q_idx: np.ndarray) -> np.ndarray:
    """Full-length per-point spacing (nearest measured anchor value)."""
    if len(spacing) == len(xyz):
        return spacing
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz[q_idx])
    _, idx = tree.query(xyz, k=1, workers=-1)
    return spacing[idx]


def _adaptive_ball_pivot_mesh(
    cloud: PointCloud,
    params: Phase2Params,
    *,
    sheet_gap_fallback_voxel: float | None = None,
) -> tuple[TriangleMesh, dict[str, Any]]:
    """Adaptive BPA: spacing-derived radii + sheet-gap cap + bridge rejection."""
    import open3d as o3d

    if cloud.normals is None:
        raise ValueError("adaptive BPA requires the fused cloud's oriented normals")
    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    nrm = np.asarray(cloud.normals, dtype=np.float64)

    # Optional size guard (never changes geometry, only bounds memory).
    if len(xyz) > params.max_input_points:
        keep = np.sort(np.random.default_rng(0).choice(
            len(xyz), params.max_input_points, replace=False))
        xyz, nrm = xyz[keep], nrm[keep]
        log.warning("phase2_input_subsampled", kept=len(keep))

    spacing, q_idx = spacing_field(xyz, k=params.spacing_k)
    sp_full = _local_spacing_at(xyz, spacing, q_idx)
    # Measure the cloud's own minimum real sheet separation (Part 6 evidence)
    # and hand it to the radius schedule as the bridge cap. Falls back to the
    # spacing-p95 cap when the cloud has no detected sheets.
    sheet_gap = None
    layer_info: dict[str, Any] = {}
    try:
        from app.services.dense_diagnostics import detect_layers

        lay = detect_layers(xyz, sheet_gap_fallback_voxel or 1.0, normals=nrm)
        if lay.get("status") == "measured" and lay.get("min_separation_m"):
            sheet_gap = float(lay["min_separation_m"])
            layer_info = {
                k: lay.get(k) for k in (
                    "layer_region_count", "layer_pair_count",
                    "median_layer_separation_m", "min_separation_m")
            }
    except Exception as lay_exc:  # diagnostics must never block meshing
        layer_info = {"status": f"unavailable: {lay_exc}"}
    radii, rad_info = _radii_schedule(spacing, params, sheet_gap_m=sheet_gap)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    if cloud.rgb is not None:
        rgb = np.asarray(cloud.rgb)
        if len(rgb) == len(xyz):
            pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    t0 = time.perf_counter()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    # Patch-seam weld at r1/2: same-surface patches from different radius
    # passes touch at shared points; r1/2 stays BELOW the measured sampling
    # noise (median spacing) so only duplicate/co-located samples weld —
    # real gaps (≥ 1.5 × fusion voxel by construction) cannot close.
    weld_eps = radii[0] / params.patch_weld_divisor
    mesh = mesh.merge_close_vertices(weld_eps)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    bpa_seconds = time.perf_counter() - t0

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("adaptive BPA produced no triangles")

    # --- bridge-triangle rejection (evidence-based, counted) --------------
    # All checks are per-FACE (a face is rejected when ANY of its evidence
    # tests fails); per-vertex arrays are gathered through ``faces``.
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    d_vert, nearest = tree.query(verts, k=1, workers=-1)
    sp_at_vert = sp_full[nearest]
    # A face is supported when all three of its vertices sit within
    # support_factor × local spacing of an actual dense point.
    support_ok = (d_vert <= params.support_factor * sp_at_vert)[faces].all(axis=1)

    tri = verts[faces]                                   # (F, 3, 3)
    e = np.stack([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ], axis=1)
    # Per-triangle budget: max of the three vertices' local spacing budgets.
    budget = params.max_edge_factor * sp_at_vert[faces].max(axis=1)[:, None]
    edge_ok = np.all(e <= budget, axis=1)
    # Normal coherence: a triangle whose winding fights the fused surface
    # normals of all three vertices is a flipped/bridged patch.
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn_n = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_u = np.divide(fn, fn_n, out=np.zeros_like(fn), where=fn_n > 1e-30)
    vn = nrm[faces].mean(axis=1)
    vn = vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30)
    coherence = np.abs((fn_u * vn).sum(axis=1))
    normal_ok = (coherence > 0.1) | (fn_n[:, 0] <= 1e-30)

    keep_f = support_ok & edge_ok & normal_ok
    rejected = {
        "unsupported_vertex": int((~support_ok).sum()),
        "oversized_edge": int((~edge_ok & support_ok).sum()),
        "normal_incoherent": int((~normal_ok & support_ok & edge_ok).sum()),
    }
    faces_kept = faces[keep_f]
    if len(faces_kept) == 0:
        raise ValueError("bridge rejection removed every triangle")
    used = np.unique(faces_kept)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    out_faces = remap[faces_kept]
    out = TriangleMesh(
        vertices=verts[used], faces=out_faces,
        meta={"method": "ball_pivot_adaptive"},
    )
    stats = {
        "radii": rad_info,
        "sheet_gap_evidence": layer_info,
        "patch_weld_eps_m": round(float(weld_eps), 4),
        "bpa_seconds": round(bpa_seconds, 1),
        "triangles_before_rejection": int(len(faces)),
        "rejected_triangles": rejected,
        "triangles_kept": int(len(faces_kept)),
        "vertices_before_rejection": int(len(verts)),
        "vertices_after_rejection": int(len(used)),
        "input_points": int(len(xyz)),
        "median_vertex_spacing_m": round(float(np.median(sp_at_vert[used])), 4),
        "labels": "MEASURED — all thresholds from the spacing field",
    }
    return out, stats


# ---------------------------------------------------------------------------
# Parts 9/10/14 — mesh evaluation + artifact contract
# ---------------------------------------------------------------------------

def evaluate_mesh(
    mesh: TriangleMesh,
    dense_xyz: np.ndarray,
    voxel: float,
    *,
    layer_scale: str = "auto",
    dense_sample: int = 100_000,
) -> dict[str, Any]:
    """Full metric set for one candidate mesh (mandate §8/§9/§10/§14).

    ``layer_scale='auto'`` measures layering at the mesh's OWN vertex scale
    (median edge length) — the Phase 1B detector's min-separation gate is
    1.5 × the scale argument, which is correct only when that scale matches
    the point set being measured. Measuring a 6 m-spaced mesh with a 2.3 m
    gate (fusion-voxel scale) counts same-surface vertices as sheets.
    """
    from scipy.spatial import cKDTree

    from app.services.dense_diagnostics import LAYER_MIN_SEP_FACTOR, detect_layers, extended_mesh_quality

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    out: dict[str, Any] = {"vertices": int(mesh.n), "faces": int(mesh.m)}
    if mesh.n == 0 or len(f) == 0:
        return {"status": "empty_mesh", **out}

    # Edge/area statistics.
    tri = v[f]
    e = np.stack([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ], axis=1)
    edges = e.ravel()
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out.update({
        "median_edge_length": round(float(np.median(edges)), 4),
        "p90_edge_length": round(float(np.percentile(edges, 90)), 4),
        "median_triangle_area": round(float(np.median(areas)), 4),
        "p90_triangle_area": round(float(np.percentile(areas, 90)), 4),
    })

    ext = extended_mesh_quality(v, f, voxel)
    out.update({
        "degenerate_faces": ext.get("degenerate_faces"),
        "non_manifold_edges": ext.get("non_manifold_edges"),
        "boundary_edges": ext.get("boundary_edges"),
        "normal_conflict_percent": ext.get("normal_conflict_percent"),
        "x_extent": ext.get("x_extent_m"),
        "y_extent": ext.get("y_extent_m"),
        "z_extent": ext.get("z_extent_m"),
    })

    # Components.
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(mesh.n, mesh.n)).tocsr()
    n_comp, labels = connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels)
    out["components"] = int(n_comp)
    out["largest_component_percent"] = round(float(100.0 * comp_sizes.max() / mesh.n), 2)

    # Dense → mesh (sampled dense points → nearest mesh vertex; vertex-based
    # approximation of point-to-surface, honest label).
    dense_q = dense_xyz if len(dense_xyz) <= dense_sample else dense_xyz[
        np.random.default_rng(0).choice(len(dense_xyz), dense_sample, replace=False)]
    d_dense_mesh = cKDTree(v).query(dense_q, k=1, workers=-1)[0]
    out.update({
        "dense_to_mesh_median": round(float(np.median(d_dense_mesh)), 4),
        "dense_to_mesh_p90": round(float(np.percentile(d_dense_mesh, 90)), 4),
        "dense_to_mesh_p95": round(float(np.percentile(d_dense_mesh, 95)), 4),
        "dense_unsupported_percent": round(
            float(100.0 * (d_dense_mesh > voxel).mean()), 2),
        "dense_support_threshold_m": round(float(voxel), 3),
    })

    # Mesh → dense support (every vertex within one fusion voxel of the cloud).
    d_mesh_dense = cKDTree(dense_xyz).query(v, k=1, workers=-1)[0]
    out["mesh_support_percent"] = round(float(100.0 * (d_mesh_dense <= voxel).mean()), 2)
    out["mesh_to_dense_median"] = round(float(np.median(d_mesh_dense)), 4)

    # Layering at the mesh's own scale.
    if layer_scale == "auto":
        scale = max(out["median_edge_length"], 1e-6) / LAYER_MIN_SEP_FACTOR
    else:
        scale = float(voxel)
    layers = detect_layers(v, scale)
    out["layered_percent"] = layers.get("layered_points_percent")
    out["layer_regions"] = layers.get("layer_region_count")
    out["layer_pair_count"] = layers.get("layer_pair_count")
    out["layer_min_separation_m"] = layers.get("min_separation_m")
    out["layer_measurement_scale"] = (
        f"min_sep = {LAYER_MIN_SEP_FACTOR} × {round(scale, 3)} m (median mesh edge / {LAYER_MIN_SEP_FACTOR})")
    out["labels"] = {
        "support": "MEASURED (threshold = fusion voxel)",
        "dense_to_mesh": "MEASURED — nearest-mesh-VERTEX approximation of point-to-surface",
        "layers": "MEASURED at the mesh's own vertex scale (see layer_measurement_scale)",
        "extents_topology": "MEASURED",
    }
    return out


def _benchmark_poisson(
    cloud: PointCloud,
    params: Phase2Params,
    dense_xyz: np.ndarray,
    voxel: float,
) -> tuple[TriangleMesh | None, dict[str, Any]]:
    """Screened Poisson — DIAGNOSTIC ONLY (watertight closure invents geometry).

    Returns ``(mesh | None, evaluation)``; the mesh is returned so callers can
    persist it, but Poisson output is never promotion-eligible.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(cloud.xyz))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals))
    t0 = time.perf_counter()
    try:
        mesh, _dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=params.poisson_depth)
    except Exception as exc:  # pragma: no cover
        return None, {"status": f"failed: {exc}", "method": "poisson"}
    mesh.remove_unreferenced_vertices()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if len(faces) == 0:
        return None, {"status": "no_triangles", "method": "poisson"}
    out = TriangleMesh(vertices=verts, faces=faces, meta={"method": "poisson"})
    ev = evaluate_mesh(out, dense_xyz, voxel)
    ev.update({
        "method": "poisson_diagnostic",
        "poisson_depth": params.poisson_depth,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "note": "watertight closure INVENTS geometry in unobserved regions — "
                "boundary_edges ≈ 0 is the closure signature, not quality",
    })
    return out, ev


