"""Mesh quality audit — the ONE owner of every mesh-quality measurement.

Before this module the audit lived inline in ``dense_reconstruction.py`` and
was copied into ``scripts/rebuild_mesh_run.py``; the copy had already
diverged (it dropped ``layering`` and ``occlusion_consistency``), and a
rebuilt mesh left the run's stored reports contradicting each other. Both
paths now call :func:`audit_mesh_quality`, and :func:`refresh_run_mesh_fields`
is the single owner of the field map that says which report carries which
mesh number — so one rebuilt mesh can never leave a run's records
self-inconsistent.

Nothing here mutates geometry: every number is measured off the mesh and
cloud the caller passes in. The audit degrades honestly — when an input a
sub-measurement needs is absent, the reason is recorded in
``audit_notes`` instead of the key silently disappearing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.logging_config import get_logger

log = get_logger("drone_recon.services.mesh_quality")

#: Where the ``viewer_glb`` detail block lives inside the run's reports. One
#: map, one owner: every value is derived from the audit or the viewer-LOD
#: stats, never re-typed by hand. ``pipeline_report.json`` carries the block
#: TWICE (stage detail + its performance mirror), which is exactly how a
#: hand-rolled refresh left one copy stale.
_VIEWER_GLB_PATHS: dict[str, tuple[str, ...]] = {
    "pipeline_report.json": (
        "stages.dense.detail.viewer_glb",
        "performance.stages.dense.details.viewer_glb",
    ),
    "manifest.json": ("stages.dense.detail.viewer_glb",),
    "performance.json": ("stages.dense.details.viewer_glb",),
}


@dataclass
class MeshAudit:
    """Result of one mesh-quality audit.

    ``quality`` is the complete ``mesh_quality_report.json`` content;
    ``layer_payload`` is ``layer_source_classification.json`` and
    ``classification`` is the stage's ``layer_classification`` entry (both
    ``None`` when the inputs they need were not supplied).
    """

    quality: dict[str, Any]
    layer_payload: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def audit_mesh_quality(
    mesh,
    cloud_xyz: np.ndarray,
    voxel_m: float,
    *,
    method: str | None = None,
    views: list[Any] | None = None,
    dense_layers: dict[str, Any] | None = None,
) -> MeshAudit:
    """Measure one mesh against the cloud it came from.

    Thresholds are derived from the run's fusion voxel, not taste: a triangle
    edge much larger than the voxel spans empty space (a bridge); the
    dominant component should carry the scene; a mostly-down-facing surface
    means flipped normals (terrain faces up under an aerial survey).
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from app.services.mesh_generator import edge_connectivity

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    nv = int(mesh.n)
    n_faces = int(mesh.m)
    notes: list[str] = []

    tri = vertices[faces]
    edges = np.stack([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ], axis=1)
    max_edge = edges.max(axis=1)
    giant = int((max_edge > 10.0 * voxel_m).sum())
    edge_n, _edge_labels, edge_sizes = edge_connectivity(faces)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nv, nv)).tocsr()
    n_comp, labels = connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nz = cross[:, 2] / (np.linalg.norm(cross, axis=1) + 1e-12)

    quality: dict[str, Any] = {
        "vertices": nv,
        "faces": n_faces,
        "edge_median_m": round(float(np.median(max_edge)), 3),
        "edge_p95_m": round(float(np.percentile(max_edge, 95)), 3),
        "edge_max_m": round(float(max_edge.max()), 2),
        "giant_faces_gt_10voxels": giant,
        "giant_face_threshold_m": round(10.0 * voxel_m, 2),
        "components": int(n_comp),
        "largest_component_pct": round(float(100.0 * comp_sizes.max() / nv), 2),
        "components_gt_100_verts": int((comp_sizes > 100).sum()),
        # The vertex count above is NOT how the surface renders: triangles that
        # merely share a vertex still render as separate shards. On
        # Video_Mission_7721c8 the vertex metric read "largest 93.8%" while the
        # RENDERED connectivity was 128,566 patches with the largest holding
        # 35.9% of faces. Both are reported so fragmentation cannot be hidden.
        "edge_components": int(edge_n),
        "edge_largest_component_pct": round(
            float(100.0 * edge_sizes.max() / n_faces), 2),
        "faces_in_small_edge_patches_pct": round(
            float(100.0 * edge_sizes[edge_sizes <= 50].sum() / n_faces), 2),
        "up_facing_pct": round(float(100.0 * (nz > 0).mean()), 2),
        "down_facing_pct": round(float(100.0 * (nz < 0).mean()), 2),
        "note": (
            "production method: ball_pivot (observed-surface BPA; up/down "
            "splits reflect genuinely observed opposing surfaces, not a "
            "watertight shell)"
            if method == "ball_pivot" else f"production method: {method}"
        ),
    }

    # Boundary/hole + component-geometry evidence (holes, component bounding
    # boxes, curtain-span audit). The edge tally packs each edge into one
    # int64 and counts with np.unique: the Python dict this replaced inserted
    # ~7M (min, max) tuples for a 2.3M-face mesh and cost ~24 s; the numbers
    # are identical (an edge with count 1 is a boundary edge). A failure here
    # degrades to the base metrics already measured (it never loses the file).
    try:
        _f = faces.astype(np.int64)
        _lo = np.minimum(_f[:, [0, 1, 2]], _f[:, [1, 2, 0]])
        _hi = np.maximum(_f[:, [0, 1, 2]], _f[:, [1, 2, 0]])
        _uniq, _counts = np.unique((_lo * (nv + 1) + _hi).ravel(), return_counts=True)
        del _uniq
        z_span = tri[:, :, 2].max(axis=1) - tri[:, :, 2].min(axis=1)
        quality["boundary_edges"] = int((_counts == 1).sum())
        quality["unique_edges"] = int(_counts.size)
        quality["faces_zspan_gt_10m"] = int((z_span > 10.0).sum())
        comp_ids = np.argsort(comp_sizes)[::-1][: min(5, len(comp_sizes))]
        quality["component_bboxes"] = [
            {
                "vertices": int(comp_sizes[cid]),
                "bbox_min": [round(float(x), 1) for x in vertices[labels == cid].min(axis=0)],
                "bbox_max": [round(float(x), 1) for x in vertices[labels == cid].max(axis=0)],
            }
            for cid in comp_ids
        ]
    except Exception as exc:  # noqa: BLE001 - degrade to the measured base
        notes.append(f"boundary/hole metrics failed: {exc}")
        log.warning("mesh_quality_extras_failed", error=str(exc))

    # Phase 1B: mesh-side layers, dense->mesh support, component classes and
    # the cross-view corroboration of the FINAL dense cloud. These need the
    # dense layer measurement and the depth views; without them the reason is
    # recorded rather than the keys vanishing.
    layer_payload: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    if dense_layers is None:
        notes.append("layering/dense_support/occlusion_consistency not measured: "
                     "no dense layer measurement supplied")
    else:
        try:
            from app.services.dense_diagnostics import (
                classify_layer_source,
                classify_mesh_components,
                cross_view_consistency,
                detect_layers,
                extended_mesh_quality,
                mesh_support,
            )

            mesh_layers = detect_layers(vertices, voxel_m)
            quality["layering"] = mesh_layers
            sup_dist, _ = cKDTree(cloud_xyz).query(vertices, k=1, workers=-1)
            # Both calls below reuse work this audit already did: the vertex
            # component labels (one connected_components pass, not two) and the
            # nearest-dense distances (one KD-tree query, not two). The numbers
            # are unchanged — same graph, same tree, same query.
            support = mesh_support(
                vertices, cloud_xyz, threshold_m=voxel_m, faces=faces,
                support_dist=(sup_dist if len(cloud_xyz) <= 4_000_000 else None))
            quality["dense_support"] = support
            comp_report = classify_mesh_components(
                vertices, faces, sup_dist, voxel_m,
                labels=labels, n_comp=int(n_comp))
            quality["component_classification"] = {
                "component_count": comp_report.get("component_count"),
                "class_counts": comp_report.get("class_counts"),
                "rule": comp_report.get("rule"),
            }
            occl: dict[str, Any] | None = None
            if views:
                occl = cross_view_consistency(cloud_xyz, views, voxel_m)
                quality["occlusion_consistency"] = occl
            else:
                notes.append("occlusion_consistency not measured: no depth views "
                             "supplied (read the run's depth stage artifacts)")
            classification = classify_layer_source(
                dense_layers, mesh_layers, None, cloud_xyz, vertices, voxel_m)
            layer_payload = {
                **classification,
                "dense_layers": dense_layers,
                "mesh_layers": mesh_layers,
                "mesh_support": support,
                "mesh_components": comp_report,
                "occlusion_consistency": occl,
            }
            quality["extended"] = extended_mesh_quality(vertices, faces, voxel_m)
        except Exception as exc:  # noqa: BLE001 - layer diagnostics are optional
            notes.append(f"phase-1B layer diagnostics failed: {exc}")
            log.warning("layer_diagnostics_failed", error=str(exc))
    if notes:
        quality["audit_notes"] = notes
    return MeshAudit(quality=quality, layer_payload=layer_payload,
                     classification=classification, notes=notes)


def viewer_glb_detail(lod_stats: dict[str, Any]) -> dict[str, Any]:
    """The ``viewer_glb`` detail block the run's reports carry.

    Built from the LOD stage stats so the GLB facts are derived from the
    artifact that was actually written. ``glb_vertices`` is the count the
    writer put IN THE FILE (``glb_vertices``), not the decimated mesh's shared
    vertex count (``vertices``): a textured LOD splits vertices once per UV
    seam, so on Video_Mission_9ab1aa the file holds 3,299,997 vertices where the
    pre-split mesh held 788,074 and the viewer correctly reported the former.
    """
    return {
        "textured": bool(lod_stats.get("textured")),
        "texture_source": lod_stats.get("texture_source"),
        "texture_fallback_reason": lod_stats.get("texture_fallback_reason"),
        "views_used": (lod_stats.get("texture") or {}).get("views_used"),
        "glb_faces": lod_stats.get("faces"),
        "glb_vertices": lod_stats.get("glb_vertices", lod_stats.get("vertices")),
    }


def _dotted(container: dict[str, Any], path: str) -> Any:
    node: Any = container
    for part in path.split("."):
        node = node[part]
    return node


def refresh_run_mesh_fields(
    workspace: Path,
    audit: MeshAudit,
    viewer_lod: dict[str, Any] | None = None,
    mesh_rel: str = "mesh/mesh.ply",
) -> dict[str, list[str]]:
    """Rewrite every mesh-bearing field of a run's reports from ONE audit.

    The pipeline writes these values inline as it runs; this is the offline
    counterpart (used by ``scripts/rebuild_mesh_run.py``) so a rebuilt mesh
    updates all of them together. Returns ``{file: [field, ...]}``.
    """
    updated: dict[str, list[str]] = {}

    # The canonical file first: the audit IS its content.
    (workspace / "mesh_quality_report.json").write_text(
        json.dumps(audit.quality, indent=2))
    updated["mesh_quality_report.json"] = sorted(audit.quality.keys())

    dense_path = workspace / "dense_report.json"
    if dense_path.is_file():
        report = json.loads(dense_path.read_text())
        stages = report.setdefault("stages", {})
        stages["mesh"] = audit.quality
        touched = ["stages.mesh"]
        if audit.classification is not None:
            stages["layer_classification"] = audit.classification
            touched.append("stages.layer_classification")
        if viewer_lod:
            stages["viewer_lod"] = viewer_lod
            touched.append("stages.viewer_lod")
        for export in report.get("exports", []):
            path = str(export.get("path", ""))
            if path.endswith(("mesh/mesh.ply", "mesh/mesh_full.ply", "mesh/mesh.ply")):
                export["vertices"] = audit.quality["vertices"]
                export["faces"] = audit.quality["faces"]
                touched.append(f"exports[{path.rsplit('/', 1)[-1]}]")
        dense_path.write_text(json.dumps(report, indent=2))
        updated["dense_report.json"] = touched

    if audit.layer_payload is not None:
        (workspace / "layer_source_classification.json").write_text(
            json.dumps(audit.layer_payload, indent=2))
        updated["layer_source_classification.json"] = [
            "mesh_layers", "mesh_support", "mesh_components",
            "occlusion_consistency", "classification"]

    if viewer_lod:
        detail = viewer_glb_detail(viewer_lod)
        for name, dotted_paths in _VIEWER_GLB_PATHS.items():
            path = workspace / name
            if not path.is_file():
                continue
            doc = json.loads(path.read_text())
            touched: list[str] = []
            for dotted in dotted_paths:
                try:
                    node = _dotted(doc, dotted)
                except (KeyError, TypeError):
                    continue
                node.update(detail)
                touched.append(f"{dotted}.*")
            if touched:
                path.write_text(json.dumps(doc, indent=2))
                updated.setdefault(name, []).extend(touched)
    return updated
