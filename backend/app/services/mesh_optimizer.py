"""Mesh optimization — triangle decimation with quality presets.

Uses Open3D's quadric-error decimation when the optional ``open3d`` package
is installed; otherwise falls back to a real iterative shortest-edge collapse
that preserves manifold topology (an edge is only collapsed when the local
result keeps every edge shared by at most two faces). Presets express the
*target fraction of triangles to keep*: ``ultra`` 0.70, ``high`` 0.50,
``medium`` 0.30, ``low`` 0.12.

The fallback is deliberately conservative (cap on iterations, boundary
edges collapse first so outer silhouettes survive longest) and reports how
far it got when a preset target cannot be reached without breaking topology.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.mesh_optimizer")

PRESET_FRACTIONS = {"ultra": 0.70, "high": 0.50, "medium": 0.30, "low": 0.12}


@dataclass
class DecimateParams:
    preset: str | None = None
    target_fraction: float | None = None
    target_faces: int | None = None
    max_iterations: int = 20_000


def _edge_counts(faces: list[tuple[int, int, int]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            e = (x, y) if x < y else (y, x)
            counts[e] = counts.get(e, 0) + 1
    return counts


def _face_component_count(faces: list[tuple[int, int, int]]) -> int:
    """Number of connected components of faces sharing edges (vectorised)."""
    if not faces:
        return 0
    arr = np.asarray(faces, dtype=np.int64)
    tri = np.stack([
        np.column_stack([arr[:, 0], arr[:, 1]]),
        np.column_stack([arr[:, 1], arr[:, 2]]),
        np.column_stack([arr[:, 2], arr[:, 0]]),
    ], axis=1).reshape(-1, 2)
    pairs = np.sort(tri, axis=1)
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1, 3)
    n_e = int(inverse.max()) + 1
    m = len(arr)
    row = inverse.ravel()
    col = np.repeat(np.arange(m), 3)
    graph = csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(n_e, m))
    adj = (graph.T @ graph).astype(np.int8)
    adj.setdiag(0)
    n, _ = connected_components(adj, directed=False)
    return int(n)


def _collapse_valid(u: int, v: int, faces: list[tuple[int, int, int]],
                    base_components: int) -> bool:
    """True if collapsing vertex u into v keeps the mesh manifold *and*
    does not disconnect the surface (component count preserved)."""
    scratch: list[tuple[int, int, int]] = []
    for t in faces:
        if u in t and v in t:
            continue  # face spanning the collapsed edge dies
        if u in t:
            nt = tuple(v if x == u else x for x in t)
            if len(set(nt)) == 3:
                scratch.append(nt)
        else:
            scratch.append(t)
    if not scratch:
        return False
    if _face_component_count(scratch) > base_components:
        return False  # would tear the surface apart
    counts = _edge_counts(scratch)
    # Faces not containing u or v are untouched, so validating the 1-ring of
    # the collapsed vertex is sufficient for manifoldness.
    return all(c <= 2 for c in counts.values())


def _collapse_edge(u: int, v: int, faces: list[tuple[int, int, int]]) -> None:
    """In place: remove vertex u, redirecting its faces to v; faces spanning
    the collapsed edge are deleted. v stays isolated-free by construction."""
    for i, t in enumerate(faces):
        if u in t and v in t:
            faces[i] = (-1, -1, -1)
        elif u in t:
            nt = tuple(v if x == u else x for x in t)
            faces[i] = nt if len(set(nt)) == 3 else (-1, -1, -1)
    for i in range(len(faces) - 1, -1, -1):
        if faces[i][0] < 0:
            faces.pop(i)


def _fallback_decimate(
    mesh: TriangleMesh,
    target: int,
    max_iterations: int,
) -> tuple[TriangleMesh, dict]:
    verts = mesh.vertices
    faces = [tuple(map(int, t)) for t in mesh.faces]
    iterations = 0
    removed = 0
    base_components = _face_component_count(faces)
    while len(faces) > target and iterations < max_iterations:
        iterations += 1
        counts = _edge_counts(faces)
        scored: list[tuple[float, int, int]] = []
        for (a, b), c in counts.items():
            if c not in (1, 2):
                continue
            length = float(np.linalg.norm(verts[a] - verts[b]))
            scored.append((length * (1.0 if c == 1 else 1.5), a, b))
        if not scored:
            break
        collapsed = False
        for _score, u, v in sorted(scored, key=lambda s: s[0]):
            if _collapse_valid(u, v, faces, base_components):
                _collapse_edge(u, v, faces)
                removed += 1
                collapsed = True
                break
        if not collapsed:
            break  # no further valid collapse — topology would break

    # Strip isolated vertices and remap faces.
    used = sorted({x for t in faces for x in t})
    remap = {old: new for new, old in enumerate(used)}
    new_faces = np.array([[remap[x] for x in t] for t in faces], dtype=np.int64)

    def _sel(arr):
        return arr[np.asarray(used, dtype=np.int64)] if arr is not None else None

    out = TriangleMesh(
        vertices=verts[np.asarray(used, dtype=np.int64)],
        faces=new_faces,
        colors=_sel(mesh.colors),
        confidence=_sel(mesh.confidence),
        labels=_sel(mesh.labels),
        meta=dict(mesh.meta),
    )
    stats = {
        "method": "edge_collapse",
        "before_faces": mesh.m,
        "after_faces": out.m,
        "removed_faces": removed,
        "iterations": iterations,
        "reached_target": out.m <= target,
    }
    return out, stats


def _open3d_decimate(mesh: TriangleMesh, target: int) -> tuple[TriangleMesh, dict]:
    """Guarded quadric-edge decimation (production path when open3d present)."""
    import open3d as o3d  # pragma: no cover - environment dependent

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    simplified = m.simplify_quadric_decimation(target_number_of_triangles=int(target))
    # simplify_quadric_decimation keeps ALL original vertices (only the face
    # list shrinks); drop the now-unreferenced ones or the exported mesh (and
    # the viewer GLB built from it) carries megabytes of dead vertices.
    simplified.remove_unreferenced_vertices()
    simplified.remove_degenerate_triangles()
    verts = np.asarray(simplified.vertices, dtype=np.float64)
    faces = np.asarray(simplified.triangles, dtype=np.int64)
    # Nearest-original colors ride along.
    from scipy.spatial import cKDTree

    tree = cKDTree(mesh.vertices)
    _, idx = tree.query(verts)
    colors = mesh.colors[idx] if mesh.colors is not None else None
    confidence = mesh.confidence[idx] if mesh.confidence is not None else None
    out = TriangleMesh(vertices=verts, faces=faces, colors=colors, confidence=confidence,
                       meta=dict(mesh.meta))
    return out, {"method": "open3d_qem", "before_faces": mesh.m, "after_faces": out.m,
                 "removed_faces": mesh.m - out.m, "reached_target": out.m <= target}


def decimate(mesh: TriangleMesh, params: DecimateParams | None = None) -> tuple[TriangleMesh, dict]:
    """Reduce *mesh* toward the preset / target triangle budget."""
    params = params or DecimateParams(preset=settings.mesh.decimate_preset)
    if params.target_fraction is not None:
        target = max(4, int(round(params.target_fraction * mesh.m)))
    elif params.target_faces is not None:
        target = max(4, int(params.target_faces))
    elif params.preset is not None:
        if params.preset not in PRESET_FRACTIONS:
            raise ValueError(f"unknown preset '{params.preset}' — choose from {sorted(PRESET_FRACTIONS)}")
        target = max(4, int(round(PRESET_FRACTIONS[params.preset] * mesh.m)))
    else:
        raise ValueError("decimate needs preset, target_fraction or target_faces")
    if mesh.m <= target:
        return mesh.copy(), {"method": "none", "before_faces": mesh.m, "after_faces": mesh.m,
                             "removed_faces": 0, "reached_target": True}
    started = time.perf_counter()
    try:
        import open3d  # noqa: F401

        out, stats = _open3d_decimate(mesh, target)
    except ImportError:
        out, stats = _fallback_decimate(mesh, target, params.max_iterations)
    stats["took_ms"] = round((time.perf_counter() - started) * 1000, 2)
    log.info("mesh_decimated", before=stats["before_faces"], after=stats["after_faces"],
             method=stats["method"])
    return out, stats


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class MeshOptimizationStage(PipelineStage):
    name = "mesh_optimization"
    description = "Triangle decimation at the configured quality preset"
    artifact_rel = "mesh/optimized_mesh.ply"
    dependencies = ("mesh_generation",)

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "base_mesh.ply").exists():
            raise StageNotApplicable("no base mesh — run mesh_generation first")

    def execute(self) -> None:
        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "base_mesh.ply")
        optimized, stats = decimate(mesh)
        path = self.artifact_path()
        assert path is not None
        optimized.save_ply(path, normals=False)
        self._count = optimized.m
        self._detail = stats
        self._outputs = [{"kind": "mesh", "name": "optimized_mesh", "path": str(path),
                          "faces": optimized.m}]
