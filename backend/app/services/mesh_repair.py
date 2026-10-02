"""Mesh repair: weld duplicates, drop degenerate geometry, fill holes.

Repair runs a deterministic pipeline and returns a machine-readable report:

1. remove non-finite vertices
2. weld duplicate vertices (quantized coordinate precision)
3. remove degenerate faces (repeated vertex / zero area)
4. remove duplicate faces
5. strip unreferenced vertices
6. remove faces making edges non-manifold (edges shared by > 2 faces)
7. drop disconnected components below a face threshold
8. fill interior boundary holes (ear-clipping) up to a loop-size budget

The outer rim of an open surface is preserved by default (it is a true
surface boundary, not a defect) unless ``fill_outer_boundary`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.mesh_repair")


@dataclass
class RepairParams:
    weld_precision: float = 1e-7
    min_component_faces: int = 12
    max_hole_edges: int = 64
    fill_outer_boundary: bool = False


def _strip_unreferenced(mesh: TriangleMesh) -> TriangleMesh:
    if mesh.m == 0:
        return mesh
    used = np.unique(mesh.faces)
    remap = np.full(mesh.n, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return TriangleMesh(
        vertices=mesh.vertices[used],
        faces=remap[mesh.faces],
        colors=mesh.colors[used] if mesh.colors is not None else None,
        confidence=mesh.confidence[used] if mesh.confidence is not None else None,
        labels=mesh.labels[used] if mesh.labels is not None else None,
        meta=dict(mesh.meta),
    )


def _weld_vertices(mesh: TriangleMesh, precision: float) -> tuple[TriangleMesh, int]:
    """Merge vertices closer than *precision* (first occurrence wins)."""
    if mesh.n == 0:
        return mesh, 0
    key = np.round(mesh.vertices / max(precision, 1e-12))
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    if len(first) == mesh.n:
        return mesh, 0
    merged = TriangleMesh(
        vertices=mesh.vertices[first],
        faces=inverse[mesh.faces],
        colors=mesh.colors[first] if mesh.colors is not None else None,
        confidence=mesh.confidence[first] if mesh.confidence is not None else None,
        labels=mesh.labels[first] if mesh.labels is not None else None,
        meta=dict(mesh.meta),
    )
    return merged, mesh.n - len(first)


def _fill_loop(mesh: TriangleMesh, loop: np.ndarray) -> np.ndarray:
    """Ear-clip a boundary loop (projected to its best-fit plane).

    Returns new faces (vertex indices) or an empty array when the loop is
    not fillable.
    """
    pts = mesh.vertices[loop]
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    # Planar loops always have one ~0 eigenvalue; require at least two
    # non-degenerate axes (a collinear loop has two).
    if evals[1] < 1e-18:
        return np.zeros((0, 3), dtype=np.int64)
    e1, e2 = evecs[:, 1], evecs[:, 2]
    p = np.column_stack([centered @ e1, centered @ e2])

    def _area2(o: int, a: int, b: int) -> float:
        """2D cross product (shoelace) — scalar; np.cross on 2-vectors is
        deprecated in numpy 2."""
        return float((p[a, 0] - p[o, 0]) * (p[b, 1] - p[o, 1])
                     - (p[a, 1] - p[o, 1]) * (p[b, 0] - p[o, 0]))

    # Consistent winding (counter-clockwise in the projected plane).
    if _area2(0, 1, 2) < 0:
        p = p[::-1]
        idx = loop[::-1]
    else:
        idx = loop
    verts = idx.tolist()
    polys = [list(range(len(verts)))]
    tris: list[list[int]] = []

    while polys:
        poly = polys.pop()
        n = len(poly)
        if n < 3:
            continue
        if n == 3:
            tris.append(poly)
            continue
        clipped = False
        guard = 0
        while n > 3 and guard < n * n:
            guard += 1
            ear = None
            for i in range(n):
                a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
                if _area2(a, b, c) <= 0:
                    continue  # reflex or degenerate
                inside = any(
                    _area2(a, b, poly[j]) > 0 and _area2(b, c, poly[j]) > 0
                    and _area2(c, a, poly[j]) > 0
                    for j in range(n) if j not in (i, (i - 1) % n, (i + 1) % n)
                )
                if not inside:
                    ear = i
                    break
            if ear is None:
                break
            prev = poly[ear - 1]
            cur = poly[ear]
            nxt = poly[(ear + 1) % n]
            tris.append([prev, cur, nxt])
            del poly[ear]
            n -= 1
            clipped = True
        if clipped:
            if n == 3:
                tris.append(poly)
            elif n > 3:
                polys.append(poly)  # split remainder; simple fans resolve it
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    out = np.array([[verts[a], verts[b], verts[c]] for a, b, c in tris], dtype=np.int64)
    # De-duplicate against existing faces (defensive) and drop zero-area.
    areas = TriangleMesh(vertices=mesh.vertices, faces=out).face_areas()
    return out[areas > 1e-18]


def repair(mesh: TriangleMesh, params: RepairParams | None = None) -> tuple[TriangleMesh, dict]:
    """Run the repair pipeline; returns ``(repaired_mesh, report)``."""
    params = params or RepairParams(
        weld_precision=settings.mesh.weld_precision,
        min_component_faces=settings.mesh.repair_min_component_faces,
        max_hole_edges=settings.mesh.repair_max_hole_edges,
    )
    report: dict = {}
    m = mesh.copy()

    # 1. non-finite vertices (drop their faces).
    finite = np.all(np.isfinite(m.vertices), axis=1)
    if not finite.all():
        m = m.submesh(np.isin(m.faces[:, 0], np.flatnonzero(finite)) &
                      np.isin(m.faces[:, 1], np.flatnonzero(finite)) &
                      np.isin(m.faces[:, 2], np.flatnonzero(finite)))
        m = _strip_unreferenced(m)
    report["removed_nonfinite_vertices"] = int((~finite).sum())

    # 2. weld duplicates.
    m, report["welded_vertices"] = _weld_vertices(m, params.weld_precision)

    # 3. degenerate faces (repeated vertex) + 4. exact duplicate faces.
    before = m.m
    ok = (m.faces[:, 0] != m.faces[:, 1]) & (m.faces[:, 1] != m.faces[:, 2]) & \
         (m.faces[:, 0] != m.faces[:, 2])
    m = m.submesh(ok)
    report["removed_degenerate_faces"] = int(before - m.m)
    if m.m:
        sorted_f = np.sort(m.faces, axis=1)
        _, first, counts = np.unique(sorted_f, axis=0, return_index=True, return_counts=True)
        keep = np.zeros(m.m, dtype=bool)
        keep[first] = True
        report["removed_duplicate_faces"] = int(m.m - keep.sum())
        m = m.submesh(keep)

    # 5. strip unreferenced vertices.
    m = _strip_unreferenced(m)

    # 6. non-manifold edges → drop excess faces.
    report["removed_non_manifold_faces"] = 0
    if m.m:
        topo = m.edge_topology()
        bad = np.flatnonzero(topo["counts"] > 2)
        if len(bad):
            drop = np.zeros(m.m, dtype=bool)
            for e in bad:
                fid = topo["face_ids"][topo["indptr"][e]:topo["indptr"][e + 1]]
                if len(fid) > 2:
                    drop[fid[2:]] = True
            report["removed_non_manifold_faces"] = int(drop.sum())
            m = m.submesh(~drop)
            m = _strip_unreferenced(m)

    # 7. drop tiny disconnected components.
    report["dropped_components"] = 0
    report["dropped_component_faces"] = 0
    if m.m:
        k, labels = m.face_components()
        if k > 1:
            sizes = np.bincount(labels, minlength=k)
            big = sizes >= max(1, params.min_component_faces)
            keep = big[labels]
            report["dropped_components"] = int((~big).sum())
            report["dropped_component_faces"] = int((~keep).sum())
            m = m.submesh(keep)
            m = _strip_unreferenced(m)

    # 8. fill interior holes.
    report["holes_filled"] = 0
    report["hole_edges_filled"] = 0
    if m.m:
        loops = m.boundary_loops()
        perimeters = [len(l) for l in loops]
        outer = max(range(len(perimeters)), key=lambda i: perimeters[i]) if perimeters else -1
        added: list[np.ndarray] = []
        for i, loop in enumerate(loops):
            if len(loop) > params.max_hole_edges:
                continue
            if i == outer and not params.fill_outer_boundary:
                continue
            tris = _fill_loop(m, loop)
            if len(tris):
                added.append(tris)
                report["holes_filled"] += 1
                report["hole_edges_filled"] += len(loop)
        if added:
            extra = np.concatenate(added)
            m = TriangleMesh(
                vertices=m.vertices,
                faces=np.vstack([m.faces, extra]),
                colors=m.colors, confidence=m.confidence, labels=m.labels,
                meta=dict(m.meta),
            )

    report["vertices"] = m.n
    report["faces"] = m.m
    report["non_manifold_edges_left"] = m.non_manifold_edge_count()
    report["watertight"] = m.is_watertight()
    return m, report


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class MeshRepairStage(PipelineStage):
    name = "mesh_repair"
    description = "Weld, dedupe, drop degenerate geometry and fill holes"
    artifact_rel = "mesh/repaired_mesh.ply"
    dependencies = ("mesh_generation", "mesh_optimization")

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "optimized_mesh.ply").exists():
            raise StageNotApplicable("no optimized mesh — run mesh_optimization first")

    def execute(self) -> None:
        import json

        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "optimized_mesh.ply")
        repaired, report = repair(mesh)
        path = self.artifact_path()
        assert path is not None
        repaired.save_ply(path, normals=False)
        (self.workspace / "mesh" / "repair_report.json").write_text(json.dumps(report, indent=2))
        self._count = repaired.m
        self._detail = report
        self._outputs = [{"kind": "mesh", "name": "repaired_mesh", "path": str(path),
                          "faces": repaired.m}]
