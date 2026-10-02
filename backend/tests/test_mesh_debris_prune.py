"""Debris-patch prune — the mesh must render as one surface, not a shard cloud.

Measured defect this pins (Video_Mission_7721c8, 2.39M faces, 1.01 m median
edge): the finished mesh was 128,566 patches under RENDERED (shared-edge)
connectivity, the largest holding 35.9% of faces, with 32% of all faces in
<=50-face patches sitting a median 9.2 edge-lengths away from the main
surface. The previous island prune judged VERTEX connectivity, which cannot
see patches that merely share a vertex, so it reported a healthy-looking
"3,317 components, 93.8% largest" while the viewer showed scattered shards.

Contracts pinned here:
1. A small patch far from the main surface is dropped (it is debris).
2. A small patch touching the main surface is KEPT (it is real geometry).
3. The main component is never a candidate, however small its share.
4. Pure deletion: no surviving vertex moves, so the mesh still sits on the
   measured cloud.
5. The quality report counts rendered (edge) connectivity, not vertex.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("open3d")

from app.services.mesh_generator import (  # noqa: E402
    TriangleMesh,
    _prune_isolated_debris,
    edge_connectivity,
)


def _grid(x0: float, y0: float, z: float, n: int, step: float, n_tri: int) -> tuple[np.ndarray, np.ndarray]:
    """A flat triangulated patch of ``n_tri`` triangles on a z plane."""
    verts = []
    faces = []
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append([x0 + i * step, y0 + j * step, z])
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return np.array(verts, dtype=np.float64), np.array(faces[:n_tri], dtype=np.int64)


def _mesh(parts: list[tuple[np.ndarray, np.ndarray]]) -> TriangleMesh:
    vs = []
    fs = []
    off = 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return TriangleMesh(vertices=np.vstack(vs), faces=np.vstack(fs))


def test_small_patch_far_from_the_surface_is_dropped() -> None:
    step = 1.0
    main_v, main_f = _grid(0, 0, 0, 20, step, 800)  # 800-triangle main surface
    # A 40-face patch 30 m away — 30 median edge lengths of debris.
    far_v, far_f = _grid(300, 0, 30, 5, step, 40)
    mesh = _mesh([(main_v, main_f), (far_v, far_f)])
    assert edge_connectivity(mesh.faces)[0] == 2

    out, meta = _prune_isolated_debris(mesh, distance_factor=3.0, max_faces=200)

    assert meta["patches_dropped"] == 1
    assert meta["faces_dropped"] == 40
    assert edge_connectivity(out.faces)[0] == 1
    assert out.m == 800


def test_small_patch_touching_the_surface_is_kept() -> None:
    step = 1.0
    main_v, main_f = _grid(0, 0, 0, 20, step, 800)
    # 40 faces only 1 m from the main surface: real observed surface, not debris.
    near_v, near_f = _grid(0, 0, 1.0, 5, step, 40)
    mesh = _mesh([(main_v, main_f), (near_v, near_f)])

    out, meta = _prune_isolated_debris(mesh, distance_factor=3.0, max_faces=200)

    assert meta["patches_dropped"] == 0
    assert out.m == 840
    assert meta["near_surface_patch_faces_kept"] == 40


def test_main_component_is_never_dropped() -> None:
    # Main surface is small (12 faces) and sits far from a huge second patch.
    small_v, small_f = _grid(0, 0, 0, 2, 1.0, 8)
    big_v, big_f = _grid(500, 0, 0, 20, 1.0, 800)
    mesh = _mesh([(small_v, small_f), (big_v, big_f)])
    n_before, _labels, sizes = edge_connectivity(mesh.faces)
    assert n_before == 2
    main_is_big = int(np.argmax(sizes)) == 1

    out, meta = _prune_isolated_debris(mesh, distance_factor=3.0, max_faces=200)

    # The big patch is the main component, so it survives; the 8-face main
    # candidate is never considered.
    assert main_is_big
    assert out.m == 800
    assert meta["patches_dropped"] == 1


def test_deletion_moves_nothing() -> None:
    step = 1.0
    main_v, main_f = _grid(0, 0, 0, 10, step, 200)
    far_v, far_f = _grid(200, 0, 0, 3, step, 12)
    mesh = _mesh([(main_v, main_f), (far_v, far_f)])
    before = set(map(tuple, np.round(np.asarray(mesh.vertices), 6)))

    out, _meta = _prune_isolated_debris(mesh, distance_factor=3.0, max_faces=200)

    after = set(map(tuple, np.round(np.asarray(out.vertices), 6)))
    assert after <= before  # survivors are the original vertices, unmoved
    assert len(after) < len(before)


def test_quality_metric_counts_rendered_connectivity_not_vertices() -> None:
    """A patch sharing only VERTICES with the surface must count as separate.

    This is the exact blindness that let the report read healthy while the
    viewer showed 128k shards: vertex adjacency joins the two patches, edge
    adjacency does not.
    """
    main_v, main_f = _grid(0, 0, 0, 10, 1.0, 200)
    n_main = len(main_v)
    # One standalone triangle that reuses main vertex 0 -> joined by VERTEX
    # only, exactly the case the old metric could not see.
    extra_v = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0]])
    mesh = TriangleMesh(
        vertices=np.vstack([main_v, extra_v]),
        faces=np.vstack([main_f, np.array([[0, n_main, n_main + 1]], dtype=np.int64)]),
    )

    n_edge, labels, sizes = edge_connectivity(mesh.faces)

    assert n_edge == 2  # vertex-only join does NOT stitch the surface
    assert sizes.max() == 200
    # Whereas vertex adjacency would call this one component:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(mesh.faces)
    nv = len(mesh.vertices)
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    n_vtx, _ = connected_components(
        sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nv, nv)).tocsr(),
        directed=False,
    )
    assert n_vtx == 1
