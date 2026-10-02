"""Triangle mesh representation, file IO, and topology helpers.

``TriangleMesh`` is the canonical Phase 7 mesh object shared by the mesh
generator / optimizer / repairer / texturer / semantic stages. Vertices and
faces are parallel arrays; optional per-vertex ``colors`` / ``confidence`` /
``labels`` ride along so geometry never gets detached from its dense-cloud
provenance.

The PLY writer/reader are symmetric (binary little-endian, vertices then
faces); OBJ export is provided for downstream tools. Topology helpers
(edges, boundaries, connected components, watertightness) are vectorised
over numpy arrays.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from app.logging_config import get_logger
from app.services.pointcloud import PointCloud

log = get_logger("drone_recon.services.mesh")


@dataclass
class TriangleMesh:
    """A triangle mesh with optional per-vertex attributes."""

    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int — vertex indices
    colors: np.ndarray | None = None  # (N, 3) uint8
    confidence: np.ndarray | None = None  # (N,) float64 in [0, 1]
    labels: np.ndarray | None = None  # (N,) int — per-vertex semantic label
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3), got {self.faces.shape}")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("face indices out of vertex range")
        n = len(self.vertices)
        for name, arr in (("colors", self.colors), ("confidence", self.confidence)):
            if arr is not None:
                arr = np.asarray(arr)
                setattr(self, name, arr)
                if len(arr) != n:
                    raise ValueError(f"{name} length != vertex count {n}")
        if self.labels is not None:
            self.labels = np.asarray(self.labels, dtype=np.int64)
            if len(self.labels) != n:
                raise ValueError("labels length != vertex count")

    # ----------------------------------------------------------------- props

    @property
    def n(self) -> int:
        return len(self.vertices)

    @property
    def m(self) -> int:
        return len(self.faces)

    def copy(self) -> TriangleMesh:
        return TriangleMesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            colors=self.colors.copy() if self.colors is not None else None,
            confidence=self.confidence.copy() if self.confidence is not None else None,
            labels=self.labels.copy() if self.labels is not None else None,
            meta=dict(self.meta),
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def center(self) -> np.ndarray:
        lo, hi = self.bounds()
        return (lo + hi) / 2.0

    def extents(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    # ------------------------------------------------------------ conversion

    @classmethod
    def from_pointcloud(cls, cloud: PointCloud) -> TriangleMesh:
        """Vertex data directly from a dense PointCloud (no faces yet)."""
        return cls(
            vertices=np.asarray(cloud.xyz, dtype=np.float64),
            colors=cloud.rgb.copy() if cloud.rgb is not None else None,
            confidence=cloud.confidence.copy() if cloud.confidence is not None else None,
            meta=dict(cloud.meta),
        )

    def to_pointcloud(self, normals: bool = True) -> PointCloud:
        return PointCloud(
            xyz=self.vertices,
            rgb=self.colors,
            confidence=self.confidence,
            normals=self.vertex_normals() if normals else None,
        )

    # ------------------------------------------------------------ geometry

    def face_areas(self) -> np.ndarray:
        a = self.vertices[self.faces[:, 0]]
        b = self.vertices[self.faces[:, 1]]
        c = self.vertices[self.faces[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def face_normals(self, normalized: bool = True) -> np.ndarray:
        """Per-face unit normals (zero rows for degenerate faces)."""
        a = self.vertices[self.faces[:, 0]]
        b = self.vertices[self.faces[:, 1]]
        c = self.vertices[self.faces[:, 2]]
        n = np.cross(b - a, c - a)
        if not normalized:
            return n
        norms = np.linalg.norm(n, axis=1)
        out = np.zeros_like(n)
        ok = norms > 1e-30
        out[ok] = n[ok] / norms[ok, None]
        return out

    def face_centroids(self) -> np.ndarray:
        return self.vertices[self.faces].mean(axis=1)

    def vertex_normals(self) -> np.ndarray:
        """Area-weighted vertex normals (zero rows for isolated vertices)."""
        fn = self.face_normals(normalized=False)
        acc = np.zeros((self.n, 3))
        np.add.at(acc, self.faces[:, 0], fn)
        np.add.at(acc, self.faces[:, 1], fn)
        np.add.at(acc, self.faces[:, 2], fn)
        norms = np.linalg.norm(acc, axis=1)
        out = np.zeros_like(acc)
        ok = norms > 1e-30
        out[ok] = acc[ok] / norms[ok, None]
        return out

    # ------------------------------------------------------------ topology

    def edge_topology(self) -> dict:
        """Unique undirected edges with per-edge incident face ids.

        Returns ``{"edges": (E, 2), "counts": (E,), "face_ids": flat list,
        "indptr": (E+1,)}`` — face ids for edge *e* are
        ``face_ids[indptr[e]:indptr[e+1]]``.
        """
        f = self.faces
        # Face-major edge rows: [face0 e0, face0 e1, face0 e2, face1 e0, ...].
        tri = np.stack([
            np.column_stack([f[:, 0], f[:, 1]]),
            np.column_stack([f[:, 1], f[:, 2]]),
            np.column_stack([f[:, 2], f[:, 0]]),
        ], axis=1).reshape(-1, 2)
        rows = np.sort(tri, axis=1)
        edges, inverse = np.unique(rows, axis=0, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(edges))
        face_ids = np.repeat(np.arange(len(f)), 3)
        order = np.argsort(inverse, kind="stable")
        sorted_inv = inverse[order]
        face_ids = face_ids[order]
        # CSR index pointers over sorted face ids.
        indptr = np.zeros(len(edges) + 1, dtype=np.int64)
        np.add.at(indptr, sorted_inv + 1, 1)
        indptr = np.cumsum(indptr)
        return {"edges": edges, "counts": counts, "face_ids": face_ids, "indptr": indptr}

    def boundary_edges(self) -> np.ndarray:
        """Edges shared by exactly one face (the mesh boundary)."""
        topo = self.edge_topology()
        return topo["edges"][topo["counts"] == 1]

    def non_manifold_edge_count(self) -> int:
        return int((self.edge_topology()["counts"] > 2).sum())

    def is_watertight(self) -> bool:
        return len(self.boundary_edges()) == 0

    def euler_characteristic(self) -> int:
        topo = self.edge_topology()
        return int(self.n - len(topo["edges"]) + self.m)

    def boundary_loops(self) -> list[np.ndarray]:
        """Vertex-index cycles around each boundary hole (manifold input)."""
        topo = self.edge_topology()
        be = topo["edges"][topo["counts"] == 1]
        if len(be) == 0:
            return []
        adj: dict[int, list[int]] = {}
        for u, v in be:
            adj.setdefault(int(u), []).append(int(v))
            adj.setdefault(int(v), []).append(int(u))
        loops: list[np.ndarray] = []
        visited: set[int] = set()
        for start in list(adj):
            if start in visited or len(adj[start]) != 2:
                continue
            loop = [start]
            prev, cur = -1, start
            visited.add(start)
            while True:
                nxts = [v for v in adj[cur] if v != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt == start:
                    break
                if nxt in visited:
                    break
                visited.add(nxt)
                loop.append(nxt)
                prev, cur = cur, nxt
                if len(loop) > len(be):
                    break
            if len(loop) >= 3:
                loops.append(np.array(loop, dtype=np.int64))
        return loops

    def face_components(self) -> tuple[int, np.ndarray]:
        """Connected components of faces sharing edges; returns (k, labels (M,))."""
        if self.m == 0:
            return 0, np.zeros(0, dtype=np.int64)
        topo = self.edge_topology()
        indptr = topo["indptr"]
        face_ids = topo["face_ids"]
        row = np.repeat(np.arange(len(topo["edges"])), np.diff(indptr))
        col = face_ids
        graph = csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)),
                           shape=(len(topo["edges"]), self.m))
        adj = (graph.T @ graph).astype(np.int8)
        adj.setdiag(0)
        n, labels = connected_components(adj, directed=False)
        return int(n), labels

    def submesh(self, face_mask: np.ndarray) -> TriangleMesh:
        """New mesh containing only the given faces (vertices remapped)."""
        keep_faces = self.faces[face_mask]
        vert_ids = np.unique(keep_faces)
        remap = np.full(self.n, -1, dtype=np.int64)
        remap[vert_ids] = np.arange(len(vert_ids))
        faces = remap[keep_faces]
        def _sel(arr):
            return arr[vert_ids] if arr is not None else None
        return TriangleMesh(
            vertices=self.vertices[vert_ids],
            faces=faces,
            colors=_sel(self.colors),
            confidence=_sel(self.confidence),
            labels=_sel(self.labels),
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ IO

    def save_ply(self, path: Path, normals: bool = False) -> None:
        """Binary little-endian PLY with vertices (colors/confidence/labels
        when present) followed by triangular faces."""
        n, m = self.n, self.m
        rgb = self.colors if self.colors is not None else np.full((n, 3), 150, dtype=np.uint8)
        conf = self.confidence if self.confidence is not None else np.ones(n, dtype=np.float64)
        labels = self.labels if self.labels is not None else np.zeros(n, dtype=np.int64)
        vn = self.vertex_normals() if normals else None

        lines = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
                 "property float x", "property float y", "property float z",
                 "property uchar red", "property uchar green", "property uchar blue",
                 "property float confidence", "property uchar label"]
        if vn is not None:
            lines += ["property float nx", "property float ny", "property float nz"]
        lines += [f"element face {m}", "property list uchar int vertex_indices", "end_header"]
        path.write_text("\n".join(lines) + "\n")

        with open(path, "ab") as f:
            vfmt = "<3f3BfB" + ("3f" if vn is not None else "")
            for i in range(n):
                f.write(struct.pack(vfmt, *self.vertices[i], *rgb[i], float(conf[i]),
                                    int(np.clip(labels[i], 0, 255)),
                                    *([*vn[i]] if vn is not None else [])))
            for face in self.faces:
                f.write(struct.pack("<B3i", 3, *face))

    @classmethod
    def read_ply(cls, path: Path) -> TriangleMesh:
        """Read a binary PLY mesh (our writer format or a compatible subset)."""
        raw = path.read_bytes()
        header_end = raw.index(b"end_header\n") + len(b"end_header\n")
        header = raw[:header_end].decode("ascii")
        body = raw[header_end:]

        vertex_props: list[tuple[str, str]] = []
        n_vertices = 0
        n_faces = 0
        for line in header.splitlines():
            parts = line.split()
            if line.startswith("property "):
                if parts[1] == "list":
                    pass  # face properties (list int vertex_indices)
                elif n_faces == 0:
                    vertex_props.append((parts[-1], parts[-2]))
            elif line.startswith("element vertex "):
                n_vertices = int(parts[-1])
            elif line.startswith("element face "):
                n_faces = int(parts[-1])

        sizes = {"float": 4, "uchar": 1, "int": 4}
        offsets: dict[str, int] = {}
        offset = 0
        for name, ptype in vertex_props:
            offsets[name] = offset
            offset += sizes[ptype]
        stride = offset

        def row(i: int) -> np.ndarray:
            return np.frombuffer(body, dtype=np.uint8, count=stride,
                                 offset=i * stride)

        verts = np.empty((n_vertices, 3), dtype=np.float64)
        rgb = np.empty((n_vertices, 3), dtype=np.uint8)
        conf = np.ones(n_vertices, dtype=np.float64)
        labels = np.zeros(n_vertices, dtype=np.int64)
        vn = None
        if "nx" in offsets:
            vn = np.empty((n_vertices, 3), dtype=np.float64)

        def fval(row_arr: np.ndarray, name: str) -> float:
            o = offsets[name]
            return row_arr[o:o + 4].view(np.float32)[0]

        def uval(row_arr: np.ndarray, name: str) -> int:
            return int(row_arr[offsets[name]])

        def ival(row_arr: np.ndarray, name: str) -> int:
            o = offsets[name]
            return int(row_arr[o:o + 4].view(np.int32)[0])

        for i in range(n_vertices):
            r = row(i)
            verts[i] = [fval(r, "x"), fval(r, "y"), fval(r, "z")]
            if "red" in offsets:
                rgb[i] = [uval(r, "red"), uval(r, "green"), uval(r, "blue")]
            if "confidence" in offsets:
                conf[i] = fval(r, "confidence")
            if "label" in offsets:
                labels[i] = uval(r, "label")
            if vn is not None:
                vn[i] = [fval(r, "nx"), fval(r, "ny"), fval(r, "nz")]

        faces = np.empty((n_faces, 3), dtype=np.int64)
        fb = body[stride * n_vertices:]
        for i in range(n_faces):
            count = fb[i * 13]
            if count != 3:
                raise ValueError(f"unsupported PLY face with {count} vertices")
            faces[i] = np.frombuffer(fb, dtype=np.int32, count=3,
                                     offset=i * 13 + 1)
        mesh = cls(vertices=verts, faces=faces,
                   colors=rgb if "red" in offsets else None,
                   confidence=conf if "confidence" in offsets else None,
                   labels=labels if "label" in offsets else None)
        if vn is not None:
            mesh.meta["vertex_normals"] = vn
        return mesh

    def save_obj(self, path: Path, normals: bool = True) -> None:
        """Wavefront OBJ (vertices + optional normals + faces)."""
        with open(path, "w") as f:
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            vn = self.vertex_normals() if normals and self.m else None
            if vn is not None:
                for nrm in vn:
                    f.write(f"vn {nrm[0]:.6f} {nrm[1]:.6f} {nrm[2]:.6f}\n")
            for face in self.faces:
                if vn is not None:
                    f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
