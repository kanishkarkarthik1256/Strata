"""Per-vertex reconstruction confidence overlay.

Each vertex's confidence combines (a) the dense-cloud per-point confidence
transferred by nearest point, (b) the semantic per-face confidence voted to
the vertex, and (c) the mission GPS quality (a global scalar from the
georeferencing report when present). The result is written as a vertex
colour overlay (red = low → green = high) plus statistics and weak-region
clusters for the dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.confidence_overlay")

WEAK_THRESHOLD = 0.4


def vertex_confidence(
    mesh: TriangleMesh,
    cloud_confidence: np.ndarray | None = None,
    cloud_xyz: np.ndarray | None = None,
    semantic_conf: np.ndarray | None = None,
    gps_quality: float | None = None,
    search_radius: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Combine confidence sources into per-vertex values in [0, 1]."""
    conf = np.ones(mesh.n, dtype=np.float64)
    sources = []
    if cloud_confidence is not None and cloud_xyz is not None:
        tree = cKDTree(cloud_xyz)
        r = search_radius or float(np.median(tree.query(cloud_xyz, k=2)[0][:, 1]) * 6)
        d, idx = tree.query(mesh.vertices, k=1)
        near = d <= r
        transferred = np.ones(mesh.n)
        transferred[near] = cloud_confidence[idx[near]]
        conf *= np.clip(transferred, 0, 1)
        sources.append("dense_cloud")
    if semantic_conf is not None:
        per_vert = np.full(mesh.n, 0.5)
        acc = np.zeros(mesh.n)
        np.add.at(acc, mesh.faces.ravel(), np.repeat(semantic_conf, 3))
        counts = np.bincount(mesh.faces.ravel(), minlength=mesh.n)
        nz = counts > 0
        per_vert[nz] = acc[nz] / counts[nz]
        conf *= np.clip(per_vert, 0.2, 1.0)
        sources.append("semantic")
    if gps_quality is not None:
        conf *= np.clip(0.6 + 0.4 * gps_quality / 100.0, 0.6, 1.0)
        sources.append("gps")
    conf = np.clip(conf, 0.0, 1.0)

    weak = _weak_regions(mesh, conf < WEAK_THRESHOLD, min_vertices=8)
    stats = {
        "mean_confidence": round(float(conf.mean()), 4),
        "median_confidence": round(float(np.median(conf)), 4),
        "p10": round(float(np.percentile(conf, 10)), 4),
        "low_share": round(float((conf < WEAK_THRESHOLD).mean()), 4),
        "sources": sources,
        "weak_regions": weak,
    }
    return conf, stats


def _weak_regions(mesh: TriangleMesh, low: np.ndarray, min_vertices: int) -> list[dict]:
    """Centroids of connected low-confidence vertex clusters."""
    if not low.any():
        return []
    edges = mesh.edge_topology()["edges"]
    keep = low[edges[:, 0]] & low[edges[:, 1]]
    if not keep.any():
        return []
    e = edges[keep]
    row = np.concatenate([e[:, 0], e[:, 1]])
    col = np.concatenate([e[:, 1], e[:, 0]])
    g = csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(mesh.n, mesh.n))
    n, labels = connected_components(g, directed=False)
    regions = []
    for c in range(n):
        verts = mesh.vertices[labels == c]
        if len(verts) < min_vertices:
            continue
        regions.append({
            "centroid": verts.mean(axis=0).tolist(),
            "vertices": int(len(verts)),
            "radius_m": round(float(np.max(np.linalg.norm(verts - verts.mean(axis=0), axis=1))), 4),
        })
    return regions


def confidence_color(conf: np.ndarray) -> np.ndarray:
    """red→green vertex colours for the overlay."""
    t = np.clip(conf, 0, 1)
    rgb = np.stack([np.clip(2 - 2 * t, 0, 1), np.clip(2 * t, 0, 1), np.zeros_like(t)], axis=1)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class ConfidenceOverlayStage(PipelineStage):
    name = "confidence_overlay"
    description = "Per-vertex confidence overlay + weak-region clusters"
    artifact_rel = "confidence/mesh_confidence.npy"
    dependencies = ("mesh_generation", "mesh_optimization", "mesh_repair", "semantic")

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run mesh_repair first")

    def execute(self) -> None:
        import numpy as np  # noqa: F401

        from app.services.pointcloud import read_ply

        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "repaired_mesh.ply")
        cloud = None
        dense_path = self.workspace / "dense" / "dense_model.ply"
        if dense_path.exists():
            cloud = read_ply(dense_path)
        sem_conf_path = self.workspace / "semantic" / "semantic_confidence.npy"
        sem_conf = None
        if sem_conf_path.exists():
            sem_conf = np.load(sem_conf_path)
        gps_quality = _gps_quality(self.workspace)
        conf, stats = vertex_confidence(
            mesh,
            cloud_confidence=cloud.confidence if cloud is not None else None,
            cloud_xyz=cloud.xyz if cloud is not None else None,
            semantic_conf=sem_conf,
            gps_quality=gps_quality,
        )
        conf_dir = self.workspace / "confidence"
        conf_dir.mkdir(parents=True, exist_ok=True)
        np.save(conf_dir / "mesh_confidence.npy", conf)
        overlay = mesh.copy()
        overlay.colors = confidence_color(conf)
        overlay.confidence = conf
        overlay_path = conf_dir / "mesh_confidence.ply"
        overlay.save_ply(overlay_path)
        (conf_dir / "confidence_report.json").write_text(json.dumps(stats, indent=2))
        self._count = mesh.n
        self._detail = stats
        self._outputs = [
            {"kind": "data", "name": "mesh_confidence", "path": str(conf_dir / "mesh_confidence.npy")},
            {"kind": "mesh", "name": "confidence_overlay", "path": str(overlay_path)},
        ]


def _gps_quality(workspace: Path) -> float | None:
    """GPS quality scalar from the georeferencing report (0-100) or None."""
    path = workspace / "georef" / "gps_report.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("gps_quality", {}).get("gps_score")
    except (OSError, ValueError):
        return None
