"""Coverage prediction over the reconstructed scene.

Derives *measured* evidence maps from the twin artifacts and labels them as
such (``source: measured``): reconstruction confidence per grid cell comes
straight from the confidence overlay, occupancy from the mesh itself, and
class shares from the semantic overlay. ``blind`` cells are *measured*
low-confidence cells — they are where a follow-up pass adds value; predicted
gains of a candidate path are computed separately in the path optimizer and
are labeled simulation estimates.

Rasters are coarse JSON grids (``planning.coverage_grid_cells`` per axis) so
they can be rendered directly as heatmaps by a client; per-cell geometry is
included for geospatial overlays when the mesh is ENU-aligned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.coverage_predictor")


def build_coverage(mesh: TriangleMesh, labels: np.ndarray | None,
                   conf: np.ndarray | None, cells: int | None = None) -> dict:
    """Coverage/blind/confidence rasters + class shares from measured data."""
    n = int(cells or settings.planning.coverage_grid_cells)
    xy = mesh.vertices[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    ix = np.clip(((xy[:, 0] - lo[0]) / span[0] * (n - 1)).astype(np.int64), 0, n - 1)
    iy = np.clip(((xy[:, 1] - lo[1]) / span[1] * (n - 1)).astype(np.int64), 0, n - 1)
    flat = iy * n + ix
    occ = np.bincount(flat, minlength=n * n).reshape(n, n) > 0
    conf_mean = None
    if conf is not None and len(conf) == mesh.n:
        acc = np.zeros(n * n)
        np.add.at(acc, flat, conf)
        cnt = np.bincount(flat, minlength=n * n).astype(float)
        z = cnt > 0
        grid_conf = np.full(n * n, np.nan)
        grid_conf[z] = acc[z] / cnt[z]
        conf_mean = grid_conf.reshape(n, n)

    blind = np.zeros((n, n), dtype=bool)
    if conf_mean is not None:
        blind = occ & np.isnan(conf_mean)  # observed but unscored
        ok = ~np.isnan(conf_mean)
        blind |= ok & (conf_mean < settings.planning.blind_conf_threshold)
    blind = blind & occ

    shares: dict[str, float] = {}
    if labels is not None and len(labels) == mesh.m:
        class_names = list(settings.semantic.classes)
        for c, name in enumerate(class_names):
            cnt = int((labels == c).sum())
            if cnt:
                shares[name] = round(cnt / mesh.m, 4)
    n_cells_occ = int(occ.sum())
    return {
        "grid": {"cells": n, "cell_size_m": round(float(span[0] / n), 3),
                 "extent_lo": [round(float(v), 2) for v in lo],
                 "extent_hi": [round(float(v), 2) for v in hi]},
        "observed": [[bool(v) for v in row] for row in occ],
        "confidence": [[None if np.isnan(v) else round(float(v), 4) for v in row]
                       for row in conf_mean] if conf_mean is not None else None,
        "blind": [[bool(v) for v in row] for row in blind],
        "blind_cells": int(blind.sum()),
        "observed_cells": n_cells_occ,
        "blind_share": round(float(blind.sum() / max(n_cells_occ, 1)), 4),
        "class_shares": shares,
        "weak_regions": _weak_from_conf(mesh, conf) if conf is not None else [],
        "source": "measured",
        "method": "mesh_occupancy_confidence_overlay",
    }


def _weak_from_conf(mesh: TriangleMesh, conf: np.ndarray) -> list[dict]:
    """Connected low-confidence clusters (same contract as confidence overlay)."""
    if len(conf) != mesh.n or not (conf < settings.planning.blind_conf_threshold).any():
        return []
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    low = conf < settings.planning.blind_conf_threshold
    edges = mesh.edge_topology()["edges"]
    keep = low[edges[:, 0]] & low[edges[:, 1]]
    if not keep.any():
        return []
    e = edges[keep]
    row = np.concatenate([e[:, 0], e[:, 1]])
    col = np.concatenate([e[:, 1], e[:, 0]])
    g = csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(mesh.n, mesh.n))
    n_comp, lab = connected_components(g, directed=False)
    out = []
    for c in range(n_comp):
        verts = mesh.vertices[lab == c]
        if len(verts) < 8:
            continue
        out.append({"centroid": [round(float(v), 2) for v in verts.mean(axis=0)],
                    "radius_m": round(float(np.max(np.linalg.norm(
                        verts - verts.mean(axis=0), axis=1))), 2),
                    "vertices": int(len(verts))})
    return out


def _load(workspace: Path) -> tuple[TriangleMesh, np.ndarray | None, np.ndarray | None]:
    mesh = TriangleMesh.read_ply(workspace / "mesh" / "repaired_mesh.ply")
    labels = None
    lp = workspace / "semantic" / "semantic_labels.npy"
    if lp.exists():
        labels = np.load(lp).astype(np.int64)
    conf = None
    cp = workspace / "confidence" / "mesh_confidence.npy"
    if cp.exists():
        conf = np.load(cp).astype(np.float64)
    return mesh, labels, conf


@register
class CoveragePredictionStage(PipelineStage):
    name = "coverage_prediction"
    description = "Measured coverage / blind-spot / confidence rasters from the twin"
    artifact_rel = "intel/coverage_prediction.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run the digital-twin chain first")

    def execute(self) -> None:
        mesh, labels, conf = _load(self.workspace)
        report = build_coverage(mesh, labels, conf)
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "coverage_prediction.json").write_text(json.dumps(report, indent=2))
        self._count = report["blind_cells"]
        self._detail = {"blind_share": report["blind_share"],
                        "blind_cells": report["blind_cells"],
                        "observed_cells": report["observed_cells"]}
        self._outputs = [{"kind": "data", "name": "coverage_prediction",
                          "path": str(intel_dir / "coverage_prediction.json")}]
        self.progress(1.0, {"blind_share": report["blind_share"]})
