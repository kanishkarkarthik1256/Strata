"""Multi-mission change detection — voxel-occupancy diff of two missions.

Both missions must be aligned into a common frame to compare honestly. When
both workspaces carry an ENU-aligned dense model (``georef/dense_model_enu.ply``
— produced by the pipeline's georef stage), the clouds live in the same
ENU frame and are directly comparable. Without that common frame a naive
cloud diff would measure *misalignment*, not change, so the stage refuses
to run (clean ``StageNotApplicable``) instead of reporting bogus change.

Outputs (``intel/change_report.json``):

* occupancy voxelization of both missions at a shared resolution,
* added / removed clusters (new structures, removed structures) with bbox +
  volume proxies,
* a semantic change summary when per-face labels can be rasterised onto each
  cloud (label change counts over nearest neighbours),
* a difference point cloud (``change_delta.ply``) for visualization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register
from app.services.pointcloud import PointCloud, read_ply, save_ply

log = get_logger("drone_recon.services.change_detection")


def _load_enu_cloud(workspace: Path) -> PointCloud | None:
    cand = workspace / "georef" / "dense_model_enu.ply"
    if cand.exists():
        return read_ply(cand)
    # Fall back to the dense model itself only when an ENU alignment matrix
    # exists and equals the identity case is provable — which it is not in
    # general. So: no ENU artifact → not comparable.
    return None


def _voxel_occupy(xyz: np.ndarray, origin: np.ndarray, size: float, extent: int) -> set[tuple[int, int, int]]:
    idx = np.floor((xyz - origin) / size).astype(np.int64)
    idx = idx[(np.abs(idx) < extent).all(axis=1)]
    return set(map(tuple, idx.tolist()))


def diff_missions(workspace_a: Path, workspace_b: Path,
                  voxel_m: float = 0.5) -> dict:
    """Compare two georeferenced missions. *a* is the baseline, *b* current."""
    cloud_a = _load_enu_cloud(workspace_a)
    cloud_b = _load_enu_cloud(workspace_b)
    if cloud_a is None or cloud_b is None:
        raise StageNotApplicable(
            "both missions need georeferenced dense models (georef/dense_model_enu.ply) "
            "to be compared in a common frame — run the pipeline with GPS for each")

    a_xyz, b_xyz = cloud_a.xyz, cloud_b.xyz
    lo = np.minimum(a_xyz.min(axis=0), b_xyz.min(axis=0))
    hi = np.maximum(a_xyz.max(axis=0), b_xyz.max(axis=0))
    voxel = max(float(voxel_m), 0.2)
    extent = int(np.ceil(float(np.max(hi - lo)) / voxel)) + 2
    occ_a = _voxel_occupy(a_xyz, lo, voxel, extent)
    occ_b = _voxel_occupy(b_xyz, lo, voxel, extent)
    added = occ_b - occ_a   # present now, absent at baseline
    removed = occ_a - occ_b  # present at baseline, gone now

    def _cluster(voxels: set, label: str) -> list[dict]:
        """Connected component clustering of voxel centres (6-connectivity)."""
        if not voxels:
            return []
        from collections import deque

        cells = set(voxels)
        clusters = []
        while cells:
            start = cells.pop()
            queue = deque([start])
            comp = []
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if abs(dx) + abs(dy) + abs(dz) != 1:
                                continue
                            nxt = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                            if nxt in cells:
                                cells.remove(nxt)
                                queue.append(nxt)
            if len(comp) >= 8:  # ≥ 1 m³ at 0.5 m voxels
                arr = np.asarray(comp, dtype=float) * voxel + lo
                clusters.append({
                    "type": label,
                    "voxel_count": len(comp),
                    "volume_m3": round(len(comp) * voxel**3, 1),
                    "centroid": [round(float(v), 2) for v in arr.mean(axis=0)],
                    "bbox_min": [round(float(v), 2) for v in arr.min(axis=0)],
                    "bbox_max": [round(float(v), 2) for v in arr.max(axis=0)],
                })
        clusters.sort(key=lambda c: -c["voxel_count"])
        return clusters

    added_clusters = _cluster(added, "added")
    removed_clusters = _cluster(removed, "removed")

    # Difference cloud for visualization (the actual changed voxels).
    delta = np.vstack([
        np.asarray(list(added), dtype=float) * voxel + lo if added else np.zeros((0, 3)),
        np.asarray(list(removed), dtype=float) * voxel + lo if removed else np.zeros((0, 3)),
    ]) if (added or removed) else np.zeros((0, 3))
    delta_cloud = PointCloud(xyz=delta, confidence=np.ones(len(delta)))
    change_dir = workspace_b / "intel"
    change_dir.mkdir(parents=True, exist_ok=True)
    save_ply(change_dir / "change_delta.ply", delta_cloud)

    # Semantic change summary: majority label per occupied voxel, compared
    # between missions (requires semantic overlays in both workspaces).
    sem = _semantic_summary(workspace_a, workspace_b, lo, voxel, extent)

    total_added = sum(c["voxel_count"] for c in added_clusters)
    total_removed = sum(c["voxel_count"] for c in removed_clusters)
    max_dim = float(np.max(hi - lo))
    change_index = round(min(1.0, (total_added + total_removed) * voxel**3
                             / max(max_dim**2 * 1.0, 1.0)), 4)
    return {
        "method": "voxel_occupancy_diff",
        "voxel_size_m": voxel,
        "baseline_job": workspace_a.name,
        "current_job": workspace_b.name,
        "added_clusters": added_clusters,
        "removed_clusters": removed_clusters,
        "added_count": len(added_clusters),
        "removed_count": len(removed_clusters),
        "changed_voxels": {"added": len(added), "removed": len(removed)},
        "semantic_changes": sem,
        "change_index": change_index,
        "scene_extent_m": [round(float(v), 2) for v in (hi - lo)],
    }


def _labeled_voxel_map(workspace: Path, lo: np.ndarray, voxel: float,
                       extent: int) -> tuple[dict, list[str]] | None:
    """Majority semantic label per occupied voxel in the *common (ENU)* frame.

    The geo-aligned mesh (``georef/repaired_mesh_enu.ply``) shares vertex
    order with the semantic overlay mesh, so labels attach to the ENU
    vertices by index — giving both missions one comparable label raster.
    """
    from app.services.mesh import TriangleMesh

    enu_path = workspace / "georef" / "repaired_mesh_enu.ply"
    ov_path = workspace / "semantic" / "semantic_overlay.ply"
    if not enu_path.exists() or not ov_path.exists():
        return None
    try:
        enu = TriangleMesh.read_ply(enu_path)
        overlay = TriangleMesh.read_ply(ov_path)
    except (OSError, ValueError):
        return None
    if overlay.labels is None or enu.n == 0 or enu.n != overlay.n:
        return None
    idx = np.floor((enu.vertices - lo) / voxel).astype(np.int64)
    ok = (np.abs(idx) < extent).all(axis=1)
    keys = list(map(tuple, idx[ok].tolist()))
    labs = overlay.labels[ok]
    from collections import defaultdict

    votes: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for k, lab in zip(keys, labs.tolist()):
        votes[k][lab] += 1
    names = list(settings.semantic.classes)
    best = {k: max(cnt.items(), key=lambda kv: kv[1])[0] for k, cnt in votes.items()}
    return best, names


def _semantic_summary(wa: Path, wb: Path, lo: np.ndarray, voxel: float,
                      extent: int) -> dict:
    """Per-class added/removed/converted shares from per-mission label maps."""
    if voxel > 1.0:
        return {}  # coarse voxels would blur semantic change
    a = _labeled_voxel_map(wa, lo, voxel, extent)
    b = _labeled_voxel_map(wb, lo, voxel, extent)
    if not a or not b:
        return {}
    map_a, names = a
    map_b, _ = b
    both = set(map_a) & set(map_b)
    converted = sum(1 for k in both if map_a[k] != map_b[k])
    added_cells = set(map_b) - set(map_a)
    removed_cells = set(map_a) - set(map_b)

    def _shares(cells, lookup) -> dict[str, float]:
        counts: dict[int, int] = {}
        for k in cells:
            lab = lookup.get(k)
            if lab is not None:
                counts[lab] = counts.get(lab, 0) + 1
        total = sum(counts.values()) or 1
        return {names[i]: round(c / total, 3) for i, c in counts.items() if i < len(names)}

    return {
        "added_classes": _shares(added_cells, map_b),
        "removed_classes": _shares(removed_cells, map_a),
        "converted_cells": converted,
        "notes": "labels come from each mission's semantic overlay; converted = "
                  "cells occupied in both with different majority labels",
    }


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class ChangeDetectionStage(PipelineStage):
    name = "change_detection"
    description = "Voxel-occupancy change between this mission and a baseline"
    artifact_rel = "intel/change_report.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        cfg = self.workspace / "intel" / "change_config.json"
        if not cfg.exists():
            raise StageNotApplicable("no change_config.json with a baseline_job — "
                                     "run /api/intel/change to compare two missions")

    def execute(self) -> None:
        cfg = json.loads((self.workspace / "intel" / "change_config.json").read_text())
        baseline = Path(cfg["baseline_job"])
        if not baseline.is_absolute():
            baseline = self.workspace.parent / baseline
        report = diff_missions(baseline, self.workspace, voxel_m=cfg.get("voxel_m", 0.5))
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "change_report.json").write_text(json.dumps(report, indent=2))
        self._count = report["added_count"] + report["removed_count"]
        self._detail = {"added": report["added_count"], "removed": report["removed_count"],
                        "change_index": report["change_index"]}
        self._outputs = [{"kind": "data", "name": "change_report", "path": str(intel_dir / "change_report.json")},
                         {"kind": "pointcloud", "name": "change_delta", "path": str(intel_dir / "change_delta.ply")}]
        self.progress(1.0, {"added": report["added_count"], "removed": report["removed_count"]})
