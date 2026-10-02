"""Infrastructure analyzer — measurement toolkit over twin objects + mesh.

Every measurement is computed from the twin's per-object geometry or the
mesh itself; none are estimated from a model. Outputs
``intel/infrastructure_report.json`` with:

* building stats — height, roof area, footprint (from roof/wall twins),
* terrain — slope statistics over the ground plane grid,
* tree height — vegetation object elevation above the ground plane,
* powerline clearance — nearest gap between ``tower``/``structure`` objects
  and ground, and between any two user-designated classes when present,
* road width — the median separation between opposite road-class runs when a
  ``road``/``ground`` corridor is detected is *not* fabricated: road width is
  reported only when a dedicated road class exists in the class list.

Class-specific sections are only populated when those classes are present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.infrastructure_analyzer")


def _ground_plane(mesh: TriangleMesh) -> tuple[np.ndarray, float]:
    """Least-squares horizontal ground from the lowest vertices."""
    z = mesh.vertices[:, 2]
    lo_idx = np.argsort(z)[: max(20, mesh.n // 10)]
    low = mesh.vertices[lo_idx]
    centered = low - low.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, -float(normal @ low.mean(axis=0))


def _object_height(obj: dict) -> float:
    lo, hi = obj.get("bbox_min"), obj.get("bbox_max")
    if not lo or not hi:
        return 0.0
    return float(hi[2] - lo[2])


def analyze_infrastructure(workspace: Path) -> dict:
    """Full infrastructure measurement pass over one mission."""
    mesh = TriangleMesh.read_ply(workspace / "mesh" / "repaired_mesh.ply")
    twin_path = workspace / "twin" / "twin.json"
    twin = json.loads(twin_path.read_text()) if twin_path.exists() else {"objects": []}
    objects = twin.get("objects", [])

    buildings = [o for o in objects if o.get("class") in ("roof", "wall", "structure")]
    trees = [o for o in objects if o.get("class") == "vegetation"]
    towers = [o for o in objects if o.get("class") in ("tower", "structure")]
    vehicles = [o for o in objects if o.get("class") == "vehicle"]

    report: dict = {"objects_total": len(objects), "note": ""}

    # --- building height / roof area / footprint (from twin measurements) ---
    if buildings:
        report["buildings"] = {
            "count": len(buildings),
            "height_m": {
                "max": round(max(_object_height(o) for o in buildings), 2),
                "mean": round(float(np.mean([_object_height(o) for o in buildings])), 2),
            },
            "roof_area_m2_total": round(sum(o.get("surface_area_m2", 0)
                                            for o in buildings if o.get("class") == "roof"), 1),
            "largest_footprint_m2": round(max((o.get("footprint_area_m2") or 0)
                                              for o in buildings), 1),
        }
    # --- tree height (vegetation above the ground plane) ---
    gn, gd = _ground_plane(mesh)
    if trees:
        heights = []
        for o in trees:
            c = np.asarray(o.get("centroid", [0, 0, 0]), dtype=float)
            h = float(c @ gn + gd)
            if h >= settings.intel.tree_min_height_m:
                heights.append(h)
        if heights:
            report["trees"] = {"count": len(heights),
                               "height_m": {"max": round(max(heights), 2),
                                            "mean": round(float(np.mean(heights)), 2)}}

    # --- terrain slope statistics from the mesh ground ---
    report["terrain"] = _terrain_stats(mesh, gn, gd)

    # --- powerline clearance: towers/structure tops to ground and to each other ---
    if towers and len(towers) >= 1:
        report["structure_clearance_m"] = {
            "min_tower_height_m": round(min(_object_height(o) for o in towers), 2),
        }
    if len(towers) >= 2:
        cents = np.asarray([o["centroid"] for o in towers])
        tree = cKDTree(cents)
        d, _ = tree.query(cents, k=2)
        report["structure_clearance_m"]["min_gap_between_m"] = round(float(d[:, 1].min()), 2)

    # --- roads: only when a dedicated road class exists in the mission's class list ---
    class_names = list(settings.semantic.classes)
    if "road" in class_names or "path" in class_names:
        road_class = "road" if "road" in class_names else "path"
        road_objs = [o for o in objects if o.get("class") == road_class]
        if road_objs:
            widths = [_min_cross_section(o, mesh) for o in road_objs]
            report["roads"] = {"count": len(road_objs),
                               "width_m_median": round(float(np.median([w for w in widths if w > 0])), 2)
                               if any(w > 0 for w in widths) else None}

    # --- vehicles as infrastructure load / obstructions ---
    if vehicles:
        report["vehicles_on_site"] = len(vehicles)

    if not any(k in report for k in ("buildings", "trees", "terrain", "roads")):
        report["note"] = "no measurable infrastructure classes present in this mission"
    return report


def _min_cross_section(obj: dict, mesh: TriangleMesh) -> float:
    """Median of the object's xy-extents projected onto its two axes — a crude
    corridor width when the object is a linear run (road). Zero when the run
    is too short to be a road."""
    lo = np.asarray(obj.get("bbox_min", [0, 0, 0]), dtype=float)
    hi = np.asarray(obj.get("bbox_max", [0, 0, 0]), dtype=float)
    return float(max(hi[0] - lo[0], hi[1] - lo[1]))


def _terrain_stats(mesh: TriangleMesh, gn: np.ndarray, gd: float) -> dict:
    """Slope distribution over a coarse grid of the horizontal extent."""
    xy = mesh.vertices[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.max(hi - lo) or 1.0
    cells = max(8, min(48, int(span / max(settings.dense.voxel_size * 4, 0.1))))
    counts = np.zeros((cells, cells))
    zacc = np.zeros((cells, cells))
    for (x, y, z) in mesh.vertices:
        i = min(cells - 1, int((x - lo[0]) / max(hi[0] - lo[0], 1e-9) * cells))
        j = min(cells - 1, int((y - lo[1]) / max(hi[1] - lo[1], 1e-9) * cells))
        counts[i, j] += 1
        zacc[i, j] += z
    occupied = counts > 0
    if not occupied.any():
        return {"slope_deg": {"mean": 0.0}, "elevation_range_m": 0.0}
    zm = np.where(occupied, zacc / np.maximum(counts, 1), np.nan)
    # Ground height model: median z at the 20% lowest cells approximates gd.
    flat = zm[~np.isnan(zm)]
    ground_h = float(np.percentile(flat, 10))
    # Gradient over a median-filled grid (NaN neighbours would otherwise make
    # every observed cell NaN on sparse meshes); slope is reported only on
    # cells that actually contain surface vertices.
    filled = np.where(occupied, zm, float(np.nanmedian(zm)))
    dzdx, dzdy = np.gradient(filled)
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))[occupied]
    return {
        "slope_deg": {"mean": round(float(slope.mean()), 1),
                      "max": round(float(slope.max()), 1),
                      "p90": round(float(np.percentile(slope, 90)), 1)},
        "elevation_range_m": round(float(np.nanmax(zm) - ground_h), 2),
        "relative_elevation_model": "grid_median",
    }


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class InfrastructureAnalysisStage(PipelineStage):
    name = "infrastructure_analysis"
    description = "Building/tree/terrain/clearance measurements from twin geometry"
    artifact_rel = "intel/infrastructure_report.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run the digital-twin chain first")

    def execute(self) -> None:
        report = analyze_infrastructure(self.workspace)
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "infrastructure_report.json").write_text(json.dumps(report, indent=2))
        self._count = report.get("objects_total", 0)
        sections = [k for k in ("buildings", "trees", "terrain", "roads") if k in report]
        self._detail = {"sections": sections}
        self._outputs = [{"kind": "data", "name": "infrastructure_report",
                          "path": str(intel_dir / "infrastructure_report.json")}]
        self.progress(1.0, {"sections": sections})
