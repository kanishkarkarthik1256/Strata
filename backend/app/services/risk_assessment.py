"""Risk assessment — explainable grid analysis over geometry + damage.

Risk is computed from measurable scene evidence only (``intel/risk_report.json``):

* a ground grid over the mesh extent with per-cell slope (gradient of the
  cell-mean elevation), occupancy and flood/debris exposure buffers,
* ``risk_zones`` — connected clusters of cells that are flooded, debris-
  exposed or steeper than ``risk_slope_deg``, each with its dominant cause,
* ``flood_risk`` / ``landslide_risk`` — damage findings cross-referenced
  with structure exposure (distance from roof/wall objects to flood cells),
* ``accessibility`` — emergency-vehicle proxy: cells traversable at
  ``access_slope_deg`` reachable from the grid boundary (flood cells block),
  per-structure reachability, and blocked-route causes,
* an ``overall_risk`` level (Critical/High/Medium/Low) with the reasoning
  strings that produced it — never a bare label.

Unobserved cells contribute nothing: slope is only defined where the mesh
has vertices, and flood/debris buffers only exist around actual findings.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import label as nd_label
from scipy.spatial import cKDTree

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.risk_assessment")

_LEVEL_NUM = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
_LEVELS = ("Low", "Medium", "High", "Critical")


def _num_to_level(n: int) -> str:
    return _LEVELS[min(max(int(n) - 1, 0), 3)]


def _slope_grid(mesh: TriangleMesh, cells: int) -> dict:
    """Cell-mean elevation grid + per-cell slope over the mesh xy extent.

    Resolution is capped by the surface sampling density: a grid finer than
    the vertex spacing would scatter occupancy into isolated cells and break
    the connectivity analysis, so ``cells`` is never finer than roughly one
    cell per nearest-neighbour spacing.
    """
    xy = mesh.vertices[:, :2]
    z = mesh.vertices[:, 2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    # nearest-neighbour spacing from a bounded vertex sample
    sample = mesh.vertices[np.linspace(0, mesh.n - 1, min(mesh.n, 2048)).astype(int)]
    if len(sample) > 4:
        from scipy.spatial import cKDTree

        tree = cKDTree(sample[:, :2])
        d, _ = tree.query(sample[:, :2], k=2)
        spacing = float(np.median(d[:, 1])) if d.shape[1] > 1 else 1.0
    else:
        spacing = float(np.sqrt(np.mean(span**2)))
    extent = float(np.max(span))
    cap = max(8, int(extent / (max(spacing, 1e-6) * 1.1)))
    nx = ny = max(8, min(int(cells), cap))
    ix = np.clip(((xy[:, 0] - lo[0]) / span[0] * (nx - 1)).astype(np.int64), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - lo[1]) / span[1] * (ny - 1)).astype(np.int64), 0, ny - 1)
    flat = iy * nx + ix
    cnt = np.bincount(flat, minlength=nx * ny).reshape(ny, nx).astype(np.float64)
    zacc = np.bincount(flat, weights=z, minlength=nx * ny).reshape(ny, nx)
    occupied = cnt > 0
    zm = np.full((ny, nx), np.nan)
    zm[occupied] = zacc[occupied] / cnt[occupied]
    # Smooth-fill NaNs so np.gradient stays finite; slope is only *used* on
    # occupied cells, so the fill only affects the field near their edges.
    filled = np.where(occupied, zm, float(np.nanmedian(zm)) if occupied.any() else 0.0)
    sx, sy = span[0] / nx, span[1] / ny
    dz = np.gradient(filled, sy, sx)  # per-metre gradient (rows=y, cols=x)
    slope = np.degrees(np.arctan(np.hypot(dz[0], dz[1])))
    return {"occupied": occupied, "slope": slope, "zm": zm, "sx": sx, "sy": sy,
            "lo": lo, "nx": nx, "ny": ny}


def _cell_centers(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) world coordinates of each cell centre (y-indexed rows first)."""
    ny, nx = grid["ny"], grid["nx"]
    xs = grid["lo"][0] + (np.arange(nx) + 0.5) * grid["sx"]
    ys = grid["lo"][1] + (np.arange(ny) + 0.5) * grid["sy"]
    X, Y = np.meshgrid(xs, ys)  # X[i, j], Y[i, j] → (x, y) of cell (i, j)
    return X, Y


def _exposure_mask(centroid: np.ndarray, area_m2: float, buffer_m: float,
                   X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cells within *buffer* + an area-equivalent radius of a damage finding."""
    radius = float(np.sqrt(max(area_m2, 1.0) / np.pi)) + float(buffer_m)
    d2 = (X - centroid[0]) ** 2 + (Y - centroid[1]) ** 2
    return d2 <= radius**2


def assess_risk(mesh: TriangleMesh, damage: dict | None, twin: dict | None,
                cells: int | None = None) -> dict:
    """Deterministic, explainable risk over mesh + damage findings + twin."""
    damage = damage or {"findings": [], "counts": {}}
    twin = twin or {"objects": []}
    findings = damage.get("findings", [])
    structures = [o for o in twin.get("objects", [])
                  if o.get("class") in ("roof", "wall", "structure")
                  and o.get("bbox_min") and o.get("centroid")]

    grid = _slope_grid(mesh, int(cells or settings.intel.risk_grid_cells))
    X, Y = _cell_centers(grid)
    occupied, slope = grid["occupied"], grid["slope"]

    cause = np.zeros((grid["ny"], grid["nx"]), dtype=np.int8)  # 0 none,1 flood,2 debris,3 slope
    flood_mask = np.zeros_like(occupied)
    debris_mask = np.zeros_like(occupied)
    flood_sev = 0
    debris_sev = 0
    for f in findings:
        if f.get("type") == "flood" and f.get("centroid"):
            flood_mask |= _exposure_mask(np.asarray(f["centroid"], dtype=float)[:2],
                                         f.get("affected_area_m2", 0.0),
                                         settings.intel.flood_buffer_m, X, Y)
            flood_sev = max(flood_sev, _LEVEL_NUM.get(f.get("severity", "Low"), 1))
        elif f.get("type") in ("debris_field", "collapse_candidate") and f.get("centroid"):
            debris_mask |= _exposure_mask(np.asarray(f["centroid"], dtype=float)[:2],
                                          f.get("affected_area_m2", 0.0),
                                          settings.intel.debris_buffer_m, X, Y)
            debris_sev = max(debris_sev, _LEVEL_NUM.get(f.get("severity", "Low"), 1))
    steep = occupied & (slope > settings.intel.risk_slope_deg)
    cause[flood_mask] = 1
    cause[debris_mask & (cause == 0)] = 2
    cause[steep & (cause == 0)] = 3
    risk_mask = flood_mask | debris_mask | steep

    # ---- risk zones (connected risk cells with dominant cause) ----
    zones = []
    if risk_mask.any():
        labels, n_zones = nd_label(risk_mask)
        for z in range(1, n_zones + 1):
            cells_in = labels == z
            n_cells = int(cells_in.sum())
            if n_cells < 2:
                continue
            cause_hist = np.bincount(cause[cells_in], minlength=4)
            dom = int(np.argmax(cause_hist[1:])) + 1
            cy, cx = np.nonzero(cells_in)
            area = n_cells * grid["sx"] * grid["sy"]
            zones.append({
                "cause": {1: "flood", 2: "debris", 3: "steep_slope"}[dom],
                "level": "High" if dom == 1 else "Medium",
                "cells": n_cells,
                "area_m2": round(float(area), 1),
                "centroid": [round(float(X[cy.mean().astype(int), cx.mean().astype(int)]), 2),
                             round(float(Y[cy.mean().astype(int), cx.mean().astype(int)]), 2)],
                "note": "largest exposure cause within the zone",
            })

    # ---- flood exposure of structures ----
    exposed = _structures_in_mask(structures, flood_mask, grid)
    exposed_ids = [s.get("uuid", str(i)) for i, s in enumerate(exposed)]

    # ---- accessibility: boundary-reachable passable cells ----
    passable = occupied & (slope <= settings.intel.access_slope_deg) & ~flood_mask
    reachable = np.zeros_like(passable)
    if passable.any():
        lab, _ = nd_label(passable)
        ring = np.zeros_like(passable, dtype=bool)
        ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        border_comps = np.unique(lab[ring])
        border_comps = border_comps[border_comps > 0]
        for c in border_comps:
            reachable |= lab == c

    blocked = []
    reachable_n = 0
    total_n = len(structures)
    occ_ys, occ_xs = np.nonzero(occupied)
    if total_n and len(occ_xs):
        occ_centers = np.stack([grid["lo"][0] + (occ_xs + 0.5) * grid["sx"],
                                grid["lo"][1] + (occ_ys + 0.5) * grid["sy"]], axis=1)
        tree = cKDTree(occ_centers)
        for s in structures:
            c = np.asarray(s.get("centroid", [0, 0, 0]), dtype=float)[:2]
            d, k = tree.query(c)
            if d > max(grid["sx"], grid["sy"]) * 3:  # structure far outside the mesh
                continue
            yc, xc = occ_ys[k], occ_xs[k]
            if reachable[yc, xc]:
                reachable_n += 1
                continue
            if flood_mask[yc, xc]:
                blocked.append({"uuid": s.get("uuid"), "class": s.get("class"),
                                "cause": "flooded_access"})
            elif slope[yc, xc] > settings.intel.access_slope_deg:
                blocked.append({"uuid": s.get("uuid"), "class": s.get("class"),
                                "cause": "slope_exceeds_vehicle_access"})
            else:
                blocked.append({"uuid": s.get("uuid"), "class": s.get("class"),
                                "cause": "isolated_by_terrain"})

    share = reachable_n / max(total_n, 1)
    if total_n == 0:
        access_level, access_grade = "Low", "not_applicable_no_structures"
    elif reachable_n == total_n:
        access_level, access_grade = "Low", "Good"
    elif share >= 0.6:
        access_level, access_grade = "Medium", "Limited"
    else:
        access_level, access_grade = "High", "Restricted"

    # ---- indicator levels with reasoning ----
    indicators: list[dict] = []
    reasoning: list[str] = []
    nums = []

    if flood_sev > 0:
        near = len(exposed_ids) > 0
        num = min(4, flood_sev + (1 if near else 0))
        level = _num_to_level(num)
        r = (f"flood extent {_describe_findings(findings, 'flood')}"
             + ("; overlaps built-up footprints" if near else ""))
        indicators.append({"name": "flood_risk", "level": level, "reasoning": r})
        reasoning.append(r)
        nums.append(num)
    steep_share = float(steep.sum() / max(occupied.sum(), 1))
    if steep_share > 0:
        num = 3 if steep_share >= 0.15 else 2 if steep_share >= 0.05 else 1
        level = _num_to_level(num)
        r = (f"{steep_share * 100:.1f}% of observed ground is steeper than "
             f"{settings.intel.risk_slope_deg:.0f}° (landslide indicator)")
        indicators.append({"name": "landslide_risk", "level": level, "reasoning": r})
        reasoning.append(r)
        nums.append(num)
    if debris_sev > 0:
        num = min(4, debris_sev + (1 if blocked else 0))
        level = _num_to_level(num)
        r = "debris fields detected" + (", obstructing access" if blocked else "")
        indicators.append({"name": "debris_risk", "level": level, "reasoning": r})
        reasoning.append(r)
        nums.append(num)
    if total_n > 0 and blocked:
        level = "High" if access_level in ("High", "Medium") else access_level
        r = (f"{len(blocked)} of {total_n} structures unreachable by emergency "
             f"vehicle ({share * 100:.0f}% reachable)")
        indicators.append({"name": "infrastructure_accessibility", "level": level, "reasoning": r})
        reasoning.append(r)
        nums.append(_LEVEL_NUM[level])

    level = _num_to_level(max(nums)) if nums else "Low"
    if not nums:
        reasoning.append("no flood, debris, steep-ground or access-blocking evidence found")
    overall = {"level": level, "indicators": indicators, "reasoning": reasoning,
               "method": "grid_accessibility_analysis"}

    impassable_share = float((occupied & ~passable).sum() / max(occupied.sum(), 1))
    terrain = {
        "mean_slope_deg": round(float(slope[occupied].mean()), 2) if occupied.any() else 0.0,
        "max_slope_deg": round(float(slope[occupied].max()), 2) if occupied.any() else 0.0,
        "p90_slope_deg": round(float(np.percentile(slope[occupied], 90)), 2) if occupied.any() else 0.0,
        "impassable_share": round(impassable_share, 4),
        "flooded_share": round(float(flood_mask.sum() / max(occupied.sum(), 1)), 4),
    }
    return {
        "overall_risk": overall,
        "flood_risk": {"level": _num_to_level(flood_sev) if flood_sev else "Low",
                       "exposed_structures": exposed_ids} if flood_sev else None,
        "risk_zones": zones,
        "terrain": terrain,
        "accessibility": {
            "grade": access_grade, "structures_total": total_n,
            "structures_reachable": reachable_n, "blocked_routes": blocked,
            "cell_size_m": round(float(max(grid["sx"], grid["sy"])), 3),
        },
        "parameters": {"slope_deg": settings.intel.risk_slope_deg,
                       "access_slope_deg": settings.intel.access_slope_deg,
                       "flood_buffer_m": settings.intel.flood_buffer_m,
                       "debris_buffer_m": settings.intel.debris_buffer_m},
    }


def _structures_in_mask(structures: list[dict], mask: np.ndarray,
                        grid: dict) -> list[dict]:
    """Twin structures whose nearest occupied cell lies in the mask."""
    occ_ys, occ_xs = np.nonzero(grid["occupied"])
    if not len(occ_xs) or not structures:
        return []
    occ_centers = np.stack([grid["lo"][0] + (occ_xs + 0.5) * grid["sx"],
                            grid["lo"][1] + (occ_ys + 0.5) * grid["sy"]], axis=1)
    tree = cKDTree(occ_centers)
    out = []
    for s in structures:
        c = np.asarray(s.get("centroid", [0, 0, 0]), dtype=float)[:2]
        d, k = tree.query(c)
        if d > max(grid["sx"], grid["sy"]) * 3:
            continue
        if mask[occ_ys[k], occ_xs[k]]:
            out.append(s)
    return out


def _describe_findings(findings: list[dict], ftype: str) -> str:
    areas = [f for f in findings if f.get("type") == ftype]
    if not areas:
        return "(no extent measured)"
    tot = sum(f.get("affected_area_m2", 0.0) for f in areas)
    return f"≈ {tot:.0f} m²"


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class RiskAssessmentStage(PipelineStage):
    name = "risk_assessment"
    description = "Explainable flood/landslide/accessibility risk over the scene grid"
    artifact_rel = "intel/risk_report.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run the digital-twin chain first")
        if not (self.workspace / "intel" / "damage_report.json").exists():
            raise StageNotApplicable("no damage_report.json — run damage_assessment first")

    def execute(self) -> None:
        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "repaired_mesh.ply")
        damage = json.loads((self.workspace / "intel" / "damage_report.json").read_text())
        twin_path = self.workspace / "twin" / "twin.json"
        twin = json.loads(twin_path.read_text()) if twin_path.exists() else {"objects": []}
        report = assess_risk(mesh, damage, twin)
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "risk_report.json").write_text(json.dumps(report, indent=2))
        self._count = len(report["risk_zones"])
        self._detail = {"overall_risk": report["overall_risk"]["level"],
                        "zones": self._count,
                        "blocked_routes": len(report["accessibility"]["blocked_routes"])}
        self._outputs = [{"kind": "data", "name": "risk_report", "path": str(intel_dir / "risk_report.json")}]
        self.progress(1.0, {"level": report["overall_risk"]["level"]})
