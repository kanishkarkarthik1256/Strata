"""Damage assessment — geometric evidence rules over the semantic mesh.

Damage is *not* inferred from nothing: each finding below is computed from
measurable geometry and carries the evidence that produced it. Anything that
would require a pre-event baseline (new/removed structures, vegetation loss,
construction progress) is reported under ``requires_baseline`` instead of
being fabricated — the multi-mission change detector feeds those when a
second mission is available.

Findings (type → rule):

* ``flood`` — water-labelled faces form a connected area above
  ``flood_min_area_m2``; severity grows with extent and structure proximity.
* ``debris_field`` — many small disconnected elevated fragments of manmade /
  unclassified geometry near the ground plane (the signature of rubble).
* ``collapse_candidate`` — debris clusters overlapping the horizontal
  footprint of a standing structure (roof/wall object).

Severity: Low/Medium/High/Critical from affected-area and exposure rules,
with a human-readable ``rationale``. Every finding gets world/GPS
coordinates when the mesh is in the ENU frame and local metres otherwise.
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

log = get_logger("drone_recon.services.damage_detector")

_AREA_PER_FINDING = 4.0


def _load_workspace(workspace: Path) -> tuple[TriangleMesh, np.ndarray, list[str]]:
    mesh = TriangleMesh.read_ply(workspace / "mesh" / "repaired_mesh.ply")
    labels = np.load(workspace / "semantic" / "semantic_labels.npy")
    class_names = list(settings.semantic.classes)
    return mesh, labels.astype(np.int64), class_names


def _class_id(class_names: list[str], name: str) -> int | None:
    return class_names.index(name) if name in class_names else None


def _face_centroids_areas(mesh: TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    return mesh.face_centroids(), mesh.face_areas()


def _ground_plane(mesh: TriangleMesh, labels: np.ndarray, class_names: list[str]) -> tuple[np.ndarray, float]:
    """Best-fit plane over ground-labelled face centroids (n·x + d = 0)."""
    gid = _class_id(class_names, "ground")
    ground = mesh.face_centroids()[labels == gid] if gid is not None else None
    if ground is None or len(ground) < 20:
        return np.array([0.0, 0.0, 1.0]), -float(np.quantile(mesh.vertices[:, 2], 0.05))
    centered = ground - ground.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, -float(normal @ ground.mean(axis=0))


def _severity(area_m2: float, share: float, near_structures: bool, exposed_people: bool = False) -> tuple[str, str]:
    """Deterministic severity from extent + exposure. Returns (level, rationale)."""
    reason = (f"affected area ≈ {area_m2:.0f} m² ({share * 100:.1f}% of the scene)")
    if near_structures:
        reason += ", overlaps built-up footprints"
    if exposed_people:
        reason += ", coincides with inhabited structures"
    if area_m2 > 800 or share > 0.2:
        level = "Critical"
    elif area_m2 > 200 or (share > 0.05 and near_structures):
        level = "High"
    elif area_m2 > _AREA_PER_FINDING * 2 or near_structures:
        level = "Medium"
    else:
        level = "Low"
    return level, reason


def assess_flood(mesh: TriangleMesh, labels: np.ndarray, class_names: list[str],
                 centroids: np.ndarray, areas: np.ndarray) -> dict | None:
    """Connected flood surface from the water class."""
    wid = _class_id(class_names, "water")
    if wid is None:
        return None
    water = labels == wid
    if not water.any():
        return None
    sub = mesh.submesh(water)
    n_comp, comp = sub.face_components()
    best_area, best_idx = 0.0, -1
    for c in range(n_comp):
        a = float(sub.face_areas()[comp == c].sum())
        if a > best_area:
            best_area, best_idx = a, c
    if best_area < settings.intel.flood_min_area_m2:
        return None
    cm = comp == best_idx
    cxyz = sub.face_centroids()[cm]
    area = float(sub.face_areas()[cm].sum())
    share = float(area / max(mesh.face_areas().sum(), 1e-9))
    # Structure proximity: distance from flood cells to roof/wall/vehicle cells.
    manmade = np.zeros(mesh.m, dtype=bool)
    for cls in ("roof", "wall", "structure", "vehicle"):
        cid = _class_id(class_names, cls)
        if cid is not None:
            manmade |= labels == cid
    near = False
    if manmade.any():
        tree = cKDTree(centroids[manmade])
        d, _ = tree.query(cxyz)
        near = bool((d < settings.intel.flood_buffer_m).any())
    level, reason = _severity(area, share, near)
    return {
        "type": "flood", "severity": level, "confidence": round(0.6 + 0.35 * min(area / 400.0, 1.0), 3),
        "affected_area_m2": round(area, 1), "scene_share": round(share, 4),
        "centroid": [round(float(v), 2) for v in cxyz.mean(axis=0)],
        "bbox_min": [round(float(v), 2) for v in sub.vertices.min(axis=0)],
        "bbox_max": [round(float(v), 2) for v in sub.vertices.max(axis=0)],
        "near_structures": near, "evidence": "water-class connected surface",
        "method": "semantic_geometry", "rationale": reason,
    }


def assess_debris(mesh: TriangleMesh, labels: np.ndarray, class_names: list[str],
                  centroids: np.ndarray, areas: np.ndarray,
                  gn: np.ndarray, gd: float) -> dict | None:
    """Debris: small disconnected fragments of manmade/unclassified geometry
    close to the ground plane and under ``debris_max_height_m``."""
    cand = np.zeros(mesh.m, dtype=bool)
    for cls in ("structure", "unclassified", "wall"):
        cid = _class_id(class_names, cls)
        if cid is not None:
            cand |= labels == cid
    cand &= ~np.isin(labels, [_class_id(class_names, "ground")])
    if not cand.any():
        return None
    sub = mesh.submesh(cand)
    n_comp, comp = sub.face_components()
    cxyz = sub.face_centroids()
    cheight = cxyz @ gn + gd
    good = (cheight > 0.3) & (cheight < settings.intel.debris_max_height_m)
    cluster_areas = []
    for c in range(n_comp):
        sel = (comp == c) & good
        if sel.any():
            a = float(sub.face_areas()[sel].sum())
            if _AREA_PER_FINDING <= a:
                cluster_areas.append(a)
    if not cluster_areas:
        return None
    total = float(np.sum(cluster_areas))
    share = float(total / max(mesh.face_areas().sum(), 1e-9))
    # Centroid over the qualifying fragments.
    frag_cent = cxyz[(comp >= 0) & good].mean(axis=0)
    level, reason = _severity(total, share, near_structures=False)
    level = "High" if len(cluster_areas) >= 12 else level  # extensive fragmentation
    if len(cluster_areas) >= 12:
        reason += f", {len(cluster_areas)} individual fragments"
    return {
        "type": "debris_field", "severity": level,
        "confidence": round(0.5 + 0.4 * min(len(cluster_areas) / 15.0, 1.0), 3),
        "affected_area_m2": round(total, 1), "scene_share": round(share, 4),
        "fragment_count": len(cluster_areas),
        "centroid": [round(float(v), 2) for v in frag_cent],
        "evidence": f"{len(cluster_areas)} elevated manmade fragments near ground",
        "method": "fragmentation_geometry", "rationale": reason,
    }


def assess_collapse(debris: dict | None, twin: dict) -> dict | None:
    """Collapse candidate: debris that overlaps a standing structure's
    footprint (a roof/wall twin object whose bbox envelope the debris enters)."""
    structures = [o for o in twin.get("objects", [])
                  if o.get("class") in ("roof", "wall", "structure")
                  and o.get("confidence", 0) >= settings.intel.min_concept_confidence]
    if debris is None or not structures:
        return None
    dc = np.asarray(debris["centroid"], dtype=float)
    near = False
    for o in structures:
        lo = np.asarray(o.get("bbox_min", dc), dtype=float)
        hi = np.asarray(o.get("bbox_max", dc), dtype=float)
        if np.all(lo - 12.0 <= dc) and np.all(dc <= hi + 12.0):
            near = True
            break
    if not near:
        return None
    conf = round(min(0.85, debris["confidence"] + 0.15), 3)
    return {
        "type": "collapse_candidate", "severity": debris["severity"],
        "confidence": conf, "affected_area_m2": debris["affected_area_m2"],
        "scene_share": debris["scene_share"], "centroid": debris["centroid"],
        "near_structures": True,
        "evidence": "debris overlaps a roof/wall footprint",
        "method": "structural_footprint_overlap", "rationale": debris["rationale"],
        "note": "single-mission proxy — confirm with a baseline mission via change_detection",
    }


def assess_damage(workspace: Path, twin: dict | None = None) -> dict:
    """Full damage assessment over one mission's semantic mesh."""
    mesh, labels, class_names = _load_workspace(workspace)
    if twin is None:
        twin_path = workspace / "twin" / "twin.json"
        twin = json.loads(twin_path.read_text()) if twin_path.exists() else {"objects": []}
    centroids, areas = _face_centroids_areas(mesh)
    gn, gd = _ground_plane(mesh, labels, class_names)

    findings = []
    flood = assess_flood(mesh, labels, class_names, centroids, areas)
    if flood:
        findings.append(flood)
    debris = assess_debris(mesh, labels, class_names, centroids, areas, gn, gd)
    if debris:
        findings.append(debris)
    collapse = assess_collapse(debris, twin)
    if collapse:
        # The compound finding supersedes its underlying debris entry.
        findings = [f for f in findings if f["type"] != "debris_field"]
        findings.append(collapse)

    severity_counts = {lvl: sum(1 for f in findings if f["severity"] == lvl)
                       for lvl in ("Low", "Medium", "High", "Critical")}
    return {
        "findings": findings,
        "counts": severity_counts,
        "max_severity": max(severity_counts, key=severity_counts.get) if findings else "None",
        "requires_baseline": [
            "new_structure", "removed_structure", "vegetation_loss",
            "construction_progress", "fire_damage_change",
        ],
        "note": "change-type damage needs a second mission — see change_detection",
        "method": "classical_semantic_geometry",
    }


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class DamageAssessmentStage(PipelineStage):
    name = "damage_assessment"
    description = "Geometric flood/debris/collapse findings with severity + rationale"
    artifact_rel = "intel/damage_report.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run the digital-twin chain first")
        if not (self.workspace / "semantic" / "semantic_labels.npy").exists():
            raise StageNotApplicable("no semantic labels — run semantic first")

    def execute(self) -> None:
        report = assess_damage(self.workspace)
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "damage_report.json").write_text(json.dumps(report, indent=2))
        self._count = len(report["findings"])
        self._detail = {"findings": report["counts"], "max_severity": report["max_severity"]}
        self._outputs = [{"kind": "data", "name": "damage_report", "path": str(intel_dir / "damage_report.json")}]
        self.progress(1.0, {"findings": self._count})
