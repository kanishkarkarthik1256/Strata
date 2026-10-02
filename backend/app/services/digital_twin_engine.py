"""Digital twin engine — objects, measurements and scene graph.

Objects are extracted from the semantic mesh: each connected run of faces
sharing the same non-terrain class becomes one :class:`TwinObject` with a
UUID, class, bounding box, surface area, footprint (2D convex hull), volume
(divergence theorem, only when the run is watertight) and a mean semantic
confidence. Per-object face sets are persisted as masks so the mesh and
texture references stay exact.

The scene graph encodes measured spatial relations (gap, vertical overlap)
between objects as plain edges with relation types derived from the classes.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.digital_twin_engine")

OBJECT_CLASSES = {"structure", "roof", "wall", "vegetation", "water", "vehicle", "building", "tree", "debris"}


@dataclass
class TwinObject:
    """One semantic object extracted from the mesh."""

    uuid: str
    class_name: str
    face_count: int
    vertex_count: int
    bbox_min: list[float]
    bbox_max: list[float]
    height_m: float
    surface_area_m2: float
    footprint_area_m2: float
    volume_m3: float | None
    volume_closed: bool
    confidence: float
    centroid: list[float]

    def to_dict(self) -> dict:
        return {
            "uuid": self.uuid,
            "class": self.class_name,
            "face_count": self.face_count,
            "vertex_count": self.vertex_count,
            "bbox_min": [round(v, 4) for v in self.bbox_min],
            "bbox_max": [round(v, 4) for v in self.bbox_max],
            "height_m": round(self.height_m, 4),
            "surface_area_m2": round(self.surface_area_m2, 4),
            "footprint_area_m2": round(self.footprint_area_m2, 4),
            "volume_m3": round(self.volume_m3, 4) if self.volume_m3 is not None else None,
            "volume_closed": self.volume_closed,
            "confidence": round(self.confidence, 4),
            "centroid": [round(v, 4) for v in self.centroid],
            "measurements": {
                "height_m": round(self.height_m, 4),
                "surface_area_m2": round(self.surface_area_m2, 4),
                "footprint_area_m2": round(self.footprint_area_m2, 4),
                "volume_m3": round(self.volume_m3, 4) if self.volume_m3 is not None else None,
            },
        }


def _volume_closed(sub: TriangleMesh) -> tuple[float | None, bool]:
    """Volume of a watertight component via the divergence theorem."""
    if not sub.is_watertight():
        return None, False
    a = sub.vertices[sub.faces[:, 0]]
    b = sub.vertices[sub.faces[:, 1]]
    c = sub.vertices[sub.faces[:, 2]]
    cross = np.cross(b - a, c - a)
    vol = float(np.sum(a * cross) / 6.0)
    return abs(vol), True


def _footprint_area(xyz: np.ndarray) -> float:
    """2D (XY) convex-hull footprint of an object."""
    if len(xyz) < 3:
        return 0.0
    try:
        hull = ConvexHull(xyz[:, :2])
        return float(hull.volume)
    except Exception:
        return 0.0


def extract_objects(mesh: TriangleMesh, semantic_labels: np.ndarray,
                    semantic_conf: np.ndarray, min_faces: int = 6,
                    class_names: list[str] | None = None) -> list[TwinObject]:
    """Segment non-terrain faces into objects by (class, connected run)."""
    names = class_names or list(settings_semantic_classes())
    objects: list[TwinObject] = []
    for cls in names:
        if cls not in OBJECT_CLASSES:
            continue
        lbl = names.index(cls) if cls in names else -1
        if lbl < 0:
            continue
        mask = semantic_labels == lbl
        if not mask.any():
            continue
        sub = mesh.submesh(mask)
        k, comp = sub.face_components()
        for ci in range(k):
            cm = comp == ci
            if int(cm.sum()) < min_faces:
                continue
            obj_sub = sub.submesh(cm)
            verts = obj_sub.vertices
            obj_areas = obj_sub.face_areas()
            lo, hi = obj_sub.bounds()
            volume, closed = _volume_closed(obj_sub)
            confidence = float(np.mean(semantic_conf[mask][cm])) if semantic_conf is not None else 0.5
            objects.append(TwinObject(
                uuid=uuid.uuid4().hex,
                class_name=cls,
                face_count=int(cm.sum()),
                vertex_count=obj_sub.n,
                bbox_min=lo.tolist(), bbox_max=hi.tolist(),
                height_m=float(hi[2] - lo[2]),
                surface_area_m2=float(obj_areas.sum()),
                footprint_area_m2=_footprint_area(verts),
                volume_m3=volume, volume_closed=closed,
                confidence=confidence,
                centroid=verts.mean(axis=0).tolist(),
            ))
    objects.sort(key=lambda o: -o.surface_area_m2)
    log.info("twin_objects_extracted", count=len(objects))
    return objects


def settings_semantic_classes() -> list[str]:
    from app.config.settings import settings

    return list(settings.semantic.classes)


def build_scene_graph(mesh: TriangleMesh, objects: list[TwinObject],
                      gap_factor: float = 0.2) -> dict:
    """Spatial adjacency between objects (bbox proximity + vertical overlap)."""
    nodes = [o.to_dict() for o in objects]
    edges = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            amin, amax = np.array(a.bbox_min), np.array(a.bbox_max)
            bmin, bmax = np.array(b.bbox_min), np.array(b.bbox_max)
            gap = float(np.max(np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))))
            scale = max(float(np.max(np.maximum(amax - amin, bmax - bmin))), 1e-6)
            if gap > gap_factor * scale:
                continue
            overlap_ratio = float(np.maximum(0.0, np.minimum(amax[2], bmax[2]) - np.maximum(amin[2], bmin[2])))
            vertical = float(np.maximum(0.0, overlap_ratio))
            edges.append({
                "source": a.uuid, "target": b.uuid,
                "source_class": a.class_name, "target_class": b.class_name,
                "relation": _relation(a.class_name, b.class_name, vertical > 0),
                "gap_m": round(gap, 4),
                "vertical_overlap_m": round(vertical, 4),
            })
    return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}


def _relation(c1: str, c2: str, vertical: bool) -> str:
    pairs = {
        frozenset({"roof", "wall"}): "supports",
        frozenset({"roof", "structure"}): "part_of",
        frozenset({"wall", "structure"}): "part_of",
        frozenset({"structure", "vegetation"}): "adjacent",
    }
    return pairs.get(frozenset({c1, c2}), "near" if not vertical else "touches")


def save_object_masks(mesh: TriangleMesh, objects: list[TwinObject],
                      semantic_labels: np.ndarray, out_dir: Path) -> None:
    """Persist per-object face masks for exact mesh/texture references."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = settings_semantic_classes()
    for obj in objects:
        lbl = names.index(obj.class_name) if obj.class_name in names else -1
        if lbl < 0:
            continue
        np.save(out_dir / f"{obj.uuid}.npy", np.flatnonzero(semantic_labels == lbl))


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class DigitalTwinStage(PipelineStage):
    name = "digital_twin"
    description = "Object extraction, measurements and scene graph"
    artifact_rel = "twin/twin.json"
    dependencies = ("mesh_generation", "mesh_optimization", "mesh_repair", "semantic")

    def validate_inputs(self) -> None:
        mesh_path = self.workspace / "mesh" / "repaired_mesh.ply"
        if not mesh_path.exists():
            raise StageNotApplicable("no repaired mesh — run mesh_repair first")
        if not (self.workspace / "semantic" / "semantic_labels.npy").exists():
            raise StageNotApplicable("no semantic labels — run semantic first")

    def execute(self) -> None:
        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "repaired_mesh.ply")
        labels = np.load(self.workspace / "semantic" / "semantic_labels.npy")
        conf = np.load(self.workspace / "semantic" / "semantic_confidence.npy")
        objects = extract_objects(mesh, labels, conf)
        graph = build_scene_graph(mesh, objects)

        twin_dir = self.workspace / "twin"
        twin_dir.mkdir(parents=True, exist_ok=True)
        save_object_masks(mesh, objects, labels, twin_dir / "masks")
        twin = {
            "job_id": self.job_id,
            "object_count": len(objects),
            "objects": [o.to_dict() for o in objects],
            "scene_graph": graph,
            "classes": {
                c: sum(1 for o in objects if o.class_name == c)
                for c in sorted({o.class_name for o in objects})
            },
        }
        (twin_dir / "twin.json").write_text(json.dumps(twin, indent=2))
        (twin_dir / "scene_graph.json").write_text(json.dumps(graph, indent=2))
        self._count = len(objects)
        self._detail = {"object_count": len(objects), "classes": twin["classes"]}
        self._outputs = [
            {"kind": "data", "name": "twin", "path": str(twin_dir / "twin.json")},
            {"kind": "data", "name": "scene_graph", "path": str(twin_dir / "scene_graph.json")},
        ]
