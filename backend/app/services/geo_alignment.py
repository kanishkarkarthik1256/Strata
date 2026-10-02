"""Geospatial alignment of the twin outputs.

Reuses the similarity transform computed during the georeferencing stage
(``georef/alignment.json`` — WGS84 → ENU, Umeyama fit) and applies it to the
repaired mesh and to every twin object. Outputs:

* ``georef/repaired_mesh_enu.ply`` — the mesh in the ENU frame
* ``georef/objects.geojson`` — semantic objects as GeoJSON footprints
* ``georef/crs.json`` — CRS provenance (EPSG/ENU anchor when known)

When no GPS telemetry was available the alignment stage skips cleanly and
outputs stay in the local SfM frame (as documented in the report).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.geo_alignment")

CRS_NOTE = "outputs stay in the local SfM frame (metres) unless georeferenced"


def load_alignment(workspace: Path) -> tuple[np.ndarray, dict] | None:
    """The georef similarity transform (row-vector convention) if it exists."""
    path = workspace / "georef" / "alignment.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if "matrix" not in data:
        return None
    return np.asarray(data["matrix"], dtype=np.float64), data


def apply_alignment_to_mesh(mesh: TriangleMesh, transform: np.ndarray) -> TriangleMesh:
    """Apply the 4x4 row-vector transform to a mesh (vertices only)."""
    out = mesh.copy()
    homo = np.hstack([out.vertices, np.ones((out.n, 1))])
    out.vertices = (homo @ transform.T)[:, :3]
    return out


def objects_to_geojson(objects: list[dict], transform: np.ndarray | None,
                       crs: dict | None) -> dict:
    """Semantic objects as GeoJSON FeatureCollection.

    Footprints are the object bbox ground rectangle transformed into the
    ENU frame (or left local when no transform is available).
    """
    features = []
    for obj in objects:
        lo = np.array(obj["bbox_min"], dtype=np.float64)
        hi = np.array(obj["bbox_max"], dtype=np.float64)
        corners = np.array([
            [lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]],
        ])
        centroid = np.array(obj["centroid"], dtype=np.float64)
        if transform is not None:
            zc = np.hstack([corners, np.full((5, 1), lo[2])])
            zc = (np.hstack([zc, np.ones((5, 1))]) @ transform.T)[:, :2]
            cen = (np.hstack([centroid, np.ones(1)]) @ transform.T)[:3]
            corners = zc
            centroid = cen
        features.append({
            "type": "Feature",
            "properties": {
                "uuid": obj["uuid"],
                "class": obj["class"],
                "confidence": obj["confidence"],
                "height_m": obj["height_m"],
                "surface_area_m2": obj["surface_area_m2"],
                "footprint_area_m2": obj["footprint_area_m2"],
                "volume_m3": obj["volume_m3"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [corners.tolist()],
            },
            "centroid": centroid.tolist(),
        })
    return {
        "type": "FeatureCollection",
        "crs": crs or {"type": "none", "note": CRS_NOTE},
        "features": features,
    }


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class GeoAlignmentStage(PipelineStage):
    name = "geo_alignment"
    description = "ENU alignment of the mesh + GeoJSON object export"
    artifact_rel = "georef/repaired_mesh_enu.ply"
    dependencies = ("mesh_generation", "mesh_optimization", "mesh_repair", "digital_twin")

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run mesh_repair first")

    def execute(self) -> None:
        alignment = load_alignment(self.workspace)
        georef_dir = self.workspace / "georef"
        georef_dir.mkdir(parents=True, exist_ok=True)
        if alignment is None:
            crs_path = georef_dir / "crs.json"
            if not crs_path.exists():
                crs_path.write_text(json.dumps(
                    {"type": "none", "note": CRS_NOTE}, indent=2))
            raise StageNotApplicable(
                "no georeferencing alignment — outputs stay in the local SfM frame; "
                "rerun the pipeline with GPS telemetry to georeference")

        transform, meta = alignment
        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "repaired_mesh.ply")
        enu = apply_alignment_to_mesh(mesh, transform)
        mesh_path = georef_dir / "repaired_mesh_enu.ply"
        enu.save_ply(mesh_path, normals=False)

        crs = {"type": "enu_local", "anchor": meta.get("scale"),
               "note": "aligned to the ENU frame computed by the georeferencing stage"}
        (georef_dir / "crs.json").write_text(json.dumps(crs, indent=2))

        twin_path = self.workspace / "twin" / "twin.json"
        if twin_path.exists():
            twin = json.loads(twin_path.read_text())
            geojson = objects_to_geojson(twin.get("objects", []), transform, crs)
            (georef_dir / "objects.geojson").write_text(json.dumps(geojson, indent=2))
            for obj in geojson["features"]:
                oid = obj["properties"]["uuid"]
                for t in twin["objects"]:
                    if t["uuid"] == oid:
                        t["centroid_enu"] = [round(v, 4) for v in obj["centroid"]]
            (twin_path).write_text(json.dumps(twin, indent=2))

        self._count = enu.m
        self._detail = {"aligned": True, "mesh_enu": str(mesh_path)}
        self._outputs = [
            {"kind": "mesh", "name": "repaired_mesh_enu", "path": str(mesh_path)},
            {"kind": "data", "name": "objects_geojson", "path": str(georef_dir / "objects.geojson")},
        ]
