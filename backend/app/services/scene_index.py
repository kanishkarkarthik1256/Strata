"""Scene index — the machine-readable manifest for one digital twin.

Aggregates every Phase 7 artifact (mesh, LODs, texture, semantic labels,
twin objects, confidence, georeferencing) into one ``scene_index.json`` plus
a compact analytics export, so dashboards and exports have a single source
of truth for a job.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.logging_config import get_logger
from app.services.pipeline_stage import PipelineStage, register

log = get_logger("drone_recon.services.scene_index")


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _artifact(path: Path) -> dict | None:
    if path.exists():
        return {"path": str(path), "bytes": path.stat().st_size}
    return None


def build_scene_index(job_id: str, workspace: Path) -> dict:
    """Collect existing artifacts into the scene manifest."""
    index: dict = {
        "job_id": job_id,
        "artifacts": {},
        "quality": {},
        "export_formats": [],
    }

    for key, rel in (
        ("base_mesh_report", "mesh/base_mesh.json"),
        ("optimized_mesh", "mesh/optimized_mesh.ply"),
        ("repaired_mesh", "mesh/repaired_mesh.ply"),
        ("texture_atlas", "texture/texture_atlas.png"),
        ("semantic_overlay", "semantic/semantic_overlay.ply"),
        ("mesh_confidence", "confidence/mesh_confidence.ply"),
        ("mesh_enu", "georef/repaired_mesh_enu.ply"),
        ("objects_geojson", "georef/objects.geojson"),
    ):
        art = _artifact(workspace / rel)
        if art:
            index["artifacts"][key] = art

    for key, rel in (
        ("mesh_repair", "mesh/repair_report.json"),
        ("lods", "mesh/lod/manifest.json"),
        ("texture", "texture/texture_report.json"),
        ("semantic", "semantic/semantic_report.json"),
        ("twin", "twin/twin.json"),
        ("scene_graph", "twin/scene_graph.json"),
        ("confidence", "confidence/confidence_report.json"),
        ("detections", "objects/detections.json"),
    ):
        data = _json(workspace / rel)
        if data is not None:
            index[key] = data

    crs = _json(workspace / "georef" / "crs.json")
    index["georef"] = {"crs": crs or {"type": "none",
                                      "note": "local SfM frame (no GPS alignment)"}}
    if index.get("twin"):
        index["quality"]["object_count"] = index["twin"].get("object_count", 0)
    if index.get("semantic"):
        index["quality"]["semantic_method"] = index["semantic"].get("method")
    if index.get("confidence"):
        index["quality"]["mean_confidence"] = index["confidence"].get("mean_confidence")

    for rel, fmt in (
        ("mesh/repaired_mesh.ply", "ply"),
        ("mesh/repaired_mesh.obj", "obj"),
        ("texture/texture_atlas.png", "texture_png"),
        ("georef/objects.geojson", "geojson"),
        ("georef/repaired_mesh_enu.ply", "ply_enu"),
    ):
        if (workspace / rel).exists():
            index["export_formats"].append(fmt)
    return index


# ---------------------------------------------------------------------------
# Pipeline stage (terminal)
# ---------------------------------------------------------------------------


@register
class SceneIndexStage(PipelineStage):
    name = "scene_index"
    description = "Aggregate the digital-twin scene manifest"
    artifact_rel = "twin/scene_index.json"

    def execute(self) -> None:
        twin_dir = self.workspace / "twin"
        twin_dir.mkdir(parents=True, exist_ok=True)
        index = build_scene_index(self.job_id, self.workspace)
        path = twin_dir / "scene_index.json"
        path.write_text(json.dumps(index, indent=2))
        (twin_dir / "analytics.json").write_text(json.dumps({
            "job_id": self.job_id,
            "objects": index.get("twin", {}).get("object_count", 0),
            "classes": index.get("twin", {}).get("classes", {}),
            "semantic_method": index.get("semantic", {}).get("method"),
            "confidence_mean": index.get("confidence", {}).get("mean_confidence"),
            "export_formats": index.get("export_formats", []),
        }, indent=2))
        twin = index.get("twin", {})
        self._count = twin.get("object_count", 0)
        self._detail = {"artifacts": sorted(index["artifacts"])}
        self._outputs = [{"kind": "data", "name": "scene_index", "path": str(path)}]
