"""Level-of-detail generation for streaming viewers.

Every mesh is decimated into the configured LOD chain (default
``[1.0, 0.5, 0.25, 0.12, 0.06]`` → LOD0…LOD4) using :func:`mesh_optimizer.decimate`.
The manifest lists each level with face counts so a viewer can switch by
projected size. LOD0 is a copy of the repaired mesh.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.mesh_optimizer import DecimateParams, decimate
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.lod_generator")


def generate_lods(mesh: TriangleMesh, ratios: list[float] | None = None,
                  out_dir: Path | None = None) -> tuple[list[dict], list[Path]]:
    """Decimate *mesh* into LOD levels; returns ``(manifest, paths)``."""
    ratios = ratios or list(settings.mesh.lod_ratios)
    manifest: list[dict] = []
    paths: list[Path] = []
    for level, ratio in enumerate(ratios):
        if ratio >= 1.0:
            lod = mesh.copy()
            method = "copy"
        else:
            lod, stats = decimate(mesh, DecimateParams(target_fraction=ratio))
            method = stats["method"]
        if out_dir is not None:
            path = out_dir / f"lod{level}.ply"
            lod.save_ply(path, normals=False)
            paths.append(path)
        manifest.append({
            "level": level, "ratio": ratio, "method": method,
            "faces": lod.m, "vertices": lod.n,
            "path": str(path) if out_dir is not None else None,
        })
    return manifest, paths


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class LodStage(PipelineStage):
    name = "lod"
    description = "Multi-resolution LOD chain for streaming"
    artifact_rel = "mesh/lod/manifest.json"
    dependencies = ("mesh_generation", "mesh_optimization", "mesh_repair")

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run mesh_repair first")

    def execute(self) -> None:
        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "repaired_mesh.ply")
        lod_dir = self.workspace / "mesh" / "lod"
        lod_dir.mkdir(parents=True, exist_ok=True)
        manifest, _paths = generate_lods(mesh, out_dir=lod_dir)
        (lod_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._count = len(manifest)
        self._detail = {"levels": manifest}
        self._outputs = [
            {"kind": "mesh", "name": f"lod{entry['level']}",
             "path": entry["path"], "faces": entry["faces"]}
            for entry in manifest
        ]
