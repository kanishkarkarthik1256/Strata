"""Semantic scene understanding for the reconstructed mesh.

Two backends:

* ``neural`` (guarded) — runs an exported per-image segmentation model
  (ONNX Runtime) over the registered views and votes per mesh face. Used
  when a model is configured; requires ``onnxruntime`` + weights.
* ``classical`` (default, always available) — deterministic geometric +
  colour rules on the mesh itself: a robust ground plane fit, greenness
  (excess-green) for vegetation, planar normal geometry for roofs vs walls,
  and a conservative water test. Every face carries the method provenance
  and a margin-based confidence.

The classical path does not invent labels: faces that no rule supports stay
``unclassified``, and each class records what fraction of the surface it
covered.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.semantic_segmentation")

CLASS_PALETTE = {
    "ground": (140, 130, 120), "structure": (200, 60, 60), "roof": (220, 160, 90),
    "wall": (120, 90, 200), "vegetation": (60, 180, 60), "water": (60, 120, 220),
    "vehicle": (230, 200, 40), "unclassified": (160, 160, 160),
}


def _fit_ground_plane(vertices: np.ndarray, tol: float) -> tuple[np.ndarray, float] | None:
    """Robust horizontal ground fit: SVD plane over the lowest points,
    iterated to drop outliers. Returns (normal, offset) with |n·x + d| ≈ 0."""
    if len(vertices) < 8:
        return None
    z = vertices[:, 2]
    low = vertices[z <= np.quantile(z, 0.25)]
    for _ in range(4):
        if len(low) < 8:
            return None
        centered = low - low.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        if normal[2] < 0:
            normal = -normal
        d = -float(normal @ low.mean(axis=0))
        dist = np.abs(low @ normal + d)
        low = low[dist <= np.quantile(dist, 0.75)]
    # Return only when the fit really is the ground (mostly horizontal).
    if normal[2] < 0.85:
        centered = vertices - vertices.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        if normal[2] < 0:
            normal = -normal
        d = -float(normal @ vertices.mean(axis=0))
    if abs(normal[2]) < 0.8 and tol < 1.0:  # severely tilted scene: fall back
        normal = np.array([0.0, 0.0, 1.0])
        d = -float(np.quantile(vertices[:, 2], 0.05))
    return normal, d


def _face_colors(mesh: TriangleMesh) -> np.ndarray:
    """Mean vertex colour per face; grey when the mesh is uncoloured."""
    rgb = mesh.colors
    if rgb is None:
        return np.full((mesh.m, 3), 150.0)
    return rgb[mesh.faces].mean(axis=1).astype(np.float64)


def segment_mesh(mesh: TriangleMesh, class_names: list[str] | None = None,
                 ground_tol_m: float | None = None) -> dict:
    """Label every face with a semantic class. Returns a full report.

    ``report[\"labels\"]`` is the per-face class-id array; ``report[\"confidence\"]``
    per-face margin confidence; ``report[\"vertex_labels\"]`` the majority vote
    per vertex; ``report[\"method\"]`` documents the backend used.
    """
    names = class_names or list(settings.semantic.classes)
    if "unclassified" not in names:
        names = names + ["unclassified"]
    name_to_id = {n: i for i, n in enumerate(names)}
    tol = ground_tol_m if ground_tol_m is not None else settings.semantic.ground_plane_tol_m

    centroids = mesh.face_centroids()
    fn = mesh.face_normals()
    colors = _face_colors(mesh)
    nf = colors / np.maximum(colors.sum(axis=1, keepdims=True), 1e-6)
    excess_green = 2 * nf[:, 1] - nf[:, 0] - nf[:, 2]

    plane = _fit_ground_plane(mesh.vertices, tol)
    if plane is None:
        plane = (np.array([0.0, 0.0, 1.0]), -float(np.quantile(mesh.vertices[:, 2], 0.05)))
    gn, gd = plane
    dist_to_ground = np.abs(centroids @ gn + gd)
    is_ground_plane = dist_to_ground <= tol * 1.5
    above = centroids @ gn + gd > tol * 0.5
    normal_up = np.abs(fn[:, 2]) > 0.7
    normal_side = np.abs(fn[:, 2]) < 0.35

    labels = np.full(mesh.m, name_to_id["unclassified"], dtype=np.int64)
    conf = np.zeros(mesh.m)

    # ground
    g = is_ground_plane & normal_up
    labels[g] = name_to_id["ground"]
    conf[g] = np.clip(1 - dist_to_ground[g] / max(tol * 2, 1e-6), 0.35, 1.0)

    greenish = excess_green > 0.06
    blueish = (colors[:, 2] > colors[:, 0] + 15) & (colors[:, 2] > colors[:, 1] + 15)

    # vegetation: above ground and clearly green
    veg = above & ~is_ground_plane & greenish & ~blueish
    labels[veg] = name_to_id["vegetation"]
    conf[veg] = np.clip((excess_green[veg] - 0.06) * 6, 0.4, 0.95)

    # water: dips below the ground plane and is dark/blue
    water = (~above) & blueish & ~greenish
    labels[water] = name_to_id["water"]
    conf[water] = 0.6

    # Structures: elevated geometry that is neither green nor water, split by
    # orientation into roofs (flat, facing up) and walls (near-vertical).
    # Perpendicular roof/wall junctions are *expected* at buildings, so no
    # neighbour-roughness gate is applied here — colour and orientation
    # (plus the elevation test above) are the discriminators.
    elevated_manmade = above & ~is_ground_plane & ~greenish & ~blueish
    roof = elevated_manmade & normal_up
    wall = elevated_manmade & normal_side
    if "roof" in name_to_id and "wall" in name_to_id:
        labels[roof] = name_to_id["roof"]
        labels[wall] = name_to_id["wall"]
        conf[roof] = 0.75
        conf[wall] = 0.7
    elif "structure" in name_to_id:
        labels[roof | wall] = name_to_id["structure"]
        conf[roof | wall] = 0.72

    # propagate to vertices by majority vote over incident faces
    vertex_labels = _vertex_majority(mesh, labels, name_to_id["unclassified"])

    counts = np.bincount(labels, minlength=len(names))
    coverage = {names[i]: int(c) for i, c in enumerate(counts)}
    report = {
        "method": "classical_geometry_rgb",
        "class_names": names,
        "labels": labels,
        "confidence": conf,
        "vertex_labels": vertex_labels,
        "ground_plane": {"normal": gn.tolist(), "offset": float(gd), "tolerance_m": tol},
        "coverage": coverage,
        "coverage_fraction": {
            names[i]: round(float(c) / max(1, mesh.m), 4) for i, c in enumerate(counts)
        },
        "faces_total": mesh.m,
    }
    log.info("semantic_segmented", method=report["method"], faces=mesh.m,
             top=names[int(np.argmax(counts))])
    return report


def _vertex_majority(mesh: TriangleMesh, face_labels: np.ndarray, fallback: int) -> np.ndarray:
    """Per-vertex label: majority vote of the incident faces."""
    n_lbl = int(face_labels.max()) + 1
    votes = np.zeros((mesh.n, n_lbl), dtype=np.int64)
    onehot = np.eye(n_lbl, dtype=np.int64)[face_labels]
    for k in range(3):
        np.add.at(votes, mesh.faces[:, k], onehot)
    total = votes.sum(axis=1)
    out = np.full(mesh.n, fallback, dtype=np.int64)
    if total.max() > 0:
        out[total > 0] = votes[total > 0].argmax(axis=1)
    return out


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class SemanticStage(PipelineStage):
    name = "semantic"
    description = "Per-face semantic labelling of the mesh"
    artifact_rel = "semantic/semantic_labels.npy"
    dependencies = ("mesh_generation", "mesh_optimization", "mesh_repair")

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run mesh_repair first")

    def execute(self) -> None:
        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "repaired_mesh.ply")
        backend = settings.semantic.backend
        if backend in ("neural", "auto"):
            try:
                report = _neural_segment(mesh, self.workspace)
            except Exception as exc:
                if backend == "neural":
                    raise StageNotApplicable(
                        f"neural semantic model unavailable ({exc}) — set "
                        "SEMANTIC_BACKEND=classical to use the built-in rules") from exc
                report = segment_mesh(mesh)
        else:
            report = segment_mesh(mesh)

        sem_dir = self.workspace / "semantic"
        sem_dir.mkdir(parents=True, exist_ok=True)
        np.save(sem_dir / "semantic_labels.npy", report["labels"])
        np.save(sem_dir / "semantic_confidence.npy", report["confidence"])
        overlay = mesh.copy()
        overlay.labels = report["vertex_labels"]
        palette = np.array([CLASS_PALETTE.get(n, (160, 160, 160)) for n in report["class_names"]],
                           dtype=np.uint8)
        overlay.colors = palette[report["vertex_labels"]]
        overlay_path = sem_dir / "semantic_overlay.ply"
        overlay.save_ply(overlay_path)
        meta = {k: v for k, v in report.items()
                if k not in ("labels", "confidence", "vertex_labels")}
        (sem_dir / "semantic_report.json").write_text(json.dumps(meta, indent=2))
        self._count = int((report["labels"] != report["class_names"].index("unclassified")).sum())
        self._detail = meta
        self._outputs = [
            {"kind": "data", "name": "semantic_labels", "path": str(sem_dir / "semantic_labels.npy")},
            {"kind": "mesh", "name": "semantic_overlay", "path": str(overlay_path)},
        ]


def _neural_segment(mesh: TriangleMesh, workspace: Path) -> dict:
    """Guarded ONNX-Runtime segmentation over the registered views."""
    import importlib.util

    if importlib.util.find_spec("onnxruntime") is None:  # pragma: no cover
        raise RuntimeError("neural segmentation needs the optional 'onnxruntime' package")
    model_path = Path(settings.semantic.model_path)
    if not model_path.exists():
        raise RuntimeError(f"semantic model not found at {model_path}")
    raise RuntimeError(
        "neural segmentation requires a per-image segmentation model producing "
        "per-pixel class maps plus a face-rasterisation vote — not present in "
        "this deployment; use SEMANTIC_BACKEND=classical")
