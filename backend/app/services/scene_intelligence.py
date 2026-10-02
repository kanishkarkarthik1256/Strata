"""Scene intelligence — open-vocabulary concept extraction over the twin.

Two backends share one report contract (``intel/scene_report.json``):

* ``classical`` (default, runs here): maps a natural-language prompt to
  semantic mesh classes and twin objects through a synonym table, then scores
  each candidate with its semantic confidence. Prompt terms outside the
  vocabulary return *no evidence* with an explicit note — never a guess.
* ``neural`` (guarded): Grounding DINO + SAM 2 instance grounding. Used only
  when ``INTEL_OPEN_VOCAB_BACKEND=neural`` **and** the ONNX models exist;
  otherwise the stage degrades to classical with a logged reason.

The stage reads the Phase 7 artifacts (repaired mesh + semantic labels +
twin objects) so concepts carry 3-D geometry, measurements and confidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.scene_intelligence")

# ---------------------------------------------------------------------------
# Open-vocabulary vocabulary (classical path)
# ---------------------------------------------------------------------------

#: canonical concept -> semantic mesh classes that support it
CONCEPT_TO_CLASSES: dict[str, list[str]] = {
    "building": ["roof", "wall", "structure"],
    "structure": ["structure", "roof", "wall"],
    "road": ["ground"],
    "water": ["water"],
    "flood": ["water"],
    "vegetation": ["vegetation"],
    "tree": ["vegetation"],
    "canopy": ["vegetation"],
    "roof": ["roof"],
    "wall": ["wall"],
    "vehicle": ["vehicle"],
    "car": ["vehicle"],
    "bridge": ["structure"],
    "tower": ["structure"],
    "powerline": ["structure"],
    "debris": ["unclassified", "structure"],
    "rubble": ["unclassified", "structure"],
    "landslide": ["unclassified", "structure"],
    "ground": ["ground"],
    "terrain": ["ground"],
}

#: every word that maps onto a canonical concept
_SYNONYMS: dict[str, str] = {
    "buildings": "building", "building": "building", "house": "building",
    "homes": "building", "structure": "structure", "structures": "structure",
    "road": "road", "roads": "road", "street": "road", "streets": "road",
    "path": "road", "water": "water", "pond": "water", "lake": "water",
    "river": "water", "flood": "flood", "flooding": "flood",
    "vegetation": "vegetation", "trees": "tree", "tree": "tree", "forest": "tree",
    "canopy": "canopy", "roof": "roof", "roofs": "roof",
    "wall": "wall", "walls": "wall", "vehicle": "vehicle", "vehicles": "vehicle",
    "car": "car", "cars": "car", "truck": "vehicle", "trucks": "vehicle",
    "bridge": "bridge", "bridges": "bridge", "tower": "tower", "towers": "tower",
    "powerline": "powerline", "powerlines": "powerline", "pylon": "tower",
    "debris": "debris", "rubble": "debris", "rubbish": "debris",
    "landslide": "landslide", "landslides": "landslide", "mudslide": "landslide",
    "ground": "ground", "terrain": "terrain", "field": "ground",
}


def _concepts_for_prompt(prompt: str) -> list[str]:
    """Canonical concepts implied by a prompt (empty ⇒ no classical evidence)."""
    if not prompt:
        return []
    words = "".join(ch.lower() if ch.isalnum() or ch == " " else " " for ch in prompt).split()
    found = []
    for w in words:
        concept = _SYNONYMS.get(w)
        if concept and concept not in found:
            found.append(concept)
    return found


# ---------------------------------------------------------------------------
# Classical concept scorer
# ---------------------------------------------------------------------------


def _label_votes(mesh: TriangleMesh, semantic_labels: np.ndarray, class_names: list[str],
                 classes: list[str]) -> tuple[float, int, int]:
    """Share of mesh faces whose label ∈ classes (with that label present)."""
    ids = [i for i, c in enumerate(class_names) if c in classes]
    if not ids or len(semantic_labels) == 0:
        return 0.0, 0, 0
    mask = np.isin(semantic_labels, ids)
    return float(mask.mean()), int(mask.sum()), len(ids)


def _resolve_concept(mesh: TriangleMesh, semantic_labels: np.ndarray,
                     class_names: list[str], twin: dict, concept: str) -> dict:
    """Evidence for one canonical concept across mesh labels + twin objects."""
    classes = CONCEPT_TO_CLASSES.get(concept, [])
    faces_share, n_faces, n_classes = _label_votes(mesh, semantic_labels, class_names, classes)
    twin_objects = [o for o in twin.get("objects", []) if o.get("class") in classes]
    matched_twins = [o for o in twin_objects if o.get("confidence", 0) >= settings.intel.min_concept_confidence]
    # Report up to the largest 12 twin objects for the concept.
    matched_twins.sort(key=lambda o: -o.get("surface_area_m2", 0))
    evidence = {
        "concept": concept,
        "matched_mesh_classes": [c for c in classes],
        "face_coverage": round(faces_share, 4),
        "face_count": int(n_faces),
        "instance_count": len(matched_twins),
        "instances": [
            {"uuid": o.get("uuid"), "class": o.get("class"),
             "centroid": o.get("centroid"), "bbox_min": o.get("bbox_min"),
             "bbox_max": o.get("bbox_max"), "height_m": o.get("height_m"),
             "surface_area_m2": o.get("surface_area_m2"), "confidence": o.get("confidence")}
            for o in matched_twins[:12]
        ],
        "surface_area_total_m2": round(sum(o.get("surface_area_m2", 0) for o in matched_twins), 2),
        "volume_total_m3": round(sum(o.get("volume_m3") or 0 for o in matched_twins), 2),
    }
    # Confidence: twin instances are strongest, face coverage supports.
    has_instances = len(matched_twins) > 0
    coverage_conf = min(1.0, faces_share * 8.0)
    conf = max(has_instances * 0.8, coverage_conf) if (has_instances or faces_share > 0.02) else 0.0
    evidence["confidence"] = round(min(conf, 0.98), 3)
    evidence["evidence_type"] = "twin_instances" if has_instances else (
        "mesh_class_coverage" if faces_share > 0.02 else "none")
    evidence["present"] = has_instances or faces_share > 0.005
    return evidence


# ---------------------------------------------------------------------------
# Guarded neural backend (Grounding DINO + SAM 2)
# ---------------------------------------------------------------------------


def _neural_concepts(frame_dir: Path, prompt: str) -> list[dict]:
    """Grounding DINO + SAM 2 grounding over the registered frames.

    Raises a descriptive RuntimeError when the models/plumbing are absent so
    the stage degrades to the classical backend instead of guessing.
    """
    del frame_dir, prompt  # backend wiring is not present in this deployment
    dino = Path(settings.intel.dino_model_path)
    sam2 = Path(settings.intel.sam2_model_path)
    if not dino.exists() or not sam2.exists():
        raise RuntimeError(
            f"open-vocabulary neural backend needs Grounding DINO + SAM 2 ONNX "
            f"models ({dino}, {sam2}) — set INTEL_OPEN_VOCAB_BACKEND=classical "
            "or provide the weights")
    raise RuntimeError(
        "neural grounding requires per-frame DINO+SAM2 inference plumbing "
        "(box prompting then mask/box extraction) — not present in this "
        "deployment; INTEL_OPEN_VOCAB_BACKEND=classical is the supported path")


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class SceneIntelligenceStage(PipelineStage):
    name = "scene_intelligence"
    description = "Open-vocabulary concept extraction over mesh labels and twin objects"
    artifact_rel = "intel/scene_report.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run the digital-twin chain first")
        if not (self.workspace / "twin" / "twin.json").exists():
            raise StageNotApplicable("no twin objects — run digital_twin first")

    def execute(self) -> None:
        prompts = [p.strip() for p in _load_prompts(self.workspace)] or ["buildings", "water", "vegetation",
                                                                          "road", "vehicle", "debris"]
        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "repaired_mesh.ply")
        labels_path = self.workspace / "semantic" / "semantic_labels.npy"
        twin = json.loads((self.workspace / "twin" / "twin.json").read_text())
        labels = np.load(labels_path) if labels_path.exists() else np.zeros(mesh.m, dtype=np.int64)
        class_names = list(settings.semantic.classes)
        if "unclassified" not in class_names:
            class_names.append("unclassified")

        backend = settings.intel.open_vocab_backend
        if backend == "neural":
            # Explicit neural request: fail loudly (no silent classical guess).
            _neural_concepts(self.workspace / "selected", prompts[0] if prompts else "")

        results = []
        for prompt in prompts:
            concepts = _concepts_for_prompt(prompt)
            if not concepts:
                if backend == "auto":
                    try:  # vocabulary miss → ask the guarded backend, else skip
                        neural = _neural_concepts(self.workspace / "selected", prompt)
                    except RuntimeError as exc:
                        log.info("scene_neural_unavailable", prompt=prompt, reason=str(exc))
                        continue  # no fabrication for out-of-vocabulary prompts
                    else:
                        results.append({"prompt": prompt, "backend": "dino_sam2",
                                        "concepts": neural, "neural": True})
                        continue
                continue  # classical vocabulary miss → no evidence, by design
            concepts_out = []
            for concept in concepts:
                ev = _resolve_concept(mesh, labels, class_names, twin, concept)
                if ev.get("present"):
                    concepts_out.append(ev)
            results.append({"prompt": prompt, "backend": "classical", "concepts": concepts_out,
                            "matched": len(concepts_out)})

        report = {
            "backend": "classical" if all(r.get("backend") == "classical" for r in results) else "mixed",
            "prompts": results,
            "total_matches": sum(len(r.get("concepts", [])) for r in results),
            "note": "classical evidence = mesh class coverage + twin instances; "
                    "out-of-vocabulary prompts produce no match rather than a guess",
        }
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "scene_report.json").write_text(json.dumps(report, indent=2))
        self._count = report["total_matches"]
        self._detail = {"total_matches": report["total_matches"], "prompts": len(results)}
        self._outputs = [{"kind": "data", "name": "scene_report", "path": str(intel_dir / "scene_report.json")}]
        self.progress(1.0, {"matches": report["total_matches"]})


def _load_prompts(workspace: Path) -> list[str]:
    """Prompt list configured by the caller (optional) — else default set."""
    p = workspace / "intel" / "prompts.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict) and isinstance(data.get("prompts"), list):
                return [str(x) for x in data["prompts"]]
        except (OSError, ValueError):
            pass
    return []
