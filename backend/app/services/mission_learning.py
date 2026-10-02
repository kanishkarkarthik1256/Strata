"""Mission learning stage — closes the feedback loop.

After a mission workspace has measured artifacts this stage:

1. collects the measured record (``mission_history.record_from_workspace``),
2. appends it to the deployment history store,
3. runs similar-mission retrieval + the learning summary (both honor the
   ``min_history_for_ml`` gate — nothing is labelled a learned prediction
   below it),
4. validates plan predictions vs measured outcomes (prediction error row).

Outputs ``intel/learning_report.json`` and, when a plan existed, writes the
validation row. Historical similarity evidence is explicitly labelled
``historical`` in every payload.
"""

from __future__ import annotations

import json

from app.logging_config import get_logger
from app.services.mission_history import (
    append_record,
    learning_summary,
    load_history,
    record_from_workspace,
    similar_missions,
    validate_mission,
)
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.mission_learning")


@register
class MissionLearningStage(PipelineStage):
    name = "mission_learning"
    description = "Persist mission record + historical learning summary + prediction validation"
    artifact_rel = "intel/learning_report.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists() \
                and not (self.workspace / "pipeline_report.json").exists():
            raise StageNotApplicable("no completed mission artifacts — nothing to learn from")

    def execute(self) -> None:
        record = record_from_workspace(self.workspace)
        # keep only meaningful records (a mission that produced mesh)
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no reconstructed mesh — no learning record recorded")
        append_record(record)
        records = load_history()
        similar = similar_missions(record, records)
        learning = learning_summary(records)
        validation = validate_mission(record)
        report = {
            "mission": record,
            "similar_missions": similar,
            "learning": learning,
            "validation": validation,
            "history_total": len(records),
            "label": "historical observations + explicit data gates — no "
                     "unvalidated ML predictions",
        }
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "learning_report.json").write_text(json.dumps(report, indent=2))
        (intel_dir / "validation.json").write_text(json.dumps(validation, indent=2))
        self._count = len(records)
        self._detail = {"history_total": len(records),
                        "learning_status": learning.get("status"),
                        "similar": similar.get("count", similar.get("status")),
                        "mission_success": record.get("mission_success")}
        self._outputs = [
            {"kind": "data", "name": "learning_report", "path": str(intel_dir / "learning_report.json")},
            {"kind": "data", "name": "validation", "path": str(intel_dir / "validation.json")},
        ]
        self.progress(1.0, {"history_total": len(records)})
