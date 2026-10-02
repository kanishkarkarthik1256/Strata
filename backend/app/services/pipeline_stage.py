"""Pipeline plugin stage interface (Phase 7 plugin architecture).

Every Phase 7 module implements :class:`PipelineStage` and registers itself
via :func:`register`. The orchestrator auto-discovers registered stages
(:func:`discover`) and executes any requested chain after the core
frames → sparse → depth → dense → georef stages, using the same
retry / cancellation / timeline machinery.

A stage owns one artifact keyed by :attr:`PipelineStage.artifact_rel`
(relative to the job workspace). When that artifact exists the orchestrator
skips the stage (resume-after-crash); stages without an artifact always run.

Lifecycle
---------
``initialize()`` → ``validate_inputs()`` (may raise :class:`StageNotApplicable`
to skip cleanly) → ``resume()`` if a checkpoint exists → ``execute()`` →
``checkpoint()`` → ``export_results()``. On failure ``rollback()`` removes
this run's outputs so a retry starts clean; ``cleanup()`` runs last.

All stages publish ``stage:<name>`` progress events, poll cancellation via the
``cancel_check`` callable, and report timing/counts in ``summary()``.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar

from app.logging_config import get_logger

log = get_logger("drone_recon.services.pipeline_stage")

ProgressFn = Callable[[str, float, dict], None]
CancelFn = Callable[[], bool]


class StageNotApplicable(Exception):
    """Inputs are valid but the stage has nothing to do (clean skip, not a failure)."""


class StageCancelled(Exception):
    """Raised inside a stage when cancellation was requested."""


class PipelineStage(ABC):
    """Base class for auto-discovered reconstruction pipeline stages."""

    name: ClassVar[str] = ""
    version: ClassVar[str] = "1.0.0"
    description: ClassVar[str] = ""
    #: Plugin stages that must have produced their artifacts first.
    dependencies: ClassVar[tuple[str, ...]] = ()
    #: Workspace-relative artifact that marks this stage complete (None = always runs).
    artifact_rel: ClassVar[str | None] = None

    def __init__(
        self,
        job_id: str,
        workspace: Path,
        publish: ProgressFn | None = None,
        cancel_check: CancelFn | None = None,
    ) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must define a stage name")
        self.job_id = job_id
        self.workspace = Path(workspace)
        self.publish = publish or (lambda ev, frac, payload: None)
        self.cancel_check = cancel_check or (lambda: False)
        self._resumed = False
        self._started_ms = 0.0
        self._count: int = 0
        self._detail: dict[str, Any] = {}
        self._outputs: list[dict] = []

    # ------------------------------------------------------------------ path

    def artifact_path(self) -> Path | None:
        """Absolute path of this stage's completion artifact (if any)."""
        if self.artifact_rel is None:
            return None
        path = self.workspace / self.artifact_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def done(self) -> bool:
        """True when the completion artifact already exists (rerun → skip)."""
        path = self.artifact_path()
        return path is not None and path.exists()

    @property
    def stage_dir(self) -> Path:
        """Shared per-job mesh/twin output directory (created on demand)."""
        d = self.workspace / "mesh"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------- lifecycle

    def initialize(self) -> None:
        """One-time setup (default: nothing)."""

    def validate_inputs(self) -> None:
        """Verify prerequisites exist; raise StageNotApplicable to skip cleanly."""

    def has_checkpoint(self) -> bool:
        """True when a previous run checkpoint exists for this stage."""
        return self._checkpoint_path().exists()

    def resume(self) -> None:
        """Adopt state from a previous interrupted run (default: mark resumed)."""
        try:
            data = json.loads(self._checkpoint_path().read_text())
        except (OSError, ValueError):
            return
        if data.get("finished") and data.get("version") == self.version:
            self._resumed = True
            self._count = data.get("count", 0)
            self._detail = data.get("detail", {})

    def checkpoint(self) -> None:
        """Persist a checkpoint record (written after a successful execute)."""
        path = self._checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "stage": self.name,
            "version": self.version,
            "finished": True,
            "updated_at": time.time(),
            "count": self._count,
            "detail": self._detail,
            "outputs": self._outputs,
        }, indent=2))

    def rollback(self) -> None:
        """Remove outputs written by this run so a retry starts clean."""
        for out in self._outputs:
            rel = out.get("path")
            if not rel:
                continue
            try:
                Path(rel).unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best effort
                log.debug("rollback_unlink_failed", stage=self.name, path=rel)

    def cleanup(self) -> None:
        """Release resources after the stage finished (default: nothing)."""

    def export_results(self) -> list[dict]:
        """Return artifact metadata for the run report."""
        return list(self._outputs)

    # ---------------------------------------------------------------- state

    def summary(self) -> dict:
        return {"count": self._count, "detail": self._detail,
                "resumed": self._resumed, "outputs": self._outputs}

    def progress(self, frac: float, payload: dict | None = None) -> None:
        self.publish(f"stage:{self.name}", frac, payload or {})

    def is_cancelled(self) -> bool:
        if self.cancel_check():
            raise StageCancelled(f"{self.name} cancelled")
        return False

    def tick(self, started_wall: float) -> dict:
        """Performance metrics for this stage run."""
        return {"duration_ms": round((time.perf_counter() - started_wall) * 1000, 2),
                "name": self.name, "version": self.version}

    def publish_metrics(self, metrics: dict | None = None) -> None:
        """Emit a ``metrics:<name>`` event with stage performance/count data.

        Called by the orchestrator after a successful run so dashboards and
        the SSE stream observe per-stage throughput (default: emits the
        tick-style summary when no explicit metrics dict is supplied).
        """
        payload = metrics or {"name": self.name, "version": self.version,
                              "count": self._count}
        self.publish(f"metrics:{self.name}", 1.0, payload)

    # -------------------------------------------------------------- helpers

    def _checkpoint_path(self) -> Path:
        d = self.workspace / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.name}.json"

    @abstractmethod
    def execute(self) -> None:
        """Run the stage. Set ``self._count`` / ``self._detail`` / ``self._outputs``."""


# ---------------------------------------------------------------------------
# Registry + auto-discovery
# ---------------------------------------------------------------------------

STAGE_REGISTRY: dict[str, type[PipelineStage]] = {}

#: Default autonomous twin chain (dense cloud → textured, semantic digital twin).
MESH_CHAIN = (
    "mesh_generation",
    "mesh_optimization",
    "mesh_repair",
    "texture",
    "semantic",
    "object_detection",
    "digital_twin",
    "confidence_overlay",
    "geo_alignment",
    "lod",
    "scene_index",
)

#: Phase 8 intelligence chain (runs after the digital twin exists).
INTEL_CHAIN = (
    "environmental_intelligence",
    "scene_intelligence",
    "damage_assessment",
    "infrastructure_analysis",
    "risk_assessment",
    "mission_recommendation",
    "change_detection",
    "rag_index",
    "report_generation",
)

#: Phase 9 mission-planning chain (follow-up mission design + learning).
PLANNING_CHAIN = (
    "coverage_prediction",
    "mission_planning",
    "mission_learning",
)


def register(cls: type[PipelineStage]) -> type[PipelineStage]:
    """Class decorator: add a stage to the auto-discovery registry."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define class attribute 'name'")
    if cls.name in STAGE_REGISTRY and STAGE_REGISTRY[cls.name] is not cls:
        raise ValueError(f"duplicate stage name '{cls.name}'")
    STAGE_REGISTRY[cls.name] = cls
    return cls


_STAGE_MODULES = (
    "app.services.mesh_generator",
    "app.services.mesh_optimizer",
    "app.services.mesh_repair",
    "app.services.texture_blender",
    "app.services.semantic_segmentation",
    "app.services.object_detector",
    "app.services.digital_twin_engine",
    "app.services.confidence_overlay",
    "app.services.geo_alignment",
    "app.services.lod_generator",
    "app.services.scene_index",
    # Phase 8 intelligence stages
    "app.services.environment_detector",
    "app.services.scene_intelligence",
    "app.services.damage_detector",
    "app.services.infrastructure_analyzer",
    "app.services.risk_assessment",
    "app.services.mission_recommender",
    "app.services.change_detection",
    "app.services.rag_engine",
    "app.services.report_generator",
    # Phase 9 planning stages
    "app.services.coverage_predictor",
    "app.services.mission_planner",
    "app.services.mission_learning",
)


def discover() -> dict[str, type[PipelineStage]]:
    """Import every module that registers a stage (idempotent)."""
    import importlib

    for mod in _STAGE_MODULES:
        importlib.import_module(mod)
    return STAGE_REGISTRY


def get_stage(name: str) -> type[PipelineStage]:
    discover()
    if name not in STAGE_REGISTRY:
        raise ValueError(f"unknown pipeline stage '{name}' — registered: {sorted(STAGE_REGISTRY)}")
    return STAGE_REGISTRY[name]
