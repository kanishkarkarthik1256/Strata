"""Autonomous pipeline orchestrator.

Runs the full chain — frame extraction → sparse reconstruction → per-view
depth generation → dense reconstruction → georeferencing/reports — from a
single entry point, with:

* **resume/recovery** — every stage is keyed to a workspace artifact
  (``selected/``, ``poses.json``, ``depth/*.npy``, ``dense_report.json``);
  stages whose artifact already exists are skipped, so an interrupted run
  continues from the last completed stage and per-frame depth caching makes
  the depth stage resumable even mid-stage.
* **retries** — a failing stage is retried (PIPE_STAGE_RETRIES) before the
  pipeline reports failure.
* **cancellation** — checked between stages and between depth frames via the
  streaming engine's per-job cancel flag.
* **timeline + profiling** — every stage records start/finish/duration and
  throughput; the timeline, performance profile, and per-stage state are
  published as streaming events and persisted to ``pipeline_report.json``
  (written incrementally, so a crashed run still leaves a readable report).
"""

from __future__ import annotations

import json
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.dense_reconstruction import run_dense_reconstruction
from app.services.depth_generator import StereoParams, generate_view_depths
from app.services.depth_prefetch import start_depth_prefetch, stop_depth_prefetch
from app.services.depth_refinement import RefineParams
from app.services.frame_extractor import extract_frames
from app.services.georeferencing import (
    align_to_enu,
    analyze_gps_track,
    crs_metadata,
    filter_valid_gps_points,
    wgs84_to_enu,
)
from app.services.image_files import count_image_files
from app.services.performance_profiler import PerformanceProfiler
from app.services.pointcloud import read_ply, save_ply
from app.services.sparse_reconstruction import run_sparse_reconstruction
from app.services.stage_caching import (
    compute_stage_fingerprint,
    describe_fingerprint_mismatch,
    is_stage_cache_valid,
    load_stage_details,
    save_stage_fingerprint,
)
from app.services.streaming_engine import engine

log = get_logger("drone_recon.services.pipeline_orchestrator")

Progress = Callable[[str, float, dict], None]

STAGES = ("frames", "sparse", "depth", "dense", "georef")


def _frames_complete(workspace: Path) -> bool:
    """Frames are complete only with a selection SfM can consume.

    One selected frame is not a usable reconstruction input: SfM needs
    multi-view geometry, and a single-frame cache made Retry a no-op (the
    stage reported complete, sparse re-failed identically).
    """
    selected = workspace / "selected"
    return selected.is_dir() and count_image_files(selected) >= 2


def _sparse_complete(workspace: Path) -> bool:
    """Sparse is complete only with a registered multi-camera trajectory.

    A single-camera poses.json is not a reconstruction — depth cannot run on
    it and dense correctly refuses it. Caching it made Retry skip sparse and
    re-fail downstream without ever re-running the broken stage.
    """
    poses_path = workspace / "poses.json"
    if not poses_path.exists():
        return False
    try:
        return len(json.loads(poses_path.read_text()).get("frames", [])) >= 2
    except (OSError, ValueError):
        return False


def _depth_complete(workspace: Path) -> bool:
    """Depth is complete when every registered view is mapped OR excluded by name,
    AND enough maps survived to feed reconstruction.

    Requiring a map for *every* view contradicted the conditioning contract:
    hover/low-parallax views are deliberately refused and named by the depth
    stage itself (21 on flight_to_tower_7511dc, 3 on Bundestag11_561869), so
    the check could never pass for a run with a legitimate exclusion — the
    record then said ``resume.depth: false`` and every resume silently re-ran
    the depth stage it had already completed. An unnamed missing view still
    counts as incomplete.

    But named exclusions cannot outvote maps: a run whose usable maps
    (generated + cached + failed-execution) number fewer than two is not a
    completed depth stage — it is a degenerate one. Caching it made Retry a
    no-op loop: dense re-failed the audit identically while depth reported
    "complete" (sunset_06cfea: 1 map + 9 exclusions, 10-camera trajectory).
    """
    poses_path = workspace / "poses.json"
    if not poses_path.exists():
        return False
    try:
        frames = json.loads(poses_path.read_text()).get("frames", [])
    except (OSError, ValueError):
        return False
    if not frames or len(frames) < 2:
        return False
    # The trailing view of an open trajectory legitimately has no depth map.
    depth_dir = workspace / "depth"
    ids = [f["frame_id"] for f in frames]
    missing = [fid for fid in ids[:-1]
               if not (depth_dir / f"{fid}.npy").exists()]
    if missing:
        excluded = load_stage_details("depth", workspace).get("excluded") or []
        named = {str(fid) for fid in excluded}
        if not named or not all(fid in named for fid in missing):
            return False
    # The usability floor counts MAPS ON DISK — the artifacts the dense audit
    # actually reads — never the stage's bookkeeping lists, which a legacy
    # record may not carry. What fusion consumes is what counts.
    maps_on_disk = sum(1 for fid in ids if (depth_dir / f"{fid}.npy").exists())
    return maps_on_disk >= 2


def _dense_complete(workspace: Path) -> bool:
    report_path = workspace / "dense_report.json"
    if not report_path.exists():
        return False
    try:
        data = json.loads(report_path.read_text())
        return data.get("status") == "completed" and (workspace / "dense" / "dense_model.ply").exists()
    except Exception:
        return False


ARTIFACT_CHECKS = {
    "frames": _frames_complete,
    "sparse": _sparse_complete,
    "depth": _depth_complete,
    "dense": _dense_complete,
}


@dataclass
class StageState:
    name: str
    status: str = "pending"  # pending|running|completed|skipped|failed|cancelled
    start_ms: float = 0.0
    end_ms: float = 0.0
    duration_ms: float = 0.0
    count: int = 0
    error: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "count": self.count,
            "error": self.error,
            "detail": self.detail,
        }


@dataclass
class PipelineRequest:
    """One autonomous pipeline run."""

    extraction_mode: str | None = None  # None = resolve from target_fps/settings
    every_n: int | None = None
    target_fps: float | None = None
    top_percent: float | None = None
    quality_threshold: float | None = None
    # Duration-aware keyframe budget (Phase: performance). None = resolve
    # from video duration: clamp(duration_s × fps_budget, 60, 200). 0
    # disables the budget entirely (all frames that pass the quality gates
    # are kept — the pre-budget behaviour).
    frame_budget: int | None = None
    depth_backend: str = "auto"
    frame_stride: int = 1
    max_depth_views: int = 200
    force: list[str] = field(default_factory=list)
    stereo: StereoParams | None = None
    refine: RefineParams | None = None
    gps: dict | None = None  # project-level WGS84 anchor {lat, lon, alt}
    # External telemetry CSV (timestamp,latitude,longitude,altitude) supplied
    # alongside the video. None => VIDEO_ONLY / embedded-only telemetry.
    telemetry_csv: str | None = None  # workspace-relative filename
    # Phase 7: auto-discovered PipelineStage plugins to run after the core
    # chain (e.g. "mesh_generation"). Empty = classic Phase 6 behaviour.
    plugins: list[str] = field(default_factory=list)


def request_params(request: PipelineRequest) -> dict:
    """The request fields that form a stage fingerprint.

    Single owner: the resume path and tooling must serialise EXACTLY this
    dict, or a resume cannot reproduce a stored fingerprint (which is what
    silently re-ran the sparse stage once).
    """
    return {
        "extraction_mode": request.extraction_mode,
        "every_n": request.every_n,
        "target_fps": request.target_fps,
        "top_percent": request.top_percent,
        "quality_threshold": request.quality_threshold,
        "frame_budget": request.frame_budget,
        "depth_backend": request.depth_backend,
        "frame_stride": request.frame_stride,
        "max_depth_views": request.max_depth_views,
    }


def run_autonomous_pipeline(job_id: str, request: PipelineRequest) -> dict:
    """Execute (or resume) the full pipeline for *job_id*."""
    workspace = settings.storage.project_dir(job_id)
    profiler = PerformanceProfiler(job_id, workspace)
    stages: dict[str, StageState] = {}
    start_wall = time.perf_counter()
    publish = lambda ev, payload: engine.publish(job_id, ev, payload)

    # Input identity gate (spec §5): the video about to be consumed must be
    # the exact uploaded file. Fails the run instead of silently proceeding.
    try:
        input_identity = _verify_input_identity(job_id, workspace)
    except FileNotFoundError:
        input_identity = {}
        log.warning("input_identity_skipped", job_id=job_id,
                    reason="no video in workspace yet (resume/rescue path)")

    # Validate requested plugin stages once, up front.
    from app.services.pipeline_stage import discover, get_stage

    discover()
    for p in request.plugins:
        get_stage(p)  # raises ValueError for unknown stage names
    names = STAGES + tuple(request.plugins)
    plugin_checks = {}
    for p in request.plugins:
        rel = get_stage(p).artifact_rel
        plugin_checks[p] = (lambda w, r=rel: (w / r).exists()) if rel else None

    params = request_params(request)

    def stage_publish(name: str, frac: float, detail: dict | None = None) -> None:
        st = stages[name]
        st.detail = {**(detail or {})}
        publish(
            f"stage:{name}",
            {"progress": round(frac, 3), "status": st.status, "duration_ms": round(st.duration_ms, 2), **st.detail},
        )

    def report_publish(report: dict) -> None:
        publish("pipeline_update", report)
        _write_report(workspace, report)

    # ------------------------------------------------------------ run stages
    try:
        for name in names:
            if engine.is_cancelled(job_id):
                return _finish(job_id, stages, start_wall, workspace, publish, report_publish, profiler=profiler, status="cancelled", input_identity=input_identity)
            stages[name] = StageState(name=name)
            check = ARTIFACT_CHECKS.get(name, plugin_checks.get(name))
            fingerprint = compute_stage_fingerprint(name, workspace, params)
            
            # Artifact check must pass AND (if fingerprint file exists, fingerprint must match)
            artifact_present = check is not None and check(workspace)
            fp_file = workspace / f".fingerprint_{name}.json"
            if fp_file.exists():
                cache_valid = artifact_present and is_stage_cache_valid(name, workspace, fingerprint, check)
            else:
                cache_valid = artifact_present

            cache_invalidation: dict | None = None
            if not cache_valid and artifact_present and name not in request.force:
                # Artifacts for this stage exist but will NOT be reused: record
                # and log the reason. Recomputing an expensive stage must never
                # be silent (a resume once re-ran a 15-minute sparse stage).
                cache_invalidation = describe_fingerprint_mismatch(name, workspace, params)
                log.warning("stage_recomputed", job_id=job_id, stage=name,
                            reason=cache_invalidation.get("reason"),
                            explanation=cache_invalidation)

            if name in request.force or not cache_valid:
                profiler.start_stage(name)
                try:
                    _run_stage(
                        job_id, name, request, workspace, stages[name],
                        publish, stage_publish, report_publish,
                    )
                    if cache_invalidation:
                        stages[name].detail = {**stages[name].detail,
                                               "cache_invalidation": cache_invalidation}
                    save_stage_fingerprint(name, workspace, fingerprint,
                                           params=params, details=stages[name].detail)
                    profiler.complete_stage(
                        name,
                        input_count=stages[name].detail.get("input_count", 0),
                        output_count=stages[name].count,
                        throughput_unit="views/sec" if name in ("frames", "depth") else "items/sec",
                        cache_hit=False,
                        cache_key=fingerprint,
                        details=stages[name].detail,
                    )
                except Exception as exc:
                    profiler.fail_stage(name, str(exc))
                    raise
            else:
                stages[name].status = "skipped"
                stages[name].detail = {"reason": "artifact_present", "cache_hit": True}
                if name == "sparse":
                    # A reused sparse stage still describes a telemetry placement
                    # the user must be able to see: a refused per-window
                    # correction completes the stage, and this detail is its
                    # only trace outside the sparse report on disk.
                    stages[name].detail.update(cached_sparse_detail(workspace))
                profiler.skip_stage(name, reason="artifact_present", cache_key=fingerprint)
                publish(f"stage:{name}", {"status": "skipped", "progress": 1.0, "reason": "artifact_present"})
        report = _finish(job_id, stages, start_wall, workspace, publish, report_publish, profiler=profiler, status="completed", input_identity=input_identity)
        return report
    except _Cancelled:
        return _finish(job_id, stages, start_wall, workspace, publish, report_publish, profiler=profiler, status="cancelled", input_identity=input_identity)
    except Exception as exc:
        log.exception("pipeline_failed", job_id=job_id, error=str(exc))
        return _finish(job_id, stages, start_wall, workspace, publish, report_publish, profiler=profiler, status="failed", error=str(exc), input_identity=input_identity)


def _run_stage(
    job_id: str,
    name: str,
    request: PipelineRequest,
    workspace: Path,
    state: StageState,
    publish,
    stage_publish,
    report_publish,
) -> None:
    """Execute one stage with retry + cancellation handling."""
    retries = max(0, settings.pipeline.stage_retries)
    state.status = "running"
    stage_publish(name, 0.0)  # announce the stage has started so live status works from stage 1
    attempts = 0
    while True:
        if engine.is_cancelled(job_id):
            state.status = "cancelled"
            raise _Cancelled()
        attempts += 1
        state.start_ms = time.perf_counter() * 1000
        try:
            if name == "frames":
                _stage_frames(job_id, request, workspace, state, stage_publish)
            elif name == "sparse":
                # Depth Anything's per-frame output is pose-independent, and
                # this stage (mapping + Ceres prior pass + conversion) holds
                # roughly one of four cores for several hundred seconds while
                # the depth stage waits on it. Compute the raw maps now; the
                # depth stage stops the prefetch and does anchoring only, on
                # the final post-BA poses, so every artifact stays identical.
                start_depth_prefetch(workspace)
                try:
                    _stage_sparse(job_id, workspace, state, stage_publish)
                except BaseException:
                    stop_depth_prefetch()
                    raise
            elif name == "depth":
                stop_depth_prefetch()
                _stage_depth(job_id, request, workspace, state, publish)
            elif name == "dense":
                _stage_dense(job_id, workspace, state)
            elif name == "georef":
                _stage_georef(job_id, request, workspace, state)
            else:
                _run_plugin_stage(job_id, name, workspace, state,
                                  publish, stage_publish, retries)
                if state.status == "skipped":
                    return  # stage cleanly not applicable — keep the skip
                state.status = "completed"
                state.end_ms = time.perf_counter() * 1000
                state.duration_ms = state.end_ms - state.start_ms
                publish(f"stage:{name}", {
                    "status": "completed", "progress": 1.0, "count": state.count,
                    "duration_ms": round(state.duration_ms, 2), "detail": state.detail,
                })
                return
            state.status = "completed"
            state.end_ms = time.perf_counter() * 1000
            state.duration_ms = state.end_ms - state.start_ms
            publish(f"stage:{name}", {
                "status": "completed", "progress": 1.0, "count": state.count,
                "duration_ms": round(state.duration_ms, 2), "detail": state.detail,
            })
            return
        except _Cancelled:
            publish(f"stage:{name}", {"status": "cancelled", "progress": 0.0})
            raise
        except Exception as exc:
            if attempts > retries:
                state.status = "failed"
                state.error = str(exc)
                state.end_ms = time.perf_counter() * 1000
                state.duration_ms = state.end_ms - state.start_ms
                publish(f"stage:{name}", {"status": "failed", "error": str(exc)})
                log.warning("stage_failed", stage=name, job_id=job_id, error=str(exc))
                raise
            log.warning("stage_retry", stage=name, job_id=job_id, attempt=attempts, error=str(exc))


def _run_plugin_stage(job_id, name, workspace, state, publish, stage_publish, retries):
    """Run one auto-discovered PipelineStage with retry/cancel/skip semantics.

    A StageNotApplicable marks the stage ``skipped`` (clean, non-fatal); any
    other error retries then fails (with rollback of this run's outputs).
    """
    from app.services.pipeline_stage import StageCancelled, StageNotApplicable, get_stage

    attempts = 0
    state.status = "running"
    while True:
        if engine.is_cancelled(job_id):
            state.status = "cancelled"
            publish(f"stage:{name}", {"status": "cancelled", "progress": 0.0})
            raise _Cancelled()
        attempts += 1
        state.start_ms = time.perf_counter() * 1000
        stage = get_stage(name)(
            job_id, workspace,
            publish=lambda ev, frac, payload: stage_publish(name, frac, payload),
            cancel_check=lambda: engine.is_cancelled(job_id),
        )
        try:
            stage.initialize()
            try:
                stage.validate_inputs()
                if stage.has_checkpoint():
                    stage.resume()
                stage.execute()
            except StageNotApplicable as exc:
                state.status = "skipped"
                state.end_ms = time.perf_counter() * 1000
                state.duration_ms = state.end_ms - state.start_ms
                state.detail = {"reason": "not_applicable", "detail": str(exc)}
                publish(f"stage:{name}", {"status": "skipped", "progress": 0.0,
                                           "reason": "not_applicable", "detail": str(exc)})
                log.info("plugin_stage_skipped", stage=name, job_id=job_id, reason=str(exc))
                return
            stage.checkpoint()
            summary = stage.summary()
            state.count = summary["count"]
            state.detail = {**summary["detail"], "outputs": summary["outputs"]}
            state.end_ms = time.perf_counter() * 1000
            state.duration_ms = state.end_ms - state.start_ms
            stage.publish_metrics({"duration_ms": round(state.duration_ms, 2),
                                   "count": state.count, "version": stage.version})
            stage.cleanup()
            publish(f"stage:{name}", {"status": "completed", "progress": 1.0,
                                       "count": state.count, "duration_ms": round(state.duration_ms, 2),
                                       "detail": state.detail})
            return
        except StageCancelled:
            state.status = "cancelled"
            publish(f"stage:{name}", {"status": "cancelled", "progress": 0.0})
            raise _Cancelled()
        except Exception as exc:
            try:
                stage.rollback()
            except Exception:
                pass
            if attempts > retries:
                state.status = "failed"
                state.error = str(exc)
                state.end_ms = time.perf_counter() * 1000
                state.duration_ms = state.end_ms - state.start_ms
                publish(f"stage:{name}", {"status": "failed", "error": str(exc)})
                log.warning("plugin_stage_failed", stage=name, job_id=job_id, error=str(exc))
                raise
            log.warning("plugin_stage_retry", stage=name, job_id=job_id,
                        attempt=attempts, error=str(exc))


def _stage_frames(
    job_id: str,
    request: PipelineRequest,
    workspace: Path,
    state: StageState,
    stage_publish=None,
) -> None:
    import cv2

    video_path = _video_path(workspace)
    # Explicit mode wins; else a caller-supplied rate implies "target_fps";
    # else the configured default ("every_n"). Without this, a payload that
    # sets only target_fps was silently ignored (mode defaulted to "every_n").
    mode = request.extraction_mode or (
        "target_fps" if request.target_fps else settings.pipeline.default_extraction_mode
    )
    every_n = request.every_n or settings.pipeline.default_every_n
    fps = request.target_fps or settings.pipeline.default_target_fps
    # Dataset calibration (converted from cameras.txt / copied intrinsics.json
    # at ingest): drives one-time undistortion at extraction so every later
    # stage sees pinhole geometry consistent with the K used for unprojection.
    calibration = None
    intr_file = workspace / "intrinsics.json"
    if intr_file.is_file():
        try:
            import json as _json

            calibration = _json.loads(intr_file.read_text())
        except (OSError, ValueError):
            calibration = None
    # Duration-aware keyframe budget: the candidate rate stays dense enough
    # to judge quality, but SfM/depth/dense only process the geometrically
    # diverse keyframes the budget keeps. Default budget scales with video
    # duration (keyframe_fps_budget keyframes/s of footage, default 1.3 —
    # benchmark-calibrated: gates pass at ~160 frames on the calibration run,
    # fail at 150), clamped to [60, 200] — bounded reconstruction cost per
    # run regardless of clip length. frame_budget=0 disables the budget
    # (pre-budget behaviour); an explicit positive value overrides the
    # duration default.
    frame_budget = request.frame_budget
    if frame_budget is None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        duration_s = n_video / fps_video if fps_video > 0 else 0.0
        frame_budget = int(min(200, max(60, round(duration_s * settings.pipeline.keyframe_fps_budget))))
    result = extract_frames(
        video_path,
        workspace,
        extraction_mode=mode,
        every_n=every_n,
        target_fps=fps,
        interval_sec=None,
        top_percent=request.top_percent,
        quality_threshold=request.quality_threshold,
        calibration=calibration,
        frame_budget=frame_budget or None,
        # Candidate decode is the frames stage's real cost; publish the
        # fraction onto the stage event so the Processing page moves during it.
        progress=(
            (lambda frac: stage_publish("frames", frac)) if stage_publish else None
        ),
    )
    state.count = result["selected_count"]
    state.detail = {
        "frame_budget": frame_budget,
        "frame_budget_applied": result.get("frame_budget_applied", False),
    }


def cached_sparse_detail(workspace: Path) -> dict:
    """Provenance a SKIPPED sparse stage can still report from its artifacts."""
    path = workspace / "reconstruction_report.json"
    if not path.is_file():
        return {}
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(report, dict):
        return {}
    out: dict = {"placement": placement_summary(report)}
    localization = report.get("localization")
    if isinstance(localization, dict) and localization:
        out["localization"] = localization
    return out


def placement_summary(report: dict) -> dict:
    """Placement provenance for the stage detail — what the UI must be able to say.

    A sparse stage can SUCCEED while its telemetry placement was degraded: the
    piecewise correction is refused when it cannot re-explain the observations,
    and the run continues on the rigid global similarity. The numbers belong in
    the stage detail so a degraded placement is visible without reading logs.
    """
    placement = ((report.get("trajectory_alignment") or {}).get("placement")) or {}
    refused = ((placement.get("piecewise") or {}).get("refused")) or None
    applied = placement.get("applied") or {}
    return {
        "mode": placement.get("placement_mode"),
        "matched_cameras": placement.get("cameras_matched"),
        "match_percent": placement.get("match_percent"),
        "points_retained_fraction": applied.get("points_retained_fraction"),
        "refused": refused,
    }


def _stage_sparse(job_id: str, workspace: Path, state: StageState, stage_publish) -> None:
    selected_dir = workspace / "selected" if (workspace / "selected").is_dir() else workspace / "frames"
    telemetry_csv = workspace / "telemetry.csv"
    pose_csv = workspace / "flight_poses.csv"
    if not pose_csv.is_file() and telemetry_csv.is_file() and _has_metric_pose_columns(telemetry_csv):
        pose_csv = telemetry_csv
    video_fps: float | None = None
    source_json = workspace / "source.json"
    if source_json.is_file():
        try:
            video_fps = float(json.loads(source_json.read_text()).get("fps")) or None
        except (OSError, ValueError):
            video_fps = None
    report = run_sparse_reconstruction(
        selected_dir, workspace, project_id=job_id,
        total_frames=0, selected_frames=state.count or count_image_files(selected_dir),
        flight_poses_csv=pose_csv,
        intrinsics_path=workspace / "intrinsics.json",
        telemetry_csv=telemetry_csv if telemetry_csv.is_file() else None,
        video_fps=video_fps,
        # Substage milestones land on the stage:sparse event the live status
        # endpoint reads (previously the longest stage reported 0% until done).
        progress=lambda block, frac: stage_publish("sparse", frac, {"substage": block}),
    )
    state.count = report["reconstruction"]["num_cameras"]
    state.detail = {
        "localization": report.get("localization", {}),
        "placement": placement_summary(report),
    }
    # Postcondition: a COMPLETED sparse stage must have produced its minimum
    # artifacts — registered cameras and the poses.json downstream consumes.
    if state.count <= 0 or not (workspace / "poses.json").exists():
        raise ValueError(
            f"SfM reported success without valid artifacts "
            f"(registered_cameras={state.count}, poses_json={(workspace / 'poses.json').exists()})"
        )


def _has_metric_pose_columns(path: Path) -> bool:
    """Return whether a telemetry CSV contains complete per-frame poses."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            fields = {field.strip().lower() for field in next(csv.reader(handle))}
    except (OSError, UnicodeError, StopIteration):
        return False
    return {"frame_id", "x", "y", "z", "qw", "qx", "qy", "qz"}.issubset(fields)


def _stage_depth(job_id: str, request: PipelineRequest, workspace: Path, state: StageState, publish) -> None:
    from app.services.depth_generator import DepthSummary

    summary: DepthSummary = DepthSummary()

    def cb(idx: int, total: int, frame_id: str, backend: str) -> bool:
        if engine.is_cancelled(job_id):
            return False
        frac = (idx + 1) / max(1, total)
        # The per-frame fraction must land on the stage event itself:
        # _live_status derives the UI's progress from stage:<name> history,
        # so a dedicated depth_progress event is never seen by the poller and
        # the Processing page sat at 0% for this entire ~40%-of-run stage.
        publish("stage:depth", {
            "status": "running", "progress": round(frac, 3),
            "frame": idx + 1, "total": total, "frame_id": frame_id,
            "backend": backend,
        })
        state.detail = {"frame": idx + 1, "total": total, "backend": backend}
        return True

    summary = generate_view_depths(
        workspace,
        backend=request.depth_backend,
        frame_stride=request.frame_stride,
        max_views=request.max_depth_views,
        stereo=request.stereo,
        refine=request.refine,
        progress=cb,
    )
    if engine.is_cancelled(job_id) and not summary.generated:
        state.status = "cancelled"
        raise _Cancelled()
    if not summary.generated and not summary.cached:
        # Never report this without the measured cause: the dominant
        # per-view reason is the whole diagnosis for the operator. Exclusion
        # reasons count too — a run whose every view was REFUSED by name
        # (60/60 on London_Mission_ca3c5b) previously reported
        # "dominant_failure=none recorded", hiding the entire diagnosis.
        counts: dict[str, int] = {}
        for reason in summary.failure_reasons.values():
            counts[reason.split(":", 1)[0]] = counts.get(reason.split(":", 1)[0], 0) + 1
        for reason in summary.excluded_reasons.values():
            counts[reason] = counts.get(reason, 0) + 1
        dominant = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        example = next(
            iter(summary.failure_reasons.values()),
            next(iter(summary.excluded_reasons.values()), None),
        )
        raise ValueError(
            "no depth maps could be generated — "
            f"failed={len(summary.failed)} "
            f"excluded={len(summary.excluded) or len(summary.excluded_reasons)} "
            f"dominant_reason={dominant or 'none recorded'} "
            f"example={example}"
        )
    state.count = len(summary.generated) + len(summary.cached)
    state.detail = summary.to_dict()
    publish("depth_complete", summary.to_dict())


def _stage_dense(job_id: str, workspace: Path, state: StageState) -> None:
    from app.services.dense_reconstruction import DenseParams

    params = DenseParams.from_settings()
    # Scene-scale awareness: the defaults fit near-field clips. A metric
    # reconstruction whose cameras sit hundreds of metres from the scene
    # (oblique aerial survey) needs proportionally looser fusion parameters,
    # derived from the recorded geometry — never hardcoded per-dataset.
    try:
        import json as _json

        poses_path = workspace / "poses.json"
        if poses_path.is_file():
            frames = _json.loads(poses_path.read_text()).get("frames", [])
            if frames:
                centers = np.array([f["t"] for f in frames], dtype=np.float64)
                path_span = float(np.linalg.norm(centers.max(0) - centers.min(0)))
                # Scene distance, not path span, drives scale: a nadir flight
                # moves little but the ground sits ~altitude below every
                # camera (airport3: 29 m path, 178 m scene). The altitude
                # term is only meaningful for telemetry runs (ENU, z-up);
                # video-only SfM frames are arbitrary, so there path span
                # remains the sole signal.
                scene_dist = path_span
                try:
                    loc = _json.loads((workspace / "reconstruction_report.json").read_text()).get("localization", {})
                    if loc.get("mode") == "telemetry_assisted":
                        scene_dist = max(scene_dist, float(np.median(centers[:, 2])))
                except (OSError, ValueError):
                    pass
                if scene_dist > 50:  # far-field scene
                    params.voxel_size = max(params.voxel_size, round(scene_dist / 500.0, 2))
                    params.max_depth_m = max(params.max_depth_m, scene_dist * 2.0)
                    params.ror_radius_m = max(params.ror_radius_m, scene_dist / 50.0)
    except Exception as exc:  # fall back to configured defaults, honestly
        log.warning("dense_scale_adaptation_failed", error=str(exc))

    report = run_dense_reconstruction(job_id, params)
    quality = report.get("quality", {})
    state.count = quality.get("point_count", 0)
    state.detail = {"dense_score": quality.get("dense_score"), "grade": quality.get("grade"),
                    "voxel_size": params.voxel_size, "max_depth_m": params.max_depth_m}
    # Textured-GLB facts for the Processing page: textured / fallback reason /
    # views used. The field map lives in mesh_quality so the offline rebuild
    # path writes exactly these fields from the same LOD stats.
    lod = report.get("stages", {}).get("viewer_lod", {})
    if lod:
        from app.services.mesh_quality import viewer_glb_detail

        state.detail["viewer_glb"] = viewer_glb_detail(lod)
    # Every timed dense substage, by name. A slow dense run has to be
    # attributable from performance.json alone: the post-fusion chain (mesh
    # build, mesh audit, texturing, LOD) once ran ~692 s with no substage
    # duration recorded anywhere, so the stage total jumped and nothing named
    # the cause. Payload-only entries (mesh quality, viewer LOD stats) carry no
    # duration and are skipped, never invented.
    substages = {
        name: float(det["duration_ms"])
        for name, det in report.get("stages", {}).items()
        if isinstance(det, dict) and det.get("duration_ms") is not None
    }
    if substages:
        state.detail["substages_ms"] = substages
    _write_combined_model(workspace)


def _write_combined_model(workspace: Path) -> None:
    """Write ``combined_model.ply`` — dense surface + sparse scaffold, one frame.

    Both artifacts already live in the reconstruction's world frame (the dense
    stage consumes poses.json, the sparse stage wrote it), so no alignment is
    performed — the merge is a pure concatenation. Sparse points carry
    ``confidence=1.0`` (they passed the reprojection + depth-uncertainty
    gates) and their observing-view count; residuals are NOT merged because
    the two stages use different units. Missing inputs skip the artifact
    rather than failing the run.
    """
    from app.services.pointcloud import PointCloud, read_ply, save_ply

    sparse_path = workspace / "sparse_model.ply"
    dense_path = workspace / "dense" / "dense_model.ply"
    out_path = workspace / "combined_model.ply"
    if not (sparse_path.is_file() and dense_path.is_file()):
        return
    if out_path.exists():
        # Stale-output guard: the combined file is a pure concatenation of its
        # two inputs, so an existing file is valid only while BOTH inputs are
        # older than it. A dense rerun (or artifact restore) that lands a
        # newer dense cloud must not leave the viewer serving the old merge —
        # the default point-cloud view reads THIS file.
        if (out_path.stat().st_mtime > sparse_path.stat().st_mtime
                and out_path.stat().st_mtime > dense_path.stat().st_mtime):
            return
        log.info("combined_model_stale_regenerating",
                 out_mtime=out_path.stat().st_mtime,
                 sparse_mtime=sparse_path.stat().st_mtime,
                 dense_mtime=dense_path.stat().st_mtime)
    try:
        sparse = read_ply(sparse_path)
        dense = read_ply(dense_path)
        conf = np.concatenate([
            np.ones(sparse.n),
            dense.confidence if dense.confidence is not None else np.full(dense.n, 0.5),
        ])
        obs = None
        if sparse.observations is not None or dense.observations is not None:
            obs = np.concatenate([
                sparse.observations if sparse.observations is not None else np.ones(sparse.n, dtype=np.int32),
                dense.observations if dense.observations is not None else np.ones(dense.n, dtype=np.int32),
            ])
        combined = PointCloud(
            xyz=np.concatenate([sparse.xyz, dense.xyz]),
            rgb=np.concatenate([sparse.ensure_colors(), dense.ensure_colors()]),
            confidence=conf,
            observations=obs,
            meta={
                "combined_from": ["sparse_model.ply", "dense/dense_model.ply"],
                "sparse_points": int(sparse.n),
                "dense_points": int(dense.n),
                "note": "single world frame; no alignment applied",
            },
        )
        save_ply(out_path, combined)
        log.info("combined_model_written", job_id=workspace.name,
                 points=combined.n, sparse=sparse.n, dense=dense.n)
    except Exception as exc:
        log.warning("combined_model_failed", job_id=workspace.name, error=str(exc))


def _stage_georef(job_id: str, request: PipelineRequest, workspace: Path, state: StageState) -> None:
    """Georeference outputs only when a sufficient GPS track exists.

    Three-mode telemetry model (see app.services.telemetry): video-only runs
    complete honestly with GPS reported unavailable; telemetry-bearing runs
    must pass a sufficiency check (≥3 distinct matched samples spanning a
    trajectory) before any georeferenced artifact is written. Coordinates are
    never silently converted from the local SfM frame to geographic ones.
    """
    import csv

    from app.services.telemetry import (
        MIN_GEOFERENCE_SAMPLES,
        TelemetryError,
        TelemetryMode,
        format_telemetry_detection,
        load_telemetry_with_schema,
        resolve_mode,
        synchronize,
    )

    poses_path = workspace / "poses.json"
    pose_frames = json.loads(poses_path.read_text()).get("frames", []) if poses_path.exists() else []
    embedded = [f["gps"] for f in pose_frames if f.get("gps")]
    anchor = request.gps
    result: dict = {
        "crs": None,
        "gps_quality": None,
        "alignment": None,
        "sync": None,
        "note": "",
    }

    # ---- classify mode + synchronize external telemetry with frames ------
    telemetry_csv_path: Path | None = None
    if request.telemetry_csv:
        candidate = workspace / request.telemetry_csv
        if candidate.is_file():
            telemetry_csv_path = candidate
        else:
            result["note"] = (
                f"external telemetry file '{request.telemetry_csv}' not found — "
                "treated as video-only"
            )
    elif (workspace / "telemetry.csv").is_file():
        # Auto-discovery parity with the sparse stage (which always checks
        # ``workspace/telemetry.csv``): a telemetry file in the workspace is
        # the run's telemetry whether or not THIS request repeated it — the
        # UI's retry/resume sends an empty request, and a run whose upload
        # attached telemetry must not degrade to video-only georeferencing
        # because of that (measured: airport1 retry dropped telemetry_csv
        # from the queue payload, so georef wrote an honest-but-wrong
        # VIDEO_ONLY detail while sparse was telemetry-placed).
        telemetry_csv_path = workspace / "telemetry.csv"
    embedded_gps = anchor or (embedded[0] if embedded else None)
    mode = resolve_mode(embedded_gps=embedded_gps, telemetry_csv_path=telemetry_csv_path)

    sync_report = None
    if telemetry_csv_path is not None:
        frame_timestamps = _frame_timestamps(workspace)
        try:
            samples, schema_report = load_telemetry_with_schema(telemetry_csv_path)
            # One-per-video-frame logs without a timestamp column: derive the
            # time base from frame numbers (same derivation/owner as the
            # sparse trajectory path — the two must never disagree).
            if samples and all(s.timestamp_sec is None for s in samples):
                from app.services.telemetry import derive_timestamps_from_frame_numbers
                vfps = vdur = None
                try:
                    src = json.loads((workspace / "source.json").read_text())
                    vfps = float(src.get("fps")) or None
                    vdur = float(src.get("duration_sec")) or None
                except (OSError, ValueError, TypeError):
                    pass
                samples, note = derive_timestamps_from_frame_numbers(
                    samples, vfps or 0.0, video_duration_sec=vdur)
                if note and all(s.timestamp_sec is not None for s in samples):
                    result["timestamp_derivation"] = note
        except TelemetryError as exc:
            # Genuinely undetectable telemetry must not fail the run and must
            # not produce georef artifacts — fall through honestly as
            # video-only, with the detection diagnostics in the note.
            result["note"] = f"external telemetry unusable ({exc}) — outputs stay in local coordinates"
            result["telemetry_mode"] = TelemetryMode.VIDEO_ONLY.value
            state.detail = result
            state.status = "completed"
            log.warning("georef_telemetry_rejected", job_id=job_id, error=str(exc))
            return
        # Schema artifacts (telemetry_normalized.csv / telemetry_schema.json /
        # telemetry_quality.json) are written next to telemetry.csv by
        # load_telemetry_with_schema — the workspace holds them (§12/§16).
        result["telemetry_artifacts"] = schema_report.get("artifacts") or {}
        result["telemetry_detection"] = {
            "fields": schema_report["schema"].get("fields"),
            "capabilities": schema_report.get("capabilities"),
            "delimiter": schema_report["schema"].get("detected_delimiter"),
            "header_row": schema_report["schema"].get("header_row"),
            "timestamp_interpretation": schema_report["schema"].get("timestamp_interpretation"),
            "mapping_confidence": schema_report.get("quality", {}).get("mapping_confidence"),
        }
        result["telemetry_detection_summary"] = format_telemetry_detection(schema_report)
        sync_report = synchronize(frame_timestamps, samples)
        result["sync"] = sync_report.to_dict()

        if not sync_report.has_sufficient_track:
            reason = (
                f"only {sync_report.matched_frames} matched frame(s) with valid GPS "
                f"(need ≥{MIN_GEOFERENCE_SAMPLES} spanning a trajectory)"
                if sync_report.gps_available
                else "no telemetry sample carried valid GPS values"
            )
            result["note"] = (
                f"GPS telemetry present but insufficient for georeferencing: {reason} — "
                "outputs stay in local coordinates"
            )
            result["telemetry_mode"] = sync_report.mode
            state.detail = result
            state.status = "completed"
            log.info("georef_insufficient_telemetry", job_id=job_id,
                     matched=sync_report.matched_frames, samples=sync_report.telemetry_samples)
            return

        # Sufficient external track: matched frame order defines GPS points
        # (matched_samples mirrors matched_frame_ids by construction).
        by_frame = dict(zip(sync_report.matched_frame_ids, sync_report.matched_samples))
        gps_points = [
            {"lat": by_frame[f["frame_id"]].latitude,
             "lon": by_frame[f["frame_id"]].longitude,
             "alt": by_frame[f["frame_id"]].altitude_m if by_frame[f["frame_id"]].altitude_m is not None else 0.0}
            for f in pose_frames if f.get("frame_id") in by_frame
        ]
        if not gps_points:
            gps_points = filter_valid_gps_points(embedded)
    else:
        # Embedded / project-anchor path (existing behaviour).
        raw_gps = list(embedded)
        if not raw_gps and anchor:
            raw_gps = [anchor]  # project-level fix (e.g. container telemetry)
        gps_points = filter_valid_gps_points(raw_gps)

    if not gps_points:
        missing = result.get("note")
        result["note"] = (
            f"{missing} — no GPS telemetry available; georeferencing skipped "
            "(outputs stay in local coordinates)" if missing else
            "no GPS telemetry available — georeferencing skipped (outputs stay in local coordinates)"
        )
        result["telemetry_mode"] = resolve_mode(embedded_gps=None, telemetry_csv_path=None).value
        state.detail = result
        state.status = "completed"
        return

    georef_dir = workspace / "georef"
    georef_dir.mkdir(parents=True, exist_ok=True)
    state.count = len(gps_points)

    # Fewer than two DISTINCT fixes is not a trajectory: there is nothing to
    # compare the reconstruction against, so no track is written and no
    # alignment is attempted (fixes are recorded in the report only). This
    # keeps video-only-style honesty for degenerate telemetry inputs.
    if len({(round(p["lat"], 7), round(p["lon"], 7)) for p in gps_points}) < 2:
        result["gps_quality"] = analyze_gps_track(
            np.array([gps_points[0]["lat"]]),
            np.array([gps_points[0]["lon"]]),
            np.array([float(gps_points[0].get("alt") or 0.0)]),
        ).to_dict()
        result["crs"] = crs_metadata(
            float(gps_points[0]["lat"]),
            float(gps_points[0]["lon"]),
            float(gps_points[0].get("alt") or 0.0),
        )
        result["note"] = (
            "GPS fixes do not span a trajectory (single or identical fixes): "
            "insufficient trajectory information — outputs stay in local coordinates "
            "(fixes recorded only)"
        )
        result["telemetry_mode"] = (
            sync_report.mode if sync_report else _embedded_mode(embedded or ([anchor] if anchor else []))
        )
        with open(georef_dir / "gps_report.json", "w") as f:
            json.dump(result, f, indent=2)
        state.detail = result
        state.status = "completed"
        log.info("georef_degenerate_track", job_id=job_id, points=len(gps_points))
        return

    lats = np.array([p["lat"] for p in gps_points])
    lons = np.array([p["lon"] for p in gps_points])
    alts = np.array([float(p.get("alt", 0.0)) for p in gps_points])
    quality = analyze_gps_track(lats, lons, alts)
    crs = crs_metadata(float(lats[0]), float(lons[0]), float(alts[0]))

    # GPS track CSV + ENU positions.
    enu = wgs84_to_enu(lats, lons, alts, float(lats[0]), float(lons[0]), float(alts[0]))
    with open(georef_dir / "gps_track.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "lat", "lon", "alt_m", "east_m", "north_m", "up_m"])
        for i in range(len(lats)):
            writer.writerow([i, round(float(lats[i]), 7), round(float(lons[i]), 7), round(float(alts[i]), 3),
                             *[round(float(v), 3) for v in enu[i]]])

    # Rigid/similarity alignment of the reconstruction to ENU when every
    # registered camera has a GPS fix AND the fixes span a real trajectory
    # (≥2 *distinct* positions — identical fixes are degenerate and cannot
    # constrain rotation). A single anchor point can never align: alignment
    # stays None and the note says so honestly.
    alignment = None
    fixes_span_trajectory = len({(p["lat"], p["lon"]) for p in gps_points}) >= 2
    if (
        pose_frames
        and len(pose_frames) == len(gps_points)
        and len(pose_frames) >= 2
        and fixes_span_trajectory
    ):
        source = np.array([np.asarray(f["t"], dtype=np.float64) for f in pose_frames])
        transform, scale = align_to_enu(source, enu)
        alignment = {"scale": round(float(scale), 6), "matrix": transform.tolist(),
                     "applied_to": []}
        _apply_alignment(workspace, georef_dir, transform, alignment)
        result["alignment"] = alignment
    elif not fixes_span_trajectory:
        result["note"] = (
            "GPS fixes do not span a trajectory (identical or single fix): "
            "alignment not possible — outputs stay in local coordinates"
        )

    result["gps_quality"] = quality.to_dict()
    result["crs"] = crs
    result["telemetry_mode"] = (
        sync_report.mode if sync_report else _embedded_mode(embedded or (anchor and [anchor] or []))
    )
    with open(georef_dir / "gps_report.json", "w") as f:
        json.dump(result, f, indent=2)
    state.detail = result
    state.status = "completed"
    log.info("georef_stage_done", job_id=job_id, score=quality.gps_score,
             grade=quality.grade, aligned=bool(alignment))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _frame_timestamps(workspace: Path) -> list[tuple[str, float]]:
    """(frame_id, timestamp_sec) for kept frames, from the extraction report."""
    report_path = workspace / "quality_report.json"
    if not report_path.is_file():
        return []
    try:
        frames = json.loads(report_path.read_text()).get("frames", [])
    except (OSError, json.JSONDecodeError):
        return []
    return [
        (str(f["filename"]).removesuffix(".jpg"), float(f["timestamp_sec"]))
        for f in frames
        if f.get("kept") and f.get("filename") and f.get("timestamp_sec") is not None
    ]


def _embedded_mode(embedded: list | None) -> str:
    from app.services.telemetry import TelemetryMode

    return (
        TelemetryMode.VIDEO_WITH_EMBEDDED_TELEMETRY.value
        if embedded else TelemetryMode.VIDEO_ONLY.value
    )


def _sparse_localization_mode(report: dict, workspace) -> str | None:
    """Localization mode from the sparse stage, wherever it was recorded.

    A cache-hit sparse stage reports only ``{reason, cache_hit}`` in the
    pipeline report, so fall back to the sparse stage's own on-disk report —
    otherwise a telemetry-placed run would be re-labelled VIDEO_ONLY.
    """
    mode = (report.get("stages", {}).get("sparse", {}).get("detail", {})
            .get("localization", {}) or {}).get("mode")
    if mode:
        return mode
    for name in ("sparse_rerun_report.json", "reconstruction_report.json"):
        try:
            data = json.loads((workspace / name).read_text())
        except (OSError, ValueError):
            continue
        mode = (data.get("localization", {}) or {}).get("mode")
        if mode:
            return mode
    return None


class _Cancelled(Exception):
    pass


def _finish(
    job_id: str,
    stages: dict[str, StageState],
    start_wall: float,
    workspace: Path,
    publish,
    report_publish,
    profiler: PerformanceProfiler | None = None,
    *,
    status: str,
    error: str = "",
    input_identity: dict | None = None,
) -> dict:
    total_ms = (time.perf_counter() - start_wall) * 1000
    profile = _profile(stages, total_ms)
    perf_data = {}
    if profiler:
        perf_data = profiler.finalize_and_save(status=status, error=error)

    report = {
        "job_id": job_id,
        "status": status,
        "error": error,
        "run_time_ms": round(total_ms, 2),
        "input_identity": input_identity or {},
        # SIH compliance: this pipeline reconstructs from ONE continuous
        # input sequence (uploaded video or single dataset pass). The marker
        # makes the single-pass contract explicit and auditable per run.
        "single_pass_enforced": True,
        "input_sequence_count": 1,
        "stages": {name: st.to_dict() for name, st in stages.items()},
        "profile": profile,
        "performance": perf_data,
        "resume": {
            stage: ARTIFACT_CHECKS.get(stage, lambda _w: False)(workspace) for stage in STAGES
        },
    }
    # Metric validation for EVERY completed run: the no-reference engine path
    # measures relative consistency + trajectory agreement and persists through
    # the artifact's single writer. Best-effort — a validation failure must
    # never fail an otherwise completed reconstruction; the Reports page then
    # honestly shows "no report" instead of a broken one.
    if status == "completed":
        try:
            from app.services.metric_validation import validate_run_internal

            validate_run_internal(workspace, run_id=job_id, persist=True)
        except Exception as mv_exc:
            log.warning("metric_validation_failed", job_id=job_id, error=str(mv_exc))
    report_publish(report)
    return report


def _profile(stages: dict[str, StageState], total_ms: float) -> dict:
    durations = {n: s.duration_ms for n, s in stages.items() if s.status == "completed"}
    bottleneck = max(durations, key=durations.get) if durations else ""
    return {
        "total_ms": round(total_ms, 2),
        "stage_durations_ms": {k: round(v, 2) for k, v in durations.items()},
        "bottleneck_stage": bottleneck,
        "stage_count": len(durations),
    }


def _write_report(workspace: Path, report: dict) -> None:
    try:
        with open(workspace / "pipeline_report.json", "w") as f:
            json.dump(report, f, indent=2)
        video_name = ""
        try:
            video_name = _video_path(workspace).name
        except Exception:
            pass
        from app.services.provenance import manifest_source

        # The dependency record is measured at run start (real environment
        # facts: torch/pycolmap versions, GPU presence, checkpoint) so the
        # run-level capability table shows this run's actual evidence.
        try:
            from app.services.phase95_validator import _audit_environment

            dependencies = _audit_environment()
        except Exception:
            dependencies = {}

        manifest = {
            "run_id": report.get("job_id"),
            "dataset": video_name or report.get("job_id"),
            "mission": f"Mission {report.get('job_id')}",
            "status": report.get("status"),
            "pipeline_version": "0.1.0",
            "source": manifest_source(workspace),
            "input_identity": report.get("input_identity"),
            "stages": report.get("stages", {}),
            "metrics": report.get("profile", {}),
            "dependencies": dependencies,
            "telemetry": report.get("stages", {}).get("georef", {}).get("detail", {}).get("sync")
            or (
                # Sparse may have been telemetry-placed even when georef never
                # ran (sparse failure). VIDEO_ONLY would then be a lie.
                {"mode": "VIDEO_WITH_EXTERNAL_TELEMETRY"}
                if _sparse_localization_mode(report, workspace) == "telemetry_assisted"
                else {"mode": report.get("stages", {}).get("georef", {})
                      .get("detail", {}).get("telemetry_mode", "VIDEO_ONLY")}
            ),
        }
        with open(workspace / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        # A run just changed on disk — drop the discovery cache so the UI
        # sees it without waiting out the TTL.
        from app.services.run_service import invalidate_runs_cache

        invalidate_runs_cache()
    except OSError:
        pass


def _video_path(workspace: Path) -> Path:
    # Case-insensitive extension match: DJI cameras write .MP4 (Flight_to_tower
    # ships video.MP4, which a lowercase-only glob silently missed).
    wanted = {".mp4", ".mov", ".avi", ".mkv"}
    candidates = sorted(p for p in workspace.iterdir() if p.suffix.lower() in wanted)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"no video file found in {workspace}")


def _verify_input_identity(job_id: str, workspace: Path) -> dict:
    """Verify the pipeline consumes exactly the uploaded video (spec §5).

    Compares the video the stages will actually read (``_video_path``) with
    the upload-time provenance record (``source.json``: filename, size,
    sha256). Any mismatch is a hard error — the pipeline must never silently
    process a different file than the user uploaded. Returns the verified
    record for logging/reporting.
    """
    video_path = _video_path(workspace)
    size = video_path.stat().st_size
    identity: dict = {
        "input_video": video_path.name,
        "input_video_size_bytes": size,
        "input_video_sha256": None,
    }
    source_file = workspace / "source.json"
    if not source_file.exists():
        log.warning(
            "input_identity_unverified",
            job_id=job_id,
            input_video=video_path.name,
            reason="no upload-time provenance record",
        )
        return identity
    try:
        record = json.loads(source_file.read_text())
    except (OSError, ValueError):
        log.warning("input_identity_unverified", job_id=job_id, reason="unreadable provenance record")
        return identity

    expected_name = record.get("original_filename")
    expected_size = record.get("file_size_bytes")
    expected_sha = record.get("sha256")

    from app.services.provenance import sha256_of

    actual_sha = sha256_of(video_path)
    identity["input_video_sha256"] = actual_sha

    mismatches = []
    if expected_name:
        # Upload storage sanitizes the client filename (spaces → underscores
        # etc.), so compare against the same sanitization, not the raw name.
        from app.services.upload_service import safe_filename

        try:
            expected_stored = safe_filename(expected_name)
        except Exception:
            expected_stored = expected_name
        if video_path.name != expected_stored:
            mismatches.append(f"filename {video_path.name!r} != uploaded {expected_stored!r}")
    if expected_size is not None and size != expected_size:
        mismatches.append(f"size {size} != uploaded {expected_size}")
    if expected_sha and actual_sha != expected_sha:
        mismatches.append("sha256 mismatch")
    if mismatches:
        raise ValueError(
            "INPUT VIDEO IDENTITY CHECK FAILED for job "
            f"{job_id}: " + "; ".join(mismatches) +
            " — the pipeline would not process the user's uploaded video."
        )
    log.info(
        "INPUT VIDEO",
        job_id=job_id,
        input_video=video_path.name,
        size_bytes=size,
        sha256=actual_sha[:16] + "…",
        kind=record.get("kind", "unknown"),
    )
    return identity


def _apply_alignment(workspace: Path, georef_dir: Path, transform: np.ndarray, alignment: dict) -> None:
    """Write ENU-aligned copies of the sparse/dense models when they exist."""
    for name, src_name, out_name in (
        ("sparse", "sparse_model.ply", "sparse_model_enu.ply"),
        ("dense", "dense/dense_model.ply", "dense_model_enu.ply"),
        ("combined", "combined_model.ply", "combined_model_enu.ply"),
    ):
        src = workspace / src_name
        if not src.exists():
            continue
        cloud = read_ply(src)
        cloud.xyz = (np.hstack([cloud.xyz, np.ones((cloud.n, 1))]) @ transform.T)[:, :3]
        save_ply(georef_dir / out_name, cloud)
        alignment["applied_to"].append(name)
    with open(georef_dir / "alignment.json", "w") as f:
        json.dump(alignment, f, indent=2)
