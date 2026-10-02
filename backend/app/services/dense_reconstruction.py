"""Dense reconstruction orchestrator.

Runs the Phase 6 pipeline for one job workspace::

    uploads/<job_id>/
        selected/ | frames/       images of the registered views
        depth/<frame_id>.npy      per-view depth maps (float32 meters)
        poses.json                per-view intrinsics + extrinsics + GPS
    ─────────────────────────────────────────────────────────────
        dense/dense_model.ply     cleaned, normalised dense cloud
        dense_report.json         full run report (quality, twin, exports)

Input contract
--------------
Depth maps are produced by a multi-view stereo engine (COLMAP
``patch_match_stereo`` when the binary is available — see the ``colmap``
config group — or an external depth estimator such as Depth Anything).
``poses.json`` is the sparse reconstruction output with one entry per
registered view:

.. code-block:: json

    {"frames": [{"frame_id": "frame_000000", "K": [[...]], "R": [[...]],
                 "t": [...], "gps": {"lat": .., "lon": .., "alt": ..}}]}

Pose convention: ``X_world = R @ X_cam + t``, ``K`` is the pinhole matrix.
If either depth maps or poses are missing the pipeline fails with a clear
message rather than fabricating data.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.depth_fusion import DepthView, FusionParams, fuse_depth_views
from app.services.digital_twin import DigitalTwin
from app.services.point_statistics import analyze_dense_quality
from app.services.pointcloud import PointCloud, export_cloud, read_ply, save_ply
from app.services.pointcloud_optimizer import OptimizeParams, optimize_cloud
from app.services.streaming_engine import engine

log = get_logger("drone_recon.services.dense_reconstruction")


@dataclass
class DenseParams:
    """Runtime overrides for one dense run (defaults from settings.dense)."""

    voxel_size: float = 0.05
    sor_k: int = 20
    sor_std_ratio: float = 2.0
    ror_radius_m: float = 0.4
    ror_min_neighbors: int = 5
    normal_k: int = 20
    min_confidence: float = 0.05
    max_points_per_view: int = 1_000_000
    max_depth_m: float = 200.0
    min_depth_m: float = 0.2

    @classmethod
    def from_settings(cls) -> DenseParams:
        d = settings.dense
        return cls(
            voxel_size=d.voxel_size,
            sor_k=d.sor_k,
            sor_std_ratio=d.sor_std_ratio,
            ror_radius_m=d.ror_radius_m,
            ror_min_neighbors=d.ror_min_neighbors,
            normal_k=d.normal_k,
            min_confidence=d.min_confidence,
            max_points_per_view=d.max_points_per_view,
            max_depth_m=d.max_depth_m,
            min_depth_m=d.min_depth_m,
        )


def _poisson_depth_for_fusion(cloud: PointCloud, voxel_size: float) -> int:
    """Choose the finest Poisson grid justified by the fused measurement scale."""
    extent = float(np.max(cloud.xyz.max(axis=0) - cloud.xyz.min(axis=0)))
    target_cell = max(0.05, 1.5 * float(voxel_size))
    if extent <= target_cell:
        return 6
    depth = int(math.ceil(math.log2(extent / target_cell)))
    return max(6, min(int(settings.mesh.poisson_depth), depth))


@dataclass
class DenseReport:
    """Aggregated dense run result, serialisable to JSON."""

    job_id: str = ""
    status: str = "running"
    stages: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    twin: dict = field(default_factory=dict)
    exports: list[dict] = field(default_factory=list)
    run_time_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stages": self.stages,
            "quality": self.quality,
            "twin": self.twin,
            "exports": self.exports,
            "run_time_ms": round(self.run_time_ms, 2),
            "error": self.error,
        }


def run_dense_reconstruction(job_id: str, params: DenseParams | None = None) -> dict:
    """Run the full dense pipeline for *job_id*. Returns the report dict."""
    start = time.perf_counter()
    params = params or DenseParams.from_settings()
    workspace = settings.storage.project_dir(job_id)
    report = DenseReport(job_id=job_id)
    publish = _make_publisher(job_id)

    #: Every substage is timed. This stage once took 579 s with NO substage
    #: duration recorded anywhere, so the largest cost in the pipeline was
    #: unattributable from its own report; the cheap fix is to measure each
    #: step against the previous one.
    _last_stage_t = start

    def stage(name: str, frac: float, payload: dict | None = None) -> None:
        nonlocal _last_stage_t
        now = time.perf_counter()
        report.stages[name] = {
            "progress": round(frac, 3),
            "duration_ms": round((now - _last_stage_t) * 1000.0, 2),
            "since_start_ms": round((now - start) * 1000.0, 2),
            **(payload or {}),
        }
        _last_stage_t = now
        # Substage detail keeps its own event (test_dense pins stage:depth_fusion);
        # the fraction ALSO lands on stage:dense, which is what the live status
        # endpoint and the Processing page read. Without it the dense stage —
        # meshing, texturing, LOD — reported 0% for its entire run.
        publish(f"stage:{name}", {**(payload or {}), "progress": frac})
        publish(
            "stage:dense",
            {"status": "running", "progress": frac, "substage": name, **(payload or {})},
        )

    try:
        publish("started", {"voxel_size": params.voxel_size})
        stage("load", 0.05)

        # --- locate inputs ---------------------------------------------------
        image_dir = _first_existing(workspace / "selected", workspace / "frames")
        depth_dir = _first_existing(workspace / "depth")
        poses_path = workspace / "poses.json"
        if depth_dir is None:
            # Report the path actually resolved for THIS run instead of a
            # hardcoded uploads/ template that may not be the storage root.
            _fail(report, publish,
                  f"No depth maps found — expected {workspace / 'depth'}/<frame>.npy "
                  "(run the depth stage first)")
        if not poses_path.exists():
            _fail(report, publish,
                  f"No poses.json found — expected {poses_path} from sparse reconstruction")

        poses = _load_poses(poses_path)
        if not poses:
            _fail(report, publish, "poses.json contains no registered frames")

        # --- depth fusion ----------------------------------------------------
        stage("depth_fusion", 0.15, {"views": len(poses), "gps_views": sum(1 for p in poses if p.get("gps"))})
        views = _build_views(poses, depth_dir, image_dir, params)
        # Stale-map guard: a map on disk written before the sparse-anchor gate
        # existed fails today's standard. The gate function the depth stage
        # applies to its live fits and its cached maps decides here too — one
        # policy owner, shared with the offline mesh rebuild so its cross-view
        # audit measures the same view set fusion used. Refused maps never
        # fuse; the exclusion is named in the report rather than silently
        # shrinking the view count.
        sparse_path_gate = workspace / "sparse_model.ply"
        sparse_xyz_gate = read_ply(sparse_path_gate).xyz if sparse_path_gate.is_file() else None
        views, gate_refused = _apply_anchor_gate(
            views, depth_dir, sparse_xyz_gate, poses)
        if gate_refused:
            report.stages["anchor_gate_exclusions"] = {
                "refused_views": gate_refused,
                "count": len(gate_refused),
                "note": "maps failing the sparse-anchor accuracy gate are excluded from fusion",
            }
        if not views:
            _fail(report, publish,
                  "None of the registered views have a matching depth map in the depth directory")

        # --- scene-adaptive depth ceiling -------------------------------------
        # The configured ceiling (settings.dense.max_depth_m, 200 m) is a
        # NEAR-FIELD default: far-field flights (airport1: scene 500-970 m)
        # zero every unprojected measurement — the audit's cross-view check
        # finds no comparable pairs (NOT_EVALUATED → G fails) and fusion
        # would fuse nothing. Derive the ceiling from the artifacts
        # themselves: the deepest valid depth the kept maps carry, plus
        # headroom for their far tail. NEVER shrinks the configured value —
        # near-field runs keep their explicit ceiling; adaptation only
        # extends it for scenes beyond it.
        effective_max_depth = scene_depth_ceiling(
            views, params.max_depth_m,
            sparse_xyz=sparse_xyz_gate, poses=poses,
        )
        if effective_max_depth > params.max_depth_m:
            log.info("fusion_depth_ceiling_adapted",
                     configured_m=params.max_depth_m,
                     effective_m=round(effective_max_depth, 1),
                     note="far-field scene — ceiling derived from the depth maps themselves")

        # --- honest fusion resolution -----------------------------------------
        # Measure how far adjacent views' depth unprojections actually sit
        # from each other in world space. A voxel finer than this disagreement
        # cannot merge anything: every measurement lands in its own cell and
        # "fusion" degenerates to per-view concatenation (obs=1 everywhere).
        # Set the voxel to the measured disagreement (bounded below by the
        # configured default) so overlapping views genuinely co-cycle and
        # multi-view support becomes measurable. Recorded in the report.
        # One pass, not two: measure_cross_view_offset is defined as the
        # median of these same pair statistics, so computing both meant
        # projecting and tree-building the whole view set twice.
        cross_view_stats = measure_cross_view_stats(views)
        measured_offset = cross_view_stats.get("median_disagreement_m")
        if measured_offset is not None and measured_offset > params.voxel_size:
            new_voxel = round(min(measured_offset, 30.0), 2)
            log.info("fusion_voxel_adapted", measured_offset_m=round(measured_offset, 3),
                     old_voxel=params.voxel_size, new_voxel=new_voxel)
            params.voxel_size = new_voxel

        # --- run & verify depth diagnostics before TSDF unprojection ----------
        from app.services.depth_diagnostics import run_depth_diagnostics

        diag_report = run_depth_diagnostics(workspace, max_depth_m=effective_max_depth)
        report.stages["depth_diagnostics"] = diag_report.to_dict()
        if not diag_report.can_feed_tsdf:
            # Name the failing criteria and the artifact counts. A bare
            # "Criteria A-G" refusal is undebuggable from the UI; the reason
            # must be in the error the user actually sees.
            criteria = diag_report.acceptance_criteria or {}
            failed_criteria = sorted(k for k, v in criteria.items() if not v)
            n_audited = len(diag_report.frames_diagnostics)
            mv = diag_report.multi_view_depth_consistency
            cross_view = mv.get("verdict") if isinstance(mv, dict) else mv
            cross_note = (
                mv.get("note") if isinstance(mv, dict) else None
            )
            detail = (
                f"failing criteria: {', '.join(failed_criteria) or 'none recorded'}; "
                f"depth maps audited: {n_audited}; "
                f"depth_model: {diag_report.depth_model}; "
                f"cross_view: {cross_view}"
                + (f" ({cross_note})" if cross_note and cross_note != "no comparable frame pairs with sufficient overlap" else "")
                + f"; effective_max_depth_m: {effective_max_depth:.1f}"
            )
            _fail(
                report,
                publish,
                "Depth maps failed diagnostic quality audit (Criteria A-G) — "
                f"cannot feed into TSDF reconstruction [{detail}]",
            )

        fusion_params = FusionParams(
            voxel_size=params.voxel_size,
            min_depth=params.min_depth_m,
            max_depth=effective_max_depth,
            pixel_noise_px=settings.dense.pixel_noise_px,
            max_points_per_view=params.max_points_per_view,
            max_total_points=settings.dense.max_fusion_points,
        )
        from app.services.depth_fusion import fuse_depth_views_with_provenance

        raw, fusion_provenance = fuse_depth_views_with_provenance(views, fusion_params)
        stage("depth_fusion", 0.35, {"fused_points": raw.n,
                                     "effective_max_depth_m": round(effective_max_depth, 1)})
        publish("fused", {"points": raw.n})
        # Record AFTER the 0.35 stage() call — stage() replaces the stage dict,
        # so keys written before it are lost (measured: both fields came out
        # null in run 5578f4 despite the measurement having run).
        report.stages["depth_fusion"]["measured_cross_view_offset_m"] = (
            round(measured_offset, 3) if measured_offset is not None else None)
        report.stages["depth_fusion"]["fusion_voxel_m"] = params.voxel_size
        report.stages["cross_view_stats"] = {
            "median_disagreement_m": cross_view_stats.get("median_disagreement_m"),
            "p95_disagreement_m": cross_view_stats.get("p95_disagreement_m"),
            "separation_buckets": cross_view_stats.get("separation_buckets"),
            "pairs": cross_view_stats.get("pairs", [])[:12],
            "labels": cross_view_stats.get("labels"),
        }
        # Part 26 contract: dense_model_raw.ply = fused pre-filter cloud.
        try:
            dense_dir = workspace / "dense"
            dense_dir.mkdir(parents=True, exist_ok=True)
            export_cloud(dense_dir / "dense_model_raw.ply", raw, "ply")
            report.exports.append({"format": "ply", "path": str(dense_dir / "dense_model_raw.ply"), "points": raw.n})
            # The mandated pre-fusion debug artifact is this same cloud under
            # a debug name; alias it rather than encoding 146 MB twice.
            _link_or_copy(dense_dir / "dense_model_raw.ply",
                          workspace / "diagnostics" / "dense_pre_fusion_debug.ply")
        except Exception as raw_err:
            log.warning("dense_raw_export_failed", error=str(raw_err))
        # Part 5 provenance: persist the dominant-observer assignment so every
        # fused point is traceable to its source frame + pixel. Kept as a
        # sidecar (not in the PLY) — PLY has no schema for it and the dense
        # export must stay standard.
        try:
            if fusion_provenance["source_frame_id"].size == raw.n and raw.n > 0:
                # Four per-point arrays. As JSON text this was a 91 MB document
                # costing ~15 s of .tolist()/json.dump per run (and ~1 GB of
                # temporary Python floats). The arrays are numeric and
                # homogeneous, so a compressed binary sidecar beside a small
                # JSON index is the same information for a fraction of the
                # cost — and the index keeps the artifact human-inspectable.
                arrays_path = workspace / "dense_fusion_provenance.npz"
                np.savez_compressed(
                    arrays_path,
                    source_frame_id=fusion_provenance["source_frame_id"].astype(np.int32),
                    fusion_weight=fusion_provenance["fusion_weight"].astype(np.float32),
                    source_pixel_u=fusion_provenance["source_pixel_u"].astype(np.float32),
                    source_pixel_v=fusion_provenance["source_pixel_v"].astype(np.float32),
                )
                prov_payload = {
                    "voxel_size": float(params.voxel_size),
                    "views": fusion_provenance["source_frame_names"],
                    "points": int(raw.n),
                    "arrays": arrays_path.name,
                    "array_keys": [
                        "source_frame_id", "fusion_weight",
                        "source_pixel_u", "source_pixel_v",
                    ],
                    "note": (
                        "per-point arrays live in the sibling .npz (float32/int32); "
                        "this file is the index"
                    ),
                }
                with open(workspace / "dense_fusion_provenance.json", "w") as f:
                    json.dump(prov_payload, f)
        except Exception as prov_err:
            log.warning("fusion_provenance_write_failed", error=str(prov_err))

        # Mandated debug artifacts: pre-fusion (raw concatenated per-view
        # measurements, exactly what fusion received) vs post-fusion (the
        # merged cloud this pipeline actually ships). Genuinely different
        # stages — the pre-fusion file preserves per-measurement provenance.

        # --- filter + normals + confidence ------------------------------------
        # Filter radii must live on the same scale as the fusion grid: a
        # radius-of-neighbours search finer than the voxel rejects nearly
        # every legitimately-fused point (measured: 0.4 m ROR on a 1.99 m
        # voxel deleted 97% of the cloud). Scale the neighbourhood radius
        # with the voxel; keep the SOR statistics (scale-free).
        ror_radius = max(params.ror_radius_m, 2.0 * params.voxel_size)
        opt = optimize_cloud(
            raw,
            OptimizeParams(
                voxel_size=params.voxel_size,
                sor_k=params.sor_k,
                sor_std_ratio=params.sor_std_ratio,
                ror_radius_m=ror_radius,
                ror_min_neighbors=params.ror_min_neighbors,
                normal_k=params.normal_k,
                min_confidence=params.min_confidence,
            ),
            camera_centers=[np.asarray(v["t"], dtype=np.float64) for v in poses],
            progress=lambda n, f, p: stage(n, f, p),
        )
        cloud = opt.cloud
        stage("optimize", 0.75, {"filter": opt.filter_stats, "points": cloud.n})
        publish("optimized", {"points": cloud.n})
        #: Points the denoise chain produced, BEFORE ghost rejection. The
        #: reported removal percentage keeps this basis so "removed_percent"
        #: still means what it always meant (filter removal) instead of
        #: silently absorbing a different decision.
        filtered_n = int(cloud.n)

        # --- ghost rejection (cross-view contradiction) ------------------------
        # A fused measurement carries its own depth uncertainty: σZ/Z is the
        # project's MAX_RELATIVE_DEPTH_ERR, so the error a point is allowed
        # to have grows with its observing range. A measurement that an
        # observing view contradicts by a closer surface, by MORE than that
        # own uncertainty can explain, is a ghost — the view looked at that
        # direction and saw something else in front. Removing it here is
        # what keeps duplicate-layer / hallucinated geometry out of the
        # mesh. Points that NO sampled view can read are never removed:
        # unobservable is a coverage fact, not evidence of error, and the
        # count is reported separately so a coverage hole can never be
        # counted as an accuracy failure. Every removal is named below.
        ghost_rejection: dict = {"status": "not_run"}
        try:
            from app.services.dense_diagnostics import (
                contradiction_mask,
                contradiction_report,
                cross_view_evidence,
                depth_uncertainty_budget,
                min_camera_range,
            )

            centers = np.asarray([np.asarray(p["t"], dtype=np.float64) for p in poses])
            rng_m = min_camera_range(cloud.xyz, centers)
            budget = depth_uncertainty_budget(rng_m)
            evidence = cross_view_evidence(cloud.xyz, views, budget)
            ghost_rejection = contradiction_report(evidence)
            ghost_rejection["budget_rule"] = (
                "MAX_RELATIVE_DEPTH_ERR x nearest observing range"
            )
            if budget.size:
                ghost_rejection["budget_m"] = {
                    "median": round(float(np.median(budget)), 3),
                    "p95": round(float(np.percentile(budget, 95)), 3),
                    "max": round(float(budget.max()), 3),
                }
            mask = contradiction_mask(evidence)
            if mask.any():
                cloud = cloud.slice(~mask)
            ghost_rejection["points_after"] = int(cloud.n)
            log.info(
                "dense_ghost_rejection",
                removed=ghost_rejection.get("removed_points"),
                judged=ghost_rejection.get("judged_points"),
                unobservable=ghost_rejection.get("unobservable_points"),
                points_after=int(cloud.n),
            )
        except Exception as ghost_err:
            log.warning("ghost_rejection_failed", error=str(ghost_err))
            ghost_rejection = {"status": f"failed: {ghost_err}"}
        stage("ghost_rejection", 0.78, {
            "removed_points": ghost_rejection.get("removed_points"),
            "removed_pct_of_cloud": ghost_rejection.get("removed_pct_of_cloud"),
            "unobservable_points": ghost_rejection.get("unobservable_points"),
            "points": int(cloud.n),
        })
        # Full account AFTER stage() — stage() replaces the stage dict, so keys
        # written before the call are lost (same trap as depth_fusion). The
        # stage's own timing/progress keys are merged back in.
        report.stages["ghost_rejection"] = {
            **report.stages.get("ghost_rejection", {}), **ghost_rejection,
        }
        publish("ghosts_rejected", {"points": int(cloud.n),
                                   "removed": ghost_rejection.get("removed_points")})

        # --- digital twin + quality -------------------------------------------
        # noise_percent (the score's noise component) counts *outliers only*
        # (SOR + ROR + confidence floor) — measurements the filters judged
        # spurious. Voxel-merge dedup is a many-to-one average of agreeing
        # measurements of the same surface cell, not noise; counting it made
        # the component track scene size instead of data quality.
        fstats = opt.filter_stats
        outliers_removed = int(fstats.get("sor_removed", 0)) + int(fstats.get("ror_removed", 0)) + int(fstats.get("low_conf_removed", 0))
        noise_pct = 0.0
        if raw.n > 0:
            noise_pct = 100.0 * outliers_removed / raw.n
        # Basis = the denoise chain's own input, so the reported removal
        # percentage still describes the filters alone; ghost rejection is
        # reported separately (it is a different decision, with its own rule).
        removed_pct = 0.0
        if raw.n > 0:
            removed_pct = 100.0 * max(0, raw.n - filtered_n) / raw.n
        twin = DigitalTwin(job_id, voxel_size=params.voxel_size)
        twin.update(cloud, label="full_run")
        stage("twin", 0.85, {"points": cloud.n, "noise_percent": round(noise_pct, 2)})

        quality = analyze_dense_quality(cloud, params.voxel_size, noise_percent=noise_pct)
        report.quality = quality.to_dict()
        report.twin = twin.to_dict()
        publish("analyzed", {"score": quality.dense_score, "grade": quality.grade})

        # --- Phase 1B: multi-layer detection on the fused cloud ----------------
        # (mesh-side detection runs after meshing; classification compares both)
        try:
            from app.services.dense_diagnostics import detect_layers

            dense_layers = detect_layers(
                cloud.xyz, params.voxel_size,
                normals=cloud.normals if getattr(cloud, "normals", None) is not None else None,
            )
            report.stages["dense_layers"] = dense_layers
        except Exception as layer_err:
            log.warning("dense_layer_detection_failed", error=str(layer_err))
            dense_layers = {"status": f"failed: {layer_err}"}

        # --- sparse<->dense consistency (mandate §22) --------------------------
        # The dense cloud must share the sparse scaffold's coordinate frame:
        # measure NN distances from every sparse point into the dense cloud.
        # Points whose offset exceeds their OWN depth-uncertainty budget
        # (σZ/Z ≤ MAX_RELATIVE_DEPTH_ERR · min observing range — the same
        # model the triangulator accepts them under) contradict the surface
        # they claim to anchor; they are recorded per-point in
        # sparse_dense_error.ply and excluded from the consistent subset
        # (sparse_model_consistent.ply) used for the combined model.
        try:
            from scipy.spatial import cKDTree

            from app.services.camera_pose_estimator import MAX_RELATIVE_DEPTH_ERR

            sparse_path = workspace / "sparse_model.ply"
            if sparse_path.is_file() and cloud.n > 0:
                s_cloud = read_ply(sparse_path)
                if s_cloud.n >= 1:
                    tree = cKDTree(cloud.xyz)
                    nn_dist, _ = tree.query(s_cloud.xyz, k=1, workers=-1)
                    report.stages["sparse_dense_consistency"] = {
                        "correspondences": int(s_cloud.n),
                        "median_m": round(float(np.median(nn_dist)), 4),
                        "p95_m": round(float(np.percentile(nn_dist, 95)), 4),
                        "rmse_m": round(float(np.sqrt(np.mean(nn_dist ** 2))), 4),
                        "within_3m_pct": round(float((nn_dist < 3.0).mean() * 100.0), 2),
                    }
                    # Uncertainty-budget screen: budget = MAX_RELATIVE_DEPTH_ERR
                    # × min range to any observing camera (sparse stage's own
                    # acceptance model). An offset beyond budget means the
                    # track triangulated to a position the surface evidence
                    # contradicts — a ghost match, not a scaffold point.
                    poses = _load_poses(poses_path)
                    if poses:
                        centers = np.asarray([np.asarray(p["t"], float) for p in poses])
                        # Chunked min-range: avoids materialising the full
                        # (N_sparse × N_poses) difference tensor.
                        chunk = 8192
                        min_range = np.empty(s_cloud.n, dtype=np.float64)
                        for lo in range(0, s_cloud.n, chunk):
                            hi = min(lo + chunk, s_cloud.n)
                            min_range[lo:hi] = np.linalg.norm(
                                s_cloud.xyz[lo:hi, None, :] - centers[None, :, :], axis=2
                            ).min(axis=1)
                        budget = MAX_RELATIVE_DEPTH_ERR * min_range
                        consistent = nn_dist <= np.maximum(budget, 1e-6)
                        n_drop = int((~consistent).sum())
                        # Coverage-vs-contradiction decomposition of the
                        # dropped tail (validation protocol §4: accuracy is
                        # not coverage). A dropped point NO kept view reads
                        # valid depth for sits in a coverage hole — its NN
                        # distance says nothing about geometric accuracy. A
                        # dropped point whose observing view DOES read depth
                        # there contradicts the fused surface — that is the
                        # accuracy tail. Reporting the split keeps a refusal
                        # created coverage hole from masquerading as error.
                        drop_decomp: dict = {"uncovered": 0, "contradicted": 0,
                                             "uncovered_median_m": None,
                                             "contradicted_median_m": None}
                        if n_drop and views:
                            from app.services.geometry import project_world_to_pixel

                            dropped_idx = np.nonzero(~consistent)[0]
                            n_drop_pts = dropped_idx.size
                            obs = np.zeros(n_drop_pts, dtype=np.int32)
                            dmin = np.full(n_drop_pts, np.inf)
                            dmax = np.full(n_drop_pts, -np.inf)
                            for view in views:
                                depth_arr = np.asarray(view.depth, dtype=np.float64)
                                vh, vw = depth_arr.shape
                                u, v, z = project_world_to_pixel(
                                    s_cloud.xyz[dropped_idx], view.R, view.t, view.K)
                                inb = (np.isfinite(u) & np.isfinite(v) & (z > 0.2)
                                       & (u >= 0) & (u < vw - 1) & (v >= 0) & (v < vh - 1))
                                if not inb.any():
                                    continue
                                ui = np.clip(u[inb].astype(np.int64), 0, vw - 1)
                                vi = np.clip(v[inb].astype(np.int64), 0, vh - 1)
                                vals = depth_arr[vi, ui]
                                ok = np.isfinite(vals) & (vals > 0.2)
                                if not ok.any():
                                    continue
                                rows = np.nonzero(inb)[0][ok]
                                dvals = vals[ok]
                                obs += np.bincount(rows, minlength=n_drop_pts).astype(np.int32)
                                np.minimum.at(dmin, rows, dvals)
                                np.maximum.at(dmax, rows, dvals)
                            covered = obs > 0
                            nn_drop = nn_dist[dropped_idx]
                            spread = dmax - dmin
                            pt_budget = np.maximum(budget[dropped_idx], 1e-6)
                            maps_disagree = covered & (obs >= 2) & (spread > pt_budget)
                            maps_agree = covered & (obs >= 2) & ~maps_disagree
                            single_view = covered & (obs == 1)
                            drop_decomp = {
                                "uncovered": int((~covered).sum()),
                                "contradicted": int(covered.sum()),
                                "uncovered_median_m": round(float(np.median(nn_drop[~covered])), 3) if (~covered).any() else None,
                                "contradicted_median_m": round(float(np.median(nn_drop[covered])), 3) if covered.any() else None,
                                "contradicted_maps_mutually_disagree": int(maps_disagree.sum()),
                                "contradicted_maps_mutually_agree": int(maps_agree.sum()),
                                "contradicted_single_view": int(single_view.sum()),
                                "note": "uncovered = no kept view reads valid depth at the point "
                                        "(coverage hole, not an accuracy measurement); "
                                        "contradicted = kept views read depth there but the fused "
                                        "surface disagrees; maps_mutually_disagree separates "
                                        "residual cross-view gauge inconsistency from a "
                                        "sparse-side local fault",
                            }
                        report.stages["sparse_dense_consistency"].update({
                            "screen": "nn_dist <= MAX_RELATIVE_DEPTH_ERR * min_range_to_camera",
                            "screened_correspondences": int(consistent.sum()),
                            "screened_dropped": n_drop,
                            "screened_median_m": round(float(np.median(nn_dist[consistent])), 4) if consistent.any() else None,
                            "screened_p95_m": round(float(np.percentile(nn_dist[consistent], 95)), 4) if consistent.any() else None,
                            "screened_rmse_m": round(float(np.sqrt(np.mean(nn_dist[consistent] ** 2))), 4) if consistent.any() else None,
                            "dropped_decomposition": drop_decomp,
                        })
                        save_ply(workspace / "sparse_dense_error.ply", PointCloud(
                            xyz=s_cloud.xyz,
                            rgb=s_cloud.ensure_colors(),
                            confidence=s_cloud.confidence if s_cloud.confidence is not None else np.ones(s_cloud.n),
                            observations=s_cloud.observations,
                            residual=nn_dist.astype(np.float32),  # per-point NN error (m)
                            meta={"note": "residual property = NN distance to dense cloud (m); "
                                          f"{n_drop} points exceed their depth-uncertainty budget"},
                        ))
                        save_ply(workspace / "sparse_model_consistent.ply", s_cloud.slice(consistent))
                        report.stages["sparse_dense_consistency"]["artifacts"] = [
                            "sparse_dense_error.ply", "sparse_model_consistent.ply"]
                    log.info("sparse_dense_consistency", job_id=job_id,
                             **report.stages["sparse_dense_consistency"])
        except Exception as consist_err:
            log.warning("sparse_dense_consistency_skipped", job_id=job_id, error=str(consist_err))

        # --- persist outputs ---------------------------------------------------
        stage("export", 0.95)
        dense_dir = workspace / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        model_path = dense_dir / "dense_model.ply"
        export_cloud(model_path, cloud, "ply")
        # Mandated post-fusion debug artifact = the shipped cloud, aliased.
        # (It has to be the shipped cloud, so it is aliased here rather than
        # written before ghost rejection had run.)
        try:
            (workspace / "diagnostics").mkdir(parents=True, exist_ok=True)
            _link_or_copy(model_path, workspace / "diagnostics" / "dense_post_fusion_debug.ply")
        except OSError as dbg_err:
            log.warning("postfusion_debug_write_failed", error=str(dbg_err))
        report.exports = [
            {"format": "ply", "path": str(model_path), "points": cloud.n},
        ]

        # The post-fusion chain was the one large untimed region: on the
        # reference run it ran ~692 s and showed up only as the gap between
        # the last substage and run_time_ms, so nothing named the cause. Each
        # step below closes a timed segment; the payload keys the mesh steps
        # write (mesh, viewer_lod, …) keep their own names, so no measurement
        # is overwritten by a timer.
        stage("export_cloud", 0.955)

        # --- surface mesh reconstruction --------------------------------------
        try:
            mesh_dir = workspace / "mesh"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = mesh_dir / "mesh.ply"
            from app.services.mesh_generator import MeshParams, generate_mesh

            # Part 13: mesh INPUT contract — record what the mesher receives.
            try:
                from scipy.spatial import cKDTree as _KD

                _d1, _ = _KD(cloud.xyz).query(cloud.xyz, k=min(2, cloud.n), workers=-1) if cloud.n > 1 else (np.zeros(1), None)
                _sp = _d1[:, 1] if (_d1.ndim == 2 and _d1.shape[1] > 1) else _d1
                _bb = cloud.xyz.max(axis=0) - cloud.xyz.min(axis=0)
                _vol = max(float(_bb[0] * _bb[1] * _bb[2]), 1e-9)
                _nrm_ok = (cloud.normals is not None and np.isfinite(cloud.normals).all()
                           and (np.linalg.norm(cloud.normals, axis=1) > 0.5).mean())
                report.stages["mesh_input"] = {
                    "mesh_input_point_count": int(cloud.n),
                    "mesh_input_density_pts_per_m3": round(float(cloud.n / _vol), 4),
                    "mesh_input_spacing_median_m": round(float(np.median(_sp)), 3),
                    "mesh_input_spacing_p95_m": round(float(np.percentile(_sp, 95)), 3),
                    "mesh_input_normal_quality": round(float(_nrm_ok), 3) if isinstance(_nrm_ok, float) else None,
                    "source": "filtered fused cloud (observed surfaces only — not per-view concatenation)",
                    "labels": "MEASURED",
                }
            except Exception as mi_err:
                log.warning("mesh_input_contract_failed", error=str(mi_err))
            stage("mesh_input_scan", 0.96)

            # Production mesh: BPA (observed surface; 'auto' resolves to it).
            # Poisson runs as a DIAGNOSTIC side-artifact only — its watertight
            # closure can invent geometry in unobserved regions.
            # Sheet-gap guard (measured, self-anchored): when the fused cloud
            # carries stacked duplicate sheets (double-bounce / depth edges),
            # the BPA weld and candidate triangles that bridge the measured
            # sheet separation are refused so the mesh cannot zig-zag between
            # two copies of one surface. OFF whenever no sheets are measured.
            _sg = None
            try:
                # Reuse the layer measurement the dense stage already made on
                # THIS cloud (same xyz, same normals, same voxel) instead of
                # running the detector again for the same answer.
                _layers = dense_layers
                if _layers.get("status") == "measured" and _layers.get("layer_pair_count"):
                    _gap = _layers.get("median_layer_separation_m")
                    if _gap:
                        from app.services.mesh_generator import SheetGapParams
                        _sg = SheetGapParams(
                            enabled=True,
                            sheet_gap_m=float(_gap),
                            sheet_gap_source=(
                                f"detect_layers on fused cloud: {_layers.get('layer_pair_count')} pairs, "
                                f"median separation {_gap:.3f} m"),
                        )
            except Exception as sg_err:
                log.warning("sheet_gap_measure_failed", error=str(sg_err))
            mesh_depth = _poisson_depth_for_fusion(cloud, params.voxel_size)
            mesh_obj, mesh_stats = generate_mesh(
                cloud,
                MeshParams(method="auto", sheet_gap=_sg, poisson_depth=mesh_depth),
            )
            report.stages["mesh_resolution"] = {
                "poisson_depth": mesh_depth,
                "fusion_voxel_m": params.voxel_size,
                "rule": "Poisson cell size is no finer than 1.5x the measured fusion voxel",
            }
            # ONE encode. mesh.ply, base_mesh.ply and mesh_full.ply are the
            # same geometry under three names (the stage name, the legacy
            # name and the authoritative name); encoding it three times cost
            # ~29 s and 122 MB of pure duplication per run. Aliases link to
            # the single encode instead.
            mesh_obj.save_ply(mesh_path, normals=False)
            _link_or_copy(mesh_path, mesh_dir / "base_mesh.ply")
            # Provenance sidecar for EVERY mesh generation (method, BPA radii,
            # input-cloud + mesh hashes, sheet-gap params, rejected count).
            try:
                from app.services.mesh_generator import mesh_provenance_record

                (mesh_dir / "base_mesh.json").write_text(json.dumps(
                    mesh_provenance_record(mesh_obj, cloud, workspace, mesh_stats,
                                           mesh_rel="mesh/mesh.ply"), indent=2))
            except Exception as prov_err:
                log.warning("mesh_provenance_record_failed", error=str(prov_err))
            # Phase 1B Part 20/26 contract: mesh_full.ply is the authoritative
            # artifact (same geometry as mesh.ply, canonical name).
            try:
                _link_or_copy(mesh_path, mesh_dir / "mesh_full.ply")
                report.exports.append({"format": "ply", "path": str(mesh_dir / "mesh_full.ply"),
                                       "vertices": mesh_obj.n, "faces": mesh_obj.m, "role": "authoritative"})
            except Exception as full_err:
                log.warning("mesh_full_export_failed", error=str(full_err))
            stage("mesh_build", 0.97)
            # Mesh quality audit — ONE owner (app.services.mesh_quality), the
            # same function the offline rebuild path calls, so a rebuilt mesh
            # can never leave this run's stored records self-contradicting.
            # Thresholds derive from the fusion voxel, not taste: a triangle
            # edge much larger than the voxel spans empty space (a bridge);
            # the dominant component should carry the scene; a mostly-down-
            # facing surface means flipped normals.
            try:
                from app.services.mesh_quality import audit_mesh_quality

                audit = audit_mesh_quality(
                    mesh_obj,
                    cloud.xyz,
                    params.voxel_size,
                    method=str(mesh_stats.get("method")),
                    views=views,
                    dense_layers=dense_layers,
                )
                for note in audit.notes:
                    log.warning("mesh_quality_audit_note", note=note)
                with open(workspace / "mesh_quality_report.json", "w") as f:
                    json.dump(audit.quality, f, indent=2)
                report.stages["mesh"] = audit.quality
                if audit.classification is not None:
                    report.stages["layer_classification"] = audit.classification
                if audit.layer_payload is not None:
                    with open(workspace / "layer_source_classification.json", "w") as f:
                        json.dump(audit.layer_payload, f, indent=2)
            except Exception as mq_err:
                log.warning("mesh_quality_report_failed", error=str(mq_err))
            stage("mesh_audit", 0.98)

            # --- viewer LOD (Part 5/6) -----------------------------------
            # The browser gets a decimated GLB, never the full PLY. This
            # runs inside the dense stage AFTER mesh validation: the
            # pipeline never waits on the browser. The GLB is textured from
            # the registered source views (conditioning exclusions apply);
            # when texturing genuinely fails the fallback is COLOR_0 vertex
            # colors with a named reason in the stats.
            try:
                from app.services.mesh_glb import decimate_for_viewer, export_viewer_glb
                from app.services.textured_glb import build_texture_payload

                # Texture the LOD THAT SHIPS: UV rows index the GLB's own face
                # order, so the payload must be baked on the same decimated
                # surface the writer emits — baking on the full mesh leaves UVs
                # that describe a surface the viewer never receives.
                lod_mesh, lod_dec_stats = decimate_for_viewer(mesh_obj)
                try:
                    texture = build_texture_payload(lod_mesh, workspace)
                except Exception as tex_err:  # defensive: build_texture_payload itself names its failures
                    log.warning("viewer_texture_build_failed", error=str(tex_err))
                    from app.services.textured_glb import _vertex_color_fallback

                    texture = _vertex_color_fallback(f"texture build failed: {tex_err}")
                lod_stats = export_viewer_glb(lod_mesh, mesh_dir / "mesh_viewer.glb",
                                              texture=texture, full_mesh=mesh_obj)
                lod_stats["decimation"] = lod_dec_stats
                report.stages["viewer_lod"] = lod_stats
            except Exception as lod_err:
                log.warning("viewer_lod_failed", error=str(lod_err))
            stage("viewer_lod_build", 0.99)

            report.exports.append({"format": "mesh_ply", "path": str(mesh_path), "vertices": mesh_obj.n, "faces": mesh_obj.m})
            log.info("dense_surface_mesh_complete", job_id=job_id, vertices=mesh_obj.n, faces=mesh_obj.m)
        except Exception as mesh_err:
            log.warning("dense_surface_mesh_skipped", job_id=job_id, error=str(mesh_err))

        report.status = "completed"
        report.run_time_ms = (time.perf_counter() - start) * 1000
        # Phase 1B Part 26: dense_quality_report.json (dense-side view only;
        # mesh support + classification live in layer_source_classification.json
        # and mesh_quality_report.json once meshing completes).
        try:
            from app.services.dense_diagnostics import dense_quality_report

            dqr = dense_quality_report(
                report.quality, dense_layers,
                support={},  # filled by mesh stage into layer_source_classification
                voxel=params.voxel_size,
                counts={
                    "raw": int(raw.n),
                    "filtered": int(filtered_n),
                    "ghost_rejected": int(ghost_rejection.get("removed_points") or 0),
                    "shipped": int(cloud.n),
                    "removed_pct": round(removed_pct, 2),
                    "outlier_removed": outliers_removed,
                    "outlier_pct": round(noise_pct, 2),
                },
            )
            with open(workspace / "dense_quality_report.json", "w") as f:
                json.dump(dqr, f, indent=2)
        except Exception as dqr_err:
            log.warning("dense_quality_report_failed", error=str(dqr_err))
        with open(workspace / "dense_report.json", "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        publish("complete", report.to_dict())
        log.info(
            "dense_reconstruction_complete",
            job_id=job_id,
            points=cloud.n,
            score=round(quality.dense_score, 2),
            grade=quality.grade,
            time_ms=round(report.run_time_ms, 2),
        )
        return report.to_dict()

    except Exception as exc:
        report.status = "failed"
        report.error = str(exc)
        report.run_time_ms = (time.perf_counter() - start) * 1000
        try:
            with open(workspace / "dense_report.json", "w") as f:
                json.dump(report.to_dict(), f, indent=2)
        except OSError:
            pass
        publish("error", {"error": str(exc)})
        log.exception("dense_reconstruction_failed", job_id=job_id, error=str(exc))
        raise


def _make_publisher(job_id: str):
    def publish(event_type: str, payload: dict[str, Any]) -> None:
        engine.publish(job_id, event_type, payload)
    return publish


def _fail(report: DenseReport, publish, message: str):
    report.error = message
    publish("error", {"error": message})
    log.error("dense_input_missing", error=message)
    raise ValueError(message)


def _link_or_copy(src: Path, dest: Path) -> None:
    """Make *dest* the same file as *src*, preferring a hard link.

    Several artifacts in this pipeline are byte-identical copies of one
    cloud or mesh under different names (stage name, legacy name, mandated
    debug name). Encoding each one separately cost seconds and hundreds of
    megabytes of pure duplication per run. A hard link keeps them as real,
    independently-readable files while the bytes exist once; a copy is the
    fallback for filesystems that cannot link.
    """
    import os
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dest.exists() and not dest.is_symlink():
            dest.unlink()
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.is_dir():
            return p
    return None


def _load_poses(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    frames = data.get("frames", data) if isinstance(data, dict) else data
    return list(frames)


def _apply_anchor_gate(
    views: list[DepthView],
    depth_dir: Path,
    sparse_xyz: np.ndarray | None,
    poses: list[dict],
) -> tuple[list[DepthView], list[str]]:
    """Drop depth maps the sparse-anchor gate refuses.

    One owner: the dense stage and the offline mesh rebuild
    (``scripts/rebuild_mesh_run.py``) both call this, so a rebuilt run's
    cross-view occlusion audit measures the same view set fusion used.
    Returns ``(kept_views, refused_frame_ids)``.
    """
    if sparse_xyz is None or len(sparse_xyz) == 0:
        return views, []
    from app.services.depth_generator import depth_map_passes_anchor_gate

    kept: list[DepthView] = []
    refused: list[str] = []
    for view in views:
        depth_path = _find_depth(depth_dir, view.frame_id)
        if depth_path is not None and not depth_map_passes_anchor_gate(
            depth_path, view.frame_id, sparse_xyz, poses,
        ):
            refused.append(view.frame_id)
            log.warning("dense_view_map_fails_anchor_gate", frame_id=view.frame_id,
                        note="depth map fails the sparse-anchor gate — excluded from fusion")
            continue
        kept.append(view)
    return kept, refused


def scene_depth_ceiling(
    views: list[DepthView],
    configured_max_m: float,
    sparse_xyz: np.ndarray | None = None,
    poses: list[dict] | None = None,
) -> float:
    """Fusion/audit depth ceiling: the configured value, extended for far field.

    A far-field scene (oblique airport survey: ground 500-970 m from the
    camera) is INVISIBLE to a near-field ceiling — every unprojected
    measurement is zeroed before it can fuse or be cross-checked, which the
    audit reports as "cross_view: NOT_EVALUATED" and fusion as an empty
    cloud. The honest ceiling is scene-derived, anchored to the deepest
    SfM-VERIFIED geometry (P99.5 camera-space depth of the sparse points
    across all poses, so horizon extrapolation the triangulation never saw
    cannot set it) plus 10% headroom. When no sparse geometry exists, fall
    back to the maps' own P99.9 depth (a lone garbage pixel cannot set the
    ceiling). The configured value is a floor, never a target: a ceiling can
    only be EXTENDED, so near-field runs behave exactly as configured.
    """
    verified_max = 0.0
    if sparse_xyz is not None and len(sparse_xyz) and poses:
        X = np.asarray(sparse_xyz, dtype=np.float64)
        z_all = []
        for pose in poses:
            R = np.asarray(pose["R"], dtype=np.float64)
            t = np.asarray(pose["t"], dtype=np.float64)
            z = ((X - t) @ R)[:, 2]  # stored convention: t = camera centre
            z = z[z > 0]
            if z.size:
                z_all.append(z)
        if z_all:
            z_cat = np.concatenate(z_all)
            verified_max = float(np.percentile(z_cat, 99.5))
    if verified_max <= 0.0:
        deepest = 0.0
        for v in views:
            d = np.asarray(v.depth, dtype=np.float64)
            valid = d[np.isfinite(d) & (d > 0)]
            if valid.size:
                deepest = max(deepest, float(np.percentile(valid, 99.9)))
        verified_max = deepest
    if verified_max <= configured_max_m:
        return configured_max_m
    return round(verified_max * 1.1, 1)


def measure_cross_view_offset(
    views: list[DepthView], max_pairs: int = 12, query_cap: int = 250_000
) -> float | None:
    """Median NN distance between adjacent views' unprojected depth points.

    This is the honest scale of cross-view disagreement of the aligned depth
    maps in world space: two views seeing the same ground produce two point
    sets whose typical offset IS the achievable fusion resolution. Returned
    in metres; None when fewer than two viewable pairs exist.
    """
    stats = measure_cross_view_stats(views, max_pairs=max_pairs, query_cap=query_cap)
    return stats.get("median_disagreement_m")


def measure_cross_view_stats(views: list[DepthView], max_pairs: int = 12, query_cap: int = 250_000) -> dict:
    """Per-pair cross-view disagreement + camera separation (Phase 1B Part 4).

    Disagreement as a function of view separation is the diagnostic that
    separates depth-map noise (flat with separation) from pose/scale drift
    (grows with separation). Returns{"pairs": [{frames, camera_separation_m,
    median_nn_m}], "median_disagreement_m", "separation_buckets"}.

    Measurement definition: queries are *sampled* pixels of view A (every
    ``sample_step``-th, capped); the NN reference set is the **dense**
    unprojection of view B. Building the tree on B's dense surface is what
    makes this a geometric-disagreement measure — sampling BOTH sides (the
    pre-2026-09 behaviour) made the median NN read the aliasing pitch of
    the query grid itself (measured: 1.94 m at step 60 vs 0.25 m at full
    density), which inflated the fusion voxel to that alias pitch and
    coarsened the entire dense cloud.
    """
    if len(views) < 2:
        return {"pairs": [], "median_disagreement_m": None}
    from scipy.spatial import cKDTree

    QUERY_CAP = 250_000   # sampled query points per view (side A)
    TREE_CAP = 500_000    # reference points per view (side B)

    unproj_cache: dict[tuple[int, int], np.ndarray] = {}

    def _unproj(vi: int, cap: int) -> np.ndarray:
        # Adjacent pairs share views (pair i's B is pair i+1's A), so cache
        # by (view, cap).
        #
        # The stride is applied to the MASK, before unprojecting, and is 2D
        # on both axes. The row-major stride this replaces kept every
        # step-th element of the flattened mask — every step-th COLUMN of
        # every row — which both biases the sample (a ~60:1 anisotropic
        # query grid) and unprojected every valid pixel of the view (8.3 M
        # on a 4K map) only to discard 98% of them.
        key = (vi, cap)
        cached = unproj_cache.get(key)
        if cached is not None:
            return cached
        view = views[vi]
        from app.services.geometry import camera_to_world, unproject_pixel_to_camera

        z = np.asarray(view.depth, dtype=np.float64)
        valid = np.isfinite(z) & (z > 0.2)
        count = int(valid.sum())
        if cap > 0 and count > cap:
            step = int(np.ceil(np.sqrt(count / float(cap))))
            stride = np.zeros_like(valid)
            stride[::step, ::step] = True
            valid &= stride
        idx = np.nonzero(valid)
        cam = unproject_pixel_to_camera(idx[1], idx[0], z[idx], view.K)
        out = camera_to_world(cam, view.R, view.t)
        unproj_cache[key] = out
        return out

    pairs: list[dict] = []
    for i in range(0, min(len(views) - 1, max_pairs)):
        # Side A: a capped, isotropic sample of view i's surface.
        A = _unproj(i, max(50, query_cap))
        # Side B: the densest reference surface the cap allows — the NN
        # distance must reflect B's surface, not our own sampling grid.
        B = _unproj(i + 1, TREE_CAP)
        if len(A) < 50 or len(B) < 50:
            continue
        d, _ = cKDTree(B).query(A, k=1, workers=-1)
        sep = float(np.linalg.norm(np.asarray(views[i].t) - np.asarray(views[i + 1].t)))
        pairs.append({
            "frames": [views[i].frame_id, views[i + 1].frame_id],
            "camera_separation_m": round(sep, 2),
            "median_nn_m": round(float(np.median(d)), 3),
        })
    if not pairs:
        return {"pairs": [], "median_disagreement_m": None}
    meds = np.asarray([p["median_nn_m"] for p in pairs])
    seps = np.asarray([p["camera_separation_m"] for p in pairs])
    buckets: dict[str, float | None] = {}
    for name, lo, hi in (("sep_lt5m", 0, 5), ("sep_5_20m", 5, 20), ("sep_gt20m", 20, 1e9)):
        m = (seps >= lo) & (seps < hi)
        buckets[name] = round(float(np.median(meds[m])), 3) if m.any() else None
    return {
        "pairs": pairs,
        "median_disagreement_m": round(float(np.median(meds)), 3),
        "p95_disagreement_m": round(float(np.percentile(meds, 95)), 3),
        "separation_buckets": buckets,
        "labels": {"disagreement": "MEASURED",
                   "note": "Median NN from sampled view-A pixels to the DENSE view-B "
                           "unprojection; true cross-view surface disagreement, "
                           "lower-bounded by view B's pixel pitch (~0.1 m at 1080p)."},
    }


def _build_views(poses: list[dict], depth_dir: Path, image_dir: Path | None, params: DenseParams) -> list[DepthView]:
    """Assemble DepthView objects for poses that have a matching depth map."""
    views: list[DepthView] = []
    for pose in poses:
        fid = pose["frame_id"]
        depth_path = _find_depth(depth_dir, fid)
        if depth_path is None:
            continue
        depth = np.load(depth_path).astype(np.float64)
        if depth.ndim != 2:
            log.warning("depth_not_2d", frame_id=fid, shape=depth.shape)
            continue
        # A map may be stored at the depth model's own grid rather than the
        # frame's. The intrinsics that go with THIS map travel in its
        # sidecar; the frame-scale lets the colour branch below sample the
        # frame-resolution images. Legacy maps resolve to the frame
        # geometry, so older runs fuse exactly as before.
        from app.services.depth_generator import read_depth_geometry

        K_grid, scale_x, scale_y = read_depth_geometry(depth_path, pose)
        rgb = None
        if image_dir is not None:
            for ext in (".jpg", ".jpeg", ".png"):
                cand = image_dir / f"{fid}{ext}"
                if cand.exists():
                    import cv2
                    img = cv2.imread(str(cand))
                    if img is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    break
        views.append(
            DepthView(
                frame_id=fid,
                depth=depth,
                rgb=rgb,
                K=K_grid,
                R=np.asarray(pose["R"], dtype=np.float64),
                t=np.asarray(pose["t"], dtype=np.float64),
                frame_scale=(scale_x, scale_y),
            )
        )
    return views


def _find_depth(depth_dir: Path, frame_id: str) -> Path | None:
    for name in (f"{frame_id}.npy", f"{frame_id}_depth.npy"):
        cand = depth_dir / name
        if cand.exists():
            return cand
    return None


def load_dense_report(job_id: str) -> dict | None:
    """Read the persisted report for a job, or None if the job never ran."""
    path = settings.storage.project_dir(job_id) / "dense_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def export_dense_format(job_id: str, fmt: str) -> Path:
    """Export the dense model in *fmt* on demand (cached in dense/exports)."""
    workspace = settings.storage.project_dir(job_id)
    dense_dir = workspace / "dense"
    model_path = dense_dir / "dense_model.ply"
    if not model_path.exists():
        raise FileNotFoundError(f"No dense model for job {job_id} — run dense reconstruction first")
    cache_dir = dense_dir / "exports"
    cache_dir.mkdir(parents=True, exist_ok=True)
    from app.services.pointcloud import EXPORT_EXTENSIONS, read_ply

    out = cache_dir / f"dense_model{EXPORT_EXTENSIONS[fmt]}"
    if not out.exists():
        export_cloud(out, read_ply(model_path), fmt)
    return out
