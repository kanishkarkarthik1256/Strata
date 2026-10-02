"""Sparse reconstruction orchestrator — chains the full SfM pipeline.

Pipeline:
1. Load selected frames
2. Extract features (SuperPoint/SIFT)
3. Match features (LightGlue/BFMatcher)
4. Select image pairs
5. Geometric verification (RANSAC)
6. Camera pose estimation (COLMAP/OpenCV)
7. Bundle adjustment
8. Mission analysis + confidence estimation
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.camera_pose_estimator import (
    estimate_poses,
    _quat_to_rotation_matrix,
    SCENE_RANGE_TOLERANCE,
)
from app.services.confidence_estimator import estimate_confidence
from app.services.dashboard import build_dashboard
from app.services.feature_extractor import FeatureExtractor
from app.services.feature_matcher import FeatureMatcher
from app.services.geometric_verifier import verify_matches
from app.services.image_files import count_image_files, list_image_files
from app.services.mission_analyzer import analyze_mission
from app.services.pair_selector import ImageInfo, compute_image_entropy, select_pairs
from app.services.reprojection import compute_overall_reprojection_error
from app.services.trajectory_optimizer import generate_trajectory

log = get_logger("drone_recon.services.sparse_reconstruction")

_MAX_TELEMETRY_ATTITUDE_DISAGREEMENT_DEG = 10.0


def _cross_validate_rpy(rows: list[dict], max_samples: int = 50) -> dict | None:
    """Cross-validate the log's roll/pitch/yaw against its quaternion.

    Both encode the same physical orientation, so the quaternion stays the
    single canonical constraint and this check only validates the adapter's
    Euler convention (gimbal camera frame: view-axis elevation = pitch,
    view azimuth = yaw; see dji_srt_telemetry._rpy_to_quat) before trusting
    it in provenance. Returns None when rpy columns are absent.
    """
    if not any(r.get("roll") for r in rows[:20]):
        return None
    from app.services.dji_srt_telemetry import _rpy_to_quat

    errs = []
    step = max(1, len(rows) // max_samples)
    for row in rows[::step]:
        try:
            q = np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])])
            q = q / max(1e-12, np.linalg.norm(q))
            R_q = _quat_to_rotation_matrix(q)
            qw, qx, qy, qz = _rpy_to_quat(
                float(row["roll"]), float(row["pitch"]), float(row["yaw"])
            )
            R_e = _quat_to_rotation_matrix(np.array([qw, qx, qy, qz]))
            cos_theta = np.clip((np.trace(R_q.T @ R_e) - 1.0) / 2.0, -1.0, 1.0)
            errs.append(np.degrees(np.arccos(cos_theta)))
        except (KeyError, TypeError, ValueError):
            continue
    if len(errs) < 3:
        return None
    errs = np.array(errs)
    return {
        "convention_tested": "gimbal camera frame (view elev=pitch, azimuth=yaw)",
        "median_angle_diff_deg": round(float(np.median(errs)), 4),
        "max_angle_diff_deg": round(float(errs.max()), 4),
        "samples": int(len(errs)),
        "note": "quaternion is canonical; rpy used for cross-validation only (no double-count)",
    }


def run_sparse_reconstruction(
    selected_dir: Path,
    output_dir: Path,
    *,
    project_id: str = "",
    total_frames: int = 0,
    selected_frames: int = 0,
    gps_data: dict[str, tuple[float, float]] | None = None,
    flight_poses_csv: Path | None = None,
    intrinsics_path: Path | None = None,
    telemetry_csv: Path | None = None,
    video_fps: float | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> dict:
    """Run the full sparse reconstruction pipeline.

    Two localization paths:

    * **telemetry-assisted** (``flight_poses_csv`` given): camera centres and
      orientations come from the flight log (frame_id, x/y/z, quaternion, ...
      — an explicitly valid input for drone photogrammetry), sparse points
      are triangulated from the *visual* matches through those poses, and the
      result is metric when the log is metric.
    * **video-only** (default): COLMAP incremental SfM as before, with a
      hard honesty gate — a reconstruction whose camera path spans more than
      1.2× the median camera→point distance has degenerate translation (a
      real oblique aerial survey is ~0.1–0.5) and must not silently feed a
      distorted downstream model.

    Returns a dict with all results for the API response.
    """
    start_time = time.perf_counter()

    # Shared helper: case-insensitive extensions (DJI writes .JPG).
    frame_files = list_image_files(selected_dir)
    if not frame_files:
        raise ValueError(f"No frames found in {selected_dir}")

    log.info("reconstruction_started", frames=len(frame_files), project_id=project_id)

    # Wall clock per block, reported as ``timings_ms``. This stage used to
    # report only its total (1166 s on the run under measurement), which left
    # most of it unattributable without replaying the whole pipeline. Every
    # block that can plausibly dominate gets a timer; whatever is left over
    # after summing them is the honest remainder.
    timings: dict[str, float] = {}

    def _elapsed(key: str, started: float) -> None:
        timings[key] = round((time.perf_counter() - started) * 1000.0, 1)

    def _at(frac: float, block: str) -> None:
        """Announce a sparse substage milestone (0..1) to the live poller.

        The orchestrator forwards this onto the ``stage:sparse`` event, which
        is what the Processing page reads; without it the longest stage of a
        run reported 0% until it completed.
        """
        if progress is not None:
            try:
                progress(block, min(1.0, max(0.0, frac)))
            except Exception:  # progress must never break reconstruction
                log.exception("sparse_progress_publish_failed", block=block)

    # Stage 1: OpenCV pre-pass — REQUIRED only for telemetry-assisted
    # localization (verified match tracks drive triangulate_with_known_poses).
    # In the video-only path COLMAP re-does extraction/matching internally, so
    # this full SIFT+BF pipeline would be discarded work (measured ~350 s on a
    # 191-frame run): skip it and take the equivalent feature counts from the
    # COLMAP database instead. Each image is decoded exactly ONCE and reused
    # for both the entropy score and feature extraction.
    needs_opencv_prepass = flight_poses_csv is not None and flight_poses_csv.is_file()
    telemetry_pose_only = (
        needs_opencv_prepass
        and _has_pose_columns(flight_poses_csv)
        and _telemetry_attitude_is_consistent(flight_poses_csv)
    )
    if telemetry_pose_only:
        needs_opencv_prepass = False
    features: dict = {}
    feature_counts: dict[str, int] = {}
    extractor = FeatureExtractor(max_keypoints=settings.colmap.max_features)

    # Stage 2: Pair selection (cheap; runs before decoding so the telemetry
    # path only extracts features for frames actually in a selected pair).
    # GPS priors join here when the flight log carries geodetic columns.
    log.info("stage", stage="pair_selection", progress=0.2)
    _at(0.15, "pair_selection")
    image_infos: list[ImageInfo] = []
    for idx, f in enumerate(frame_files):
        name = f.stem
        info = ImageInfo(
            frame_id=name,
            index=idx,
            timestamp_sec=float(idx),
            file_path=str(f),
            entropy=0.0,
        )
        # Add GPS if available
        if gps_data and name in gps_data:
            info.gps_lat, info.gps_lon = gps_data[name]
        image_infos.append(info)
    if gps_data:
        log.info("pair_gps_priors", frames_with_gps=sum(
            1 for i in image_infos if i.gps_lat is not None))

    # OpenCV pre-pass (telemetry path only): each image is decoded exactly
    # ONCE and consumed for both the entropy score (adaptive pair scoring)
    # and feature extraction — nothing holds more than one image at a time.
    # Video-only: COLMAP re-does extraction/matching internally and selects
    # its own pairs, so this pass (measured ~350 s on a 191-frame run) and
    # the entropy decode would be discarded work — skipped entirely.
    if needs_opencv_prepass:
        log.info("stage", stage="feature_extraction", progress=0.1)
        _t0 = time.perf_counter()
        for idx, f in enumerate(frame_files):
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            image_infos[idx].entropy = compute_image_entropy(gray)
            feat = extractor.extract(f.stem, gray)
            features[f.stem] = feat
            feature_counts[f.stem] = len(feat.keypoints)
        _elapsed("prepass_feature_extraction", _t0)
        log.info(
            "opencv_prepass_complete", frames=len(features),
            entropy_scored=sum(1 for i in image_infos if i.entropy > 0),
        )

    _t0 = time.perf_counter()
    pairs = select_pairs(image_infos, strategy="adaptive")
    _elapsed("pair_selection", _t0)

    # Video-only: COLMAP matched internally, so `features` stays empty and
    # the matching loop below is a no-op — the report carries the COLMAP
    # backend instead.
    # Stage 3: Feature matching (telemetry path only).
    matcher = FeatureMatcher()
    matches = []
    match_qualities = []
    if needs_opencv_prepass:
        log.info("stage", stage="feature_matching", progress=0.3)

    _t0 = time.perf_counter()
    for frame_a, frame_b in pairs:
        if frame_a in features and frame_b in features:
            result = matcher.match(features[frame_a], features[frame_b])
            if result.num_inliers > 5:
                matches.append(result)
                match_qualities.append(result.confidence)
    _elapsed("prepass_matching", _t0)

    # Stage 4: Geometric verification (telemetry path only)
    verified = []
    if needs_opencv_prepass:
        log.info("stage", stage="geometric_verification", progress=0.5)
        _t0 = time.perf_counter()
        for match in matches:
            feat_a = features.get(match.frame_a)
            feat_b = features.get(match.frame_b)
            if feat_a is not None and feat_b is not None:
                v = verify_matches(match, feat_a.keypoints, feat_b.keypoints)
                if v.passed:
                    verified.append(v)
        _elapsed("prepass_verification", _t0)

    # Stage 5: Camera localization — telemetry-assisted when a flight log is
    # present, otherwise COLMAP incremental SfM. Substage timings (which
    # backend consumed the wall clock, and where) are recorded on
    # reconstruction.detail by both paths and persisted in the report.
    reconstruction = None
    telemetry_info: dict = {}
    gps_priors: dict[str, tuple[float, float, float]] | None = None
    _t_pose = time.perf_counter()
    if flight_poses_csv is not None and flight_poses_csv.is_file():
        log.info("stage", stage="pose_estimation", progress=0.6, mode="telemetry_assisted")
        _at(0.55, "pose_estimation")
        reconstruction, telemetry_info, gps_priors = _telemetry_assisted_poses(
            selected_dir, features, verified, frame_files,
            flight_poses_csv, intrinsics_path,
        )
        if reconstruction is not None and reconstruction.num_registered > 0:
            log.info(
                "telemetry_localization_accepted",
                cameras=reconstruction.num_registered,
                points=reconstruction.num_points,
                mean_reproj=round(reconstruction.mean_reproj_error, 3),
                **telemetry_info,
            )
        else:
            reason = telemetry_info.get("reason", "no usable telemetry poses")
            log.warning("telemetry_localization_unusable", reason=reason)
            telemetry_info["reason"] = reason
            reconstruction = None  # fall through to video-only SfM

    if reconstruction is None:
        log.info("stage", stage="pose_estimation", progress=0.6, mode="video_only")
        _at(0.4, "pose_estimation")
        reconstruction = estimate_poses(selected_dir, project_id,
                                        frame_files=frame_files)
        _at(0.6, "pose_estimation")
        # Video-only: the per-frame feature counts come from the COLMAP DB
        # (no OpenCV extraction pass ran — see Stage 1).
        feature_counts = dict(reconstruction.feature_counts)
    _pose_estimation_s = time.perf_counter() - _t_pose
    reconstruction.detail.setdefault("pose_estimation_ms", round(_pose_estimation_s * 1000.0, 1))

    # Contract: sparse may only claim success when at least one camera was
    # registered. Zero registered cameras is NOT successful reconstruction —
    # fail here with the real diagnostics instead of writing a "completed"
    # report (and no poses.json) that downstream stages would trip over.
    if reconstruction.num_registered <= 0:
        # ``feature_counts`` is only populated by the OpenCV pre-pass, which
        # the video-only path skips (COLMAP re-does extraction internally) —
        # reporting ``len(features)`` from a skipped pass printed a 0 that
        # belonged to the pass, not the data (the COLMAP DB knows the truth).
        counts = feature_counts or dict(reconstruction.feature_counts or {})
        frames_with_features = sum(1 for c in counts.values() if c)
        raise ValueError(
            "SfM produced zero registered cameras; reconstruction cannot continue. "
            f"attempted_frames={len(frame_files)} "
            f"frames_with_features={frames_with_features} "
            f"pairs={len(pairs)} matches={len(matches)} verified={len(verified)} "
            f"pose_backend={reconstruction.backend}"
        )

    # Single-pass trajectory placement (root-cause fix): with an external
    # telemetry CSV, a video-only monocular reconstruction has NO metric
    # scale (measured ~17x gauge collapse on flight_to_tower_7511dc).
    # Place it into the telemetry ENU frame via the MEASURED similarity
    # (time-matched correspondences), run the composite trajectory gate,
    # and hand BA the fix positions as SOFT metric priors.  Scale is a
    # measurement — reported, never silent, never an arbitrary stretch.
    trajectory_block: dict = {}
    metric_targets: dict[str, np.ndarray] | None = None
    placement_succeeded = False
    _t0 = time.perf_counter()
    if (
        reconstruction.backend not in ("telemetry_triangulation",)
        and telemetry_csv is not None and telemetry_csv.is_file()
    ):
        from app.services.trajectory_sync import (
            camera_timestamps_from_quality_report,
            compare_trajectories,
            composite_trajectory_gate,
            align_reconstruction_to_telemetry,
            pose_jump_report,
            temporal_sync_report,
            telemetry_position_prior,
        )
        try:
            video_duration = None
            try:
                source_meta = json.loads(
                    (output_dir / "source.json").read_text()
                ) if (output_dir / "source.json").is_file() else {}
                video_duration = float(source_meta.get("duration_sec")) or None
            except (OSError, ValueError, TypeError):
                video_duration = None
            prior, prior_report = telemetry_position_prior(
                telemetry_csv,
                video_fps=video_fps,
                video_duration_sec=video_duration,
            )
            qr = workspace_qr = output_dir / "quality_report.json"
            camera_ts = (
                camera_timestamps_from_quality_report(qr) if qr.is_file() else {}
            )
            if not camera_ts:
                raise ValueError("quality_report.json missing camera timestamps")
            targets, place_report = align_reconstruction_to_telemetry(
                reconstruction, prior, camera_ts
            )
            if not targets:
                raise ValueError(
                    f"trajectory placement failed: {place_report.get('error', 'no time-matched correspondences')}"
                )
            metric_targets = targets
            sync_report = temporal_sync_report(
                qr, telemetry_csv, video_fps or 30.0,
                video_duration_sec=video_duration,
            )
            # PRE-BA gate: temporal sync ONLY.  Shape/continuity are judged
            # AFTER bundle adjustment — the metric_targets priors are the
            # mechanism that pulls drifted camera segments back, so gating
            # on the pre-BA trajectory would reject the state BA exists to
            # repair.  Sync, however, cannot be fixed downstream: if frames
            # cannot be paired with the telemetry of their own moment, no
            # later stage can repair it — fail fast here.
            if not sync_report.get("pass"):
                trajectory_block = {
                    "telemetry_prior": {
                        k: v for k, v in prior_report.items()
                        if k != "rejected_samples"
                    },
                    "temporal_sync": sync_report,
                    "placement": {
                        k: v for k, v in place_report.items()
                        if k != "rotation_3x3"
                    },
                    "composite_gate": {
                        "pass": False,
                        "components": {"temporal_sync": False},
                        "error": "temporal sync failed pre-BA (unfixable downstream)",
                    },
                }
                with open(output_dir / "trajectory_gate_failure.json", "w") as f:
                    json.dump(trajectory_block, f, indent=2, default=str)
                raise ValueError(
                    "Temporal-sync gate FAILED pre-BA: frames cannot be paired "
                    f"with the telemetry of their own moment. "
                    f"matched={sync_report.get('matched_frames')} "
                    f"unmatched={sync_report.get('unmatched_frames')} "
                    f"max_dt={sync_report.get('max_frame_to_telemetry_dt_s')}s "
                    f"(threshold {sync_report.get('max_dt_threshold_s')}s)"
                )
            placement_succeeded = True
            # A previous failed attempt's gate-failure artifact must not
            # survive a now-passing run — the UI would show a stale refusal.
            stale_gate = output_dir / "trajectory_gate_failure.json"
            if stale_gate.is_file():
                stale_gate.unlink()
            trajectory_block = {
                "telemetry_prior": {
                    k: v for k, v in prior_report.items()
                    if k != "rejected_samples"
                },
                "temporal_sync": sync_report,
                "placement": {
                    k: v for k, v in place_report.items()
                    if k != "rotation_3x3"
                },
                "composite_gate": {"pass": None, "components": {},
                                   "note": "shape/continuity judged post-BA"},
            }
            log.info(
                "trajectory_placement_complete",
                matched=place_report.get("cameras_matched"),
                scale=place_report.get("similarity_scale_visual_per_telemetry"),
                robust_inliers=place_report.get("robust_fit", {}).get("inliers"),
                robust_outliers=place_report.get("robust_fit", {}).get("outliers"),
                sync_pass=sync_report.get("pass"),
                next="shape+continuity gate after BA",
            )
        except Exception as exc:
            # No silent fallback: placement failure is recorded and the gate
            # reports it; video-only geometry continues ONLY because the
            # honesty gate below still applies (it cannot pass a collapsed
            # gauge without telemetry).
            trajectory_block = {
                "placement_error": str(exc),
                "composite_gate": {"pass": False, "components": {},
                                   "error": str(exc)},
            }
            log.warning("trajectory_placement_failed", error=str(exc))
    _elapsed("trajectory_placement", _t0)

    # Honesty gate (video-only SfM only): a camera path that spans much more
    # than the scene distance means the translation gauge collapsed (measured
    # 2.48 on the collapsed airport1 run; the real flight log gives ~0.33).
    # Such a reconstruction must NOT silently feed dense fusion.  Skipped
    # when trajectory placement SUCCEEDED (telemetry supplied the metric
    # scale; the composite trajectory gate then judges the result).
    scale_ratio = 0.0
    placement_ok = bool(trajectory_block.get("composite_gate", {}).get("pass", None) is None) or bool(
        trajectory_block.get("composite_gate", {}).get("pass")
    )
    if (
        reconstruction.backend not in ("telemetry_triangulation",)
        and reconstruction.points3d and not placement_ok
    ):
        centers = np.array([c.position for c in reconstruction.cameras.values()])
        xyz = np.array([p.position for p in reconstruction.points3d])
        path_span = float(np.linalg.norm(centers.max(0) - centers.min(0)))
        d = np.linalg.norm(xyz[None, :, :] - centers[:, None, :], axis=2)
        flat = d[d > 0]
        med_scene = float(np.median(flat)) if flat.size else 0.0
        scale_ratio = path_span / med_scene if med_scene > 0 else 0.0
        if scale_ratio > 1.2:
            # Message must reflect reality: when telemetry WAS provided but
            # could not be applied, telling the user to "provide telemetry"
            # is false — name what happened to it instead.
            if telemetry_csv is not None and telemetry_csv.is_file():
                tele_note = (
                    "Flight telemetry WAS provided for this run but could not "
                    "be applied: "
                    + str(trajectory_block.get("placement_error")
                          or trajectory_block.get("composite_gate", {}).get("error")
                          or "placement did not produce a trusted trajectory")
                )
            else:
                tele_note = (
                    "provide flight telemetry (GPS/flight metadata) for "
                    "telemetry-assisted localization"
                )
            raise ValueError(
                "SfM translation is degenerate: the camera path spans "
                f"{scale_ratio:.2f}x the median camera-to-point distance "
                f"(path_span={path_span:.2f}, median_scene_distance={med_scene:.2f}). "
                "A coherent reconstruction cannot be produced from this geometry; "
                f"{tele_note} "
                f"diagnostics: registered={reconstruction.num_registered} "
                f"points={reconstruction.num_points} "
                f"reproj={reconstruction.mean_reproj_error:.3f}"
            )

    # Composite trajectory gate: on failure the trajectory is NOT trusted
    # for dense work — fail the sparse stage with the full diagnostics so
    # the run stops before depth/fusion on a bad trajectory (no silent
    # downstream repair).
    if trajectory_block and trajectory_block.get("composite_gate", {}).get("pass") is False:
        gate_err = trajectory_block.get("composite_gate", {}).get("error")
        raise ValueError(
            "Trajectory gate FAILED — reconstruction halted before dense "
            f"stages. components={trajectory_block.get('composite_gate', {}).get('components')} "
            f"detail={gate_err or trajectory_block.get('placement_error', '')} "
            f"sync={trajectory_block.get('temporal_sync')} "
            f"placement={trajectory_block.get('placement')}"
        )

    # Persist poses + sparse cloud so downstream stages (dense pipeline,
    # georeferencing) can run without re-running SfM.
    _t0 = time.perf_counter()
    if reconstruction.cameras:
        _write_poses_json(reconstruction, output_dir)
        _write_sparse_ply(reconstruction, output_dir)
    _elapsed("persist_geometry_pre_ba", _t0)
    # applied as a soft positional prior (never a hard constraint). The
    # metrics reported are re-measured by re-projection, not library labels.
    log.info("stage", stage="bundle_adjustment", progress=0.7)
    _at(0.65, "bundle_adjustment")
    from app.services.bundle_adjustment import run_bundle_adjustment
    # Prior sigma: a placed initialization is already accurate to the GNSS
    # noise scale, so priors bind tightly (they polish, not drag); the
    # video-only-no-telemetry path keeps the loose default.  Chosen from the
    # measured telemetry quality (quantized ~1 m fixes -> ~2 m noise).
    ba_sigma = 2.0 if metric_targets is not None else 5.0
    _t0 = time.perf_counter()
    run_explicit_ba = reconstruction.backend != "telemetry_triangulation" and not (
        placement_succeeded and bool(telemetry_info.get("reason"))
    )
    ba_result = run_bundle_adjustment(
        reconstruction,
        use_colmap=run_explicit_ba,
        gps_priors=gps_priors,
        metric_targets=metric_targets,
        gps_prior_sigma_m=ba_sigma,
        # Dataset/calibrated intrinsics are measurements — hold K fixed
        # unless this reconstruction estimated it itself (video-only SfM).
        refine_intrinsics=reconstruction.backend != "telemetry_triangulation",
    )
    reconstruction.mean_reproj_error = ba_result.final_error
    _elapsed("bundle_adjustment", _t0)

    # POST-BA composite gate: trajectory SHAPE and POSE CONTINUITY are
    # judged here, on the trajectory BA actually produced — the metric
    # priors exist precisely to pull drifted segments back, so the pre-BA
    # state must not be the thing that fails the run.  Sync was proven
    # pre-BA (cannot be repaired downstream); shape/continuity are the
    # verdict on the FINAL trajectory that feeds dense.
    _t0 = time.perf_counter()
    if trajectory_block and metric_targets is not None:
        try:
            from app.services.trajectory_sync import (
                compare_trajectories as _cmp,
                composite_trajectory_gate as _gate,
                pose_jump_report as _pjr,
            )
            post_centers = {
                name: np.asarray(cam.position, dtype=np.float64)
                for name, cam in reconstruction.cameras.items()
            }
            centers_by_stem = {
                Path(name).stem: name for name in post_centers
            }
            ts_names = sorted(camera_ts, key=lambda nm: camera_ts[nm])
            matched_names = [
                centers_by_stem[nm] for nm in ts_names if nm in centers_by_stem
            ]
            shape_centers = np.array([post_centers[nm] for nm in matched_names])
            shape_ts = np.array([camera_ts[Path(nm).stem] for nm in matched_names])
            shape_report = _cmp(shape_centers, prior, visual_timestamps=shape_ts)
            continuity_report = _pjr(shape_centers, shape_ts)
            gate = _gate(sync_report, shape_report, continuity_report)
            trajectory_block["trajectory_shape"] = {
                k: v for k, v in shape_report.items()
                if k != "similarity_transform"
            }
            trajectory_block["pose_continuity"] = continuity_report
            trajectory_block["composite_gate"] = gate
            trajectory_block["gate_stage"] = "post_ba"
            log.info(
                "trajectory_gate_post_ba",
                gate_pass=gate["pass"],
                gate_components=gate["components"],
                vel_dir_median=shape_report.get("velocity_direction", {}).get("median_cosine"),
                len_ratio=shape_report.get("cumulative_path_length", {}).get("ratio"),
                jumps=continuity_report.get("jump_count"),
            )
            if not gate["pass"]:
                log.warning(
                    "trajectory_gate_failed",
                    components=gate["components"],
                    action="halting before dense stages (post-BA verdict)",
                    shape=json.dumps(trajectory_block["trajectory_shape"]),
                    continuity=json.dumps(trajectory_block["pose_continuity"]),
                )
                with open(output_dir / "trajectory_gate_failure.json", "w") as f:
                    json.dump(trajectory_block, f, indent=2, default=str)
        except Exception as exc:
            # Gate evaluation itself failed — record it; the explicit False
            # pass below decides the run, missing evidence is NOT a pass.
            trajectory_block["gate_error"] = str(exc)
            trajectory_block.setdefault("composite_gate", {})["pass"] = False
            trajectory_block["composite_gate"]["error"] = f"post-BA gate error: {exc}"
            log.warning("trajectory_gate_evaluation_failed", error=str(exc))
        if trajectory_block.get("composite_gate", {}).get("pass") is False:
            raise ValueError(
                "Trajectory gate FAILED post-BA — reconstruction halted before "
                f"dense stages. components={trajectory_block['composite_gate'].get('components')} "
                f"detail={trajectory_block['composite_gate'].get('error', '')} "
                f"shape={trajectory_block.get('trajectory_shape')} "
                f"continuity={trajectory_block.get('pose_continuity')}"
            )

    _elapsed("post_ba_gate", _t0)

    # Re-persist the FINAL (post-BA, gated) state: the pre-BA write above
    # exists so a gated failure still leaves diagnosable artifacts, but
    # downstream stages (depth/dense/georef) must consume exactly the
    # trajectory the gate validated — BA moves cameras/points after the
    # early write (measured: pre-BA z range −87→+14 m vs post-BA coherent).
    _t0 = time.perf_counter()
    if reconstruction.cameras:
        _write_poses_json(reconstruction, output_dir)
        _write_sparse_ply(reconstruction, output_dir)
    _elapsed("persist_geometry_post_ba", _t0)

    # Stage 7: Trajectory generation
    log.info("stage", stage="trajectory", progress=0.8)
    _at(0.8, "trajectory")
    _t0 = time.perf_counter()
    trajectory = generate_trajectory(reconstruction.cameras)
    _elapsed("trajectory_generation", _t0)

    # Stage 8: Mission analysis
    log.info("stage", stage="analysis", progress=0.9)
    _at(0.9, "analysis")
    _t0 = time.perf_counter()
    analysis = analyze_mission(
        reconstruction,
        trajectory,
        total_frames=total_frames,
        selected_frames=selected_frames or len(frame_files),
        feature_counts=feature_counts,
        match_qualities=match_qualities,
    )  # match_qualities is [] in video-only mode (COLMAP matched internally)
    _elapsed("mission_analysis", _t0)

    # Stage 9: Confidence estimation
    _t0 = time.perf_counter()
    reproj_errors = compute_overall_reprojection_error(reconstruction.cameras, reconstruction.points3d)
    confidence = estimate_confidence(
        reconstruction, feature_counts=feature_counts, reproj_errors=reproj_errors
    )

    # Stage 10: Dashboard
    dashboard = build_dashboard(reconstruction, analysis, confidence, trajectory, len(frame_files))
    _elapsed("confidence_and_dashboard", _t0)

    # Build report
    elapsed = (time.perf_counter() - start_time) * 1000
    report = {
        "project_id": project_id,
        "status": "completed",
        "pipeline_time_ms": round(elapsed, 2),
        # Per-block wall clock (ms). ``unattributed_ms`` is what is left of the
        # stage total after every named timer — a non-zero remainder means a
        # block is still untimed, which is the signal to instrument it rather
        # than to guess.
        "timings_ms": {
            **timings,
            # The pose backend owns this block; its own split is reported under
            # stages.pose_estimation, so only the parent is added here (adding
            # both would double-count the same wall clock).
            "pose_estimation": getattr(reconstruction, "detail", {}).get("pose_estimation_ms"),
            "unattributed_ms": round(
                elapsed - sum(timings.values())
                - float(getattr(reconstruction, "detail", {}).get("pose_estimation_ms") or 0.0),
                1,
            ),
        },
        "localization": {
            # "telemetry_assisted" covers BOTH paths where telemetry supplied
            # the metric frame: the telemetry_triangulation backend AND a
            # successful telemetry placement of video-only SfM (the placement
            # + prior-BA gave the model metric scale + ENU positions).
            "mode": (
                "telemetry_assisted"
                if (reconstruction.backend == "telemetry_triangulation" or placement_succeeded)
                else "video_only"
            ),
            **telemetry_info,
            "path_span": round(reconstruction.path_span, 3) if reconstruction.path_span else None,
            "median_scene_distance": round(reconstruction.median_scene_distance, 3) if reconstruction.median_scene_distance else None,
            "video_only_scale_ratio": round(scale_ratio, 3) if scale_ratio else None,
        },
        "stages": {
            "feature_extraction": {
                "backend": "colmap" if not needs_opencv_prepass else extractor._backend,
                "frames": len(frame_files),
                "per_frame_source": "colmap_db" if not needs_opencv_prepass else "opencv",
            },
            "feature_matching": {"matches": len(matches), "backend": matcher._backend},
            "geometric_verification": {"verified": len(verified), "total": len(matches)},
            "pose_estimation": {
                "registered": reconstruction.num_registered,
                "backend": reconstruction.backend,
                # Substage wall clock, measured inside the pose backend
                # (COLMAP: extraction/matching/mapping; telemetry: the
                # whole triangulation). This is what made a 1166 s sparse
                # stage attributable — see camera_pose_estimator.detail.
                **{k: v for k, v in getattr(reconstruction, "detail", {}).items()
                   if k.endswith("_ms")},
                "feature_cap": getattr(reconstruction, "detail", {}).get("feature_cap", {}),
            },
            "bundle_adjustment": {
                "backend": ba_result.backend,
                "joint": ba_result.backend == "pycolmap_joint",
                "cameras_optimized": ba_result.num_cameras if ba_result.converged else 0,
                "points_optimized": ba_result.num_points if ba_result.converged else 0,
                "observations": ba_result.num_observations,
                "initial_error_px": ba_result.initial_error,
                "final_error_px": ba_result.final_error,
                "converged": ba_result.converged,
                "degraded": ba_result.degraded,
                "total_ms": round(ba_result.ba_time_ms, 2),
                # Where the BA call's own wall clock went. On this cloud the
                # COLMAP text round trip (not the solve) is the cost, so it is
                # reported rather than folded into one number.
                "timings_ms": ba_result.timings,
                "gps_prior_cameras": ba_result.gps_prior_cameras,
                "gps_prior_rms_m": ba_result.gps_prior_rms_m,
                "gps_prior_rms_before_m": ba_result.gps_prior_rms_before_m,
                "scope_note": (
                    "A: COLMAP internal incremental/global BA ran inside "
                    "pose_estimation (backend=colmap). B: the STRATA explicit "
                    "joint BA is THIS section (backend=pycolmap_joint when it "
                    "actually solved; 'unavailable' means it did not)."
                ),
            },
        },
        "tracks": _track_statistics(reconstruction),
        "gps": _gps_statistics(reconstruction, gps_priors, ba_result),
        "trajectory_alignment": trajectory_block or {"mode": "video_only_no_telemetry"},
        "reconstruction": {
            "num_cameras": reconstruction.num_registered,
            "num_points": reconstruction.num_points,
            "mean_reproj_error": round(reconstruction.mean_reproj_error, 4),
            "quality_gates": getattr(reconstruction, "quality_gates", None),
        },
        "trajectory": {
            "points": len(trajectory.points),
            "total_length": trajectory.total_length,
            "smoothness": trajectory.smoothness,
        },
        "analysis": {
            "mission_score": analysis.mission_score,
            "grade": analysis.grade,
            "suggestions": analysis.suggestions,
        },
        "confidence": {
            "mean_camera": confidence.mean_camera_confidence,
            "mean_point": confidence.mean_point_confidence,
        },
        "dashboard": {
            "mission_score": dashboard.mission_score,
            "grade": dashboard.grade,
            "registered_cameras": dashboard.registered_cameras,
            "sparse_points": dashboard.sparse_points,
            "camera_registration_percent": dashboard.camera_registration_percent,
        },
    }

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "reconstruction_report.json", "w") as f:
        json.dump(report, f, indent=2)

    log.info(
        "reconstruction_complete",
        cameras=reconstruction.num_registered,
        points=reconstruction.num_points,
        score=analysis.mission_score,
        time_ms=round(elapsed, 2),
    )

    return report


def _telemetry_assisted_poses(
    selected_dir: Path,
    features: dict,
    verified: list,
    frame_files: list,
    flight_poses_csv: Path,
    intrinsics_path: Path | None,
) -> tuple[object, dict, dict[str, tuple[float, float, float]] | None]:
    """Build known poses from a flight log and triangulate visual matches.

    Returns (ReconstructionResult | None, info dict, gps_priors). None means
    the log was unusable for this run (missing fps, no matching frames) and
    video-only SfM should take over — the reason is reported, never swallowed.
    gps_priors is frame_id -> (lat, lon, alt) for frames whose log row carried
    geodetic coordinates (None when the log has no GPS columns).
    """
    from app.services.camera_pose_estimator import (
        PAIR_BASELINE_PARALLAX_PX,
        SUPPAIR_MIN_PARALLAX_PX,
        _quat_to_rotation_matrix,
        triangulate_with_known_poses,
    )

    info: dict = {"telemetry_source": flight_poses_csv.name}
    try:
        with open(flight_poses_csv, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        return None, {"reason": f"flight log unreadable: {exc}"}, None
    required = {"frame_id", "x", "y", "z", "qw", "qx", "qy", "qz"}
    if not rows or not required.issubset(rows[0].keys()):
        return None, {"reason": "flight log lacks pose columns (frame_id,x,y,z,qw,qx,qy,qz)"}, None

    # frame_id -> video frame index. The MovingDrone log is 1:1 with video
    # frames (verified: 0..N-1 with no gaps); extracted frame_i.k files were
    # taken at video frame index * stride, so telemetry joins via the
    # extractor's quality_report (index -> frame_num) when present.
    stride = 1
    qr_path = selected_dir.parent / "quality_report.json"
    frame_num_by_index: dict[int, int] = {}
    if qr_path.is_file():
        try:
            qr = json.loads(qr_path.read_text())
            kept = [f for f in qr.get("frames", []) if f.get("kept")]
            frame_num_by_index = {f["index"]: f.get("frame_num", f["index"]) for f in kept}
        except (OSError, ValueError):
            frame_num_by_index = {}
    if frame_num_by_index:
        indices = sorted(frame_num_by_index)
        stride = max(1, min(
            (b - a for a, b in zip(indices, indices[1:]) if b > a), default=1,
        ))

    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    gps_by_frame: dict[str, tuple[float, float, float]] = {}
    center_by_frame_num: dict[int, np.ndarray] = {}
    pose_by_frame_id: dict[int, dict] = {}
    for row in rows:
        try:
            fid = int(float(row["frame_id"]))
            C = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            q = np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])])
        except (KeyError, TypeError, ValueError):
            continue
        q = q / max(1e-12, np.linalg.norm(q))
        # GPS provenance (Part 8 of the SIH audit fix): the log's geodetic
        # tuple travels WITH the pose it was measured with. Absent/blank
        # columns simply leave the frame without a prior — never fabricated.
        gps_row: tuple[float, float, float] | None = None
        try:
            lat_raw = (row.get("latitude") or row.get("lat") or "").strip()
            lon_raw = (row.get("longitude") or row.get("lon") or "").strip()
            alt_raw = (row.get("altitude") or row.get("alt") or "").strip()
            if lat_raw and lon_raw:
                gps_row = (float(lat_raw), float(lon_raw), float(alt_raw or 0.0))
        except (TypeError, ValueError):
            gps_row = None
        pose_by_frame_id[fid] = {"C": C, "q": q, "row": row, "gps": gps_row}

    n_with_gps = 0
    for frame_file in frame_files:
        stem = frame_file.stem  # frame_000000
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        video_frame = frame_num_by_index.get(idx, idx * stride)
        pose = pose_by_frame_id.get(video_frame)
        if pose is not None:
            poses[stem] = (pose["C"], _quat_to_rotation_matrix(pose["q"]))
            if pose.get("gps") is not None:
                gps_by_frame[stem] = pose["gps"]
                n_with_gps += 1
            center_by_frame_num[video_frame] = pose["C"]
    info["gps_rows_in_log"] = sum(1 for p in pose_by_frame_id.values() if p.get("gps") is not None)
    info["gps_frames_matched"] = n_with_gps
    info["frames_matched"] = len(poses)
    info["selected_frames"] = len(frame_files)
    if len(poses) < 2:
        return None, {**info, "reason": "fewer than 2 selected frames matched a telemetry pose"}, None

    # Orientation cross-validation: the log carries BOTH a quaternion and
    # roll/pitch/yaw for the same physical orientation. The quaternion is the
    # canonical orientation (consumed above); rpy is used ONLY to verify the
    # source's Euler convention — never as an independent constraint, which
    # would double-count the same measurement.
    #
    # A large RPY/quaternion discrepancy means the RPY→quaternion adapter
    # (dji_srt_telemetry._rpy_to_quat) uses the wrong Euler convention for
    # this log — it does NOT mean the quaternion is wrong. The quaternion is
    # measured directly by the flight controller and is the canonical pose.
    # Demoting RPY disagreement to a warning-only diagnostic (instead of
    # rejecting the entire telemetry) prevents an RPY convention bug from
    # forcing a full COLMAP fallback when valid position+quaternion data is
    # available.
    rpy_check = _cross_validate_rpy(rows)
    if rpy_check is not None:
        info["rpy_quaternion_crosscheck"] = rpy_check
        if rpy_check["median_angle_diff_deg"] > _MAX_TELEMETRY_ATTITUDE_DISAGREEMENT_DEG:
            log.warning(
                "telemetry_rpy_convention_mismatch",
                median_diff_deg=rpy_check["median_angle_diff_deg"],
                note=(
                    "quaternion is canonical and will be used; RPY convention "
                    "adapter disagrees — RPY fields are unreliable for this log"
                ),
            )

    # Intrinsics: dataset-supplied (intrinsics.json) when available, else a
    # dataset fov_vertical, else the existing heuristic. Scaled K never guessed.
    K = None
    if intrinsics_path is not None and intrinsics_path.is_file():
        try:
            intr = json.loads(intrinsics_path.read_text())
            sample = cv2.imread(str(frame_files[0]))
            h, w = sample.shape[:2]
            if intr.get("width") == w and intr.get("height") == h:
                K = np.array([
                    [float(intr["fx"]), 0, float(intr["cx"])],
                    [0, float(intr["fy"]), float(intr["cy"])],
                    [0, 0, 1],
                ])
                info["intrinsics_source"] = "dataset_intrinsics_json"
        except (OSError, ValueError, KeyError):
            K = None
    if K is None:
        sample = cv2.imread(str(frame_files[0]))
        if sample is None:
            return None, {**info, "reason": "frames unreadable"}, None
        h, w = sample.shape[:2]
        fov_row = next((r for r in rows if r.get("fov_vertical")), None)
        try:
            fov_deg = float(fov_row["fov_vertical"]) if fov_row else 0.0
        except (TypeError, ValueError):
            fov_deg = 0.0
        if fov_deg > 0:
            fy = (h / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
            K = np.array([[fy, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]])
            info["intrinsics_source"] = "fov_vertical_from_flight_log"
        else:
            f = max(w, h) * 0.8
            K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])
            info["intrinsics_source"] = "heuristic_fallback"

    # Verified matches -> keypoint-index pairs for triangulation.
    verified_pairs = []
    total_inliers = 0
    for v in verified:
        m = v.match
        mask = v.inlier_mask
        if mask is None:
            mask = np.ones(len(m.matches), dtype=bool)
        sel = m.matches[mask]
        if len(sel) >= 8:
            verified_pairs.append((m.frame_a, m.frame_b, sel[:, 0], sel[:, 1]))
            total_inliers += len(sel)
    info["verified_pairs"] = len(verified_pairs)
    info["verified_inliers"] = total_inliers
    if not verified_pairs:
        result = _pose_only_result(poses, K, frame_files)
        info.update({"reason": "no geometrically verified matches", "fast_path": "telemetry_poses"})
        return result, info, (gps_by_frame or None)

    # Baseline gate: drop pairs whose telemetry baseline cannot yield
    # verifiable parallax at the flight altitude. Verification cannot
    # constrain geometry on near-zero-baseline pairs, so false matches
    # survive there and triangulate near the lens (the airport3 failure).
    altitude = float(np.median([p[0][2] for p in poses.values()])) if poses else 0.0
    if altitude > 0.0 and len(poses) >= 2:
        min_baseline = PAIR_BASELINE_PARALLAX_PX * altitude / float(K[0, 0])
        kept_pairs = []
        dropped = 0
        for frame_a, frame_b, idx_a, idx_b in verified_pairs:
            ca = poses.get(frame_a, (None, None))[0]
            cb = poses.get(frame_b, (None, None))[0]
            if ca is not None and cb is not None and np.linalg.norm(ca - cb) < min_baseline:
                dropped += 1
            else:
                kept_pairs.append((frame_a, frame_b, idx_a, idx_b))
        verified_pairs = kept_pairs
        info["min_pair_baseline_m"] = round(min_baseline, 3)
        info["pairs_dropped_short_baseline"] = dropped
        info["flight_altitude_m"] = round(altitude, 1)
        info["verified_pairs_after_baseline_gate"] = len(verified_pairs)
        log.info(
            "telemetry_pair_baseline_gate",
            altitude_m=round(altitude, 1),
            min_baseline_m=round(min_baseline, 3),
            dropped=dropped,
            kept=len(verified_pairs),
        )
        if not verified_pairs:
            result = _pose_only_result(poses, K, frame_files)
            info.update({
                "reason": f"all verified pairs below telemetry baseline floor ({min_baseline:.2f} m)",
                "fast_path": "telemetry_poses",
            })
            return result, info, (gps_by_frame or None)

        # Depth-usable pair guarantee. The σZ/Z ≈ e/(f·θ) gate needs
        # b·f/Z ≥ e/MAX_REL ≈ 15 px of expected parallax; pairs narrower than
        # that triangulate with unusable depth precision even when genuine.
        # Ensure every frame has a telemetry-spaced partner: for each frame
        # march forward along the trajectory to the first frame at least
        # ``depth_baseline`` away and match+verify that pair if not already
        # present. Bounded (≤ one added pair per frame) and honest — real
        # matches, real verification, only the pair CHOICE is telemetry-led.
        depth_baseline = SUPPAIR_MIN_PARALLAX_PX * altitude / float(K[0, 0])
        order = sorted(poses)  # zero-padded stems sort in trajectory order
        centers = [poses[s][0] for s in order]
        existing = {(a, b) for a, b, _, _ in verified_pairs}
        matcher = FeatureMatcher()
        added = 0
        for i in range(len(order) - 1):
            j = i + 1
            while j < len(order) and np.linalg.norm(centers[j] - centers[i]) < depth_baseline:
                j += 1
            if j >= len(order):
                break
            pair = (order[i], order[j])
            if pair in existing or (pair[1], pair[0]) in existing:
                continue
            fa, fb = features.get(pair[0]), features.get(pair[1])
            if fa is None or fb is None:
                continue
            m = matcher.match(fa, fb)
            if m.num_inliers <= 8:
                continue
            v = verify_matches(m, fa.keypoints, fb.keypoints)
            if not v.passed:
                continue
            mask = v.inlier_mask
            if mask is None:
                mask = np.ones(len(m.matches), dtype=bool)
            sel = m.matches[mask]
            if len(sel) >= 8:
                verified_pairs.append((pair[0], pair[1], sel[:, 0], sel[:, 1]))
                existing.add(pair)
                added += 1
        info["depth_baseline_m"] = round(depth_baseline, 3)
        info["telemetry_spaced_pairs_added"] = added
        info["pairs_after_spacing"] = len(verified_pairs)
        log.info(
            "telemetry_spaced_pairs",
            depth_baseline_m=round(depth_baseline, 3),
            added=added,
            total=len(verified_pairs),
        )
        if not verified_pairs:
            result = _pose_only_result(poses, K, frame_files)
            info.update({"reason": "no depth-usable pairs after telemetry spacing", "fast_path": "telemetry_poses"})
            return result, info, (gps_by_frame or None)

    image_paths = {f.stem: f for f in frame_files}
    result = triangulate_with_known_poses(
        features, verified_pairs, poses, K, image_paths, gps_by_frame=gps_by_frame)
    result.telemetry_assisted = True
    result.telemetry_source = flight_poses_csv.name
    return result, info, (gps_by_frame or None)


def _has_pose_columns(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            fields = {field.strip().lower() for field in next(csv.reader(handle))}
    except (OSError, UnicodeError, StopIteration):
        return False
    return {"frame_id", "x", "y", "z", "qw", "qx", "qy", "qz"}.issubset(fields)


def _telemetry_attitude_is_consistent(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            check = _cross_validate_rpy(list(csv.DictReader(handle)))
    except (OSError, UnicodeError):
        return False
    return check is None or check["median_angle_diff_deg"] <= _MAX_TELEMETRY_ATTITUDE_DISAGREEMENT_DEG


def _pose_only_result(
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    intrinsics: np.ndarray,
    frame_files: list[Path],
):
    from app.services.camera_pose_estimator import CameraPose, ReconstructionResult

    cameras = {}
    for image_id, frame_file in enumerate(frame_files, start=1):
        pose = poses.get(frame_file.stem)
        if pose is None:
            continue
        center, rotation = pose
        cameras[frame_file.stem] = CameraPose(
            image_id=image_id,
            frame_id=frame_file.stem,
            position=np.asarray(center, dtype=np.float64),
            rotation=np.asarray(rotation, dtype=np.float64),
            quaternion=np.zeros(4, dtype=np.float64),
            intrinsics=np.asarray(intrinsics, dtype=np.float64),
            distortion=np.zeros(0, dtype=np.float64),
            is_estimated=False,
        )
    return ReconstructionResult(
        cameras=cameras,
        num_registered=len(cameras),
        num_points=0,
        mean_reproj_error=0.0,
        backend="telemetry_triangulation",
        telemetry_assisted=True,
        detail={"telemetry_pose_only": True},
    )


def _write_poses_json(result, output_dir: Path) -> None:
    """Persist per-view poses as ``poses.json`` (the dense pipeline's input).

    Pose convention: ``X_world = R @ X_cam + t``. Saves one entry per
    registered camera with intrinsics, extrinsics, and GPS when present on
    the camera record.
    
    Note: frame_id is stored WITHOUT the file extension for compatibility
    with the depth generator and other downstream stages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, cam in result.cameras.items():
        # Strip file extension from frame_id for downstream compatibility
        frame_id = Path(name).stem if '.' in name else name
        entry = {
            "frame_id": frame_id,
            "K": np.asarray(cam.intrinsics, dtype=np.float64).tolist(),
            "R": np.asarray(cam.rotation, dtype=np.float64).tolist(),
            "t": np.asarray(cam.position, dtype=np.float64).tolist(),
        }
        gps = getattr(cam, "gps", None)
        if gps:
            entry["gps"] = {
                "lat": float(gps.get("lat") or gps.get("latitude")),
                "lon": float(gps.get("lon") or gps.get("longitude")),
                "alt": float(gps.get("alt") or gps.get("altitude", 0.0)),
            }
        frames.append(entry)
    with open(output_dir / "poses.json", "w") as f:
        json.dump({"frames": frames}, f, indent=2)
    log.info("poses_persisted", path=str(output_dir / "poses.json"), cameras=len(frames))


def _write_sparse_ply(result, output_dir: Path) -> None:
    """Persist the sparse 3D points as ``sparse_model.ply``."""
    if not result.points3d:
        return
    from app.services.pointcloud import PointCloud, save_ply

    xyz = np.vstack([p.position for p in result.points3d])
    colors = np.vstack([p.color for p in result.points3d])
    tracks = np.array([p.track_length for p in result.points3d], dtype=np.int32)
    reproj = np.array([p.mean_reproj_error for p in result.points3d], dtype=np.float64)
    conf = np.clip(tracks / 5.0, 0.0, 1.0).astype(np.float64)

    # Ghost screen at export (measured, physically derived — see the
    # SCENE_RANGE_TOLERANCE rationale in camera_pose_estimator). A point
    # whose minimum range to any registered camera exceeds the tolerance x
    # the camera-corridor extent sits at a range this flight never imaged;
    # far-field ghost matches on Flight_to_tower sat 2-9x beyond the
    # corridor while real structure stayed inside it.
    if result.cameras:
        centers = np.asarray([np.asarray(c.position, dtype=np.float64) for c in result.cameras.values()])
        if len(centers) >= 2:
            span = float(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))
            min_range = np.linalg.norm(xyz[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
            in_range = min_range <= SCENE_RANGE_TOLERANCE * span
            n_dropped = int((~in_range).sum())
            # The screen may only prune when a well-supported MAJORITY of the
            # reconstruction survives it: it assumes near-field structure
            # around the camera corridor, which holds for nadir GNSS-anchored
            # flights but inverts on far-field forward-motion scenes — there
            # the corridor under-spans the real scene and the screen would
            # keep a junk minority (London: 12,041 of 12,141 points dropped
            # from a healthy reproj-0.45 px reconstruction; the 106 kept
            # poisoned every view's conditioning and the depth stage refused
            # the whole run). Majority + track support + an absolute floor.
            kept_n = int(in_range.sum())
            min_track = int(np.percentile(tracks[in_range], 50)) if kept_n else 0
            majority_ok = kept_n >= max(0.5 * len(xyz), 100) and (
                n_dropped <= kept_n) and min_track >= 2
            if n_dropped and majority_ok:
                xyz, colors, tracks, reproj, conf = (
                    xyz[in_range], colors[in_range], tracks[in_range], reproj[in_range], conf[in_range])
                log.info("sparse_ghost_range_screen", dropped=n_dropped, kept=kept_n,
                         corridor_span_m=round(span, 1))
            elif n_dropped:
                log.warning("sparse_ghost_range_screen_refused",
                            note="screen would drop the majority of the reconstruction — corridor-relative "
                                 "range screen does not describe this scene shape; keeping all points",
                            dropped_would=n_dropped, kept_would=kept_n, total=len(xyz),
                            corridor_span_m=round(span, 1))
    save_ply(
        output_dir / "sparse_model.ply",
        PointCloud(xyz=xyz, rgb=colors, confidence=conf, observations=tracks, residual=reproj),
    )
    log.info("sparse_model_persisted", path=str(output_dir / "sparse_model.ply"), points=len(xyz))


def _track_statistics(result) -> dict:
    """Track-quality report (Parts 1-2): observation provenance + conditioning.

    Percentages are MEASURED from the reconstruction's own points. Gates are
    the constants the triangulator already enforces — reported here, not
    invented for this summary.
    """
    pts = result.points3d
    if not pts:
        return {"points": 0}
    obs = np.array([p.track_length for p in pts], dtype=np.int64)
    reproj = np.array([p.mean_reproj_error for p in pts], dtype=np.float64)
    angles = np.array([p.min_triangulation_angle_deg for p in pts], dtype=np.float64)
    rel_depth = np.array([p.rel_depth_uncertainty for p in pts], dtype=np.float64)
    n = len(pts)

    def _pct(mask: np.ndarray) -> float:
        return round(float(mask.mean() * 100.0), 2)

    from app.services.camera_pose_estimator import (
        MAX_RELATIVE_DEPTH_ERR,
        MAX_TRACK_REPROJ_PX,
        MIN_TRIANGULATION_ANGLE_DEG,
    )

    # Per-point frame_ids for the worst tracks (diagnostic payload, bounded).
    worst = np.argsort(reproj)[::-1][:20]
    worst_points = [
        {
            "point_id": int(pts[i].point_id),
            "frame_ids": [f for f, _ in pts[i].observations],
            "observations": int(pts[i].track_length),
            "mean_reproj_error_px": round(float(reproj[i]), 3),
            "min_triangulation_angle_deg": round(float(angles[i]), 4),
        }
        for i in worst
    ]

    return {
        "points": n,
        "observation_count_pct": {
            "2": _pct(obs == 2),
            "3": _pct(obs == 3),
            "4": _pct(obs == 4),
            "5": _pct(obs == 5),
            "6+": _pct(obs >= 6),
        },
        "observations_median": float(np.median(obs)),
        "observations_p95": float(np.percentile(obs, 95)),
        "observations_max": int(obs.max()),
        "reprojection_median_px": round(float(np.median(reproj)), 4),
        "reprojection_p95_px": round(float(np.percentile(reproj, 95)), 4),
        "min_triangulation_angle_median_deg": round(float(np.median(angles)), 4),
        "min_triangulation_angle_p05_deg": round(float(np.percentile(angles, 5)), 4),
        "gates": {
            "min_triangulation_angle_deg": MIN_TRIANGULATION_ANGLE_DEG,
            "max_track_reproj_px": MAX_TRACK_REPROJ_PX,
            "max_relative_depth_err": MAX_RELATIVE_DEPTH_ERR,
        },
        "gate_flags": {
            "low_angle_track_percent": _pct(angles < MIN_TRIANGULATION_ANGLE_DEG * 1.5),
            "high_reprojection_track_percent": _pct(reproj > MAX_TRACK_REPROJ_PX * 0.5),
            "uncertain_depth_track_percent": _pct(rel_depth > MAX_RELATIVE_DEPTH_ERR * 0.5),
        },
        "worst_tracks": worst_points,
    }


def _gps_statistics(result, gps_priors, ba_result=None) -> dict:
    """GPS provenance report (Part 8): what GPS entered the system and where
    it went. Labels are MEASURED counts, never inferred availability.

    Two counts here answer two different questions and used to read as a
    contradiction: ``gps_priors_supplied_to_ba`` counted the frame_id ->
    (lat, lon, alt) dict, which is EMPTY on the telemetry-placed path (the
    caller passes explicit metric ENU targets instead), while the BA stage
    reported ``gps_prior_cameras: 16`` for the same run. Both numbers were
    true about different inputs. Each now states its own meaning, and the
    BA's own source label is carried through so the pair cannot be misread.
    """
    cams = result.cameras
    with_gps = [c for c in cams.values() if getattr(c, "gps", None)]
    stats = {
        "registered_cameras": len(cams),
        "cameras_with_gps_record": len(with_gps),
        # Frames the flight log could supply geodetic coordinates for.
        "frames_with_geodetic_gps": len(gps_priors or {}),
        "poses_json_carries_gps": bool(with_gps),
    }
    if ba_result is not None:
        # What BA actually did with telemetry (the authoritative count).
        stats["ba_priors_supplied_from"] = getattr(ba_result, "gps_prior_source", "none")
        stats["ba_cameras_priored"] = int(getattr(ba_result, "gps_prior_cameras", 0) or 0)
        stats["ba_prior_rms_m"] = getattr(ba_result, "gps_prior_rms_m", None)
    if with_gps:
        lats = [c.gps["lat"] for c in with_gps]
        lons = [c.gps["lon"] for c in with_gps]
        alts = [c.gps.get("alt", 0.0) for c in with_gps]
        stats["lat_range"] = [round(min(lats), 6), round(max(lats), 6)]
        stats["lon_range"] = [round(min(lons), 6), round(max(lons), 6)]
        stats["alt_range_m"] = [round(min(alts), 2), round(max(alts), 2)]
    return stats
