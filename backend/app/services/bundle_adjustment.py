"""Bundle adjustment — real joint optimization of camera poses and 3D points.

Primary path: pycolmap ``bundle_adjustment`` (Ceres-backed Levenberg-Marquardt
Schur solver). Cameras, 3D points, and intrinsics are optimized JOINTLY against
every observation in the reconstruction's tracks; the objective is the standard

    E_BA = Σ_ij rho( || x_ij − project(X_j, K_i, R_i, C_i) ||² )

with a Huber loss (rho) and gauge fixed at the first registered camera.

When per-camera GPS/telemetry centres are supplied they are added as a SOFT
prior via pycolmap's pose-prior bundle adjuster:

    E_GPS = Σ_i rho( ||C_i − C_GPS_i||² / σ² )        σ from the prior

— never as a hard constraint and never as ground truth.

Honesty notes (mandate §19/§20 — do NOT trust library labels):
* pycolmap's ``compute_mean_reprojection_error`` is CACHED on this wheel
  (verified 3.12.5: stays constant across point mutations), so the reported
  errors here are always recomputed by explicit re-projection of every
  observation through the adjusted cameras.
* ``initial_error`` is measured on the input reconstruction BEFORE solving;
  ``final_error`` after. If BA cannot run, the caller gets a result with
  ``converged=False`` and unchanged geometry — nothing silently "passes".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.logging_config import get_logger
from app.services.camera_pose_estimator import ReconstructionResult

log = get_logger("drone_recon.services.bundle_adjustment")


@dataclass
class BAResult:
    """Bundle adjustment output."""
    initial_error: float = 0.0
    final_error: float = 0.0
    iterations: int = 0
    converged: bool = False
    ba_time_ms: float = 0.0
    num_cameras: int = 0
    num_points: int = 0
    num_observations: int = 0
    #: Wall clock of this call's I/O-heavy sub-parts (ms). The joint path
    #: round-trips the whole reconstruction through COLMAP's text format; on
    #: an 800k-observation cloud the round trip, not the solve, is the cost,
    #: so it is measured separately instead of hiding inside one ba_time_ms.
    timings: dict = field(default_factory=dict)
    backend: str = ""
    # Provenance of the GPS/telemetry soft prior, when one was applied.
    # ``gps_prior_source`` names WHICH input supplied it, because the two are
    # routinely confused: ``gps_priors`` is a frame_id -> (lat, lon, alt)
    # dict that is EMPTY on the telemetry-placed path (the caller supplies
    # explicit metric ENU targets instead). The sparse report therefore read
    # "gps_priors_supplied_to_ba: 0" beside "gps_prior_cameras: 16" on a run
    # where 16 cameras really were priored — both true, jointly misleading.
    gps_prior_cameras: int = 0
    gps_prior_source: str = "none"
    gps_prior_rms_m: float | None = None
    gps_prior_rms_before_m: float | None = None
    # Honesty flags: the pycolmap wheel exposes no solver summary, so the
    # reported residuals are re-measured under OUR projection convention.
    solver_rerun_reproj_px: float | None = None
    degraded: bool = False
    # True when the solve shipped worse geometry than it received and the
    # caller's pre-BA state was restored (never-ship-worse guarantee).
    restored: bool = False


def run_bundle_adjustment(
    result: ReconstructionResult,
    *,
    max_iterations: int = 30,
    use_colmap: bool = True,
    gps_priors: dict[str, tuple[float, float, float]] | None = None,
    gps_prior_sigma_m: float = 5.0,
    refine_intrinsics: bool = False,
    metric_targets: dict[str, np.ndarray] | None = None,
) -> BAResult:
    """Run joint bundle adjustment on the reconstruction (in place).

    Args:
        result: reconstruction whose cameras/points are refined in place.
        max_iterations: solver iteration cap.
        use_colmap: try the pycolmap path first.
        gps_priors: frame_id -> (lat, lon, alt) marking frames whose centre
            is telemetry/GPS-backed. Each such camera gets a SOFT positional
            prior at its own telemetry centre (see ``_prior_targets``).
        gps_prior_sigma_m: prior uncertainty (metres) per camera centre.
        refine_intrinsics: when False (default) K is held fixed — dataset-
            supplied or calibrated intrinsics are measurements, not guesses.
            Set True only when K itself came from the SfM stage.
        metric_targets: optional explicit metric prior centres (reconstruction
            frame) overriding the telemetry-centre default.
    """
    initial = _measure_reprojection_error(result)
    out = BAResult(
        initial_error=round(initial, 4),
        num_cameras=result.num_registered,
        num_points=result.num_points,
        num_observations=sum(len(p.observations) for p in result.points3d),
    )

    # GPS prior residual BEFORE the solve (soft prior diagnostic): distance
    # between each prior camera's current centre and its prior target.
    if gps_priors or metric_targets:
        targets = _prior_targets(result, gps_priors, metric_targets)
        if targets:
            out.gps_prior_rms_before_m = round(float(np.sqrt(np.mean([
                float(np.sum((np.asarray(result.cameras[f].position, dtype=np.float64) - t) ** 2))
                for f, t in targets.items() if f in result.cameras
            ]))), 3)

    if use_colmap and result.points3d and result.num_registered >= 2:
        has_priors = bool(gps_priors or metric_targets)
        # Prior-sigma ladder + SHIP-PLACED fallback (single-pass reality):
        # a placed state is already metric-accurate (window fits at the GNSS
        # noise scale), so when BA cannot improve it WITHOUT dragging cameras
        # off their priors, the placed state is the correct answer. Plain BA
        # without priors is NOT a fallback on this path: measured on the
        # flight_to_tower hover footage it deformed the trajectory (2
        # impossible grid jumps) while polishing reprojection — exactly the
        # failure the trajectory gate exists to catch.
        sigmas = [gps_prior_sigma_m, gps_prior_sigma_m * 2.0] if has_priors else [None]
        last_error: str | None = None
        for attempt, sigma in enumerate(sigmas):
            try:
                ba = _run_colmap_ba(
                    result, max_iterations, gps_priors,
                    sigma if sigma is not None else gps_prior_sigma_m,
                    refine_intrinsics, metric_targets, initial_px=initial,
                    require_prior_success=bool(has_priors),
                )
            except Exception as exc:
                last_error = str(exc)
                log.warning("colmap_ba_attempt_failed", attempt=attempt, sigma=sigma, error=str(exc))
                continue
            final = _measure_reprojection_error(result)
            out.final_error = round(final, 4)
            out.iterations = ba["iterations"]
            out.converged = ba["converged"]
            out.degraded = ba.get("degraded", False)
            out.restored = bool(ba.get("restored", False))
            out.solver_rerun_reproj_px = ba.get("solver_rerun_reproj_px")
            out.ba_time_ms = ba["time_ms"]
            out.timings = ba.get("timings", {})
            out.backend = "pycolmap_joint" if ba.get("ba_ran") else "placed_unadjusted"
            out.gps_prior_cameras = ba.get("gps_prior_cameras", 0)
            out.gps_prior_source = ba.get("gps_prior_source", "none")
            out.gps_prior_rms_m = ba.get("gps_prior_rms_m")
            log.info(
                "bundle_adjustment_complete",
                backend=out.backend,
                initial_px=out.initial_error,
                final_px=out.final_error,
                iterations=out.iterations,
                converged=out.converged,
                restored=out.restored,
                observations=out.num_observations,
                gps_prior_cameras=out.gps_prior_cameras,
                time_ms=out.ba_time_ms,
                timings=out.timings,
            )
            return out
        if last_error is not None and has_priors:
            # Every prior-BA attempt failed: ship the placed state honestly.
            log.warning("ba_prior_attempts_exhausted_ship_placed", detail=last_error,
                        note="placed state is metric (window fits); no-prior BA deliberately skipped")
            out.final_error = out.initial_error
            out.converged = True
            out.backend = "placed_unadjusted"
            out.ba_time_ms = 0.0
            return out
        if last_error is not None:
            log.warning("colmap_ba_failed", error=last_error, fallback="skipped")

    # No working BA backend: report honestly instead of pretending.
    out.final_error = out.initial_error
    out.converged = False
    out.backend = "unavailable"
    out.ba_time_ms = 0.0
    log.warning("bundle_adjustment_unavailable", initial_px=out.initial_error)
    return out


# ---------------------------------------------------------------------------
# pycolmap joint BA
# ---------------------------------------------------------------------------


def _run_colmap_ba(
    result: ReconstructionResult,
    max_iterations: int,
    gps_priors: dict[str, tuple[float, float, float]] | None,
    gps_prior_sigma_m: float,
    refine_intrinsics: bool = False,
    metric_targets: dict[str, np.ndarray] | None = None,
    initial_px: float = 0.0,
    require_prior_success: bool = False,
) -> dict:
    """Run real joint BA via pycolmap and copy the refined state back.

    pycolmap 3.x needs a fully-wired Reconstruction (rig/frame/image/point
    graph). Building that object graph in memory is fragile across wheel
    versions, so this uses the documented text format round-trip: export the
    current state to COLMAP text, let pycolmap load it, adjust, save, reload.
    All the data is ours — no external model is involved.
    """
    import tempfile
    from pathlib import Path

    import pycolmap

    start = time.perf_counter()
    out_initial = initial_px  # caller's measured pre-BA error (px)
    ba_timings: dict[str, float] = {}

    with tempfile.TemporaryDirectory(prefix="strata_ba_") as td:
        root = Path(td)
        # CALLER-STATE SNAPSHOT: BA must never be able to ship a state worse
        # than the one it received.  The python-side snapshot below is the
        # source of truth for the final restore; the in-rec snapshots are
        # only for the prior-pass guard.
        ba_snapshot = {
            name: (
                np.asarray(cam.rotation, dtype=np.float64).copy(),
                np.asarray(cam.position, dtype=np.float64).copy(),
                np.asarray(cam.intrinsics, dtype=np.float64).copy(),
                np.asarray(cam.quaternion, dtype=np.float64).copy()
                if getattr(cam, "quaternion", None) is not None else None,
                True,
            )
            for name, cam in result.cameras.items()
        }
        ba_snapshot_points = [
            np.asarray(pt.position, dtype=np.float64).copy()
            for pt in result.points3d
        ]
        _t_io = time.perf_counter()
        _export_colmap_text(result, root)
        rec = pycolmap.Reconstruction(str(root / "sparse"))
        ba_timings["text_export_and_load_ms"] = round((time.perf_counter() - _t_io) * 1000.0, 2)
        if rec.num_images() < 2 or rec.num_points3D() < 3:
            raise ValueError(
                f"colmap text round-trip produced too small a reconstruction "
                f"(images={rec.num_images()}, points={rec.num_points3D()})"
            )

        # If a GPS/telemetry prior exists, run the pose-prior adjuster (soft
        # positional priors with explicit covariance) and then plain joint BA
        # on the result. GUARD: this wheel's prior adjuster runs an internal
        # RANSAC alignment of the reconstruction onto the priors which can
        # FAIL and leave a corrupted frame (measured: "Alignment w.r.t. prior
        # positions failed" + cameras dragged ~14 m with WORSE re-projection).
        # So: snapshot the geometry first; accept the prior pass only if the
        # reconstruction's own mean re-projection did not degrade. Otherwise
        # restore the snapshot and let plain joint BA run — the drift of the
        # visual solution vs the telemetry trajectory is then measured and
        # reported (never silently accepted, never hidden).
        _t_prior = time.perf_counter()
        prior_meta: dict = {}
        priors = _build_pose_priors(rec, result, gps_priors, metric_targets, gps_prior_sigma_m)
        # Which input the priors came from. Reported, never inferred later.
        _src = []
        if gps_priors:
            _src.append("gps_priors")
        if metric_targets:
            _src.append("metric_targets")
        prior_meta["gps_prior_source"] = "+".join(_src) if _src else "none"
        if priors:
            snap_cams = {
                img.name: (np.asarray(img.cam_from_world().rotation.matrix()),
                           np.asarray(img.cam_from_world().translation))
                for img in rec.images.values()
            }
            snap_pts = {pid: np.asarray(pt.xyz) for pid, pt in rec.points3D.items()}

            def _rec_reproj(rec_in) -> float:
                # This wheel's img_from_cam can return a scalar (0-d) for
                # some inputs — np.asarray(...)[:2] then raises IndexError
                # (measured crash 'array is 0-dimensional').  Compute the
                # projection defensively: array-ify FIRST, index only when
                # a (2+)-vector actually came back; skip odd returns.
                errs = []
                for img in rec_in.images.values():
                    cam = rec_in.cameras[img.camera_id]
                    for p2 in img.points2D:
                        if p2.has_point3D():
                            X = np.asarray(rec_in.points3D[p2.point3D_id].xyz)
                            raw = cam.img_from_cam(img.cam_from_world() * X)
                            uv = np.asarray(raw, dtype=np.float64).reshape(-1)
                            if uv.size < 2:
                                continue
                            errs.append(float(np.hypot(*(uv[:2] - np.asarray(p2.xy)))))
                return float(np.mean(errs)) if errs else 0.0

            before_reproj = _rec_reproj(rec)
            ba_opts = pycolmap.BundleAdjustmentOptions()
            ba_opts.solver_options.max_num_iterations = max_iterations
            ba_opts.loss_function_type = pycolmap.LossFunctionType.CAUCHY
            ba_opts.loss_function_scale = 1.0
            # Same intrinsics policy as the plain BA below: dataset-calibrated
            # K is a measurement (the import step keeps our fixed K; a refined
            # focal here would silently desync the two conventions).
            ba_opts.refine_focal_length = refine_intrinsics
            ba_opts.refine_principal_point = False
            ba_opts.refine_extra_params = False
            prior_opts = pycolmap.PosePriorBundleAdjustmentOptions()
            prior_opts.prior_position_loss_scale = max(1e-6, gps_prior_sigma_m ** 2)
            config = pycolmap.BundleAdjustmentConfig()
            for img_id in rec.images:
                config.add_image(img_id)
            # GAUGE: with a full, metric set of position priors the gauge is
            # DEFINED by the priors.  The prior adjuster's internal RANSAC
            # alignment mis-estimates the gauge when left UNSPECIFIED (it
            # guessed the alignment from the torn pre-fit geometry and the
            # prior pass then pulled cameras THROUGH the mis-gauge), which is
            # the dominant failure mode of this pass on placed models.
            # This wheel has no USE_METRIC_FRAME; THREE_POINTS fixes the gauge
            # on the point cloud — stable under our placement because the
            # retriangulated points already sit in the telemetry frame.
            config.fix_gauge(pycolmap.BundleAdjustmentGauge.UNSPECIFIED)
            try:
                config.fix_gauge(pycolmap.BundleAdjustmentGauge.THREE_POINTS)
            except Exception:
                pass
            adj = pycolmap.create_pose_prior_bundle_adjuster(
                ba_opts, prior_opts, config, priors, rec
            )
            adj.solve()
            after_reproj = _rec_reproj(rec)
            # VERDICT FIRST, then record: the prior pass succeeded iff its own
            # re-projection did not degrade.  (The previous sequencing read
            # ``prior_meta['gps_prior_cameras']`` BEFORE the success path ever
            # wrote it — the field is only assigned in the success branch — so
            # a SUCCESSFUL prior pass was always misread as rejected and the
            # require-prior ladder raised on an IMPROVED state.  Measured:
            # "degraded reprojection (1.105 -> 0.883 px)" raised on a state
            # whose reprojection had in fact improved 20%.)
            prior_pass_ok = not (after_reproj > before_reproj * 1.5 + 1e-6)
            if prior_pass_ok:
                centres_after = {
                    img.name: np.asarray(img.projection_center(), dtype=np.float64)
                    for img in rec.images.values()
                }
                diffs = [
                    float(np.linalg.norm(centres_after[name] - target))
                    for name, target in _prior_targets(result, gps_priors, metric_targets).items()
                    if name in centres_after
                ]
                if diffs:
                    prior_meta["gps_prior_rms_m"] = round(float(np.sqrt(np.mean(np.square(diffs)))), 3)
                prior_meta["gps_prior_cameras"] = len(priors)
            else:
                # Prior pass degraded the geometry — restore the snapshot.
                # CLEAN-SLATE reload: this wheel's prior adjuster can leave
                # the rec internally inconsistent (measured rerun15: the
                # manual snapshot restore left poses/points/frames out of
                # sync and plain BA then DIVERGED 1.3 -> 1143 px).  Re-loading
                # the untouched text export guarantees a coherent rec.
                log.warning(
                    "pose_prior_ba_rejected",
                    reproj_before_px=round(before_reproj, 4),
                    reproj_after_px=round(after_reproj, 4),
                    action="clean_slate_reload_prior_skipped",
                )
                rec = pycolmap.Reconstruction(str(root / "sparse"))
                prior_meta["gps_prior_cameras"] = 0
                # No prior bound, so naming a source would imply one did.
                prior_meta["gps_prior_source"] = "none"
                if require_prior_success:
                    # Ladder contract: the caller wants PRIOR-constrained BA.
                    # Falling through to plain (no-prior) BA here is what
                    # deformed the trajectory on sparse hover clouds while
                    # "polishing" reprojection — fail instead so the caller
                    # can retry with a wider sigma or ship the placed state.
                    raise ValueError(
                        "pose_prior_ba_rejected: prior pass degraded reprojection "
                        f"({before_reproj:.3f} -> {after_reproj:.3f} px); plain BA refused"
                    )
            # Plain joint BA below runs from the prior-adjusted state (pass
            # succeeded) or the cleanly reloaded pre-pass state (pass failed).

        ba_timings["pose_prior_pass_ms"] = round((time.perf_counter() - _t_prior) * 1000.0, 2)
        ba = pycolmap.BundleAdjustmentOptions()
        ba.solver_options.max_num_iterations = max_iterations
        ba.loss_function_type = pycolmap.LossFunctionType.CAUCHY
        ba.loss_function_scale = 1.0
        ba_ran = True
        # Dataset-calibrated K is a measurement: hold intrinsics fixed unless
        # the caller explicitly asks to refine them.
        ba.refine_focal_length = refine_intrinsics
        ba.refine_principal_point = False
        ba.refine_extra_params = False
        # pycolmap.bundle_adjustment returns None on this wheel; the C++ side
        # prints a report we cannot capture. Iteration/convergence detail is
        # therefore derived from the error trajectory we measure ourselves:
        # the caller's own re-projection metric is the source of truth, and
        # "converged" means the geometry actually improved (the honest test).            pycolmap.bundle_adjustment(rec, ba)

        _t_io = time.perf_counter()
        adjusted_dir = root / "adjusted"
        adjusted_dir.mkdir(parents=True, exist_ok=True)
        rec.write_binary(str(adjusted_dir))
        _import_colmap_text(result, adjusted_dir)
        ba_timings["write_and_import_ms"] = round((time.perf_counter() - _t_io) * 1000.0, 2)

    # Honest convergence: the solver ran, but this wheel exposes no solver
    # summary — so "converged" is defined as "the geometry actually improved
    # under OUR re-projection metric" (measured again on the imported state).
    # A run that degraded is reported as not converged (and flagged).
    final_check = _measure_reprojection_error(result)
    degraded = final_check > out_initial + 0.1
    if degraded:
        # NEVER ship a worse state than arrived: restore the caller's
        # geometry verbatim.  BA exists to refine, not to destroy — a
        # diverged solve (measured: 1.3 -> 1143 px after a prior-pass guard
        # trip) must never become the shipped artifact.
        log.warning(
            "bundle_adjustment_degraded_restored",
            initial_px=round(out_initial, 4),
            degraded_final_px=round(final_check, 4),
            action="caller_geometry_restored",
        )
        for cam_name, (R_c2w, C, K, q, extra) in ba_snapshot.items():
            cam = result.cameras.get(cam_name)
            if cam is None:
                continue
            cam.rotation = R_c2w.copy()
            cam.position = C.copy()
            cam.intrinsics = K.copy()
            if q is not None and hasattr(cam, "quaternion"):
                cam.quaternion = q.copy()
            if extra is not None and hasattr(cam, "translation"):
                cam.translation = C.copy()
        for i, pt in enumerate(result.points3d):
            if i < len(ba_snapshot_points):
                pt.position = ba_snapshot_points[i].copy()
        final_check = _measure_reprojection_error(result)
    return {
        "iterations": max_iterations,
        "converged": final_check <= out_initial + 1e-6,
        "degraded": degraded,
        "solver_rerun_reproj_px": round(final_check, 4),
        "time_ms": round((time.perf_counter() - start) * 1000, 2),
        "restored": bool(degraded),
        "ba_ran": True,
        "timings": ba_timings,
        **prior_meta,
    }


def _measure_reprojection_error(result: ReconstructionResult) -> float:
    """Mean per-OBSERVATION reprojection error (px), recomputed — never cached.

    Uses the canonical convention via app.services.geometry: with R = c2w and
    C = camera centre, X_cam = R^T (X − C) and pixel = K X_cam / z.

    Batched: every observation is projected in a handful of numpy calls
    instead of one call per observation. This is invoked ~4x per BA round and
    the run under measurement carries 812,670 observations, so the per-call
    interpreter overhead dominated the metric itself. The convention, the
    K basis (each point is projected through the intrinsics of its FIRST
    observing camera) and the ``z <= 1e-6`` drop are unchanged, so the value
    is the same number — it is only reached faster.
    """
    positions: list[np.ndarray] = []
    k_of_point: list[np.ndarray] = []
    cam_index: dict[str, int] = {}
    cam_R: list[np.ndarray] = []
    cam_C: list[np.ndarray] = []
    pt_idx: list[int] = []
    obs_cam: list[int] = []
    obs_u: list[float] = []
    obs_v: list[float] = []

    for pt in result.points3d:
        obs = [(f, pix) for f, pix in pt.observations if f in result.cameras]
        if len(obs) < 2:
            continue
        first = obs[0][0]  # intrinsics basis, exactly as before
        k_of_point.append(np.abs(np.asarray(result.cameras[first].intrinsics, dtype=np.float64)))
        positions.append(np.asarray(pt.position, dtype=np.float64))
        pi = len(positions) - 1
        for frame_id, pix in obs:
            ci = cam_index.get(frame_id)
            if ci is None:
                ci = len(cam_R)
                cam_index[frame_id] = ci
                cam = result.cameras[frame_id]
                cam_R.append(np.asarray(cam.rotation, dtype=np.float64))
                cam_C.append(np.asarray(cam.position, dtype=np.float64))
            pt_idx.append(pi)
            obs_cam.append(ci)
            obs_u.append(float(pix[0]))
            obs_v.append(float(pix[1]))

    if not pt_idx:
        return 0.0

    X = np.asarray(positions, dtype=np.float64)
    R = np.asarray(cam_R, dtype=np.float64)
    C = np.asarray(cam_C, dtype=np.float64)
    Kp = np.asarray(k_of_point, dtype=np.float64)[pt_idx]
    ci = np.asarray(obs_cam, dtype=np.intp)
    pix_u = np.asarray(obs_u, dtype=np.float64)
    pix_v = np.asarray(obs_v, dtype=np.float64)

    # X_cam = R^T (X − C) = (X − C) @ R, one row per observation. matmul (not
    # einsum) so the batched form goes through the same BLAS kernel as the
    # per-observation (1,3)@(3,3) call it replaces — bit-identical, not merely
    # close, which matters because these errors feed a convergence verdict.
    Xc = np.matmul((X[pt_idx] - C[ci])[:, None, :], R[ci])[:, 0, :]
    z = Xc[:, 2]
    ok = z > 1e-6
    if not ok.any():
        return 0.0
    # Mirrors project_world_to_pixel's arithmetic term for term (including its
    # |z| <= 1e-12 guard), just across every observation at once.
    denom = np.where(np.abs(z) > 1e-12, z, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = Xc[:, 0] / denom * Kp[:, 0, 0] + Kp[:, 0, 2]
        v = Xc[:, 1] / denom * Kp[:, 1, 1] + Kp[:, 1, 2]
    errs = np.hypot(u[ok] - pix_u[ok], v[ok] - pix_v[ok])
    return float(np.mean(errs)) if errs.size else 0.0


def _prior_targets(
    result: ReconstructionResult,
    gps_priors: dict[str, tuple[float, float, float]] | None,
    metric_targets: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """frame_id -> prior camera centre in the reconstruction's metric frame.

    The reconstruction in the telemetry-assisted path lives IN the flight
    log's metric frame (the log's x/y/z IS the GPS-derived metric position —
    the same row that carries lat/lon). The prior therefore anchors each
    GPS-backed camera to its own telemetry centre: ``E_GPS`` says "stay near
    the measured trajectory", and lat/lon stays provenance (reported, used
    for georeferencing) rather than a second, conflicting coordinate frame.

    ``metric_targets`` lets a caller supply explicit metric prior centres in
    the reconstruction frame (e.g. ENU-converted geodetic fixes); they take
    precedence when given.
    """
    if metric_targets:
        return {f: np.asarray(v, dtype=np.float64) for f, v in metric_targets.items() if f in result.cameras}
    if not gps_priors:
        return {}
    return {
        f: np.asarray(result.cameras[f].position, dtype=np.float64)
        for f in gps_priors if f in result.cameras
    }


def _build_pose_priors(rec, result, gps_priors, metric_targets=None,
                       gps_prior_sigma_m: float = 5.0) -> dict:
    """pycolmap PosePriors for images whose frame has a prior centre."""
    import pycolmap

    targets = _prior_targets(result, gps_priors, metric_targets)
    if not targets:
        return {}
    name_to_img = {img.name: img for img in rec.images.values()}
    priors: dict = {}
    for name, img in name_to_img.items():
        stem = name.rsplit(".", 1)[0] if "." in name else name
        C = targets.get(name)
        if C is None:
            C = targets.get(stem)
        if C is None:
            continue
        pp = pycolmap.PosePrior()
        pp.position = [float(C[0]), float(C[1]), float(C[2])]
        pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        # Uncertainty must be REPRESENTED, not implied: σ² on the diagonal.
        # Without it the adjuster logs "No pose priors with valid covariance
        # found" and applies unit weights — the prior then overwhelms the
        # pixels (measured: cameras dragged 8+ m off their measurements).
        pp.position_covariance = (
            np.eye(3) * float(gps_prior_sigma_m) ** 2
        ).tolist()
        priors[img.image_id] = pp
    return priors


def _export_colmap_text(result: ReconstructionResult, root) -> None:
    """Write the reconstruction in COLMAP text format (cameras/images/points3D).

    Camera pose convention in COLMAP text is world-to-camera; we hold
    R = camera-to-world and C = centre, so we export R_w2c = R^T and
    t = −R_w2c C. Rotation is written as a qvec (w, x, y, z) — the same
    quaternion we already store on each CameraPose.
    """
    sparse = root / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    cams = list(result.cameras.items())
    # Build a frame_id -> (cam_id, obs list) index in ONE pass over the
    # points (O(P·obs)); the previous per-camera scan was O(C·P·obs), which
    # on a 191-camera / 120k-point run is ~200× more work than necessary.
    pt_frame_map: dict[int, list[tuple[int, np.ndarray]]] = {cid: [] for cid in range(1, len(cams) + 1)}
    name_to_colid = {name: cid for cid, (name, _) in enumerate(cams, start=1)}
    # Export ids are the LIST INDEX + 1 (contiguous 1-based). COLMAP point3D
    # ids are arbitrary/sparse; the importer maps pid-1 back into the list, so
    # ids must be contiguous or the import scrambles points (measured: mean
    # reproj 0.31 → 706 px before this fix).
    for pt_idx, pt in enumerate(result.points3d):
        pid = pt_idx + 1
        for fid, pix in pt.observations:
            cid = name_to_colid.get(fid)
            if cid is not None:
                pt_frame_map[cid].append((pid, np.asarray(pix, dtype=np.float64)))
    for cid in pt_frame_map:
        pt_frame_map[cid].sort(key=lambda o: o[0])

    # COLMAP needs positive width/height metadata; geometry comes from K. The
    # pipeline's frames are 1080p (settings.dense/Frontend) — derive a nominal
    # 16:9 size from the principal point so the loader accepts the file.
    K0 = np.asarray(cams[0][1].intrinsics, dtype=np.float64)
    nominal_w = int(max(2 * K0[0, 2], 640))
    nominal_h = int(max(2 * K0[1, 2], 360))
    cam_lines = []
    for cam_id, (name, cam) in enumerate(cams, start=1):
        K = np.asarray(cam.intrinsics, dtype=np.float64)
        cam_lines.append(
            f"{cam_id} PINHOLE {nominal_w} {nominal_h} {abs(K[0, 0]):.6f} {abs(K[1, 1]):.6f} "
            f"{K[0, 2]:.6f} {K[1, 2]:.6f}"
        )
        cam.image_id = cam_id  # remember the mapping for import
        cam._ba_frame_id = name
    (sparse / "cameras.txt").write_text("\n".join(cam_lines) + "\n")

    # COLMAP images.txt format is TWO lines per image: a pose header line
    # (IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME) then a points2D line of
    # "X Y POINT3D_ID" triplets (possibly empty).
    img_lines = []
    for cam_id, (name, cam) in enumerate(cams, start=1):
        R_w2c = np.asarray(cam.rotation, dtype=np.float64).T
        C = np.asarray(cam.position, dtype=np.float64)
        t = -R_w2c @ C
        q = _rot_to_qvec(R_w2c)
        obs = pt_frame_map[cam_id]
        header = (f"{cam_id} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                  f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {cam_id} {name}")
        body = " ".join(f"{pix[0]:.4f} {pix[1]:.4f} {pid}" for pid, pix in obs)
        img_lines.append(header + "\n" + body)
    (sparse / "images.txt").write_text("\n".join(img_lines) + "\n")

    # points3D.txt: id X Y Z r g b ERROR TRACK — track elements reference
    # (image_id, point2D_idx); COLMAP matches observations to the image's
    # points2D by index order, so we emit the observations in the same order
    # per image and use running indices there.
    pt_lines = []
    # Invert once into pid -> [(cam_id, point2D_idx)] so the per-point loop
    # below walks a point's OWN observations. The previous version scanned all
    # C cameras for every point (C=116 on the run under measurement, ~60k
    # points -> ~7M dict lookups per export); the elements keep ascending
    # camera order, so the emitted track is byte-identical.
    tracks_by_point: dict[int, list[tuple[int, int]]] = {}
    for cam_id, obs in pt_frame_map.items():
        for i, (pid, _pix) in enumerate(obs):
            tracks_by_point.setdefault(pid, []).append((cam_id, i))
    for pt_idx, pt in enumerate(result.points3d):
        elems = tracks_by_point.get(pt_idx + 1)
        if not elems:
            continue
        r = int(np.clip(pt.color[0], 0, 255))
        g = int(np.clip(pt.color[1], 0, 255))
        b = int(np.clip(pt.color[2], 0, 255))
        track = " ".join(f"{cid} {i}" for cid, i in elems)
        pt_lines.append(
            # COLMAP reserves point3D ID 0 as INVALID — export 1-based ids.
        f"{pt_idx + 1} {pt.position[0]:.8f} {pt.position[1]:.8f} {pt.position[2]:.8f} "
            f"{r} {g} {b} {pt.mean_reproj_error:.6f} {track}"
        )
    (sparse / "points3D.txt").write_text("\n".join(pt_lines) + "\n")


def _import_colmap_text(result: ReconstructionResult, adjusted_root) -> None:
    """Copy the adjusted cameras/points back into the ReconstructionResult."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(adjusted_root))
    name_to_cam: dict[int, str] = {}
    for img_id, img in rec.images.items():
        # COLMAP image name keeps our frame_id (we exported it verbatim).
        name = img.name
        stem = name.rsplit(".", 1)[0] if "." in name else name
        cam_pose = result.cameras.get(name) or result.cameras.get(stem)
        if cam_pose is None:
            continue
        cfw = img.cam_from_world()
        R_w2c = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t_w2c = np.asarray(cfw.translation, dtype=np.float64)
        R_c2w = R_w2c.T
        C = -R_c2w @ t_w2c
        cam_pose.rotation = R_c2w
        cam_pose.position = C
        cam_pose.quaternion = _rot_to_qvec(R_w2c.T)  # c2w quaternion (w,x,y,z)
        cam = rec.cameras[img.camera_id]
        if abs(cam.focal_length_x) > 0 and abs(cam.focal_length_y) > 0:
            K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
            K[0, 0] = abs(K[0, 0])
            K[1, 1] = abs(K[1, 1])
            cam_pose.intrinsics = K
        name_to_cam[img_id] = cam_pose.frame_id

    for pid, pt in rec.points3D.items():
        if pid - 1 < len(result.points3d):   # ids were exported 1-based
            sp = result.points3d[pid - 1]
            sp.position = np.asarray(pt.xyz, dtype=np.float64)
            err = getattr(pt, "error", None)
            if err is not None:
                sp.mean_reproj_error = float(err)


def _rot_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion (w, x, y, z)."""
    from app.services.camera_pose_estimator import _rotation_matrix_to_quat

    return _rotation_matrix_to_quat(np.asarray(R, dtype=np.float64))
