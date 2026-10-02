"""Camera pose estimation — PyCOLMAP incremental SfM with OpenCV fallback.

Estimates camera intrinsics, extrinsics, and 3D point positions from
matched features across image pairs.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from app.config.settings import settings
from app.logging_config import get_logger

#: World up direction for the telemetry frame (MovingDrone ENU: +z is up —
#: the same frame the flight log's x/y/z and altitude live in).
WORLD_UP = np.array([0.0, 0.0, 1.0])
from app.services.image_files import list_image_files

log = get_logger("drone_recon.services.camera_pose_estimator")


def _colmap_device():
    """Best COLMAP compute device: cuda when torch sees a GPU, else cpu.

    Depth Anything and COLMAP SIFT share the same CUDA availability signal,
    so one probe drives both. Falls back to CPU transparently — identical
    extraction/matching semantics, only throughput changes.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return pycolmap.Device.cuda
    except ImportError:
        pass
    return pycolmap.Device.cpu


def _extract_features_with_fallback(db_path: Path, image_dir: Path, sift_options, camera_mode, camera_model: str) -> str:
    """extract_features preferring CUDA SiftGPU, falling back to CPU.

    Returns the device actually used ("cuda" | "cpu"). CPU-only pycolmap
    wheels raise when handed Device.cuda — that is an environment property,
    not a caller error, so it downgrades with a log line instead of failing
    the run.
    """
    device = _colmap_device()
    if device == pycolmap.Device.cuda:
        try:
            pycolmap.extract_features(
                database_path=str(db_path),
                image_path=str(image_dir),
                camera_mode=camera_mode,
                camera_model=camera_model,
                sift_options=sift_options,
                device=device,
            )
            return "cuda"
        except RuntimeError as exc:
            log.warning("colmap_sift_gpu_unavailable_falling_back_cpu", error=str(exc)[:200])
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(image_dir),
        camera_mode=camera_mode,
        camera_model=camera_model,
        sift_options=sift_options,
    )
    return "cpu"


def _match_with_fallback(db_path: Path, matcher_type: str, sift_match_options) -> str:
    """match_* preferring CUDA, falling back to CPU. Returns device used."""
    device = _colmap_device()
    match_fn = pycolmap.match_exhaustive if matcher_type == "exhaustive" else pycolmap.match_sequential
    if device == pycolmap.Device.cuda:
        try:
            match_fn(database_path=str(db_path), sift_options=sift_match_options, device=device)
            return "cuda"
        except RuntimeError as exc:
            log.warning("colmap_match_gpu_unavailable_falling_back_cpu", error=str(exc)[:200])
    match_fn(database_path=str(db_path), sift_options=sift_match_options)
    return "cpu"



@dataclass
class CameraPose:
    """Estimated pose for a single camera."""
    image_id: int
    frame_id: str
    position: np.ndarray  # (3,) — translation in world coords
    rotation: np.ndarray  # (3,3) — rotation matrix
    quaternion: np.ndarray  # (4,) — w, x, y, z
    intrinsics: np.ndarray  # (3,3) — camera matrix
    distortion: np.ndarray  # distortion coefficients
    is_estimated: bool = False


@dataclass
class SparsePoint3D:
    """A single 3D point in the reconstruction."""
    point_id: int
    position: np.ndarray  # (3,)
    color: np.ndarray  # (3,) uint8 RGB
    track_length: int = 0
    mean_reproj_error: float = 0.0
    # Observation-level provenance (Part 1): every observing frame and the
    # pixel the track was measured at. Kept for track diagnostics and
    # bundle adjustment; never exported into the PLY itself.
    observations: list[tuple[str, np.ndarray]] = field(default_factory=list)
    # Conditioning statistics (Parts 2/16): smallest observing baseline's
    # subtended angle and the point's relative depth uncertainty (e/(f·θ)).
    min_triangulation_angle_deg: float = 0.0
    rel_depth_uncertainty: float = 0.0


@dataclass
class ReconstructionResult:
    """Full result of sparse reconstruction."""
    cameras: dict[str, CameraPose] = field(default_factory=dict)
    points3d: list[SparsePoint3D] = field(default_factory=list)
    image_pairs: list[tuple[str, str]] = field(default_factory=list)
    num_registered: int = 0
    num_points: int = 0
    mean_reproj_error: float = 0.0
    reconstruction_time_ms: float = 0.0
    backend: str = "colmap"
    # Telemetry-assisted localization provenance (empty for video-only SfM).
    telemetry_assisted: bool = False
    telemetry_source: str = ""
    path_span: float = 0.0
    median_scene_distance: float = 0.0
    # Mandated sparse quality-gate report (None when no telemetry pass ran).
    quality_gates: dict | None = None
    # Compute devices actually used for SIFT extraction and matching
    # ("cuda" | "cpu") — provenance for GPU/CPU benchmarking.
    detail: dict = field(default_factory=dict)
    # Per-frame feature counts (COLMAP DB or OpenCV extractor — whichever ran).
    feature_counts: dict[str, int] = field(default_factory=dict)


def estimate_poses(
    selected_dir: Path,
    project_id: str,
    *,
    use_colmap: bool = True,
    camera_model: str = "PINHOLE",
    frame_files: list[Path] | None = None,
) -> ReconstructionResult:
    """Run full sparse reconstruction on selected frames.

    Tries PyCOLMAP first, falls back to OpenCV incremental estimation.
    ``frame_files`` (ordered) is threaded to the COLMAP backend so it can
    report the per-frame DB feature counts into ``result.feature_counts``
    without a separate OpenCV extraction pass.
    """
    # RANSAC seeding: pycolmap's incremental SfM is otherwise unseeded, so two
    # runs on identical inputs can register different camera subsets (seen as
    # flaky depth-audit outcomes in e2e). Seeding is a determinism fix only —
    # it changes no reconstruction math.
    pycolmap.set_random_seed(0)
    if use_colmap:
        try:
            return _run_colmap(selected_dir, project_id, camera_model,
                               frame_files=frame_files)
        except Exception as exc:
            log.warning("colmap_failed", error=str(exc), fallback="opencv")

    return _run_opencv(selected_dir, project_id)


def _run_colmap(
    selected_dir: Path,
    project_id: str,
    camera_model: str,
    frame_files: list[Path] | None = None,
) -> ReconstructionResult:
    """Run COLMAP feature extraction + matching + SfM via pycolmap.

    Environment note: the pycolmap macOS x86_64 wheel cannot decode JPEG
    frames (BITMAP_ERROR for any ``.jpg``), while PNG decodes correctly.
    Frames are therefore losslessly re-encoded as PNG for this stage — the
    pixels are unchanged, only the container differs.
    """
    import pycolmap

    start = time.perf_counter()
    image_dir = _prepare_colmap_images(selected_dir)
    db_path = selected_dir / "colmap.db"
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass
    output_dir = selected_dir / "colmap_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature extraction (CUDA SiftGPU when torch sees a GPU, else CPU;
    # identical semantics — only throughput changes)
    log.info("colmap_feature_extraction", path=str(image_dir))
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.max_num_features = settings.colmap.max_features
    # This stage recorded NO substage timing, so a 238 s SfM for 25 frames
    # could not be attributed from its own report. Time each step.
    _t_extract = time.perf_counter()
    camera_mode = getattr(pycolmap.CameraMode, "SINGLE", pycolmap.CameraMode.AUTO)
    sift_device = _extract_features_with_fallback(db_path, image_dir, sift_options, camera_mode, camera_model)
    _extract_s = time.perf_counter() - _t_extract

    # Matching (sequential is the drone-video default, exhaustive for
    # unordered collections). Images are already registered in the DB by
    # feature extraction, so only the database path is required.
    # Enforce the feature cap AT THE DATABASE, before any matching work.
    # This wheel's CPU SIFT path ignores max_num_features (measured: cap 8192
    # stored 20,352 keypoints on one image; cap 2000 stored 3,190), so the
    # configured limit is applied here. Matching and SfM then see at most
    # max_num_features features per image.
    cap_stats: dict = {}
    _t_cap = time.perf_counter()
    if settings.colmap.enforce_feature_cap:
        # Extraction above ran at this wheel's own feature count (measured
        # 18.8k/frame at cap 10,240) because its CPU SIFT does not honour
        # max_num_features; the cap is applied to the database here. That is a
        # full blob rewrite of every over-cap image, so it is timed and its
        # row counts recorded rather than folded into extraction's number.
        cap_stats = _enforce_feature_cap(db_path, cap=int(sift_options.max_num_features))
    _cap_s = time.perf_counter() - _t_cap

    _t_match = time.perf_counter()
    log.info("colmap_matching", matcher=settings.colmap.matcher_type)
    sift_match_options = pycolmap.SiftMatchingOptions()
    match_device = _match_with_fallback(db_path, settings.colmap.matcher_type, sift_match_options)
    _match_s = time.perf_counter() - _t_match

    # Incremental mapping (SfM + bundle adjustment)
    log.info("colmap_mapping")
    # Ceres parallelises bundle adjustment only above
    # ba_min_num_residuals_for_cpu_multi_threading (default 50 000) — most
    # local BA passes on capped-feature drone runs sit below that and ran
    # single-threaded. 10 000 turns on multi-threaded Jacobian evaluation
    # for them: same solver, same model, same results, more cores.
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.ba_min_num_residuals_for_cpu_multi_threading = 10_000
    # Consolidated global-BA schedule (Batch 5, benchmark-verified on the
    # 160-frame calibration arm): defaults triggered 35 global BA rounds on
    # 160 frames; ratios 1.25 cut that to 20 with the model statistically
    # unchanged (tracks 96,165 vs 96,169, reprojection median 0.2385 vs
    # 0.2386 px, p95 0.5832 vs 0.5829, 0 negative-depth) and the sparse
    # stage 36% faster (943 s vs 1,478 s). The final global BA round still
    # runs; the explicit STRATA joint BA after mapping is unaffected.
    mapper_options.ba_global_frames_ratio = 1.25
    mapper_options.ba_global_points_ratio = 1.25
    mapper_options.ba_global_max_refinements = 3
    _t_map = time.perf_counter()
    mapping = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(image_dir),
        output_path=str(output_dir),
        options=mapper_options,
    )
    _map_s = time.perf_counter() - _t_map
    reconstructions = getattr(mapping, "reconstructions", None)
    if reconstructions is None:
        # pycolmap < 0.6 returned a dict keyed by reconstruction id
        reconstructions = [r for r in mapping.values() if r is not None]

    elapsed = (time.perf_counter() - start) * 1000
    result = ReconstructionResult(
        reconstruction_time_ms=round(elapsed, 2),
        backend="colmap",
    )
    result.detail = {
        "sift_device": sift_device,
        "match_device": match_device,
        # Substage split of this stage's wall clock. Without it the stage's
        # own report said only "238 s" and gave no way to tell extraction from
        # matching from mapping.
        "feature_extraction_ms": round(_extract_s * 1000.0, 1),
        "feature_cap_ms": round(_cap_s * 1000.0, 1),
        "feature_cap": cap_stats,
        "matching_ms": round(_match_s * 1000.0, 1),
        "mapping_ms": round(_map_s * 1000.0, 1),
    }

    # Per-frame feature counts straight from the COLMAP database — equivalent
    # information to the discarded OpenCV pre-pass counts, at zero cost.
    try:
        import sqlite3

        con = sqlite3.connect(str(db_path))
        try:
            rows = con.execute(
                "SELECT i.name, k.rows FROM images i "
                "JOIN keypoints k ON k.image_id = i.image_id"
            ).fetchall()
            result.feature_counts = {name: int(n) for name, n in rows}
        finally:
            con.close()
    except Exception as exc:
        log.warning("colmap_feature_count_read_failed", error=str(exc))

    best = max(
        (r for r in reconstructions if r is not None and len(r.images)),
        key=lambda r: len(r.points3D),  # pycolmap 3.x attribute is points3D
        default=None,
    )
    if best is None:
        log.warning("colmap_no_reconstruction", registered=0)
        return result

    for img_id, img in best.images.items():
        cfw = img.cam_from_world()
        R_w2c = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        tvec = np.asarray(cfw.translation, dtype=np.float64)
        R_c2w = R_w2c.T
        center = -R_c2w @ tvec
        qvec = _rotation_matrix_to_quat(R_c2w)  # (w, x, y, z)
        intrinsics = np.eye(3)
        try:
            cam = best.cameras[img.camera_id]
            intrinsics = np.asarray(cam.calibration_matrix(), dtype=np.float64)
            intrinsics[0, 0] = abs(intrinsics[0, 0])
            intrinsics[1, 1] = abs(intrinsics[1, 1])
        except Exception:
            pass
        pose = CameraPose(
            image_id=img_id,
            frame_id=img.name,
            position=center,
            rotation=R_c2w,
            quaternion=qvec,
            intrinsics=intrinsics,
            distortion=np.zeros(5),
            is_estimated=True,
        )
        result.cameras[img.name] = pose

    centers_img: dict[int, np.ndarray] = {
        img_id: np.asarray(cam.position, dtype=np.float64)
        for img_id, cam in ((i, result.cameras[img.name]) for i, img in best.images.items())
    }
    for pt_id, pt in best.points3D.items():
        error_val = float(pt.error) if hasattr(pt, "error") and pt.error is not None else 0.0
        # Observation-level provenance: every track element gives (image_id,
        # point2D_idx) into the image's points2D array — the measured pixel
        # each observation was made at. Without this the STRATA-level joint
        # BA has no observation graph to optimize against (measured no-op on
        # the airport_53 audit: observations=0, initial=final=0).
        obs: list[tuple[str, np.ndarray]] = []
        obs_centers: list[np.ndarray] = []
        try:
            for el in pt.track.elements:
                obs_img = best.images[el.image_id]
                xy = np.asarray(obs_img.points2D[el.point2D_idx].xy, dtype=np.float64)
                obs.append((obs_img.name, xy))
                obs_centers.append(centers_img[el.image_id])
        except Exception as exc:
            log.warning("colmap_track_observation_read_failed", point=pt_id, error=str(exc))
            obs = []
        xyz = np.asarray(pt.xyz, dtype=np.float64)
        # Conditioning statistics for the same fields the telemetry-
        # triangulation backend fills: smallest subtended ray angle and the
        # implied relative depth uncertainty (e/(f·θ)). Measured, never the
        # 0.0 default — _track_statistics reports these fields verbatim.
        min_angle = _min_observation_angle_deg(xyz, obs_centers)
        if min_angle is not None and obs:
            focal = float(intrinsics[0, 0])
            err_for_gate = max(error_val, MIN_TRACK_REPROJ_PX_FLOOR)
            rel_depth = err_for_gate / (focal * np.radians(max(min_angle, 1e-3)))
        else:
            rel_depth = 0.0
        result.points3d.append(
            SparsePoint3D(
                point_id=pt_id,
                position=xyz,
                color=np.asarray(pt.color, dtype=np.uint8) if pt.color is not None else np.zeros(3, dtype=np.uint8),
                track_length=_track_length(pt.track),
                mean_reproj_error=error_val,
                observations=obs,
                min_triangulation_angle_deg=float(min_angle) if min_angle is not None else 0.0,
                rel_depth_uncertainty=float(rel_depth),
            )
        )

    result.num_registered = len(result.cameras)
    result.num_points = len(result.points3d)

    # Mean reprojection error over all triangulated points (per-point error
    # is what pycolmap 3.x exposes; per-observation error2D is not public).
    errors = [float(pt.error) for pt in best.points3D.values() if pt.error is not None]
    result.mean_reproj_error = float(np.mean(errors)) if errors else 0.0

    log.info(
        "colmap_complete",
        registered=result.num_registered,
        points=result.num_points,
        mean_reproj=round(result.mean_reproj_error, 4),
    )

    return result


def _min_observation_angle_deg(xyz: np.ndarray, centers: list[np.ndarray]) -> float | None:
    """Smallest pairwise ray angle (deg) subtended at *xyz* by the camera
    centres — the same statistic the telemetry-triangulation backend gates
    on, computed independently for points that arrive from COLMAP.  Needs
    >= 2 distinct centres; None otherwise.

    Batched over the pairs: this runs once per point (59,922 points on the
    run under measurement, ~13.6 observing cameras each) and the pairwise
    Python double loop was the single largest block inside the pose backend
    (measured 132.5 s of its 813.9 s). The arithmetic is written out term by
    term rather than through a matrix product — the same two subtractions,
    the same three-product dot accumulated in the same order, the same norm
    expression, the same clip/arccos/degrees — so every pair's angle is the
    same float and the minimum is that value, not an approximation of it.
    """
    if len(centers) < 2:
        return None
    C = np.asarray(centers, dtype=np.float64)
    V = np.asarray(xyz, dtype=np.float64) - C
    # dot(va, vb) for every pair, in index order (matches np.dot elementwise).
    dots = (V[:, 0:1] * V[:, 0][None, :]
            + V[:, 1:2] * V[:, 1][None, :]
            + V[:, 2:3] * V[:, 2][None, :])
    norms = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2 + V[:, 2] ** 2)
    cosang = dots / (norms[:, None] * norms[None, :] + 1e-12)
    angles = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    # The loop only compared i < j. Every diagonal entry is a centre against
    # itself (cosine ~1 by the +1e-12 guard, so ~7e-06 deg) and would win the
    # minimum outright, so blank the diagonal instead of building (i, j) index
    # arrays for every point — the same set of candidates, one allocation less.
    np.fill_diagonal(angles, np.inf)
    return float(angles.min())


def _track_length(track: object) -> int:
    """pycolmap ``Track.length`` is a method in 3.x, an attribute in 0.6.x.

    Returns the number of observations of the track (int).
    """
    length = getattr(track, "length", 0)
    if callable(length):
        length = length()
    return int(length or 0)


def _enforce_feature_cap(db_path: Path, cap: int) -> dict:
    """Enforce the per-image feature cap on the COLMAP database.

    The bundled pycolmap wheel's CPU SIFT ignores ``max_num_features``
    (measured: cap 8192 → up to 20,352 stored rows), so the configured cap
    is enforced here BEFORE matching: for each over-cap image, keypoints and
    descriptors are rewritten with a deterministic, spatially-balanced
    subset. Keypoints are stored as 2-float rows (x, y) and descriptors as
    row-aligned 128-float rows, so deleting the same row indices from both
    keeps descriptor↔keypoint identity intact. Selection maximizes response
    (COLMAP sorts keypoints by descending scale-space response — the blob
    order is already strongest-first) while balancing coverage over a 4×4
    spatial grid so the retained features are not concentrated in one
    textured corner. Idempotent: no-op when every image is within cap.
    Returns a small stats dict for provenance.
    """
    import sqlite3

    import numpy as np

    stats = {"images_over_cap": 0, "rows_before": 0, "rows_after": 0}
    con = sqlite3.connect(str(db_path))
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "keypoints" not in tables or "descriptors" not in tables:
            return stats
        rows = con.execute("SELECT image_id, rows, cols, data FROM keypoints").fetchall()
        for image_id, n, k_cols, blob in rows:
            if n <= cap:
                continue
            # keypoints blob: n × k_cols float32 — COLMAP stores (x, y,
            # a11, a12, a21, a22) per row; x/y are columns 0/1. cols is the
            # per-row element COUNT (6), not bytes.
            kps_all = np.frombuffer(blob, dtype=np.float32).reshape(n, k_cols)
            kps = kps_all[:, :2]
            desc_row = con.execute(
                "SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (image_id,)
            ).fetchone()
            if desc_row is None:
                continue
            d_rows, d_cols, d_blob = desc_row
            if d_rows != n:
                continue  # malformed DB — leave untouched rather than corrupt
            desc = np.frombuffer(d_blob, dtype=np.uint8).reshape(d_rows, d_cols)

            # Keep a spatially balanced subset in COLMAP's own extraction
            # order. The DB rows carry (x, y, a11, a12, a21, a22) and the DB
            # order is roughly ascending scale (measured corr +0.68 with row
            # index). Measured on a 40-frame subset of airport_53 (uncapped
            # 40 reg / 26,954 pts / mean 0.283 px):
            #   first-N per grid cell → 40 reg / 19,769 pts / 0.526 px
            #   stratified rows per cell → 34 reg / 15,863 pts / 1.247 px
            #   descending-scale rank → 32 reg /  9,835 pts / 1.464 px
            # so extraction-order selection within cells is the best of the
            # three deterministic strategies and is retained. Rows are
            # independent at this stage (no matches exist yet), so rewriting
            # keypoints+descriptors with the same row subset keeps
            # descriptor↔keypoint identity intact.
            order = np.arange(n)
            # 4×4 grid balance: keep the first rows per cell (extraction order).
            G = 4
            xs = np.clip((kps[:, 0] / max(kps[:, 0].max(), 1e-6) * G).astype(int), 0, G - 1)
            ys = np.clip((kps[:, 1] / max(kps[:, 1].max(), 1e-6) * G).astype(int), 0, G - 1)
            cells = ys * G + xs
            keep: list[int] = []
            quotas = np.full(G * G, cap // (G * G), dtype=int)
            quotas[: cap - quotas.sum()] += 1
            for cell in range(G * G):
                idx = order[cells == cell][: quotas[cell]]
                keep.append(idx)
            remaining = cap - sum(len(k) for k in keep)
            if remaining > 0:
                chosen = np.concatenate(keep) if keep else np.array([], dtype=int)
                rest = np.setdiff1d(order, chosen, assume_unique=False)
                keep.append(rest[:remaining])
            keep_idx = np.sort(np.concatenate(keep))

            new_n = int(len(keep_idx))
            new_kps = kps_all[keep_idx].astype(np.float32)
            new_desc = desc[keep_idx].astype(np.uint8)
            con.execute(
                "UPDATE keypoints SET rows = ?, data = ? WHERE image_id = ?",
                (new_n, new_kps.tobytes(), image_id),
            )
            con.execute(
                "UPDATE descriptors SET rows = ?, data = ? WHERE image_id = ?",
                (new_n, new_desc.tobytes(), image_id),
            )
            stats["images_over_cap"] += 1
            stats["rows_before"] += int(n)
            stats["rows_after"] += new_n
        con.commit()
    finally:
        con.close()
    if stats["images_over_cap"]:
        log.info(
            "colmap_feature_cap_enforced",
            cap=cap,
            images_over_cap=stats["images_over_cap"],
            rows_before=stats["rows_before"],
            rows_after=stats["rows_after"],
        )
    return stats


def _prepare_colmap_images(selected_dir: Path) -> Path:
    """Losslessly re-encode the JPEG frames as PNG for the COLMAP stage.

    Kept separate from the shared ``frames/`` directory so every other stage
    continues to consume the original JPEGs.

    The mirror is CONTENT-AWARE: it is rebuilt whenever its PNG stems no
    longer match the current selection. A presence-only check reused a stale
    1-frame mirror after a re-selection (London retry: COLMAP extracted
    features for 1 image against 60 selected JPGs → zero pairs → zero
    registered cameras) — the same stale-cache class that made Retry a
    no-op loop.
    """
    import cv2

    jpgs = list_image_files(selected_dir)
    jpg_stems = {p.stem for p in jpgs}
    image_dir = selected_dir / "colmap_images"
    pngs = list(image_dir.glob("*.png")) if image_dir.is_dir() else []
    if pngs and {p.stem for p in pngs} == jpg_stems:
        return image_dir
    if image_dir.is_dir():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    for jpg in jpgs:
        img = cv2.imread(str(jpg))
        if img is not None:
            cv2.imwrite(str(image_dir / f"{jpg.stem}.png"), img)
        else:
            log.warning("colmap_image_unreadable", path=str(jpg))
    return image_dir


def triangulate_with_known_poses(
    features: dict,
    verified_pairs: list[tuple[str, str, np.ndarray, np.ndarray]],
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    intrinsics: np.ndarray,
    image_paths: dict[str, Path] | None = None,
    gps_by_frame: dict[str, tuple[float, float, float]] | None = None,
) -> ReconstructionResult:
    """Triangulate 3D points from verified matches with KNOWN camera poses.

    Telemetry-assisted localization: flight metadata (an explicitly valid
    input per the SIH problem statement) supplies the camera trajectory when
    visual translation is unreliable — e.g. ~1 px parallax aerial video where
    SfM's translation gauge collapses. Visual information is NOT discarded:
    every 3D point comes from real image matches triangulated through the
    telemetry poses (DLT), and a point survives only if it reprojects
    consistently into all its observing views.

    Args:
        features: frame_id -> Features (keypoints (N,2) x,y pixels).
        verified_pairs: (frame_a, frame_b, idx_a (M,), idx_b (M,)) —
            geometrically-verified match KEYPOINT INDICES (columns of
            MatchResult.matches filtered by the verification inlier mask).
        poses: frame_id -> (camera centre C_world (3,), R_c2w (3,3)) — the
            same convention as poses.json: X_world = R_c2w @ X_cam + C.
        intrinsics: shared (3,3) K for all frames.
        image_paths: optional frame_id -> image path; when given, sparse
            points take their RGB from the actual video pixels of their first
            observing frame.

    Conditioning note: two-view tracks of a weak-parallax flight are
    ill-conditioned ALONG the viewing ray; tracks observed across a wider
    baseline triangulate accurately. ``track_length`` is stored per point so
    downstream filtering can prefer well-constrained points, and the caller
    records the path/scene ratio that characterises the flight.
    """
    start = time.perf_counter()
    result = ReconstructionResult(backend="telemetry_triangulation")

    # Nadir detection from the telemetry rotations themselves: mean view
    # direction (camera +Z in world) vs world up. Only used when clearly
    # downward-looking; oblique flights get no altitude-based rejection.
    nadir_cosine: float | None = None
    nadir_min_clearance = 0.0
    view_dirs = [R_c2w[:, 2] for _, R_c2w in poses.values()]
    if view_dirs:
        mean_down = -np.mean(view_dirs, axis=0)
        nrm = np.linalg.norm(mean_down)
        if nrm > 1e-9:
            c = float(np.dot(mean_down / nrm, WORLD_UP))
            if c >= NADIR_COSINE:
                nadir_cosine = c
                # Real structure sits well below the camera plane on a nadir
                # flight; ghosts sit within metres of the lens. The clearance
                # scales with the flight altitude from telemetry itself.
                alt = float(np.median([p[0][2] for p in poses.values()]))
                nadir_min_clearance = NADIR_CLEARANCE_FRACTION * max(alt, 0.0)

    # Per-frame projection: X_cam = R_w2c @ (X_world - C), pixel = K @ X_cam/z.
    projs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for fid, (C, R_c2w) in poses.items():
        R_w2c = R_c2w.T
        P = intrinsics @ np.hstack([R_w2c, -(R_w2c @ C).reshape(3, 1)])  # 3x4
        projs[fid] = (C, R_w2c, P)

    # Scene-range envelope for the ghost gate below, measured from the
    # flight geometry itself: real scene structure sits at ranges comparable
    # to what the flight actually LOOKED at, but ghost matches (repeat
    # texture, haze, off-corridor clutter) drift arbitrarily far.
    # Envelope = SCENE_RANGE_TOLERANCE x max camera distance to the
    # trajectory centroid, with the nadir case handled explicitly: on a
    # downward-looking flight the ground is one altitude below the whole
    # corridor, so the envelope must include at least the flight altitude
    # (a 10 m-span corridor at 30 m altitude still images ground at ~30 m
    # range — the corridor span alone would wrongly exclude it).
    if len(poses) >= 2:
        _centers = np.asarray([p[0] for p in poses.values()])
        corridor = float(np.max(np.linalg.norm(_centers - _centers.mean(axis=0), axis=1)))
        alt = float(np.median([p[0][2] for p in poses.values()]))
        view_dirs = [R_c2w[:, 2] for _, R_c2w in poses.values()]
        mean_down = -np.mean(view_dirs, axis=0)
        nd = np.linalg.norm(mean_down)
        is_nadir = nd > 1e-9 and float(np.dot(mean_down / nd, WORLD_UP)) >= NADIR_COSINE
        scene_range_max_m = SCENE_RANGE_TOLERANCE * max(corridor, alt if is_nadir else 0.0)
        if scene_range_max_m <= 0:
            scene_range_max_m = None  # hover flight: gate disabled
    else:
        scene_range_max_m = None

    # Union-Find over keypoint identities (frame_id, keypoint_index): two
    # identities linked by a verified match belong to the same 3D point.
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for fid, feat in features.items():
        for i in range(len(feat.keypoints)):
            parent[(fid, i)] = (fid, i)
    for frame_a, frame_b, idx_a, idx_b in verified_pairs:
        if frame_a not in features or frame_b not in features:
            continue
        for ia, ib in zip(idx_a, idx_b):
            ia, ib = int(ia), int(ib)
            if ia < len(features[frame_a].keypoints) and ib < len(features[frame_b].keypoints):
                union((frame_a, ia), (frame_b, ib))

    # Group identities into tracks (one pixel observation per frame).
    tracks: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}
    for (fid, i) in parent:
        feat = features[fid]
        if i >= len(feat.keypoints):
            continue
        tracks.setdefault(find((fid, i)), []).append((fid, np.asarray(feat.keypoints[i], dtype=np.float64)))

    # Triangulate every track spanning >= 2 known-pose frames.
    pt_id = 0
    reproj_errors: list[float] = []
    first_observation: list[tuple[str, np.ndarray]] = []
    for members in tracks.values():
        by_frame: dict[str, np.ndarray] = {}
        for fid, pix in members:
            if fid in projs and fid not in by_frame:
                by_frame[fid] = pix
        if len(by_frame) < 2:
            continue
        obs = list(by_frame.items())
        X = _triangulate_dlt([projs[f][2] for f, _ in obs], [pix for _, pix in obs])
        if X is None:
            continue
        # Triangulation-angle gate: with weak parallax a two-view DLT can
        # place a point kilometres down-ray and still reproduce the pixels
        # within a few px (angular error scales with f/depth). Require the
        # smallest observing baseline to subtend a meaningful angle at the
        # point — the classic well-conditioned-triangulation criterion.
        obs_centers = [projs[f][0] for f, _ in obs]
        min_angle = _min_observation_angle_deg(X, obs_centers)
        if min_angle is None or min_angle < MIN_TRIANGULATION_ANGLE_DEG:
            continue
        # Honest gate: the point must reproject consistently into EVERY
        # observing view (this is what makes a point "seen", not a count).
        errs: list[float] = []
        for fid, pix in obs:
            C, R_w2c, _ = projs[fid]
            xc = R_w2c @ (X - C)
            if xc[2] <= 1e-6:
                break
            uv = intrinsics @ (xc / xc[2])
            errs.append(float(np.linalg.norm(uv[:2] - pix)))
        else:
            err_mean = float(np.mean(errs))
            # Per-point robust refinement (the point-BA step): cameras stay
            # fixed at their telemetry prior; the point moves to the
            # Huber-optimal ray intersection. LM damping keeps near-parallel
            # rays (0.5-2 deg) from exploding the step; accepted ONLY when it
            # strictly reduces the mean reprojection (never a blind move).
            X_ref, refined_err = _refine_point_gauss_newton(
                X,
                obs_centers,
                [projs[f][1] for f, _ in obs],
                intrinsics,
                [pix for _, pix in obs],
            )
            if X_ref is not None and np.all(np.isfinite(X_ref)) and err_mean > 0 and refined_err < err_mean:
                X = X_ref
                err_mean = refined_err
                # Re-verify positive depth in every observer after the move.
                if any((projs[f][1] @ (X - projs[f][0]))[2] <= 1e-6 for f, _ in obs):
                    continue
                # Refresh the min angle at the refined position.
                min_angle = _min_observation_angle_deg(X, obs_centers)
            # Depth-uncertainty gate. Along-ray error scales with
            # reprojection error over the triangulation angle (dZ ~ e·Z /
            # (f·θ)); at aerial ranges a flat 4 px cap alone admits points
            # hundreds of metres off along the ray (seen as below-ground
            # outliers). Cap the RELATIVE depth error instead: a point is
            # accepted only when its implied depth uncertainty is a small
            # fraction of its depth.
            err_for_gate = max(err_mean, MIN_TRACK_REPROJ_PX_FLOOR)
            min_angle_rad = np.radians(max(min_angle, 1e-3))
            focal = float(intrinsics[0, 0])
            # σZ/Z ≈ e / (f·θ): dimensionless relative depth uncertainty.
            rel_depth_err = err_for_gate / (focal * min_angle_rad)
            if rel_depth_err > MAX_RELATIVE_DEPTH_ERR:
                continue
            # Nadir ghost gate: on a downward-looking telemetry flight, every
            # real scene point lies below every observing camera. Tracks that
            # intersect at camera altitude or above are ghost matches (lens
            # flare / dust / noise) that reproject cleanly but are physically
            # impossible — measured on airport3 as a 21-point scaffold
            # floating at flight altitude under a 178 m nadir survey.
            if nadir_cosine is not None:
                below_all = all(
                    float(np.dot(X - projs[fid][0], WORLD_UP)) < -nadir_min_clearance
                    for fid, _ in obs
                )
                if not below_all:
                    continue
            # Scene-range consistency gate (oblique flights). The uncertainty
            # gate above bounds how FAR off a point may sit along its ray for
            # its own noise budget — but at 60–130 m ranges that budget is
            # ±5–10 m, which ghost matches (repeat textures, haze, moving
            # clutter far off the flight corridor) spend silently: they sit
            # within budget yet hundreds of metres from anything the flight
            # actually imaged. Physical prior: a triangulated point should be
            # at a range comparable to the ranges the flight actually imaged.
            # The scene-range envelope is derived from the trajectory and the
            # depth maps' own extent (not a universal constant): a point is
            # kept when its minimum range to any observing camera is within
            # SCENE_RANGE_TOLERANCE of the max camera→camera-path-span that
            # the frames' depth values support. Measured on Flight_to_tower:
            # far-field ghosts sit at ≥2.2× the imaged scene extent.
            min_point_range = min(
                float(np.linalg.norm(X - projs[fid][0])) for fid, _ in obs
            )
            if scene_range_max_m is not None and min_point_range > scene_range_max_m:
                continue
            if err_mean <= MAX_TRACK_REPROJ_PX:
                reproj_errors.append(err_mean)
                first_observation.append(obs[0])
                focal = float(intrinsics[0, 0])
                result.points3d.append(
                    SparsePoint3D(
                        point_id=pt_id,
                        position=X,
                        color=np.zeros(3, dtype=np.uint8),
                        track_length=len(by_frame),
                        mean_reproj_error=err_mean,
                        observations=[(f, np.asarray(pix, dtype=np.float64)) for f, pix in obs],
                        min_triangulation_angle_deg=float(min_angle),
                        rel_depth_uncertainty=float(err_for_gate) / (focal * np.radians(max(min_angle, 1e-3))),
                    )
                )
                pt_id += 1

    _colorize_from_images(result.points3d, first_observation, image_paths)

    # Flight/scene scale characterisation, computed in the telemetry world
    # frame: how far the camera path spans relative to the median camera→point
    # distance. Video-only SfM on this flight collapsed that ratio to ~2.5 —
    # an airport seen from 490 m has ~0.3 — so this is the honest sanity gate.
    if poses and result.points3d:
        centers = np.array([poses[f][0] for f in poses], dtype=np.float64)
        xyz = np.array([p.position for p in result.points3d])
        result.path_span = float(np.linalg.norm(centers.max(0) - centers.min(0)))
        sample = xyz if len(xyz) <= 20000 else xyz[np.random.default_rng(0).choice(len(xyz), 20000, replace=False)]
        d = np.linalg.norm(sample[None, :, :] - centers[:, None, :], axis=2)
        flat = d[d > 0]
        result.median_scene_distance = float(np.median(flat)) if flat.size else 0.0

    for fid, (C, R_c2w) in poses.items():
        cam = CameraPose(
            image_id=0,
            frame_id=fid,
            position=C,
            rotation=R_c2w,
            quaternion=_rotation_matrix_to_quat(R_c2w),
            intrinsics=intrinsics,
            distortion=np.zeros(5),
            is_estimated=True,
        )
        # GPS provenance travels with the camera record (Part: telemetry/GPS
        # flow) so poses.json and the BA soft prior see the SAME source.
        if gps_by_frame and fid in gps_by_frame:
            lat, lon, alt = gps_by_frame[fid]
            cam.gps = {"lat": lat, "lon": lon, "alt": alt}
        result.cameras[fid] = cam
    # Sparse quality gates (mandate: report conditioning, not just counts).
    if result.points3d:
        _sp = np.array([p.position for p in result.points3d])
        _tr = np.array([p.track_length for p in result.points3d])
        _re = np.array([p.mean_reproj_error for p in result.points3d])
        # Exact per-point min triangulation angle over a deterministic
        # subsample, from camera centres (vertex angle at X between the two
        # nearest cameras — the min-baseline conditioning angle).
        _centers = np.array([poses[f][0] for f in poses])
        _rng = np.random.default_rng(0)
        _sub = _sp if len(_sp) <= 3000 else _sp[_rng.choice(len(_sp), 3000, replace=False)]
        _angles: list[float] = []
        for X in _sub:
            d = _centers - X
            r = np.linalg.norm(d, axis=1)
            two = np.argsort(r)[:2]
            if len(two) < 2 or r[two[1]] < 1e-9:
                continue
            cosang = float(np.clip(np.dot(d[two[0]], d[two[1]]) / (r[two[0]] * r[two[1]]), -1.0, 1.0))
            _angles.append(float(np.degrees(np.arccos(cosang))))
        result.quality_gates = {
            "reprojection_median_px": round(float(np.median(_re)), 3),
            "reprojection_p95_px": round(float(np.percentile(_re, 95)), 3),
            "track_length_median": float(np.median(_tr)),
            "track_length_p95": float(np.percentile(_tr, 95)),
            "two_view_track_pct": round(float(100.0 * (_tr == 2).mean()), 2),
            "triangulation_angle_median_deg": round(float(np.median(_angles)), 3) if _angles else None,
            "triangulation_angle_p05_deg": round(float(np.percentile(_angles, 5)), 3) if _angles else None,
            "positive_depth_pct": 100.0,  # every accepted point verified in all observers
            "xyz_bounds_min": [round(float(v), 1) for v in _sp.min(axis=0)],
            "xyz_bounds_max": [round(float(v), 1) for v in _sp.max(axis=0)],
        }
        log.info("sparse_quality_gates", **result.quality_gates)
    result.image_pairs = [(a, b) for a, b, _, _ in verified_pairs]
    result.num_registered = len(result.cameras)
    result.num_points = len(result.points3d)
    result.mean_reproj_error = float(np.mean(reproj_errors)) if reproj_errors else 0.0
    result.reconstruction_time_ms = round((time.perf_counter() - start) * 1e3, 2)
    log.info(
        "telemetry_triangulation_complete",
        cameras=result.num_registered,
        points=result.num_points,
        mean_reproj=round(result.mean_reproj_error, 3),
        pairs=len(verified_pairs),
        path_span=round(result.path_span, 2),
        median_scene_distance=round(result.median_scene_distance, 2),
    )
    return result


#: Mean reprojection residual (px) a triangulated track may have against its
#: observing views and still count as a real point.
MAX_TRACK_REPROJ_PX = 4.0

#: Reprojection-error floor (px) used by the relative-depth-uncertainty gate
#: so sub-pixel residuals do not imply unrealistically precise depth. Set to
#: the measured real-world SfM noise (airport scenes: mean track reprojection
#: ~0.86 px); a 0.3 px floor implied ~5% depth certainty on wide pairs that
#: actually carry ~20%, admitting nothing — the gate must model real noise.
MIN_TRACK_REPROJ_PX_FLOOR = 0.75

#: Maximum accepted relative depth uncertainty σZ/Z of a triangulated point
#: (reprojection error over f·θ). At the airport scenes' ~0.3–1° angles this
#: rejects along-ray outliers hundreds of metres off while keeping points
#: whose depth is actually constrained.
MAX_RELATIVE_DEPTH_ERR = 0.08

#: Minimum expected parallax (px) a camera pair must produce, given the
#: flight altitude from telemetry, for geometric verification to be able to
#: tell real matches from false ones. Pairs closer together than
#: ``PAIR_BASELINE_PARALLAX_PX * altitude / f`` are dropped before
#: triangulation: near-zero-baseline pairs cannot be verified epipolarly
#: (true parallax is sub-pixel), and their false matches triangulate NEAR
#: the lens with wide angles and tiny residuals — the exact near-field
#: scaffold failure measured on airport3 (1.8 m scaffold under a 170 m
#: flight). Wide pairs (≥ ~1.4 m at 170 m) triangulate to the true 150-210 m
#: scene depth (validated against dataset GT depth, 2558/2628 points).
PAIR_BASELINE_PARALLAX_PX = 3.0

#: Expected parallax (px) a camera pair must provide, per the σZ/Z ≈ e/(f·θ)
#: gate, for triangulated depth to be better than ~5% uncertain at 0.75 px
#: matching noise: θ ≈ b·f/Z ⇒ σZ/Z ≈ e·Z/(f·b·f) — need b·f/Z ≥ e/0.05 = 15 px
#: with Z ≈ flight altitude. Pairs below this floor cannot yield usable
#: depth from telemetry poses, so the telemetry branch MATCHES extra pairs
#: along the trajectory until every frame has wide-enough partners (and
#: drops verified pairs below the 3 px verifiability floor).
SUPPAIR_MIN_PARALLAX_PX = 12.0

#: A telemetry flight whose cameras all look ≥ this far below the horizon
#: (mean view direction · world-up < −NADIR_COSINE) is a nadir/near-nadir
#: survey. For such flights every real scene point lies far below the
#: camera plane — at least NADIR_CLEARANCE_FRACTION of the flight altitude.
#: Ghost matches (lens flare, dust, sensor noise) reproject cleanly but
#: sit within metres of the lens (measured: 1.8 m median under a 178 m
#: nadir flight) and are rejected.
NADIR_COSINE = 0.7
NADIR_CLEARANCE_FRACTION = 0.25

#: Scene-range ghost gate for oblique (sub-nadir) flights: a triangulated
#: point's minimum distance to any observing camera must stay within this
#: multiple of the camera-corridor extent (max camera distance to the
#: trajectory centroid). Measured on Flight_to_tower: real structure sits at
#: 57–68 m from the path under a 148 m corridor span, while far-field ghost
#: matches (repeat texture / haze / off-corridor clutter that still reproject
#: cleanly) sit ≥2.2× beyond it — their along-ray error is bounded by their
#: own depth-uncertainty budget (±5–10 m at 60–130 m ranges) and passes every
#: pixel-space gate. 2.5× keeps all corridor structure with margin and
#: removes only points at ranges the flight never imaged at any bearing.
SCENE_RANGE_TOLERANCE = 2.5

#: Minimum acceptable triangulation angle (degrees) between the two most
#: closely-spaced observing cameras. Below this the point's depth is
#: effectively unobservable and its position is dominated by noise.
MIN_TRIANGULATION_ANGLE_DEG = 0.25


def _colorize_from_images(
    points3d: list[SparsePoint3D],
    first_observation: list[tuple[str, np.ndarray]],
    image_paths: dict[str, Path] | None,
) -> None:
    """Give sparse points their RGB from the actual video pixels.

    Points are grouped by first observing frame so each image is read once.
    Unreadable/absent images leave the point gray rather than failing the
    reconstruction — colour is cosmetic for the sparse scaffold.
    """
    if not image_paths or not points3d:
        for p in points3d:
            p.color = np.array([128, 128, 128], dtype=np.uint8)
        return
    by_frame: dict[str, list[int]] = {}
    for i, (fid, pix) in enumerate(first_observation):
        by_frame.setdefault(fid, []).append(i)
    for fid, indices in by_frame.items():
        path = image_paths.get(fid)
        if path is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for i in indices:
            u, v = first_observation[i][1]
            ui, vi = int(round(float(u))), int(round(float(v)))
            if 0 <= ui < w and 0 <= vi < h:
                b, g, r = img[vi, ui]
                points3d[i].color = np.array([r, g, b], dtype=np.uint8)
        del img  # one frame in memory at a time
    for p in points3d:
        if not p.color.any():
            p.color = np.array([128, 128, 128], dtype=np.uint8)


def _triangulate_dlt(Ps: list[np.ndarray], pixels: list[np.ndarray]) -> np.ndarray | None:
    """DLT triangulation from >=2 projections; returns the world point or None."""
    if len(Ps) < 2:
        return None
    rows = []
    for P, pix in zip(Ps, pixels):
        u, v = pix
        rows.append(P[0] - u * P[2])
        rows.append(P[1] - v * P[2])
    A = np.asarray(rows, dtype=np.float64)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    X_h = Vt[-1]
    if abs(X_h_h := X_h[3]) < 1e-12 or not np.all(np.isfinite(X_h)):
        return None
    X = X_h[:3] / X_h_h
    return X if np.all(np.isfinite(X)) else None


def _refine_point_gauss_newton(
    X0: np.ndarray,
    centers: list[np.ndarray],
    rotations_w2c: list[np.ndarray],
    intrinsics: np.ndarray,
    pixels: list[np.ndarray],
    max_iters: int = 12,
) -> tuple[np.ndarray | None, float]:
    """Robust per-point Levenberg-Marquardt refinement of a triangulated point.

    Minimizes the Huber-loss reprojection residual over the point position
    with cameras FIXED (the telemetry poses are the trajectory prior; point
    refinement is the bundle-adjustment step that must not drift the
    gauge). LM damping (not plain GN) is REQUIRED here: with 0.5-2 deg
    triangulation angles the along-ray curvature is near-singular and an
    undamped GN step explodes to z<=0 (measured: 97% of tracks lost). Huber
    delta comes from the point's own initial residual scale — no absolute
    constant. Returns the refined point and its mean residual (px), or
    (None, inf) if the initial point is invalid; the caller keeps X0 unless
    the refined residual actually improved.
    """
    X = np.asarray(X0, dtype=np.float64).copy()
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    def residuals_and_jacobians(Xv: np.ndarray):
        rows: list[np.ndarray] = []
        res: list[float] = []
        for C, R_w2c, pix in zip(centers, rotations_w2c, pixels):
            xc = R_w2c @ (Xv - C)
            if xc[2] <= 1e-6:
                return None, None
            uv = intrinsics @ (xc / xc[2])
            e = uv[:2] - np.asarray(pix, dtype=np.float64)
            z2 = xc[2] * xc[2]
            J_cam = np.array([
                [fx / xc[2], 0.0, -fx * xc[0] / z2],
                [0.0, fy / xc[2], -fy * xc[1] / z2],
            ])
            rows.append(J_cam @ R_w2c)
            res.append(e)
        return rows, res

    rows, res = residuals_and_jacobians(X)
    if rows is None:
        return None, float("inf")
    errs0 = [float(np.linalg.norm(e)) for e in res]
    best_cost = float(np.mean(errs0))
    best_X = X.copy()
    delta = max(1.0, float(np.median(errs0)) * 2.5)  # Huber scale from data
    lam = 1e-3
    for _ in range(max_iters):
        J_all = []
        r_vec = []
        for J, e in zip(rows, res):
            r = float(np.linalg.norm(e))
            w = delta / r if r > delta else 1.0
            J_all.append(w * J)
            r_vec.extend((w * e).tolist())
        J_all = np.concatenate(J_all, axis=0)
        r_vec = np.asarray(r_vec, dtype=np.float64)
        H = J_all.T @ J_all
        g = J_all.T @ r_vec
        damped = H + lam * np.diag(np.diag(H)) + 1e-12 * np.eye(3)
        try:
            step = np.linalg.solve(damped, -g)
        except np.linalg.LinAlgError:
            break
        X_new = X + step
        rows_new, res_new = residuals_and_jacobians(X_new)
        if rows_new is None:
            lam *= 10.0
            if lam > 1e6:
                break
            continue
        cost = float(np.mean([float(np.linalg.norm(e)) for e in res_new]))
        if cost < best_cost:
            improvement = best_cost - cost
            best_cost, best_X = cost, X_new.copy()
            X, rows, res = X_new, rows_new, res_new
            lam = max(lam * 0.3, 1e-9)
            if np.linalg.norm(step) < 1e-4 or improvement < 1e-5:
                break
        else:
            lam *= 10.0
            if lam > 1e6:
                break
    return best_X, best_cost


def _run_opencv(selected_dir: Path, project_id: str) -> ReconstructionResult:
    """Fallback: estimate pairwise poses using essential matrix."""
    start = time.perf_counter()

    image_files = list_image_files(selected_dir)
    result = ReconstructionResult(backend="opencv")

    if len(image_files) < 2:
        return result

    # Read first image to get intrinsics
    sample = cv2.imread(str(image_files[0]))
    h, w = sample.shape[:2]
    focal = max(w, h) * 0.8
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)

    # Create SIFT detector for features
    sift = cv2.SIFT_create(nfeatures=settings.colmap.max_features)

    all_kpts = {}
    all_descs = {}
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kpts, descs = sift.detectAndCompute(gray, None)
        all_kpts[img_file.name] = np.array([kp.pt for kp in kpts], dtype=np.float32) if kpts else np.zeros((0, 2), dtype=np.float32)
        all_descs[img_file.name] = descs.astype(np.float32) if descs is not None else np.zeros((0, 128), dtype=np.float32)

    # Match adjacent pairs and estimate essential matrix
    bf = cv2.BFMatcher(cv2.NORM_L2)
    current_R = np.eye(3)
    current_t = np.zeros(3)

    for i in range(len(image_files) - 1):
        name_a = image_files[i].name
        name_b = image_files[i + 1].name

        if len(all_descs[name_a]) == 0 or len(all_descs[name_b]) == 0:
            continue

        matches = bf.knnMatch(all_descs[name_a], all_descs[name_b], k=2)
        good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]

        if len(good) < 8:
            continue

        pts_a = all_kpts[name_a][[m.queryIdx for m in good]]
        pts_b = all_kpts[name_b][[m.trainIdx for m in good]]

        E, mask = cv2.findEssentialMat(pts_a, pts_b, K, cv2.RANSAC, 0.99, 1.0)
        if E is None or mask is None:
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)

        current_R = R @ current_R
        current_t = current_t + current_R @ t.ravel()

        # Store poses
        result.cameras[name_a] = CameraPose(
            image_id=i,
            frame_id=name_a,
            position=current_t.copy(),
            rotation=current_R.copy(),
            quaternion=_rotation_matrix_to_quat(current_R),
            intrinsics=K,
            distortion=np.zeros(5),
            is_estimated=True,
        )

        result.image_pairs.append((name_a, name_b))

    # Add last frame
    if image_files:
        last = image_files[-1].name
        if last not in result.cameras:
            result.cameras[last] = CameraPose(
                image_id=len(image_files) - 1,
                frame_id=last,
                position=current_t.copy(),
                rotation=current_R.copy(),
                quaternion=_rotation_matrix_to_quat(current_R),
                intrinsics=K,
                distortion=np.zeros(5),
                is_estimated=True,
            )

    elapsed = (time.perf_counter() - start) * 1000
    result.num_registered = len(result.cameras)
    result.reconstruction_time_ms = round(elapsed, 2)

    log.info(
        "opencv_pose_estimation_complete",
        registered=result.num_registered,
        pairs=len(result.image_pairs),
    )

    return result


def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)
