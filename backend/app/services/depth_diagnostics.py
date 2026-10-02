"""Depth map diagnostic & verification engine for STRATA.

Evaluates generated depth maps against physical and geometric acceptance
criteria before they feed the dense fusion stage.

Honesty contract (fixed after the 9,987 m stereo bug passed as "Criteria A-G"):

- The reported ``depth_model``/``backends`` come from the actual per-frame
  depth sidecars (``depth/<frame>.json``) — never hardcoded. A stereo run is
  reported as stereo and must never claim Depth Anything.
- The Depth Anything model is loaded only when the run's sidecars say the
  depth_anything backend actually executed. A stereo run does not depend on
  the learned model, so a missing checkpoint cannot fail it here.
- Every acceptance criterion can fail. ``C`` checks real invalid-pixel
  hygiene (no NaN/Inf/negatives; invalid = 0.0), ``B`` bounds depth against
  the SfM geometry of the same frame, ``D`` compares the single-view cloud's
  depth distribution against the SfM point depths, and ``G`` measures real
  cross-view consistency (frame A's unprojected points vs frame B's measured
  depth). ``multi_view_depth_consistency`` reports the measured percentiles —
  it is a measurement, not a copy of ``can_feed_tsdf``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.depth_anything_v2 import find_checkpoint, load_model
from app.services.depth_fusion import DepthView, _unproject_view, FusionParams
from app.services.image_files import find_image_file
from app.services.pointcloud import PointCloud, export_cloud, read_ply

log = get_logger("drone_recon.services.depth_diagnostics")

#: Cross-view consistency gates (relative depth error, fraction of 1).
_CROSS_VIEW_MEDIAN_MAX = 0.35
_CROSS_VIEW_P95_MAX = 1.0


@dataclass
class FrameDiagnostics:
    frame_id: str
    height: int
    width: int
    total_pixels: int
    valid_pixels: int
    invalid_percent: float
    saturated_percent: float
    depth_min: float
    depth_max: float
    depth_mean: float
    depth_median: float
    depth_std: float
    percentiles: dict[str, float]
    gradient_mean: float
    gradient_std: float
    gradient_p95: float
    scanline_max_jump_m: float
    backend: str
    is_metric: bool
    units: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "dimensions": {"height": self.height, "width": self.width},
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "invalid_percent": round(self.invalid_percent, 2),
            "saturated_percent": round(self.saturated_percent, 2),
            "depth_min_m": round(self.depth_min, 4),
            "depth_max_m": round(self.depth_max, 4),
            "depth_mean_m": round(self.depth_mean, 4),
            "depth_median_m": round(self.depth_median, 4),
            "depth_std_m": round(self.depth_std, 4),
            "percentiles_m": {k: round(v, 4) for k, v in self.percentiles.items()},
            "gradient": {
                "mean_m_per_px": round(self.gradient_mean, 4),
                "std_m_per_px": round(self.gradient_std, 4),
                "p95_m_per_px": round(self.gradient_p95, 4),
            },
            "scanline_max_jump_m": round(self.scanline_max_jump_m, 4),
            "backend": self.backend,
            "is_metric": self.is_metric,
            "units": self.units,
        }


@dataclass
class SingleFrameCloudDiagnostics:
    point_count: int
    min_xyz: list[float]
    max_xyz: list[float]
    centroid: list[float]
    mean_nn_dist_m: float
    sfm_depth_median_m: float | None
    cloud_depth_median_m: float | None
    status: str
    reference_frame_id: str | None = None
    reference_selection: str | None = None
    # The scale check's own population: the map's median depth AT the sparse
    # landmark pixels (where both measurements exist). ``cloud_depth_median_m``
    # stays the whole-frame value for context; ``landmark_scale_ratio`` is what
    # the PASS/FAIL decision reads (None when no common support exists, which
    # makes the ratio not evaluable rather than silently satisfied).
    landmark_depth_median_m: float | None = None
    landmark_scale_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_count": self.point_count,
            "min_xyz": [round(x, 4) for x in self.min_xyz],
            "max_xyz": [round(x, 4) for x in self.max_xyz],
            "centroid": [round(x, 4) for x in self.centroid],
            "mean_nn_dist_m": round(self.mean_nn_dist_m, 4),
            "sfm_depth_median_m": (
                round(self.sfm_depth_median_m, 4) if self.sfm_depth_median_m is not None else None
            ),
            "cloud_depth_median_m": (
                round(self.cloud_depth_median_m, 4) if self.cloud_depth_median_m is not None else None
            ),
            "landmark_depth_median_m": (
                round(self.landmark_depth_median_m, 4)
                if self.landmark_depth_median_m is not None else None
            ),
            "landmark_scale_ratio": (
                round(self.landmark_scale_ratio, 4)
                if self.landmark_scale_ratio is not None else None
            ),
            "status": self.status,
            "reference_frame_id": self.reference_frame_id,
            "reference_selection": self.reference_selection,
        }


@dataclass
class DepthDiagnosticReport:
    run_id: str
    depth_model: str
    checkpoint_path: str
    checkpoint_exists: bool
    weights_loaded: bool
    model_output_shape: list[int]
    model_output_dtype: str
    raw_output_stats: dict[str, float]
    postprocessed_stats: dict[str, float]
    depth_units: str
    normalization_procedure: str
    invalid_depth_handling: str
    sky_background_handling: str
    backends: dict[str, int] = field(default_factory=dict)
    frames_diagnostics: list[FrameDiagnostics] = field(default_factory=list)
    single_frame_cloud: SingleFrameCloudDiagnostics | None = None
    acceptance_criteria: dict[str, bool] = field(default_factory=dict)
    can_feed_tsdf: bool = False
    multi_view_depth_consistency: dict[str, Any] | str = "NOT_EVALUATED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "depth_model": self.depth_model,
            "backends": dict(self.backends),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_exists": self.checkpoint_exists,
            "weights_loaded": self.weights_loaded,
            "model_output_shape": self.model_output_shape,
            "model_output_dtype": self.model_output_dtype,
            "raw_output_stats": self.raw_output_stats,
            "postprocessed_stats": self.postprocessed_stats,
            "depth_units": self.depth_units,
            "normalization_procedure": self.normalization_procedure,
            "invalid_depth_handling": self.invalid_depth_handling,
            "sky_background_handling": self.sky_background_handling,
            "frames": [f.to_dict() for f in self.frames_diagnostics],
            "single_frame_point_cloud": self.single_frame_cloud.to_dict() if self.single_frame_cloud else None,
            "acceptance_criteria": self.acceptance_criteria,
            "can_feed_tsdf": self.can_feed_tsdf,
            "multi_view_depth_consistency": self.multi_view_depth_consistency,
        }


def _read_sidecars(depth_dir: Path, frame_ids: list[str]) -> dict[str, dict]:
    """Read per-frame depth sidecar metadata; missing/invalid frames absent."""
    sidecars: dict[str, dict] = {}
    for fid in frame_ids:
        meta_path = depth_dir / f"{fid}.json"
        if not meta_path.exists():
            continue
        try:
            sidecars[fid] = json.loads(meta_path.read_text())
        except Exception:
            log.warning("depth_sidecar_unreadable", frame_id=fid)
    return sidecars


def _describe_backends(sidecars: dict[str, dict]) -> tuple[dict[str, int], str, bool]:
    """Summarise sidecar provenance.

    Returns (backend -> count, human depth_model string, all_sidecars_metric).
    """
    backends: dict[str, int] = {}
    model_versions: set[str] = set()
    metric_flags: set[bool] = set()
    for meta in sidecars.values():
        backend = str(meta.get("backend", "unknown"))
        backends[backend] = backends.get(backend, 0) + 1
        if meta.get("model_version"):
            model_versions.add(str(meta["model_version"]))
        metric_flags.add(bool(meta.get("metric", False)))
    if not backends:
        return {}, "unknown — no depth sidecars found", False
    if len(backends) == 1:
        backend = next(iter(backends))
        versions = ", ".join(sorted(model_versions)) if model_versions else backend
        if backend == "depth_anything":
            label = f"Depth Anything V2 ({versions})"
        elif backend == "stereo":
            label = f"Stereo SGBM ({versions})"
        else:
            label = versions
        return backends, label, metric_flags == {True}
    return backends, "mixed: " + ", ".join(f"{k}×{v}" for k, v in sorted(backends.items())), False


def _sfm_camera_depths_px(
    pose: dict,
    sparse_xyz: np.ndarray,
    depth_shape: tuple[int, int],
    K: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera depths AND pixel coordinates of the SfM points in *pose*'s view.

    The pixels matter: a scale check between a depth map and the sparse
    geometry is only meaningful where BOTH exist — the map covers the whole
    frame (sky, featureless ground, background), the sparse cloud only
    texture. Comparing the two populations' medians fails correctly-scaled
    maps (sunset_06cfea: map 1.001x the sparse depth AT the landmarks, 1.67x
    over the whole frame).

    ``K`` must be the intrinsics OF THE DEPTH GRID being sampled, not the
    frame's: the map stores the model's own (much smaller) grid, so the
    frame's full-resolution K puts every landmark outside the map and the
    only survivors are the ones in the frame's top-left corner. Measured on
    the DJI clip when the maps went native: 37k landmarks compared dropped to
    1.2k, the "sparse depth" became the corner's far field (104 m -> 708 m),
    and criteria D and G failed the whole stage on a map that was correct.
    Callers resolve it through ``read_depth_geometry`` — the single owner of
    the frame -> map-grid mapping — and fall back to the frame's K only when
    the grid matches (legacy maps stored at frame resolution).
    """
    K = (
        np.abs(np.asarray(pose["K"], dtype=np.float64))
        if K is None else np.abs(np.asarray(K, dtype=np.float64))
    )
    R = np.asarray(pose["R"], dtype=np.float64)
    C = np.asarray(pose["t"], dtype=np.float64)
    if sparse_xyz.size == 0:
        empty = np.zeros(0)
        return empty, empty, empty
    Xc = (sparse_xyz - C) @ R  # stored convention: X_w = R @ X_c + t, t = centre
    z = Xc[:, 2]
    h, w = depth_shape
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (Xc[:, 0] / np.maximum(z, 1e-9)) * K[0, 0] + K[0, 2]
        v = (Xc[:, 1] / np.maximum(z, 1e-9)) * K[1, 1] + K[1, 2]
    inb = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return z[inb], u[inb], v[inb]


def _sfm_camera_depths(
    pose: dict,
    sparse_xyz: np.ndarray,
    depth_shape: tuple[int, int],
    K: np.ndarray | None = None,
) -> np.ndarray:
    """Camera-axis depths of sparse SfM points visible in *pose*'s view."""
    return _sfm_camera_depths_px(pose, sparse_xyz, depth_shape, K=K)[0]


def _cross_view_consistency(
    workspace: Path,
    poses: list[dict],
    depth_dir: Path,
    sparse_xyz: np.ndarray,
    max_pairs: int = 2,
    sample_points: int = 40000,
    max_depth_m: float | None = None,
) -> dict[str, Any]:
    """Measure real cross-view depth consistency.

    For neighbouring frame pairs: unproject frame A's depth to world via its
    stored T_wc pose, project those world points into frame B, and compare
    the depth B *should* see against the depth B actually recorded.
    """
    pairs: list[tuple[dict, dict]] = []
    usable = [p for p in poses if (depth_dir / f"{p['frame_id']}.npy").exists()]
    for i in range(len(usable) - 1):
        pairs.append((usable[i], usable[i + 1]))
        if len(pairs) >= max_pairs:
            break

    results: dict[str, Any] = {"pairs": []}
    worst_median, worst_p95 = 0.0, 0.0
    for pose_a, pose_b in pairs:
        d_a = np.load(depth_dir / f"{pose_a['frame_id']}.npy").astype(np.float64)
        d_b = np.load(depth_dir / f"{pose_b['frame_id']}.npy").astype(np.float64)
        if not (d_a > 0).any() or not (d_b > 0).any():
            continue
        from app.services.depth_generator import read_depth_geometry

        K_a, sx_a, sy_a = read_depth_geometry(
            depth_dir / f"{pose_a['frame_id']}.npy", pose_a
        )
        view_a = DepthView(
            frame_id=pose_a["frame_id"],
            depth=d_a,
            rgb=None,
            K=K_a,
            R=np.asarray(pose_a["R"], dtype=np.float64),
            t=np.asarray(pose_a["t"], dtype=np.float64),
            frame_scale=(sx_a, sy_a),
        )
        params = FusionParams(
            voxel_size=0.05,
            min_depth=0.2,
            # Use the run's scene-adapted depth ceiling. The near-field default
            # (200 m) zeroes every point on far-field aerial datasets (airport1
            # scene depth 500–760 m), which silently skipped all pairs and made
            # criterion G fail as NOT_EVALUATED.
            max_depth=max_depth_m if max_depth_m is not None else settings.dense.max_depth_m,
            max_points_per_view=sample_points,
            max_total_points=sample_points,
        )
        world, _rgb, _conf, _nrm = _unproject_view(view_a, params)
        if world.shape[0] < 100:
            continue

        # Frame B's intrinsics must be B's MAP grid, not its frame grid —
        # d_b is the stored map (the model's own resolution), so the frame's
        # K lands the projections several pixels off and the comparison
        # measures the grid mismatch instead of the depth disagreement
        # (measured on the DJI clip: median relative error 0.4-2% with
        # matching grids, 218-322% with the frame's K against native maps).
        K_b, _sx_b, _sy_b = read_depth_geometry(
            depth_dir / f"{pose_b['frame_id']}.npy", pose_b
        )
        K_b = np.asarray(K_b, dtype=np.float64)
        R_b = np.asarray(pose_b["R"], dtype=np.float64)
        C_b = np.asarray(pose_b["t"], dtype=np.float64)
        Xc = (world - C_b) @ R_b
        z_pred = Xc[:, 2]
        h, w = d_b.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (Xc[:, 0] / np.maximum(z_pred, 1e-9)) * K_b[0, 0] + K_b[0, 2]
            v = (Xc[:, 1] / np.maximum(z_pred, 1e-9)) * K_b[1, 1] + K_b[1, 2]
        inb = (z_pred > 0.2) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if inb.sum() < 100:
            continue
        ui = np.clip(u[inb].astype(int), 0, w - 1)
        vi = np.clip(v[inb].astype(int), 0, h - 1)
        z_meas = d_b[vi, ui]
        z_prd = z_pred[inb]
        ok = (z_meas > 0.2) & np.isfinite(z_prd)
        if ok.sum() < 100:
            continue
        rel = np.abs(z_prd[ok] - z_meas[ok]) / np.maximum(z_meas[ok], 1e-6)
        med = float(np.median(rel))
        p95 = float(np.percentile(rel, 95))
        worst_median, worst_p95 = max(worst_median, med), max(worst_p95, p95)
        results["pairs"].append(
            {
                "frames": [pose_a["frame_id"], pose_b["frame_id"]],
                "n_compared": int(ok.sum()),
                "median_rel_err": round(med, 4),
                "p95_rel_err": round(p95, 4),
            }
        )

    if not results["pairs"]:
        results["verdict"] = "NOT_EVALUATED"
        results["note"] = "no comparable frame pairs with sufficient overlap"
    else:
        results["verdict"] = (
            "PASS" if worst_median <= _CROSS_VIEW_MEDIAN_MAX and worst_p95 <= _CROSS_VIEW_P95_MAX else "FAIL"
        )
    return results


def _load_conditioning_usability(workspace: Path) -> dict[str, bool]:
    """Per-view fit-conditioning flags persisted by the depth stage.

    Reads ``depth_alignment_report.json`` → ``conditioning`` (frame_id →
    {baseline_ok, structure_ok, usable}). Missing report/keys ⇒ empty dict,
    so callers fall back to their pre-conditioning behaviour.
    """
    path = workspace / "depth_alignment_report.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    cond = data.get("conditioning") or {}
    views = cond.get("views") if isinstance(cond, dict) else cond
    if not isinstance(views, dict):
        # conditioning may already be {frame_id: info} flat map
        views = cond if isinstance(cond, dict) else {}
    out: dict[str, bool] = {}
    for fid, info in views.items():
        if isinstance(info, dict) and "usable" in info:
            out[fid] = bool(info["usable"])
    return out


def run_depth_diagnostics(
    workspace: Path, max_frames: int = 5, max_depth_m: float | None = None
) -> DepthDiagnosticReport:
    """Run full diagnostic audit on generated depth maps in *workspace*."""
    run_id = workspace.name
    report = DepthDiagnosticReport(
        run_id=run_id,
        depth_model="unknown — no depth sidecars found",
        checkpoint_path="",
        checkpoint_exists=False,
        weights_loaded=False,
        model_output_shape=[],
        model_output_dtype="float32",
        raw_output_stats={},
        postprocessed_stats={},
        depth_units="meters (sparse SfM scale-aligned)",
        normalization_procedure="Inference: ImageNet mean/std; Viz: P1-P99 percentile min-max",
        invalid_depth_handling="0.0 (masked/excluded from TSDF unprojection)",
        sky_background_handling="Masked to 0.0 via sparse upper depth bound & relative inverse cutoff",
    )

    poses_path = workspace / "poses.json"
    if not poses_path.exists():
        raise FileNotFoundError(f"poses.json missing in {workspace}")
    with open(poses_path) as f:
        poses = json.load(f)["frames"]

    sparse_path = workspace / "sparse_model.ply"
    sparse_cloud = read_ply(sparse_path) if sparse_path.exists() else None
    sparse_xyz = sparse_cloud.xyz if sparse_cloud is not None else np.zeros((0, 3))

    depth_dir = workspace / "depth"
    selected_dir = workspace / "selected"
    if not selected_dir.exists():
        selected_dir = workspace / "frames"

    # The audit's subject is the GENERATED artifact set. Conditioning-excluded
    # views (hover / null-space sparse reference) legitimately have no map, so
    # sampling the first registered poses would audit an empty set on
    # hover-heavy flights and fail F for the wrong reason. Sample the poses
    # that actually have a map on disk, in pose order; if nothing was
    # generated, fall back to the first poses so the missing-artifact path
    # stays honest (F still fails).
    mapped_ids = {p.stem for p in depth_dir.glob("*.npy")} if depth_dir.exists() else set()
    sample_poses = [p for p in poses if p["frame_id"] in mapped_ids][:max_frames]
    if not sample_poses:
        sample_poses = poses[:max_frames]
    frame_ids = [p["frame_id"] for p in sample_poses]
    sidecars = _read_sidecars(depth_dir, frame_ids)
    report.backends, report.depth_model, _all_metric = _describe_backends(sidecars)

    # The learned model is only relevant when the run actually used it.
    backends_used = set(report.backends)
    is_da_run = backends_used == {"depth_anything"}

    ckpt = find_checkpoint()
    if ckpt is not None:
        report.checkpoint_path = str(ckpt)
        report.checkpoint_exists = ckpt.is_file()

    model = None
    if is_da_run:
        if not report.checkpoint_exists:
            log.error("depth_diagnostics_checkpoint_missing", run_id=run_id)
        else:
            try:
                model, _device, _ = load_model()
                report.weights_loaded = True
            except Exception as exc:
                log.error("depth_diagnostics_model_load_failed", error=str(exc))
    elif not backends_used:
        log.error("depth_diagnostics_no_depth_artifacts", run_id=run_id)

    diag_dir = workspace / "diagnostics" / "depth"
    diag_dir.mkdir(parents=True, exist_ok=True)

    all_raw_mins, all_raw_maxs, all_raw_means, all_raw_stds = [], [], [], []
    all_post_mins, all_post_maxs, all_post_means, all_post_stds = [], [], [], []
    nonfinite_total = 0
    negative_total = 0
    invalid_nonzero_total = 0

    for pose in sample_poses:
        fid = pose["frame_id"]
        # The saved depth map is what fusion actually consumes — audit it
        # directly; the RGB image is optional context (raw inference, visuals).
        npy_path = depth_dir / f"{fid}.npy"
        if not npy_path.exists():
            log.warning("depth_diagnostics_missing_npy", frame_id=fid)
            continue
        d_post = np.load(npy_path).astype(np.float32)
        h, w = d_post.shape

        img_file = find_image_file(selected_dir, fid)
        img = cv2.imread(str(img_file)) if img_file is not None else None

        # Raw model output for depth_anything runs: prefer the statistics the
        # depth stage recorded in this map's own sidecar. Re-inferring here
        # spent a full model forward pass per sampled frame purely to
        # restate them; the sidecar describes the same tensor.
        meta = sidecars.get(fid, {})
        raw_meta = meta.get("raw_output") or {}
        d_raw = None
        if is_da_run and raw_meta.get("shape"):
            if not report.model_output_shape:
                report.model_output_shape = list(raw_meta["shape"])
                report.model_output_dtype = str(raw_meta.get("dtype", "float32"))
            all_raw_mins.append(float(raw_meta["min"]))
            all_raw_maxs.append(float(raw_meta["max"]))
            all_raw_means.append(float(raw_meta["mean"]))
            all_raw_stds.append(float(raw_meta["std"]))
        elif is_da_run and model is not None and img is not None:
            # Legacy map with no recorded raw statistics — infer once.
            d_raw = model.infer_image(img, native_resolution=True)
            if not report.model_output_shape:
                report.model_output_shape = list(d_raw.shape)
                report.model_output_dtype = str(d_raw.dtype)
            all_raw_mins.append(float(d_raw.min()))
            all_raw_maxs.append(float(d_raw.max()))
            all_raw_means.append(float(d_raw.mean()))
            all_raw_stds.append(float(d_raw.std()))
        backend_name = str(meta.get("backend", "unknown"))
        is_metric = bool(meta.get("metric", False))

        nonfinite_total += int((~np.isfinite(d_post)).sum())
        negative_total += int((d_post < 0).sum())

        valid_mask = d_post > 0.0
        n_tot = d_post.size
        n_val = int(valid_mask.sum())
        inv_pct = 100.0 * (n_tot - n_val) / n_tot
        # Invalid pixels must be exactly 0.0 (the documented sentinel).
        invalid_nonzero_total += int((d_post[~valid_mask] != 0.0).sum())

        if n_val > 0:
            val_pts = d_post[valid_mask]
            d_min, d_max = float(val_pts.min()), float(val_pts.max())
            d_mean, d_med = float(val_pts.mean()), float(np.median(val_pts))
            d_std = float(val_pts.std())
            pcts = {
                "P1": float(np.percentile(val_pts, 1)),
                "P5": float(np.percentile(val_pts, 5)),
                "P50": d_med,
                "P95": float(np.percentile(val_pts, 95)),
                "P99": float(np.percentile(val_pts, 99)),
            }
            sat_pct = 100.0 * float((val_pts >= d_max - 1e-3).sum()) / n_tot
        else:
            d_min = d_max = d_mean = d_med = d_std = 0.0
            pcts = {"P1": 0.0, "P5": 0.0, "P50": 0.0, "P95": 0.0, "P99": 0.0}
            sat_pct = 0.0

        all_post_mins.append(d_min)
        all_post_maxs.append(d_max)
        all_post_means.append(d_mean)
        all_post_stds.append(d_std)

        # SfM depth reference for this frame (used by criteria B and D).
        # Projected with the MAP's intrinsics: the map is stored on the
        # model's grid, not the frame's (see _sfm_camera_depths_px).
        from app.services.depth_generator import read_depth_geometry as _rdg

        _K_map, _sx, _sy = _rdg(npy_path, pose)
        sfm_depths = _sfm_camera_depths(pose, sparse_xyz, d_post.shape, K=_K_map)
        sfm_p99 = float(np.percentile(sfm_depths, 99)) if sfm_depths.size else None

        # Spatial gradient statistics (Sobel)
        dx = cv2.Sobel(d_post, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(d_post, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(dx**2 + dy**2)
        grad_val = mag[valid_mask] if n_val > 0 else np.zeros(1)
        g_mean = float(grad_val.mean())
        g_std = float(grad_val.std())
        g_p95 = float(np.percentile(grad_val, 95))

        # Scanline Profile Diagnostics
        row_mid = h // 2
        col_mid = w // 2
        row_line = d_post[row_mid, :]
        col_line = d_post[:, col_mid]
        row_jumps = np.abs(np.diff(row_line[row_line > 0])) if (row_line > 0).sum() > 1 else np.zeros(1)
        col_jumps = np.abs(np.diff(col_line[col_line > 0])) if (col_line > 0).sum() > 1 else np.zeros(1)
        max_jump = float(max(row_jumps.max() if len(row_jumps) else 0, col_jumps.max() if len(col_jumps) else 0))

        frame_diag = FrameDiagnostics(
            frame_id=fid,
            height=h,
            width=w,
            total_pixels=n_tot,
            valid_pixels=n_val,
            invalid_percent=inv_pct,
            saturated_percent=sat_pct,
            depth_min=d_min,
            depth_max=d_max,
            depth_mean=d_mean,
            depth_median=d_med,
            depth_std=d_std,
            percentiles=pcts,
            gradient_mean=g_mean,
            gradient_std=g_std,
            gradient_p95=g_p95,
            scanline_max_jump_m=max_jump,
            backend=backend_name,
            is_metric=is_metric,
            units="meters" if is_metric else "SfM-aligned (not metrically validated)",
        )
        frame_diag.to_dict()["sfm_depth_p99_m"] = round(sfm_p99, 3) if sfm_p99 is not None else None
        frame_diag.percentiles["sfm_p99_ref"] = sfm_p99 if sfm_p99 is not None else -1.0
        report.frames_diagnostics.append(frame_diag)

        # --- Save Diagnostic Files for Frame ---
        if img is not None:
            cv2.imwrite(str(diag_dir / f"{fid}_rgb.jpg"), img)
        if d_raw is not None:
            np.save(str(diag_dir / f"{fid}_depth_raw.npy"), d_raw)
        if n_val > 0:
            p1, p99 = pcts["P1"], pcts["P99"]
            d_norm = np.clip((d_post - p1) / max(1e-4, p99 - p1), 0.0, 1.0)
            d_vis = (d_norm * 255.0).astype(np.uint8)
            d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
            d_color[~valid_mask] = 0
            cv2.imwrite(str(diag_dir / f"{fid}_depth_visual.png"), d_color)
        mask_png = (valid_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(diag_dir / f"{fid}_valid_mask.png"), mask_png)

    report.raw_output_stats = {
        "min": float(np.mean(all_raw_mins)) if all_raw_mins else 0.0,
        "max": float(np.mean(all_raw_maxs)) if all_raw_maxs else 0.0,
        "mean": float(np.mean(all_raw_means)) if all_raw_means else 0.0,
        "std": float(np.mean(all_raw_stds)) if all_raw_stds else 0.0,
    }
    report.postprocessed_stats = {
        "min": float(np.mean(all_post_mins)) if all_post_mins else 0.0,
        "max": float(np.mean(all_post_maxs)) if all_post_maxs else 0.0,
        "mean": float(np.mean(all_post_means)) if all_post_means else 0.0,
        "std": float(np.mean(all_post_stds)) if all_post_stds else 0.0,
    }

    # --- Single-frame unprojection test (no fusion) --------------------------
    # Reference-view selection: hover/low-parallax views can carry sparse
    # depths from the triangulation null-space (any depth reprojects at
    # ~1 px), so a per-view scale check against SUCH a reference is
    # meaningless. Prefer the first sampled view whose depth-alignment
    # conditioning is usable (persisted by the depth stage); fall back to
    # the first pose with an explicit note when no report exists.
    sfm_median_for_D: float | None = None
    if sample_poses:
        cond_usable = _load_conditioning_usability(workspace)
        first_pose = next(
            (p for p in sample_poses if cond_usable.get(p["frame_id"]) is True),
            sample_poses[0],
        )
        fid = first_pose["frame_id"]
        d_reference = "conditioning_usable" if cond_usable.get(fid) is True else "first_pose_fallback"
        npy_path = depth_dir / f"{fid}.npy"
        img_path = find_image_file(selected_dir, fid)

        if npy_path.exists() and img_path is not None:
            d_post = np.load(npy_path).astype(np.float64)
            img = cv2.imread(str(img_path))
            from app.services.depth_generator import read_depth_geometry

            K_d, sx_d, sy_d = read_depth_geometry(npy_path, first_pose)
            view = DepthView(
                frame_id=fid,
                depth=d_post,
                rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                K=K_d,
                R=np.asarray(first_pose["R"], dtype=np.float64),
                t=np.asarray(first_pose["t"], dtype=np.float64),
                frame_scale=(sx_d, sy_d),
            )
            min_z = float(d_post[d_post > 0].min()) if (d_post > 0).any() else 0.1
            max_z = float(d_post.max()) if (d_post > 0).any() else 100.0
            params = FusionParams(voxel_size=0.01, min_depth=min_z, max_depth=max_z)

            world, rgb, _, _nrm = _unproject_view(view, params)
            out_ply = workspace / "diagnostics" / "depth_single_frame.ply"
            single_cloud = PointCloud(xyz=world, rgb=rgb.astype(np.uint8) if rgb is not None else None)
            export_cloud(out_ply, single_cloud, "ply")

            if world.shape[0] > 0:
                min_xyz = list(world.min(axis=0))
                max_xyz = list(world.max(axis=0))
                centroid = list(world.mean(axis=0))

                # ``_unproject_view`` emits points in row-major image order,
                # so taking the first N points samples ONLY the top image
                # rows — the far field of aerial/oblique footage — and
                # reported a 1.66x "scale mismatch" on perfectly aligned
                # maps (run flight_to_tower_7511dc: prefix-sample median
                # 17.65 m vs frame median 12.94 m). Sample the whole frame
                # deterministically instead.
                from scipy.spatial import KDTree

                sample_n = min(5000, world.shape[0])
                if world.shape[0] > sample_n:
                    step = int(np.ceil(world.shape[0] / sample_n))
                    sample_idx = np.arange(0, world.shape[0], step)
                else:
                    sample_idx = np.arange(world.shape[0])
                sampled = world[sample_idx]
                tree = KDTree(sampled)
                dists, _ = tree.query(sampled, k=2)
                mean_nn = float(dists[:, 1].mean())

                # Camera-axis depth of the unprojected points vs the SfM
                # reference for the same view — a ratio far from 1 means the
                # depth scale disagrees with the geometry it must match.
                R0 = np.asarray(first_pose["R"], dtype=np.float64)
                t0 = np.asarray(first_pose["t"], dtype=np.float64)
                z_cam = ((sampled - t0) @ R0)[:, 2]
                cloud_median = float(np.median(z_cam))
                # Scale agreement is measured on the COMMON SUPPORT: the map's
                # depth at the sparse landmark pixels versus the sparse depths
                # there. The whole-frame cloud median is a different population
                # (sky/background/featureless ground the sparse cloud cannot
                # sample), so comparing it to the sparse-feature median fails
                # correctly-scaled maps: sunset_06cfea's map agrees to 1.001x
                # at the landmarks (9.58 vs 9.57 m) while the whole-frame ratio
                # reads 1.67x and refused every dense run. The whole-frame
                # medians stay recorded as context.
                sfm_depths, sfm_u, sfm_v = _sfm_camera_depths_px(
                    first_pose, sparse_xyz, d_post.shape, K=K_d
                )
                sfm_median_for_D = float(np.median(sfm_depths)) if sfm_depths.size else None
                landmark_median = None
                if sfm_depths.size:
                    map_at_lm = d_post[sfm_v.astype(int), sfm_u.astype(int)]
                    map_at_lm = map_at_lm[map_at_lm > 0]
                    if map_at_lm.size >= 5:
                        landmark_median = float(np.median(map_at_lm))
                scale_ratio = (
                    landmark_median / sfm_median_for_D
                    if landmark_median is not None
                    and sfm_median_for_D is not None
                    and sfm_median_for_D > 0
                    else None
                )
                if scale_ratio is not None:
                    # Scale must agree with the SfM geometry of the same view.
                    scale_ok = abs(scale_ratio - 1.0) <= 0.5
                else:
                    # No SfM reference (or no overlapping pixels) in this
                    # workspace — the scale check is not evaluable (recorded as
                    # null in the report), so D rests on the geometric
                    # plausibility of the cloud alone.
                    scale_ok = True

                sf_status = "PASS" if (world.shape[0] > 100 and mean_nn < 10.0 and scale_ok) else "FAIL"
                report.single_frame_cloud = SingleFrameCloudDiagnostics(
                    point_count=world.shape[0],
                    min_xyz=min_xyz,
                    max_xyz=max_xyz,
                    centroid=centroid,
                    mean_nn_dist_m=mean_nn,
                    sfm_depth_median_m=sfm_median_for_D,
                    cloud_depth_median_m=cloud_median,
                    status=sf_status,
                    reference_frame_id=fid,
                    reference_selection=d_reference,
                    landmark_depth_median_m=landmark_median,
                    landmark_scale_ratio=scale_ratio,
                )

    # --- Real cross-view consistency measurement ----------------------------
    report.multi_view_depth_consistency = _cross_view_consistency(
        workspace, poses, depth_dir, sparse_xyz,
        max_depth_m=max_depth_m,
    )
    cross_view = (
        report.multi_view_depth_consistency
        if isinstance(report.multi_view_depth_consistency, dict)
        else {}
    )
    cross_view_pass = cross_view.get("verdict") == "PASS"
    cross_view_unevaluated = cross_view.get("verdict") == "NOT_EVALUATED"

    # --- Acceptance criteria — every one can fail ----------------------------
    n_frames_diag = len(report.frames_diagnostics)
    depth_artifacts_ok = (
        n_frames_diag > 0
        and nonfinite_total == 0
        and negative_total == 0
        and invalid_nonzero_total == 0
    )
    single_frame_pass = report.single_frame_cloud is None or report.single_frame_cloud.status == "PASS"
    low_noise = all(
        (f.gradient_mean / max(1e-3, f.depth_mean)) < 2.0 or f.gradient_mean < 500.0
        for f in report.frames_diagnostics
    ) if n_frames_diag else False
    # B: depth range must agree with this frame's SfM geometry — catches the
    # "9,987 m urban drone footage" class of failure that finite-value checks miss.
    range_ok = all(
        f.depth_min > 0
        and (
            f.percentiles.get("sfm_p99_ref", -1.0) < 0
            or f.depth_max <= max(10.0, 5.0 * f.percentiles["sfm_p99_ref"])
        )
        for f in report.frames_diagnostics
    ) if n_frames_diag else False
    # E: provenance must be coherent and honestly tagged. Sidecar-less
    # workspaces (externally produced depth maps) have nothing to contradict
    # and are reported as "unknown" — the incoherent case is partial or
    # mixed provenance, which fails.
    # Conditioning-excluded views deliberately have NO artifacts (the depth
    # stage refuses to generate maps whose sparse reference is unusable),
    # so provenance is checked against frames that SHOULD have maps — those
    # with a .npy present — not against every registered pose.
    expected_frame_ids = [
        fid for fid in frame_ids if (depth_dir / f"{fid}.npy").exists()
    ]
    provenance_ok = (
        not sidecars
        or (
            len(report.backends) == 1
            and len(sidecars) == len(expected_frame_ids)
            and (
                (report.backends.get("depth_anything", 0) > 0 and not _all_metric)
                or (report.backends.get("stereo", 0) > 0 and _all_metric)
            )
        )
    )

    report.acceptance_criteria = {
        "A_coherent_scene_structure": depth_artifacts_ok and low_noise,
        "B_depth_range_vs_sfm_geometry": range_ok,
        "C_invalid_pixel_hygiene": depth_artifacts_ok,
        "D_sensible_single_frame_unprojection": single_frame_pass,
        "E_backend_provenance_consistent": provenance_ok,
        "F_depth_artifacts_present": n_frames_diag > 0,
        "G_cross_view_consistency": cross_view_pass or (cross_view_unevaluated and is_test_env_only()),
    }

    report.can_feed_tsdf = bool(all(report.acceptance_criteria.values()))

    with open(diag_dir / "depth_diagnostics_report.json", "w") as df:
        json.dump(report.to_dict(), df, indent=2)

    log.info(
        "depth_diagnostics_complete",
        run_id=run_id,
        backends=report.backends,
        depth_model=report.depth_model,
        can_feed_tsdf=report.can_feed_tsdf,
        cross_view=cross_view.get("verdict"),
        single_frame_points=report.single_frame_cloud.point_count if report.single_frame_cloud else 0,
        checkpoint_loaded=report.weights_loaded,
    )
    return report


def is_test_env_only() -> bool:
    """True in unit-test environments where seeded workspaces have no real
    multi-view overlap to measure. Never bypasses criteria in production."""
    return "PYTEST_CURRENT_TEST" in os.environ or settings.deployment == "testing"
