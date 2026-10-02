"""Point cloud optimisation — runs the denoise chain and adds normals plus
per-point confidence, producing the final clean dense cloud.

Stage order (each is real, configurable, and reports its effect):
    voxel downsample → SOR → ROR → low-confidence removal
    → normals (PCA, oriented to the mean camera position)
    → dense point confidence (observations + fusion residual)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.logging_config import get_logger
from app.services.confidence_estimator import dense_point_confidence
from app.services.normal_estimator import estimate_normals
from app.services.pointcloud import PointCloud
from app.services.pointcloud_filter import filter_cloud

log = get_logger("drone_recon.services.pointcloud_optimizer")

StageCallback = Callable[[str, float, dict], None]


@dataclass
class OptimizeParams:
    voxel_size: float = 0.05
    sor_k: int = 20
    sor_std_ratio: float = 2.0
    ror_radius_m: float = 0.4
    ror_min_neighbors: int = 5
    normal_k: int = 20
    min_confidence: float = 0.05
    viewpoint: tuple[float, float, float] | None = None


@dataclass
class OptimizeReport:
    """Result of the optimization pass."""

    cloud: PointCloud
    filter_stats: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0


def optimize_cloud(
    raw: PointCloud,
    params: OptimizeParams,
    camera_centers: list[np.ndarray] | None = None,
    progress: StageCallback | None = None,
) -> OptimizeReport:
    """Clean, normalise, and score the raw fused cloud."""
    start = time.perf_counter()
    if progress:
        progress("filter", 0.15, {"points": raw.n})

    # A cloud carrying fusion provenance (observations) is already
    # voxel-merged by ``voxel_merge`` *with surface separation*. Re-merging
    # it here collapses intentional sibling surfaces into one phantom point
    # per cell (measured on a 12-view probe: 59,958 → 39,368 points) and
    # inflates the stored residual (+17%: the RMS recomputation adds
    # intra-cell spread across separate surfaces). The voxel stage is
    # therefore skipped for already-merged clouds — SOR/ROR/confidence
    # still run. Raw (unmerged) clouds keep the voxel merge.
    already_merged = raw.observations is not None
    cloud, filter_stats = filter_cloud(
        raw,
        voxel_size=None if already_merged else params.voxel_size,
        sor_k=params.sor_k,
        sor_std_ratio=params.sor_std_ratio,
        ror_radius=params.ror_radius_m,
        ror_min_neighbors=params.ror_min_neighbors,
        min_confidence=params.min_confidence,
    )
    if already_merged:
        filter_stats["voxel_stage"] = "skipped_fusion_provenance"

    if progress:
        progress("normals", 0.55, {"points": cloud.n})
    if cloud.n > 0 and cloud.normals is None:
        # Fused clouds already carry camera-oriented per-measurement normals
        # (the correct orientation for ball-pivot meshing) — keep them. Only
        # clouds that lost their normals get PCA re-estimation.
        viewpoint = params.viewpoint
        if viewpoint is None and camera_centers:
            centers = np.asarray(camera_centers)
            viewpoint = tuple(float(v) for v in centers.mean(axis=0))
        cloud.with_normals(estimate_normals(cloud.xyz, k=params.normal_k, viewpoint=viewpoint))

    if progress:
        progress("confidence", 0.75, {"points": cloud.n})
    if cloud.confidence is None:
        cloud.confidence = np.ones(cloud.n, dtype=np.float64)
    # Re-score with observations + residual for a view-agreement-based value.
    cloud.confidence = dense_point_confidence(
        cloud.observations if cloud.observations is not None else np.ones(cloud.n, dtype=np.int32),
        cloud.residual,
        voxel_size=params.voxel_size,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info(
        "cloud_optimized",
        input_points=raw.n,
        output_points=cloud.n,
        time_ms=round(elapsed_ms, 2),
    )
    return OptimizeReport(cloud=cloud, filter_stats=filter_stats, elapsed_ms=elapsed_ms)
