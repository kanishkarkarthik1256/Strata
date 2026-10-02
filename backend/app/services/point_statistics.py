"""Dense reconstruction quality statistics.

Computes a defensible metric set for an aerial dense cloud:

* **mean spacing** — mean distance to nearest neighbour (kd-tree)
* **coverage %** — 2.5D footprint: share of occupied XY cells inside the
  convex hull of the occupied footprint at ``voxel_size`` resolution
  (bounding box only as a degenerate-shape fallback)
* **occlusion %** — share of footprint cells whose point count falls below
  half the median column count: cells that *should* be covered but are
  under-observed are treated as occluded / missing surface
* **noise %** — share of measurements removed by filtering (fed in from the
  filter stage)
* **dense score (0-100)** — weighted combination of coverage, mean
  confidence, spacing quality, residual quality, and noise penalty

Also exposes :func:`footprint_cells` (used by the digital twin for weak
region detection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from app.logging_config import get_logger
from app.services.pointcloud import PointCloud

log = get_logger("drone_recon.services.point_statistics")

_GRADES = [(85, "Excellent"), (70, "Good"), (50, "Fair"), (0, "Poor")]


@dataclass
class DenseQuality:
    """Full quality report for a dense cloud."""

    point_count: int = 0
    mean_spacing: float = 0.0
    mean_confidence: float = 0.0
    coverage_percent: float = 0.0
    occlusion_percent: float = 0.0
    noise_percent: float = 0.0
    sparse_cell_count: int = 0
    dense_score: float = 0.0
    grade: str = "Poor"
    component_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "point_count": self.point_count,
            "mean_spacing": round(self.mean_spacing, 6),
            "mean_confidence": round(self.mean_confidence, 4),
            "coverage_percent": round(self.coverage_percent, 2),
            "occlusion_percent": round(self.occlusion_percent, 2),
            "noise_percent": round(self.noise_percent, 2),
            "sparse_cell_count": self.sparse_cell_count,
            "dense_score": round(self.dense_score, 2),
            "grade": self.grade,
            "components": {k: round(v, 4) for k, v in self.component_scores.items()},
        }


def footprint_cells(
    cloud: PointCloud,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise the cloud to 2.5D XY columns.

    Returns (cell_ids (N, 2) int, cell_centers (C, 2) float, counts (C,) int)
    with one row per *unique* occupied cell.
    """
    q = np.floor(cloud.xyz[:, :2] / voxel_size).astype(np.int64)
    # Exact cell identity: unique over the (i, j) pair itself. A scalar
    # packing like ``i * BIG + j`` is NOT a bijection for signed indices
    # (floor-div/mod do not round-trip negatives), which aliased every
    # y-negative cell and inflated the footprint denominator by orders of
    # magnitude — reporting a false 0.02% coverage. Keep the pair.
    uniq_q, inverse, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    centers = (uniq_q + 0.5) * voxel_size
    return uniq_q, centers, counts


def _footprint_denominator(cells: np.ndarray, voxel_size: float) -> int:
    """Grid-cell count of the footprint the cloud occupies.

    Convex hull of the occupied cell centers (area / cell area), falling back
    to the occupied bounding box when the hull is undefined (fewer than 3
    non-collinear cells — a line or single row has no 2D footprint interior
    and the bbox is then the honest denominator).
    """
    if len(cells) < 3:
        return max(1, len(cells))
    centers = (cells + 0.5) * voxel_size
    try:
        hull_area = float(ConvexHull(centers).volume)  # 2D: volume == area
    except QhullError:
        nx = int(cells[:, 0].max() - cells[:, 0].min() + 1)
        ny = int(cells[:, 1].max() - cells[:, 1].min() + 1)
        return max(1, nx * ny)
    return max(len(cells), int(round(hull_area / (voxel_size * voxel_size))))


def analyze_dense_quality(
    cloud: PointCloud,
    voxel_size: float,
    *,
    noise_percent: float = 0.0,
    residual_scale: float | None = None,
) -> DenseQuality:
    """Analyze the density/coverage quality of a dense cloud."""
    n = cloud.n
    if n == 0:
        return DenseQuality()

    tree = cKDTree(cloud.xyz)
    dist, _ = tree.query(cloud.xyz, k=2)
    spacing = dist[:, 1]
    mean_spacing = float(spacing.mean())

    # 2.5D footprint coverage: occupied cells over the *footprint-shaped*
    # denominator — the convex hull of the occupied column centers — not the
    # bounding box. An L-shaped or diagonal site has an almost-empty bbox
    # perimeter (measured: edge occupancy 1-2%, only 145 interior holes), so
    # the bbox denominator reported the site's *shape* as missing coverage.
    # Degenerate footprints (collinear/single-cell, where no hull exists)
    # fall back to the bounding-box denominator.
    cells, _, counts = footprint_cells(cloud, voxel_size)
    footprint_cells_total = _footprint_denominator(cells, voxel_size)
    coverage = 100.0 * len(counts) / max(1, footprint_cells_total)

    median_count = float(np.median(counts))
    threshold = max(1.0, 0.5 * median_count)
    sparse = counts[counts < threshold]
    occlusion = 100.0 * len(sparse) / max(1, len(counts))

    mean_conf = float(cloud.confidence.mean()) if cloud.confidence is not None else 1.0

    residual_scale = residual_scale or voxel_size
    residual = cloud.residual if cloud.residual is not None else np.zeros(n)
    residual_score = float(np.clip(1.0 - residual.mean() / max(residual_scale, 1e-9), 0.0, 1.0))

    spacing_quality = float(np.clip(1.0 - mean_spacing / max(2.0 * voxel_size, 1e-9), 0.0, 1.0))
    coverage_score = float(np.clip(coverage / 100.0, 0.0, 1.0))
    noise_score = float(np.clip(1.0 - noise_percent / 100.0, 0.0, 1.0))

    components = {
        "coverage": coverage_score,
        "mean_confidence": float(np.clip(mean_conf, 0.0, 1.0)),
        "spacing": spacing_quality,
        "residual": residual_score,
        "noise": noise_score,
    }
    score = 100.0 * (
        0.30 * components["coverage"]
        + 0.25 * components["mean_confidence"]
        + 0.15 * components["spacing"]
        + 0.15 * components["residual"]
        + 0.15 * components["noise"]
    )

    grade = next(g for cutoff, g in _GRADES if score >= cutoff)
    result = DenseQuality(
        point_count=n,
        mean_spacing=mean_spacing,
        mean_confidence=mean_conf,
        coverage_percent=coverage,
        occlusion_percent=occlusion,
        noise_percent=noise_percent,
        sparse_cell_count=int(len(sparse)),
        dense_score=float(score),
        grade=grade,
        component_scores=components,
    )
    log.info(
        "dense_quality_analyzed",
        points=n,
        score=round(result.dense_score, 2),
        grade=result.grade,
        coverage=round(result.coverage_percent, 2),
        occlusion=round(result.occlusion_percent, 2),
    )
    return result


def weak_regions(
    cloud: PointCloud,
    voxel_size: float,
    max_regions: int = 10,
) -> list[dict]:
    """Centroids of the most under-covered footprint columns.

    These are candidate weak / occluded regions for mission intelligence.
    """
    _, centers, counts = footprint_cells(cloud, voxel_size)
    if len(counts) < 8:
        return []
    threshold = max(1.0, 0.5 * float(np.median(counts)))
    weak = counts < threshold
    if not weak.any():
        return []
    order = np.argsort(counts[weak])[:max_regions]
    return [
        {
            "x": round(float(centers[weak][i, 0]), 4),
            "y": round(float(centers[weak][i, 1]), 4),
            "cell_count": int(counts[weak][i]),
        }
        for i in order
    ]
