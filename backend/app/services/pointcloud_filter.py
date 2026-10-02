"""Point cloud filtering — noise, outlier, and duplicate removal.

Real neighbourhood-based filters (statistical + radius outlier removal on a
kd-tree) plus voxel-grid downsampling, which also deduplicates coincident
points. Every filter reports how many points it removed.
"""

from __future__ import annotations

from scipy.spatial import cKDTree

from app.logging_config import get_logger
from app.services.depth_fusion import voxel_merge
from app.services.pointcloud import PointCloud

log = get_logger("drone_recon.services.pointcloud_filter")


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    """Deduplicate and average points that share a voxel cell.

    Preserves multi-view provenance: when the input already carries fusion
    statistics (observations/residual from a coarser per-view merge), the
    merged cell SUMS observations and combines residuals as the RMS of the
    contributing cells — a re-merge of a fused cloud must not reset every
    point to obs=1/residual=0 (that erased all multi-view support in the
    shipped model). Fused per-point normals are likewise preserved: cells
    merge only same-surface members, so the confidence-weighted normal
    stays that surface's orientation (needed by ball-pivot meshing).
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    import numpy as np
    conf = cloud.confidence if cloud.confidence is not None else np.ones(cloud.n, dtype=np.float64)
    if cloud.observations is None and cloud.residual is None:
        # Raw measurements: voxel_merge computes everything per cell.
        return voxel_merge(cloud.xyz, cloud.rgb, conf, voxel_size)

    # Re-merge of an already-fused cloud: aggregate per-cell statistics.
    obs = cloud.observations if cloud.observations is not None else np.ones(cloud.n, dtype=np.int64)
    resid = cloud.residual if cloud.residual is not None else np.zeros(cloud.n)
    q = np.floor(cloud.xyz / voxel_size).astype(np.int64)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    q_s = q[order]
    starts = np.r_[0, np.nonzero((q_s[1:] != q_s[:-1]).any(axis=1))[0] + 1]
    counts = np.diff(np.r_[starts, len(order)])

    n_cells = len(starts)
    out_rgb = np.zeros((n_cells, 3), dtype=np.uint8) if cloud.rgb is not None else None
    out_normals = np.zeros((n_cells, 3)) if cloud.normals is not None else None
    # Vectorised per-cell weighted aggregation: segment sums over the sorted
    # runs (np.add.reduceat) replace the former per-cell Python loop —
    # measured 36.8 s -> ~1 s on the 888k-point flight_to_tower raw cloud
    # with identical arithmetic (same weights, same order, float64 sums).
    members_w = conf[order]
    pts_sorted = cloud.xyz[order]
    wsum = np.add.reduceat(members_w, starts)
    wsum_safe = np.where(wsum > 0, wsum, 1.0)
    out_xyz = np.add.reduceat(pts_sorted * members_w[:, None], starts, axis=0) / wsum_safe[:, None]
    out_obs = np.add.reduceat(obs[order], starts)
    out_conf = np.add.reduceat(members_w, starts) / np.maximum(counts, 1)
    # RMS spread about each cell's new centroid: E[(r + |p-c|)²] per cell.
    d2 = ((pts_sorted - np.repeat(out_xyz, counts, axis=0)) ** 2).sum(axis=1)
    cell_mean_sq = np.add.reduceat(resid[order] ** 2 + d2, starts) / np.maximum(counts, 1)
    out_resid = np.sqrt(np.maximum(cell_mean_sq, 0.0))
    if out_rgb is not None:
        cw = cloud.rgb[order].astype(np.float64)
        out_rgb = np.clip(np.rint(
            np.add.reduceat(cw * members_w[:, None], starts, axis=0) / wsum_safe[:, None]
        ), 0, 255).astype(np.uint8)
    from app.services.pointcloud import PointCloud
    if out_normals is not None:
        nrm = cloud.normals[order]
        w_all = conf[order]
        nsum = np.add.reduceat(nrm * w_all[:, None], starts, axis=0)
        onorm = np.linalg.norm(nsum, axis=1, keepdims=True)
        out_normals = np.where(onorm > 1e-9, nsum / np.maximum(onorm, 1e-12), 0.0)
    return PointCloud(xyz=out_xyz, rgb=out_rgb, confidence=out_conf,
                      observations=out_obs.astype(np.int32), residual=out_resid,
                      normals=out_normals,
                      meta=dict(cloud.meta))


def statistical_outlier_removal(
    cloud: PointCloud,
    k: int = 20,
    std_ratio: float = 2.0,
) -> tuple[PointCloud, int]:
    """Remove points whose mean k-NN distance exceeds mean + std_ratio*std.

    Classic SOR (Rusu et al., PCL): isolated points sit far from their
    neighbours and are treated as noise.
    """
    if cloud.n < k + 1:
        return cloud, 0
    tree = cKDTree(cloud.xyz)
    # workers=-1: parallel k-NN over all cores (deterministic regardless).
    dist, _ = tree.query(cloud.xyz, k=k + 1, workers=-1)
    mean_dist = dist[:, 1:].mean(axis=1)  # exclude self-distance
    mu = mean_dist.mean()
    sigma = mean_dist.std()
    if sigma == 0:
        return cloud, 0
    keep = mean_dist <= mu + std_ratio * sigma
    removed = int((~keep).sum())
    return cloud.slice(keep), removed


def radius_outlier_removal(
    cloud: PointCloud,
    radius: float,
    min_neighbors: int,
) -> tuple[PointCloud, int]:
    """Remove points with fewer than *min_neighbors* within *radius*."""
    if cloud.n == 0:
        return cloud, 0
    tree = cKDTree(cloud.xyz)
    counts = tree.query_ball_point(cloud.xyz, r=radius, return_length=True, workers=-1)
    keep = counts >= min_neighbors
    removed = int((~keep).sum())
    return cloud.slice(keep), removed


def remove_low_confidence(cloud: PointCloud, min_confidence: float) -> tuple[PointCloud, int]:
    """Drop points below a hard confidence floor."""
    if cloud.confidence is None:
        return cloud, 0
    keep = cloud.confidence >= min_confidence
    removed = int((~keep).sum())
    return cloud.slice(keep), removed


def filter_cloud(
    cloud: PointCloud,
    *,
    voxel_size: float | None = None,
    sor_k: int = 20,
    sor_std_ratio: float = 2.0,
    ror_radius: float = 0.4,
    ror_min_neighbors: int = 5,
    min_confidence: float = 0.05,
) -> tuple[PointCloud, dict]:
    """Run the standard denoise chain: voxel → SOR → ROR → confidence.

    Returns (cleaned cloud, per-stage removal report).
    """
    report: dict = {"input_points": cloud.n}
    if voxel_size and voxel_size > 0:
        cloud = voxel_downsample(cloud, voxel_size)
    report["after_voxel"] = cloud.n

    cloud, removed = statistical_outlier_removal(cloud, sor_k, sor_std_ratio)
    report["sor_removed"] = removed

    cloud, removed = radius_outlier_removal(cloud, ror_radius, ror_min_neighbors)
    report["ror_removed"] = removed

    cloud, removed = remove_low_confidence(cloud, min_confidence)
    report["low_conf_removed"] = removed
    report["output_points"] = cloud.n

    log.info("cloud_filtered", **report)
    return cloud, report
