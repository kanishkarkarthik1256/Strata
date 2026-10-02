"""Dense-score metric definitions — regression pins.

Two definitions changed after the measured diagnosis on airport_53_8f0e90
(65.82 "Fair" while every artifact was healthy):

1. Coverage denominator = convex hull of occupied footprint cells, not the
   XY bounding box. An L-shaped site has a nearly empty bbox perimeter
   (edge occupancy 1-2%), so the bbox reported the site's *shape* as missing
   coverage (61.8% bbox vs 91.3% hull on the real cloud).
2. The noise component counts filter-judged *outliers only* (SOR + ROR +
   confidence floor). Voxel-merge dedup is a many-to-one average of agreeing
   measurements of one surface cell — counting it made the component track
   scene size instead of data quality (43.17% "noise" of which 41.7 pp was
   dedup, 1.5 pp real outliers).
"""

from __future__ import annotations

import numpy as np

from app.services.point_statistics import (
    _footprint_denominator,
    analyze_dense_quality,
    footprint_cells,
)
from app.services.pointcloud import PointCloud
from app.services.pointcloud_filter import filter_cloud


def _cloud(xyz: np.ndarray, **extra) -> PointCloud:
    return PointCloud(xyz=xyz, **extra)


class TestHullCoverage:
    def test_l_shape_scores_high_not_punished_by_bbox(self):
        """L-shaped footprint: bbox denominator ~2x the occupied cells; the
        hull matches the footprint, so coverage must stay high."""
        v = 1.0
        xs, ys = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
        cells = np.stack([xs.ravel(), ys.ravel()], axis=1)
        # Drop the upper-right quadrant → L shape.
        keep = ~((cells[:, 0] >= 10) & (cells[:, 1] >= 10))
        cells = cells[keep]
        xyz = np.c_[cells * v + v / 2, np.zeros(len(cells))]
        q = analyze_dense_quality(_cloud(xyz), voxel_size=v)
        # Hull denominator = footprint area (400 - 100 = 300 cells) → ~100%.
        assert q.coverage_percent > 90.0

    def test_bbox_gap_still_counts_as_gap(self):
        """Two separated blobs sharing a y-band: the convex hull of occupied
        centers IS the full rectangle (the gap is inside the hull), so the
        gap is honestly counted — but the hull denominator can never exceed
        the bbox one, and interior holes still read as missing coverage."""
        v = 1.0
        blob_a = np.stack(np.meshgrid(np.arange(10), np.arange(10), indexing="ij"), -1).reshape(-1, 2)
        blob_b = np.stack(np.meshgrid(np.arange(40, 50), np.arange(10), indexing="ij"), -1).reshape(-1, 2)
        cells = np.vstack([blob_a, blob_b])
        xyz = np.c_[cells * v + v / 2, np.zeros(len(cells))]
        q = analyze_dense_quality(_cloud(xyz), voxel_size=v)
        # 200 occupied / ~440 hull cells (centers-span triangle ≈ 0.5·45·11 m²).
        assert 35.0 < q.coverage_percent < 50.0  # measured: 45.35
        # Hull never scores higher than the bbox would on the same cloud:
        hull_cells = _footprint_denominator(cells, v)
        bbox_cells = (cells[:, 0].max() - cells[:, 0].min() + 1) * (cells[:, 1].max() - cells[:, 1].min() + 1)
        assert hull_cells <= bbox_cells

    def test_full_grid_is_perfect(self):
        v = 1.0
        xs, ys = np.meshgrid(np.arange(15), np.arange(15), indexing="ij")
        cells = np.stack([xs.ravel(), ys.ravel()], axis=1)
        xyz = np.c_[cells * v + v / 2, np.zeros(len(cells))]
        q = analyze_dense_quality(_cloud(xyz), voxel_size=v)
        assert q.coverage_percent > 99.0

    def test_degenerate_line_falls_back_to_bbox(self):
        """A single row of cells has no 2D hull — bbox denominator, coverage 1."""
        v = 1.0
        cells = np.stack([np.arange(20), np.zeros(20)], axis=1)
        xyz = np.c_[cells * v + v / 2, np.zeros(len(cells))]
        q = analyze_dense_quality(_cloud(xyz), voxel_size=v)
        assert q.coverage_percent == 100.0

    def test_footprint_denominator_units(self):
        """Denominator is in cells: hull area / cell area, floored at the
        occupied count; degenerate inputs return honest small values."""
        v = 0.5
        corners = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])  # square footprint
        # Centers span 1.0 m x 1.0 m → area 1.0 m² → 4 cells at v=0.5.
        assert _footprint_denominator(corners, v) == 4
        # Diagonal line: no 2D hull (QhullError) → bbox fallback.
        diag = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        assert _footprint_denominator(diag, v) == 4 * 4
        assert _footprint_denominator(np.array([[0, 0]]), v) == 1


class TestOutlierOnlyNoise:
    def test_dedup_is_not_noise(self):
        """Dense duplicate measurements of one surface: voxel merge collapses
        them (huge raw→filtered delta), SOR/ROR remove nothing — the noise
        component must reflect the filters, not the dedup."""
        rng = np.random.default_rng(7)
        # 400k measurements packed into a 10x10 m plane (spacing ~2.5 cm).
        xyz = np.c_[rng.uniform(0, 10, 400_000), rng.uniform(0, 10, 400_000), np.zeros(400_000)]
        raw = _cloud(xyz)
        cleaned, stats = filter_cloud(raw, voxel_size=0.05, sor_k=20, sor_std_ratio=2.0,
                                      ror_radius=0.4, ror_min_neighbors=5, min_confidence=0.05)
        dedup_removed = raw.n - cleaned.n
        outlier_removed = stats["sor_removed"] + stats["ror_removed"] + stats["low_conf_removed"]
        assert dedup_removed > 300_000          # dedup dominates
        assert outlier_removed < dedup_removed * 0.05  # filters barely fire on clean geometry

        noise_pct = 100.0 * outlier_removed / raw.n
        q = analyze_dense_quality(cleaned, voxel_size=0.05, noise_percent=noise_pct)
        assert q.component_scores["noise"] > 0.95

    def test_real_outliers_still_count(self):
        """Genuine isolated noise must still be judged: add far-flung junk
        points and the noise component must drop measurably... via the
        percentage the caller feeds in (SOR/ROR findings)."""
        rng = np.random.default_rng(11)
        surface = np.c_[rng.uniform(0, 10, 50_000), rng.uniform(0, 10, 50_000), np.zeros(50_000)]
        junk = np.c_[rng.uniform(0, 10, 2_500), rng.uniform(0, 10, 2_500), rng.uniform(30, 40, 2_500)]
        xyz = np.vstack([surface, junk])
        raw = _cloud(xyz)
        cleaned, stats = filter_cloud(raw, voxel_size=0.05, sor_k=20, sor_std_ratio=2.0,
                                      ror_radius=0.4, ror_min_neighbors=5, min_confidence=0.05)
        outlier_removed = stats["sor_removed"] + stats["ror_removed"] + stats["low_conf_removed"]
        assert outlier_removed > 1_000          # junk got caught
        noise_pct = 100.0 * outlier_removed / raw.n
        q = analyze_dense_quality(cleaned, voxel_size=0.05, noise_percent=noise_pct)
        assert q.component_scores["noise"] < 0.98  # noisy data scores below clean

    def test_old_definition_would_have_scored_dedup_as_noise(self):
        """Pin the behavioral delta: with dedup excluded, a heavily-deduped
        clean cloud's noise component is ~1, whereas raw-removed percentage
        (the old input) would have dragged it near zero."""
        rng = np.random.default_rng(3)
        xyz = np.c_[rng.uniform(0, 10, 200_000), rng.uniform(0, 10, 200_000), np.zeros(200_000)]
        raw = _cloud(xyz)
        cleaned, stats = filter_cloud(raw, voxel_size=0.05, sor_k=20, sor_std_ratio=2.0,
                                      ror_radius=0.4, ror_min_neighbors=5, min_confidence=0.05)
        old_removed_pct = 100.0 * (raw.n - cleaned.n) / raw.n
        outlier_removed = stats["sor_removed"] + stats["ror_removed"] + stats["low_conf_removed"]
        new_noise_pct = 100.0 * outlier_removed / raw.n
        assert old_removed_pct > 80.0
        assert 1 - new_noise_pct / 100 > 0.95 >= 1 - old_removed_pct / 100 - 0.001


class TestOptimizerSkipsReMerge:
    def test_fused_cloud_keeps_sibling_surfaces(self):
        """optimize_cloud must not re-voxel-merge a cloud that already went
        through voxel_merge (fusion provenance = observations): the second
        merge collapsed intentional sibling surfaces (measured 59,958 →
        39,368 on a 12-view probe) and inflated residual +17%."""
        rng = np.random.default_rng(5)
        # Two parallel sheets 0.3 m apart inside the same XY footprint —
        # voxel_merge keeps them as siblings; a blind re-merge averages the
        # bands into phantom mid-points.
        n = 40_000
        sheet_a = np.c_[rng.uniform(0, 20, n), rng.uniform(0, 20, n), np.full(n, 0.0)]
        sheet_b = np.c_[rng.uniform(0, 20, n), rng.uniform(0, 20, n), np.full(n, 0.3)]
        xyz = np.vstack([sheet_a, sheet_b])
        fused = PointCloud(xyz=xyz, observations=np.full(2 * n, 5, dtype=np.int32),
                           residual=np.full(2 * n, 0.01))
        from app.services.pointcloud_optimizer import OptimizeParams, optimize_cloud
        report = optimize_cloud(fused, OptimizeParams(voxel_size=0.05))
        assert report.filter_stats.get("voxel_stage") == "skipped_fusion_provenance"
        # Sheets preserved: both z-levels still present, no phantom mid-band.
        zs = report.cloud.xyz[:, 2]
        assert float(np.abs(zs).min()) < 0.02 or float(np.abs(zs - 0.3).min()) < 0.02
        kept_a = int((np.abs(zs) < 0.05).sum())
        kept_b = int((np.abs(zs - 0.3) < 0.05).sum())
        assert kept_a > 30_000 and kept_b > 30_000

    def test_raw_cloud_still_gets_voxel_merge(self):
        """Unmerged (raw measurement) clouds keep the voxel-merge stage."""
        rng = np.random.default_rng(6)
        xyz = np.c_[rng.uniform(0, 5, 30_000), rng.uniform(0, 5, 30_000), np.zeros(30_000)]
        raw = PointCloud(xyz=xyz)  # no observations → not yet merged
        from app.services.pointcloud_optimizer import OptimizeParams, optimize_cloud
        report = optimize_cloud(raw, OptimizeParams(voxel_size=0.05))
        assert "after_voxel" in report.filter_stats  # voxel stage ran
        assert report.cloud.n < raw.n                 # and deduped
