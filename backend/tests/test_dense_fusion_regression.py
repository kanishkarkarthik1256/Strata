"""Regression tests pinning the dense-fusion defect found in run 1ae5b0.

History: the dense stage exported per-view concatenation as "fusion"
(obs=1 / residual=0 / constant confidence for every point), and the filter
chain's ROR radius (0.4 m default) was finer than the measured fusion voxel
(1.99 m), annihilating 97% of the cloud. These tests fail if either defect
is reintroduced.
"""

import numpy as np
import pytest

from app.services.depth_fusion import DepthView, FusionParams, fuse_depth_views
from app.services.pointcloud import PointCloud
from app.services.pointcloud_filter import radius_outlier_removal, voxel_downsample

_LOOK_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _make_K(size: int, focal: float = 60.0) -> np.ndarray:
    return np.array(
        [[focal, 0, (size - 1) / 2], [0, focal, (size - 1) / 2], [0, 0, 1]],
        dtype=np.float64,
    )


def _shifted_plane_views(n_views: int = 4, size: int = 48, z: float = 10.0):
    """Overlapping depth views of a z=0 plane from shifted nadir cameras.

    Views overlap 75% so every central pixel is seen by all cameras — the
    fused voxels must reflect that support.
    """
    K = _make_K(size)
    views = []
    for i in range(n_views):
        # 0.5 m offsets keep the same voxel grid aligned across views.
        t = np.array([0.25 * i, 0.25 * i, z], dtype=np.float64)
        depth = np.full((size, size), z, dtype=np.float64)
        rgb = np.full((size, size, 3), 128, dtype=np.uint8)
        views.append(DepthView(frame_id=f"v{i}", depth=depth, rgb=rgb, K=K, R=_LOOK_DOWN.copy(), t=t))
    return views


class TestFusionHonesty:
    def test_fused_support_reflects_overlap(self):
        """Fusion cannot silently export obs=1 everywhere.

        Four 75%-overlapping views of one plane: every central voxel is
        observed 4x, so obs must exceed 1 over a substantial fraction and
        residual must be non-zero where views disagree by half a pixel.
        """
        views = _shifted_plane_views()
        cloud = fuse_depth_views(views, FusionParams(voxel_size=0.5))
        obs = cloud.observations
        assert obs is not None and obs.max() >= 2
        frac_multi = float((obs > 1).mean())
        assert frac_multi > 0.2, f"only {frac_multi:.2%} of voxels have multi-view support"
        assert cloud.residual is not None and float(np.median(cloud.residual)) > 0

    def test_voxel_downsample_preserves_support(self):
        """Re-voxelizing a fused cloud must not reset obs to 1 (the
        merge-of-merges trap that produced constant-confidence exports)."""
        views = _shifted_plane_views()
        cloud = fuse_depth_views(views, FusionParams(voxel_size=0.5))
        down = voxel_downsample(cloud, voxel_size=0.5)
        assert down.observations is not None
        assert down.observations.max() >= 2
        assert float(np.median(down.observations)) == pytest.approx(float(np.median(cloud.observations)), rel=0.5)

    def test_ror_radius_cannot_annihilate_cloud(self):
        """A neighbourhood radius finer than the point spacing must not
        delete the bulk of a fused cloud (the 0.4 m-on-1.99 m-voxel bug:
        97% of points removed). Guard: the coupled radius used by the
        dense stage keeps >=80% of a legitimately fused cloud."""
        views = _shifted_plane_views()
        cloud = fuse_depth_views(views, FusionParams(voxel_size=0.5))
        coupled = max(0.4, 2.0 * 0.5)  # same formula as the dense stage
        cleaned, removed = radius_outlier_removal(cloud, radius=coupled, min_neighbors=5)
        assert cleaned.n >= 0.8 * cloud.n, f"coupled ROR removed {removed}/{cloud.n}"

        # Sanity that ROR still works as a filter: the sub-voxel radius
        # (the historical bug) genuinely does devastate this cloud.
        _, removed_bad = radius_outlier_removal(cloud, radius=0.4 * 0.4, min_neighbors=5)
        assert removed_bad > 0.5 * cloud.n

    def test_fusion_params_cannot_match_concatenation(self):
        """Concatenation signature check: fused output must be strictly
        coarser than the sum of raw pixels and carry aggregated support."""
        views = _shifted_plane_views()
        cloud = fuse_depth_views(views, FusionParams(voxel_size=0.5))
        raw = sum(int((v.depth > 0).sum()) for v in views)
        assert cloud.n < raw
        obs = cloud.observations
        assert obs is not None and (obs > 1).any()
        # Constant confidence is the concatenation fingerprint.
        assert float(np.std(cloud.confidence)) > 0


# ---------------------------------------------------------------------------
# Surface separation (Part 3): incompatible layers must never be averaged.
# ---------------------------------------------------------------------------

from app.services.depth_fusion import voxel_merge  # noqa: E402


class TestSurfaceSeparation:
    def test_crossing_layers_not_averaged(self):
        """Terrain + wall measurements sharing one voxel fuse into TWO points
        with per-layer normals — never one phantom point with a blended
        normal."""
        rng = np.random.default_rng(3)
        pa = np.column_stack([rng.uniform(0, 1, 12), rng.uniform(0, 1, 12), 0.1 + rng.normal(0, 0.02, 12)])
        na = np.tile([0.0, 0.0, 1.0], (12, 1)) + rng.normal(0, 0.05, (12, 3))
        pb = np.column_stack([0.9 + rng.normal(0, 0.02, 8), rng.uniform(0, 1, 8), rng.uniform(0, 1, 8)])
        nb = np.tile([1.0, 0.0, 0.0], (8, 1)) + rng.normal(0, 0.05, (8, 3))
        cloud = voxel_merge(np.vstack([pa, pb]), None,
                            np.concatenate([np.full(12, 0.9), np.full(8, 0.8)]),
                            1.0, normals=np.vstack([na, nb]))
        assert cloud.n == 2, f"layers must stay separate, got {cloud.n} points"
        assert cloud.observations.tolist() == [12, 8], "per-layer support must be preserved"
        fn = cloud.normals[np.argmin(cloud.xyz[:, 2])]
        wn = cloud.normals[np.argmax(cloud.xyz[:, 0])]
        assert fn[2] > 0.9 and wn[0] > 0.9, "per-layer normals must survive the merge"

    def test_single_surface_never_splits(self):
        """One smooth plane across many voxels: zero splits, consensus
        normals — separation must not fragment legitimate surfaces."""
        rng = np.random.default_rng(5)
        p = np.column_stack([rng.uniform(0, 5, 60), rng.uniform(0, 1, 60), 0.1 + rng.normal(0, 0.02, 60)])
        n = np.tile([0.0, 0.0, 1.0], (60, 1)) + rng.normal(0, 0.05, (60, 3))
        cloud = voxel_merge(p, None, np.full(60, 0.9), 1.0, normals=n)
        assert cloud.n == 5
        assert cloud.meta["surface_split_measurements"] == 0
        assert np.allclose(cloud.normals[:, 2], 1.0, atol=0.05)

    def test_separation_deterministic(self):
        """Same input twice → identical output (mandate: deterministic)."""
        rng = np.random.default_rng(3)
        pa = np.column_stack([rng.uniform(0, 1, 12), rng.uniform(0, 1, 12), np.full(12, 0.1)])
        na = np.tile([0.0, 0.0, 1.0], (12, 1))
        pb = np.column_stack([np.full(8, 0.9), rng.uniform(0, 1, 8), rng.uniform(0, 1, 8)])
        nb = np.tile([1.0, 0.0, 0.0], (8, 1))
        args = (np.vstack([pa, pb]), None,
                np.concatenate([np.full(12, 0.9), np.full(8, 0.8)]), 1.0)
        c1 = voxel_merge(*args, normals=np.vstack([na, nb]))
        c2 = voxel_merge(*args, normals=np.vstack([na, nb]))
        assert np.array_equal(c1.xyz, c2.xyz) and np.array_equal(c1.observations, c2.observations)
