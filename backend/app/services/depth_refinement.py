"""Depth map refinement — every filter is optional and independently enabled.

Filters (real OpenCV kernels, all operate on float32 depth in meters):
* ``median``  — cv2.medianBlur on a scaled integer image (removes speckle)
* ``bilateral`` — cv2.bilateralFilter (edge-aware denoising)
* ``edge_preserving`` — cv2.edgePreservingFilter (smooths flat areas,
  preserves depth discontinuities)
* ``hole_fill`` — masked median inpainting of zero/invalid pixels
* ``clamp`` — clip to [min_depth, max_depth]

Only pixels that were originally valid may produce new depth: holes are
filled from valid neighbours, never invented where no data exists nearby.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.logging_config import get_logger

log = get_logger("drone_recon.services.depth_refinement")


@dataclass
class RefineParams:
    median: bool = False
    median_k: int = 5
    bilateral: bool = False
    bilateral_d: int = 5
    bilateral_sigma: float = 0.1  # in meters (depth units)
    edge_preserving: bool = False
    hole_fill: bool = False
    hole_median_k: int = 9
    min_depth: float = 0.2
    max_depth: float = 200.0

    @classmethod
    def from_settings(cls) -> RefineParams:
        from app.config.settings import settings

        d = settings.dense
        return cls(
            median=d.refine_median,
            median_k=d.refine_median_k,
            bilateral=d.refine_bilateral,
            edge_preserving=d.refine_edge_preserving,
            hole_fill=d.refine_hole_fill,
            min_depth=d.min_depth_m,
            max_depth=d.max_depth_m,
        )

    def enabled(self) -> bool:
        return any([self.median, self.bilateral, self.edge_preserving, self.hole_fill])


def refine_depth(depth: np.ndarray, params: RefineParams) -> np.ndarray:
    """Apply the enabled filters to a float32 depth map (invalid = <=0).

    All filters run directly in float meters (integer scaling saturates for
    depths beyond ~25 cm); invalid pixels stay 0.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid0 = depth > 0
    if not params.enabled() or not valid0.any():
        return depth

    out = depth.copy()

    if params.median:
        k = params.median_k if params.median_k % 2 == 1 else params.median_k + 1
        out = _windowed_median(out, k)

    if params.bilateral:
        sigma_color = max(10.0, params.bilateral_sigma * 1000.0)
        out = cv2.bilateralFilter(out, params.bilateral_d, sigma_color, sigma_color)

    if params.edge_preserving:
        out = cv2.edgePreservingFilter(out, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.05)

    if params.hole_fill:
        kk = params.hole_median_k if params.hole_median_k % 2 == 1 else params.hole_median_k + 1
        out = _fill_holes(out, kk)
        # Only keep filled values where the original data had local support;
        # deep gaps (beyond the filter radius) stay empty.
        support = _local_support(valid0, kk)
        out[~valid0 & (support < 0.01)] = 0.0
    else:
        # Median/bilateral/edge filters must not invent depth in invalid pixels.
        out[~valid0] = 0.0

    # Clamp depth range without raising zeros to min_depth (zeros mean no data).
    positive = out > 0
    out[positive] = np.clip(out[positive], params.min_depth, params.max_depth)
    return out.astype(np.float32)


def _local_support(valid0: np.ndarray, k: int) -> np.ndarray:
    """Fraction of originally-valid pixels inside each k x k window."""
    k = k if k % 2 == 1 else k + 1
    return cv2.boxFilter(valid0.astype(np.float32), -1, (k, k), borderType=cv2.BORDER_REPLICATE)


def _windowed_median(depth: np.ndarray, k: int) -> np.ndarray:
    """Median filter on float32 depth for any odd kernel size."""
    if k == 1:
        return depth.copy()
    # cv2.medianBlur supports CV_32F for k <= 5; use it when possible.
    if k <= 5:
        return cv2.medianBlur(depth, k)
    from numpy.lib.stride_tricks import sliding_window_view

    pad = k // 2
    padded = np.pad(depth, pad, mode="edge")
    windows = sliding_window_view(padded, (k, k))
    return np.median(windows, axis=(-2, -1)).astype(np.float32)


def _fill_holes(depth: np.ndarray, k: int) -> np.ndarray:
    """Fill zero pixels from the median of valid neighbours (iterative)."""
    out = depth.copy()
    for _ in range(2):
        invalid = out <= 0
        if not invalid.any():
            break
        kk = k if k % 2 == 1 else k + 1
        med = _windowed_median(out, kk)
        fill = invalid & (med > 0)
        out[fill] = med[fill]
    return out
