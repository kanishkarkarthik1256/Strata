"""Quality analyzer — composes individual quality signals into a composite score.

Each sub-analyzer is a pure function on a BGR frame (np.ndarray).
The composite is a configurable weighted average, all scores normalized to 0.0–1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config.settings import settings
from app.services.blur_detector import blur_score
from app.services.duplicate_detector import is_consecutive_duplicate

#: Width of the downscaled proxy frame used for the two dominant quality
#: costs (dense optical flow + SSIM). Measured on 1920x1080 aerial frames:
#: Farneback 543 ms + SSIM 146 ms at full res vs 43 ms + 9 ms at 480 px,
#: with identical keep/reject decisions on sampled real pairs. Blur and
#: sharpness stay full-res: they are cheap (18 ms) and their thresholds
#: are calibrated at capture resolution (Laplacian variance is NOT
#: scale-invariant, so it must not move to the proxy).
QUALITY_PROXY_WIDTH = 480


def _proxy(frame: np.ndarray) -> np.ndarray:
    """Downscaled copy for expensive quality signals (motion, SSIM)."""
    h, w = frame.shape[:2]
    if w <= QUALITY_PROXY_WIDTH:
        return frame
    scale = QUALITY_PROXY_WIDTH / w
    return cv2.resize(frame, (QUALITY_PROXY_WIDTH, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


@dataclass
class FrameQuality:
    """All quality signals for a single frame."""
    blur: float = 0.0
    sharpness: float = 0.0
    exposure: float = 0.0
    motion: float = 0.0
    composite: float = 0.0
    rejection_reason: str = ""


def analyze_frame(
    frame: np.ndarray,
    prev_frame: np.ndarray | None = None,
) -> FrameQuality:
    """Run all quality checks on a single frame and return scores.

    All sub-scores are 0.0–1.0 where higher is better.
    """
    raw_blur = blur_score(frame)
    # Normalize blur: 0 = perfectly blurry, 1 = perfectly sharp
    # Typical range: 0–500+, threshold ~100
    blur_norm = min(raw_blur / max(settings.processing.blur_threshold * 3, 1), 1.0)

    sharpness = _sharpness_score(frame)
    exposure = _exposure_score(frame)
    # Motion + duplicate SSIM run on the downscaled proxy: they dominate the
    # extraction cost (87% measured) and their decisions are scale-stable.
    proxy = _proxy(frame)
    proxy_prev = _proxy(prev_frame) if prev_frame is not None else None
    motion = _motion_score(proxy, proxy_prev)
    # Flow magnitudes shrink with the image; rescale to the full-res pixel
    # units the /10 normalization was calibrated for.
    motion = min(motion * (frame.shape[1] / max(proxy.shape[1], 1)), 1.0)

    # Weighted composite
    composite = (
        0.30 * blur_norm +
        0.25 * sharpness +
        0.25 * exposure +
        0.20 * (1.0 - motion)  # lower motion = better
    )

    # Hard rejection checks
    reason = ""
    if raw_blur < settings.processing.blur_threshold:
        reason = "blurry"
    elif exposure < 0.1:
        reason = "overexposed" if _is_overexposed(frame) else "underexposed"
    elif is_consecutive_duplicate(proxy, proxy_prev, motion=motion):
        reason = "duplicate"

    return FrameQuality(
        blur=blur_norm,
        sharpness=sharpness,
        exposure=exposure,
        motion=motion,
        composite=round(composite, 4),
        rejection_reason=reason,
    )


def _sharpness_score(frame: np.ndarray) -> float:
    """Edge density via Canny — higher means more detail."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size
    return min(edge_ratio * 5, 1.0)  # normalize: ~20% edge pixels = 1.0


def _exposure_score(frame: np.ndarray) -> float:
    """0.0 = terrible exposure, 1.0 = ideal.

    Penalizes both overexposure and underexposure by measuring how much
    of the histogram falls in the extreme ends.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()

    # Fraction of pixels in extreme dark (<10) and extreme bright (>245)
    dark_frac = hist[:10].sum()
    bright_frac = hist[246:].sum()
    extreme_frac = dark_frac + bright_frac

    # Ideal: ~5% or less in extremes → score 1.0
    # Terrible: >50% in extremes → score 0.0
    return max(0.0, min(1.0, 1.0 - (extreme_frac - 0.05) / 0.45))


def _motion_score(frame: np.ndarray, prev_frame: np.ndarray | None) -> float:
    """0.0 = no motion, 1.0 = extreme motion.

    Uses dense optical flow magnitude as a proxy for motion blur.
    """
    if prev_frame is None:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    if gray.shape != prev_gray.shape:
        prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = float(mag.mean())

    # Normalize: mean flow > 10px = max motion
    return min(mean_mag / 10.0, 1.0)


def _is_overexposed(frame: np.ndarray) -> bool:
    """Check if frame is primarily overexposed vs underexposed."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) > 200
