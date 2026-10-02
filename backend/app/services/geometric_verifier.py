"""Geometric verification — RANSAC essential and fundamental matrix estimation.

Rejects false correspondences, computes inlier ratios and reprojection errors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.logging_config import get_logger
from app.services.feature_matcher import MatchResult

log = get_logger("drone_recon.services.geometric_verifier")


@dataclass
class VerificationResult:
    """Result of geometric verification on a match."""
    match: MatchResult
    inlier_mask: np.ndarray  # (M,) bool
    num_inliers: int
    inlier_ratio: float
    fundamental_matrix: np.ndarray | None = None  # 3x3
    essential_matrix: np.ndarray | None = None  # 3x3
    reprojection_error: float = 0.0
    verification_time_ms: float = 0.0
    passed: bool = False


def verify_matches(
    match: MatchResult,
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    image_shape: tuple[int, int] | None = None,
    ransac_threshold: float = 3.0,
    min_inlier_ratio: float = 0.3,
) -> VerificationResult:
    """Verify a match set using RANSAC fundamental matrix estimation.

    Returns VerificationResult with inlier mask and quality metrics.
    """
    start = time.perf_counter()

    if len(match.matches) < 8:
        return VerificationResult(
            match=match,
            inlier_mask=np.zeros(len(match.matches), dtype=bool),
            num_inliers=0,
            inlier_ratio=0.0,
            passed=False,
        )

    # Extract matched keypoint coordinates
    pts_a = kpts_a[match.matches[:, 0]]
    pts_b = kpts_b[match.matches[:, 1]]

    # Fundamental matrix with RANSAC
    try:
        F, inlier_mask_cv = cv2.findFundamentalMat(
            pts_a.astype(np.float64),
            pts_b.astype(np.float64),
            cv2.FM_RANSAC,
            ransac_threshold,
            0.99,
        )
    except AttributeError:
        # Headless OpenCV may not have findFundamentalMat
        F, inlier_mask_cv = None, None

    if inlier_mask_cv is None:
        inlier_mask = np.zeros(len(match.matches), dtype=bool)
    else:
        inlier_mask = inlier_mask_cv.ravel().astype(bool)

    num_inliers = int(inlier_mask.sum())
    inlier_ratio = num_inliers / len(match.matches) if len(match.matches) > 0 else 0.0

    # Compute reprojection error for inliers
    reproj_error = 0.0
    if num_inliers > 0 and F is not None:
        reproj_error = _compute_reprojection_error(pts_a[inlier_mask], pts_b[inlier_mask], F)

    elapsed = (time.perf_counter() - start) * 1000
    passed = inlier_ratio >= min_inlier_ratio and num_inliers >= 10

    return VerificationResult(
        match=match,
        inlier_mask=inlier_mask,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        fundamental_matrix=F,
        reprojection_error=round(reproj_error, 4),
        verification_time_ms=round(elapsed, 2),
        passed=passed,
    )


def _compute_reprojection_error(pts_a: np.ndarray, pts_b: np.ndarray, F: np.ndarray) -> float:
    """Symmetric epipolar distance as reprojection error proxy."""
    ones_a = np.hstack([pts_a, np.ones((len(pts_a), 1))])
    ones_b = np.hstack([pts_b, np.ones((len(pts_b), 1))])

    # l = F * p'
    lines = (F @ ones_b.T).T  # (N, 3)
    d1 = np.abs(np.sum(ones_a * lines, axis=1))
    d1_norm = d1 / np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2 + 1e-10)

    # l' = F^T * p
    lines_t = (F.T @ ones_a.T).T
    d2 = np.abs(np.sum(ones_b * lines_t, axis=1))
    d2_norm = d2 / np.sqrt(lines_t[:, 0] ** 2 + lines_t[:, 1] ** 2 + 1e-10)

    return float(np.mean((d1_norm + d2_norm) / 2))


def verify_batch(
    verified: list[VerificationResult],
    min_inlier_ratio: float = 0.3,
) -> list[VerificationResult]:
    """Filter a batch of verification results, keeping only those that passed."""
    passed = [v for v in verified if v.passed and v.inlier_ratio >= min_inlier_ratio]
    log.info(
        "batch_verification",
        total=len(verified),
        passed=len(passed),
    )
    return passed
