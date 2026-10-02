"""Duplicate frame detection.

Two strategies:
1. SSIM against the previous frame — catches near-identical consecutive frames.
2. Perceptual hash against recently kept frames — catches near-static redundancy.

Every duplicate verdict is vetoable by MEASURED inter-frame change: a frame
whose optical flow against its comparison frame exceeds the pipeline's motion
threshold is moving (parallax is accumulating), whatever the cheap signal
says. Without the veto, high-fps or slow-drift footage is eaten alive —
London (59.94 fps): 149/150 candidates SSIM-rejected at 0.167 s spacing;
sunset: 22/32 pHash-rejected while flow showed real motion.

Pure functions — no side effects, no DB access.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config.settings import settings

#: Motion-score floor above which a frame is MOVING, not redundant.
#: Measured basis: London's consecutive 59.94 fps candidates carry ≈ 1.4 px
#: mean flow (motion 0.14) — real parallax SfM matches easily — while a
#: genuinely static pair measures < 0.3 px. The pipeline's motion_threshold
#: (0.4, calibrated for motion-blur risk) sits ABOVE both London's 0.14 and
#: sunset's dup-rejected p50 of 0.39, so it cannot serve as the duplicate
#: veto: the floor must separate "cheap hash/SSIM says similar" from
#: "flow says the camera moved", and flow does that far below 0.4.
MOTION_FLOOR = 0.05


def ssim_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Structural similarity between two frames. Returns 0.0–1.0."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # Ensure same size
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))

    # Compute SSIM via means, variances, covariance
    mu_a = cv2.GaussianBlur(gray_a.astype(np.float64), (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b.astype(np.float64), (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(gray_a.astype(np.float64) ** 2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(gray_b.astype(np.float64) ** 2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(gray_a.astype(np.float64) * gray_b.astype(np.float64), (11, 11), 1.5) - mu_ab

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu_ab + C1) * (2 * sigma_ab + C2)) / \
               ((mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2))

    return float(ssim_map.mean())


def is_consecutive_duplicate(
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    threshold: float | None = None,
    motion: float | None = None,
) -> bool:
    """True if frame is too similar to the previous frame.

    ``motion`` is the frame's measured motion score (already computed by the
    caller) normalized like _motion_score: mean full-res flow / 10. A moving
    frame is never a duplicate — at the default candidate cadence, SSIM
    ≥ 0.85 coexists with real parallax (London: flow ≈ 1.4 px between
    consecutive 59.94 fps candidates, SSIM ≈ 0.86), and rejecting those
    frames starves SfM into collapse (1 kept frame → zero registered cameras).
    """
    if prev_frame is None:
        return False
    if threshold is None:
        threshold = settings.processing.frame_overlap_threshold
    if ssim_score(frame, prev_frame) < threshold:
        return False
    if motion is not None and motion >= MOTION_FLOOR:
        return False  # measured change: parallax is accumulating, not a duplicate
    return True


def _phash(frame: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Compute perceptual hash of a frame as a boolean array."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4))
    # DCT-based pHash
    resized = resized.astype(np.float64)
    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    return dct_low >= median


def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    """Number of differing bits between two perceptual hashes."""
    return int(np.sum(hash_a != hash_b))


def is_duplicate_of_kept(
    frame: np.ndarray,
    kept_hashes: list[np.ndarray],
    threshold: int | None = None,
    motion: float | None = None,
) -> bool:
    """True if frame's perceptual hash is too close to any kept frame.

    Same motion veto as the SSIM check: pHash's coarse layout bits barely
    move under slow drift (sunset: hamming 2–8 vs kept frames while flow
    showed measured change), so a hash match on a moving frame is a hash
    artifact, not redundancy.
    """
    if not kept_hashes:
        return False
    if motion is not None and motion >= MOTION_FLOOR:
        return False
    if threshold is None:
        threshold = settings.processing.dedup_hash_threshold
    h = _phash(frame)
    return any(hamming_distance(h, kh) <= threshold for kh in kept_hashes)


def compute_phash(frame: np.ndarray) -> np.ndarray:
    """Public accessor — compute and return the perceptual hash."""
    return _phash(frame)
