"""Intelligent pair selection — avoids brute-force matching.

Selects image pairs using temporal proximity, image overlap estimation,
and optional GPS-based distance scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger

log = get_logger("drone_recon.services.pair_selector")


@dataclass
class ImageInfo:
    """Metadata for a frame used in pair selection."""
    frame_id: str
    index: int
    timestamp_sec: float
    file_path: str
    gps_lat: float | None = None
    gps_lon: float | None = None
    entropy: float = 0.0


def compute_image_entropy(image: np.ndarray) -> float:
    """Shannon entropy of the grayscale histogram — higher = more texture."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def select_pairs(
    images: list[ImageInfo],
    *,
    strategy: str = "adaptive",
    temporal_window: int = 10,
    max_pairs: int = 500,
    min_overlap_score: float = 0.1,
) -> list[tuple[str, str]]:
    """Select which image pairs to match.

    Strategies:
    - "sequential": consecutive pairs only
    - "temporal": within a sliding window
    - "adaptive": temporal + overlap scoring + GPS distance
    """
    if strategy == "sequential":
        return _sequential_pairs(images)
    if strategy == "temporal":
        return _temporal_pairs(images, temporal_window, max_pairs)
    return _adaptive_pairs(images, temporal_window, max_pairs, min_overlap_score)


def _sequential_pairs(images: list[ImageInfo]) -> list[tuple[str, str]]:
    """Each frame paired with its immediate successor."""
    return [(images[i].frame_id, images[i + 1].frame_id) for i in range(len(images) - 1)]


def _temporal_pairs(
    images: list[ImageInfo], window: int, max_pairs: int
) -> list[tuple[str, str]]:
    """Pairs within a sliding temporal window."""
    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, min(len(images), i + window + 1)):
            pairs.append((images[i].frame_id, images[j].frame_id))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _adaptive_pairs(
    images: list[ImageInfo],
    window: int,
    max_pairs: int,
    min_overlap: float,
) -> list[tuple[str, str]]:
    """Score candidate pairs by overlap likelihood and keep the best."""
    candidates = []

    for i in range(len(images)):
        for j in range(i + 1, min(len(images), i + window + 1)):
            score = _pair_score(images[i], images[j])
            if score >= min_overlap:
                candidates.append((score, images[i].frame_id, images[j].frame_id))

    # Sort by score descending, take top N
    candidates.sort(reverse=True)
    pairs = [(a, b) for _, a, b in candidates[:max_pairs]]

    log.info(
        "pairs_selected",
        strategy="adaptive",
        total_candidates=len(candidates),
        selected=len(pairs),
    )

    return pairs


def _pair_score(a: ImageInfo, b: ImageInfo) -> float:
    """Score how likely two images are to have useful overlap.

    Returns 0.0–1.0 where higher = more likely to match well.
    """
    # Temporal proximity: closer frames more likely to overlap
    time_diff = abs(a.timestamp_sec - b.timestamp_sec)
    temporal_score = 1.0 / (1.0 + time_diff)

    # Entropy: higher entropy = more features to match
    entropy_score = min((a.entropy + b.entropy) / 16.0, 1.0)

    # GPS distance (if available)
    gps_score = 1.0
    if a.gps_lat is not None and b.gps_lat is not None:
        dist = _haversine(a.gps_lat, a.gps_lon, b.gps_lat, b.gps_lon)
        # Frames within 100m are good candidates
        gps_score = max(0.0, 1.0 - dist / 100.0)

    return 0.4 * temporal_score + 0.3 * entropy_score + 0.3 * gps_score


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS coordinates."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
