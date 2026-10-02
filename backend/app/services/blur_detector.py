"""Blur detection via Laplacian variance.

Pure functions — take a BGR frame (np.ndarray), return a float score.
Higher score = sharper. Configurable threshold for rejection.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config.settings import settings


def blur_score(frame: np.ndarray) -> float:
    """Laplacian variance — higher means sharper.

    Works on any BGR frame. Converts to grayscale internally.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(frame: np.ndarray, threshold: float | None = None) -> bool:
    """True if the frame's blur score is below the threshold."""
    if threshold is None:
        threshold = settings.processing.blur_threshold
    return blur_score(frame) < threshold
