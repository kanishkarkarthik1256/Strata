"""AI feature extraction — SuperPoint with OpenCV SIFT fallback.

Tries to use SuperPoint (deep learning) for keypoint/descriptor extraction.
Falls back to OpenCV SIFT when SuperPoint weights are unavailable.

All functions are stateless — the extractor is a thin wrapper around model state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger

log = get_logger("drone_recon.services.feature_extractor")


@dataclass
class Features:
    """Extraction result for a single frame."""
    frame_id: str
    keypoints: np.ndarray  # (N, 2) float32 — x, y
    descriptors: np.ndarray  # (N, D) float32
    scores: np.ndarray  # (N,) float32 — keypoint confidence
    extraction_time_ms: float = 0.0
    device: str = "cpu"
    backend: str = "sift"


class FeatureExtractor:
    """Extracts keypoints and descriptors from images.

    Attempts SuperPoint first, falls back to SIFT.
    """

    def __init__(self, max_keypoints: int = 4096, nms_radius: int = 4):
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self._superpoint = None
        self._device = settings.ai.device
        self._backend = "sift"
        self._init_backend()

    def _init_backend(self) -> None:
        """Try to load SuperPoint, fall back to SIFT."""
        try:
            import torch
            from lightglue import SuperPoint

            self._superpoint = SuperPoint(
                max_num_keypoints=self.max_keypoints,
                nms_radius=self.nms_radius,
            ).eval()

            if self._device == "cuda" and torch.cuda.is_available():
                self._superpoint = self._superpoint.cuda()
                self._device = "cuda"
            else:
                self._device = "cpu"

            self._backend = "superpoint"
            log.info("feature_extractor_loaded", backend="superpoint", device=self._device)
        except (ImportError, OSError) as exc:
            log.info("superpoint_unavailable", reason=str(exc), fallback="sift")
            self._backend = "sift"
            self._sift = cv2.SIFT_create(nfeatures=self.max_keypoints)

    def extract(self, frame_id: str, image: np.ndarray) -> Features:
        """Extract features from a single BGR or grayscale image.

        Grayscale input is accepted so callers that already decoded to gray
        (entropy scoring) avoid a second color decode of the same file.
        """
        if self._backend == "superpoint" and self._superpoint is not None:
            return self._extract_superpoint(frame_id, image)
        return self._extract_sift(frame_id, image)

    def extract_batch(self, frames: list[tuple[str, np.ndarray]]) -> list[Features]:
        """Extract features from multiple frames."""
        return [self.extract(fid, img) for fid, img in frames]

    def _extract_superpoint(self, frame_id: str, image: np.ndarray) -> Features:
        """Extract using SuperPoint."""
        import torch

        start = time.perf_counter()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        tensor = torch.from_numpy(gray).float() / 255.0
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        if self._device == "cuda":
            tensor = tensor.cuda()

        with torch.no_grad():
            pred = self._superpoint({"image": tensor})

        kpts = pred["keypoints"][0].cpu().numpy()
        desc = pred["descriptors"][0].cpu().numpy()
        scores = pred["scores"][0].cpu().numpy()

        elapsed = (time.perf_counter() - start) * 1000

        return Features(
            frame_id=frame_id,
            keypoints=kpts.astype(np.float32),
            descriptors=desc.astype(np.float32),
            scores=scores.astype(np.float32),
            extraction_time_ms=round(elapsed, 2),
            device=self._device,
            backend="superpoint",
        )

    def _extract_sift(self, frame_id: str, image: np.ndarray) -> Features:
        """Extract using OpenCV SIFT."""
        start = time.perf_counter()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)

        elapsed = (time.perf_counter() - start) * 1000

        if not keypoints:
            return Features(
                frame_id=frame_id,
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=np.zeros((0, 128), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                extraction_time_ms=round(elapsed, 2),
                device="cpu",
                backend="sift",
            )

        kpts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        desc = descriptors.astype(np.float32) if descriptors is not None else np.zeros((len(keypoints), 128), dtype=np.float32)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)

        # Sort by score and take top N
        order = np.argsort(-scores)[: self.max_keypoints]
        kpts = kpts[order]
        desc = desc[order]
        scores = scores[order]

        return Features(
            frame_id=frame_id,
            keypoints=kpts,
            descriptors=desc,
            scores=scores,
            extraction_time_ms=round(elapsed, 2),
            device="cpu",
            backend="sift",
        )
