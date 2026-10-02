"""COLMAP text-format helpers (cameras.txt) — dataset calibration intake.

Flight_to_tower (JB3D) ships a COLMAP ``cameras.txt`` with one RADIAL camera
calibrated at 3840x2160 while the video decodes at that same resolution. The
sparse stage consumes a simple ``intrinsics.json``; this module converts the
COLMAP text model into that schema, scaling focal length and principal point
to the actual video resolution when they differ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.logging_config import get_logger

log = get_logger(__name__)

# COLMAP distortion models that map onto pycolmap's identically named models.
_SUPPORTED = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"}


def camera_txt_to_intrinsics(cameras_txt: Path, video_w: int, video_h: int) -> dict | None:
    """Parse the first PINHOLE-family camera from a COLMAP cameras.txt.

    Returns ``{model, fx, fy, cx, cy, k1, k2, width, height, orig_width,
    orig_height, scale}`` scaled to (video_w, video_h), or None when the
    model is unsupported/undecodable (caller proceeds with its fallback).
    """
    try:
        for raw in cameras_txt.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
            if len(parts) < 5:
                continue
            model = parts[1].upper()
            if model not in _SUPPORTED:
                log.info("cameras_txt_model_skipped", model=parts[1],
                         supported=sorted(_SUPPORTED))
                continue
            w, h = int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]
            sx = video_w / w
            sy = video_h / h
            if model == "SIMPLE_PINHOLE":
                f, cx, cy = params[0], params[1], params[2]
                fx = fy = f
                out = {"model": "PINHOLE", "k1": 0.0, "k2": 0.0}
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                out = {"model": "PINHOLE", "k1": 0.0, "k2": 0.0}
            elif model == "SIMPLE_RADIAL":
                f, cx, cy, k1 = params[0], params[1], params[2], params[3]
                fx = fy = f
                out = {"model": "RADIAL", "k1": k1, "k2": 0.0}
            elif model == "RADIAL":
                f, cx, cy, k1, k2 = params[:5]
                fx = fy = f
                out = {"model": "RADIAL", "k1": k1, "k2": k2}
            else:  # OPENCV
                fx, fy, cx, cy, k1, k2 = params[:6]
                out = {"model": "OPENCV", "k1": k1, "k2": k2}
            out.update({
                "fx": fx * sx,
                "fy": fy * sy,
                "cx": cx * sx,
                "cy": cy * sy,
                "width": video_w,
                "height": video_h,
                "orig_width": w,
                "orig_height": h,
                "scale": sx,
            })
            return out
    except (OSError, ValueError, IndexError) as exc:
        log.warning("cameras_txt_parse_failed", error=str(exc))
        return None
    return None
