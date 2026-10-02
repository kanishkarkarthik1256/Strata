"""Canonical camera geometry — the single source of truth for coordinate transforms.

Convention (STRATA-wide):

* ``R`` is the camera-to-world rotation (world-from-camera).
* ``C`` (stored as ``t`` in poses.json) is the camera centre in world coords.
* ``K`` maps camera-space coordinates to pixels (K @ X_cam = lambda * [u,v,1]).

World point:

    X_world = R @ X_camera + C

Projection:

    x ~ K @ R.T @ (X_world - C)

Every stage (sparse, dense, validation) must use these helpers; no stage may
reinterpret the pose translation or re-derive its own convention.
"""

from __future__ import annotations

import numpy as np


def camera_to_world(X_cam: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """X_world = R @ X_camera + C  (X: (...,3), batched)."""
    X_cam = np.asarray(X_cam, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    return X_cam @ R.T + C


def world_to_camera(X_world: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """X_camera = R.T @ (X_world - C)  (X: (...,3), batched)."""
    X_world = np.asarray(X_world, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    return (X_world - C) @ R


def project_world_to_camera(X_world: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Alias of world_to_camera for readability at call sites."""
    return world_to_camera(X_world, R, C)


def unproject_pixel_to_camera(
    u: np.ndarray, v: np.ndarray, depth: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """X_camera = Z * inv(K) @ [u,v,1]  (u,v,Z: (N,), K: (3,3)).

    ``depth`` is the camera-space Z (positive in front of the camera).
    """
    K = np.asarray(K, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.column_stack([x, y, z])


def project_world_to_pixel(
    X_world: np.ndarray, R: np.ndarray, C: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to pixels; returns (u, v, Z_cam).

    Points behind the camera (Z <= 0) yield NaN pixels; callers gate on Z > 0.
    """
    Xc = world_to_camera(X_world, R, C)
    K = np.asarray(K, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = Xc[:, 0] / np.where(np.abs(z) > 1e-12, z, np.nan) * fx + cx
        v = Xc[:, 1] / np.where(np.abs(z) > 1e-12, z, np.nan) * fy + cy
    return u, v, z


def camera_ray(X_world: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Unit world-space viewing ray from camera centre toward X (batched)."""
    d = np.asarray(X_world, dtype=np.float64) - np.asarray(C, dtype=np.float64)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(n, 1e-12)
