"""Surface normal estimation — PCA over k-nearest neighbours.

For every point the covariance of its k-NN neighbourhood is diagonalised;
the eigenvector of the smallest eigenvalue is the surface normal. Normals
are then flipped to point toward a reference viewpoint (e.g. the mean
camera position), which orients them consistently for aerial captures.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from app.logging_config import get_logger
from app.services.pointcloud import PointCloud

log = get_logger("drone_recon.services.normal_estimator")


def estimate_normals(
    points: np.ndarray,
    k: int = 20,
    viewpoint: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Estimate unit normals for *points* (N, 3).

    Parameters
    ----------
    points:
        World coordinates (N, 3).
    k:
        Neighbourhood size for the PCA fit.
    viewpoint:
        Reference point the normals are oriented toward. When *None* the
        sign is arbitrary (unit normals still returned).

    Returns
    -------
    (N, 3) float64 unit normals.
    """
    n = len(points)
    if n == 0:
        return np.zeros((0, 3))
    if n <= k:
        k = max(n - 1, 1)

    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    nn = points[idx[:, 1:]]  # exclude self

    # Covariance of each neighbourhood (3x3) in one batched einsum.
    centered = nn - points[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / k

    # Smallest eigenvector per neighbourhood.
    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]

    # Normalise (guard degenerate flat/empty neighbourhoods).
    norm = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, norm[:, None], out=np.zeros_like(normals), where=norm[:, None] > 1e-12)

    if viewpoint is not None:
        to_view = np.asarray(viewpoint, dtype=np.float64) - points
        facing = np.einsum("ij,ij->i", normals, to_view)
        normals[facing < 0] *= -1.0

    return normals


def add_normals(cloud: PointCloud, k: int = 20, viewpoint: tuple[float, float, float] | None = None) -> PointCloud:
    """Attach normals to *cloud* (in place) and return it."""
    cloud.with_normals(estimate_normals(cloud.xyz, k=k, viewpoint=viewpoint))
    return cloud
