"""Texture projection — choose the best camera for every mesh face.

Per face the projector scores each registered view by how directly the face
points at the camera (viewing-angle term) and how close the camera is
(distance term). A view only counts when the face's centroid projects
inside the image frustum *and* is not occluded — occlusion is tested
against the per-view depth map (camera-space z) when one is available.

Faces no camera sees (back-facing, out of frustum, or occluded everywhere)
are left unassigned (-1) and keep their dense-cloud vertex colour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.logging_config import get_logger
from app.services.mesh import TriangleMesh

log = get_logger("drone_recon.services.texture_projector")


@dataclass
class TextureView:
    """One registered camera + its image for texturing."""

    frame_id: str
    image: np.ndarray  # (H, W, 3) RGB uint8
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3) — X_world = R @ X_cam + t
    t: np.ndarray  # (3,)
    depth: np.ndarray | None = None  # (Hd, Wd) camera-space z, metres
    #: Intrinsics OF THE DEPTH MAP's grid. Native-resolution maps live on a
    #: coarser grid than the frame; sampling them with frame-grid pixels
    #: reads the wrong cells (or crashes — measured on Video_Mission_9ab1aa:
    #: "index 1011 is out of bounds for axis 0 with size 518", which silently
    #: degraded the whole viewer GLB to vertex colors). None = frame grid.
    K_depth: np.ndarray | None = None


@dataclass
class TextureProjection:
    """Per-face best-camera assignment."""

    assignment: np.ndarray  # (M,) int — index into views, -1 = unseen
    score: np.ndarray  # (M,) float — best view score (0 for unseen)
    visible_fraction: float  # share of faces textured
    view_counts: dict = field(default_factory=dict)  # frame_id -> faces

    def to_dict(self) -> dict:
        return {
            "faces_total": int(len(self.assignment)),
            "faces_textured": int((self.assignment >= 0).sum()),
            "visible_fraction": round(self.visible_fraction, 4),
            "view_counts": {str(k): int(v) for k, v in self.view_counts.items()},
        }


def _project_to_image(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                      world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into one camera (X_cam = Rᵀ(X_world − t)).
    Returns (uv, depth) — uv pixel coords, depth along camera z."""
    cam = (world - t) @ R  # (…, 3)
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = cam[:, :2] / depth[:, None]
    uv = (uv @ K[:2, :2].T) + K[:2, 2]
    return uv, depth


def scene_camera_distance_cap(views: list[TextureView], configured_cap: float) -> float:
    """Per-face camera-distance cap: the configured value, extended for far field.

    The configured cap (settings.mesh.texture_max_camera_dist, 500 m) assumes
    a near-field scene; on an oblique far-field survey (airport1: ground
    500-970 m from the camera) every face is rejected and texturing falls
    back to vertex colours with "no face has a visible camera". Derive the
    honest cap from the geometry the run actually verified: the P99.5
    camera-space depth of the sparse points (their far tail is real oblique
    coverage), plus 10% headroom. The configured value is a floor, never a
    target: the cap can only be EXTENDED, so near-field runs behave exactly
    as configured.
    """
    if not views:
        return configured_cap
    deepest = 0.0
    for v in views:
        d = np.asarray(v.depth, dtype=np.float64) if v.depth is not None else None
        if d is None:
            continue
        valid = d[np.isfinite(d) & (d > 0)]
        if valid.size:
            deepest = max(deepest, float(np.percentile(valid, 99.5)))
    if deepest <= configured_cap:
        return configured_cap
    return round(deepest * 1.1, 1)


def select_best_views(
    mesh: TriangleMesh,
    views: list[TextureView],
    max_camera_dist: float = 500.0,
    min_cos_angle: float = 0.1,
) -> TextureProjection:
    """Pick the best (angle + distance, occlusion-checked) view per face."""
    centroids = mesh.face_centroids()
    normals = mesh.face_normals()
    # Orient normals toward the camera cluster: triangulation winding is
    # arbitrary, so flip faces that point away from the mean viewpoint.
    if views:
        centers = np.stack([v.t for v in views if v.image is not None])
        if len(centers):
            mean_view = centers.mean(axis=0)
            flip = ((mean_view[None, :] - centroids) * normals).sum(axis=1) < 0
            normals[flip] = -normals[flip]
    m = mesh.m
    assignment = np.full(m, -1, dtype=np.int64)
    score = np.zeros(m)
    view_counts = {v.frame_id: 0 for v in views}

    for vi, view in enumerate(views):
        if view.image is None:
            continue
        H, W = view.image.shape[:2]
        cam_center = view.t  # X_world = R X_cam + t → centre sits at t
        to_cam = cam_center[None, :] - centroids
        dist = np.linalg.norm(to_cam, axis=1)
        valid_dist = (dist < max_camera_dist) & (dist > 1e-9)
        if not valid_dist.any():
            continue
        cosang = (to_cam[valid_dist] / dist[valid_dist, None] * normals[valid_dist]).sum(axis=1)
        front = cosang > min_cos_angle
        idx = np.flatnonzero(valid_dist)[front]
        if len(idx) == 0:
            continue
        sub = idx
        uv, depth = _project_to_image(view.K, view.R, view.t, centroids[sub])
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < W - 1) &
                  (uv[:, 1] >= 0) & (uv[:, 1] < H - 1) &
                  (depth > 0.01))
        sub = sub[inside]
        if len(sub) == 0:
            continue
        uv, depth = uv[inside], depth[inside]
        # Occlusion against the depth map (if present for this view).
        # The map lives on its OWN grid: project into it with K_depth and
        # index with the map's shape — never the image's (the legacy code
        # reused frame-grid uv against a 518-row map, which crashed on any
        # native-resolution run and skipped occlusion silently otherwise).
        if view.depth is not None:
            Kd = view.K_depth if view.K_depth is not None else view.K
            uv_d, _ = _project_to_image(Kd, view.R, view.t, centroids[sub])
            Hd, Wd = view.depth.shape[:2]
            duv = np.round(uv_d).astype(np.int64)
            duv[:, 0] = np.clip(duv[:, 0], 0, Wd - 1)
            duv[:, 1] = np.clip(duv[:, 1], 0, Hd - 1)
            z_map = view.depth[duv[:, 1], duv[:, 0]]
            visible = (z_map > 0) & (depth <= z_map * 1.02 + 0.05)
            sub = sub[visible]
            uv = uv[visible]
            depth = depth[visible]
        if len(sub) == 0:
            continue
        dterm = 1.0 / (1.0 + depth / max_camera_dist)
        # (angle already restricted to front-facing; recompute per visible)
        csub = centroids[sub]
        to_cam = cam_center[None, :] - csub
        dn = np.linalg.norm(to_cam, axis=1)
        ca = (to_cam / dn[:, None] * normals[sub]).sum(axis=1)
        s = ca * dterm
        better = s > score[sub]
        new_faces = sub[better]
        assignment[new_faces] = vi
        score[new_faces] = s[better]
        view_counts[view.frame_id] += int(better.sum())

    visible = assignment >= 0
    visible_fraction = float(visible.mean()) if m else 0.0
    log.info("texture_views_selected", faces_textured=int(visible.sum()), total=m,
             fraction=round(visible_fraction, 3))
    return TextureProjection(assignment=assignment, score=score,
                             visible_fraction=visible_fraction,
                             view_counts=view_counts)
