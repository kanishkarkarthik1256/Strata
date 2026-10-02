"""Texture atlas baking with per-image exposure compensation.

Each textured face's source triangle is projected into its atlas tile with a
perspective-free affine warp (valid for the small triangles that tile a
surface mesh). Seam artifacts between images sharing a scene are reduced by
a real exposure-compensation step: every image's contribution is rescaled by
a gain that drives its sampled mean colour toward the global mean of the
views actually used.

Faces that no view sees keep their dense-cloud vertex colour (baked into the
mesh by the earlier stages), so the mesh never loses appearance.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register
from app.services.texture_projector import TextureView, select_best_views
from app.services.uv_mapper import AtlasLayout, build_atlas_layout

log = get_logger("drone_recon.services.texture_blender")


def _gain_compensation(means: list[float]) -> list[float]:
    """Per-image gains driving sampled mean colours to the global mean."""
    if not means:
        return []
    target = float(np.mean(means))
    if target <= 0:
        return [1.0] * len(means)
    return [float(np.clip(target / m, 0.6, 1.6)) if m > 0 else 1.0 for m in means]


def build_texture_atlas(
    mesh: TriangleMesh,
    views: list[TextureView],
    layout: AtlasLayout,
    min_cell_sample: int = 4,
) -> tuple[np.ndarray, dict]:
    """Bake the atlas image. Returns ``(atlas, stats)``; atlas is in the same
    channel order as the source images (cv2 BGR when loaded via cv2)."""
    from app.services.texture_projector import scene_camera_distance_cap

    camera_dist_cap = scene_camera_distance_cap(
        views, settings.mesh.texture_max_camera_dist)
    projection = select_best_views(mesh, views, max_camera_dist=camera_dist_cap)
    atlas = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    if projection.visible_fraction <= 0:
        return atlas, {**projection.to_dict(), "atlas_written": False,
                       "camera_dist_cap_m": camera_dist_cap,
                       "note": "no face has a visible camera — mesh keeps vertex colours"}
    # Exposure compensation from mean colours of each used view.
    view_ids = [v for v in range(len(views)) if (projection.assignment == v).any()]
    means: list[float] = []
    verts = mesh.vertices
    for vi in view_ids:
        faces = np.flatnonzero(projection.assignment == vi)
        pts = verts[mesh.faces[faces]].reshape(-1, 3)
        uv, depth = _project_all(views[vi], pts)
        ok = np.isfinite(uv).all(axis=1) & (depth > 0)
        if not ok.any():
            means.append(0.0)
            continue
        sampled = views[vi].image
        H, W = sampled.shape[:2]
        uu = np.clip(uv[ok, 0].astype(np.int64), 0, W - 1)
        vv = np.clip(uv[ok, 1].astype(np.int64), 0, H - 1)
        means.append(float(sampled[vv, uu].mean()))
    gains = _gain_compensation(means)
    gain_by_view: dict[int, float] = dict(zip(view_ids, gains))

    def _bake_faces(vi: int, faces: np.ndarray) -> int:
        """Bake ``faces`` (mesh face indices) from view ``vi``.

        Phase 1 is fully vectorised (projection, crop boxes, and closed-form
        affine solves for all faces at once). Production rasterization keeps
        OpenCV's exact warpAffine semantics; the experimental remap path is
        available separately but is not used because its interpolation and
        border rounding are not byte-identical.
        """
        img = views[vi].image
        H, W = img.shape[:2]
        gain = gain_by_view.get(vi, 1.0)

        tris = mesh.faces[faces]
        src, depth = _project_all(views[vi], verts[tris].reshape(-1, 3))
        src = src.reshape(-1, 3, 2)
        depth = depth.reshape(-1, 3)
        ok = np.isfinite(src).all(axis=(1, 2)) & (depth.min(axis=1) > 0)
        if not ok.any():
            return 0
        faces = faces[ok]
        src = src[ok]

        # Crop boxes (same clamp arithmetic as the per-face original).
        x0 = np.floor(src[:, :, 0].min(axis=1)).astype(np.int64) - 1
        y0 = np.floor(src[:, :, 1].min(axis=1)).astype(np.int64) - 1
        x1 = np.ceil(src[:, :, 0].max(axis=1)).astype(np.int64) + 2
        y1 = np.ceil(src[:, :, 1].max(axis=1)).astype(np.int64) + 2
        x0c = np.clip(x0, 0, W)
        y0c = np.clip(y0, 0, H)
        x1c = np.clip(x1, 0, W)
        y1c = np.clip(y1, 0, H)
        valid = ((x1c - x0c) >= 1) & ((y1c - y0c) >= 1)
        if not valid.any():
            return 0
        faces = faces[valid]
        src = src[valid]
        x0c, y0c, x1c, y1c = x0c[valid], y0c[valid], x1c[valid], y1c[valid]

        # Destination corners relative to each tile's own origin, and the
        # src->tile affine per face, solved as one batched 3x3 system —
        # the same closed form cv2.getAffineTransform applies per face.
        tile = layout.uv_px[faces]  # (F, 3, 2)
        tile_origin = np.stack(
            [tile[:, 0, 0].astype(np.int64), tile[:, 0, 1].astype(np.int64)], axis=-1
        )[:, None, :]
        dst_rel = tile - tile_origin
        src_rel = src - np.stack([x0c, y0c], axis=1)[:, None, :]
        hom = np.concatenate([src_rel, np.ones((len(faces), 3, 1))], axis=2)
        try:
            aff = np.linalg.solve(hom, dst_rel).transpose(0, 2, 1)  # (F, 2, 3)
        except np.linalg.LinAlgError:
            return _bake_faces_per_face(
                atlas, img, faces, layout, src, x0c, y0c, x1c, y1c, gain
            )

        cell = layout.cell
        # OpenCV remap is retained as an experimental benchmark path, but its
        # interpolation and border rounding are not byte-identical to
        # warpAffine. Keep production output on the exact reference rasterizer.
        return _bake_faces_per_face(
            atlas, img, faces, layout, src, x0c, y0c, x1c, y1c, gain
        )

    # ---- bake all views (loop over views, batched per view) ----
    textured = 0
    for vi in view_ids:
        faces = np.flatnonzero(projection.assignment == vi)
        if len(faces) == 0:
            continue
        textured += _bake_faces(vi, faces)

    stats = {
        **projection.to_dict(),
        "camera_dist_cap_m": camera_dist_cap,
        "faces_baked": textured,
        "faces_textured": textured,
        "atlas": [layout.width, layout.height],
        "view_gains": {str(vi): round(g, 4) for vi, g in gain_by_view.items()},
        "atlas_written": True,
    }
    log.info("texture_atlas_baked", faces=textured, width=layout.width, height=layout.height)
    return atlas, stats


def _inv_affine23(aff: np.ndarray) -> np.ndarray:
    """Inverse of a batch of (F, 2, 3) affine matrices.

    aff maps src_rel → dst_rel:  dst = aff @ [sx, sy, 1].
    Returns (F, 2, 3) mapping dst_rel → src_rel.
    """
    a, b, c = aff[:, 0, 0], aff[:, 0, 1], aff[:, 0, 2]
    d, e, f = aff[:, 1, 0], aff[:, 1, 1], aff[:, 1, 2]
    det = a * e - b * d
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)  # guard degenerate
    inv = np.empty_like(aff)
    inv[:, 0, 0] = e / det
    inv[:, 0, 1] = -b / det
    inv[:, 0, 2] = (b * f - c * e) / det
    inv[:, 1, 0] = -d / det
    inv[:, 1, 1] = a / det
    inv[:, 1, 2] = (c * d - a * f) / det
    return inv


def _bake_faces_per_face(
    atlas: np.ndarray,
    img: np.ndarray,
    faces: np.ndarray,
    layout: AtlasLayout,
    src: np.ndarray,
    x0c: np.ndarray,
    y0c: np.ndarray,
    x1c: np.ndarray,
    y1c: np.ndarray,
    gain: float,
) -> int:
    """Correctness fallback for faces whose source triangles are singular."""
    count = 0
    for k, face in enumerate(faces):
        source = src[k]
        source_rel = source - np.array([x0c[k], y0c[k]])
        destination = layout.uv_px[face]
        destination_rel = destination - destination[0]
        try:
            affine = cv2.getAffineTransform(
                source_rel.astype(np.float32), destination_rel.astype(np.float32)
            )
        except cv2.error:
            continue
        crop = img[y0c[k]:y1c[k], x0c[k]:x1c[k]]
        warped = cv2.warpAffine(crop.astype(np.float32), affine, (layout.cell, layout.cell))
        warped = np.clip(warped * gain, 0, 255).astype(np.uint8)
        x_t, y_t = map(int, destination[0])
        roi = atlas[y_t:y_t + layout.cell, x_t:x_t + layout.cell]
        roi[:] = np.where(np.any(warped > 0, axis=2)[:, :, None], warped, roi)
        count += 1
    return count


def _bake_faces_batched_remap(
    atlas: np.ndarray,
    img: np.ndarray,
    faces: np.ndarray,
    layout: AtlasLayout,
    aff: np.ndarray,       # (F, 2, 3)  src_rel -> dst_rel
    x0c: np.ndarray,       # (F,) crop x-origin in source
    y0c: np.ndarray,       # (F,) crop y-origin in source
    x1c: np.ndarray,       # (F,) crop x-end in source
    y1c: np.ndarray,       # (F,) crop y-end in source
    cell: int,
    gain: float,
) -> int:
    """Rasterize all *faces* from one view into *atlas* in a single batched
    cv2.remap pass.

    Builds a face-index label map and per-pixel source-coordinate maps over
    the bounding box of this view's atlas tiles, then calls remap once.
    The result is composited into the atlas with the same "new-pixel overwrites
    old" rule the per-face loop used.
    """
    H, W = img.shape[:2]
    tile0s = layout.uv_px[faces]          # (F, 3, 2)
    x_ts = tile0s[:, 0, 0].astype(np.int64)
    y_ts = tile0s[:, 0, 1].astype(np.int64)

    # Bounding box of all tiles this view writes to.
    min_x = max(0, int(x_ts.min()))
    max_x = min(layout.width, int(x_ts.max()) + cell)
    min_y = max(0, int(y_ts.min()))
    max_y = min(layout.height, int(y_ts.max()) + cell)
    bw, bh = max_x - min_x, max_y - min_y
    if bw <= 0 or bh <= 0:
        return 0

    # ---- face-index label map (which face owns each pixel in the bbox) ----
    face_idx = np.full((bh, bw), -1, dtype=np.int32)
    for k in range(len(faces)):
        face_idx[
            y_ts[k] - min_y : y_ts[k] + cell - min_y,
            x_ts[k] - min_x : x_ts[k] + cell - min_x,
        ] = k

    # ---- inverse affine: atlas-pixel-relative -> source-crop-relative ----
    inv = _inv_affine23(aff)              # (F, 2, 3)

    # ---- build xmap / ymap: source coords for every pixel in the bbox ----
    # For pixel (py, px) in the bbox belonging to face k:
    #   src_x = inv[k,0,0]*(px-x_ts[k]) + inv[k,0,1]*(py-y_ts[k]) + inv[k,0,2] + x0c[k]
    #   src_y = inv[k,1,0]*(px-x_ts[k]) + inv[k,1,1]*(py-y_ts[k]) + inv[k,1,2] + y0c[k]
    yy, xx = np.mgrid[min_y:max_y, min_x:max_x]
    safe_idx = np.maximum(face_idx, 0)
    px_rel = xx - x_ts[safe_idx]          # (bh, bw) — tile-relative x
    py_rel = yy - y_ts[safe_idx]          # (bh, bw) — tile-relative y
    px_rel = np.where(face_idx >= 0, px_rel, 0.0)
    py_rel = np.where(face_idx >= 0, py_rel, 0.0)

    src_x = (inv[:, 0, 0][face_idx] * px_rel
             + inv[:, 0, 1][face_idx] * py_rel
             + inv[:, 0, 2][safe_idx]
             + x0c[safe_idx])
    src_y = (inv[:, 1, 0][face_idx] * px_rel
             + inv[:, 1, 1][face_idx] * py_rel
             + inv[:, 1, 2][safe_idx]
             + y0c[safe_idx])

    # Pixels with no face, or whose source coords fall outside the CROP
    # (not just the full image), get a sentinel so remap returns 0.
    # This matches the per-face warpAffine behaviour: warpAffine on a crop
    # returns 0 for any output pixel whose input maps outside the crop.
    in_crop = (src_x >= x0c[safe_idx]) & (src_x < x1c[safe_idx])
    in_crop &= (src_y >= y0c[safe_idx]) & (src_y < y1c[safe_idx])
    valid = (face_idx >= 0) & in_crop
    xmap = np.where(valid, src_x, -1e6).astype(np.float32)
    ymap = np.where(valid, src_y, -1e6).astype(np.float32)

    # ---- single batched remap ----
    warped = cv2.remap(img, xmap, ymap, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped = np.clip(warped.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    # ---- composite into atlas (new pixels overwrite old, same as per-face) ----
    mask = np.any(warped > 0, axis=2)     # (bh, bw)
    roi = atlas[min_y:max_y, min_x:max_x]
    roi[:] = np.where(mask[:, :, None], warped, roi)
    return int(len(faces))



def _project_all(view: TextureView, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into one camera: uv pixel coords + depth.
    Pose convention: X_world = R X_cam + t, so X_cam = Rᵀ (X_world − t)."""
    cam = (world - view.t) @ view.R
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = cam[:, :2] / depth[:, None]
    uv = (uv @ view.K[:2, :2].T) + view.K[:2, 2]
    return uv, depth


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class TextureStage(PipelineStage):
    name = "texture"
    description = "Best-view texture projection into a UV atlas"
    artifact_rel = "texture/texture_atlas.png"
    dependencies = ("mesh_generation", "mesh_optimization", "mesh_repair")

    def validate_inputs(self) -> None:
        if not (self.workspace / "mesh" / "repaired_mesh.ply").exists():
            raise StageNotApplicable("no repaired mesh — run mesh_repair first")

    def execute(self) -> None:
        poses_path = self.workspace / "poses.json"
        image_dir = self.workspace / "selected"
        if not image_dir.is_dir():
            image_dir = self.workspace / "frames"
        views, missing = _load_views(poses_path, image_dir, self.workspace / "depth")
        if not views:
            raise StageNotApplicable("no registered views with images available for texturing")

        mesh = TriangleMesh.read_ply(self.workspace / "mesh" / "repaired_mesh.ply")
        layout = build_atlas_layout(mesh.m, atlas_width=settings.mesh.atlas_width)
        atlas, stats = build_texture_atlas(mesh, views, layout)

        tex_dir = self.workspace / "texture"
        tex_dir.mkdir(parents=True, exist_ok=True)
        atlas_path = tex_dir / "texture_atlas.png"
        # Atlas is RGB (views loaded RGB); cv2 writes BGR — convert.
        cv2.imwrite(str(atlas_path), cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR))
        np.save(tex_dir / "uv.npy", layout.uv_norm)
        # Per-face view index (into views) used for texturing.
        np.save(tex_dir / "face_view.npy", _projection_of(mesh, views, stats))
        meta = {
            "frame_ids": [v.frame_id for v in views],
            "atlas_width": layout.width, "atlas_height": layout.height,
            "faces_textured": stats.get("faces_textured", 0),
            "faces_baked": stats.get("faces_baked", 0),
            "view_gains": stats.get("view_gains", {}),
            "missing_frames": missing,
            "note": "" if stats.get("faces_textured", 0) else
                    "no faces textured — mesh keeps vertex colours",
        }
        (tex_dir / "texture_report.json").write_text(json.dumps(meta, indent=2))
        self._count = int(stats.get("faces_textured", 0))
        self._detail = meta
        self._outputs = [
            {"kind": "texture", "name": "texture_atlas", "path": str(atlas_path)},
            {"kind": "data", "name": "uv", "path": str(tex_dir / "uv.npy")},
        ]


def _load_views(poses_path: Path, image_dir: Path, depth_dir: Path) -> tuple[list[TextureView], list[str]]:
    """Load TextureViews for registered frames that have an image on disk."""
    if not poses_path.exists():
        return [], ["no poses.json"]
    import json as _json

    from app.services.depth_generator import read_depth_geometry

    frames = _json.loads(poses_path.read_text()).get("frames", [])
    views: list[TextureView] = []
    missing: list[str] = []
    for pose in frames:
        fid = pose["frame_id"]
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = image_dir / f"{fid}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            missing.append(fid)
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            missing.append(fid)
            continue
        # TextureView.image is documented RGB and the GLB/atlas consumers
        # write these bytes straight into color assets — cv2's native BGR
        # shipped red/blue-swapped textures on every textured run.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = None
        K_depth = None
        for name in (f"{fid}.npy", f"{fid}_depth.npy"):
            dpath = depth_dir / name
            if dpath.exists():
                try:
                    depth = np.load(dpath)
                except (OSError, ValueError):
                    depth = None
                # Native-resolution maps live on a coarser grid than the
                # frame; the occlusion test must project into THAT grid.
                # read_depth_geometry returns the frame K for legacy maps,
                # so both schemas resolve correctly here.
                if depth is not None:
                    K_depth, _, _ = read_depth_geometry(dpath, pose)
                break
        views.append(TextureView(
            frame_id=fid, image=img,
            K=np.asarray(pose["K"], dtype=np.float64),
            R=np.asarray(pose["R"], dtype=np.float64),
            t=np.asarray(pose["t"], dtype=np.float64),
            depth=depth.astype(np.float64) if depth is not None else None,
            K_depth=K_depth,
        ))
    return views, missing


def _projection_of(mesh: TriangleMesh, views: list[TextureView], stats: dict) -> np.ndarray:
    """Per-face view index used for texturing (recomputed deterministically)."""
    proj = select_best_views(mesh, views, max_camera_dist=settings.mesh.texture_max_camera_dist)
    return proj.assignment
