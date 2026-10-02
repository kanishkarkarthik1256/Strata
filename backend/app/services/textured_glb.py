"""Textured viewer GLB — production promotion of the texture experiment.

The viewer LOD (``mesh_viewer.glb``) is a downstream derivative of the
authoritative mesh: this module derives a UV-atlas texture for it from the
registered source views (Depth Anything conditioning exclusions apply — a
view the depth stage refused for its gauge is a view we do not texture with)
and writes it through :func:`mesh_glb.export_viewer_glb`, which embeds the
atlas as a PNG and emits TEXCOORD_0.

Authoritative artifacts (mesh.ply / base_mesh.ply / mesh_full.ply) are never
read here for modification and never written — the texture is baked onto the
decimated LOD copy. When texturing genuinely fails for a run (no views, no
atlas written, UV overflow), the fallback is a COLOR_0 vertex-color GLB and
the payload names ``texture_fallback_reason`` — never a silent downgrade.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.texture_blender import build_texture_atlas
from app.services.texture_projector import TextureView
from app.services.uv_mapper import AtlasLayout, build_atlas_layout

log = get_logger("drone_recon.services.textured_glb")

#: Atlas width — matches settings.mesh.atlas_width's default. The one-tile-
#: per-face grid on a decimated LOD keeps texel density usable at this size
#: (256k faces ⇒ ~14 px cells) while the embedded PNG stays a few MB.
TEXTURED_ATLAS_WIDTH = 8192
#: Faces beyond this keep the vertex-color fallback. Aligned with the viewer
#: LOD budget itself: the decimated GLB can never exceed it by construction,
#: so a well-formed run always textures; the cap only guards misuse.
from app.services.mesh_glb import viewer_triangle_budget as _viewer_triangle_budget


@dataclass
class TexturePayload:
    """Everything the GLB writer needs to emit a textured primitive."""

    image_bytes: bytes
    mime_type: str  # "image/jpeg" for the photographic atlas, "image/png" otherwise
    uv: np.ndarray  # (M, 3, 2) float32, normalized
    source: str  # provenance shown in the payload ("registered_views")
    fallback_reason: str | None = None  # set ONLY on the vertex-color fallback
    stats: dict | None = None


def build_texture_payload(mesh: TriangleMesh, workspace) -> TexturePayload:
    """Bake a UV atlas for *mesh* (the LOD copy) from the run's usable views.

    Never returns None: every failure mode returns the vertex-color fallback
    payload with its named reason, so a non-textured GLB is always an
    EXPLAINED downgrade in the stage payload, never a silent one.
    """
    started = time.perf_counter()
    views, excluded = _texture_views(workspace)
    if not views:
        return _vertex_color_fallback("no registered views with images available for texturing")
    face_count = mesh.m
    atlas_cap = _viewer_triangle_budget()
    if face_count > atlas_cap:
        log.warning("texture_atlas_too_large", faces=face_count, cap=atlas_cap)
        return _vertex_color_fallback(
            f"mesh has {face_count} faces — beyond the {atlas_cap}-face atlas cap"
        )
    try:
        layout = build_atlas_layout(face_count, atlas_width=TEXTURED_ATLAS_WIDTH)
    except ValueError as exc:
        log.warning("texture_atlas_layout_failed", error=str(exc))
        return _vertex_color_fallback(f"atlas layout failed: {exc}")
    atlas, stats = build_texture_atlas(mesh, views, layout)
    if not stats.get("atlas_written"):
        # No face has a visible camera — an atlas would be all black.
        return _vertex_color_fallback(
            f"no face has a visible camera ({stats.get('note', 'no visible faces')})"
        )
    # Photographic atlas → JPEG: the source frames are lossy JPEGs, so JPEG
    # adds negligible further loss while keeping the embedded image (and the
    # GLB the browser downloads) bounded — measured on the 8k real-run atlas:
    # PNG 107 MB, q90 34.5 MB, q85 28.4 MB. q85 chosen: one extra lossy
    # generation on already-lossy sources, sub-1% PSNR delta vs q90.
    # The atlas is RGB (views are loaded RGB); cv2 encodes BGR, so convert —
    # encoding RGB bytes as BGR shipped red/blue-swapped textures.
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        log.warning("texture_image_encode_failed")
        return _vertex_color_fallback("atlas image encoding failed")
    stats["views_used"] = len(views)
    stats["conditioning_excluded_views"] = excluded
    stats["atlas_bytes"] = int(len(buf))
    stats["took_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return TexturePayload(
        image_bytes=buf.tobytes(),
        mime_type="image/jpeg",
        uv=layout.uv_norm,
        source="registered_views",
        stats=stats,
    )


def _texture_views(workspace) -> tuple[list[TextureView], list[str]]:
    """Load TextureViews for frames the depth stage found conditioning-usable.

    ``depth_alignment_report.json`` records per-view conditioning; views the
    depth stage excluded (hover null-space, unalignable anchor) are excluded
    here too — we do not texture with a view whose depth gauge was refused.
    Absent report (older runs): all registered frames with an image are used.
    """
    from app.services.texture_blender import _load_views

    poses_path = workspace / "poses.json"
    selected = workspace / "selected"
    image_dir = selected if selected.is_dir() else workspace / "frames"
    views, missing = _load_views(poses_path, image_dir, workspace / "depth")
    conditions: dict = {}
    report = workspace / "depth_alignment_report.json"
    if report.is_file():
        try:
            # to_report() persists the conditioning map under "conditioning".
            conditions = json.loads(report.read_text()).get("conditioning", {})
        except (OSError, ValueError):
            conditions = {}
    if not conditions:
        return views, missing
    usable = [v for v in views if conditions.get(v.frame_id, {}).get("usable", True)]
    dropped = sorted(set(v.frame_id for v in views) - set(v.frame_id for v in usable))
    return usable, missing + dropped


def _vertex_color_fallback(reason: str) -> TexturePayload:
    """Honest fallback marker: the GLB writer emits COLOR_0 and the payload
    carries ``texture_fallback_reason`` so the downgrade is never silent."""
    return TexturePayload(image_bytes=b"", mime_type="image/png", uv=np.zeros((0, 3, 2)),
                          source="vertex_colors", fallback_reason=reason)
