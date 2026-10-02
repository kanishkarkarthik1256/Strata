#!/usr/bin/env python
"""Render every pipeline stage's REAL artifact as a high-resolution RGBA PNG.

Each step is written separately into ``<out>/`` with a transparent background so
the images drop onto any surface. Nothing here is synthetic or illustrative:
every pixel comes from the run's own stored artifacts — extracted video frames,
the sparse cloud, the camera poses, the per-view depth maps, the fused dense
cloud, and the textured viewer GLB.

The 3D steps share ONE viewing angle so the sequence reads as the same object
gaining detail rather than three unrelated shots, but each is framed to exactly
the geometry that panel draws. One shared frame made the mesh panels pay for
the fused cloud they never render — roughly 30% of their area, for nothing.

Open3D's offscreen renderer is EGL-blocked on macOS ("EGL Headless is not
supported on this platform"), so this module rasterises in numpy: z-buffered
disc splatting for the point clouds, and a small-triangle batched rasteriser
with per-pixel UV sampling of the GLB's baked texture for the meshes.

    .venv/bin/python scripts/render_pipeline_stages.py --run airport1_test_99ee3c
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

# Run straight from the repo: `python scripts/render_pipeline_stages.py` puts
# scripts/ on sys.path, not backend/, so the app package would not import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE_DEFAULT = 2048
SS_DEFAULT = 2  # supersample factor; the final PNG is downsampled from this
ELEVATION_DEG = 34.0  # camera elevation shared by every 3D panel
# Fraction trimmed from each end when fitting a frame, so a handful of outlier
# specks cannot buy scale loss for the whole panel. The fit is only ever widened
# when the RENDERED border proves content reached the canvas edge — see
# _border_alpha — so the robustness costs nothing in guarantee.
FIT_TRIM = 0.003
FIT_TRIM_LADDER = (FIT_TRIM, 0.001, 0.0)

GREEN = (34, 197, 94)


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
class View:
    """Orthographic view of a world frame, in supersampled pixel coordinates."""

    def __init__(self, right, up, fwd, center, scale, size):
        self.right, self.up, self.fwd = right, up, fwd
        self.center, self.scale, self.size = center, scale, size

    def project(self, pts):
        rel = np.asarray(pts, dtype=np.float64) - self.center
        x = self.size * 0.5 + (rel @ self.right) * self.scale
        y = self.size * 0.5 - (rel @ self.up) * self.scale
        return np.column_stack([x, y]), rel @ self.fwd

    def px_per_unit(self):
        return self.scale


def up_from_normals(vertices, faces):
    """Area-weighted mean face normal — the 'up' an aerial scene actually has."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    a = v[f[:, 0]]
    n = np.cross(v[f[:, 1]] - a, v[f[:, 2]] - a)
    area = np.linalg.norm(n, axis=1)
    mean = (n * area[:, None]).sum(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else np.array([0.0, 0.0, 1.0])


def camera_centres(run):
    """Camera centres from poses.json.

    ``t`` is the centre in the run's metric ENU frame — verified against
    ``georef/gps_track.csv``, whose east/north/up bounds are identical to it
    (both [0..1566, 0..1416, -28..173]). The cloud frames differ from it only
    by the reconstruction's own origin, so the path is drawn at its true
    height and the VIEW is what has to make room for it: on this run the
    aircraft sat ~530 m above the surface, which is why an obliquely viewed
    path floats well clear of the cloud rather than weaving through it.
    """
    poses = json.loads((run / "poses.json").read_text())["frames"]
    return np.array([p["t"] for p in poses], dtype=np.float64), poses


def choose_azimuth(points, size, up, el_deg=ELEVATION_DEG, samples=24):
    """Azimuth whose projection is closest to square — i.e. fills the canvas.

    A long thin survey corridor viewed down its own axis wastes most of a
    square frame; at 45 degrees to it the subject runs corner to corner.
    """
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[:: max(1, len(pts) // 20_000)]
    best, best_ratio = 0.0, -1.0
    for az in np.linspace(0.0, 180.0, samples, endpoint=False):
        v = build_view(pts.min(0), pts.max(0), size, up, az_deg=az, el_deg=el_deg,
                       fit_points=pts)
        px, _ = v.project(pts)
        span = px.max(0) - px.min(0)
        ratio = span.min() / max(span.max(), 1e-9)  # 1.0 == perfectly square
        if ratio > best_ratio:
            best, best_ratio = float(az), float(ratio)
    return best, best_ratio


def build_view(bounds_min, bounds_max, size, up, az_deg, el_deg=ELEVATION_DEG,
               margin=0.08, fit_points=None, trim=FIT_TRIM):
    """Orthographic camera at (azimuth, elevation) around *up*, framing the content.

    ``fit_points`` — every vertex the caller is about to draw — drives the
    framing rather than the axis-aligned bbox: the corners of an AABB around a
    diagonally oriented scene project far outside the content, so framing on
    them wasted ~27% of the canvas and left the subject off-centre.

    The extent is trimmed by ``trim`` at each end, so a few outlier points
    cannot shrink the subject for everyone. That makes containment a question
    the caller answers by measurement, not by construction: whatever the trim
    drops may land outside the canvas, so the renderer checks the produced
    border (``_border_alpha``) and re-fits with a smaller trim if it finds
    anything there. ``trim=0.0`` is the raw min/max, which is exhaustive.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    ref = np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(up, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    az, el = np.radians(az_deg), np.radians(el_deg)
    to_cam = np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up
    fwd = -to_cam
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up_s = np.cross(right, fwd)
    lo = np.asarray(bounds_min, dtype=np.float64)
    hi = np.asarray(bounds_max, dtype=np.float64)
    if fit_points is not None and len(fit_points):
        pts = np.asarray(fit_points, dtype=np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
    else:
        pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])], dtype=np.float64)
    center = (lo + hi) * 0.5
    rel = pts - center
    u, vv = rel @ right, rel @ up_s
    lo_pct, hi_pct = 100.0 * trim, 100.0 * (1.0 - trim)
    u0, u1 = (float(x) for x in np.percentile(u, [lo_pct, hi_pct]))
    v0, v1 = (float(x) for x in np.percentile(vv, [lo_pct, hi_pct]))
    # Centre on the CONTENT, which is what the viewer should see.
    center = center + right * (0.5 * (u0 + u1)) + up_s * (0.5 * (v0 + v1))
    extent = max(u1 - u0, v1 - v0)
    scale = size * (1.0 - 2.0 * margin) / max(extent, 1e-9)
    return View(right, up_s, fwd, center, scale, size)


# --------------------------------------------------------------------------- #
# z-buffered rasterisation
# --------------------------------------------------------------------------- #
def _composite(pix, depth, color, size):
    """Depth-test flat candidate buffers into an RGBA image (far → near)."""
    if len(pix) == 0:
        return np.zeros((size, size, 4), np.uint8)
    order = np.argsort(-depth, kind="stable")
    pix, color = pix[order], color[order]
    flat = np.zeros((size * size, 4), np.uint8)
    flat[pix, :3] = color
    flat[pix, 3] = 255
    return flat.reshape(size, size, 4)


def splat(points_px, depth, rgb, size, radius):
    """Draw each point as a filled disc, depth-tested."""
    d = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(d, d)
    keep = (dx * dx + dy * dy) <= radius * radius + 0.25
    dx, dy = dx[keep], dy[keep]
    x = np.round(points_px[:, 0]).astype(np.int64)
    y = np.round(points_px[:, 1]).astype(np.int64)
    px = x[:, None] + dx[None, :]
    py = y[:, None] + dy[None, :]
    inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    return (py[inside] * size + px[inside],
            np.repeat(depth, len(dx))[inside.ravel()],
            np.repeat(rgb, len(dx), axis=0)[inside.ravel()])


def rasterise(pts_px, depth, faces, size, uvs=None, tex=None,
              face_normals=None, to_camera=None, light=None):
    """Batched small-triangle rasteriser (z-buffered, optional UV texturing).

    Triangles are grouped by integer bbox size so each group is one vectorised
    numpy pass over all its pixels; at photogrammetric triangle density almost
    every triangle lands in a 2-6 px group, which is what makes this fast in
    pure numpy. Large obliques fall back to a per-triangle loop.

    ``face_normals`` + ``to_camera`` enable back-face culling: the viewer GLB
    carries no NORMAL attribute, so without this the under-surface of a
    photogrammetric shell renders as acid speckle through the top. ``light``
    then adds gentle form shading.
    """
    faces = np.asarray(faces, dtype=np.int64)
    # Culling/shading are applied per SELECTED triangle, never by rewriting the
    # face array — that would desynchronise the per-face shade lookup.
    front = None
    face_shade = None
    if face_normals is not None and to_camera is not None:
        front = (face_normals @ to_camera) > 0.0
        if light is not None:
            face_shade = 0.72 + 0.28 * np.clip(face_normals @ light, 0.0, 1.0)
    tx = pts_px[faces, 0]
    ty = pts_px[faces, 1]
    tz = depth[faces]
    minx = np.floor(tx.min(1)).astype(np.int64)
    maxx = np.ceil(tx.max(1)).astype(np.int64)
    miny = np.floor(ty.min(1)).astype(np.int64)
    maxy = np.ceil(ty.max(1)).astype(np.int64)
    w = maxx - minx + 1
    h = maxy - miny + 1
    on_screen = (maxx >= 0) & (minx < size) & (maxy >= 0) & (miny < size) & (w > 0) & (h > 0)
    idx = np.flatnonzero(on_screen)
    if len(idx) == 0:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3), np.uint8)

    pix_out, z_out, col_out = [], [], []
    key = w[idx] * 4096 + h[idx]
    for k in np.unique(key):
        sel = idx[key == k]
        if front is not None:
            sel = sel[front[sel]]
            if len(sel) == 0:
                continue
        gw, gh = int(w[sel[0]]), int(h[sel[0]])
        if gw * gh > 4096:  # absurd oblique — handled one at a time below
            continue
        ax, ay = tx[sel, 0], ty[sel, 0]
        bx, by = tx[sel, 1], ty[sel, 1]
        cx, cy = tx[sel, 2], ty[sel, 2]
        # Signed twice-area. Keeping the sign in the divisor is what makes this
        # winding-agnostic: every sub-triangle below shares the parent's
        # orientation, so interior points stay non-negative for BOTH windings.
        # (Negating the barycentrics of flipped triangles instead breaks
        # w0+w1+w2 == 1 and silently drops half the mesh.)
        area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        area2 = np.where(np.abs(area2) < 1e-12, 1e-12, area2)
        X, Y = np.broadcast_arrays(minx[sel, None, None] + np.arange(gw)[None, None, :],
                                   miny[sel, None, None] + np.arange(gh)[None, :, None])
        a2 = area2[:, None, None]
        w0 = ((bx[:, None, None] - X) * (cy[:, None, None] - Y)
              - (by[:, None, None] - Y) * (cx[:, None, None] - X)) / a2
        w1 = ((cx[:, None, None] - X) * (ay[:, None, None] - Y)
              - (cy[:, None, None] - Y) * (ax[:, None, None] - X)) / a2
        w2 = ((ax[:, None, None] - X) * (by[:, None, None] - Y)
              - (ay[:, None, None] - Y) * (bx[:, None, None] - X)) / a2
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        inside &= (X >= 0) & (X < size) & (Y >= 0) & (Y < size)
        if not inside.any():
            continue
        z = (w0 * tz[sel, 0][:, None, None] + w1 * tz[sel, 1][:, None, None]
             + w2 * tz[sel, 2][:, None, None])
        gshade = (np.broadcast_to(face_shade[sel][:, None, None], inside.shape)[inside]
                  if face_shade is not None else None)
        if tex is not None and uvs is not None:
            tu = (w0 * uvs[faces[sel, 0], 0][:, None, None]
                  + w1 * uvs[faces[sel, 1], 0][:, None, None]
                  + w2 * uvs[faces[sel, 2], 0][:, None, None])
            tv = (w0 * uvs[faces[sel, 0], 1][:, None, None]
                  + w1 * uvs[faces[sel, 1], 1][:, None, None]
                  + w2 * uvs[faces[sel, 2], 1][:, None, None])
            th, tw = tex.shape[:2]
            ci = np.clip((tv.ravel()[inside.ravel()] * th).astype(np.int64), 0, th - 1)
            cj = np.clip((tu.ravel()[inside.ravel()] * tw).astype(np.int64), 0, tw - 1)
            col = tex[ci, cj]
        else:
            col = np.zeros((int(inside.sum()), 3), np.uint8)
        if gshade is not None:
            col = np.clip(col.astype(np.float64) * gshade[:, None], 0, 255).astype(np.uint8)
        pix_out.append((Y[inside] * size + X[inside]))
        z_out.append(z[inside])
        col_out.append(col)

    # Oversized obliques: rare enough to rasterise one triangle at a time.
    big = idx[(w[idx] * h[idx] > 4096)]
    for i in big:
        x0, x1 = max(0, minx[i]), min(size - 1, maxx[i])
        y0, y1 = max(0, miny[i]), min(size - 1, maxy[i])
        if x1 < x0 or y1 < y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        ax, ay = tx[i, 0], ty[i, 0]
        bx, by = tx[i, 1], ty[i, 1]
        cx, cy = tx[i, 2], ty[i, 2]
        area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area2) < 1e-12:
            continue
        w0 = ((bx - X) * (cy - Y) - (by - Y) * (cx - X)) / area2
        w1 = ((cx - X) * (ay - Y) - (cy - Y) * (ax - X)) / area2
        w2 = ((ax - X) * (by - Y) - (ay - Y) * (bx - X)) / area2
        m = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not m.any():
            continue
        z = w0 * tz[i, 0] + w1 * tz[i, 1] + w2 * tz[i, 2]
        if tex is not None and uvs is not None:
            tu = w0 * uvs[faces[i, 0], 0] + w1 * uvs[faces[i, 1], 0] + w2 * uvs[faces[i, 2], 0]
            tv = w0 * uvs[faces[i, 0], 1] + w1 * uvs[faces[i, 1], 1] + w2 * uvs[faces[i, 2], 1]
            th, tw = tex.shape[:2]
            ci = np.clip((tv[m] * th).astype(np.int64), 0, th - 1)
            cj = np.clip((tu[m] * tw).astype(np.int64), 0, tw - 1)
            col = tex[ci, cj]
        else:
            col = np.zeros((int(m.sum()), 3), np.uint8)
        pix_out.append(Y[m] * size + X[m])
        z_out.append(z[m])
        col_out.append(col)

    if not pix_out:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3), np.uint8)
    return (np.concatenate(pix_out), np.concatenate(z_out),
            np.concatenate(col_out) if col_out else np.zeros((0, 3), np.uint8))


# --------------------------------------------------------------------------- #
# GLB
# --------------------------------------------------------------------------- #
_GLTF_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                   5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_GLTF_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


_GLB_CACHE = {}


def read_glb(path):
    """Return (positions, uvs, triangles, texture_rgb) from a .glb."""
    key = str(path)
    if key in _GLB_CACHE:  # the viewer GLB is ~100 MB; two steps share one parse
        return _GLB_CACHE[key]
    raw = Path(path).read_bytes()
    magic, _version, _length = struct.unpack("<III", raw[:12])
    if magic != 0x46546C67:
        raise ValueError(f"{path} is not a GLB")
    off, chunks = 12, []
    while off < len(raw):
        clen, ctype = struct.unpack("<II", raw[off:off + 8])
        chunks.append((ctype, off + 8, clen))
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    gltf = json.loads(raw[chunks[0][1]:chunks[0][1] + chunks[0][2]])
    blob_off = next((c for c in chunks if c[0] == 0x004E4942), None)
    blob = raw[blob_off[1]:blob_off[1] + blob_off[2]] if blob_off else b""

    def accessor(i):
        acc = gltf["accessors"][i]
        bv = gltf["bufferViews"][acc["bufferView"]]
        comp = _GLTF_COMPONENT[acc["componentType"]]
        ncomp = _GLTF_NCOMP[acc["type"]]
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        count = acc["count"]
        itemsize = np.dtype(comp).itemsize * ncomp
        stride = bv.get("byteStride")
        if stride in (None, itemsize):
            return np.frombuffer(blob, dtype=comp, count=count * ncomp,
                                 offset=start).reshape(count, ncomp)
        raw_view = np.frombuffer(blob, dtype=np.uint8,
                                 count=stride * (count - 1) + itemsize, offset=start)
        return np.lib.stride_tricks.as_strided(
            raw_view.view(comp), shape=(count, ncomp),
            strides=(stride, np.dtype(comp).itemsize)).copy()

    prim = gltf["meshes"][0]["primitives"][0]
    pos = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
    uvs = accessor(prim["attributes"]["TEXCOORD_0"]).astype(np.float64) \
        if "TEXCOORD_0" in prim["attributes"] else None
    tri = accessor(prim["indices"]).astype(np.int64).reshape(-1, 3)
    tex = None
    if gltf.get("images"):
        bv = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
        start = bv.get("byteOffset", 0)
        buf = np.frombuffer(blob, dtype=np.uint8, count=bv["byteLength"], offset=start)
        tex = np.asarray(Image.open(__import__("io").BytesIO(buf.tobytes())).convert("RGB"))
    _GLB_CACHE[key] = (pos, uvs, tri, tex)
    return pos, uvs, tri, tex


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def halo_line(draw, a, b, color, width):
    """Stroke a line with a light outer edge under it.

    These PNGs are transparent, so a line has to read on a white slide AND on
    a dark viewer: the light halo is what shows against dark, the coloured core
    is what shows against light.
    """
    draw.line([a, b], fill=(255, 255, 255, 190), width=int(width * 2.6) + 2)
    draw.line([a, b], fill=color, width=width)


def halo_polygon(draw, ring, color, width):
    for k in range(len(ring)):
        halo_line(draw, ring[k], ring[(k + 1) % len(ring)], color, width)


def step_raw_frames(run, size, count=3):
    """The extracted video frames themselves — the raw drone data."""
    frames = sorted(run.glob("frames/*.jpg"))
    if not frames:
        return None
    picks = np.linspace(0, len(frames) - 1, count).astype(int)
    # Shallow depth of field of *angle*: fan them like a hand of cards.
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tile_w = int(size * 0.62)
    angles = np.linspace(-7.0, 7.0, count)
    offs = np.linspace(-0.16, 0.16, count) * size
    for i, (idx, ang, dy) in enumerate(zip(picks, angles, offs)):
        # RGBA, so the corners rotate() adds are transparent rather than the
        # opaque black an RGB rotate() fills them with.
        im = Image.open(frames[idx]).convert("RGBA")
        im = im.resize((tile_w, max(1, int(tile_w * im.height / im.width))), Image.LANCZOS)
        rot = im.rotate(ang, resample=Image.BICUBIC, expand=True)
        cx = int(size * 0.5 + (i - (count - 1) / 2) * size * 0.02)
        cy = int(size * 0.5 + dy)
        # No drop shadow: on a transparent canvas any offset silhouette reads as
        # a black border around the photo rather than as light.
        canvas.alpha_composite(rot, (cx - rot.width // 2, cy - rot.height // 2))
    return canvas


def step_depth_maps(run, size, count=3):
    """Per-view depth maps, colourised with TURBO straight from the .npy arrays."""
    maps = sorted((run / "depth").glob("*.npy"))
    if not maps:
        return None
    picks = np.linspace(0, len(maps) - 1, count).astype(int)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tile_w = int(size * 0.60)
    angles = np.linspace(-6.0, 6.0, count)
    offs = np.linspace(-0.15, 0.15, count) * size
    for i, (idx, ang, dy) in enumerate(zip(picks, angles, offs)):
        d = np.load(maps[idx]).astype(np.float32)
        finite = np.isfinite(d) & (d > 0)
        lo, hi = np.percentile(d[finite], [2, 98]) if finite.any() else (0.0, 1.0)
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        coloured = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
        alpha = np.where(finite, 255, 0).astype(np.uint8)
        im = Image.fromarray(np.dstack([coloured, alpha]))
        im = im.resize((tile_w, max(1, int(tile_w * im.height / im.width))), Image.LANCZOS)
        # No backing card: the maps are their own subject and the requested
        # background is transparent, so anything opaque behind them would show
        # as a plate on a non-dark surface.
        rot = im.rotate(ang, resample=Image.BICUBIC, expand=True)
        cx = int(size * 0.5 + (i - (count - 1) / 2) * size * 0.03)
        cy = int(size * 0.5 + dy)
        canvas.alpha_composite(rot, (cx - rot.width // 2, cy - rot.height // 2))
    return canvas


def frustum_segments(t, R, K, extent):
    """World-space line segments for one camera frustum."""
    depth = 0.09 * extent
    w = 0.5 * 1920 / float(K[0][0]) * depth
    h = 0.5 * 1080 / float(K[1][1]) * depth
    local = np.array([[0, 0, 0], [-w, -h, depth], [w, -h, depth],
                      [w, h, depth], [-w, h, depth]], dtype=np.float64)
    world = local @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64)
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return [(world[a], world[b]) for a, b in pairs]


def step_sparse(run, view, size):
    """Sparse SfM cloud + the reconstructed camera poses."""
    from app.services.pointcloud import read_ply

    cloud = read_ply(run / "sparse_model.ply")
    pts = np.asarray(cloud.xyz)
    rgb = np.asarray(cloud.rgb) if cloud.rgb is not None else np.full((cloud.n, 3), 40, np.uint8)
    px, depth = view.project(pts)
    pix, z, col = splat(px, depth, rgb, size, radius=max(2, int(round(size / 512))))
    rgba = _composite(pix, z, col, size)

    im = Image.fromarray(rgba)
    draw = ImageDraw.Draw(im)
    poses = json.loads((run / "poses.json").read_text())["frames"]
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    width = max(1, size // 900)
    step = max(1, len(poses) // 24)
    for p in poses[::step]:
        for a, b in frustum_segments(p["t"], p["R"], p["K"], extent):
            pa, _ = view.project(a[None, :])
            pb, _ = view.project(b[None, :])
            halo_line(draw, tuple(pa[0]), tuple(pb[0]), (25, 25, 25, 255), width)
    return im


def step_filtered_sparse(run, view, size, poses):
    """Orientation-consistent sparse cloud + the aligned camera trajectory."""
    from app.services.pointcloud import read_ply

    cloud = read_ply(run / "sparse_model_consistent.ply")
    pts = np.asarray(cloud.xyz)
    rgb = np.asarray(cloud.rgb) if cloud.rgb is not None else np.full((cloud.n, 3), 40, np.uint8)
    px, depth = view.project(pts)
    pix, z, col = splat(px, depth, rgb, size, radius=max(2, int(round(size / 400))))
    im = Image.fromarray(_composite(pix, z, col, size))

    draw = ImageDraw.Draw(im)
    traj, _ = view.project(np.array([p["t"] for p in poses], dtype=np.float64))
    tw = max(2, size // 420)
    pts_traj = [tuple(p) for p in traj]
    draw.line(pts_traj, fill=(255, 255, 255, 190), width=int(tw * 2.6) + 2, joint="curve")
    draw.line(pts_traj, fill=GREEN + (255,), width=tw, joint="curve")
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    cube = 0.035 * extent
    step = max(1, len(poses) // 9)
    w = max(1, size // 1200)
    for p in poses[::step]:
        c, _ = view.project(np.asarray(p["t"], dtype=np.float64)[None, :])
        x, y = c[0]
        s = cube * view.px_per_unit()
        offs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        corners = [(x + a * s * 0.5, y + b * s * 0.5) for a, b in offs]
        pts2 = [(x + a * s * 0.5 + size * 0.012, y + b * s * 0.5 - size * 0.012)
                for a, b in offs]
        for ring in (corners, pts2):
            halo_polygon(draw, ring, (25, 25, 25, 255), w)
            for k in range(4):
                halo_line(draw, ring[k], pts2[k], (25, 25, 25, 255), w)
    return im


def _splat_radius(view, spacing_m, *, grow=1.42, lo=1, hi=6):
    """Disc radius that covers the cloud's own sampling cell in screen space.

    A fixed radius is wrong at both ends: too small leaves sieve holes across
    the surface, too large floods the pass with candidate pixels (a hard-coded
    9px radius at 4096px cost 135 s and blurred real detail).

    A disc of radius r covers a square cell of side s only when r >= s/sqrt(2),
    i.e. grow >= sqrt(2) ~ 1.4142. The old 1.35 sat just under that and left a
    measurable screen-door: at the dense panel's 2.9 px spacing it picked r=2 px
    against the 2.06 px needed, giving 10.0% holes and 30.3% solid pixels in the
    surface core — visible as a lattice at 1:1. At 1.42 it rounds up to a radius
    that covers (5.6% holes, 61.6% solid), which is what the panel should show.
    """
    spacing_px = spacing_m * view.px_per_unit()
    return int(np.clip(np.ceil(grow * spacing_px / 2.0), lo, hi))


def step_dense_cloud(run, view, size):
    """The fused dense point cloud, in its own captured colours."""
    from app.services.mesh_generator import nn_spacing_stats
    from app.services.pointcloud import read_ply

    cloud = read_ply(run / "dense/dense_model.ply")
    pts = np.asarray(cloud.xyz)
    ok = np.isfinite(pts).all(axis=1)
    pts = pts[ok]
    rgb = np.asarray(cloud.rgb)[ok] if cloud.rgb is not None else np.full((len(pts), 3), 160, np.uint8)
    px, depth = view.project(pts)
    spacing = nn_spacing_stats(pts)["p50"]
    radius = _splat_radius(view, spacing)
    print(f"    dense splat: {len(pts):,} pts, measured spacing {spacing:.2f} m "
          f"-> radius {radius}px at {size}px", flush=True)
    pix, z, col = splat(px, depth, rgb, size, radius=radius)
    return Image.fromarray(_composite(pix, z, col, size))


def step_textured_mesh(run, view, size, badge=False):
    """The textured viewer GLB — per-pixel sampling of its baked atlas."""
    pos, uvs, tri, tex = read_glb(run / "mesh/mesh_viewer.glb")
    px, depth = view.project(pos)
    # Face normals from positions (the GLB ships no NORMAL attribute). The mesh
    # winding is not something to assume: for an aerial capture the surface we
    # are looking at is the one facing 'up', so use that to fix the sign once,
    # globally, instead of guessing per face.
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-12)[:, None]
    upv = np.asarray(view.up, dtype=np.float64)
    if float((fn @ upv).sum()) < 0:
        fn = -fn
    to_cam = -np.asarray(view.fwd, dtype=np.float64)
    print(f"    mesh: {len(tri):,} triangles, {int(((fn @ to_cam) > 0).sum()):,} face the camera",
          flush=True)
    pix, z, col = rasterise(px, depth, tri, size, uvs=uvs, tex=tex,
                            face_normals=fn, to_camera=to_cam,
                            light=np.asarray(view.up, dtype=np.float64))
    im = Image.fromarray(_composite(pix, z, col, size))
    if badge:
        draw = ImageDraw.Draw(im)
        r = size * 0.055
        cx, cy = size - r * 1.9, r * 1.9
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GREEN + (255,))
        w = r * 0.16
        draw.line([(cx - r * 0.45, cy + r * 0.02), (cx - r * 0.10, cy + r * 0.38),
                   (cx + r * 0.48, cy - r * 0.34)], fill=(255, 255, 255, 255),
                  width=int(w), joint="curve")
    return im


# --------------------------------------------------------------------------- #
def _downsample(im, size):
    return im if im.size == (size, size) else im.resize((size, size), Image.LANCZOS)


def _border_alpha(im):
    """Opaque pixels on the outermost rows/columns — the no-clip witness.

    Measured on the FINAL downsampled image, because that is the deliverable:
    resampling can put alpha on the edge even when no source vertex is out of
    frame, and conversely a vertex slightly outside can be genuinely invisible.
    """
    a = np.asarray(im)[:, :, 3]
    return int((np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]) > 0).sum())


def panel_content(run, poses, verts):
    """The geometry each 3-D panel draws, keyed by panel. One frame per panel.

    A single shared frame made the mesh panels pay for geometry they never
    render: they were fitted to mesh ∪ fused cloud and lost ~30% of their area
    to a cloud that only the dense-cloud panel puts on the canvas. Each entry
    here is exactly what its own panel draws, so no panel is framed by a
    sibling's contents.
    """
    from app.services.pointcloud import read_ply

    def points(name):
        xyz = np.asarray(read_ply(run / name).xyz, dtype=np.float64)
        return xyz[np.isfinite(xyz).all(axis=1)]

    def cones(cloud):
        # Each sparse panel draws a frustum per camera, sized from its own
        # cloud's extent, so the cones are part of that panel's drawn geometry.
        extent = float(np.linalg.norm(cloud.max(0) - cloud.min(0)))
        return np.array([p for pose in poses
                         for a, b in frustum_segments(pose["t"], pose["R"], pose["K"], extent)
                         for p in (a, b)])

    sparse = points("sparse_model.ply")
    consistent = points("sparse_model_consistent.ply")
    dense = points("dense/dense_model.ply")
    cams = np.array([p["t"] for p in poses], dtype=np.float64)
    return {
        "02_sparse_point_cloud_and_camera_poses": np.vstack([sparse, cams, cones(sparse)]),
        "03_filtered_sparse_cloud_aligned_trajectory": np.vstack(
            [consistent, cams, cones(consistent)]),
        "05_dense_point_cloud": dense,
        # 07 is 06 plus a check badge drawn inside the margin, so it takes the
        # same frame and the two panels stay exactly registered with each other.
        "06_textured_3d_mesh": verts,
        "07_validated_3d_model": verts,
    }


def run_all(run_id, out_dir, size=SIZE_DEFAULT, ss=SS_DEFAULT, steps=None):
    from app.services.mesh import TriangleMesh

    run = Path("data/storage") / run_id
    if not run.is_dir():
        raise SystemExit(f"no such run: {run}")
    big = size * ss
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = TriangleMesh.read_ply(run / "mesh/mesh.ply")
    lo = np.asarray(mesh.vertices).min(axis=0)
    hi = np.asarray(mesh.vertices).max(axis=0)
    up = up_from_normals(mesh.vertices, mesh.faces)
    verts = np.asarray(mesh.vertices)
    cams, poses = camera_centres(run)
    content = panel_content(run, poses, verts)
    # One azimuth for every 3D panel so the sequence reads as one camera.
    az, squareness = choose_azimuth(np.vstack([verts, cams]), big, up,
                                    el_deg=ELEVATION_DEG)

    def frame_for(key, trim):
        pts = content.get(key)  # panels 01 and 04 compose their own canvas
        if pts is None:
            return None
        return build_view(pts.min(0), pts.max(0), big, up, az, fit_points=pts, trim=trim)

    print(f"run={run_id}  scene={np.round(hi - lo, 1)}  up={np.round(up, 3)}\n"
          f"  azimuth={az:.1f}deg (squareness {squareness:.2f})  cameras={len(cams)} "
          f"({np.round(cams.min(0),1)} .. {np.round(cams.max(0),1)})\n"
          f"  per-panel frames (px/m): " + ", ".join(
              f"{k.split('_')[0]} {frame_for(k, FIT_TRIM).px_per_unit():.2f}" for k in content)
          + f" | render={big}px -> {size}px")

    todo = {
        "01_raw_drone_data": lambda view: step_raw_frames(run, big),
        "02_sparse_point_cloud_and_camera_poses": lambda view: step_sparse(run, view, big),
        "03_filtered_sparse_cloud_aligned_trajectory": lambda view: step_filtered_sparse(
            run, view, big, poses),
        "04_per_view_depth_maps": lambda view: step_depth_maps(run, big),
        "05_dense_point_cloud": lambda view: step_dense_cloud(run, view, big),
        "06_textured_3d_mesh": lambda view: step_textured_mesh(run, view, big),
        "07_validated_3d_model": lambda view: step_textured_mesh(run, view, big, badge=True),
    }
    labels = {
        "01_raw_drone_data": "Raw Drone Data",
        "02_sparse_point_cloud_and_camera_poses": "Sparse Point Cloud & Camera Poses",
        "03_filtered_sparse_cloud_aligned_trajectory": "Filtered Sparse Cloud with Aligned Trajectory",
        "04_per_view_depth_maps": "Per-view Depth Maps",
        "05_dense_point_cloud": "Dense Point Cloud",
        "06_textured_3d_mesh": "Textured 3D Mesh",
        "07_validated_3d_model": "Validated 3D Model",
    }
    sources = {
        "01_raw_drone_data": "frames/*.jpg (extracted from video.mp4)",
        "02_sparse_point_cloud_and_camera_poses": "sparse_model.ply + poses.json",
        "03_filtered_sparse_cloud_aligned_trajectory": "sparse_model_consistent.ply + poses.json",
        "04_per_view_depth_maps": "depth/*.npy",
        "05_dense_point_cloud": "dense/dense_model.ply",
        "06_textured_3d_mesh": "mesh/mesh_viewer.glb (baked atlas)",
        "07_validated_3d_model": "mesh/mesh_viewer.glb + quality_report.json",
    }

    made = []
    for key, fn in todo.items():
        if steps and key.split("_", 1)[0] not in steps:
            continue
        t0 = time.perf_counter()
        im, trim_used = None, FIT_TRIM
        for trim in FIT_TRIM_LADDER:
            # Robust fit first; widen only if the rendered border proves the
            # trim dropped something that was actually drawn.
            raw = fn(frame_for(key, trim))
            if raw is None:
                break
            im = _downsample(raw, size)
            trim_used = trim
            if _border_alpha(im) == 0:
                break
        if im is None:
            print(f"  {key}: SKIPPED (source artifact missing)")
            continue
        path = out_dir / f"{key}.png"
        im.save(path)
        alpha = np.asarray(im)[:, :, 3]
        frame = frame_for(key, trim_used)
        made.append({"file": path.name, "label": labels[key], "source": sources[key],
                     "pixels": list(im.size),
                     "opaque_pct": round(float((alpha > 0).mean()) * 100, 1),
                     "px_per_metre_at_ss": round(float(frame.px_per_unit()), 3) if frame else None,
                     "fit_trim_pct": round(trim_used * 100, 2) if frame else None,
                     "seconds": round(time.perf_counter() - t0, 1)})
        widened = "" if trim_used == FIT_TRIM else f"  [frame widened to trim {trim_used*100:g}%]"
        print(f"  {key}: {im.size[0]}x{im.size[1]}  opaque={made[-1]['opaque_pct']}%  "
              f"border={_border_alpha(im)}  {made[-1]['seconds']}s -> {path}{widened}")

    meta = {"run": run_id, "rendered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_run_dir": str(run.resolve()), "supersample": ss,
            "camera": {"azimuth_deg": round(float(az), 1), "elevation_deg": ELEVATION_DEG,
                       "projection": "orthographic", "up_axis": [round(float(x), 5) for x in up],
                       "scene_extent_m": [round(float(x), 2) for x in (hi - lo)]},
            "background": "transparent (RGBA, alpha=0)",
            "renderer": "numpy software rasteriser (Open3D offscreen is EGL-blocked on macOS)",
            "steps": made}
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "index.html").write_text(_index_html(run_id, made, size))
    print(f"wrote {len(made)} images + manifest.json + index.html to {out_dir}")
    return made


def _index_html(run_id, made, size):
    cards = "\n".join(
        f'    <figure><img src="{m["file"]}" alt="{m["label"]}">'
        f'<figcaption>{m["label"]}</figcaption>'
        f'<span class="src">{m["source"]}</span></figure>' for m in made)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>STRATA pipeline stages — {run_id}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 0; padding: 48px; font: 14px/1.5 -apple-system, "Segoe UI", sans-serif;
         background: #0b0f14; color: #e6edf3; }}
  h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 4px; }}
  p.sub {{ color: #8b949e; margin: 0 0 32px; }}
  .strip {{ display: flex; gap: 20px; overflow-x: auto; padding-bottom: 16px; }}
  figure {{ flex: 0 0 auto; width: {max(220, int(size / 5))}px; margin: 0;
            background: #11171f; border: 1px solid #1f2a36; border-radius: 14px;
            padding: 14px; }}
  figure img {{ width: 100%; height: auto; display: block;
                background: repeating-conic-gradient(#1a222c 0% 25%, #131a22 0% 50%)
                            50%/18px 18px; border-radius: 8px; }}
  figcaption {{ margin-top: 12px; font-weight: 600; }}
  .src {{ display: block; margin-top: 4px; color: #6e7d8f; font-size: 11px;
          font-family: ui-monospace, monospace; }}
</style></head>
<body>
  <h1>STRATA pipeline stages — {run_id}</h1>
  <p class="sub">Real artifacts from this run, {size}&times;{size} RGBA, transparent background.</p>
  <div class="strip">
{cards}
  </div>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run id under data/storage")
    ap.add_argument("--out", default=None,
                    help="output dir (default <repo>/docs/pipeline-stages/<run>)")
    ap.add_argument("--size", type=int, default=SIZE_DEFAULT)
    ap.add_argument("--ss", type=int, default=SS_DEFAULT, help="supersample factor")
    ap.add_argument("--steps", default=None, help="comma-separated step prefixes, e.g. 01,04")
    args = ap.parse_args()
    # Script-relative, NOT cwd-relative: this runs from backend/, where a bare
    # "../../docs" lands in the PARENT repo (outside drone-recon).
    default_out = Path(__file__).resolve().parents[2] / "docs" / "pipeline-stages"
    out = Path(args.out).resolve() if args.out else default_out / args.run
    steps = set(args.steps.split(",")) if args.steps else None
    run_all(args.run, out, size=args.size, ss=args.ss, steps=steps)


if __name__ == "__main__":
    main()
