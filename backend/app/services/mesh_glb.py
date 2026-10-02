"""Viewer LOD: decimated GLB export of the production mesh.

The authoritative reconstruction (mesh.ply) is never sent to the browser —
the viewer loads a quadric-decimated ``mesh_viewer.glb`` instead. Decimation
preserves topology (Open3D quadric-error metric when open3d is available, an
iterative shortest-edge collapse that never tears the surface otherwise),
and generation is part of the dense stage AFTER mesh validation, so the
pipeline never blocks on the browser (Part 6 of the rebuild mandate).

Export is pure numpy + pygltflib: positions, indices and either per-vertex
colors or a TEXCOORD_0 atlas — a visualization representation, not an
analysis artifact. Attributes on a primitive are always written with the
POSITION accessor's count, so a spec-compliant viewer reads the UVs the
writer intended (a per-corner UV payload is re-indexed onto a vertex set
where every vertex carries exactly one UV).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from pygltflib import GLTF2

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.mesh_optimizer import DecimateParams, decimate

log = get_logger("drone_recon.services.mesh_glb")


def viewer_triangle_budget() -> int:
    """Triangle budget for the browser LOD (MESH_VIEWER_TRIANGLE_BUDGET)."""
    return settings.mesh.viewer_triangle_budget


def decimate_for_viewer(mesh: TriangleMesh,
                        target_faces: int | None = None) -> tuple[TriangleMesh, dict]:
    """Decimate *mesh* to the viewer budget (quadric when open3d present)."""
    target = target_faces if target_faces is not None else viewer_triangle_budget()
    stats: dict = {"method": "copy", "before_faces": mesh.m, "after_faces": mesh.m}
    if mesh.m <= target:
        return mesh.copy(), stats
    out, d_stats = decimate(mesh, DecimateParams(target_faces=target))
    stats.update(d_stats)
    return out, stats


def export_viewer_glb(mesh: TriangleMesh, path, target_faces: int | None = None,
                      texture=None, full_mesh: TriangleMesh | None = None) -> dict:
    """Write a decimated binary-glTF viewer LOD; returns a stats dict.

    ``mesh`` is the surface to write — pass an already-decimated LOD when the
    texture payload was baked for it (UV rows index the GLB's own face order,
    so the payload and the written mesh must be the same surface).
    ``full_mesh`` optionally supplies the authoritative mesh for the deviation
    diagnostic when ``mesh`` is that LOD; deviation is lod-vs-full either way.

    ``texture`` is an optional :class:`~app.services.textured_glb.TexturePayload`.
    When given, the primitive carries TEXCOORD_0 + an embedded image material
    and per-vertex colors are omitted (the texture replaces them). A payload
    whose ``image_bytes`` is empty is an honest FALLBACK marker: the GLB is
    written in the classic COLOR_0 vertex-color form and the payload stats
    carry ``texture_fallback_reason`` — never a silent downgrade.
    """
    started = time.perf_counter()
    lod, d_stats = decimate_for_viewer(
        mesh, target_faces if target_faces is not None else viewer_triangle_budget())
    ref = full_mesh if full_mesh is not None else mesh
    # Fallback payload (empty image) → classic COLOR_0 export; its reason
    # still travels in the stats below.
    tex_stats = texture_stats(texture) if texture is not None else {"textured": False}
    if texture is not None and not texture.image_bytes:
        texture = None

    # Deviation between the full artifact and the viewer LOD (Part 5):
    # symmetric vertex-to-vertex NN statistics — the decimated surface must
    # stay within the full mesh's own sampling noise (local spacing), or the
    # LOD no longer represents the reconstruction.
    deviation: dict = {}
    try:
        from scipy.spatial import cKDTree

        full_v = np.asarray(ref.vertices, dtype=np.float64)
        lod_v = np.asarray(lod.vertices, dtype=np.float64)
        d_lod_full = cKDTree(full_v).query(lod_v, k=1)[0]
        sample = min(len(full_v), 50_000)
        full_sub = full_v if len(full_v) <= sample else full_v[
            np.random.default_rng(0).choice(len(full_v), sample, replace=False)]
        d_full_lod = cKDTree(lod_v).query(full_sub, k=1)[0]
        deviation = {
            "lod_to_full_median_m": round(float(np.median(d_lod_full)), 4),
            "lod_to_full_p95_m": round(float(np.percentile(d_lod_full, 95)), 4),
            "full_to_lod_median_m": round(float(np.median(d_full_lod)), 4),
            "full_to_lod_p95_m": round(float(np.percentile(d_full_lod, 95)), 4),
            "hausdorff_max_m": round(float(max(d_lod_full.max(), d_full_lod.max())), 4),
        }
    except Exception as dev_err:  # pragma: no cover - diagnostics only
        log.warning("lod_deviation_failed", error=str(dev_err))

    vertices = np.asarray(lod.vertices, dtype=np.float32)
    faces = np.asarray(lod.faces, dtype=np.uint32).reshape(-1, 3)
    colors = None
    if lod.colors is not None and texture is None:
        colors = np.asarray(lod.colors, dtype=np.float32) / 255.0

    # The UV payload is per face CORNER (each face owns its own atlas tile),
    # while POSITION is per vertex. glTF requires every attribute of a
    # primitive to have the POSITION accessor's count, so the two must be
    # re-indexed onto a shared vertex set before writing — otherwise a
    # compliant viewer reads uv[vertex_index] and samples an unrelated
    # corner's tile, which renders the surface as scrambled dark patches.
    uv_per_vertex: np.ndarray | None = None
    if texture is not None:
        uv_corner = np.asarray(texture.uv, dtype=np.float32).reshape(-1, 2)
        if len(uv_corner) != faces.size:
            raise ValueError(
                f"texture payload covers {len(uv_corner)} face corners but the "
                f"surface has {faces.size} — the UVs do not describe this mesh"
            )
        vertices, faces, uv_per_vertex = _split_vertices_by_uv(vertices, faces, uv_corner)

    # float32 is exact for indices < 2**24 — assert the viewer budget keeps us there.
    assert vertices.max(initial=0.0) < 2**24 and faces.max(initial=0.0) < 2**24

    gltf = GLTF2()
    # Buffer 0: indices+vertices. Buffer 1 (optional): colors. pygltflib
    # packs binary blobs in accessor order, so we assemble explicit blobs.
    import struct

    idx_bytes = faces.astype("<u4").tobytes()
    pos_bytes = vertices.tobytes()
    col_bytes = colors.tobytes() if colors is not None else b""

    blob = bytearray()
    buffer_views: list[dict] = []

    def _add(data: bytes, target=None) -> int:
        offset = (len(blob) + 3) & ~3  # 4-byte alignment
        blob.extend(b"\x00" * (offset - len(blob)))
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
            **({"target": target} if target is not None else {}),
        })
        blob.extend(data)
        return len(buffer_views) - 1

    iv = _add(idx_bytes, target=34963)  # ELEMENT_ARRAY_BUFFER
    pv = _add(pos_bytes, target=34962)  # ARRAY_BUFFER
    cv = _add(col_bytes, target=34962) if colors is not None else None

    # Texture payloads (TEXCOORD_0 + embedded PNG) pack after the geometry.
    image_blob = b""
    uv_view = img_view = None
    if texture is not None:
        image_blob = texture.image_bytes
        assert uv_per_vertex is not None
        uv_view = _add(uv_per_vertex.tobytes(), target=34962)
        img_view = _add(image_blob)

    min_pos, max_pos = vertices.min(axis=0), vertices.max(axis=0)
    accessor_idx = len(gltf.accessors)
    gltf.accessors.extend([
        # 0: indices (SCALAR, uint32)
        _accessor(componentType=5125, count=faces.size, type="SCALAR",
                  bufferView=iv, byte_offset=0),
        # 1: positions (VEC3, float32) with min/max (required by spec for POSITION)
        _accessor(componentType=5126, count=len(vertices), type="VEC3",
                  bufferView=pv, byte_offset=0, min=min_pos.tolist(), max=max_pos.tolist()),
    ])
    if cv is not None:
        gltf.accessors.append(
            _accessor(componentType=5126, count=len(vertices), type="VEC3",
                      bufferView=cv, byte_offset=0))
    if texture is not None:
        assert uv_per_vertex is not None
        gltf.accessors.append(
            _accessor(componentType=5126, count=len(uv_per_vertex), type="VEC2",
                      bufferView=uv_view, byte_offset=0))
        gltf.images = [_image(mimeType=texture.mime_type, bufferView=img_view)]
        gltf.samplers = [_sampler()]
        gltf.textures = [_texture()]

    gltf.bufferViews = [
        _bv(bv) for bv in buffer_views  # type: ignore[misc]
    ]
    gltf.buffers = [_buf(len(blob))]

    material = _material(vertex_colors=colors is not None, texture=texture)
    gltf.materials = [material]
    prim = _primitive(accessor_idx, vertex_colors=colors is not None,
                      textured=texture is not None)
    mesh_gltf = _mesh(prim)
    node = _node(mesh_gltf)
    scene = _scene(node)
    gltf.meshes = [mesh_gltf]
    gltf.nodes = [node]
    gltf.scenes = [scene]
    gltf.scene = 0

    # Pad to a 4-byte boundary before attaching (GLB requires it).
    while len(blob) % 4:
        blob.append(0)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(path))

    stats = {
        "viewer_lod": True,
        "target_faces": target_faces,
        "faces": int(lod.m),
        "vertices": int(lod.n),
        # Written POSITION count; equals the TEXCOORD_0 count when textured
        # (the split adds the seam vertices the per-corner UVs require).
        "glb_vertices": int(len(vertices)),
        # The artifact's real size on disk (the image blob lives INSIDE the
        # packed blob, so summing them double-counted it).
        "bytes": int(Path(path).stat().st_size),
        "deviation": deviation,
        **tex_stats,
        **d_stats,
        "took_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    log.info("viewer_glb_written", path=str(path), faces=stats["faces"], took_ms=stats["took_ms"])
    return stats


def _split_vertices_by_uv(vertices: np.ndarray, faces: np.ndarray,
                         uv_corner: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-index a mesh so every vertex carries exactly one UV.

    A per-face atlas gives each of a triangle's three corners its own UV, so
    corners that share a position still need distinct vertices — the unique
    ``(vertex, u, v)`` set is the smallest valid indexed form. Exact float32
    bit patterns key the dedupe (no tolerance, so no UV drift).
    """
    corner_vertex = faces.reshape(-1)
    key = np.empty(len(corner_vertex),
                   dtype=[("v", np.int64), ("u", np.float32), ("w", np.float32)])
    key["v"] = corner_vertex
    key["u"] = uv_corner[:, 0]
    key["w"] = uv_corner[:, 1]
    unique, inverse = np.unique(key, return_inverse=True)
    new_faces = inverse.reshape(-1, 3).astype(np.uint32)
    new_vertices = vertices[unique["v"]]
    new_uv = np.stack([unique["u"], unique["w"]], axis=1).astype(np.float32)
    return new_vertices, new_faces, new_uv


# ---------------------------------------------------------------------------
# pygltflib constructors (kept tiny; the library's objects are verbose)
# ---------------------------------------------------------------------------

def _accessor(*, componentType: int, count: int, type: str, bufferView: int,
              byte_offset: int, min=None, max=None):
    from pygltflib import Accessor

    kwargs = {}
    if min is not None:
        kwargs["min"] = min
    if max is not None:
        kwargs["max"] = max
    return Accessor(componentType=componentType, count=count, type=type,
                    bufferView=bufferView, byteOffset=byte_offset, **kwargs)


def _bv(bv: dict):
    from pygltflib import BufferView

    return BufferView(buffer=bv["buffer"], byteOffset=bv["byteOffset"],
                      byteLength=bv["byteLength"], target=bv.get("target"))


def _buf(byte_length: int):
    from pygltflib import Buffer

    return Buffer(byteLength=byte_length)


def texture_stats(texture) -> dict:
    """Honest texturing facts for the stage payload.

    A payload with an empty image is the vertex-color fallback: it reports
    ``textured: False`` PLUS its ``texture_fallback_reason`` — the downgrade
    is visible, never silent."""
    if texture is None or not texture.image_bytes:
        out = {"textured": False}
    else:
        out = {"textured": True, "texture_source": texture.source}
    if texture is not None:
        if texture.fallback_reason:
            out["texture_fallback_reason"] = texture.fallback_reason
        if texture.stats:
            out["texture"] = texture.stats
    return out


def _material(vertex_colors: bool, texture=None):
    # Vertex-color path: base color white — COLOR_0 multiplies it (works in
    # three.js and every spec-compliant viewer). Textured path: PBR base
    # color texture over the embedded PNG; no vertex colors are emitted.
    from pygltflib import Material

    if texture is not None:
        from pygltflib import PbrMetallicRoughness, TextureInfo

        mat = Material()
        mat.pbrMetallicRoughness = PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=0, texCoord=0),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.9,
        )
        return mat
    return Material()


def _primitive(base: int, vertex_colors: bool, textured: bool = False):
    from pygltflib import Primitive

    attrs = {"POSITION": base + 1}
    if vertex_colors:
        attrs["COLOR_0"] = base + 2
    if textured:
        # Accessor order: indices, POSITION, [COLOR_0], TEXCOORD_0.
        attrs["TEXCOORD_0"] = base + (3 if vertex_colors else 2)
    return Primitive(attributes=attrs, indices=base, material=0, mode=4)


def _mesh(prim):
    from pygltflib import Mesh

    return Mesh(primitives=[prim])


def _image(mimeType: str, bufferView: int):
    from pygltflib import Image

    return Image(mimeType=mimeType, bufferView=bufferView)


def _sampler():
    from pygltflib import Sampler

    return Sampler(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)


def _texture():
    from pygltflib import Texture

    return Texture(source=0, sampler=0)


def _node(mesh):
    from pygltflib import Node

    return Node(mesh=0)


def _scene(node):
    from pygltflib import Scene

    return Scene(nodes=[0])
