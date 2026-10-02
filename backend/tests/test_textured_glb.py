"""Textured viewer GLB — production texture promotion.

The viewer GLB is a downstream derivative: it must carry TEXCOORD_0 plus an
embedded image (not COLOR_0-only) when the run can be textured, and the
authoritative PLYs must be byte-identical after a textured run. A run that
cannot be textured falls back to COLOR_0 with a NAMED reason in the payload —
never a silent downgrade.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config.settings import settings
from app.services.mesh import TriangleMesh
from app.services.mesh_glb import export_viewer_glb
from app.services.mesh_glb import viewer_triangle_budget
from app.services.textured_glb import (
    _vertex_color_fallback,
    _texture_views,
    build_texture_payload,
)

# ---------------------------------------------------------------------------
# Fixtures: a workspace with poses + a visible image, and a small mesh the
# cameras actually see.
# ---------------------------------------------------------------------------


def _K() -> np.ndarray:
    return np.array([[60.0, 0, 47.5], [0, 60.0, 47.5], [0, 0, 1]])


def _look_down() -> np.ndarray:
    # Camera looks along −z world (down), aligned with world +x right/+y fwd.
    return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _flat_mesh() -> TriangleMesh:
    """One ground quad (2 triangles) at z=0 — normals point up at the cameras."""
    v = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    colors = np.full((4, 3), 127, dtype=np.uint8)
    return TriangleMesh(vertices=v, faces=f, colors=colors)


def _seeded_workspace(tmp_path: Path, *, usable: bool = True) -> Path:
    """A run directory with poses.json, one nadir image, and conditioning info."""
    workspace = tmp_path / "run_tex"
    (workspace / "selected").mkdir(parents=True)
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:, :, 2] = 200  # blue-ish ground the camera sees
    cv2.imwrite(str(workspace / "selected" / "frame_000000.jpg"), img)
    (workspace / "poses.json").write_text(json.dumps({"frames": [
        {"frame_id": "frame_000000", "K": _K().tolist(),
         "R": _look_down().tolist(), "t": [2.0, 2.0, 10.0]},
    ]}))
    conditions = {
        "frame_000000": {"usable": usable, "structure_ok": usable},
    }
    # Schema matches DepthAlignmentState.to_report(): the conditioning map
    # persists under "conditioning".
    (workspace / "depth_alignment_report.json").write_text(json.dumps({"conditioning": conditions}))
    return workspace


# ---------------------------------------------------------------------------
# GLB structure
# ---------------------------------------------------------------------------


def _load_glb(path: Path) -> dict:
    raw = path.read_bytes()
    assert raw[:4] == b"glTF", "not a GLB"
    length = struct.unpack("<I", raw[12:16])[0]
    return json.loads(raw[20:20 + length])


def _load_glb_with_binary(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    assert raw[:4] == b"glTF", "not a GLB"
    off = 12
    json_len = struct.unpack_from("<I", raw, off)[0]
    off += 8
    gltf = json.loads(raw[off:off + json_len])
    off += json_len
    bin_len = struct.unpack_from("<I", raw, off)[0]
    off += 8
    return gltf, raw[off:off + bin_len]


def _read_accessor(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    a = gltf["accessors"][index]
    view = gltf["bufferViews"][a["bufferView"]]
    base = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    dtype = {5125: "<u4", 5126: "<f4"}[a["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
    flat = np.frombuffer(binary, dtype=np.dtype(dtype),
                         count=a["count"] * width, offset=base)
    return flat.reshape(a["count"], width)


def test_textured_glb_has_texcoord_and_image(tmp_path):
    workspace = _seeded_workspace(tmp_path)
    mesh = _flat_mesh()
    texture = build_texture_payload(mesh, workspace)
    assert texture is not None and texture.image_bytes, "texturing must succeed on the visible quad"

    out = tmp_path / "mesh_viewer.glb"
    stats = export_viewer_glb(mesh, out, texture=texture)

    gltf = _load_glb(out)
    prim = gltf["meshes"][0]["primitives"][0]
    assert "TEXCOORD_0" in prim["attributes"], "textured GLB must carry UVs"
    assert "COLOR_0" not in prim["attributes"], "texture replaces vertex colors"
    assert gltf["images"], "atlas image must be embedded"
    assert gltf["images"][0]["mimeType"] == "image/jpeg"
    assert gltf["textures"], "material must reference a texture"
    # Buffer actually carries the image bytes.
    view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    assert view["byteLength"] > 0
    assert stats["textured"] is True
    assert stats["texture_source"] == "registered_views"
    assert "texture_fallback_reason" not in stats


def test_textured_glb_attributes_share_the_position_count(tmp_path):
    """glTF requires every attribute to have the POSITION accessor's count.

    Regression: the per-corner UV payload was written straight into
    TEXCOORD_0, so the GLB declared 3x more UVs than vertices. A compliant
    viewer reads uv[vertex_index], which then sampled an unrelated corner's
    atlas tile — the mesh rendered as scrambled dark patches ("no colour").
    """
    workspace = _seeded_workspace(tmp_path)
    mesh = _flat_mesh()
    texture = build_texture_payload(mesh, workspace)
    out = tmp_path / "mesh_viewer.glb"
    export_viewer_glb(mesh, out, texture=texture)

    gltf, binary = _load_glb_with_binary(out)
    prim = gltf["meshes"][0]["primitives"][0]
    positions = _read_accessor(gltf, binary, prim["attributes"]["POSITION"])
    uvs = _read_accessor(gltf, binary, prim["attributes"]["TEXCOORD_0"])
    indices = _read_accessor(gltf, binary, prim["indices"]).reshape(-1)

    assert len(positions) == len(uvs), (
        "TEXCOORD_0 count must equal the POSITION count "
        f"(got {len(uvs)} vs {len(positions)})"
    )
    assert int(indices.max()) < len(positions), "indices must address real vertices"
    assert gltf["accessors"][prim["attributes"]["TEXCOORD_0"]]["count"] == len(positions)

    # The spec-compliant read (uv[index]) must reproduce the payload's own
    # per-corner UVs — that is what makes the atlas land on the right faces.
    expected = np.asarray(texture.uv, dtype=np.float32).reshape(-1, 2)
    got = uvs[indices].astype(np.float32)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-6), "UVs are scrambled relative to the payload"


def test_reported_glb_vertices_is_the_count_in_the_file(tmp_path):
    """The ``viewer_glb`` detail must quote the GLB's real vertex count.

    Regression (live): ``viewer_glb_detail`` read the LOD's *pre-split* shared
    vertex count out of ``vertices`` while the writer records the written count
    in ``glb_vertices``. On Video_Mission_9ab1aa the reports said 788,074 while
    the file holds 3,299,997 and the viewer displayed the latter, so the record
    and the artifact disagreed by 4x.
    """
    from app.services.mesh_quality import viewer_glb_detail

    workspace = _seeded_workspace(tmp_path)
    mesh = _flat_mesh()
    texture = build_texture_payload(mesh, workspace)
    out = tmp_path / "mesh_viewer.glb"
    stats = export_viewer_glb(mesh, out, texture=texture)

    gltf = _load_glb(out)
    prim = gltf["meshes"][0]["primitives"][0]
    written = gltf["accessors"][prim["attributes"]["POSITION"]]["count"]

    detail = viewer_glb_detail(stats)
    assert detail["glb_vertices"] == written
    assert detail["glb_faces"] == stats["faces"] == gltf["accessors"][prim["indices"]]["count"] // 3
    # UV seams split vertices, so the file count is the honest one to report.
    assert stats["glb_vertices"] == written
    assert stats["vertices"] == mesh.n


def test_vertex_color_glb_unchanged_without_texture(tmp_path):
    out = tmp_path / "mesh_viewer.glb"
    stats = export_viewer_glb(_flat_mesh(), out)

    gltf = _load_glb(out)
    prim = gltf["meshes"][0]["primitives"][0]
    assert "COLOR_0" in prim["attributes"]
    assert "TEXCOORD_0" not in prim["attributes"]
    assert not gltf.get("images")
    assert stats["textured"] is False


# ---------------------------------------------------------------------------
# Authoritative-artifact safety
# ---------------------------------------------------------------------------


def test_authoritative_plys_byte_identical_after_textured_run(tmp_path):
    """A textured export must never touch the authoritative artifacts."""
    workspace = _seeded_workspace(tmp_path)
    mesh = _flat_mesh()
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    authoritative = {}
    for name in ("mesh.ply", "base_mesh.ply", "mesh_full.ply"):
        p = mesh_dir / name
        mesh.save_ply(p)
        authoritative[name] = p.read_bytes()

    texture = build_texture_payload(mesh, workspace)
    export_viewer_glb(mesh, mesh_dir / "mesh_viewer.glb", texture=texture)

    for name, before in authoritative.items():
        assert (mesh_dir / name).read_bytes() == before, f"{name} was modified"


# ---------------------------------------------------------------------------
# View selection: conditioning exclusions apply
# ---------------------------------------------------------------------------


def test_conditioning_excluded_views_are_not_texture_sources(tmp_path):
    workspace = _seeded_workspace(tmp_path, usable=False)
    views, excluded = _texture_views(workspace)
    assert views == []
    assert excluded == ["frame_000000"]


def test_usable_views_survive_conditioning_filter(tmp_path):
    workspace = _seeded_workspace(tmp_path, usable=True)
    views, excluded = _texture_views(workspace)
    assert [v.frame_id for v in views] == ["frame_000000"]
    assert excluded == []


# ---------------------------------------------------------------------------
# Honest fallbacks — every failure mode names itself
# ---------------------------------------------------------------------------


def test_no_views_falls_back_with_named_reason(tmp_path):
    workspace = tmp_path / "run_noviews"
    workspace.mkdir()
    (workspace / "poses.json").write_text(json.dumps({"frames": []}))
    payload = build_texture_payload(_flat_mesh(), workspace)
    assert payload.image_bytes == b""
    assert payload.fallback_reason
    assert "no registered views" in payload.fallback_reason

    stats = export_viewer_glb(_flat_mesh(), tmp_path / "g.glb", texture=payload)
    assert stats["textured"] is False
    assert stats["texture_fallback_reason"] == payload.fallback_reason


def test_no_visible_camera_falls_back_with_named_reason(tmp_path):
    """A view whose camera cannot see any face: atlas not written → fallback."""
    workspace = tmp_path / "run_blind"
    (workspace / "selected").mkdir(parents=True)
    cv2.imwrite(str(workspace / "selected" / "frame_000000.jpg"),
                np.zeros((96, 96, 3), dtype=np.uint8))
    # Camera far to the side, looking down away from the quad.
    (workspace / "poses.json").write_text(json.dumps({"frames": [
        {"frame_id": "frame_000000", "K": _K().tolist(),
         "R": _look_down().tolist(), "t": [500.0, 500.0, 10.0]},
    ]}))
    (workspace / "depth_alignment_report.json").write_text(
        json.dumps({"conditions": {"frame_000000": {"usable": True}}}))

    payload = build_texture_payload(_flat_mesh(), workspace)
    assert payload.image_bytes == b""
    assert payload.fallback_reason
    assert "no face has a visible camera" in payload.fallback_reason


def test_face_count_overflow_falls_back_with_named_reason(tmp_path):
    workspace = _seeded_workspace(tmp_path)
    # The atlas cap IS the viewer triangle budget (one knob owns the LOD
    # size); the payload builder refuses beyond it by name.
    cap = viewer_triangle_budget()
    assert cap > 0
    big = TriangleMesh(
        vertices=np.zeros((cap + 1, 3)),
        faces=np.zeros((cap + 1, 3), dtype=np.int64),
    )
    payload = build_texture_payload(big, workspace)
    assert payload.image_bytes == b""
    assert "atlas cap" in payload.fallback_reason


def test_fallback_glb_is_still_a_valid_color_glb(tmp_path):
    payload = _vertex_color_fallback("texture build failed: injected")
    out = tmp_path / "mesh_viewer.glb"
    stats = export_viewer_glb(_flat_mesh(), out, texture=payload)

    gltf = _load_glb(out)
    prim = gltf["meshes"][0]["primitives"][0]
    assert "COLOR_0" in prim["attributes"], "fallback must keep vertex colors"
    assert "TEXCOORD_0" not in prim["attributes"]
    assert stats["textured"] is False
    assert stats["texture_fallback_reason"] == "texture build failed: injected"
