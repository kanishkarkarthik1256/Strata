"""
verify_atlas_bake.py — Prove the batched atlas bake is byte-identical to the
per-face reference and measure the speedup.
"""

from __future__ import annotations

import time
import sys
import numpy as np
import cv2


def main():
    from app.services.uv_mapper import build_atlas_layout
    from app.services.texture_blender import (
        _inv_affine23,
        _bake_faces_batched_remap,
        _project_all,
    )
    from app.services.mesh import TriangleMesh
    from app.services.texture_projector import TextureView

    print("=" * 60)
    print("  Atlas bake verification: byte-identical + benchmark")
    print("=" * 60)

    # --- synthetic mesh ---
    rng = np.random.default_rng(0)
    n_faces = 5000
    xyz = np.column_stack([
        rng.uniform(-30, 30, n_faces * 3),
        rng.uniform(-30, 30, n_faces * 3),
        rng.uniform(0, 8, n_faces * 3),
    ]).astype(np.float64)
    faces_arr = np.array([list(range(i, i + 3)) for i in range(0, len(xyz), 3)])
    mesh = TriangleMesh(vertices=xyz, faces=faces_arr)

    # --- synthetic views ---
    views = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        radius = 45.0
        t = np.array([radius * np.cos(angle), radius * np.sin(angle), 15.0])
        look = -t
        look /= np.linalg.norm(look)
        right = np.cross(look, [0, 0, 1])
        if np.linalg.norm(right) < 0.01:
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, look)
        # R maps world→cam: X_cam = R @ (X_world - t)
        # Camera looks down +Z, so R[:,2] should point toward the scene (from camera to origin)
        R_cols = np.column_stack([right, up, look])  # look points from cam toward origin
        R = R_cols.T

        # Wider FOV
        K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float64)

        img = np.zeros((512, 512, 3), dtype=np.uint8)
        for c in range(3):
            img[:, :, c] = ((np.arange(512)[:, None] * 0.3
                             + np.arange(512) * 0.5
                             + i * 30 + c * 40) % 256).astype(np.uint8)

        views.append(TextureView(
            frame_id=f"v{i:04d}", image=img, K=K, R=R, t=t,
            depth=None, K_depth=None,
        ))

    layout = build_atlas_layout(mesh.m, atlas_width=2048)
    cell = layout.cell

    print(f"Mesh: {mesh.n} verts, {mesh.m} faces")
    print(f"Atlas: {layout.width}×{layout.height}, cell={cell}")
    print(f"Views: {len(views)}")

    # --- pick view 0 ---
    vi = 0
    view = views[vi]
    verts = mesh.vertices
    img = view.image
    H, W = img.shape[:2]
    gain = 1.0

    # Debug projection
    from app.services.texture_blender import _project_all
    centroids = verts[faces_arr].mean(axis=1)
    uv_c, dep_c = _project_all(view, centroids)
    print(f"  UV range: [{uv_c.min():.1f}, {uv_c.max():.1f}]×[{uv_c[:,1].min():.1f}, {uv_c[:,1].max():.1f}]")
    print(f"  Depth range: [{dep_c.min():.1f}, {dep_c.max():.1f}]")
    print(f"  Image: {W}×{H}")
    in_image = (uv_c[:, 0] >= 0) & (uv_c[:, 0] < W) & (uv_c[:, 1] >= 0) & (uv_c[:, 1] < H) & (dep_c > 0)
    print(f"  In image + positive depth: {in_image.sum()}/{len(centroids)}")

    # Project face vertices
    all_pts = verts[faces_arr].reshape(-1, 3)
    all_src, all_depth = _project_all(view, all_pts)
    src_3 = all_src.reshape(-1, 3, 2)
    depth_3 = all_depth.reshape(-1, 3)
    ok = np.isfinite(src_3).all(axis=(1, 2)) & (depth_3.min(axis=1) > 0)
    faces = np.flatnonzero(ok)
    src = src_3[ok]
    depth = depth_3[ok]

    print(f"View {vi}: {len(faces)} visible faces (of {mesh.m})")
    if len(faces) < 10:
        print("Too few visible — trying view 1..")
        for vi2 in range(1, len(views)):
            view = views[vi2]
            all_src, all_depth = _project_all(view, all_pts)
            src_3 = all_src.reshape(-1, 3, 2)
            depth_3 = all_depth.reshape(-1, 3)
            ok = np.isfinite(src_3).all(axis=(1, 2)) & (depth_3.min(axis=1) > 0)
            faces = np.flatnonzero(ok)
            if len(faces) >= 10:
                vi = vi2
                view = views[vi]
                img = view.image
                H, W = img.shape[:2]
                src = src_3[ok]
                depth = depth_3[ok]
                print(f"Using view {vi}: {len(faces)} visible faces")
                break
        else:
            print("No view has enough visible faces.")
            return

    # --- crop boxes + affines ---
    x0 = np.floor(src[:, :, 0].min(axis=1)).astype(np.int64) - 1
    y0 = np.floor(src[:, :, 1].min(axis=1)).astype(np.int64) - 1
    x1 = np.ceil(src[:, :, 0].max(axis=1)).astype(np.int64) + 2
    y1 = np.ceil(src[:, :, 1].max(axis=1)).astype(np.int64) + 2
    x0c = np.clip(x0, 0, W)
    y0c = np.clip(y0, 0, H)
    x1c = np.clip(x1, 0, W)
    y1c = np.clip(y1, 0, H)
    valid = ((x1c - x0c) >= 1) & ((y1c - y0c) >= 1)
    faces = faces[valid]
    src = src[valid]
    x0c, y0c, x1c, y1c = x0c[valid], y0c[valid], x1c[valid], y1c[valid]

    tile = layout.uv_px[faces]
    tile_origin = np.stack(
        [tile[:, 0, 0].astype(np.int64), tile[:, 0, 1].astype(np.int64)], axis=-1
    )[:, None, :]
    dst_rel = tile - tile_origin
    src_rel = src - np.stack([x0c, y0c], axis=1)[:, None, :]
    hom = np.concatenate([src_rel, np.ones((len(faces), 3, 1))], axis=2)
    aff = np.linalg.solve(hom, dst_rel).transpose(0, 2, 1)

    print(f"Batched affine solve: {len(faces)} faces")

    # Debug: compare per-face and batched for first face
    k = 0
    face0 = faces[k]
    tile0 = layout.uv_px[face0]
    x_t0, y_t0 = int(tile0[0, 0]), int(tile0[0, 1])
    print(f"Face 0: tile at ({x_t0}, {y_t0}), cell={cell}")
    print(f"  aff[0] = {aff[k].tolist()}")
    inv0 = _inv_affine23(aff[k:k+1])[0]
    print(f"  inv[0] = {inv0.tolist()}")
    # Verify: warpAffine on crop vs remap on full image
    crop0 = img[y0c[k]:y1c[k], x0c[k]:x1c[k]]
    warp_ref = cv2.warpAffine(crop0.astype(np.float32), aff[k], (cell, cell))
    print(f"  warpAffine shape: {warp_ref.shape}, mean={warp_ref.mean():.1f}")
    # Now test remap: build xmap/ymap for just this face's tile region
    yy0, xx0 = np.mgrid[y_t0:y_t0+cell, x_t0:x_t0+cell]
    px_rel0 = xx0 - x_t0
    py_rel0 = yy0 - y_t0
    src_x0 = inv0[0,0]*px_rel0 + inv0[0,1]*py_rel0 + inv0[0,2] + x0c[k]
    src_y0 = inv0[1,0]*px_rel0 + inv0[1,1]*py_rel0 + inv0[1,2] + y0c[k]
    warp_remap = cv2.remap(img, src_x0.astype(np.float32), src_y0.astype(np.float32),
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    print(f"  remap shape: {warp_remap.shape}, mean={warp_remap.mean():.1f}")
    print(f"  diff max: {np.abs(warp_ref - warp_remap).max()}")
    print(f"  warpAffine sample [0,0]: {warp_ref[0,0].tolist()}")
    print(f"  remap sample [0,0]: {warp_remap[0,0].tolist()}")

    # --- per-face reference ---
    def bake_per_face(atlas):
        count = 0
        for k in range(len(faces)):
            face = faces[k]
            tile_k = layout.uv_px[face]
            x_t, y_t = int(tile_k[0, 0]), int(tile_k[0, 1])
            crop = img[y0c[k]:y1c[k], x0c[k]:x1c[k]]
            warped = cv2.warpAffine(crop.astype(np.float32), aff[k], (cell, cell))
            warped = np.clip(warped * gain, 0, 255).astype(np.uint8)
            mask = np.any(warped > 0, axis=2)
            roi = atlas[y_t:y_t + cell, x_t:x_t + cell]
            roi[:] = np.where(mask[:, :, None], warped, roi)
            count += 1
        return count

    atlas_ref = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    count_ref = bake_per_face(atlas_ref)
    t_ref = time.perf_counter() - t0

    # Also test: batched remap with per-face crop bounds check on xmap/ymap
    from app.services.texture_blender import _inv_affine23, _bake_faces_batched_remap
    atlas_batch2 = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    t0b = time.perf_counter()
    count_batch2 = _bake_faces_batched_remap(
        atlas_batch2, img, faces, layout, aff, x0c, y0c, x1c, y1c, cell, gain
    )
    t_batch2 = time.perf_counter() - t0b
    d2 = np.abs(atlas_ref.astype(int) - atlas_batch2.astype(int))
    print(f"Batched2 (new fn): {count_batch2} faces, {t_batch2:.4f}s, diff={int((d2>0).sum())} pixels, max={d2.max()}")

    # --- batched remap ---
    atlas_batch = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    count_batch = _bake_faces_batched_remap(
        atlas_batch, img, faces, layout, aff, x0c, y0c, x1c, y1c, cell, gain
    )
    t_batch = time.perf_counter() - t0

    # Debug: compare atlas region for face 0
    tile0 = layout.uv_px[faces[0]]
    x_t0, y_t0 = int(tile0[0, 0]), int(tile0[0, 1])
    print(f"\nDebug face 0 tile ({x_t0},{y_t0}):")
    print(f"  ref tile mean: {atlas_ref[y_t0:y_t0+cell, x_t0:x_t0+cell].mean():.1f}")
    print(f"  batch tile mean: {atlas_batch[y_t0:y_t0+cell, x_t0:x_t0+cell].mean():.1f}")
    ref_tile = atlas_ref[y_t0:y_t0+cell, x_t0:x_t0+cell]
    batch_tile = atlas_batch[y_t0:y_t0+cell, x_t0:x_t0+cell]
    print(f"  tile diff max: {np.abs(ref_tile.astype(int) - batch_tile.astype(int)).max()}")
    print(f"  ref non-zero: {(ref_tile > 0).any(axis=2).sum()}, batch non-zero: {(batch_tile > 0).any(axis=2).sum()}")

    # Check if the remap maps to the right source region
    # For face 0, check what source pixel (0,0) in the tile maps to
    inv0 = _inv_affine23(aff[0:1])[0]
    print(f"  inv0 translation: ({inv0[0,2]:.1f}, {inv0[1,2]:.1f})")
    print(f"  x0c[0]={x0c[0]}, y0c[0]={y0c[0]}")
    # Source for tile pixel (0,0): src = inv @ [0, 0, 1] + [x0c, y0c]
    sx00 = inv0[0, 2] + x0c[0]
    sy00 = inv0[1, 2] + y0c[0]
    print(f"  tile(0,0) -> source({sx00:.1f}, {sy00:.1f})")
    print(f"  source img value at that point: ", end="")
    if 0 <= sx00 < W and 0 <= sy00 < H:
        print(f"{img[int(sy00), int(sx00)].tolist()}")
    else:
        print("OUT OF BOUNDS")

    # Check source triangle vertices for face 0
    face0 = faces[0]
    tri_verts = verts[mesh.faces[face0]].reshape(-1, 3)
    src_tri, _ = _project_all(view, tri_verts)
    print(f"  Source triangle vertices (UV):")
    for j in range(3):
        print(f"    v{j}: ({src_tri[j,0]:.1f}, {src_tri[j,1]:.1f})")
    print(f"  Crop bounds: x=[{x0c[0]}, {x1c[0]}), y=[{y0c[0]}, {y1c[0]})")
    print(f"  Crop size: {x1c[0]-x0c[0]}×{y1c[0]-y0c[0]}")
    # The affine maps src_rel -> dst_rel
    # src_rel = src - [x0c, y0c]
    # For the first source vertex:
    sx_rel0 = src_tri[0, 0] - x0c[0]
    sy_rel0 = src_tri[0, 1] - y0c[0]
    dst0 = aff[0] @ [sx_rel0, sy_rel0, 1]
    print(f"  src v0 rel=({sx_rel0:.1f},{sy_rel0:.1f}) -> dst_rel=({dst0[0]:.1f},{dst0[1]:.1f})")
    # Check: does dst_rel match the tile? tile0 = layout.uv_px[face0]
    print(f"  tile0 (from layout): {tile0.tolist()}")
    print(f"  tile_origin: {tile_origin[0].tolist()}")
    print(f"  dst_rel expected: {(tile0 - tile_origin[0]).tolist()}")

    # --- results ---
    print(f"\n{'Metric':<30} {'Per-face':>12} {'Batched':>12}")
    print("-" * 54)
    print(f"{'Faces baked':<30} {count_ref:>12} {count_batch:>12}")
    print(f"{'Time (s)':<30} {t_ref:>12.4f} {t_batch:>12.4f}")
    if t_batch > 0:
        print(f"{'Speedup':<30} {'':>12} {t_ref / t_batch:>11.1f}×")

    match = np.array_equal(atlas_ref, atlas_batch)
    diff = np.abs(atlas_ref.astype(int) - atlas_batch.astype(int))
    max_diff = diff.max()
    n_diff = int((diff > 0).sum())

    print(f"\n{'Byte-identical':<30} {'YES' if match else 'NO':>12}")
    print(f"{'Max pixel diff':<30} {max_diff:>12}")
    print(f"{'Differing pixels':<30} {n_diff:>12}")

    if not match and n_diff > 0:
        print("\n⚠ Differences (sample):")
        y_idx, x_idx = np.where(diff.max(axis=2) > 0)
        for i in range(min(5, len(y_idx))):
            y, x = y_idx[i], x_idx[i]
            print(f"  ({x:4d}, {y:4d}): ref={atlas_ref[y,x].tolist()} "
                  f"batch={atlas_batch[y,x].tolist()}")

    print("\n" + "=" * 60)
    if match and n_diff == 0:
        print("  ✅ VERIFIED: batched atlas bake is byte-identical")
    else:
        print(f"  ❌ FAILED: {n_diff} differing pixels, max diff = {max_diff}")
        raise SystemExit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
