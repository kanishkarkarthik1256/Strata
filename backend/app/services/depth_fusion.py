"""Multi-view depth fusion — unprojects per-view depth maps into a single
dense point cloud and merges redundant measurements on a voxel grid.

Pipeline per view
-----------------
1. Back-project each valid depth pixel to a camera ray::

       X_cam = depth * K^{-1} [u, v, 1]

2. Map to world coordinates with the view's extrinsic pose::

       X_world = R @ X_cam + t      (R = world-from-camera rotation)

3. Assign a per-measurement confidence from a quadratic depth-noise model
   (sigma grows with depth^2 / focal length, as for triangulation-based MVS).

Across views
------------
4. Quantise every measurement to the voxel grid (``voxel_size``) and fuse
   the members of each voxel with confidence-weighted averaging:

   * position  — weighted centroid
   * colour    — confidence-weighted mean RGB
   * residual  — RMS spread of the contributing measurements (m)
   * confidence — mean per-view confidence, clipped to [0, 1]
   * observations — number of contributing views

Memory efficient: every view is processed with vectorised numpy and the
final merge is a single group-by reduction over the concatenated
measurements, so peak memory stays O(points) rather than O(points * views).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.pointcloud import PointCloud

log = get_logger("drone_recon.services.depth_fusion")


@dataclass
class DepthView:
    """One registered frame with its depth map and pose.

    Pose convention: ``X_world = R @ X_cam + t`` where ``t`` is the camera
    centre in world coordinates and ``K`` maps camera coords to pixels.
    """

    frame_id: str
    depth: np.ndarray  # (H, W) float32 — meters, 0/NaN = invalid
    rgb: Optional[np.ndarray]  # (H, W, 3) uint8 — may be None
    K: np.ndarray  # (3, 3) intrinsics OF ``depth`` (not necessarily the frame's)
    R: np.ndarray  # (3, 3) world-from-camera rotation
    t: np.ndarray  # (3,) camera centre in world
    #: Frame pixels per unit of the stored (H, W) grid: ``(sx, sy)`` with
    #: ``1.0`` when the map is stored at the frame's own resolution. Needed
    #: to sample frame-resolution colour from a coarser depth grid — the
    #: depth model infers at ~518 px, so a stored map is usually much
    #: smaller than the frame it came from.
    frame_scale: tuple[float, float] = (1.0, 1.0)


@dataclass
class FusionParams:
    voxel_size: float = 0.05
    min_depth: float = 0.2
    max_depth: float = 200.0
    pixel_noise_px: float = 0.5
    max_points_per_view: int = 1_000_000
    max_total_points: int = 8_000_000


def fuse_depth_views(views: list[DepthView], params: FusionParams | None = None) -> PointCloud:
    """Fuse multiple depth views into one dense point cloud.

    Returns an un-filtered ``PointCloud`` — filtering/normals are later
    pipeline stages.
    """
    if not views:
        raise ValueError("fuse_depth_views requires at least one view")
    params = params or FusionParams(
        voxel_size=settings.dense.voxel_size,
        min_depth=settings.dense.min_depth_m,
        max_depth=settings.dense.max_depth_m,
        pixel_noise_px=settings.dense.pixel_noise_px,
        max_points_per_view=settings.dense.max_points_per_view,
        max_total_points=settings.dense.max_fusion_points,
    )
    if params.voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    start = time.perf_counter()
    xs: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    nrm: list[np.ndarray] = []

    # Parallel fan-out: unproject every view across worker processes
    # (bit-identical math; falls back in-process for few views).
    for view, (xyz, rgb, conf, normal) in zip(
        views, _unproject_views_fanout(views, params, with_pixels=False)
    ):
        xs.append(xyz)
        if rgb is not None:
            colors.append(rgb)
        confs.append(conf)
        # Zero-vector placeholder keeps the normal rows aligned with xyz
        # for views that produced no valid normals.
        nrm.append(normal if normal is not None else np.zeros((len(xyz), 3)))

    xyz = np.concatenate(xs, axis=0)
    conf = np.concatenate(confs)
    normals = None if not nrm else np.concatenate(nrm)

    if xyz.shape[0] > params.max_total_points:
        # Deterministic stride subsample instead of a random one.
        step = int(np.ceil(xyz.shape[0] / params.max_total_points))
        xyz = xyz[::step]
        conf = conf[::step]
        rgb = None if not colors else np.concatenate(colors)[::step]
        normals = None if normals is None else normals[::step]

    rgb = None if not colors else np.concatenate(colors)
    if xyz.shape[0] == 0:
        raise ValueError(
            f"depth fusion produced no measurements: every valid pixel falls outside "
            f"the depth band [{params.min_depth:g}, {params.max_depth:g}] m — the scene "
            "is deeper than the configured ceiling (far-field footage needs a "
            "scene-adapted ceiling; see scene_depth_ceiling)"
        )
    cloud = voxel_merge(xyz, rgb, conf, params.voxel_size, normals=normals)

    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info(
        "depth_fusion_complete",
        views=len(views),
        raw_points=len(xyz),
        fused_points=cloud.n,
        voxel_size=params.voxel_size,
        time_ms=round(elapsed_ms, 2),
    )
    cloud.meta.update(
        {
            "views_fused": len(views),
            "raw_measurements": int(len(xyz)),
            "voxel_size": float(params.voxel_size),
            "fusion_time_ms": round(elapsed_ms, 2),
        }
    )
    return cloud


#: Minimum number of views before process-parallel unprojection pays off
#: (spawn + pickling overhead dominates below this).
_PARALLEL_MIN_VIEWS = 8
_PARALLEL_MAX_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))


def _unproject_views_fanout(views, params, with_pixels=False):
    """Unproject many views concurrently (bit-identical results).

    The per-view work is pure numpy, which releases the GIL on the large
    array kernels — a thread pool gives real multi-core scaling with zero
    pickling of the ~35 MB/view inputs/outputs a process pool would need.
    """
    if len(views) < _PARALLEL_MIN_VIEWS:
        return [
            _unproject_view(v, params, with_pixels=with_pixels) for v in views
        ]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=_PARALLEL_MAX_WORKERS) as exe:
        results = list(
            exe.map(lambda v: _unproject_view(v, params, with_pixels=with_pixels), views)
        )
    return results


def _unproject_view(
    view: DepthView, params: FusionParams, with_pixels: bool = False
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]] | tuple:
    """Back-project one depth map into world coordinates.

    Returns (xyz (M,3), rgb (M,3) or None, confidence (M,), normals (M,3)
    or None) — or, with ``with_pixels=True``, additionally (u (M,), v (M,))
    pixel coordinates of each measurement (fusion provenance).
    Normals come from local central differences of neighbouring
    depth pixels (tangent cross product), oriented toward the observing
    camera, and are None/invalid where any neighbour is missing or the
    depth step indicates a discontinuity — such measurements never drive a
    surface split and contribute no normal to the consensus.
    """
    depth = np.asarray(view.depth, dtype=np.float64)
    h, w = depth.shape
    fx, fy = float(view.K[0, 0]), float(view.K[1, 1])
    cx, cy = float(view.K[0, 2]), float(view.K[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("depth view intrinsics must have positive focal length")

    z = depth
    valid = np.isfinite(z) & (z >= params.min_depth) & (z <= params.max_depth)

    # Deterministic subsample for very large depth maps: a UNIFORM 2D
    # stride on both axes. The row-major stride this replaces kept every
    # ``step``-th element of the flattened mask, i.e. every step-th COLUMN
    # of EVERY row — a ~7:1 anisotropic sample of the map (measured on the
    # shipped 4K maps: 480 of 3840 columns, all 2160 rows). For the same
    # point budget a 2D stride gives a uniform footprint, which is the
    # sampling the voxel merge downstream assumes. ``step`` is chosen so
    # the kept count cannot exceed the budget.
    count = int(valid.sum())
    if count > params.max_points_per_view:
        step = int(np.ceil(np.sqrt(count / float(params.max_points_per_view))))
        stride = np.zeros_like(valid)
        stride[::step, ::step] = True
        valid &= stride

    idx = np.nonzero(valid)
    if len(idx[0]) == 0:
        return np.zeros((0, 3)), None, np.zeros(0), None

    # Pixel coordinates straight from the mask — no (h, w)-sized meshgrid,
    # which allocated two int64 maps per view (133 MB each on a 4K map).
    z_v = z[idx]
    u_v = idx[1].astype(np.float64)
    v_v = idx[0].astype(np.float64)

    # Canonical convention via app.services.geometry (t = camera centre).
    from app.services.geometry import camera_to_world, unproject_pixel_to_camera

    cam = unproject_pixel_to_camera(u_v, v_v, z_v, view.K)
    world = camera_to_world(cam, view.R, view.t)

    # Quadratic depth noise model: sigma ~ z^2 / f * pixel noise.
    sigma = z_v**2 / min(fx, fy) * params.pixel_noise_px
    conf = np.exp(-sigma / max(params.voxel_size, 1e-9))
    conf = np.clip(conf, 0.05, 1.0)

    # ------------------------------------------------------------------
    # Per-measurement normals from local depth differences.
    # ------------------------------------------------------------------
    # Pad with NaN so border/invalid neighbours are detectable.
    zp = np.full((h + 2, w + 2), np.nan)
    zp[1:-1, 1:-1] = depth
    zi = idx[0] + 1
    j = idx[1] + 1
    zl, zr = zp[zi, j - 1], zp[zi, j + 1]
    zu, zd = zp[zi - 1, j], zp[zi + 1, j]
    zc = zp[zi, j]
    # Discontinuity guard: a neighbour step > 10% of depth is an occlusion
    # edge (at the pixel angular step dz/z ~ z/f ~ few % for scene slopes;
    # 10% keeps wall-like constant-depth surfaces while dropping silhouettes).
    step_ok = (
        np.isfinite(zl) & np.isfinite(zr) & np.isfinite(zu) & np.isfinite(zd)
        & (np.abs(zl - zc) <= 0.10 * zc) & (np.abs(zr - zc) <= 0.10 * zc)
        & (np.abs(zu - zc) <= 0.10 * zc) & (np.abs(zd - zc) <= 0.10 * zc)
    )
    n_out = np.zeros((len(z_v), 3))
    n_valid = np.zeros(len(z_v), dtype=bool)
    if step_ok.any():
        # World-space tangents from unprojected neighbour pixels.
        k = np.nonzero(step_ok)[0]
        ui_k, vi_k = u_v[k], v_v[k]
        zc_k = z_v[k]
        pl = camera_to_world(unproject_pixel_to_camera(ui_k - 1.0, vi_k, zl[k], view.K), view.R, view.t)
        pr = camera_to_world(unproject_pixel_to_camera(ui_k + 1.0, vi_k, zr[k], view.K), view.R, view.t)
        pdn = camera_to_world(unproject_pixel_to_camera(ui_k, vi_k + 1.0, zd[k], view.K), view.R, view.t)
        pup = camera_to_world(unproject_pixel_to_camera(ui_k, vi_k - 1.0, zu[k], view.K), view.R, view.t)
        t_u = pr - pl
        t_v = pdn - pup
        n = np.cross(t_u, t_v)
        nn = np.linalg.norm(n, axis=1)
        good = nn > 1e-9
        n[good] /= nn[good, None]
        # Orient toward the observing camera: n . (C - X) > 0.
        to_cam = np.asarray(view.t, dtype=np.float64)[None, :] - world[k]
        flip = np.einsum("ij,ij->i", n, to_cam) < 0
        n[flip] *= -1.0
        n_out[k[good]] = n[good]
        n_valid[k[good]] = True

    rgb = None
    if view.rgb is not None:
        rgb_img = np.asarray(view.rgb, dtype=np.uint8)
        if rgb_img.shape[:2] == (h, w):
            rgb = rgb_img[idx].astype(np.float64)
        else:
            # The depth map is stored in a coarser grid than the frame the
            # colours come from (the depth model infers at ~518 px): map the
            # map's pixel coordinates back to frame pixels before sampling.
            # Without this the colour branch silently returned None for every
            # native-resolution map and the cloud shipped uncoloured.
            sx, sy = view.frame_scale
            rh, rw = rgb_img.shape[:2]
            fu = np.clip((idx[1] / sx).astype(np.int64), 0, rw - 1)
            fv = np.clip((idx[0] / sy).astype(np.int64), 0, rh - 1)
            rgb = rgb_img[fv, fu].astype(np.float64)

    # Invalid-normal measurements carry a zero vector; voxel_merge treats
    # zero normals as neutral (never triggers a split, contributes none).
    if with_pixels:
        return world, rgb, conf, n_out if n_valid.any() else None, u_v, v_v
    return world, rgb, conf, n_out if n_valid.any() else None


def voxel_merge(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    conf: np.ndarray,
    voxel_size: float,
    normals: Optional[np.ndarray] = None,
    normal_cos_opposed: float = 0.5,
) -> PointCloud:
    """Merge measurements that fall in the same voxel.

    Confidence-weighted centroid/colour, sum of observations, and RMS
    spread (residual) per voxel. Vectorised with a single group-by.

    Surface separation: measurements in one voxel are merged only when
    compatible with the voxel's consensus surface —
      (a) not normal-OPPOSED to the consensus normal (dot < 0.5, i.e. >60°
          — a genuine different-layer indicator, immune to per-pixel
          gradient noise), and
      (b) within 0.5×voxel_size of the consensus PLANE (|(p−c)·n|) — a
          plane slicing diagonally through the voxel is one surface; a
          second layer offset along the normal is not. Layers separated
          by less than ~0.5×voxel are within the fusion's own residual
          envelope and are intentionally treated as one surface.
    Incompatible measurements are grouped into sibling voxels by a
    deterministic key (compatible minority points share one sibling), so
    overlapping surface layers (terrain vs tower base, wall vs roof) are
    NEVER averaged into one phantom point. Zero-vector normals
    (discontinuity-guarded measurements) never drive a split and
    contribute no consensus normal; voxels with no consensus normal yet
    do not split on geometric criteria.
    """
    n = len(xyz)
    conf = np.asarray(conf, dtype=np.float64)
    q = np.floor(xyz / voxel_size).astype(np.int64)  # quantised coords
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    q = q[order]
    pts = xyz[order]
    c = conf[order]
    col = rgb[order] if rgb is not None else None
    nrm = normals[order] if normals is not None else None

    # Voxel run boundaries (contiguous runs after the lexsort).
    starts = np.r_[0, np.nonzero((q[1:] != q[:-1]).any(axis=1))[0] + 1]
    ends = np.r_[starts[1:], n]
    n_vox = len(starts)

    # ------------------------------------------------------------------
    # Surface separation. Score every measurement against its voxel's
    # weighted consensus (position + normal); incompatible members get a
    # sibling suffix. Two sweep passes so a large layer entering late can
    # still claim consensus before a wrong split freezes.
    # ------------------------------------------------------------------
    sib = np.zeros(n, dtype=np.int64)
    if nrm is not None:
        nrm = np.asarray(nrm, dtype=np.float64)
        raw_norm = np.linalg.norm(nrm, axis=1)
        has_n = raw_norm > 0.5
        nrm[has_n] /= raw_norm[has_n, None]
        nrm[~has_n] = 0.0

        def _bad_mask(run_c: np.ndarray, run_n: np.ndarray, has_cons: np.ndarray) -> np.ndarray:
            dot = np.einsum("ij,ij->i", nrm, run_n)
            opposed = has_n & has_cons & (dot < normal_cos_opposed)
            off = has_cons & (np.abs(np.einsum("ij,ij->i", pts - run_c, run_n)) > 0.5 * voxel_size)
            return opposed | off

        def _regroup(bad: np.ndarray, base: np.ndarray, slot0: np.ndarray) -> np.ndarray:
            """Deterministic sibling slots for ``bad`` rows.

            Compatible bad rows share one slot (checked against each
            group's first member): normal dot >= threshold AND within
            0.75×voxel along the rep's plane. ``base`` is the current sib
            array (good rows keep theirs); ``slot0`` the per-run last used
            slot.

            Hot loop over up to millions of bad rows: scalar numpy
            fancy-indexing (nrm[i] etc.) costs ~1.5 µs/dispatch — measured
            1.8M np.dot calls = 3.8 s of pure overhead. Row floats are
            pre-extracted into plain Python floats (ONE vectorised pass) so
            the per-row math is float arithmetic on the same values in the
            same order — identical results, ~10× less dispatch overhead.
            """
            bad_idx = np.nonzero(bad)[0]
            new_sib = base.copy()
            slot = slot0.copy()
            if len(bad_idx) == 0:
                return new_sib
            # One vectorised extraction instead of per-row fancy indexing.
            b_run = run_of[bad_idx].tolist()
            b_n = nrm[bad_idx].tolist()
            b_p = pts[bad_idx].tolist()
            reps: dict[int, list[tuple[int, list[float], list[float]]]] = {}
            cos_t = normal_cos_opposed
            half_v = 0.5 * voxel_size
            for pos in range(len(bad_idx)):
                i = int(bad_idx[pos])
                v_i = b_run[pos]
                n_i = b_n[pos]
                p_i = b_p[pos]
                grp = None
                for r, n_r, p_r in reps.get(v_i, ()):
                    dot = n_i[0] * n_r[0] + n_i[1] * n_r[1] + n_i[2] * n_r[2]
                    if dot >= cos_t:
                        dx = p_i[0] - p_r[0]; dy = p_i[1] - p_r[1]; dz = p_i[2] - p_r[2]
                        if abs(dx * n_r[0] + dy * n_r[1] + dz * n_r[2]) <= half_v:
                            grp = int(new_sib[r])
                            break
                if grp is None:
                    slot[v_i] += 1
                    new_sib[i] = slot[v_i]
                    reps.setdefault(v_i, []).append((i, n_i, p_i))
                else:
                    new_sib[i] = grp
            return new_sib

        def _resort() -> None:
            nonlocal pts, c, col, nrm, has_n, q, sib, starts, ends, n_vox, run_of
            key = np.lexsort((sib, run_of))
            pts, c, sib = pts[key], c[key], sib[key]
            col = col[key] if col is not None else None
            nrm = nrm[key]
            has_n = has_n[key]
            q = q[key]
            starts = np.r_[0, np.nonzero((q[1:] != q[:-1]).any(axis=1) | (sib[1:] != sib[:-1]))[0] + 1]
            ends = np.r_[starts[1:], n]
            n_vox = len(starts)
            run_of = np.repeat(np.arange(n_vox), ends - starts)

        # Sweep 1 — seed each voxel's plane with its highest-confidence
        # member (an actual surface sample): both layers of a contaminated
        # voxel separate cleanly in one pass (the minority layer is opposed
        # to / off the seed plane, the majority stays).
        run_of = np.repeat(np.arange(n_vox), ends - starts)
        by_conf = np.lexsort((-c, run_of))  # rows grouped by run, conf desc
        run_sorted = run_of[by_conf]
        blk = np.r_[0, np.nonzero(run_sorted[1:] != run_sorted[:-1])[0] + 1]
        seed_rows = by_conf[blk]  # highest-conf row per run (ties → first)
        seed_c = pts[seed_rows][run_of]
        seed_n = nrm[seed_rows][run_of]
        bad = _bad_mask(seed_c, seed_n, np.ones(n, dtype=bool))
        sib = _regroup(bad, sib, np.zeros(n_vox, dtype=np.int64)) if bad.any() else sib

        # Sweeps 2..4 — refine against per-run consensus until fixpoint.
        # Good rows KEEP their sibling slot (consistency never merges a
        # separated layer back into the majority run).
        for _ in range(3):
            if not (sib > 0).any():
                break
            _resort()
            centroid = np.add.reduceat(pts * c[:, None], starts, axis=0)
            w = np.add.reduceat(c, starts)
            centroid /= np.where(w > 0, w, 1.0)[:, None]
            wsum = np.add.reduceat(nrm * (c * has_n)[:, None], starts, axis=0)
            wn = np.linalg.norm(wsum, axis=1)
            consensus_n = wsum / np.maximum(wn, 1e-12)[:, None]
            bad = _bad_mask(
                centroid[run_of], consensus_n[run_of], (wn > 1e-6)[run_of]
            )
            if not bad.any():
                break
            slot0 = np.maximum.reduceat(sib, starts)
            new_sib = _regroup(bad, sib, slot0)
            if np.array_equal(new_sib, sib):
                break
            sib = new_sib
            _resort()

    # Group-by reductions over contiguous voxel runs. Weighted sums drive the
    # centroid/colour; plain sums drive the residual (RMS spread).
    wsum = np.add.reduceat(pts * c[:, None], starts, axis=0)  # conf-weighted xyz
    w = np.add.reduceat(c, starts)  # total weight per voxel
    obs = (ends - starts).astype(np.int32)
    w_safe = np.where(w > 0, w, 1.0)
    centroid = wsum / w_safe[:, None]

    psum = np.add.reduceat(pts, starts, axis=0)  # plain xyz sum
    sumsq = np.add.reduceat(np.einsum("ij,ij->i", pts, pts), starts)
    mean = psum / obs[:, None]
    residual = np.sqrt(np.maximum(sumsq / obs - np.einsum("ij,ij->i", mean, mean), 0.0))
    mean_conf = np.clip(w / np.maximum(obs, 1), 0.0, 1.0)

    out_normals = None
    if nrm is not None:
        nsum = np.add.reduceat(nrm * (c * has_n)[:, None], starts, axis=0)
        onorm = np.linalg.norm(nsum, axis=1, keepdims=True)
        out_normals = np.where(onorm > 1e-9, nsum / np.maximum(onorm, 1e-12), 0.0)

    colors = None
    if col is not None:
        cw = np.add.reduceat(col * c[:, None], starts, axis=0) / w_safe[:, None]
        colors = np.clip(np.rint(cw), 0, 255).astype(np.uint8)

    split = int((sib > 0).sum())
    cloud = PointCloud(
        xyz=centroid,
        rgb=colors,
        confidence=mean_conf,
        observations=obs,
        residual=residual,
        normals=out_normals,
    )
    cloud.meta["surface_split_measurements"] = split
    return cloud


def fuse_depth_views_with_provenance(
    views: list[DepthView], params: FusionParams | None = None,
) -> tuple[PointCloud, dict[str, np.ndarray]]:
    """Fuse like ``fuse_depth_views`` and additionally return provenance.

    Phase 1 Part 5: every fused point must be traceable to the measurements
    that formed it. Returns (cloud, provenance) where provenance holds
      * ``source_frame_id``  (N,) int32 — view index of the highest-confidence
        measurement in the fused point's voxel (the dominant observer; -1
        when no measurement's voxel contains the centroid),
      * ``fusion_weight``    (N,) float64 — total confidence mass of the voxel
        (Σ w_i, the same weight the centroid used),
      * ``source_pixel_u``/``source_pixel_v`` (N,) float32 — pixel of that
        dominant measurement in its source view.
    implemented by an index-parallel group-by over the SAME sorted layout
    voxel_merge uses — the merge math is untouched.
    """
    if not views:
        raise ValueError("fuse_depth_views requires at least one view")
    params = params or FusionParams(
        voxel_size=settings.dense.voxel_size,
        min_depth=settings.dense.min_depth_m,
        max_depth=settings.dense.max_depth_m,
        pixel_noise_px=settings.dense.pixel_noise_px,
        max_points_per_view=settings.dense.max_points_per_view,
        max_total_points=settings.dense.max_fusion_points,
    )
    if params.voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    start = time.perf_counter()
    xs: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    nrm: list[np.ndarray] = []
    frame_idx: list[np.ndarray] = []
    pix_u: list[np.ndarray] = []
    pix_v: list[np.ndarray] = []

    # Threaded fan-out (same kernels as fuse_depth_views): the per-view
    # unprojection is pure numpy that releases the GIL — serial measured
    # ~95 s for 35 full-res views, ~8× less wall clock under the pool.
    results = _unproject_views_fanout(views, params, with_pixels=True)
    for vi, view in enumerate(views):
        xyz, rgb, conf, normal, u_pix, v_pix = results[vi]
        xs.append(xyz)
        if rgb is not None:
            colors.append(rgb)
        confs.append(conf)
        nrm.append(normal if normal is not None else np.zeros((len(xyz), 3)))
        frame_idx.append(np.full(len(xyz), vi, dtype=np.int32))
        pix_u.append(u_pix)
        pix_v.append(v_pix)

    xyz = np.concatenate(xs, axis=0)
    conf = np.concatenate(confs)
    normals = None if not nrm else np.concatenate(nrm)
    fidx = np.concatenate(frame_idx)
    pu = np.concatenate(pix_u)
    pv = np.concatenate(pix_v)
    rgb_all = None if not colors else np.concatenate(colors)

    if xyz.shape[0] > params.max_total_points:
        step = int(np.ceil(xyz.shape[0] / params.max_total_points))
        xyz = xyz[::step]
        conf = conf[::step]
        rgb_all = None if rgb_all is None else rgb_all[::step]
        normals = None if normals is None else normals[::step]
        fidx = fidx[::step]
        pu = pu[::step]
        pv = pv[::step]

    if xyz.shape[0] == 0:
        raise ValueError(
            f"depth fusion produced no measurements: every valid pixel falls outside "
            f"the depth band [{params.min_depth:g}, {params.max_depth:g}] m — the scene "
            "is deeper than the configured ceiling (far-field footage needs a "
            "scene-adapted ceiling; see scene_depth_ceiling)"
        )
    cloud = voxel_merge(xyz, rgb_all, conf, params.voxel_size, normals=normals)

    # Provenance (Part 5): for every FUSED point, the highest-confidence raw
    # measurement whose voxel contains that point's centroid (ties broken by
    # measurement order). Fully vectorised: one lexsort of measurements by
    # (voxel key, confidence desc, index) picks each voxel's dominant
    # observer; one searchsorted maps fused centroids to their voxel. No
    # per-point Python loop — 1M+ measurements are routine.
    # Centroid quantisation: the weighted mean of members sharing a value on
    # an axis (e.g. z=10.0 → floor gives 19 vs members' 20 when the mean is
    # 9.9999…) must resolve to the MEMBERS' voxel. floor(x/s + ε) with ε =
    # half the float64 gap at that magnitude snaps exact-mean boundaries to
    # the members' cell; genuinely drifted centroids (>1e-6 rel) still
    # quantise to their own cell.
    eps = np.maximum(1.0, np.abs(cloud.xyz)).max() * np.finfo(np.float64).eps * 0.5
    q_raw = np.floor(xyz / params.voxel_size).astype(np.int64)
    q_fused = np.floor(cloud.xyz / params.voxel_size + eps / params.voxel_size).astype(np.int64)
    # Voxel-key packing: |q| stays < 2^20 for any scene < ~2M voxels per
    # axis (airport8 spans ~932 voxels at 0.5 m), so 21-bit slots cannot
    # collide at these scene scales.
    key_raw = q_raw[:, 0] * (1 << 42) + q_raw[:, 1] * (1 << 21) + q_raw[:, 2]
    key_fused = q_fused[:, 0] * (1 << 42) + q_fused[:, 1] * (1 << 21) + q_fused[:, 2]

    ordm = np.lexsort((np.arange(len(key_raw)), -conf, key_raw))
    ks = key_raw[ordm]
    starts = np.r_[0, np.nonzero(ks[1:] != ks[:-1])[0] + 1]
    best_m = ordm[starts]  # dominant measurement per occupied voxel
    voxel_of_best = ks[starts]  # sorted unique voxel keys

    pos = np.clip(np.searchsorted(voxel_of_best, key_fused), 0, len(voxel_of_best) - 1)
    hit = voxel_of_best[pos] == key_fused
    dest = np.nonzero(hit)[0]
    src = pos[dest]

    dom = np.full(cloud.n, -1, dtype=np.int32)
    dom_u = np.zeros(cloud.n, dtype=np.float32)
    dom_v = np.zeros(cloud.n, dtype=np.float32)
    dom[dest] = fidx[best_m[src]]
    dom_u[dest] = pu[best_m[src]]
    dom_v[dest] = pv[best_m[src]]
    # fusion weight = total confidence mass of the voxel (Σ w_i), recovered
    # from the merge output: mean_conf × observations.
    w_out = np.where(dom >= 0, cloud.confidence * cloud.observations, 0.0)

    provenance = {
        "source_frame_id": dom,
        "source_frame_names": [v.frame_id for v in views],
        "fusion_weight": w_out,
        "source_pixel_u": dom_u,
        "source_pixel_v": dom_v,
    }
    elapsed_ms = (time.perf_counter() - start) * 1000
    cloud.meta.update({
        "views_fused": len(views),
        "raw_measurements": int(len(xyz)),
        "voxel_size": float(params.voxel_size),
        "fusion_time_ms": round(elapsed_ms, 2),
    })
    return cloud, provenance
