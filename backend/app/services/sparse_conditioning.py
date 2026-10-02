"""Single-owner sparse-conditioning policy.

Two related questions share one module and one threshold set — a change to
the conditioning definition lands in exactly this file:

1. POINT conditioning (triangulation null-space): a sparse point whose
   observing cameras have too little baseline relative to its depth is
   unconstrained by the pixels — any depth along the ray reproduces them.
   Measured on the flight_to_tower_7511dc hover segment: ~1.9 m baselines
   against ~8.4 m depths, with the resulting depths ~2x the physically
   possible value at constant GNSS altitude, reprojecting at ~1 px (so
   reprojection metrics and BA cannot catch them). Used by
   ``trajectory_sync.retriangulate_points`` to drop null-space points
   before they poison the depth-alignment reference.

2. VIEW conditioning: a camera whose sparse reference is unusable (too
   few points, null-space structure, inverted depth ramp, no parallax)
   must not anchor a learned-depth map. Used by the depth stage's
   conditioning pre-pass; such views are EXCLUDED from generation and
   fusion entirely (logged, never silent) — fusing a map anchored to
   null-space geometry ships a second, wrong gauge into the dense model.

Threshold semantics (all documented, all measured):

- ``MIN_DEPTH_BASELINE_RATIO`` — point-level floor. Observing-baseline /
  depth below this ratio means along-ray position is not observable.
  Selected at 0.05 by the C1 sweep (scripts/c1_ratio_sweep.py): 2306
  well-conditioned points kept vs 594 at 0.10, null-space still screened
  (9427 dropped), adjacent-frame depth consistency preserved (median
  0.0345 vs 0.0361) and its p95 improved (0.208 vs 0.259).
- ``MIN_VIEW_SUPPORT`` — a view needs enough in-bounds sparse points for
  the dz/dv regression and a 2-parameter fit to be meaningful.
- ``MAX_INVERTED_STRUCTURE_FRACTION`` — SIGNED structure gate. Only a
  POSITIVE dz/dv ramp (camera-space depth increasing with image row) is
  inverted vs physics for a down-looking camera; a NEGATIVE slope is the
  correct direction and may be arbitrarily strong (real near-field
  structure). The total positive ramp across the frame (slope * height)
  must stay within this fraction of the view's median scene depth.
  Measured separation: benign flat ground ~2-5% (frame_000021: 0.0012
  m/px ≈ 2.5 m on a 53 m scene); inverted hover views ~90% (frame_000015:
  0.045 m/px ≈ 97 m). 0.15 separates them by an order of magnitude.
- ``MIN_VIEW_BASELINE_FRACTION`` — DIAGNOSTIC ONLY (not a usability gate).
  Nearest-neighbour camera translation as a fraction of the view's own
  median scene depth, reported per view. It is NOT part of ``usable``:
  the point-level screen above already guarantees conditioning with
  strictly more information (per-point TRACK baselines vs a camera
  proximity proxy). Measured on the 7511dc tower scene: a camera-proximity
  floor rejects 32/55 views whose sparse references are actually
  well-conditioned (e.g. frame_000021: structure correct, points from
  far-reaching tracks, but nearest camera 2.2 m away against 205 m scene
  depth). Hover null-space views need no separate gate — their points
  cannot pass the point-level screen (hover tracks cannot reach a 5%
  baseline), and their structure gate fails independently.
"""
from __future__ import annotations

import numpy as np

MIN_DEPTH_BASELINE_RATIO = 0.05
MIN_VIEW_SUPPORT = 40
MAX_INVERTED_STRUCTURE_FRACTION = 0.15
MIN_VIEW_BASELINE_FRACTION = 0.04

__all__ = [
    "MIN_DEPTH_BASELINE_RATIO",
    "MIN_VIEW_SUPPORT",
    "MAX_INVERTED_STRUCTURE_FRACTION",
    "MIN_VIEW_BASELINE_FRACTION",
    "point_min_baseline_m",
    "point_is_well_conditioned",
    "view_baseline_m",
    "probe_view",
]


def point_min_baseline_m(names, cams) -> float:
    """Smallest pairwise camera-centre distance (m) among observing cameras.

    ``cams`` maps camera name -> (R_w2c, C, K) tuples as built by
    ``trajectory_sync.retriangulate_points``.
    """
    if len(names) < 2:
        return 0.0
    pts = np.asarray([np.asarray(cams[n][1], dtype=np.float64) for n in names])
    d2 = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d2, np.inf)
    return float(d2.min())


def point_is_well_conditioned(
    names,
    cams,
    depth_m: float,
    *,
    min_ratio: float = MIN_DEPTH_BASELINE_RATIO,
) -> bool:
    """True when observing baselines actually constrain this point's depth."""
    if not np.isfinite(depth_m) or depth_m <= 0:
        return False
    return point_min_baseline_m(names, cams) >= min_ratio * float(depth_m)


def view_baseline_m(centers: np.ndarray, index: int) -> float:
    """Camera translation (m) from view *index* to its nearest neighbour.

    A single view has no parallax baseline (0.0), never infinity.
    """
    if len(centers) < 2:
        return 0.0
    d2 = np.linalg.norm(centers - centers[index], axis=1)
    d2[index] = np.inf
    m = float(d2.min())
    return m if np.isfinite(m) else 0.0


def probe_view(pose: dict, sparse_xyz: np.ndarray, image_wh) -> dict:
    """Measure whether a view's sparse depths are fit-worthy.

    Projects the sparse cloud into the view and regresses camera-space
    depth against image row. Returns a dict with:

      n_support               — sparse points projecting in-bounds
      baseline_m / baseline_ok— DIAGNOSTIC, filled by the caller (needs the
                                pose list); baseline_ok is NOT part of
                                usable — see MIN_VIEW_BASELINE_FRACTION
      dzdv_slope              — signed regression slope of z_cam vs row v
      z_median_m              — median camera-space scene depth
      inverted_ramp_m         — slope * frame height (total inverted ramp)
      inverted_ramp_fraction  — ramp / median depth
      structure_ok            — signed gate vs MAX_INVERTED_STRUCTURE_FRACTION
      usable                  — structure_ok AND sufficient support

    Never raises for empty inputs; unusable views return ``usable=False``
    with ``None`` fields where the measurement could not be made.
    """
    from app.services.geometry import project_world_to_pixel

    w, h = int(image_wh[0]), int(image_wh[1])
    u, v, z_cam = project_world_to_pixel(
        np.asarray(sparse_xyz, dtype=np.float64),
        np.asarray(pose["R"], dtype=np.float64),
        np.asarray(pose["t"], dtype=np.float64),
        np.abs(np.asarray(pose["K"], dtype=np.float64)),
    )
    inb = (
        np.isfinite(u) & np.isfinite(v) & np.isfinite(z_cam)
        & (z_cam > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    )
    n = int(inb.sum())
    info: dict = {
        "n_support": n,
        "baseline_m": None,
        "baseline_ok": None,
        "dzdv_slope": None,
        "z_median_m": None,
        "inverted_ramp_m": None,
        "inverted_ramp_fraction": None,
        "structure_ok": None,
        "usable": False,
    }
    if n < MIN_VIEW_SUPPORT:
        return info

    vv = v[inb]
    zz = z_cam[inb]
    slope = float(np.polyfit(vv, zz, 1)[0])
    z_med = float(np.median(zz))
    ramp = slope * float(h)
    info.update(
        {
            "dzdv_slope": round(slope, 6),
            "z_median_m": round(z_med, 3),
            "inverted_ramp_m": round(ramp, 3),
            "inverted_ramp_fraction": round(ramp / max(z_med, 1e-6), 4),
            "structure_ok": bool(
                ramp <= MAX_INVERTED_STRUCTURE_FRACTION * max(z_med, 1e-6)
            ),
        }
    )
    # Usability = sufficient support (guaranteed past the early return)
    # AND correct-direction structure. The camera-proximity baseline is
    # diagnostic only — the point-level screen owns conditioning.
    info["usable"] = bool(info["structure_ok"])
    return info
