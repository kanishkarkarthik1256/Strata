"""Dense/surface diagnostics (Phase 1B).

Everything here is DIAGNOSTIC: it measures geometry that already exists and
classifies it with explicit, documented thresholds. It never mutates the
cloud or mesh it inspects and never uses ground-truth data.

Layers
------
A "layer" is a second surface sheet inside one local neighbourhood —
e.g. terrain and a duplicate sheet ~2 m above it. Detection is per-cell:
the cloud is bucketed on a coarse grid (``cell = 4 × voxel``), each cell's
points are projected on the cell's mean normal, and gaps along that axis
larger than ``min_separation`` split the cell into distinct sheets.

Classification (layer source)
-----------------------------
The same detector runs on the dense cloud and on the mesh vertices:

* CASE A — layering originates in the dense reconstruction (the mesh only
  inherits it).
* CASE B — the dense cloud is mostly single-surface; meshing creates layers.
* CASE C — both contribute measurably.
* CASE D — insufficient evidence (too few points / detector could not run).

Support
-------
Every mesh vertex is queried against the dense cloud (kNN). A vertex is
"supported" when its nearest dense point lies within the fusion voxel —
the measured surface scale of the run. Unsupported connected regions are
reported separately so large fabricated patches cannot hide inside an
overall-good percentage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.logging_config import get_logger

log = get_logger("drone_recon.services.dense_diagnostics")

# detect_layers scales: min_sep = LAYER_MIN_SEP_FACTOR × scale, cell = 8 × scale.
# The scale argument must be the sampling scale of the point set being measured
# (fusion voxel for the dense cloud, ~median mesh edge / factor for a mesh) —
# measuring a coarser point set with a finer scale counts same-surface points
# as sheets. Centralised so callers cannot re-derive it differently.
LAYER_MIN_SEP_FACTOR = 1.5
LAYER_CELL_FACTOR = 8.0


# --------------------------------------------------------------------------
# Multi-layer detection
# --------------------------------------------------------------------------

def detect_layers(
    xyz: np.ndarray,
    voxel: float,
    *,
    normals: np.ndarray | None = None,
    max_seeds: int = 40_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Quantify multi-sheet surface structure in a point set.

    Returns layer_region_count (cells containing ≥2 sheets),
    layer_pair_count, separation statistics, layered point percentage and
    the threshold used. Empty/failed runs report counts of None with
    ``status: "insufficient_points"`` rather than zeros.
    """
    n = len(xyz)
    if n < 500 or voxel <= 0:
        return {"status": "insufficient_points", "points": int(n)}

    rng = np.random.default_rng(seed)
    # Vertical slab columns: a layer signature is two surfaces at DIFFERENT
    # heights sharing the same (x, y) footprint, so cells must be tall
    # columns (floor(x/c) × floor(y/c), full Z extent) — a 3-D cell would
    # put the two sheets in separate cells and they would never be compared.
    # Cell footprint ≈ 8 voxels keeps local context; min_sep sits just above
    # the fusion voxel (closer than that, fusion itself would have merged
    # the sheets into one surface).
    cell = max(LAYER_CELL_FACTOR * voxel, 1e-6)
    min_sep = max(LAYER_MIN_SEP_FACTOR * voxel, 1e-6)

    keys2d = np.floor(xyz[:, :2] / cell).astype(np.int64)  # column id (x, y)
    # Pack column keys (21 bits/axis is plenty for these scene sizes).
    kx = keys2d[:, 0] + 2**20
    ky = keys2d[:, 1] + 2**20
    packed = (kx << 21) | ky

    order = np.argsort(packed, kind="stable")
    packed_sorted = packed[order]
    bounds = np.flatnonzero(np.diff(packed_sorted)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    sizes = ends - starts

    # Candidate cells: enough points to mean a surface, few enough to be local.
    # Cap ≈ n/20 keeps the per-cell Python loop O(n) overall.
    cand = np.flatnonzero((sizes >= 8) & (sizes <= 4000))
    if len(cand) > max_seeds:
        cand = rng.choice(cand, size=max_seeds, replace=False)

    pair_seps: list[float] = []
    layered_pts = 0
    layered_cells = 0
    excluded = 0
    cand_pts = 0
    nrm_all = normals if (normals is not None and len(normals) == n) else None

    for ci in cand:
        s, e = starts[ci], ends[ci]
        idx = order[s:e]
        pts = xyz[idx]
        if nrm_all is not None:
            nvec = nrm_all[idx].mean(axis=0)
        else:
            # No per-point normals: project on world Z. Within one (x, y)
            # column the Z axis IS the surface-stacking direction by
            # construction, and column extents bound in-plane contamination
            # of the measured gap (in-plane spread ≈ cell footprint). No
            # planarity pre-filter: stacked sheets make a column NON-planar
            # — that is the signature we are detecting, so it must not be
            # excluded.
            nvec = np.array([0.0, 0.0, 1.0])
        norm = np.linalg.norm(nvec)
        if norm < 1e-9:
            continue
        proj = (pts - pts.mean(axis=0)) @ (nvec / norm)
        pj = np.sort(proj)
        gaps = np.diff(pj)
        # Count distinct sheets: 1 + number of gaps > min_sep.
        big = gaps > min_sep
        sheets = int(big.sum()) + 1
        cand_pts += e - s
        if sheets >= 2:
            layered_cells += 1
            layered_pts += e - s
            gs = gaps[big]
            pair_seps.extend(gs.tolist())

    measured = int(cand_pts)
    result: dict[str, Any] = {
        "status": "measured",
        "points": int(n),
        "cells_examined": int(len(cand)),
        "cells_excluded_planarity": int(excluded),
        "points_in_examined_cells": measured,
        "min_separation_m": round(float(min_sep), 3),
        "layer_region_count": int(layered_cells),
    }
    if pair_seps:
        seps = np.asarray(pair_seps)
        result.update({
            "layer_pair_count": int(len(seps)),
            "median_layer_separation_m": round(float(np.median(seps)), 3),
            "p95_layer_separation_m": round(float(np.percentile(seps, 95)), 3),
            "max_layer_separation_m": round(float(seps.max()), 3),
        })
    else:
        result.update({
            "layer_pair_count": 0,
            "median_layer_separation_m": None,
            "p95_layer_separation_m": None,
            "max_layer_separation_m": None,
        })
    layered_pct = 100.0 * layered_pts / measured if measured else 0.0
    result["layered_points_percent"] = round(float(layered_pct), 2)
    result["layered_points_percent_basis"] = "points in examined cells (8..4000 per cell)"
    return result


# --------------------------------------------------------------------------
# Layer source classification
# --------------------------------------------------------------------------

def classify_layer_source(
    dense_layers: dict,
    mesh_layers: dict,
    sparse_xyz: np.ndarray | None,
    dense_xyz: np.ndarray | None,
    mesh_v: np.ndarray | None,
    voxel: float,
) -> dict[str, Any]:
    """Evidence-based CASE A/B/C/D classification (Phase 1B Part 6)."""
    dl = dense_layers.get("layered_points_percent")
    ml = mesh_layers.get("layered_points_percent")
    ok_d = dense_layers.get("status") == "measured"
    ok_m = mesh_layers.get("status") == "measured"

    if not (ok_d and ok_m) or dl is None or ml is None:
        classification = "CASE_D_insufficient_evidence"
    elif dl < 1.0 and ml < 1.0:
        classification = "CASE_D_insufficient_evidence"  # no measurable layering anywhere
    elif dl >= 1.0 and ml >= 1.0 and max(dl, ml) <= 3.0 * min(dl, ml):
        # Comparable layering on both sides — both contribute.
        classification = "CASE_C_both_contribute"
    elif dl >= 5.0 and ml < 2.0 * dl:
        classification = "CASE_A_dense_origin"
    elif ml >= 2.0 * max(dl, 1.0):
        classification = "CASE_B_mesh_origin"
    else:
        classification = "CASE_D_insufficient_evidence"

    evidence = {
        "dense_layered_points_percent": dl,
        "mesh_layered_points_percent": ml,
        "dense_layer_pair_count": dense_layers.get("layer_pair_count"),
        "mesh_layer_pair_count": mesh_layers.get("layer_pair_count"),
        "dense_median_separation_m": dense_layers.get("median_layer_separation_m"),
        "mesh_median_separation_m": mesh_layers.get("median_layer_separation_m"),
        "threshold_rule": "mesh ≥ 2× dense (or dense <1% & mesh ≥1%) → mesh origin; "
                          "dense ≥5% and mesh not ≥2× dense → dense origin; "
                          "both ≥1% otherwise → both; below → insufficient",
    }
    limitations = [
        "Detector samples cells of LAYER_CELL_FACTOR×scale; layers finer than LAYER_MIN_SEP_FACTOR×scale separation are not counted.",
        "Percentages are relative to points in examinable cells, not the whole cloud.",
        "Mesh vertices are BPA outputs: a mesh-only layer can also be a dense layer BPA "
        "resolves differently — CASE B means the extra sheets appear only at mesh stage.",
    ]
    if sparse_xyz is not None and len(sparse_xyz) and dense_xyz is not None and len(dense_xyz):
        limitations.append(
            "Sparse cloud is not layer-scanned (100k pts, irregular density): "
            "sparse evidence is the NN-consistency report, not this detector."
        )
    return {
        "classification": classification,
        "evidence": evidence,
        "voxel_m": float(voxel),
        "confidence": "HEURISTIC — threshold rule above, not a calibrated probability",
        "limitations": limitations,
    }


# --------------------------------------------------------------------------
# Cross-view consistency + occlusion (Phase 1B Parts 4 + 8)
# --------------------------------------------------------------------------

@dataclass
class CrossViewEvidence:
    """Per-point multi-view evidence for a cloud — the single owner.

    Both the reported statistics and the contradiction filter read these
    arrays, so the two can never disagree about what a view said.

    Every array is indexed by GLOBAL cloud point. The pre-2026-09
    implementation indexed by position *within a chunk* while writing into
    global-sized accumulators, so with more than one chunk every view's
    votes landed on the same first ``chunk`` points: it reported
    ``points_with_any_observable_view == chunk`` and
    ``median_observable_views`` of 50 with a per-point sample of 8, i.e.
    statistics for points it had never looked at.
    """

    observable: np.ndarray        # int32 — sampled views that read a valid depth
    agree: np.ndarray             # int32 — views agreeing within the point's budget
    closer: np.ndarray            # int32 — views seeing a CLOSER surface here
    behind: np.ndarray            # int32 — views seeing background behind the point
    max_shortfall_m: np.ndarray   # float32 — max (z_pred - z_meas): how far in front
    min_range_m: np.ndarray       # float32 — nearest sampled camera distance
    views_sampled: int            # per-point sample size actually used (k)


def depth_uncertainty_budget(
    min_range_m: np.ndarray,
    relative_err: float | None = None,
    floor_m: float = 0.0,
) -> np.ndarray:
    """Per-point depth-uncertainty budget ``σZ/Z × range``.

    The same relative-depth model the triangulator accepts tracks under
    (``MAX_RELATIVE_DEPTH_ERR``), reused here so a measurement is judged
    against the uncertainty its own geometry can support. A single
    scene-wide voxel cannot do this: at the far end of a deep scene a
    fixed voxel is many times the point's own uncertainty, which is how
    duplicate sheets of a few metres' separation survive fusion.
    """
    from app.services.camera_pose_estimator import MAX_RELATIVE_DEPTH_ERR

    rel = MAX_RELATIVE_DEPTH_ERR if relative_err is None else float(relative_err)
    budget = rel * np.asarray(min_range_m, dtype=np.float64)
    return np.maximum(budget, float(floor_m))


def cross_view_evidence(
    cloud_xyz: np.ndarray,
    views: list[Any],
    budget_m: np.ndarray | float,
    *,
    max_views: int = 8,
    chunk: int = 250_000,
) -> CrossViewEvidence:
    """Project EVERY cloud point into its nearest views and read those maps.

    For each fused point the up to ``max_views`` cameras whose centres are
    nearest the point are chosen. The point is projected into each chosen
    view with the canonical convention (x ~ K R^T (X - C)) and the projected
    depth compared against that view's depth map:

    * ``|z_pred - z_meas| <= budget`` → the view corroborates the surface.
    * ``z_meas < z_pred - budget``    → the view sees a CLOSER surface here:
      the point is contradicted (its claimed surface is not what the view
      observed).
    * ``z_meas > z_pred + budget``    → the view sees background behind it.
    * map invalid / outside frame     → not observable.

    ``budget_m`` may be a scalar (the run's surface scale) or a per-point
    array (the uncertainty budget of ``depth_uncertainty_budget``). The
    per-point form is the honest one for filtering.

    This measures the whole cloud; it does not sample a subset.
    """
    n = len(cloud_xyz)
    if n == 0 or not views:
        return CrossViewEvidence(
            observable=np.zeros(0, np.int32), agree=np.zeros(0, np.int32),
            closer=np.zeros(0, np.int32), behind=np.zeros(0, np.int32),
            max_shortfall_m=np.zeros(0, np.float32),
            min_range_m=np.zeros(0, np.float32), views_sampled=0,
        )

    # Nearest views per point via camera centres (cheap KDTree on centres).
    from scipy.spatial import cKDTree

    centers = np.asarray([np.asarray(v.t, dtype=np.float64) for v in views])
    cent_tree = cKDTree(centers)
    k = min(max_views, len(views))
    _, view_idx = cent_tree.query(cloud_xyz, k=k)
    if k == 1:
        view_idx = view_idx[:, None]

    observable = np.zeros(n, dtype=np.int32)
    agree = np.zeros(n, dtype=np.int32)
    closer = np.zeros(n, dtype=np.int32)
    behind = np.zeros(n, dtype=np.int32)
    max_shortfall = np.zeros(n, dtype=np.float32)
    min_range = np.full(n, np.inf, dtype=np.float32)
    budget = np.broadcast_to(np.asarray(budget_m, dtype=np.float64), (n,))

    # Group points by the views that need their depth maps loaded (load each
    # view's map once).
    depth_cache: dict[int, np.ndarray] = {}
    for vi in range(len(views)):
        members = np.flatnonzero((view_idx == vi).any(axis=1))
        if len(members) == 0:
            continue
        if vi not in depth_cache:
            depth_cache[vi] = np.asarray(views[vi].depth, dtype=np.float64)
        dmap = depth_cache[vi]
        v = views[vi]
        R = np.asarray(v.R, dtype=np.float64)
        C = np.asarray(v.t, dtype=np.float64)
        K = np.asarray(v.K, dtype=np.float64)
        h, w = dmap.shape
        pts = cloud_xyz[members]
        # chunk to bound memory
        for s in range(0, len(members), chunk):
            sl = slice(s, s + chunk)
            glob = members[sl]                     # GLOBAL cloud indices
            Xc = (pts[sl] - C) @ R
            z = Xc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (Xc[:, 0] / np.maximum(z, 1e-9)) * K[0, 0] + K[0, 2]
                vv = (Xc[:, 1] / np.maximum(z, 1e-9)) * K[1, 1] + K[1, 2]
            inb = (z > 0.2) & (u >= 0) & (u < w - 1) & (vv >= 0) & (vv < h - 1)
            sel = np.flatnonzero(inb)
            if sel.size == 0:
                continue
            z_meas = dmap[
                np.clip(vv[sel].astype(np.int64), 0, h - 1),
                np.clip(u[sel].astype(np.int64), 0, w - 1),
            ]
            good = np.isfinite(z_meas) & (z_meas > 0.2)
            if not good.any():
                continue
            gi = glob[sel[good]]                   # index into the CLOUD
            z_pred = z[sel[good]]
            z_meas = z_meas[good]
            b = budget[gi]
            # Each point appears at most once per (view, chunk), so plain
            # fancy-index accumulation here cannot collide.
            range_pt = np.linalg.norm(cloud_xyz[gi] - C, axis=1)
            min_range[gi] = np.minimum(min_range[gi], range_pt)
            observable[gi] += 1
            diff = z_meas - z_pred
            agree[gi[np.abs(diff) <= b]] += 1
            behind[gi[diff > b]] += 1
            is_closer = diff < -b
            if is_closer.any():
                gc = gi[is_closer]
                closer[gc] += 1
                cur = max_shortfall[gc]
                np.maximum(cur, (-diff[is_closer]).astype(np.float32), out=cur)
                max_shortfall[gc] = cur

    min_range[~np.isfinite(min_range)] = 0.0
    return CrossViewEvidence(
        observable=observable, agree=agree, closer=closer, behind=behind,
        max_shortfall_m=max_shortfall, min_range_m=min_range, views_sampled=int(k),
    )


def cross_view_consistency(
    cloud_xyz: np.ndarray,
    views: list[Any],
    voxel_m: float,
    *,
    max_views: int = 8,
    chunk: int = 250_000,
) -> dict[str, Any]:
    """Summarise cross-view corroboration at the run's surface scale.

    Threshold = fusion voxel, i.e. the run's own surface scale. This is a
    MEASUREMENT: it never mutates the cloud. Occlusion handling in fusion
    itself is the sibling-split in ``voxel_merge`` (measurements of
    different surfaces sharing a voxel are never averaged — they become
    separate output points), and ``contradiction_mask`` turns the evidence
    below into the filtering decision when the caller wants one.
    """
    n = len(cloud_xyz)
    if n == 0 or not views:
        return {"status": "insufficient_points", "points": int(n)}

    ev = cross_view_evidence(cloud_xyz, views, float(voxel_m),
                             max_views=max_views, chunk=chunk)
    obs = ev.observable > 0
    result: dict[str, Any] = {
        "status": "measured",
        "points": int(n),
        "views_sampled_per_point_max": int(ev.views_sampled),
        "agreement_threshold_m": round(float(voxel_m), 3),
        "threshold_rule": "|z_pred - z_meas| <= fusion voxel → corroborated; z_meas closer by > voxel → contradicted",
        "points_examined": int(n),
        "whole_cloud": True,
    }
    if obs.any():
        result["points_with_any_observable_view"] = int(obs.sum())
        result["observable_fraction_pct"] = round(float(100.0 * obs.mean()), 2)
        result["cross_view_corroborated_pct"] = round(float(100.0 * (ev.agree[obs] >= 1).mean()), 2)
        result["occlusion_isolated_pct"] = round(
            float(100.0 * ((ev.agree[obs] == 0) & (ev.closer[obs] >= 1)).mean()), 2)
        result["median_observable_views"] = float(np.median(ev.observable[obs]))
        contradicted = obs & (ev.closer >= 1)
        if contradicted.any():
            result["contradicted_points"] = int(contradicted.sum())
            short = ev.max_shortfall_m[contradicted]
            result["median_contradiction_m"] = round(float(np.median(short)), 3)
            result["p95_contradiction_m"] = round(float(np.percentile(short, 95)), 3)
    result["labels"] = {
        "cross_view_corroborated_pct": "MEASURED (documented threshold)",
        "occlusion_isolated_pct": "MEASURED — points not corroborated by any sampled view "
                                  "and behind a closer surface in ≥1: duplicate-layer / hallucination candidates",
        "observable_fraction_pct": "MEASURED — share of the WHOLE cloud any sampled view reads",
    }
    return result


def min_camera_range(
    points: np.ndarray,
    camera_centers: np.ndarray,
    *,
    chunk: int = 250_000,
) -> np.ndarray:
    """Distance from every point to the nearest camera centre (m).

    The range scale the depth-uncertainty budget is derived from. The
    sparse→dense screen uses the same definition (nearest observing
    range), so a point is judged by one consistent model everywhere.
    """
    from scipy.spatial import cKDTree

    centers = np.asarray(camera_centers, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    tree = cKDTree(centers)
    out = np.empty(len(points), dtype=np.float64)
    for s in range(0, len(points), chunk):
        sl = slice(s, s + chunk)
        d, _ = tree.query(points[sl], k=1, workers=-1)
        out[sl] = d
    return out


def contradiction_mask(
    evidence: CrossViewEvidence,
    *,
    min_contradicting: int = 1,
    max_agreeing: int = 0,
) -> np.ndarray:
    """Points no observing view supports and at least one view contradicts.

    The comparison itself already used each point's own depth-uncertainty
    budget (see ``depth_uncertainty_budget``), so a point is only counted
    here when a view saw a surface closer than the point claims by MORE
    than that point's geometry can explain.

    ``max_agreeing=0`` is the conservative rule: a single corroborating
    view rescues a point. Points no view can observe are never masked —
    unobservable is not evidence of error.
    """
    return (evidence.closer >= int(min_contradicting)) & (evidence.agree <= int(max_agreeing))


def contradiction_report(
    evidence: CrossViewEvidence,
    *,
    min_contradicting: int = 1,
    max_agreeing: int = 0,
) -> dict[str, Any]:
    """The named, countable account of a contradiction-based removal.

    Reports the rule actually applied, how many points it judges, how many
    it removes, and — separately — the points it could NOT judge because no
    sampled view reads them. A coverage hole must never be reported as an
    accuracy failure, or a refusal gets counted as error.
    """
    n = len(evidence.observable)
    if n == 0:
        return {"status": "insufficient_points", "points": 0}
    obs = evidence.observable > 0
    mask = contradiction_mask(
        evidence, min_contradicting=min_contradicting, max_agreeing=max_agreeing
    )
    n_obs = int(obs.sum())
    removed = int(mask.sum())
    report: dict[str, Any] = {
        "status": "measured",
        "points": int(n),
        "rule": (
            f"contradicted by >= {int(min_contradicting)} sampled view(s) with <= "
            f"{int(max_agreeing)} corroborating, beyond the point's own "
            "depth-uncertainty budget (MAX_RELATIVE_DEPTH_ERR x observing range)"
        ),
        "views_sampled_per_point_max": int(evidence.views_sampled),
        "judged_points": n_obs,
        "removed_points": removed,
        "removed_pct_of_judged": round(float(100.0 * removed / n_obs), 2) if n_obs else 0.0,
        "removed_pct_of_cloud": round(float(100.0 * removed / n), 2),
        "unobservable_points": int(n - n_obs),
        "unobservable_note": "no sampled view reads depth here — a coverage question, not an accuracy one",
    }
    if removed:
        short = evidence.max_shortfall_m[mask]
        report["median_shortfall_m"] = round(float(np.median(short)), 3)
        report["p95_shortfall_m"] = round(float(np.percentile(short, 95)), 3)
        report["median_range_m"] = round(float(np.median(evidence.min_range_m[mask])), 3)
    return report


# --------------------------------------------------------------------------
# Mesh component classification (Phase 1B Part 17)
# --------------------------------------------------------------------------

def classify_mesh_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    support_dist: np.ndarray | None,
    threshold_m: float,
    labels: np.ndarray | None = None,
    n_comp: int | None = None,
) -> dict[str, Any]:
    """Classify every connected mesh component — never auto-delete.

    Classes:
      supported_large   ≥100 verts, median support ≤ threshold
      supported_small   <100 verts, median support ≤ threshold (a small but
                        real structure must survive)
      unsupported       median support > 2×threshold
      weakly_supported  everything between (supported size, unsupported distance)
      noise_candidate   <10 verts AND unsupported
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    nv = len(vertices)
    if nv == 0 or len(faces) == 0:
        return {"status": "insufficient_points"}
    if labels is None or n_comp is None:
        rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nv, nv)).tocsr()
        n_comp, labels = connected_components(adj, directed=False)
    n_comp = int(n_comp)

    # Group the vertices by component ONCE (stable sort) and reduce per group.
    # The boolean-mask loop this replaces scanned all nv vertices for every
    # component — O(n_comp x nv), ~10^10 comparisons on a 2.4M-vertex mesh —
    # and dominated the mesh audit. Every statistic below (count, bbox,
    # median support) is order-independent, so the numbers are identical.
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=n_comp).astype(np.int64)
    starts = np.zeros(n_comp + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    grouped_v = vertices[order]
    gstart = starts[:-1]
    bbox_min = np.minimum.reduceat(grouped_v, gstart, axis=0)
    bbox_max = np.maximum.reduceat(grouped_v, gstart, axis=0)

    have_support = support_dist is not None and len(support_dist) == nv
    med_by_comp = np.full(n_comp, np.nan)
    if have_support:
        grouped_s = np.asarray(support_dist, dtype=np.float64)[order]
        med_by_comp = np.array([
            np.median(grouped_s[starts[cid]:starts[cid + 1]])
            for cid in range(n_comp)
        ])

    comps: list[dict[str, Any]] = []
    for cid in range(n_comp):
        cnt = int(counts[cid])
        entry: dict[str, Any] = {
            "component_id": int(cid),
            "vertices": cnt,
            "bbox_min": [round(float(x), 1) for x in bbox_min[cid]],
            "bbox_max": [round(float(x), 1) for x in bbox_max[cid]],
        }
        if have_support:
            med = float(med_by_comp[cid])
            entry["median_support_distance_m"] = round(med, 3)
            if cnt < 10 and med > threshold_m:
                entry["class"] = "noise_candidate"
            elif med <= threshold_m:
                entry["class"] = "supported_large" if cnt >= 100 else "supported_small"
            elif med > 2.0 * threshold_m:
                entry["class"] = "unsupported"
            else:
                entry["class"] = "weakly_supported"
        else:
            entry["class"] = "unclassified (no support distances)"
        comps.append(entry)
    comps.sort(key=lambda e: -e["vertices"])
    by_class: dict[str, int] = {}
    for e in comps:
        by_class[e["class"]] = by_class.get(e["class"], 0) + 1
    return {
        "status": "measured",
        "component_count": int(n_comp),
        "class_counts": by_class,
        "components_top20": comps[:20],
        "rule": "supported: median support ≤ fusion voxel; unsupported: > 2×; small components are NOT auto-deleted",
    }


# --------------------------------------------------------------------------
# Dense → mesh support
# --------------------------------------------------------------------------

def mesh_support(
    mesh_v: np.ndarray,
    dense_xyz: np.ndarray,
    threshold_m: float,
    faces: np.ndarray | None = None,
    max_points: int = 4_000_000,
    support_dist: np.ndarray | None = None,
) -> dict[str, Any]:
    """Nearest-dense-point distance for every mesh vertex.

    ``threshold_m`` should be the run's fusion voxel (the measured surface
    scale): a vertex with no dense observation within one voxel is not
    carried by observed geometry.

    ``support_dist`` lets a caller that already queried these same vertices
    against this same cloud (the mesh audit does, for component support)
    pass the distances in rather than rebuild and re-query the KD-tree.
    """
    from scipy.spatial import cKDTree

    if len(mesh_v) == 0 or len(dense_xyz) == 0:
        return {"status": "insufficient_points"}
    if support_dist is not None and len(support_dist) == len(mesh_v):
        d = np.asarray(support_dist, dtype=np.float64)
    else:
        dense = dense_xyz if len(dense_xyz) <= max_points else dense_xyz[:: int(np.ceil(len(dense_xyz) / max_points))]
        d, _ = cKDTree(dense).query(mesh_v, k=1)
    supported = d <= threshold_m
    result: dict[str, Any] = {
        "status": "measured",
        "support_distance_threshold_m": round(float(threshold_m), 3),
        "supported_vertices": int(supported.sum()),
        "unsupported_vertices": int((~supported).sum()),
        "support_percent": round(float(100.0 * supported.mean()), 2),
        "median_support_distance_m": round(float(np.median(d)), 3),
        "p95_support_distance_m": round(float(np.percentile(d, 95)), 3),
    }
    if faces is not None and len(faces) and (~supported).any():
        # Connected regions of unsupported vertices (diagnostic only).
        try:
            import scipy.sparse as sp
            from scipy.sparse.csgraph import connected_components

            unsup = np.flatnonzero(~supported)
            unsup_set = np.zeros(len(mesh_v), dtype=bool)
            unsup_set[unsup] = True
            f = faces
            keep = unsup_set[f].any(axis=1)
            f2 = f[keep]
            # Relabel to compact unsupported ids; drop faces touching supported verts.
            remap = -np.ones(len(mesh_v), dtype=np.int64)
            remap[unsup] = np.arange(len(unsup))
            v2 = remap[f2]
            good = (v2 >= 0).all(axis=1)
            f3 = v2[good]
            if len(f3) >= 3:
                rows = np.concatenate([f3[:, 0], f3[:, 1], f3[:, 2]])
                cols = np.concatenate([f3[:, 1], f3[:, 2], f3[:, 0]])
                adj = sp.coo_matrix(
                    (np.ones(len(rows)), (rows, cols)), shape=(len(unsup), len(unsup))
                ).tocsr()
                n_comp, labels = connected_components(adj, directed=False)
                comp_sizes = np.bincount(labels)
                comp_sizes = comp_sizes[comp_sizes > 0]
                # bbox of each unsupported region (top 5 by size)
                boxes = []
                for cid in np.argsort(comp_sizes)[::-1][:5]:
                    sel = labels == cid
                    if sel.sum() < 3:
                        continue
                    cv = mesh_v[unsup[sel]]
                    boxes.append({
                        "vertices": int(sel.sum()),
                        "bbox_min": [round(float(x), 1) for x in cv.min(axis=0)],
                        "bbox_max": [round(float(x), 1) for x in cv.max(axis=0)],
                    })
                result["unsupported_regions"] = int(len(comp_sizes))
                result["unsupported_region_bboxes_top5"] = boxes
        except Exception as exc:  # pragma: no cover — diagnostics must never crash the stage
            log.warning("unsupported_region_analysis_failed", error=str(exc))
    return result


# --------------------------------------------------------------------------
# Extended mesh quality (Phase 1B Part 19)
# --------------------------------------------------------------------------

def extended_mesh_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
    voxel: float,
) -> dict[str, Any]:
    """Non-manifold/degenerate/normal-conflict/extents — vectorised."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    out: dict[str, Any] = {
        "x_extent_m": round(float(v[:, 0].max() - v[:, 0].min()), 2) if len(v) else None,
        "y_extent_m": round(float(v[:, 1].max() - v[:, 1].min()), 2) if len(v) else None,
        "z_extent_m": round(float(v[:, 2].max() - v[:, 2].min()), 2) if len(v) else None,
    }
    if len(f) == 0:
        return out
    # Degenerate faces: repeated vertex index or (near-)zero area.
    rep = (
        (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    )
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    degenerate = rep | (area2 <= 1e-12)
    out["degenerate_faces"] = int(degenerate.sum())

    # Edge census via sorted-pair unique (vectorised).
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    e_view = np.ascontiguousarray(e).view([("a", e.dtype), ("b", e.dtype)])
    uniq, counts = np.unique(e_view, return_counts=True)
    out["non_manifold_edges"] = int((counts > 2).sum())
    out["boundary_edges"] = int((counts == 1).sum())

    # Normal conflict: angle between the two faces sharing an interior edge.
    # Integer edge keys (a*nv+b, a<b) — searchsorted cannot handle structured
    # views, but int64 keys work and cannot collide while a<b<nv.
    interior = np.flatnonzero(counts == 2)
    if len(interior):
        nv = len(v)
        edge_key = uniq["a"].astype(np.int64) * nv + uniq["b"].astype(np.int64)
        face_edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        face_ids = np.tile(np.arange(len(f)), 3)
        fe = np.sort(face_edges, axis=1)
        fe_key = fe[:, 0].astype(np.int64) * nv + fe[:, 1].astype(np.int64)
        order = np.argsort(fe_key, kind="stable")
        fe_sorted = fe_key[order]
        fid_sorted = face_ids[order]
        pos = np.searchsorted(fe_sorted, edge_key[interior])
        pos = np.minimum(pos, len(fe_sorted) - 1)
        f1 = fid_sorted[pos]
        nxt = np.minimum(pos + 1, len(fe_sorted) - 1)
        f2 = fid_sorted[nxt]
        valid = (fe_sorted[pos] == edge_key[interior]) & (fe_sorted[nxt] == edge_key[interior]) & (f1 != f2)
        f1, f2 = f1[valid], f2[valid]
        if len(f1):
            n1 = cross[f1] / (area2[f1, None] + 1e-12)
            n2 = cross[f2] / (area2[f2, None] + 1e-12)
            dot = np.clip((n1 * n2).sum(axis=1), -1.0, 1.0)
            conflict = dot < 0.0  # faces meeting at >90° across a shared edge
            out["normal_conflict_percent"] = round(float(100.0 * conflict.mean()), 2)
            out["adjacent_face_pairs_sampled"] = int(len(f1))
    out.setdefault("normal_conflict_percent", None)
    out["threshold_note"] = "conflict = adjacent face normals with negative dot product (dihedral > 90°)"
    return out


def dense_quality_report(
    quality_dict: dict,
    dense_layers: dict,
    support: dict,
    voxel: float,
    counts: dict,
) -> dict[str, Any]:
    """Assemble dense_quality_report.json with honest per-metric labels."""
    return {
        "voxel_size_m": float(voxel),
        "counts": {
            "raw_fused": counts.get("raw"),
            "after_filter": counts.get("filtered"),
            "removed_percent": counts.get("removed_pct"),
        },
        "spacing": quality_dict.get("spacing", {}),
        "quality": quality_dict,
        "layers": dense_layers,
        "mesh_support": support,
        "labels": {
            "counts": "MEASURED",
            "spacing": "MEASURED",
            "layers": "MEASURED (documented thresholds; see detect_layers)",
            "mesh_support": "MEASURED (threshold = fusion voxel)",
            "quality_score": "HEURISTIC",
        },
    }
