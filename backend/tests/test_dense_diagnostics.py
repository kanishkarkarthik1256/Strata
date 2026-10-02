"""Phase 1B diagnostics regression tests.

Pins the evidence-based layer detector, layer-source classifier,
dense→mesh support measurement and extended mesh-quality audit.
Synthetic geometry only — no GT data, no repository mutation.
"""

import numpy as np
import pytest

from app.services.dense_diagnostics import (
    classify_layer_source,
    classify_mesh_components,
    contradiction_mask,
    contradiction_report,
    cross_view_consistency,
    cross_view_evidence,
    depth_uncertainty_budget,
    detect_layers,
    extended_mesh_quality,
    mesh_support,
    min_camera_range,
)


@pytest.fixture(scope="module")
def voxel() -> float:
    return 1.5


def _rng():
    return np.random.default_rng(0)


def test_detect_layers_finds_duplicate_sheet(voxel):
    rng = _rng()
    ground = np.column_stack([
        rng.uniform(0, 200, 4000),
        rng.uniform(0, 200, 4000),
        rng.normal(0, 0.2, 4000),
    ])
    dup = ground.copy()
    dup[:, 2] = ground[:, 2] + 6.0  # 4× voxel above ground
    res = detect_layers(np.vstack([ground, dup]), voxel)
    assert res["status"] == "measured"
    assert res["layer_region_count"] > 0
    assert res["layered_points_percent"] > 5.0
    assert res["median_layer_separation_m"] == pytest.approx(6.0, abs=1.0)


def test_detect_layers_single_surface_is_clean(voxel):
    rng = _rng()
    ground = np.column_stack([
        rng.uniform(0, 200, 4000),
        rng.uniform(0, 200, 4000),
        rng.normal(0, 0.2, 4000),
    ])
    res = detect_layers(ground, voxel)
    assert res["status"] == "measured"
    assert res["layer_pair_count"] == 0
    assert res["layered_points_percent"] == 0.0


def test_detect_layers_too_few_points_is_honest(voxel):
    res = detect_layers(np.zeros((10, 3)), voxel)
    assert res["status"] == "insufficient_points"
    assert res["points"] == 10


def test_classifier_cases(voxel):
    rng = _rng()
    ground = np.column_stack([
        rng.uniform(0, 200, 4000),
        rng.uniform(0, 200, 4000),
        rng.normal(0, 0.2, 4000),
    ])
    dup = ground.copy()
    dup[:, 2] = ground[:, 2] + 6.0
    clean = detect_layers(ground, voxel)
    layered = detect_layers(np.vstack([ground, dup]), voxel)

    # CASE B: dense clean, mesh layered → meshing creates layers.
    assert classify_layer_source(clean, layered, None, None, None, voxel)[
        "classification"] == "CASE_B_mesh_origin"
    # CASE A: dense layered, mesh clean → dense origin.
    assert classify_layer_source(layered, clean, None, None, None, voxel)[
        "classification"] == "CASE_A_dense_origin"
    # CASE C: both layered.
    assert classify_layer_source(layered, layered, None, None, None, voxel)[
        "classification"] == "CASE_C_both_contribute"
    # CASE D: detector failed on one side.
    assert classify_layer_source({"status": "insufficient_points"}, layered, None, None, None, voxel)[
        "classification"] == "CASE_D_insufficient_evidence"


def test_mesh_support_measures_fabricated_geometry(voxel):
    from scipy.spatial import Delaunay

    rng = _rng()
    dense = np.column_stack([
        rng.uniform(0, 200, 40000),
        rng.uniform(0, 200, 40000),
        rng.normal(0, 0.2, 40000),
    ])
    # Mesh vertices on the surface → fully supported.
    on_surface = dense[::4]
    tri = Delaunay(on_surface[:, :2])
    sup = mesh_support(on_surface, dense, threshold_m=voxel, faces=tri.simplices)
    assert sup["support_percent"] == pytest.approx(100.0)

    # Mesh lifted 30 m off the surface → fully unsupported, one region.
    floating = on_surface.copy()
    floating[:, 2] += 30.0
    sup2 = mesh_support(floating, dense, threshold_m=voxel, faces=tri.simplices)
    assert sup2["support_percent"] == 0.0
    assert sup2["unsupported_regions"] == 1
    box = sup2["unsupported_region_bboxes_top5"][0]
    assert box["vertices"] == len(floating)
    assert box["bbox_min"][2] > 25.0


def test_extended_mesh_quality_counts(voxel):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [5, 5, 5]], float)
    # face 3 is degenerate (repeated vertex); faces 1+2 share edge (1,2);
    # face 0+1 share two edges with compatible normals; edge (0,3) absent → boundary.
    f = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 1]])
    q = extended_mesh_quality(v, f, voxel)
    assert q["degenerate_faces"] == 1
    assert q["boundary_edges"] == 4
    assert q["non_manifold_edges"] == 1
    assert q["normal_conflict_percent"] is not None
    assert q["x_extent_m"] == pytest.approx(5.0)
    assert q["z_extent_m"] == pytest.approx(5.0)


def _depth_views_from_plane(rng):
    """Two cameras at z=-100 looking up at a z=0 plane; maps store Z-depth."""
    from app.services.depth_fusion import DepthView

    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    views = []
    for t in ([50.0, 50.0], [55.0, 50.0]):
        uu, vv = np.meshgrid(np.arange(100), np.arange(100), indexing="xy")
        dirc = np.stack([(uu - 50) / 100, (vv - 50) / 100, np.ones_like(uu, float)], axis=-1)
        C = np.array([t[0], t[1], -100.0])
        tt = (0 - C[2]) / dirc[..., 2]
        X = C[0] + dirc[..., 0] * tt
        Y = C[1] + dirc[..., 1] * tt
        inside = (X >= 0) & (X <= 100) & (Y >= 0) & (Y <= 100)
        d = np.where(inside, 100.0, 0.0)  # Z-depth of the z=0 plane
        views.append(DepthView(f"v{t}", d, None, K, np.eye(3), C))
    return views


def test_cross_view_consistency_corroborates_real_surface(voxel):
    rng = _rng()
    ground = np.column_stack([
        rng.uniform(0, 100, 5000), rng.uniform(0, 100, 5000), np.zeros(5000)])
    views = _depth_views_from_plane(rng)
    r = cross_view_consistency(ground[::5], views, voxel_m=voxel)
    assert r["status"] == "measured"
    assert r["cross_view_corroborated_pct"] == 100.0
    assert r["occlusion_isolated_pct"] == 0.0


def test_cross_view_consistency_flags_occluded_points(voxel):
    rng = _rng()
    ground = np.column_stack([
        rng.uniform(0, 100, 5000), rng.uniform(0, 100, 5000), np.zeros(5000)])
    views = _depth_views_from_plane(rng)
    behind = ground[::5].copy()
    behind[:, 2] = 20.0  # beyond the observed surface from these cameras
    r = cross_view_consistency(behind, views, voxel_m=voxel)
    assert r["cross_view_corroborated_pct"] == 0.0
    assert r["occlusion_isolated_pct"] == 100.0


def test_classify_mesh_components_never_autodeletes(voxel):
    from scipy.spatial import Delaunay

    rng = _rng()
    ground = np.column_stack([
        rng.uniform(0, 100, 5000), rng.uniform(0, 100, 5000), np.zeros(5000)])
    v = ground[::4]
    tri = Delaunay(v[:, :2])
    far = v[:100].copy()
    far[:, 0] += 500
    far[:, 2] += 40
    v2 = np.vstack([v, far])
    sd = np.concatenate([np.full(len(v), 0.5), np.full(100, 50.0)])
    f2 = np.vstack([tri.simplices, Delaunay(far[:, :2]).simplices + len(v)])
    cc = classify_mesh_components(v2, f2, sd, threshold_m=voxel)
    assert cc["component_count"] == 2
    assert cc["class_counts"] == {"supported_large": 1, "unsupported": 1}
    # the small far island (100 verts) must be classified, not deleted
    far_entry = [c for c in cc["components_top20"] if c["vertices"] == 100][0]
    assert far_entry["class"] == "unsupported"


def test_normal_conflict_detection(voxel):
    # Two consistently-wound triangles sharing edge (1,2); the second folds
    # back under the first so the face normals oppose (dot < 0) — a >90°
    # crease, the folded-sheet signature.
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.6, -0.4]], float)
    f = np.array([[0, 1, 2], [2, 1, 3]])
    q = extended_mesh_quality(v, f, voxel)
    assert q["normal_conflict_percent"] == 100.0
    # Gentle fold (both normals up): no conflict.
    v2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1.0]], float)
    q2 = extended_mesh_quality(v2, f, voxel)
    assert q2["normal_conflict_percent"] == 0.0


# ---------------------------------------------------------------------------
# Cross-view evidence / ghost rejection
# ---------------------------------------------------------------------------

def _plane_scene(n=3000, ghost_every=7, ghost_z=20.0, seed=0):
    """Points on the observed z=0 plane, every ``ghost_every``-th beyond it."""
    views = _depth_views_from_plane(_rng())
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(5, 95, n), rng.uniform(5, 95, n), np.zeros(n)])
    ghost_idx = np.arange(0, n, ghost_every)
    pts[ghost_idx, 2] = ghost_z  # farther from the cameras than the observed surface
    return pts, views, ghost_idx


def _brute_force_evidence(pts, views, budget):
    """Independent reference: one (point, view) pair at a time, no chunks."""
    from scipy.spatial import cKDTree

    centers = np.asarray([np.asarray(v.t, dtype=float) for v in views])
    _, view_idx = cKDTree(centers).query(pts, k=len(views))
    n = len(pts)
    obs = np.zeros(n, int)
    agree = np.zeros(n, int)
    closer = np.zeros(n, int)
    behind = np.zeros(n, int)
    for i in range(n):
        for vi in np.atleast_1d(view_idx[i]):
            v = views[vi]
            Xc = (pts[i] - np.asarray(v.t, float)) @ np.asarray(v.R, float)
            z = float(Xc[2])
            if z <= 0.2:
                continue
            u = Xc[0] / z * v.K[0, 0] + v.K[0, 2]
            vv = Xc[1] / z * v.K[1, 1] + v.K[1, 2]
            h, w = v.depth.shape
            if not (0 <= u < w - 1 and 0 <= vv < h - 1):
                continue
            z_meas = float(v.depth[int(vv), int(u)])
            if not np.isfinite(z_meas) or z_meas <= 0.2:
                continue
            obs[i] += 1
            diff = z_meas - z
            if abs(diff) <= budget:
                agree[i] += 1
            elif diff < -budget:
                closer[i] += 1
            else:
                behind[i] += 1
    return obs, agree, closer, behind


def test_evidence_votes_land_on_the_points_they_describe():
    """Regression: the chunk-index bug.

    Votes were accumulated with an index into the CHUNK slice while the
    accumulator was cloud-sized, so with more than one chunk every view's
    votes landed on the same first ``chunk`` points. The shipped run
    reported ``points_with_any_observable_view`` exactly equal to its
    250 000 chunk constant and a median of 50 observable views against a
    per-point sample of 8 — statistics for points never projected.

    A brute-force reference is the only thing that can pin this.
    """
    pts, views, _ = _plane_scene()
    chunk = 100
    ev = cross_view_evidence(pts, views, 1.5, chunk=chunk)

    # The bug could never accumulate more than one chunk's worth of votes.
    assert int(ev.observable.sum()) > chunk

    obs, agree, closer, behind = _brute_force_evidence(pts, views, 1.5)
    assert (ev.observable == obs).all()
    assert (ev.agree == agree).all()
    assert (ev.closer == closer).all()
    assert (ev.behind == behind).all()
    assert ev.views_sampled == len(views)


def test_evidence_max_shortfall_is_per_point():
    pts, views, ghost_idx = _plane_scene()
    ev = cross_view_evidence(pts, views, 1.5, chunk=100)
    # Points beyond the observed surface are contradicted; plane points are not.
    assert (ev.closer[ghost_idx] >= 1).all()
    on_plane = np.setdiff1d(np.arange(len(pts)), ghost_idx)
    assert (ev.closer[on_plane] == 0).all()
    assert ev.max_shortfall_m[ghost_idx].min() > 1.5
    assert ev.max_shortfall_m[on_plane].max() == 0.0


def test_summary_covers_the_whole_cloud():
    """The reported statistics must describe every point, not one chunk."""
    pts, views, _ = _plane_scene(n=2400)
    r = cross_view_consistency(pts, views, voxel_m=1.5)
    assert r["points_examined"] == len(pts)
    assert r["whole_cloud"] is True
    assert r["points_with_any_observable_view"] > 100  # far beyond a default chunk
    # A per-point sample of 2 cannot produce a median of 50 observable views.
    assert r["median_observable_views"] <= r["views_sampled_per_point_max"]


def test_contradiction_filter_removes_only_uncontradicted_by_nothing():
    pts, views, ghost_idx = _plane_scene()
    ev = cross_view_evidence(pts, views, 1.5, chunk=100)
    mask = contradiction_mask(ev)
    assert mask[ghost_idx].all()
    assert not mask[np.setdiff1d(np.arange(len(pts)), ghost_idx)].any()
    rep = contradiction_report(ev)
    assert rep["removed_points"] == len(ghost_idx)
    assert rep["judged_points"] == int((ev.observable > 0).sum())
    assert rep["removed_pct_of_cloud"] == round(100.0 * len(ghost_idx) / len(pts), 2)


def test_contradiction_filter_never_removes_unobservable_points():
    """A point no view reads is a coverage fact, not an accuracy failure."""
    rng = _rng()
    # Laterally far outside every camera's 100x100 field of view.
    pts = np.column_stack([
        rng.uniform(300, 500, 500), rng.uniform(300, 500, 500), np.zeros(500)])
    views = _depth_views_from_plane(rng)
    ev = cross_view_evidence(pts, views, 1.5, chunk=100)
    assert (ev.observable == 0).all()
    assert not contradiction_mask(ev).any()
    rep = contradiction_report(ev)
    assert rep["removed_points"] == 0
    assert rep["unobservable_points"] == 500
    assert rep["judged_points"] == 0


def test_depth_uncertainty_budget_grows_with_range():
    rng_m = np.array([10.0, 100.0, 1000.0])
    b = depth_uncertainty_budget(rng_m)
    from app.services.camera_pose_estimator import MAX_RELATIVE_DEPTH_ERR

    assert b[1] == pytest.approx(MAX_RELATIVE_DEPTH_ERR * 100.0)
    assert (np.diff(b) > 0).all()
    # A floor keeps a near-field budget from collapsing to zero.
    assert (depth_uncertainty_budget(rng_m, floor_m=0.5) >= 0.5).all()


def test_min_camera_range_is_the_nearest_centre_distance():
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    pts = np.array([[0.0, 0.0, 3.0], [10.0, 0.0, 4.0], [1000.0, 0.0, 0.0]])
    r = min_camera_range(pts, centers)
    assert r[0] == pytest.approx(3.0)
    assert r[1] == pytest.approx(4.0)
    assert r[2] == pytest.approx(990.0)


# ---------------------------------------------------------------------------
# Mesh-audit scaling guard
# ---------------------------------------------------------------------------

class _CountedArray(np.ndarray):
    """An ndarray that tallies how many of its own elements get examined.

    The mesh audit's component classifier once looped
    ``for cid in range(n_comp): mask = labels == cid`` — one full-array boolean
    comparison per component, i.e. O(n_comp x n_vertices) (~10^10 comparisons
    on the 2.36M-vertex reference mesh). Counting element touches pins that
    pattern without wall-clock timing, which would be flaky on a loaded host.

    Counted: ufunc inputs (catches ``labels == cid``), ``argsort``, and the
    source size read by a full-length boolean-mask gather. So the work of any
    per-component rescan shows up, however it is written.
    """

    def __new__(cls, arr, tally):
        obj = np.asarray(arr).view(cls)
        obj._tally = tally
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._tally = getattr(obj, "_tally", None)

    def _tick(self, n):
        if self._tally is not None:
            self._tally[0] += int(n)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Only our own arrays count; a plain companion (e.g. reduceat indices)
        # must not inflate the tally.
        for i in inputs:
            if isinstance(i, _CountedArray):
                self._tick(i.size)
        # Delegate on base arrays so ufunc.reduceat etc. see a plain ndarray.
        plain = tuple(
            np.asarray(i) if isinstance(i, _CountedArray) else i for i in inputs)
        if "out" in kwargs:
            kwargs["out"] = tuple(
                np.asarray(o) if isinstance(o, _CountedArray) else o
                for o in kwargs["out"])
        return getattr(ufunc, method)(*plain, **kwargs)

    def __getitem__(self, key):
        # A full-length row mask (1-D bool over the leading axis) is one pass.
        if (isinstance(key, np.ndarray) and key.dtype == bool
                and key.ndim == 1 and key.shape[0] == self.shape[0]):
            self._tick(self.size)
        return super().__getitem__(key)

    def argsort(self, *args, **kwargs):
        self._tick(self.size)
        return super().argsort(*args, **kwargs)


def _disjoint_component_mesh(n_comp, side, spacing=1000.0):
    """``n_comp`` disconnected triangulated ``side x side`` patches.

    Returns ``(vertices, faces, labels)`` with ``n_comp * side**2`` vertices
    and contiguous component ids, the composition the audit's connected-
    component pass would produce.
    """
    verts = []
    faces = []
    ii, jj = np.meshgrid(np.arange(side - 1), np.arange(side - 1), indexing="ij")
    a = (ii * side + jj).ravel()
    f_local = np.concatenate([
        np.stack([a, a + 1, a + side], axis=1),
        np.stack([a, a + side, a + side + 1], axis=1),
    ]).astype(np.int64)
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    for c in range(n_comp):
        v = np.column_stack([
            gx.ravel() + c * spacing, gy.ravel(), np.zeros(side * side)]).astype(float)
        verts.append(v)
        faces.append(f_local + c * side * side)
    labels = np.repeat(np.arange(n_comp, dtype=np.int64), side * side)
    return np.vstack(verts), np.vstack(faces), labels


def test_mesh_audit_component_work_does_not_scale_with_component_count(voxel):
    """The mesh audit's component step must stay linear in component count.

    Regression guard for the O(n_comp x n_vertices) mask loop that dominated
    the dense mesh audit (``classify_mesh_components``). Two meshes with the
    SAME vertex count but 4x the components must not cost ~4x more: the old
    pattern touched ``n_vertices`` elements once per component.
    """
    nv = 32 * 32 * 32  # identical for both meshes below
    v_a, f_a, lab_a = _disjoint_component_mesh(32, 32)
    v_b, f_b, lab_b = _disjoint_component_mesh(128, 16)
    assert len(v_a) == len(v_b) == nv == 128 * 16 * 16

    lab_t_a, lab_t_b = [0], [0]  # element touches on the label array
    vtx_t_a, vtx_t_b = [0], [0]  # element touches on the 2-D vertex array
    sd_a = np.full(nv, 0.5)
    sd_b = np.full(nv, 0.5)

    cc_a = classify_mesh_components(
        _CountedArray(v_a, vtx_t_a), f_a, sd_a, threshold_m=voxel,
        labels=_CountedArray(lab_a, lab_t_a), n_comp=32)
    cc_b = classify_mesh_components(
        _CountedArray(v_b, vtx_t_b), f_b, sd_b, threshold_m=voxel,
        labels=_CountedArray(lab_b, lab_t_b), n_comp=128)

    # The label array is where the old O(n_comp x nv) loop lived: it ran
    # ``labels == cid`` once per component. A linear implementation touches it
    # a small constant number of times regardless of component count.
    assert lab_t_a[0] <= 4 * nv
    assert lab_t_b[0] <= 4 * nv
    assert lab_t_b[0] <= 1.5 * lab_t_a[0]
    # The same 4x-component jump must not multiply vertex work either.
    assert vtx_t_b[0] <= 1.5 * vtx_t_a[0]
    assert vtx_t_a[0] <= 16 * nv

    # Output must stay correct, so the test cannot pass by degrading the result.
    assert cc_a["component_count"] == 32
    assert cc_b["component_count"] == 128
    assert cc_a["class_counts"] == {"supported_large": 32}
    assert cc_b["class_counts"] == {"supported_large": 128}
    assert cc_a["components_top20"][0]["vertices"] == 32 * 32
    assert cc_b["components_top20"][0]["vertices"] == 16 * 16
