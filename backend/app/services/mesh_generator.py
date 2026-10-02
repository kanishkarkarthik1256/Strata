"""Surface reconstruction: dense point cloud → triangle mesh.

Supported methods (configurable per run):

* ``ball_pivot`` — Ball-Pivot reconstruction (Open3D) over the cloud's
  fused, camera-oriented normals. This is the PRIMARY/production method:
  BPA only creates triangles when a rolling ball of measured radius touches
  three locally-supported points, so unknown space stays unknown — no
  watertight closure, no invented occlusion fill, and genuinely
  disconnected observed regions remain disconnected. Radii are derived per
  run from the measured local NN spacing (never hard-coded).
* ``poisson`` — watertight regularized reconstruction; DIAGNOSTIC ONLY. It
  can close unobserved regions (sub-surface skirts) and is never used for
  the production artifact. Requires per-point normals.
* ``surface`` — 2.5D Delaunay triangulation of the cloud projected onto its
  best-fit plane (PCA), with every candidate triangle validated against its
  TRUE 3D edge lengths under a locally-scaled budget. Legacy fallback for
  nadir-only scans; invalid for scenes where surfaces share (u,v)
  footprint (terrain + tall structure).
* ``alpha`` — 3D Delaunay alpha shape; keeps tetrahedra whose circumradius
  is below an alpha radius and extracts the boundary. Suited to closed
  volumes.

``auto`` resolves to ``ball_pivot`` when open3d is available and the cloud
carries normals; otherwise ``surface``.

Nothing here fabricates geometry: when the cloud cannot support a surface
(too few points, degenerate spread, no valid triangles after edge culling)
a :class:`MeshNotPossible` error is raised with the reason, and the caller
decides whether that means skip or fail.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.mesh import TriangleMesh
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register
from app.services.pointcloud import PointCloud, read_ply

log = get_logger("drone_recon.services.mesh_generator")

#: Above this many points, :func:`_local_spacing` measures on a subsample and
#: propagates each vertex's nearest measured spacing (bounds the k-NN query's
#: memory at ~64 bytes/point while keeping the per-edge budget local).
_MAX_EXACT_SPACING_POINTS = 4_000_000


class MeshNotPossible(Exception):
    """The point cloud cannot support the requested surface reconstruction."""


@dataclass
class SheetGapParams:
    """Sheet-gap bridge protection for BPA (mesh-quality phase, airport3 experiment).

    Every threshold is anchored to the cloud's OWN measured sheet separation
    (``detect_layers`` median observed separation) — no absolute constants.
    Two independent safeguards, each separately toggleable so the experiment
    report can attribute handle reduction to the right mechanism:

    * ``cap_weld`` — the BPA patch-merge weld (``merge_close_vertices``) must
      stay BELOW the measured sheet separation. The baseline welds at the
      full smallest BPA radius (~median spacing); when that radius exceeds
      the gap between stacked sheets, the weld merges the two sheets into
      one zig-zag surface — the handle factory. Capping the weld keeps every
      vertex (it welds LESS, never more) while refusing to fuse sheets.
    * ``reject_triangles`` — post-BPA evidence test per candidate triangle.
      A triangle is rejected ONLY when ALL three independent pieces of
      evidence agree it crosses a sheet gap:
        1. SPAN: its longest edge exceeds the measured sheet separation
           (an edge shorter than the closest sheets cannot bridge them);
        2. ORIENTATION: its face normal is near-perpendicular to the local
           dense-surface normals under it (a bridge across parallel sheets
           lies edge-on to both surfaces);
        3. EMPTINESS: both midpoints of that longest edge are farther from
           the dense cloud than the local point spacing (the edge floats
           over empty space instead of lying on observed surface).
      Rejected triangles remove CONNECTIVITY ONLY — the underlying dense
      points are never modified or filtered.
    """

    enabled: bool = False
    sheet_gap_m: float | None = None       # MEASURED median sheet separation
    sheet_gap_source: str = ""             # provenance of the measurement
    cap_weld: bool = True
    weld_sheet_factor: float = 0.5         # weld eps ≤ factor × sheet gap
    reject_triangles: bool = True
    span_budget: str = "sheet_gap"         # bridge-suspect edge > sheet gap
    normal_cos_min: float = 0.5            # |dot| < this ⇒ off-surface orientation
    midpoint_factor: float = 0.75          # midpoint empty iff d > factor × local spacing


@dataclass
class MeshParams:
    method: str = "auto"  # auto|surface|alpha|poisson|ball_pivot
    #: Budget for a 2.5D-surface triangle edge = this × the mean of its two
    #: endpoints' measured local k-NN spacing, so dense regions get tight
    #: budgets and sparse ones generous ones. 10.0 is measured, not taste:
    #: with the spacing now measured exactly (see :func:`_local_spacing`) on
    #: Video_Mission test_run_29_bedd74's 1.27M-pt fused cloud (1.26 m local
    #: spacing, 119 m of real relief — buildings, trees, a river basin),
    #: factor 4 rejects 53% of the Delaunay and shatters the result into
    #: 108,659 patches with the largest holding 44.7% of faces and 35.1% of
    #: edges on a hole boundary; 6 → 52.8%/21.2%; 10 → 93.3%/9.2% at the same
    #: cost (14 s); 16 → 97.7%/4.3%. 10 keeps a 12.6 m edge cap here, i.e. ten
    #: sampling scales, so a genuine curtain (a tower-to-ground span) is still
    #: refused while real steep relief is not.
    surface_max_edge_factor: float = 10.0
    alpha_multiplier: float = 1.5
    min_points: int = 50
    #: Octree depth for Poisson. The grid cell is (cloud extent)/2**depth, so
    #: 10 on a 500 m scene is a 0.49 m cell — the fused cloud's own sampling
    #: scale (measured 0.47-0.58 m on the airport1 fused cloud). Depth 9 (0.98 m
    #: cells) resolves coarser than the data supports and depth 11 triples the
    #: face count for detail below the noise floor.
    poisson_depth: int = 10
    #: Poisson returns a *watertight* shell around the sampled region, not just
    #: the surface: the far part of that shell is invented and dominates every
    #: distance metric unless it is removed. Two cuts are needed — the density
    #: quantile drops the low-confidence shell, the bounding box drops whatever
    #: survives outside the measured extent. Measured on the airport1 fused
    #: cloud (2.59M points): without the crop the median vertex-to-cloud
    #: distance was 1.2e6 m and the mesh had thousands of spurious components;
    #: cropped, it is 0.351 m (the cloud's own noise) with 415.
    poisson_density_quantile: float = 0.03
    #: Ball radii = factors × the measured P90 NN spacing (see
    #: :func:`nn_spacing_stats`). Rooting at the median shattered the surface.
    ball_pivot_radii_factors: list[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.33])
    #: Patch-seam weld = patch_weld_factor x smallest BPA radius (= P90
    #: spacing). Measured on furnerhem_6 (2.23M pts, 0.554 m spacing): the
    #: legacy 1.0x-median weld merged 34% of the cloud into welded blobs
    #: (mesh-to-cloud dev p95 0.264 m). 0.5x stays below the sampling noise
    #: while still stitching same-surface patches from different radius
    #: passes, and the sheet-gap cap still binds below it when sheets exist.
    #: Real gaps (>= 1.5x fusion voxel by construction) cannot close.
    patch_weld_factor: float = 0.5
    #: Hole closure (measured on sihhh_93663d, 2.32M pts, 1.06 m P90 spacing):
    #: BPA cannot pivot across a sampling gap, so it left the surface open
    #: along 30.6% of its edges (532k of 1.74M) — the ragged edges the viewer
    #: shows as holes. Filling every hole whose perimeter is within
    #: ``fill_holes_factor`` x the P90 spacing closes 24% of that boundary
    #: (+40k faces, +8 s of a 270 s mesh) and every added face sits p50 0.63 /
    #: p95 0.97 m from a real cloud point, i.e. INSIDE one measured spacing —
    #: so the fill can only cover gaps the sampling already supports. Raising
    #: the size to 4-8 m adds nothing further (the residual boundary is
    #: genuinely unobserved region), so the closure stays bounded by data.
    fill_holes: bool = True
    fill_holes_factor: float = 2.0
    #: Shatter cleanup: drop disconnected clusters that are negligible at the
    #: sampling scale (few vertices AND tiny diameter vs the smallest BPA
    #: radius). Removes BPA's boundary-noise islands without touching real
    #: observed surface; the provenance records what was pruned.
    prune_islands: bool = True
    island_min_points: int = 40
    island_max_diameter_factor: float = 3.0
    #: Debris cleanup (measured on Video_Mission_7721c8, 2.39M faces, 1.01 m
    #: median edge): the mesh was 128,566 EDGE-connected patches whose largest
    #: held only 35.9% of faces, and 32% of all faces sat in <=50-face patches
    #: a median 9.2 edge-lengths away from the main surface — the scattered
    #: shard look the viewer showed. A patch is dropped only when BOTH bounds
    #: fire: it holds <= debris_max_faces faces AND its nearest point is
    #: further than debris_distance_factor x the mesh's own median edge length
    #: from the main surface. At 3.0x that removed 17.3% of faces (61,226
    #: patches) while keeping 612,392 faces of genuinely near-surface patches,
    #: so real observed surface within ~3 m of the scene's main surface is
    #: never deleted. Set False for the legacy unfiltered output.
    prune_debris: bool = True
    debris_max_faces: int = 200
    debris_distance_factor: float = 3.0
    sheet_gap: SheetGapParams | None = None  # default OFF = exact legacy behavior

    @classmethod
    def from_settings(cls) -> MeshParams:
        m = settings.mesh
        return cls(
            method=m.method,
            surface_max_edge_factor=m.surface_max_edge_factor,
            alpha_multiplier=m.alpha_multiplier,
            min_points=m.surface_min_points,
            poisson_depth=m.poisson_depth,
            ball_pivot_radii_factors=list(m.ball_pivot_radii_factors),
        )


def nn_spacing_stats(xyz: np.ndarray, sample: int = 20_000) -> dict[str, float]:
    """Nearest-neighbour spacing DISTRIBUTION (m) of the cloud's sampling grid.

    Phase 2 fix retained: NN distances are measured against the FULL cloud
    (subsampling WITHIN the sample inflates NN distance by ~cbrt(n/sample));
    the subsample only bounds query count.

    The distribution, not just its median, is what the ball radii need
    (measured on Video_Mission_9ab1aa, 1.63M pts, 1.0 km scene): the median
    spacing was 0.665 m while P90 was 0.942 m. A ladder rooted at the MEDIAN
    (radii [0.67, 1.33, 2.66]) cannot roll across the gaps that actually
    occur, so BPA emitted 1.72M faces in 108,470 edge-components — the
    shattered mesh. The same ladder rooted at P90 (radii [0.94, 1.41, 2.20])
    produced 54,584 components with the largest holding 74.3% of faces (and
    estrel_c0cf6d improved from 18.1% to 75.0%), so the smallest ball must
    span the upper spacing percentile, not the median.
    """
    n = len(xyz)
    if n < 2:
        raise MeshNotPossible("too few points to estimate spacing")
    tree = cKDTree(xyz)
    if n <= sample:
        dist, _ = tree.query(xyz, k=2, workers=-1)
        v = dist[:, 1]
    else:
        pts = xyz[np.random.default_rng(0).choice(n, sample, replace=False)]
        dist, _ = tree.query(pts, k=2, workers=-1)
        v = dist[:, 1]
    return {f"p{p}": float(np.percentile(v, p)) for p in (50, 75, 90, 95, 99)}


def _local_spacing(xyz: np.ndarray, k: int = 6, sample: int = 50_000) -> np.ndarray:
    """Median k-NN distance per vertex — the local scale of the sampling.

    Measured against the FULL cloud. A subsampled anchor tree inflates the
    statistic by (n/sample)^(1/dimension) — measured 3.03x on the 1.27M-point
    fused cloud of Video_Mission test_run_29_bedd74 (3.83 m reported against
    1.26 m actual) — and every edge budget in :func:`_surface_mesh` is a
    multiple of this number, so the inflation silently loosened the 3D
    validity test that is supposed to stop tower-to-ground curtain
    triangles. The exact per-vertex query is also no slower (1.1 s against
    0.93 s at 1.27M points), because it queries each point against the tree
    that is already built.

    Clouds past :data:`_MAX_EXACT_SPACING_POINTS` measure on a subsample and
    give every vertex its nearest measured neighbour's spacing, so the
    per-edge budget stays local while memory stays bounded.
    """
    n = len(xyz)
    if n < 2:
        return np.full(n, np.inf)
    tree = cKDTree(xyz)
    kq = min(k, n - 1)

    def _median_knn(points: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(points, k=kq + 1, workers=-1)
        # Self-matches (distance 0, only for points that are their own
        # neighbour) must not dilute the statistic — treat them as missing.
        d = np.where(dist > 0, dist, np.inf)
        return np.median(np.sort(d, axis=1)[:, :kq], axis=1)

    if n <= _MAX_EXACT_SPACING_POINTS:
        return _median_knn(xyz)
    pts = xyz[np.random.default_rng(0).choice(n, sample, replace=False)]
    _, owner = cKDTree(pts).query(xyz, k=1, workers=-1)
    return _median_knn(pts)[owner]


def _surface_mesh(cloud: PointCloud, max_edge_factor: float) -> TriangleMesh:
    """2.5D triangulation with true-3D, locally-scaled edge validity.

    Topology comes from the 2D Delaunay on the best-fit-plane projection,
    but every candidate triangle is then checked against the ACTUAL 3D
    geometry: all three edge lengths must fit a budget derived from the
    measured local point spacing of their endpoints (max_edge_factor x the
    mean per-vertex k-NN spacing). This is scale-aware — dense regions get
    tight budgets, sparse regions generous ones — and uses no absolute
    constant, so it adapts to any scene. It is what stops the historical
    curtain failure: a tower vertex 100 m above the ground shares its
    (u,v) footprint with terrain vertices, but the 3D edge between them is
    ~100x the local spacing and the triangle is rejected. Genuinely
    disconnected observed regions therefore stay disconnected instead of
    being bridged.
    """
    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    if len(xyz) < 3:
        raise MeshNotPossible("need at least 3 points to triangulate")
    centered = xyz - xyz.mean(axis=0)
    cov = centered.T @ centered / len(xyz)
    evals, evecs = np.linalg.eigh(cov)
    if evals[1] < 1e-12:
        raise MeshNotPossible("point cloud is (near-)collinear — no surface to triangulate")
    e1, e2 = evecs[:, 1], evecs[:, 2]
    if np.cross(e1, e2) @ evecs[:, 0] < 0:
        e2 = -e2
    u = centered @ e1
    v = centered @ e2
    try:
        tri = Delaunay(np.column_stack([u, v]))
    except QhullError as exc:
        raise MeshNotPossible(f"2D Delaunay failed: {exc}") from exc
    if tri.simplices.size == 0:
        raise MeshNotPossible("triangulation produced no triangles")

    # Per-vertex local spacing → per-edge 3D budget (no global threshold).
    spacing = _local_spacing(xyz)

    simplices = tri.simplices
    # (F, 3, 2) index pairs for edges (0,1), (1,2), (2,0).
    edge_idx = simplices[:, [[0, 1], [1, 2], [2, 0]]]
    pa = xyz[edge_idx[:, :, 0]]
    pb = xyz[edge_idx[:, :, 1]]
    d3 = np.linalg.norm(pa - pb, axis=2)  # TRUE 3D edge lengths
    budget = max_edge_factor * 0.5 * (spacing[edge_idx[:, :, 0]] + spacing[edge_idx[:, :, 1]])
    keep = np.all(d3 <= budget, axis=1)
    faces = simplices[keep]
    if len(faces) == 0:
        raise MeshNotPossible("3D edge validation rejected every triangle")
    return TriangleMesh(
        vertices=xyz,
        faces=faces,
        colors=cloud.rgb.copy() if cloud.rgb is not None else None,
        confidence=cloud.confidence.copy() if cloud.confidence is not None else None,
        meta={
            "method": "surface",
            "edge_budget": "local_spacing_3d",
            "max_edge_factor": float(max_edge_factor),
            "median_spacing_m": round(float(np.median(spacing)), 4),
            "rejected_faces_3d": int((~keep).sum()),
        },
    )


def _alpha_mesh(cloud: PointCloud, alpha_multiplier: float) -> TriangleMesh:
    """3D alpha-shape surface: boundary of tetrahedra with circumradius ≤ alpha."""
    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    if len(xyz) < 4:
        raise MeshNotPossible("alpha shapes need at least 4 points")
    try:
        tri = Delaunay(xyz)
    except QhullError as exc:
        raise MeshNotPossible(f"3D Delaunay failed: {exc}") from exc

    # Circumradius per tetrahedron via the standard determinant formula.
    tet = xyz[tri.simplices]
    a, b, c, d = tet[:, 0], tet[:, 1], tet[:, 2], tet[:, 3]
    ab, ac, ad = b - a, c - a, d - a
    denom = 2.0 * np.einsum("ij,ij->i", ab, np.cross(ac, ad))
    with np.errstate(divide="ignore", invalid="ignore"):
        num = (np.linalg.norm(ab, axis=1) * np.linalg.norm(ac, axis=1) *
               np.linalg.norm(ad, axis=1))
        radius = np.abs(num / denom)
    radius[~np.isfinite(radius)] = np.inf
    finite = radius[np.isfinite(radius)]
    if not len(finite):
        raise MeshNotPossible("no tetrahedron has a finite circumradius")
    # Random 3D Delaunay complexes contain slivers with enormous circumradii;
    # alpha is anchored to the *radius distribution* (75th percentile) rather
    # than the point spacing, which keeps the well-shaped tetrahedra and
    # discards the pathological ones.
    alpha = alpha_multiplier * float(np.percentile(finite, 75))
    kept = radius <= alpha
    if kept.sum() == 0:
        raise MeshNotPossible("alpha radius removed every tetrahedron")

    # Boundary triangles = faces shared by exactly one kept tetrahedron.
    kept_tets = tri.simplices[kept]
    per_face = np.concatenate([
        kept_tets[:, [0, 1, 2]], kept_tets[:, [0, 1, 3]],
        kept_tets[:, [0, 2, 3]], kept_tets[:, [1, 2, 3]],
    ])
    per_face.sort(axis=1)
    uniq, counts = np.unique(per_face, axis=0, return_counts=True)
    faces = uniq[counts == 1]
    if len(faces) == 0:
        raise MeshNotPossible("no boundary triangles — cloud is a volume interior")
    return TriangleMesh(
        vertices=xyz,
        faces=faces,
        colors=cloud.rgb.copy() if cloud.rgb is not None else None,
        confidence=cloud.confidence.copy() if cloud.confidence is not None else None,
        meta={"method": "alpha", "alpha_m": float(alpha), "kept_tets": int(kept.sum())},
    )


def _crop_poisson_shell(mesh, cloud: PointCloud, densities: np.ndarray, quantile: float):
    """Remove Poisson's invented watertight shell, keeping the observed surface.

    Poisson solves for a watertight indicator function, so its output always
    includes a boundary shell far outside the sampled region. That shell is
    not measured geometry: on the airport1 fused cloud it put the MEDIAN
    vertex-to-cloud distance at 1.2e6 m and added thousands of components.

    Two cuts, in this order because they answer different questions:

    * the **density** cut drops the low-confidence extrapolation. Poisson's
      per-vertex density counts supporting samples, so it separates "surface
      the cloud actually constrains" from "shell closing the volume";
    * the **bounds** cut drops anything outside the cloud's measured extent,
      which is where a shell that happens to sit on a high-density fringe
      would otherwise survive.

    Neither cut moves or merges a vertex, so nothing inside the observed region
    is altered.
    """
    import open3d as o3d

    lo, hi = cloud.bounds()
    before = len(mesh.vertices)
    if len(densities) == before and quantile > 0.0:
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, quantile))
    box = o3d.geometry.AxisAlignedBoundingBox(lo.astype(float), hi.astype(float))
    mesh = mesh.crop(box)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh, {
        "vertices_before_crop": int(before),
        "vertices_after_crop": int(len(mesh.vertices)),
        "density_quantile": float(quantile),
        "note": "Poisson's watertight shell is invented geometry; only the "
                "observed surface inside the cloud's own bounds is kept",
    }


def poisson_unsupported_reason(cloud: PointCloud, params: MeshParams) -> str | None:
    """Why Poisson cannot be used on *cloud*, or None when it can.

    Poisson resolves the scene into an octree of ``2**depth`` cells across the
    dominant extent. When that cell is far finer than the cloud's own sampling
    the grid is nearly all empty cells, the solver is operating outside its
    input's support, and Open3D's implementation does not return an error — it
    dies natively. Measured: the autonomous pipeline's seeded scene (a handful
    of points over ~1 m) segfaulted the entire process out of
    ``create_from_point_cloud_poisson`` at the configured depth 10.

    The requirement is therefore derived from the data, not chosen: the octree
    cell must stay within the sampling scale the cloud actually supports. On
    the airport1 run (480 m extent, 0.47 m measured spacing, depth 10) the cell
    is 0.469 m — ratio ~1.0 — so the production cloud passes with margin.
    """
    if cloud.normals is None:
        return ("poisson reconstruction requires oriented normals — the fused "
                "cloud carries camera-oriented per-measurement normals through "
                "the chain")
    lo, hi = cloud.bounds()
    extent = float(np.max(np.asarray(hi) - np.asarray(lo)))
    if not np.isfinite(extent) or extent <= 0.0:
        return "poisson requires a non-degenerate spatial extent"
    cell = extent / float(2 ** int(params.poisson_depth))
    spacing = nn_spacing_stats(np.asarray(cloud.xyz))["p50"]
    if spacing > 0 and cell < 0.5 * spacing:
        return (f"poisson octree cell {cell:.4g} m is far finer than the "
                f"cloud's measured {spacing:.4g} m sampling")
    return None


def _poisson_mesh(cloud: PointCloud, params: MeshParams) -> TriangleMesh:
    """Screened Poisson reconstruction of the fused cloud, shell-cropped.

    Chosen as the production mesher because it is the only method here that
    reconstructs a *continuous* surface from this pipeline's fused cloud.
    The cloud carries sub-spacing depth-alignment error (measured on the
    airport1 run: 0.32 m local plane residual, 0.48 m cross-view
    disagreement, 1.95 m median separation between the stacked sheets the
    dense stage itself counts), and a triangulation that must pass through
    every measured point turns that error into a crumpled lace — the same
    cloud measured 875,914 m^2 of mesh over a 121,576 m^2 footprint (7.2x),
    23% of edges on a hole boundary and 17,040 components. Poisson integrates
    the same measurements into one implicit surface, and on that cloud it
    gives 0.2% boundary edges and 415 components at 0.351 m median deviation
    from the cloud (i.e. it follows the data's own noise, it does not smooth
    it away), in 90 s against 480 s.

    Only one implementation is kept: Open3D's. The previous pycolmap
    subprocess detour could not be density-cropped (pycolmap exposes no
    density field), so whenever it succeeded it returned the uncropped shell
    and the bad mesh — a second path that produced the wrong answer.
    """
    import open3d as o3d

    reason = poisson_unsupported_reason(cloud, params)
    if reason is not None:
        raise MeshNotPossible(reason)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(cloud.xyz))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals))
    if cloud.rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(cloud.rgb) / 255.0)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=params.poisson_depth)
    mesh, crop_meta = _crop_poisson_shell(
        mesh, cloud, np.asarray(densities), params.poisson_density_quantile)
    out = _from_open3d(mesh, cloud, "poisson")
    out.meta["poisson_depth"] = int(params.poisson_depth)
    out.meta["shell_crop"] = crop_meta
    return out


def _ball_pivot_mesh(cloud: PointCloud, params: MeshParams) -> TriangleMesh:
    """Ball-Pivot reconstruction — the observed-surface production method.

    Requires oriented normals (the fused cloud carries camera-oriented
    per-measurement normals through the whole chain). Radii are derived
    from the MEASURED local NN spacing each run: {1x, 1.5x, 2.33x} × the
    P90 spacing — the smallest ball must span the neighbour gaps that
    actually occur (rooting at the median measured on Video_Mission_9ab1aa
    left the ball below P90 and shattered the surface into 108k patches),
    while larger radii only span legitimate sparse regions. BPA never
    bridges across empty space the ball cannot roll through, so unknown
    stays unknown and genuinely disconnected regions stay disconnected.

    When ``params.sheet_gap`` is enabled, two measured sheet-gap safeguards
    apply on top (see :class:`SheetGapParams`): the patch-merge weld is
    capped below the measured sheet separation, and candidate triangles
    carrying threefold bridge evidence are rejected. Both are OFF by
    default — the legacy behavior is bit-identical when ``sheet_gap`` is
    None.
    """
    import open3d as o3d  # lazy import; surface/alpha remain dependency-free

    if cloud.normals is None:
        raise MeshNotPossible(
            "ball-pivot reconstruction requires oriented normals — the fused "
            "cloud must carry normals through the filter chain"
        )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(cloud.xyz))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals))
    if cloud.rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(cloud.rgb) / 255.0)
    spacing_stats = nn_spacing_stats(cloud.xyz)
    spacing = spacing_stats["p90"]
    radii = [spacing * f for f in params.ball_pivot_radii_factors]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    # Scale-collapse guard (measured e2e_video_1): BPA's ball sequence starts
    # from the SMALLEST radius. On clouds whose NN-spacing distribution is a
    # mixture (dense cluster subpopulation + sparser sheet — 15% of points
    # under 0.02 m next to a 0.14 m sheet there), a basis below the sheet's
    # local scale stalls the first ball at double-digit triangle counts,
    # eventually surfacing as "ball_pivot produced no triangles" after
    # welding. The P90 basis already carries the sheet's scale when the
    # sheet owns the upper decile; this retry covers the residual case where
    # even P90 sits in the dense subpopulation. Fallback only — the first
    # pass is untouched whenever it succeeds.
    if len(mesh.triangles) < 1000:
        from scipy.spatial import cKDTree as _cKDTree

        _s = cloud.xyz[np.random.default_rng(0).choice(
            len(cloud.xyz), min(20_000, len(cloud.xyz)), replace=False)]
        _d, _ = _cKDTree(cloud.xyz).query(_s, k=2, workers=-1)
        _q = float(np.percentile(_d[:, 1], 75))
        if _q > spacing * 1.3:
            log.info("bpa_scale_collapse_retry", radii_basis_spacing=round(spacing, 4),
                     upper_quartile_spacing=round(_q, 4),
                     first_pass_triangles=int(len(mesh.triangles)))
            spacing = _q
            radii = [_q * f for f in params.ball_pivot_radii_factors]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
    # BPA triangulates each radius pass independently and leaves adjacent
    # same-surface samples in separate patches; merge_close_vertices at half
    # the smallest radius stitches patches of ONE surface (its seam distance
    # is far below the ball radius that would be needed to bridge a real
    # gap), so disconnected observed regions remain disconnected. The
    # measured factor (see MeshParams.patch_weld_factor) replaced the legacy
    # 1x weld that blob-merged a third of a 2.23M-point cloud.
    weld = radii[0] * params.patch_weld_factor
    weld_source = f"{params.patch_weld_factor:.2f} x smallest BPA radius (measured patch-merge)"
    sg = params.sheet_gap
    if sg is not None and sg.enabled and sg.cap_weld and sg.sheet_gap_m:
        cap = sg.weld_sheet_factor * float(sg.sheet_gap_m)
        if cap < weld:
            weld = cap
            weld_source = (
                f"sheet-gap cap: {sg.weld_sheet_factor} x measured sheet "
                f"separation ({sg.sheet_gap_m:.3f} m) — weld can no longer "
                f"fuse stacked sheets"
            )
    merged = mesh.merge_close_vertices(weld)
    merged.remove_degenerate_triangles()
    merged.remove_duplicated_triangles()
    merged.remove_unreferenced_vertices()
    # Shatter/island cleanup: BPA emits thousands of tiny disconnected
    # clusters around noisy boundary points (measured on estrel_c0cf6d:
    # 9,820 components, largest only 90.7% of the mesh — the shattered
    # look in the viewer). A cluster is dropped when it is geometrically
    # negligible against the SAMPLING scale: ≤ island_min_points vertices
    # AND diameter ≤ island_max_diameter_factor × smallest BPA radius.
    # Both bounds together: many points in a tight clump is still a real
    # blob of debris; a long thin strip is still real observed surface.
    island_meta: dict = {"enabled": False}
    if params.prune_islands:
        import scipy.sparse as _sp
        from scipy.sparse.csgraph import connected_components as _cc

        _v = np.asarray(merged.vertices)
        _f = np.asarray(merged.triangles)
        _nv = len(_v)
        _nf = len(_f)
        # Connectivity that RENDERS is edge connectivity: two triangles are
        # one surface piece only where they share an EDGE. The graph used to
        # be built on vertices, so any two shards touching at a single vertex
        # merged into one component — on the estrel cloud that reported 3,313
        # components when the rendered mesh had 50,260, and the flecks that
        # make the viewer look shattered were never small enough to prune.
        _e = np.concatenate([_f[:, [0, 1]], _f[:, [1, 2]], _f[:, [2, 0]]])
        _e = np.sort(_e, axis=1)
        _key = _e[:, 0] * _nv + _e[:, 1]
        _order = np.argsort(_key, kind="stable")
        _face_of_edge = np.tile(np.arange(_nf), 3)[_order]
        _ks = _key[_order]
        _same = _ks[1:] == _ks[:-1]
        _ia, _ib = _face_of_edge[:-1][_same], _face_of_edge[1:][_same]
        if len(_ia):
            _n_comp, _face_comp = _cc(
                _sp.coo_matrix((np.ones(len(_ia)), (_ia, _ib)),
                               shape=(_nf, _nf)).tocsr(), directed=False)
        else:
            _n_comp, _face_comp = _nf, np.arange(_nf)
        # Per-component vertex count and bounding-box span, vectorised (the
        # former per-component Python loop ran once per component).
        _cv = np.concatenate([_face_comp, _face_comp, _face_comp])
        _vv = _f.reshape(-1)
        _sizes = np.bincount(_cv, minlength=_n_comp)  # vertex memberships
        _vmin = np.full((_n_comp, 3), np.inf)
        _vmax = np.full((_n_comp, 3), -np.inf)
        np.minimum.at(_vmin, _cv, _v[_vv])
        np.maximum.at(_vmax, _cv, _v[_vv])
        _span = np.linalg.norm(_vmax - _vmin, axis=1)
        _r1 = radii[0]
        _diam_max = params.island_max_diameter_factor * _r1
        _drop = (_sizes <= params.island_min_points) & (_span <= _diam_max)
        island_meta = {
            "enabled": True,
            "components_before": int(_n_comp),
            "components_dropped": int(_drop.sum()),
            "connectivity": "edge (rendered surface pieces)",
            "island_min_points": params.island_min_points,
            "island_diameter_cap_m": round(_diam_max, 4),
        }
        if _drop.any():
            _keep_f = ~_drop[_face_comp]
            _kept = _f[_keep_f]
            _keep_v = np.zeros(_nv, dtype=bool)
            _keep_v[_kept.reshape(-1)] = True
            _remap = -np.ones(_nv, dtype=np.int64)
            _remap[_keep_v] = np.arange(int(_keep_v.sum()))
            _merged2 = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(_v[_keep_v]),
                o3d.utility.Vector3iVector(_remap[_kept]))
            _merged2.remove_unreferenced_vertices()
            island_meta["faces_dropped"] = int((~_keep_f).sum())
            island_meta["vertices_dropped"] = int((~_keep_v).sum())
            log.info("mesh_islands_pruned", **{
                k: v for k, v in island_meta.items() if k != "enabled"})
            merged = _merged2
    out = _from_open3d(merged, cloud, "ball_pivot")
    out.meta["radii_m"] = [round(r, 4) for r in radii]
    out.meta["patch_merge_m"] = round(weld, 4)
    out.meta["patch_merge_source"] = weld_source
    out.meta["nn_spacing_m"] = round(spacing_stats["p50"], 4)
    out.meta["nn_spacing_stats_m"] = {k: round(v, 4) for k, v in spacing_stats.items()}
    out.meta["radii_basis"] = "p90 nn spacing (measured per run)"
    out.meta["island_pruning"] = island_meta
    return _apply_cloud_guards(
        out, cloud, params, method="ball_pivot",
        hole_size_m=radii[0] * params.fill_holes_factor,
        hole_source="fill_holes_factor x smallest BPA radius (P90 spacing)")


def _apply_cloud_guards(
    mesh: TriangleMesh,
    cloud: PointCloud,
    params: MeshParams,
    *,
    method: str,
    hole_size_m: float,
    hole_source: str,
) -> TriangleMesh:
    """Post-passes owned by the CLOUD, not by the triangulation method.

    Sheet-gap rejection, debris pruning and hole closure all answer questions
    about the measured cloud (where are the stacked sheets, what is the local
    sampling scale) and none of them cares how the triangles were produced.
    They live here so every observed-surface reconstruction — ball pivot or
    2.5D surface — is guarded the same way, in this order:

    1. sheet-gap rejection (when the dense stage measured stacked sheets),
    2. debris pruning of small patches far from the main surface,
    3. hole closure, strictly additive and bounded by the sampling spacing.
    """
    sg = params.sheet_gap
    if sg is not None and sg.enabled and sg.reject_triangles and sg.sheet_gap_m:
        faces_keep, rej = _sheet_gap_bridge_rejection(mesh, cloud, sg)
        n_before = mesh.m
        kept_faces = mesh.faces[faces_keep]
        used = np.unique(kept_faces)
        remap = -np.ones(mesh.n, dtype=np.int64)
        remap[used] = np.arange(len(used))
        dropped = int(remap.size - len(used))
        mesh = TriangleMesh(
            vertices=mesh.vertices[used], faces=remap[kept_faces],
            colors=mesh.colors[used] if mesh.colors is not None else None,
            meta={**mesh.meta, "method": method},
        )
        mesh.meta["sheet_gap_rejection"] = {
            **rej,
            "triangles_before": int(n_before),
            "triangles_after": int(mesh.m),
            "vertices_dropped_unreferenced": dropped,
            "note": "rejected triangles remove connectivity only; the authoritative "
                    "dense cloud is untouched (hash-verified by the caller)",
        }
    # Debris cleanup runs on the finished topology: the sheet-gap rejection can
    # itself detach pieces, and BPA's island prune judges vertex connectivity,
    # which cannot see patches that merely share vertices.
    if params.prune_debris and mesh.m > 0:
        mesh, debris_meta = _prune_isolated_debris(
            mesh, params.debris_distance_factor, params.debris_max_faces)
        mesh.meta["debris_pruning"] = debris_meta
    # Hole closure runs LAST so it is strictly additive: every step above keeps
    # exactly the topology it had, and the fill can only add faces across gaps
    # the sampling supports (see MeshParams.fill_holes for the measurements).
    if params.fill_holes and mesh.m > 0:
        mesh, fill_meta = _fill_sampling_holes(mesh, hole_size_m, source=hole_source)
        mesh.meta["hole_filling"] = fill_meta
        if fill_meta.get("faces_added"):
            log.info("mesh_holes_filled", **{
                k: v for k, v in fill_meta.items() if k != "enabled"})
    return mesh


def _fill_sampling_holes(
    mesh: TriangleMesh, hole_size: float, *, source: str
) -> tuple[TriangleMesh, dict]:
    """Close holes whose perimeter is within *hole_size* metres.

    ADDITIVE ONLY: the fill never moves, adds or removes a vertex, so every
    per-vertex attribute (colour, confidence, label) and every existing face
    survives exactly as it was. Vertices are re-derived from the cloud on the
    final mesh anyway, so nothing downstream depends on the fill's own vertex
    data — but the guards below keep that contract explicit rather than
    assumed: a fill that would introduce vertices is declined.
    """
    import open3d as o3d

    faces_before, verts_before = int(mesh.m), int(mesh.n)
    meta = {
        "enabled": True,
        "hole_size_m": round(hole_size, 4),
        "hole_size_source": source,
        "faces_before": faces_before,
        "vertices_before": verts_before,
    }
    try:
        legacy = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)))
        filled = o3d.t.geometry.TriangleMesh.from_legacy(legacy).fill_holes(
            hole_size).to_legacy()
        faces = np.asarray(filled.triangles, dtype=np.int64)
        vertices = np.asarray(filled.vertices)
    except Exception as exc:
        return mesh, {"enabled": False, "reason": f"fill_holes failed: {exc}"}
    meta["faces_after"] = int(len(faces))
    meta["faces_added"] = int(len(faces)) - faces_before
    meta["vertices_after"] = int(len(vertices))
    if len(vertices) != verts_before:
        # Vertex order/meaning would change and the per-vertex attributes could
        # no longer be mapped; keep the unfilled topology rather than guess.
        return mesh, {"enabled": False, "reason": "fill_holes introduced vertices",
                      **{k: meta[k] for k in ("hole_size_m", "faces_before",
                                              "faces_after", "faces_added")}}
    if meta["faces_added"] <= 0:
        return mesh, {**meta, "note": "no hole within the size bound"}
    out = TriangleMesh(
        vertices=np.asarray(mesh.vertices), faces=faces, colors=mesh.colors,
        confidence=mesh.confidence, labels=mesh.labels, meta=dict(mesh.meta),
    )
    return out, meta


def edge_connectivity(faces: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Triangle connectivity under shared-EDGE adjacency.

    The one definition of "is this surface connected" used by both the debris
    prune and the mesh quality report. A shared vertex alone does NOT join two
    triangles for rendering purposes, which is why vertex-adjacency counts
    (9,820 "components" while 128,566 patches were actually rendered) hid the
    fragmentation. Edge pairs are encoded as one int64 key per edge, so the
    grouping is a single sort instead of a lexsort over 7M rows.

    Returns ``(n_patches, per_triangle_labels, per_patch_face_counts)``.
    """
    import scipy.sparse as _sp
    from scipy.sparse.csgraph import connected_components as _cc

    f = np.asarray(faces, dtype=np.int64)
    nf = len(f)
    if nf == 0:
        return 0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    nv = int(f.max()) + 1
    pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]])
    lo = pairs.min(axis=1)
    hi = pairs.max(axis=1)
    key = lo * nv + hi
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    t_sorted = np.tile(np.arange(nf), 3)[order]
    same = np.diff(k_sorted) == 0
    rows, cols = t_sorted[:-1][same], t_sorted[1:][same]
    n_comp, labels = _cc(
        _sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nf, nf)).tocsr(),
        directed=False,
    )
    return n_comp, labels, np.bincount(labels, minlength=n_comp)


def _prune_isolated_debris(
    mesh: TriangleMesh,
    distance_factor: float,
    max_faces: int,
) -> tuple[TriangleMesh, dict]:
    """Drop spatially isolated debris patches from a finished mesh.

    Connectivity is measured the way the rendered surface is stitched:
    triangles joined through a shared EDGE (a shared vertex alone leaves the
    surface disconnected in the viewer). A non-main patch is debris only when
    it is both small (<= ``max_faces`` triangles) and far from the main
    surface (> ``distance_factor`` x the mesh's median edge length) — a small
    patch that touches the surface is real observed geometry and stays. The
    main component is never a candidate.

    Pure deletion: no vertex is ever moved, so the mesh keeps sitting exactly
    on the measured cloud.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if len(f) == 0:
        return mesh, {"enabled": True, "patches_before": 0, "patches_dropped": 0}
    n_comp, labels, sizes = edge_connectivity(f)
    main = int(np.argmax(sizes))
    med_edge = float(np.median(
        np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1)))
    d_max = distance_factor * med_edge
    small = np.flatnonzero(sizes <= max_faces)
    small = small[small != main]
    meta: dict = {
        "enabled": True,
        "patches_before": int(n_comp),
        "largest_patch_faces_pct": round(float(sizes[main] / len(f) * 100.0), 2),
        "median_edge_m": round(med_edge, 4),
        "max_faces": int(max_faces),
        "distance_factor": float(distance_factor),
        "distance_threshold_m": round(d_max, 4),
    }
    if len(small) == 0:
        meta.update({"patches_dropped": 0, "faces_dropped": 0})
        return mesh, meta
    tree = cKDTree(v[np.unique(f[labels == main].reshape(-1))])
    comp_of_vertex = np.full(len(v), -1, dtype=np.int64)
    np.maximum.at(comp_of_vertex, f.reshape(-1), np.repeat(labels, 3))
    small_faces = np.isin(labels, small)
    small_verts = np.unique(f[small_faces].reshape(-1))
    dist, _ = tree.query(v[small_verts], k=1)
    nearest = np.full(len(v), np.inf)
    np.minimum.at(nearest, comp_of_vertex[small_verts], dist)
    drop = small[nearest[small] > d_max]
    keep_face = ~np.isin(labels, drop)
    kept_faces = f[keep_face]
    used = np.unique(kept_faces)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    cleaned = TriangleMesh(
        vertices=v[used],
        faces=remap[kept_faces],
        colors=mesh.colors[used] if mesh.colors is not None else None,
        meta=mesh.meta,
    )
    meta.update({
        "patches_dropped": int(len(drop)),
        "faces_dropped": int(len(f) - len(kept_faces)),
        "faces_dropped_pct": round(float((len(f) - len(kept_faces)) / len(f) * 100.0), 2),
        "near_surface_patch_faces_kept": int(sizes[small].sum() - sizes[drop].sum()),
        "note": "pure deletion — no vertex is moved, so the mesh still sits on the "
                "measured cloud",
    })
    log.info("mesh_debris_pruned", patches_dropped=meta["patches_dropped"],
             faces_dropped=meta["faces_dropped"],
             faces_dropped_pct=meta["faces_dropped_pct"],
             distance_threshold_m=meta["distance_threshold_m"])
    return cleaned, meta


def _sheet_gap_bridge_rejection(
    mesh: TriangleMesh,
    cloud: PointCloud,
    sg: SheetGapParams,
) -> tuple[np.ndarray, dict]:
    """Evidence-based sheet-gap bridge rejection over BPA output triangles.

    Returns ``(keep_mask (F,), diagnostics dict)``. A face is rejected only
    when ALL three evidence tests fire (span + orientation + midpoint
    emptiness) — see :class:`SheetGapParams`. Purely diagnostic; never
    touches the cloud.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    nrm = np.asarray(cloud.normals, dtype=np.float64)
    gap = float(sg.sheet_gap_m)

    # Local sampling spacing of the DENSE cloud (per point, kNN median).
    tree = cKDTree(xyz)
    n_q = min(len(xyz), 2_000_000)
    q_idx = np.arange(len(xyz)) if len(xyz) <= n_q else np.sort(
        np.random.default_rng(0).choice(len(xyz), n_q, replace=False))
    kq = 4
    d_knn, _ = tree.query(xyz[q_idx], k=kq + 1, workers=-1)
    d_knn = np.where(d_knn > 0, d_knn, np.inf)
    sp_pts = np.median(np.sort(d_knn, axis=1)[:, :kq], axis=1)
    # sp_pts is per-queried point but gets indexed by GLOBAL cloud ids
    # (``near_mid`` from the full tree) below — scatter to full length or
    # any cloud larger than the query cap indexes out of bounds.
    sp_full = np.full(len(xyz), float(np.median(sp_pts)))
    sp_full[q_idx] = sp_pts
    sp_pts = sp_full

    # Nearest dense point for every mesh vertex (BPA vertices are input or
    # welded points, so distances are ~0 except at weld centroids).
    d_vert, near_vert = tree.query(v, k=1, workers=-1)

    tri = v[f]
    e = np.stack([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ], axis=1)
    long_edge = np.argmax(e, axis=1)             # (F,) which edge is longest
    fid_pairs = np.array([[0, 1], [1, 2], [2, 0]])
    va = f[np.arange(len(f)), fid_pairs[long_edge, 0]]
    vb = f[np.arange(len(f)), fid_pairs[long_edge, 1]]
    mids = 0.5 * (v[va] + v[vb])
    d_mid, near_mid = tree.query(mids, k=1, workers=-1)
    sp_mid = sp_pts[near_mid]

    # Face normal vs the local dense-surface orientation under the triangle.
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_u = np.divide(fn, fn_len, out=np.zeros_like(fn), where=fn_len > 1e-30)
    local_n = nrm[near_vert[f]].mean(axis=1)
    local_n = local_n / (np.linalg.norm(local_n, axis=1, keepdims=True) + 1e-30)
    dot = np.abs((fn_u * local_n).sum(axis=1))

    span = e.max(axis=1) > gap
    normal_off = dot < sg.normal_cos_min
    mid_empty = (d_mid > sg.midpoint_factor * sp_mid) & np.isfinite(sp_mid)
    reject = span & normal_off & mid_empty

    # Diagnostics: aggregates for every face, full records for a sample.
    rej_idx = np.flatnonzero(reject)
    diag: dict = {
        "enabled": True,
        "sheet_gap_m": round(gap, 4),
        "sheet_gap_source": sg.sheet_gap_source or "detect_layers median observed separation",
        "thresholds": {
            "span_budget_m": round(gap, 4),
            "normal_cos_min": sg.normal_cos_min,
            "midpoint_factor": sg.midpoint_factor,
            "rule": "reject iff span AND orientation-off AND midpoints empty (threefold evidence)",
        },
        "faces_total": int(len(f)),
        "flag_counts": {
            "span": int(span.sum()),
            "normal_off": int(normal_off.sum()),
            "midpoint_empty": int(mid_empty.sum()),
        },
        "rejected_faces": int(len(rej_idx)),
        "rejected_percent": round(float(100.0 * len(rej_idx) / max(len(f), 1)), 3),
        "rejected_evidence_stats": {},
        "kept_vs_rejected": {},
        "spatial_top_cells_10m": [],
        "rejected_records_sample": [],
        "labels": "MEASURED — thresholds anchored to the measured sheet separation",
    }
    if len(rej_idx):
        rs = rej_idx
        diag["rejected_evidence_stats"] = {
            "max_edge_m": {
                "median": round(float(np.median(e[rs].max(axis=1))), 3),
                "p95": round(float(np.percentile(e[rs].max(axis=1), 95)), 3),
                "max": round(float(e[rs].max()), 3),
            },
            "normal_abs_dot": {
                "median": round(float(np.median(dot[rs])), 3),
                "p05": round(float(np.percentile(dot[rs], 5)), 3),
            },
            "midpoint_distance_m": {
                "median": round(float(np.median(d_mid[rs])), 3),
                "p95": round(float(np.percentile(d_mid[rs], 95)), 3),
            },
        }
        keep_idx = np.flatnonzero(~reject)
        diag["kept_vs_rejected"] = {
            "kept_median_max_edge_m": round(float(np.median(e[keep_idx].max(axis=1))), 3),
            "kept_median_normal_dot": round(float(np.median(dot[keep_idx])), 3),
            "kept_median_midpoint_distance_m": round(float(np.median(d_mid[keep_idx])), 3),
        }
        # Spatial histogram (10 m grid) of rejected triangle centroids.
        cent = tri[rs].mean(axis=1)
        cells = np.floor(cent / 10.0).astype(np.int64)
        uniq_cells, counts = np.unique(cells, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1][:10]
        diag["spatial_top_cells_10m"] = [
            {"cell": uniq_cells[i].tolist(), "cell_min_m": (uniq_cells[i] * 10.0).tolist(),
             "rejected": int(counts[i])} for i in order]
        sample = rs[np.argsort(-e[rs].max(axis=1))[:2000]]
        diag["rejected_records_sample"] = [
            {
                "face": int(fi),
                "max_edge_m": round(float(e[fi].max()), 3),
                "normal_abs_dot": round(float(dot[fi]), 3),
                "mid_a_distance_m": None,
                "centroid": [round(float(c), 1) for c in tri[fi].mean(axis=0)],
            }
            for fi in sample[:2000]
        ]
    return ~reject, diag


def mesh_provenance_record(
    mesh: TriangleMesh,
    cloud: PointCloud,
    workspace: Path,
    stats: dict,
    dense_rel: str = "dense/dense_model.ply",
    mesh_rel: str = "mesh/mesh.ply",
) -> dict:
    """base_mesh.json sidecar content — provenance for EVERY mesh generation.

    Records the meshing method, BPA radii, input-cloud and mesh hashes,
    sheet-gap parameters, rejected-triangle count and a timestamp. Nothing
    is inferred from other stages: hashes are computed from the files.
    """
    import hashlib

    from app.services.mesh_phase2 import _sha256

    dense_path = workspace / dense_rel
    mesh_path = workspace / mesh_rel
    sg = stats.get("sheet_gap_rejection") or {}
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": stats.get("method"),
        "bpa_radii_m": stats.get("radii_m") or mesh.meta.get("radii_m"),
        "patch_merge_m": stats.get("patch_merge_m") or mesh.meta.get("patch_merge_m"),
        "nn_spacing_m": stats.get("nn_spacing_m") or mesh.meta.get("nn_spacing_m"),
        "island_pruning": mesh.meta.get("island_pruning") or {"enabled": False},
        "hole_filling": mesh.meta.get("hole_filling") or {"enabled": False},
        "sheet_gap": {
            "enabled": bool(stats.get("sheet_gap_enabled")),
            "sheet_gap_m": stats.get("sheet_gap_m"),
            "sheet_gap_source": stats.get("sheet_gap_source"),
            "weld_cap_applied": stats.get("weld_cap_applied"),
            "rejected_triangles": sg.get("rejected_faces"),
        },
        "input_dense_hash_sha256": _sha256(dense_path) if dense_path.is_file() else None,
        "input_dense_points": int(cloud.n),
        "mesh_hash_sha256": _sha256(mesh_path) if mesh_path.is_file() else None,
        "vertices": int(mesh.n),
        "faces": int(mesh.m),
        "took_ms": stats.get("took_ms"),
    }


def _from_open3d(mesh, cloud: PointCloud, method: str) -> TriangleMesh:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if len(faces) == 0:
        raise MeshNotPossible(f"open3d {method} produced no triangles")
    # Open3D meshes vertices to the oriented surface — nearest-cloud colors.
    colors = None
    if cloud.rgb is not None:
        tree = cKDTree(np.asarray(cloud.xyz))
        _, idx = tree.query(verts)
        colors = np.asarray(cloud.rgb)[idx]
    return TriangleMesh(
        vertices=verts, faces=faces, colors=colors,
        confidence=None, meta={"method": method},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_mesh(cloud: PointCloud, params: MeshParams | None = None) -> tuple[TriangleMesh, dict]:
    """Reconstruct a mesh from *cloud*; returns ``(mesh, stats)``.

    Raises :class:`MeshNotPossible` when the cloud cannot support a surface.
    """
    params = params or MeshParams.from_settings()
    started = time.perf_counter()
    if cloud.n < params.min_points:
        raise MeshNotPossible(f"only {cloud.n} points (need ≥ {params.min_points})")
    lo, hi = cloud.bounds()
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise MeshNotPossible("point cloud contains non-finite coordinates")

    method = params.method
    if method == "auto":
        # Poisson is the production default. Every method that must pass a
        # triangle through the measured points — 2.5D Delaunay (`surface`),
        # ball pivot, alpha — faithfully reproduces the fused cloud's own
        # depth-alignment error, because that error is real geometry in the
        # point set: the airport1 cloud measures 0.32 m of local plane
        # residual against 0.47-0.58 m sampling, and the dense stage counts
        # the stacked sheets behind it (1,062 layer regions, 1.95 m median
        # separation). The Delaunay default turned that cloud into 875,914 m^2
        # of mesh over a 121,576 m^2 footprint — the crumpled, ragged "lace"
        # the viewer showed — with 23% of its edges on a hole boundary and
        # 17,040 components, in ~480 s.
        #
        # Same cloud through `poisson`: 0.2% boundary edges, 415 components,
        # 0.351 m median vertex-to-cloud distance, in 90 s. Poisson stays
        # honest about unknown space because the watertight shell it adds is
        # cut away (see :func:`_crop_poisson_shell`); it does not smooth the
        # data either — its deviation from the cloud is the cloud's own noise.
        # `surface` and `ball_pivot` remain selectable by name.
        #
        # `auto` must never fail where the old Delaunay default succeeded, so a
        # cloud Poisson cannot support (no oriented normals, or an extent the
        # octree cannot be sampled at) falls back to `surface` rather than
        # raising — or, in Open3D's hands, crashing.
        reason = poisson_unsupported_reason(cloud, params)
        if reason is not None:
            log.info("mesh_auto_fallback", to="surface", reason=reason)
            method = "surface"
        else:
            method = "poisson"
    builders = {
        "surface": lambda: _surface_mesh(cloud, params.surface_max_edge_factor),
        "alpha": lambda: _alpha_mesh(cloud, params.alpha_multiplier),
        "poisson": lambda: _poisson_mesh(cloud, params),
        "ball_pivot": lambda: _ball_pivot_mesh(cloud, params),
    }
    if method not in builders:
        raise ValueError(f"unknown mesh method '{method}' — choose from {sorted(builders)}")
    mesh = builders[method]()
    if method in ("surface", "poisson"):
        # Same cloud-level guards ball pivot gets (sheet gaps, debris, hole
        # closure), sized by this cloud's own measured spacing rather than by
        # a ball radius. Poisson's surfaces are already closed (0.2% boundary
        # edges), so for it these guards are a no-op in practice — they are
        # applied anyway so the policy has one owner and cannot silently
        # differ between methods.
        hole = float(
            mesh.meta.get("median_spacing_m")
            or nn_spacing_stats(np.asarray(cloud.xyz))["p50"]
        ) * params.fill_holes_factor
        if hole > 0:
            mesh = _apply_cloud_guards(
                mesh, cloud, params, method=method, hole_size_m=hole,
                hole_source="fill_holes_factor x measured median local NN spacing")
    sgr = mesh.meta.get("sheet_gap_rejection") or {}
    stats = {
        "method": mesh.meta.get("method", method),
        "vertices": mesh.n,
        "faces": mesh.m,
        "took_ms": round((time.perf_counter() - started) * 1000, 2),
        "radii_m": mesh.meta.get("radii_m"),
        "patch_merge_m": mesh.meta.get("patch_merge_m"),
        "patch_merge_source": mesh.meta.get("patch_merge_source", ""),
        "nn_spacing_m": mesh.meta.get("nn_spacing_m"),
        "sheet_gap_enabled": bool(sgr),
        "sheet_gap_m": sgr.get("sheet_gap_m") if sgr else None,
        "sheet_gap_source": sgr.get("sheet_gap_source") if sgr else None,
        "weld_cap_applied": "sheet-gap cap" in str(mesh.meta.get("patch_merge_source", "")),
        "sheet_gap_rejection": mesh.meta.get("sheet_gap_rejection"),
        "hole_filling": mesh.meta.get("hole_filling") or {"enabled": False},
    }
    log.info("mesh_generated", method=stats["method"], vertices=mesh.n, faces=mesh.m)
    return mesh, stats


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class MeshGenerationStage(PipelineStage):
    name = "mesh_generation"
    description = "Surface reconstruction of the dense point cloud"
    artifact_rel = "mesh/base_mesh.ply"

    def validate_inputs(self) -> None:
        dense = self.workspace / "dense" / "dense_model.ply"
        if not dense.exists():
            raise StageNotApplicable("no dense model — run the dense stage first")

    def execute(self) -> None:
        dense = self.workspace / "dense" / "dense_model.ply"
        cloud = read_ply(dense)
        mesh, stats = generate_mesh(cloud)
        path = self.artifact_path()
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.save_ply(path, normals=False)
        mesh_alt = self.workspace / "mesh" / "mesh.ply"
        mesh.save_ply(mesh_alt, normals=False)
        (self.workspace / "mesh" / "base_mesh.json").write_text(
            json.dumps(stats, indent=2))
        self._count = mesh.m
        self._detail = stats
        self._outputs = [{"kind": "mesh", "name": "base_mesh", "path": str(path),
                          "faces": mesh.m, "vertices": mesh.n}]
