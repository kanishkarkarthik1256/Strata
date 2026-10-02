"""Held-out LiDAR reference — the ground-truth path.

Contracts pinned here:

1. LAS and LAZ both read, and the app's own LAS writer round-trips through
   the reader (writer/reader disagreement would silently misplace a tile).
   The pre-baked `.npz` form is an accepted INPUT as well as the stored
   reference: `points` is read verbatim, a grid-only archive is reported as
   having no returns rather than read as an empty cloud.
2. Rasterisation matches the sampler ``dsm_accuracy`` uses — row 0 is the
   north edge, column 0 the west edge — keeps the MAXIMUM return per cell
   (a surface, not a bare-earth average), and leaves unmeasured cells NaN
   instead of interpolating a height nobody measured.
3. A run's reference resolves from its own workspace. There is no per-run
   registry: a stale entry could attach one scene's ground truth to another
   scene's reconstruction, which is the failure this replaced.
4. With a reference present the absolute-accuracy comparison measures and the
   report says ``held_out_reference``; with none it reports ``no_reference``
   and claims nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services import dsm_accuracy as dsm
from app.services import lidar
from app.services.pointcloud import PointCloud, save_ply
from app.services.upload_service import safe_lidar_filename

# --------------------------------------------------------------------------
# fixtures: a bumpy tile whose relief is feature-rich enough that a rolled
# copy is genuinely decorrelated (the alignment gate refuses flat geometry).
# --------------------------------------------------------------------------


def _surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Aperiodic test terrain: a tilt, two blobs and fine texture.

    Deliberately NOT a sum of sines alone — a periodic surface aligns equally
    well at every period, which makes a co-registration test meaningless (an
    early version of this fixture "recovered" a 45 m offset that fitted just
    as well as the true one).
    """
    return (
        50.0
        + 0.05 * x
        - 0.03 * y
        + 2.0 * np.exp(-(((x - 30.0) ** 2 + (y - 70.0) ** 2) / 500.0))
        + 1.5 * np.exp(-(((x - 90.0) ** 2 + (y - 40.0) ** 2) / 300.0))
        + 0.6 * np.sin(x / 1.3) * np.cos(y / 1.1)
    )


def _plane_points(cell: float = 1.0, span: float = 60.0) -> np.ndarray:
    """A near-flat tile: relief per cell is ~5 cm, so a max-aggregated grid
    must reproduce the sampled heights tightly enough to catch a transposed
    or flipped axis (which would miss by tens of metres)."""
    xs = np.arange(0.0, span, cell)
    X, Y = np.meshgrid(xs, xs)
    return np.column_stack([X.ravel(), Y.ravel(), (100.0 + 0.04 * X - 0.02 * Y).ravel()])


def _bumpy_points(cell: float = 0.5, span: float = 120.0, offset=(0.0, 0.0)) -> np.ndarray:
    xs = np.arange(0.0, span, cell)
    X, Y = np.meshgrid(xs, xs)
    Z = _surface(X, Y)
    return np.column_stack(
        [X.ravel() + offset[0], Y.ravel() + offset[1], Z.ravel()]
    )


def _write_tile(path: Path, points: np.ndarray) -> Path:
    """A tile written by the app's own LAS writer (writer ↔ reader contract)."""
    from app.services.pointcloud import save_las

    path.parent.mkdir(parents=True, exist_ok=True)
    save_las(path, PointCloud(xyz=points))
    return path


# --------------------------------------------------------------------------
# 1. reading
# --------------------------------------------------------------------------


def test_las_round_trips_through_the_writers_own_scale(tmp_path):
    points = _bumpy_points(cell=2.0, span=40.0)
    tile = _write_tile(tmp_path / "tile.las", points)
    read = lidar.read_points(tile)
    assert read.xyz.shape == points.shape
    # The writer quantises to millimetres; anything larger is a real defect.
    assert np.abs(read.xyz - points).max() < 2e-3


def test_laz_reads_through_the_compressor(tmp_path):
    """LAZ is only readable with a decompressor — and must be, when present."""
    points = _bumpy_points(cell=2.0, span=40.0)
    import laspy

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])  # same mm scale the app writer uses
    header.offsets = np.array(points.min(axis=0), dtype=np.float64)
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    laz = tmp_path / "tile.laz"
    las.write(str(laz))

    read = lidar.read_points(laz)
    # the writer quantises to millimetres (header scale), so that is the floor
    assert np.allclose(read.xyz, points, atol=2e-3)
    grid_laz = lidar.load_grid(laz)
    grid_las = lidar.load_grid(_write_tile(tmp_path / "tile.las", points))
    assert np.allclose(grid_laz.height, grid_las.height, equal_nan=True)


def test_unreadable_tile_is_an_actionable_error(tmp_path):
    bad = tmp_path / "not_a_tile.las"
    bad.write_bytes(b"this is not a LAS file at all" * 10)
    with pytest.raises(lidar.LidarError) as exc:
        lidar.read_points(bad)
    assert "unreadable LiDAR file" in str(exc.value)


def test_npz_points_read_verbatim(tmp_path):
    """A pre-baked `.npz` is the array certification compares against, so its
    `points` must come back exactly — no quantisation, no unit guessing."""
    points = _bumpy_points(cell=0.5, span=20.0)
    path = tmp_path / "baked.npz"
    np.savez_compressed(path, points=points)
    read = lidar.read_points(path)
    assert np.array_equal(read.xyz, points)


def test_npz_carries_classification_through(tmp_path):
    points = _bumpy_points(cell=1.0, span=20.0)
    cls = np.full(len(points), 2, dtype=np.uint8)
    path = tmp_path / "baked.npz"
    np.savez_compressed(path, points=points, classification=cls)
    read = lidar.read_points(path)
    assert read.classification is not None
    assert np.array_equal(read.classification, cls)


def test_npz_without_points_says_so(tmp_path):
    """A pre-baked grid IS a usable reference but has no returns to rasterise:
    the error must say that rather than report an empty cloud."""
    grid = lidar.rasterise(lidar.LidarPoints(xyz=_bumpy_points(cell=2.0, span=40.0)))
    path = lidar.write_grid_npz(grid, tmp_path / "dsm.npz")
    with pytest.raises(lidar.LidarError) as exc:
        lidar.read_points(path)
    assert "no 'points' array" in str(exc.value)


def test_npz_with_malformed_points_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, points=np.zeros((10, 2)))
    with pytest.raises(lidar.LidarError) as exc:
        lidar.read_points(path)
    assert "(N, 3)" in str(exc.value)


def test_canonical_npz_resolves_as_the_reference(tmp_path):
    """The stored-reference form is part of the accepted input set."""
    grid = lidar.rasterise(lidar.LidarPoints(xyz=_bumpy_points(cell=2.0, span=40.0)))
    ref = tmp_path / lidar.REFERENCE_DIRNAME
    ref.mkdir(parents=True)
    path = lidar.write_grid_npz(grid, ref / f"{lidar.REFERENCE_STEM}.npz")
    assert lidar.resolve_reference(tmp_path) == path


def test_dataset_side_discovery_finds_an_npz_sibling(tmp_path):
    """The dataset path picks up `.npz` beside the video, not only LAS/LAZ."""
    from app.services.data_video_service import dataset_lidar_source

    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    baked = tmp_path / "lidar.npz"
    np.savez_compressed(baked, points=_bumpy_points(cell=1.0, span=10.0))
    assert dataset_lidar_source(video) == baked


# --------------------------------------------------------------------------
# 2. rasterisation
# --------------------------------------------------------------------------


def test_grid_matches_the_sampler_and_keeps_the_maximum_return(tmp_path):
    points = _plane_points()
    grid = lidar.rasterise(lidar.LidarPoints(xyz=points))

    # The sampler dsm_accuracy uses must reproduce the surface the grid was
    # built from. The residual bound is the relief across ONE cell (the tile's
    # own resolution limit): a max-aggregated cell sits at the top of the
    # cell's relief, so it cannot agree with a cell-centre sample more closely
    # than that — but a transposed or flipped grid would miss by tens of
    # metres, which is what this pins.
    z, ok = dsm._sample(grid.height, grid.bounds, grid.gsd, points[:, 0], points[:, 1])
    assert ok.sum() > 0.9 * len(points)
    assert np.abs(points[ok, 2] - z).max() < 0.1

    # row 0 is the north edge (y1), column 0 the west edge (x0)
    x0, y0, x1, y1 = grid.bounds
    plane = lambda x, y: 100.0 + 0.04 * np.asarray(x) - 0.02 * np.asarray(y)
    assert grid.height[0, 0] == pytest.approx(float(plane(x0, y1)), abs=0.1)

    # two returns in one cell: the higher one is the surface
    stacked = lidar.LidarPoints(
        xyz=np.array([[1.0, 1.0, 5.0], [1.0, 1.0, 9.0], [3.0, 3.0, 1.0]])
    )
    g2 = lidar.rasterise(stacked, gsd=1.0)
    assert np.nanmax(g2.height) == pytest.approx(9.0)
    assert int(np.nanmax(g2.counts)) == 2


def test_unmeasured_cells_stay_nan_and_the_grid_coarsens(tmp_path):
    # a tile with a hole: the middle band has no returns at all
    points = _bumpy_points(cell=1.0, span=40.0)
    hole = points[(points[:, 1] < 15.0) | (points[:, 1] > 25.0)]
    grid = lidar.rasterise(lidar.LidarPoints(xyz=hole))
    assert np.isnan(grid.height).any(), "an unmeasured cell must not be invented"
    assert 0.0 < float(np.isnan(grid.height).mean()) < 0.5

    # a cell budget forces a coarser grid rather than an unloadable one
    coarse = lidar.rasterise(lidar.LidarPoints(xyz=points), gsd=0.5, max_cells=1_000)
    assert coarse.gsd > 0.5
    assert coarse.height.size <= 1_000


def test_spacing_is_the_ground_spacing_not_a_subsample_artifact():
    """The first end-to-end run of this path reported 1.41 m for a 1 m tile.

    Cause: estimating spacing from the nearest neighbours of a random subset —
    a subsample's neighbours are farther apart by 1/√fraction, exactly √2 at
    8:1 — which silently halves the reference's raster resolution. The
    estimate must therefore match the exact full-set nearest-neighbour spacing
    once the tile is big enough to have triggered the old subsampling.
    """
    from scipy.spatial import cKDTree

    xs = np.arange(0.0, 400.0, 1.0)
    X, Y = np.meshgrid(xs, xs)
    Z = np.sin(X / 3.0) + np.cos(Y / 2.5)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    assert len(points) > 20_000, "fixture must be large enough to have subsampled"

    d, _ = cKDTree(points[:, :2]).query(points[:, :2], k=2)
    exact = float(np.median(d[:, 1]))
    estimate = lidar._ground_spacing(points)
    assert exact == pytest.approx(1.0)
    assert estimate == pytest.approx(exact, rel=0.02), (estimate, exact)

    # and the resulting grid keeps the tile's own resolution
    grid = lidar.rasterise(lidar.LidarPoints(xyz=points))
    assert grid.gsd == pytest.approx(1.0, rel=0.02)


def test_spacing_ignores_terrain_roughness():
    """A 3-D nearest-neighbour distance grows with relief; the grid must not."""
    xs = np.arange(0.0, 60.0, 0.5)
    X, Y = np.meshgrid(xs, xs)
    flat = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    # ~0.46 m of relief per 0.5 m step: a 3-D estimate would inflate hugely
    rough = np.column_stack([X.ravel(), Y.ravel(), (0.6 * np.sin(X / 1.3)).ravel()])
    assert lidar._ground_spacing(flat) == pytest.approx(0.5, rel=0.02)
    assert lidar._ground_spacing(rough) == pytest.approx(0.5, rel=0.02)


def test_grid_npz_round_trips(tmp_path):
    grid = lidar.rasterise(lidar.LidarPoints(xyz=_bumpy_points(cell=2.0, span=40.0)))
    path = lidar.write_grid_npz(grid, tmp_path / "dsm.npz")
    again = lidar.load_grid(path)
    assert np.allclose(again.height, grid.height, equal_nan=True)
    assert again.gsd == pytest.approx(grid.gsd)
    assert tuple(again.bounds) == pytest.approx(tuple(grid.bounds))


# --------------------------------------------------------------------------
# 3. resolution — no registry
# --------------------------------------------------------------------------


def test_there_is_no_per_run_reference_registry():
    assert not hasattr(dsm, "REFERENCE_DSMS")
    assert not hasattr(dsm, "ReferenceDsm")


def test_resolution_reads_the_runs_own_workspace(tmp_path):
    ws = tmp_path / "run"
    ws.mkdir()
    assert lidar.resolve_reference(ws) is None  # no reference dir at all

    ref = ws / lidar.REFERENCE_DIRNAME
    ref.mkdir()
    # canonical name wins
    canonical = ref / "lidar.las"
    canonical.write_bytes(b"")
    assert lidar.resolve_reference(ws) == canonical

    # a single tile under its survey's own name is still accepted
    canonical.unlink()
    vendor = ref / "site_2024_survey.laz"
    vendor.write_bytes(b"")
    assert lidar.resolve_reference(ws) == vendor

    # two candidates must not be guessed between
    (ref / "other_tile.las").write_bytes(b"")
    assert lidar.resolve_reference(ws) is None


# --------------------------------------------------------------------------
# 4. absolute accuracy through the real services
# --------------------------------------------------------------------------


def _run_workspace(tmp_path: Path, *, reference: bool, offset=(10.0, -5.0)) -> Path:
    """A workspace holding a mesh of the bumpy surface, optionally with a tile."""
    import open3d as o3d

    ws = tmp_path / "run_e2e"
    (ws / "mesh").mkdir(parents=True)
    (ws / "dense").mkdir()

    span, cell = 120.0, 2.0
    xs = np.arange(0.0, span, cell)
    X, Y = np.meshgrid(xs, xs)
    Z = _surface(X, Y)
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    # the reconstruction sits at a known offset from the tile's frame — the
    # service must MEASURE that placement, not assume it
    verts = verts + np.array([offset[0], offset[1], 0.0])

    idx = np.arange(len(xs) ** 2).reshape(len(xs), len(xs))
    tri = []
    for r in range(len(xs) - 1):
        for c in range(len(xs) - 1):
            a, b, d, e = idx[r, c], idx[r, c + 1], idx[r + 1, c], idx[r + 1, c + 1]
            tri += [[a, b, d], [b, e, d]]
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(np.array(tri))
    )
    o3d.io.write_triangle_mesh(str(ws / "mesh" / "mesh.ply"), mesh)
    save_ply(ws / "dense" / "dense_model.ply", PointCloud(xyz=verts))

    if reference:
        ref = ws / lidar.REFERENCE_DIRNAME
        ref.mkdir()
        # A real tile is far denser than the mesh it validates: 0.5 m returns
        # against 2 m mesh vertices, so a max-aggregated cell's relief (and
        # therefore its bias against a cell-centre sample) stays small.
        _write_tile(ref / "lidar.las", _bumpy_points(cell=0.5, span=span))
    return ws


def test_run_without_a_reference_reports_no_reference(tmp_path):
    ws = _run_workspace(tmp_path, reference=False)
    assert dsm.run_dsm_accuracy(ws) == {"status": "no_reference"}


def test_run_with_a_held_out_tile_measures_its_own_offset(tmp_path):
    ws = _run_workspace(tmp_path, reference=True, offset=(10.0, -5.0))
    result = dsm.run_dsm_accuracy(ws)
    assert result["status"] == "ok", result
    dx, dy = result["measured_offset_m"]
    # The mesh sits at (+10, −5) from the tile's frame; the offset that puts it
    # back is therefore (−10, +5). Either sign convention being wrong would be
    # off by 20 m here, far outside the tolerance.
    assert abs(dx + 10.0) <= 0.1, result
    assert abs(dy - 5.0) <= 0.1, result
    assert result["mae_m"] < 0.5
    assert result["coverage"] > 0.6
    assert result["reference_kind"].startswith("held-out LiDAR")


def test_validation_uses_the_reference_and_the_ladder_follows(tmp_path):
    from app.services.metric_validation import validate_run_internal

    with_ref = _run_workspace(tmp_path / "a", reference=True, offset=(0.0, 0.0))
    report = validate_run_internal(with_ref, persist=False)
    assert report["reference_comparison"] is not None
    assert report["validation_kind"] == "held_out_reference"
    assert report["reference_type"].startswith("independent LiDAR")
    assert report["reference_source"]
    assert report["error_decomposition"] is not None
    # A measured reference is what the capability ladder keys off — and the
    # absolute claim is the measured status, never a hardcoded sentence.
    assert report["accuracy_summary"]["absolute"] != "NOT MEASURED — NO ACCURACY REPORT"

    without_ref = _run_workspace(tmp_path / "b", reference=False)
    plain = validate_run_internal(without_ref, persist=False)
    assert plain["reference_comparison"] is None
    assert plain["reference_type"] is None
    assert plain["validation_kind"] == "internal_consistency"
    assert plain["accuracy_certified"] is False
    assert plain["accuracy_summary"]["absolute"] == "NOT MEASURED — NO ACCURACY REPORT"


# --------------------------------------------------------------------------
# 5. upload gate
# --------------------------------------------------------------------------


def test_las_laz_and_npz_are_accepted_as_a_reference():
    """The gate accepts exactly what the reader can read: LAS/LAZ returns and
    the pre-baked `.npz` form (returns, a height grid, or both)."""
    assert safe_lidar_filename("survey.las") == "survey.las"
    assert safe_lidar_filename("/tmp/../site.LAZ") == "site.LAZ"
    assert safe_lidar_filename("baked.npz") == "baked.npz"
    assert safe_lidar_filename("/tmp/../site.NPZ") == "site.NPZ"
    for bad in ("notes.txt", "poses.csv", "no_extension"):
        with pytest.raises(Exception):
            safe_lidar_filename(bad)
