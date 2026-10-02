"""Held-out LiDAR reference — the single owner of LiDAR inputs.

A LiDAR scan is a **reference, never an input**: nothing in the
frames/sparse/depth/dense/mesh path may read it. It is consumed by exactly
two readers, both of which measure absolute accuracy *after* the model
exists:

* :func:`app.services.dsm_accuracy.run_dsm_accuracy` — height-grid (DSM)
  comparison, co-registered by translation search and gated against a
  rolled null;
* :func:`app.services.metric_validation.validate_run_internal` — the
  point-cloud comparison that feeds the certification ladder.

Formats
-------``.las`` and ``.laz`` through ``laspy`` (``lazrs`` supplies the LAZ
      decompressor; ``laspy`` alone reads uncompressed LAS). A pre-baked ``.npz``
      may be supplied directly as well
carrying ``height``/``bounds``/``gsd`` is read as-is, so a reference that was
already projected keeps working unchanged.

Rasterisation rule
------------------
A LiDAR tile is a set of returns; a reconstructed model is a *surface*, so
each cell keeps the **maximum return height** — the top of the surface — and
the return count is kept beside it so "measured once, noisily" is
distinguishable from "measured densely". Cells with no return are ``NaN`` and
every comparison masks them rather than inventing an interpolated height.

Grid convention (the contract ``dsm_accuracy`` samples against):
``bounds`` is ``(x0, y0, x1, y1)`` in the file's own projected metres,
``height`` is ``(rows, cols)`` with column 0 at ``x0`` and **row 0 at
``y1``** (north-up, descending row axis), and cell ``(r, c)`` covers
``[x0 + c·gsd, x0 + (c+1)·gsd) × [y1 − (r+1)·gsd, y1 − r·gsd)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: What a user may upload and what a dataset may ship beside its video.
LAS_LAZ_EXTENSIONS = (".las", ".laz")
#: The pre-baked numpy form. It may carry ``points`` (returns, with optional
#: ``classification``), a ``height``/``bounds``/``gsd`` grid, or both.
NPZ_EXTENSION = ".npz"
#: Everything a LiDAR input may be SUPPLIED as and a reference STORED as.
LIDAR_EXTENSIONS = (*LAS_LAZ_EXTENSIONS, NPZ_EXTENSION)
#: Workspace directory holding a run's held-out reference. Deliberately a
#: dedicated directory (nothing else writes here) so "held out" is a property
#: of the path, not a convention nobody can check.
REFERENCE_DIRNAME = "reference"
#: Canonical stem inside :data:`REFERENCE_DIRNAME`.
REFERENCE_STEM = "lidar"
#: Cells beyond this force a coarser grid (a 40 km × 40 km tile at 20 cm is
#: 40 billion cells — the cap keeps a reference loadable instead of OOM).
MAX_CELLS = 16_000_000


class LidarError(RuntimeError):
    """A LiDAR reference could not be read — always actionable, never a guess."""


@dataclass(frozen=True)
class LidarPoints:
    """Returns as measured, in the file's own projected metres."""

    xyz: np.ndarray  # (N, 3) float64
    classification: np.ndarray | None = None


@dataclass(frozen=True)
class HeightGrid:
    """Rasterised surface + the return count behind each cell."""

    height: np.ndarray  # (H, W) float64, NaN where no return
    bounds: tuple[float, float, float, float]  # x0, y0, x1, y1
    gsd: float
    counts: np.ndarray | None = None


def npz_keys(path: Path) -> set[str]:
    """Array names inside a pre-baked ``.npz`` reference.

    One owner for opening the archive, so "what does this file contain?" has
    a single answer whether the caller wants returns, a grid, or to decide
    between them.
    """
    path = Path(path)
    try:
        with np.load(path) as d:
            return set(d.files)
    except Exception as exc:
        raise LidarError(f"unreadable LiDAR file {path.name}: {str(exc)[:200]}") from exc


def read_npz_points(path: Path) -> LidarPoints:
    """Returns carried by a pre-baked ``.npz`` (key ``points``).

    ``points`` is read as-is — no unit guessing and no reprojection, because
    this is the same array the certification path compares against. An
    optional ``classification`` travels with it.

    A grid-only ``.npz`` (the ``height``/``bounds``/``gsd`` form
    :func:`write_grid_npz` produces) is still a usable reference, but it has
    no returns to read; saying so is more useful than returning an empty cloud.
    """
    path = Path(path)
    files = npz_keys(path)
    if "points" not in files:
        raise LidarError(
            f"{path.name} carries no 'points' array — a pre-baked height grid "
            f"is a reference but has no returns to rasterise; supply the tile "
            f"as .las/.laz, or add a 'points' (N, 3) array"
        )
    with np.load(path) as d:
        xyz = np.asarray(d["points"], dtype=np.float64)
        classification = (
            np.asarray(d["classification"]) if "classification" in files else None
        )
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise LidarError(
            f"{path.name}: 'points' must be an (N, 3) array, got shape {xyz.shape}"
        )
    if len(xyz) == 0:
        raise LidarError(f"{path.name} contains no points")
    return LidarPoints(xyz=xyz, classification=classification)


def read_points(path: Path) -> LidarPoints:
    """Read LAS/LAZ returns (or an ``.npz``'s ``points``) as ``(N, 3)`` metres.

    ``laspy`` returns scaled floats (the header's scale/offset already
    applied), so no unit guessing happens here. A missing LAZ decompressor
    is reported as what it is, with the extra that fixes it.
    """
    path = Path(path)
    if path.suffix.lower() == NPZ_EXTENSION:
        return read_npz_points(path)
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise LidarError(
            "laspy is not installed — install the backend's declared dependencies "
            "(`pip install laspy`) to read LiDAR references"
        ) from exc

    try:
        las = laspy.read(str(path))
    except Exception as exc:  # laspy raises a family of errors for bad input
        message = str(exc)
        if path.suffix.lower() == ".laz" and "laz" in message.lower():
            raise LidarError(
                "LAZ reading needs a decompressor: install `lazrs` "
                "(`pip install lazrs`), or supply the same tile as .las"
            ) from exc
        raise LidarError(f"unreadable LiDAR file {path.name}: {message[:200]}") from exc

    xyz = np.column_stack(
        [
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        ]
    )
    if len(xyz) == 0:
        raise LidarError(f"{path.name} contains no points")

    classification = None
    try:
        classification = np.asarray(las.classification)
    except Exception:  # point format without a classification field
        classification = None
    return LidarPoints(xyz=xyz, classification=classification)


def _ground_spacing(points: np.ndarray, sample: int = 20_000) -> float:
    """Robust **ground** spacing of the returns, in metres.

    The grid resolution is derived from the data rather than assumed: a tile
    delivered at 1 m must not be rasterised onto a 10 cm grid (which would
    leave 99% of cells empty and make coverage meaningless), and rasterising a
    1 m tile onto a 1.41 m grid would silently halve the reference's detail.

    Measured two ways, because both easy estimators are wrong:

    * **never in 3-D** — terrain roughness adds to a nearest-neighbour
      distance, so a rough tile reports a coarser spacing than it was flown at
      (1.2% coarse on a 0.5 m test tile, enough to shift every cell boundary
      off the data it contains);
    * **never from a subsample's nearest neighbours** — the nearest neighbour
      of a random *subset* is farther apart than the spacing of the set, by
      ``1/√fraction`` (a 1 m tile subsampled 8:1 reports 1.41 m, exactly √2).
      A real 160,000-return tile hit precisely that on the first end-to-end
      run of this path.

    So: return density is measured on the full set through a coarse occupancy
    grid (O(n), no tree to build), using the **median over occupied cells** so
    empty margins cannot move the estimate. Small tiles fall back to the exact
    nearest-neighbour distance, which is cheap there.

    The two agree on a uniformly flown tile. They diverge on a clustered one
    (one dense patch plus a sparse margin): the occupancy median then reports
    the spacing of the *prevailing area*, not of the densest patch, and that is
    the scale wanted here — a grid resolved to one patch would leave the rest
    of the tile mostly empty.
    """
    horizontal = np.asarray(points, dtype=np.float64)[:, :2]
    n = len(horizontal)
    if n < 2:
        return 1.0
    if n <= sample:
        try:
            from scipy.spatial import cKDTree

            d, _ = cKDTree(horizontal).query(horizontal, k=2)
            nn = d[:, 1]
            finite = nn[np.isfinite(nn) & (nn > 0)]
            if len(finite):
                return float(np.median(finite))
        except Exception:  # pragma: no cover - scipy is a declared dependency
            pass

    x, y = horizontal[:, 0], horizontal[:, 1]
    span_x, span_y = float(np.ptp(x)), float(np.ptp(y))
    bbox = max(span_x * span_y, 1e-9)
    if span_x <= 0 or span_y <= 0:
        return 1.0
    # A probe cell ~4× the expected spacing holds ~16 returns, so the median
    # count over occupied cells is a stable density estimate.
    cell = 4.0 * float(np.sqrt(bbox / n))
    if not np.isfinite(cell) or cell <= 0:
        return 1.0
    ix = ((x - x.min()) / cell).astype(np.int64)
    iy = ((y - y.min()) / cell).astype(np.int64)
    keys = ix * (int(iy.max()) + 1) + iy
    _, counts = np.unique(keys, return_counts=True)
    median_count = float(np.median(counts))
    if median_count <= 0:
        return 1.0
    spacing = cell / float(np.sqrt(median_count))
    return spacing if spacing > 0 else 1.0


def rasterise(
    points: LidarPoints,
    gsd: float | None = None,
    *,
    max_cells: int = MAX_CELLS,
) -> HeightGrid:
    """Rasterise returns to a maximum-height surface grid.

    ``gsd`` defaults to the measured median return spacing. The grid is
    coarsened (doubling) until it fits ``max_cells``, and the coarsening is
    reported through the returned ``gsd`` so a caller never has to guess what
    resolution it actually got.
    """
    xyz = np.asarray(points.xyz, dtype=np.float64)
    if len(xyz) == 0:
        raise LidarError("cannot rasterise an empty point set")

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = float(np.min(y)), float(np.max(y))
    if gsd is None:
        gsd = _ground_spacing(xyz)
    gsd = float(gsd)
    if not np.isfinite(gsd) or gsd <= 0:
        gsd = 1.0

    def dims(g: float) -> tuple[int, int]:
        return (
            max(int(np.floor((y1 - y0) / g)) + 1, 1),
            max(int(np.floor((x1 - x0) / g)) + 1, 1),
        )

    rows, cols = dims(gsd)
    while rows * cols > max_cells:
        gsd *= 2.0
        rows, cols = dims(gsd)

    # row 0 is y1 (north-up); see the module docstring for the full contract.
    col = np.clip(((x - x0) / gsd).astype(np.int64), 0, cols - 1)
    row = np.clip(((y1 - y) / gsd).astype(np.int64), 0, rows - 1)
    flat = row * cols + col

    height = np.full(rows * cols, -np.inf, dtype=np.float64)
    np.maximum.at(height, flat, z)
    counts = np.bincount(flat, minlength=rows * cols).astype(np.int32)

    empty = ~np.isfinite(height)
    height[empty] = np.nan
    return HeightGrid(
        height=height.reshape(rows, cols),
        bounds=(x0, y0, x1, y1),
        gsd=gsd,
        counts=counts.reshape(rows, cols),
    )


def write_grid_npz(grid: HeightGrid, path: Path) -> Path:
    """Cache a rasterised grid in the shape ``dsm_accuracy`` already loads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        height=grid.height,
        bounds=np.asarray(grid.bounds, dtype=np.float64),
        gsd=np.asarray(grid.gsd, dtype=np.float64),
    )
    return path


def load_grid(path: Path, *, gsd: float | None = None) -> HeightGrid:
    """Load a reference grid from ``.npz`` (as-is) or LAS/LAZ (rasterised)."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        try:
            d = np.load(path)
            return HeightGrid(
                height=np.asarray(d["height"], dtype=np.float64),
                bounds=tuple(float(v) for v in d["bounds"]),
                gsd=float(d["gsd"]),
            )
        except Exception as exc:
            raise LidarError(f"unreadable reference grid {path.name}: {str(exc)[:200]}") from exc
    return rasterise(read_points(path), gsd=gsd)


def resolve_reference(workspace: Path) -> Path | None:
    """The run's held-out LiDAR reference, or None when it has none.

    Canonical name first (``reference/lidar.{las,laz,npz}``); failing that, a
    single LiDAR file anywhere in ``reference/`` is accepted, so a hand-placed
    tile with its dataset's own filename still works.
    """
    folder = Path(workspace) / REFERENCE_DIRNAME
    if not folder.is_dir():
        return None
    for ext in LIDAR_EXTENSIONS:
        candidate = folder / f"{REFERENCE_STEM}{ext}"
        if candidate.is_file():
            return candidate
    found = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in LIDAR_EXTENSIONS
    ]
    return found[0] if len(found) == 1 else None


def reference_points(workspace: Path) -> tuple[LidarPoints, Path] | None:
    """The held-out returns themselves (not a grid) plus the file they came from.

    A pre-baked ``.npz`` may carry ``points`` (with optional ``normals`` and
    ``classification``) — the shape the certification path compares against —
    in which case those are used directly instead of rasterising.
    """
    path = resolve_reference(workspace)
    if path is None:
        return None
    if path.suffix.lower() == ".npz":
        try:
            d = np.load(path)
        except Exception as exc:
            raise LidarError(f"unreadable reference {path.name}: {str(exc)[:200]}") from exc
        if "points" not in d.files:
            return None  # a height grid only: no cloud to compare point-to-point
        pts = LidarPoints(
            xyz=np.asarray(d["points"], dtype=np.float64),
            classification=(
                np.asarray(d["classification"]) if "classification" in d.files else None
            ),
        )
        return pts, path
    return read_points(path), path
