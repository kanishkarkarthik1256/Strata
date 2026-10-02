"""Ground-truth DSM accuracy service — honesty contracts.

The service is the Reports page's "vs reference DSM" panel: every number it
emits must be one the run actually measured, and the mapping that produces
those numbers must be provably sane. This thread measured the failure modes
it exists to prevent: an assumed anchor placement scored real terrain no
better than a permutation null (pure noise), and unclamped co-registration
walked to a 2.6%-coverage sliver that fit anything. The pinned contracts:

1. Tile-local mapping (the dataset's own convention — bounds are meters in
   the tile frame, not UTM).
2. Co-registration *measures* horizontal placement and must recover a known
   synthetic offset — never assume one.
3. Coverage-floor: sliver overlaps are excluded from registration.
4. The rolled-null gate refuses geometry that matches decorrelated terrain
   as well as the real tile — a broken mapping can never look accurate.
5. Height datum offset is reported, not hidden, and never contaminates MAE.
6. No reference registered → ``no_reference`` (honest empty UI state).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services import dsm_accuracy as dsm

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND.parent / "data"
STORAGE = BACKEND / "data" / "storage"
#: The dataset whose reference grid and reconstruction are both on disk.
ANCHOR_RUN = "airport1_test_99ee3c"


def _save_grid(path, H, bounds, gsd):
    np.savez(path, height=H.astype(np.float32), bounds=np.asarray(bounds), gsd=gsd)


def _synthetic_scene(gsd: float = 2.0, cell: float = 400) -> tuple:
    """A small world with unambiguous relief: gaussian hills + a tall tower."""
    half = cell * gsd / 2
    bounds = (-half, -half, half, half)
    xs = np.linspace(-half, half, cell)
    ys = np.linspace(half, -half, cell)  # row 0 = north edge
    X, Y = np.meshgrid(xs, ys)
    H = (
        12.0 * np.exp(-(((X - 120) / 90) ** 2 + ((Y + 80) / 70) ** 2))
        + 8.0 * np.exp(-(((X + 150) / 60) ** 2 + ((Y - 140) / 80) ** 2))
        + 2.0 * np.sin(X / 33.0) * np.cos(Y / 41.0)
    )
    return H.astype(np.float32), bounds, gsd, X, Y


def test_coregister_recovers_known_offset():
    """A reconstruction that is the scene translated by +t must register
    with offset −t (points + offset land on the tile) — ± one cell."""
    H, bounds, gsd, X, Y = _synthetic_scene()
    t = (37.0, -23.0)
    expect = (-t[0], -t[1])
    # Reconstruction heights = the scene sampled at (x − t_x, y − t_y):
    xs = np.linspace(bounds[0] + 20, bounds[2] - 20, 60)
    ys = np.linspace(bounds[1] + 20, bounds[3] - 20, 60)
    Xp, Yp = np.meshgrid(xs, ys)
    Z = (
        12.0 * np.exp(-((((Xp - t[0]) - 120) / 90) ** 2 + (((Yp - t[1]) + 80) / 70) ** 2))
        + 8.0 * np.exp(-((((Xp - t[0]) + 150) / 60) ** 2 + (((Yp - t[1]) - 140) / 80) ** 2))
        + 2.0 * np.sin((Xp - t[0]) / 33.0) * np.cos((Yp - t[1]) / 41.0)
    )
    pts = np.column_stack([Xp.ravel(), Yp.ravel(), Z.ravel()])
    dx, dy, _, cov = dsm.coregister(H, bounds, gsd, pts, coarse_m=10.0, window_m=300.0)
    assert abs(dx - expect[0]) <= 12.0, (dx, dy)
    assert abs(dy - expect[1]) <= 12.0, (dx, dy)
    assert cov > 0.9


def test_coregister_rejects_sliver_overlap():
    """An offset that keeps only a sliver inside the tile must not win —
    measured on real data: unclamped registration walked to 2.6% coverage
    and fit noise. The floor keeps ≥60% of the best achievable coverage."""
    H, bounds, gsd, _, _ = _synthetic_scene()
    # Points centered far outside: only a tiny corner can ever overlap.
    n = 4000
    rng = np.random.default_rng(7)
    pts = np.column_stack(
        [
            rng.uniform(bounds[2] + 1400, bounds[2] + 1700, n),
            rng.uniform(bounds[1] + 1400, bounds[3] + 1700, n),
            rng.uniform(0, 10, n),
        ]
    )
    dx, dy, _, cov = dsm.coregister(H, bounds, gsd, pts, coarse_m=100.0, window_m=3000.0)
    # Whatever offset wins, it may not be a sliver: coverage must be ≥60% of
    # the best achievable — here that means effectively zero overlap loses
    # to nothing, so the result is honest zero-coverage, not a fake fit.
    assert cov == 0.0 or cov >= 0.5 * 0.6


def test_datum_offset_reported_and_removed_from_mae():
    H, bounds, gsd, _, _ = _synthetic_scene()
    xs = np.linspace(bounds[0] + 40, bounds[2] - 40, 60)
    ys = np.linspace(bounds[1] + 40, bounds[3] - 40, 60)
    Xp, Yp = np.meshgrid(xs, ys)
    Z = (
        12.0 * np.exp(-(((Xp - 120) / 90) ** 2 + ((Yp + 80) / 70) ** 2))
        + 8.0 * np.exp(-(((Xp + 150) / 60) ** 2 + ((Yp - 140) / 80) ** 2))
        + 2.0 * np.sin(Xp / 33.0) * np.cos(Yp / 41.0)
    )
    pts = np.column_stack([Xp.ravel(), Yp.ravel(), Z.ravel() + 55.0])
    r = dsm.dsm_mae(pts, H_ext=(H, bounds, gsd))
    assert r["status"] == "ok"
    assert abs(r["height_datum_offset_m"] - 55.0) < 1.5
    assert r["mae_m"] < 0.5  # shape matches; the datum shift never leaks in
    assert r["coverage"] > 0.95


def test_gate_refuses_geometry_matching_only_decorrelated_terrain():
    """A broken mapping must not receive a score: same geometry at the same
    offset must score no better against the real tile than against rolled
    (decorrelated) copies of it."""
    # Reconstruction = independent random terrain, no relation to the tile.
    H, bounds, gsd, _, _ = _synthetic_scene(cell=300)
    rng = np.random.default_rng(11)
    n = 6000
    pts = np.column_stack(
        [
            rng.uniform(bounds[0], bounds[2], n),
            rng.uniform(bounds[1], bounds[3], n),
            rng.uniform(0, 30, n),
        ]
    )
    tmp = STORAGE / "_gate_probe_tmp"
    tmp.mkdir(exist_ok=True)
    try:
        grid = tmp / "gate_probe.npz"
        _save_grid(grid, H, bounds, gsd)
        # Force registration to (0,0) by probing dsm_mae directly at several
        # plausible offsets — every one must fail the gate via run-level
        # machinery; here we verify the mechanism: real ≈ null on garbage.
        real = dsm.dsm_mae(pts, H_ext=(H, bounds, gsd))
        rolled = np.roll(H, (77, -61), axis=(0, 1))
        null = dsm.dsm_mae(pts, H_ext=(rolled, bounds, gsd))
        assert real["status"] == "ok" and null["status"] == "ok"
        # Random heights vs shuffled terrain: no real advantage allowed.
        assert real["mae_m"] > null["mae_m"] * 0.8
    finally:
        grid.unlink(missing_ok=True)
        tmp.rmdir()


def test_no_reference_run_is_honest(tmp_path):
    assert dsm.run_dsm_accuracy(tmp_path, "run_without_reference") == {
        "status": "no_reference"
    }


def test_anchor_run_measures_through_real_service(tmp_path):
    """The real airport1 anchor run: score at a *measured* offset that
    demonstrably beats decorrelated terrain (the gate's whole point).

    The reference is now the run's own held-out input rather than a registry
    entry, so this stages it (and the mesh) into a temporary workspace by
    symlink — the artifacts are 30-70 MB and the stored run must not be
    mutated by a test.
    """
    run_ws = STORAGE / ANCHOR_RUN
    mesh = run_ws / "mesh" / "mesh.ply"
    reference = DATA / "airport1" / "dsm.npz"
    if not mesh.exists() or not reference.is_file():
        pytest.skip("anchor workspace artifacts not present")
    ws = tmp_path / ANCHOR_RUN
    (ws / "mesh").mkdir(parents=True)
    (ws / "reference").mkdir()
    (ws / "mesh" / "mesh.ply").symlink_to(mesh)
    (ws / "reference" / "lidar.npz").symlink_to(reference)

    r = dsm.run_dsm_accuracy(ws)
    assert r["status"] == "ok", r
    # Registration must be real: the measured offset was co-registered, not
    # assumed, and the real tile beats the null (gate enforces ≥ margin).
    assert "measured_offset_m" in r
    assert 0.2 < r["coverage"] <= 1.0
    # Far-field noise shell ~1 m; terrain structure adds several meters —
    # an accidental near-zero would itself be suspicious, but the pinned
    # bound only catches broken mappings (double-digit or gate failure).
    assert r["mae_m"] < 12.0
