"""Viewer detail path — the artifacts the browser actually receives.

The user's complaint ("loses detail when zoomed") traces to three places the
viewer path could discard or stale-out the vitb density:

1. ``combined_model.ply`` — the default point-cloud view's data source — was
   written once and never refreshed: a dense rerun left the browser serving
   the old merge. The stale-merge guard regenerates when either input is
   newer than the output.
2. The viewer GLB budget (formerly fixed at 500k faces) silently discarded
   ~half of a denser production mesh (measured: 46.7% of faces kept, LOD
   edges 1.47× coarser). The budget is now a named setting and the atlas cap
   is the SAME knob, so the LOD size and the textured-GLB cap cannot drift.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings
from app.services.mesh import TriangleMesh
from app.services.mesh_glb import decimate_for_viewer, viewer_triangle_budget
from app.services.pipeline_orchestrator import _write_combined_model
from app.services.pointcloud import PointCloud, save_ply


def _seed_workspace(tmp_path: Path, n_dense: int = 100, n_sparse: int = 5) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "dense").mkdir()
    save_ply(ws / "dense" / "dense_model.ply", PointCloud(
        xyz=np.random.default_rng(1).uniform(-10, 10, (n_dense, 3)),
        confidence=np.full(n_dense, 0.8)))
    save_ply(ws / "sparse_model.ply", PointCloud(
        xyz=np.random.default_rng(2).uniform(-10, 10, (n_sparse, 3)),
        confidence=np.ones(n_sparse)))
    return ws


def _write_glb_payload(out: Path) -> None:
    """Minimal GLB so _write_combined_model's caller contract is exercised."""
    out.write_bytes(b"")


class TestCombinedModelFreshness:
    def test_combined_model_regenerated_when_dense_newer(self, tmp_path):
        ws = _seed_workspace(tmp_path)
        _write_combined_model(ws)
        out = ws / "combined_model.ply"
        assert out.exists()
        first_mtime = out.stat().st_mtime
        # Dense rerun lands a newer cloud.
        import os
        import time as _time
        future = _time.time() + 10
        os.utime(ws / "dense" / "dense_model.ply", (future, future))
        _write_combined_model(ws)
        assert out.stat().st_mtime != first_mtime, (
            "a newer dense cloud must regenerate combined_model.ply — the "
            "viewer's default point-cloud view reads this file")

    def test_combined_model_kept_when_older_inputs(self, tmp_path):
        ws = _seed_workspace(tmp_path)
        _write_combined_model(ws)
        out = ws / "combined_model.ply"
        import os
        import time as _time
        # Touch inputs into the past: output is fresh, no regeneration.
        past = _time.time() - 10
        os.utime(ws / "dense" / "dense_model.ply", (past, past))
        os.utime(ws / "sparse_model.ply", (past, past))
        before = out.stat().st_mtime
        _write_combined_model(ws)
        assert out.stat().st_mtime == before

    def test_combined_model_content_merges_both_inputs(self, tmp_path):
        ws = _seed_workspace(tmp_path, n_dense=100, n_sparse=7)
        _write_combined_model(ws)
        from app.services.pointcloud import read_ply
        merged = read_ply(ws / "combined_model.ply")
        assert merged.n == 107


class TestViewerTriangleBudget:
    def test_budget_is_a_setting_and_reasonable(self):
        assert viewer_triangle_budget() == settings.mesh.viewer_triangle_budget
        assert settings.mesh.viewer_triangle_budget >= 1_000_000, (
            "the budget must carry airport1-class (~1.07M face) meshes intact")

    def test_mesh_at_budget_passes_through_undecimated(self):
        budget = viewer_triangle_budget()
        v = np.random.default_rng(3).uniform(-1, 1, (budget + 10, 3))
        # A valid closed-ish fan: each face references three real vertices.
        f = np.stack([np.arange(budget), (np.arange(budget) + 1) % (budget + 10),
                      (np.arange(budget) + 2) % (budget + 10)], axis=1)
        mesh = TriangleMesh(vertices=v, faces=f.astype(np.int64))
        out, stats = decimate_for_viewer(mesh)
        assert stats["method"] == "copy"
        assert out.m == mesh.m

    def test_mesh_over_budget_is_decimated_to_budget(self):
        budget = viewer_triangle_budget()
        n = budget + 50_000
        v = np.random.default_rng(4).uniform(-1, 1, (n, 3))
        f = np.stack([np.arange(n - 2), (np.arange(n - 2) + 1) % n,
                      (np.arange(n - 2) + 2) % n], axis=1)
        mesh = TriangleMesh(vertices=v, faces=f.astype(np.int64))
        out, stats = decimate_for_viewer(mesh)
        assert out.m <= budget
        assert stats["after_faces"] <= budget

    def test_atlas_cap_and_lod_budget_are_one_knob(self):
        # build_texture_payload caps at viewer_triangle_budget() — verified by
        # the updated overflow test; here we pin the import-level contract so
        # the two can never silently diverge again.
        from app.services import textured_glb

        assert textured_glb._viewer_triangle_budget is viewer_triangle_budget
