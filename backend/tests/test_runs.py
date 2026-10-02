"""Tests for the Phase 11.1 runs API (discovery and artifacts).

Covers: run discovery from the runs directory, manifest access, safe artifact
serving (traversal/extension rejection, 404s), and point-count extraction from
real PLY headers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.config.settings import settings
from app.services import run_service


def _make_run(base: Path, run_id: str, points: int = 1234) -> Path:
    """Create a minimal run dir with manifest + a small PLY + poses."""
    run = base / run_id
    run.mkdir(parents=True)
    manifest = {
        "run_id": run_id, "dataset": "shitan", "mission": "ms1",
        "started_at": "2026-09-09T13:12:51", "completed_at": "2026-09-09T13:20:00",
        "status": "PASS", "pipeline_version": "9.5.1",
        "stages": {"sparse": {"status": "PASS"}, "dense": {"status": "PASS"}},
        "metrics": {"total_stages": 2, "passed_stages": 2},
        "artifacts": {}, "limitations": [], "dependencies": {"pycolmap_version": "3.12.5"},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    dense = run / "dense"
    dense.mkdir()
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {points}\nproperty float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode()
    (dense / "dense_model.ply").write_bytes(header + b"\x00" * 12 * points)
    poses = {"frames": [{"frame_id": f"f{i}", "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0],
                         "gps": {"lat": 24.5, "lon": 120.9, "alt": 300.0}} for i in range(3)]}
    (run / "poses.json").write_text(json.dumps(poses))
    return run


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch):
    """Point the runs root at a temp dir for the duration of a test."""
    monkeypatch.setattr(settings.storage, "runs_dir", str(tmp_path))
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path / "storage"))
    return tmp_path


@pytest.fixture(autouse=True)
def _fresh_runs_cache():
    """list_runs() caches for 2s — isolate every test from earlier caches."""
    run_service.invalidate_runs_cache()


# ---------------------------------------------------------------- analysis --


def _write_dense_ply_xyz(run: Path, xs, ys, zs) -> Path:
    """Write a minimal binary PLY in the pipeline's own vertex layout
    (xyz + rgb + confidence + observations + residual), as save_ply does."""
    import struct

    n = len(xs)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float confidence\nproperty int observations\nproperty float residual\n"
        "end_header\n"
    ).encode()
    rows = []
    for x, y, z in zip(xs, ys, zs):
        rows.append(struct.pack("<fffBBBfif",
                                float(x), float(y), float(z), 200, 200, 200, 0.9, 3, 0.1))
    dense = run / "dense"
    dense.mkdir(exist_ok=True)
    path = dense / "dense_model.ply"
    path.write_bytes(header + b"".join(rows))
    return path


@pytest.mark.anyio
async def test_analysis_route_reports_area_for_real_surface(runs_dir: Path, client: AsyncClient):
    """The analysis endpoint computes area/elevation from the actual PLY —
    and says so. A 100 m x 50 m planar grid at z=5 must yield ≈5000 m² with
    min z 5 and max z 5 (single-plane surface, robust relief 0)."""
    run = _make_run(runs_dir, "analysis_run")
    xs, ys, zs = [], [], []
    # 0.5 m grid over 100 m x 50 m: every 1 m² cell holds ≥4 points, so the
    # density-support gate (>=3 pts/cell) counts the full ground as modelled.
    for i in range(201):
        for j in range(101):
            xs.append(i * 0.5)
            ys.append(j * 0.5)
            zs.append(5.0)
    _write_dense_ply_xyz(run, xs, ys, zs)

    res = await client.get("/api/runs/analysis_run/analysis")
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is True
    assert body["run_id"] == "analysis_run"
    site = body["site"]
    # Occupied 1 m² cells: 101 x 51 bbox, edge cells can round down.
    assert 4850 <= site["area_m2"] <= 5201, site["area_m2"]
    assert site["area_method"]
    # On a full rectangular grid the hull equals the occupied area.
    assert site["outline_area_m2"] >= site["area_m2"] * 0.97, site["outline_area_m2"]
    elev = body["elevation"]
    assert elev["min"] == pytest.approx(5.0, abs=0.01)
    assert elev["max"] == pytest.approx(5.0, abs=0.01)
    assert elev["highest"]["enu"][2] == pytest.approx(5.0, abs=0.01)
    mv = body.get("metric_validation")
    assert mv is None or "internal_validation" in mv


@pytest.mark.anyio
async def test_analysis_route_honest_when_no_surface(runs_dir: Path, client: AsyncClient):
    """A run without a dense surface gets available=false — never a fabricated
    area figure."""
    _make_run(runs_dir, "empty_run")
    res = await client.get("/api/runs/empty_run/analysis")
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is False
    assert body.get("site") is None
    assert body.get("reason")


async def test_demo_runs_tagged_by_origin_and_fast_demo_honest(runs_dir: Path, client: AsyncClient, monkeypatch):
    """Demo selection is by discovery root, never by run-id substring.

    In production the legacy precomputed-demo root (backend/outputs) IS the
    resolved runs root, so demo runs are discovered normally and tagged
    ``is_demo`` by their parent directory. fast-demo selects only tagged
    runs and refuses (404) rather than substituting an arbitrary mission.
    """
    # Mirror production: runs root == legacy demo root (backend/outputs);
    # normal upload missions live under the storage root instead.
    monkeypatch.setattr(run_service, "_LEGACY_DEMO_ROOT", runs_dir.resolve())
    _make_run(runs_dir, "shitan_ms1_20260909_131251")
    # A normal mission whose id merely *contains* "shitan".
    storage_root = settings.storage.base_dir
    _make_run(storage_root, "shitan_like_mission_20260912_000000")
    run_service.invalidate_runs_cache()

    resp = await client.get("/api/runs")
    assert resp.status_code == 200
    by_id = {r["run_id"]: r for r in resp.json()["runs"]}
    assert by_id["shitan_ms1_20260909_131251"]["is_demo"] is True
    assert by_id["shitan_like_mission_20260912_000000"]["is_demo"] is False

    resp = await client.get("/api/demo/fast-demo")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "shitan_ms1_20260909_131251"

    # With the demo root elsewhere (no tagged runs), fast-demo must 404 —
    # never fall back to a normal mission.
    monkeypatch.setattr(run_service, "_LEGACY_DEMO_ROOT", Path("/nonexistent-demo-root"))
    run_service.invalidate_runs_cache()
    resp = await client.get("/api/demo/fast-demo")
    assert resp.status_code == 404


async def test_list_runs_discovers_and_counts(runs_dir: Path, client: AsyncClient):
    _make_run(runs_dir, "shitan_ms1_20260909_131251", points=2626577)
    _make_run(runs_dir, "other_run_20260909_000000", points=99)

    resp = await client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 2
    by_id = {r["run_id"]: r for r in data["runs"]}
    main = by_id["shitan_ms1_20260909_131251"]
    assert main["status"] == "PASS"
    assert main["dense_points"] == 2626577  # read from the actual PLY header
    assert main["cameras"] == 3
    assert main["gps_points"] == 3
    assert any(a["path"] == "dense/dense_model.ply" for a in main["artifacts"])
    assert main["dependencies"]["pycolmap_version"] == "3.12.5"


async def test_run_detail_and_manifest(runs_dir: Path, client: AsyncClient):
    _make_run(runs_dir, "shitan_ms1_20260909_131251")
    resp = await client.get("/api/runs/shitan_ms1_20260909_131251")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["manifest"]["status"] == "PASS"

    resp = await client.get("/api/runs/shitan_ms1_20260909_131251/manifest")
    assert resp.status_code == 200
    assert resp.json()["dataset"] == "shitan"


async def test_manifest_404_for_unknown_run(runs_dir: Path, client: AsyncClient):
    resp = await client.get("/api/runs/not_a_real_run/manifest")
    assert resp.status_code == 404


async def test_artifact_serving(runs_dir: Path, client: AsyncClient):
    _make_run(runs_dir, "shitan_ms1_20260909_131251", points=10)
    resp = await client.get("/api/runs/shitan_ms1_20260909_131251/artifact/dense/dense_model.ply")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.content[:9] == b"ply\nforma"
    assert b"element vertex 10" in resp.content[:200]

    resp = await client.get("/api/runs/shitan_ms1_20260909_131251/artifact/poses.json")
    assert resp.status_code == 200
    assert resp.json()["frames"][0]["gps"]["lat"] == 24.5


async def test_artifact_security(runs_dir: Path, client: AsyncClient):
    _make_run(runs_dir, "shitan_ms1_20260909_131251")
    outside = runs_dir.parent / "secret.txt"
    outside.write_text("top secret")

    # traversal via encoded slash (rejected at routing or validation — either way blocked)
    resp = await client.get("/api/runs/shitan_ms1_20260909_131251/artifact/..%2F..%2Fsecret.txt")
    assert resp.status_code in (400, 404)
    # explicit traversal segment (raw ../ collapses at routing; still blocked)
    resp = await client.get("/api/runs/shitan_ms1_20260909_131251/artifact/../manifest.json")
    assert resp.status_code in (400, 404)
    # unknown extension
    resp = await client.get("/api/runs/shitan_ms1_20260909_131251/artifact/dense_model.ply.exe")
    assert resp.status_code == 400
    # valid name but missing file
    resp = await client.get("/api/runs/shitan_ms1_20260909_131251/artifact/nope.json")
    assert resp.status_code == 400
    # invalid run id
    resp = await client.get("/api/runs/..%2F..%2Fetc/manifest")
    assert resp.status_code == 404 or resp.status_code == 400


def test_run_service_path_rules(runs_dir: Path):
    _make_run(runs_dir, "shitan_ms1_20260909_131251")
    with pytest.raises(run_service.InvalidArtifactError):
        run_service.resolve_artifact("shitan_ms1_20260909_131251", "../manifest.json")
    with pytest.raises(run_service.InvalidArtifactError):
        run_service.resolve_artifact("shitan_ms1_20260909_131251", "/etc/passwd")
    with pytest.raises(run_service.InvalidArtifactError):
        run_service.resolve_artifact("shitan_ms1_20260909_131251", "dense_model.ply.exe")
    with pytest.raises(run_service.InvalidArtifactError):
        run_service.run_dir("../../etc")  # invalid characters/segments rejected before lookup