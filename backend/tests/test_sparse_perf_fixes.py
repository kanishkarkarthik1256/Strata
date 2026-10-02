"""Performance-batch regression tests.

Covers the runtime-audit fixes:
- the STRATA explicit BA must never silently no-op (backend=unavailable +
  observations=0 + initial==final==0) when a valid reconstruction with
  observations exists;
- reprojection measurement consumes observation-level pixels;
- the COLMAP feature cap is enforced at the database with descriptor/keypoint
  row integrity preserved, deterministically.
Synthetic geometry only — no run artifacts required.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from app.services.bundle_adjustment import (
    _measure_reprojection_error,
    run_bundle_adjustment,
)
from app.services.camera_pose_estimator import (
    ReconstructionResult,
    CameraPose,
    SparsePoint3D,
    _enforce_feature_cap,
)


def _synthetic_reconstruction(
    n_cams: int = 5, n_points: int = 60, noise_px: float = 0.3, seed: int = 0
) -> tuple[ReconstructionResult, dict[str, np.ndarray], np.ndarray]:
    """Small synthetic aerial geometry: cameras orbiting a ground patch.

    Returns (result, true_K_by_frame, true_points) with REAL observations
    (measured pixel per point per camera) so the BA has a graph to optimize.
    """
    rng = np.random.default_rng(seed)
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    pts = np.column_stack([
        rng.uniform(-8, 8, n_points), rng.uniform(-8, 8, n_points), rng.uniform(0, 2, n_points),
    ])
    result = ReconstructionResult(backend="colmap")
    for c in range(n_cams):
        theta = 2 * np.pi * c / n_cams
        C = np.array([15 * np.cos(theta), 15 * np.sin(theta), 12.0])
        # camera-to-world: z axis toward the origin, y roughly world -z.
        zax = -C / np.linalg.norm(C)
        yax = np.array([0.0, 0.0, -1.0])
        yax = yax - (yax @ zax) * zax
        yax /= np.linalg.norm(yax)
        xax = np.cross(yax, zax)
        R = np.column_stack([xax, yax, zax])
        name = f"frame_{c:06d}"
        result.cameras[name] = CameraPose(
            image_id=c + 1, frame_id=name, position=C, rotation=R,
            quaternion=np.array([1.0, 0, 0, 0]), intrinsics=K,
            distortion=np.zeros(5), is_estimated=True,
        )
    for p_idx, X in enumerate(pts):
        obs: list[tuple[str, np.ndarray]] = []
        for name, cam in result.cameras.items():
            xc = cam.rotation.T @ (X - cam.position)
            if xc[2] <= 0.1:
                continue
            uv = K @ xc
            pix = uv[:2] / uv[2] + rng.normal(0, noise_px, 2)
            obs.append((name, pix))
        if len(obs) >= 2:
            result.points3d.append(
                SparsePoint3D(
                    point_id=p_idx, position=X + rng.normal(0, 0.05, 3),
                    color=np.zeros(3, np.uint8), track_length=len(obs),
                    mean_reproj_error=0.5, observations=obs,
                )
            )
    result.num_registered = len(result.cameras)
    result.num_points = len(result.points3d)
    return result, {n: K for n in result.cameras}, pts


def test_ba_reports_real_solve_when_observations_exist():
    """The mandated gate: a valid model with observations must NOT yield
    backend=unavailable + observations=0 + initial==final==0."""
    result, _, _ = _synthetic_reconstruction()
    out = run_bundle_adjustment(result, max_iterations=10, refine_intrinsics=False)
    assert out.num_observations > 0, "BA must see the observation graph"
    assert not (
        out.backend == "unavailable"
        and out.num_observations == 0
        and out.initial_error == 0.0
        and out.final_error == 0.0
    ), "BA silently no-op'd on a valid reconstruction"
    assert out.backend == "pycolmap_joint"
    assert out.initial_error > 0.0, "initial residual must be measured"
    assert out.final_error <= out.initial_error + 1e-6
    assert out.converged is True
    assert out.degraded is False


def test_reprojection_measurement_consumes_observations():
    result, _, _ = _synthetic_reconstruction()
    err = _measure_reprojection_error(result)
    assert err > 0.0, "observations must drive the measured residual"
    # Corrupt every observation pixel -> error must explode, proving the
    # pixels (not cached per-point labels) are measured.
    for pt in result.points3d:
        pt.observations = [(f, pix + 500.0) for f, pix in pt.observations]
    assert _measure_reprojection_error(result) > err * 100


def test_ba_honest_when_no_observations():
    """Without observations the BA must report 'unavailable' and copy the
    initial residual — never fabricate a converged solve."""
    result, _, _ = _synthetic_reconstruction()
    for pt in result.points3d:
        pt.observations = []
    out = run_bundle_adjustment(result, max_iterations=5, refine_intrinsics=False)
    assert out.backend == "unavailable"
    assert out.num_observations == 0
    assert out.final_error == out.initial_error
    assert out.converged is False


def _make_db(path: Path, n_images: int = 2, n: int = 10000) -> None:
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY NOT NULL, "
        "rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB)"
    )
    con.execute(
        "CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY NOT NULL, "
        "rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB)"
    )
    rng = np.random.default_rng(0)
    for i in range(1, n_images + 1):
        kps = rng.random((n, 6)).astype(np.float32)
        kps[:, 0] *= 1920.0
        kps[:, 1] *= 1080.0
        desc = rng.integers(0, 255, (n, 128), dtype=np.uint8)
        con.execute("INSERT INTO keypoints VALUES (?,?,?,?)", (i, n, 6, kps.tobytes()))
        con.execute("INSERT INTO descriptors VALUES (?,?,?,?)", (i, n, 128, desc.tobytes()))
    con.commit()
    con.close()


def test_feature_cap_enforced_at_database(tmp_path: Path):
    db = tmp_path / "cap.db"
    _make_db(db, n=10000)
    stats = _enforce_feature_cap(db, cap=8192)
    assert stats["images_over_cap"] == 2
    con = sqlite3.connect(str(db))
    mx = con.execute("SELECT MAX(rows) FROM keypoints").fetchone()[0]
    mismatch = con.execute(
        "SELECT COUNT(*) FROM keypoints k JOIN descriptors d ON d.image_id=k.image_id "
        "WHERE k.rows != d.rows"
    ).fetchone()[0]
    kps_rows = con.execute("SELECT rows, cols, data FROM keypoints ORDER BY image_id").fetchall()
    con.close()
    assert mx <= 8192, "cap must hold for every image"
    assert mismatch == 0, "descriptor rows must stay aligned with keypoints"
    for rows, cols, blob in kps_rows:
        arr = np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)
        assert np.isfinite(arr).all()
        assert (arr[:, 0] >= 0).all() and (arr[:, 0] <= 1920).all()


def test_feature_cap_is_deterministic_and_idempotent(tmp_path: Path):
    db1, db2 = tmp_path / "a.db", tmp_path / "b.db"
    _make_db(db1, n=9000)
    _make_db(db2, n=9000)
    _enforce_feature_cap(db1, cap=8192)
    _enforce_feature_cap(db2, cap=8192)
    c1 = sqlite3.connect(str(db1)).execute(
        "SELECT data FROM keypoints ORDER BY image_id").fetchall()
    c2 = sqlite3.connect(str(db2)).execute(
        "SELECT data FROM keypoints ORDER BY image_id").fetchall()
    assert c1 == c2, "identical input DBs must cap identically"
    # idempotent: re-running on a capped DB changes nothing
    before = c1
    _enforce_feature_cap(db1, cap=8192)
    after = sqlite3.connect(str(db1)).execute(
        "SELECT data FROM keypoints ORDER BY image_id").fetchall()
    assert before == after


def test_feature_cap_under_cap_noop(tmp_path: Path):
    db = tmp_path / "small.db"
    _make_db(db, n=1000)
    stats = _enforce_feature_cap(db, cap=8192)
    assert stats["images_over_cap"] == 0


# ---------------------------------------------------------------------------
# Batch 4: CUDA enablement — SiftGPU auto-detect with honest CPU fallback.
# Reconstruction semantics never change with device: only throughput.
# ---------------------------------------------------------------------------

def _fake_torch(available: bool):
    """Minimal torch stub for device-detection tests."""
    import types
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: available))


def test_colmap_device_detects_cuda(monkeypatch):
    """A visible CUDA device must select Device.cuda."""
    import sys

    import pycolmap

    import app.services.camera_pose_estimator as cpe

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    assert cpe._colmap_device() == pycolmap.Device.cuda


def test_colmap_device_cpu_when_no_cuda(monkeypatch):
    """No CUDA -> Device.cpu (the current Intel-Mac environment)."""
    import sys

    import pycolmap

    import app.services.camera_pose_estimator as cpe

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    assert cpe._colmap_device() == pycolmap.Device.cpu


def test_extract_features_falls_back_to_cpu_on_gpu_error(monkeypatch):
    """CPU-only wheels raise on Device.cuda; the wrapper must downgrade, not fail."""
    import sys

    import pycolmap

    import app.services.camera_pose_estimator as cpe

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))

    calls = []

    def fake_extract(**kw):
        calls.append(kw.get("device"))
        if kw.get("device") == pycolmap.Device.cuda:
            raise RuntimeError("CUDA SIFT not compiled into this wheel")
        return None

    monkeypatch.setattr(pycolmap, "extract_features", fake_extract)
    used = cpe._extract_features_with_fallback(
        db_path=Path("x.db"), image_dir=Path("img"),
        sift_options=pycolmap.SiftExtractionOptions(),
        camera_mode=pycolmap.CameraMode.AUTO, camera_model="SIMPLE_RADIAL",
    )
    assert used == "cpu"
    assert calls == [pycolmap.Device.cuda, None]  # tried GPU, then CPU


def test_extract_features_uses_cuda_when_available(monkeypatch):
    import sys

    import pycolmap

    import app.services.camera_pose_estimator as cpe

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))

    calls = []

    def fake_extract(**kw):
        calls.append(kw.get("device"))
        return None

    monkeypatch.setattr(pycolmap, "extract_features", fake_extract)
    used = cpe._extract_features_with_fallback(
        db_path=Path("x.db"), image_dir=Path("img"),
        sift_options=pycolmap.SiftExtractionOptions(),
        camera_mode=pycolmap.CameraMode.AUTO, camera_model="SIMPLE_RADIAL",
    )
    assert used == "cuda"
    assert calls == [pycolmap.Device.cuda]


def test_match_falls_back_to_cpu_when_no_torch(monkeypatch):
    """No torch at all -> CPU matching, CUDA path never attempted."""
    import sys

    import pycolmap

    import app.services.camera_pose_estimator as cpe

    monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError

    called = {}

    def real_match(database_path, sift_options=None, device=None):
        called["device"] = device
        return None

    monkeypatch.setattr(pycolmap, "match_exhaustive", real_match)
    used = cpe._match_with_fallback(Path("x.db"), "exhaustive", pycolmap.SiftMatchingOptions())
    assert used == "cpu"
    assert called["device"] is None


def test_match_uses_cuda_when_available(monkeypatch):
    import sys

    import pycolmap

    import app.services.camera_pose_estimator as cpe

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))

    called = {}

    def fake_match(database_path, sift_options=None, device=None):
        called["device"] = device
        return None

    monkeypatch.setattr(pycolmap, "match_exhaustive", fake_match)
    used = cpe._match_with_fallback(Path("x.db"), "exhaustive", pycolmap.SiftMatchingOptions())
    assert used == "cuda"
    assert called["device"] == pycolmap.Device.cuda


def test_reconstruction_result_detail_field():
    """detail carries the device provenance through ReconstructionResult."""
    r = ReconstructionResult()
    r.detail = {"sift_device": "cpu", "match_device": "cpu"}
    assert r.detail["sift_device"] == "cpu"


# ---------------------------------------------------------------------------
# SRT telemetry compatibility (SIH): DJI .srt flight logs are accepted at
# upload, converted to the canonical flight_poses.csv, and consumed by the
# telemetry-assisted sparse path. Synthetic log only — no dataset needed.
# ---------------------------------------------------------------------------

_SRT_BLOCK = (
    "{n}\n"
    "1970-01-01 00:00:{sec:02d}.000\n"
    "[frame_cnt: {n} ]\n"
    "[GPS(7) ] [latitude: 37.4419000] [longitude: -122.1530000] "
    "[rel_alt: 50.000 abs_alt: 120.000]\n"
    "[GIMBAL] [gb_yaw: {yaw:.1f} gb_pitch: -30.0 gb_roll: 0.0]\n"
    "[focal_len: 24]\n\n"
)


def _write_srt(path: Path, n: int = 120, step_m: float = 0.5) -> None:
    """Synthetic northbound DJI SRT log: n blocks, ~3 Hz fix cadence."""
    lat0, lon0 = 37.4419, -122.1530
    dlat = step_m / 111_320.0  # ~meters north per fix
    blocks = []
    for i in range(1, n + 1):
        # One GPS fix every 10 frames (3 Hz at 30 fps); zero-order hold fills.
        fix = i - (i - 1) % 10
        lat = lat0 + (fix - 1) * dlat
        yaw = 0.0  # flying north
        blocks.append(_SRT_BLOCK.format(n=i, sec=i // 30, yaw=yaw)
                      .replace("37.4419000", f"{lat:.7f}"))
    path.write_text("\n".join(blocks), encoding="utf-8")


def test_srt_upload_extension_accepted():
    from app.services.upload_service import safe_telemetry_filename

    assert safe_telemetry_filename("DJI_0001.SRT") == "DJI_0001.SRT"


def test_srt_upload_extension_still_rejects_junk():
    from app.exceptions import InvalidVideoError
    from app.services.upload_service import safe_telemetry_filename

    with pytest.raises(InvalidVideoError):
        safe_telemetry_filename("log.exe")


def test_srt_parses_and_converts(tmp_path: Path):
    from app.services.dji_srt_telemetry import parse_srt, srt_to_flight_poses

    srt = tmp_path / "video.SRT"
    _write_srt(srt)
    frames = parse_srt(srt)
    assert len(frames) == 120
    assert frames[0].frame_cnt == 1

    out = tmp_path / "flight_poses.csv"
    prov = srt_to_flight_poses(srt, out, video_fps=30.0)
    assert prov["frames"] == 120
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("frame_id,x,y,z,qw,qx,qy,qz,latitude,longitude")
    assert len(lines) == 121
    # Northbound flight: y (ENU north) must increase monotonically.
    ys = [float(l.split(",")[2]) for l in lines[1:]]
    assert all(b >= a for a, b in zip(ys, ys[1:]))


def test_srt_conversion_rejects_non_dji(tmp_path: Path):
    from app.services.dji_srt_telemetry import parse_srt

    srt = tmp_path / "subs.SRT"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nJust subtitles.\n\n")
    with pytest.raises(ValueError):
        parse_srt(srt)


def test_srt_to_flight_poses_invalid_fps(tmp_path: Path):
    from app.services.dji_srt_telemetry import srt_to_flight_poses

    srt = tmp_path / "video.SRT"
    _write_srt(srt)
    with pytest.raises(ValueError):
        srt_to_flight_poses(srt, tmp_path / "out.csv", video_fps=0.0)


def test_single_pass_marker_in_pipeline_report_shape():
    """The report contract carries the SIH single-pass marker."""
    import inspect

    from app.services import pipeline_orchestrator as po

    src = inspect.getsource(po._finish)
    assert "single_pass_enforced" in src
    assert "input_sequence_count" in src
