"""Trajectory sync module tests.

Covers the single-pass root-cause fix: telemetry -> ENU position prior,
stale-fix collapse, jump rejection, temporal sync, similarity placement
(projection invariance), composite gate, and continuity diagnostics.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from app.services import trajectory_sync as tsync
from app.services.camera_pose_estimator import (
    CameraPose,
    ReconstructionResult,
    SparsePoint3D,
)


def _rot(axis, deg):
    """Rotation matrix: scalar deg is broadcast across the axis string."""
    from scipy.spatial.transform import Rotation as Rot

    n_axes = len(axis)
    if isinstance(deg, (list, tuple, np.ndarray)):
        angles = [list(deg)]
    else:
        angles = [[float(deg)] * n_axes]
    m = Rot.from_euler(axis, angles, degrees=True).as_matrix()
    return m[0] if m.ndim == 3 else m


def _make_result(n=20, scale=1.0):
    """Video-only-style reconstruction with arbitrary attitudes."""
    rng = np.random.default_rng(7)
    res = ReconstructionResult()
    centers = rng.normal(size=(n, 3)) * 3 * scale
    for i in range(n):
        R_c2w = _rot("xyz", 0) if False else np.eye(3)
        res.cameras[f"frame_{i:06d}.jpg"] = CameraPose(
            0, f"frame_{i:06d}.jpg", centers[i], R_c2w,
            np.array([1.0, 0, 0, 0]),
            np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]]),
            np.zeros(5),
        )
    res.points3d = [
        SparsePoint3D(i, p.copy(), np.array([128, 128, 128], dtype=np.uint8))
        for i, p in enumerate(rng.normal(size=(40, 3)) * 2 + centers.mean(0))
    ]
    res.num_registered = n
    res.num_points = len(res.points3d)
    return res, centers


class TestUmeyama:
    def test_recovers_true_similarity(self):
        rng = np.random.default_rng(0)
        V = rng.normal(size=(20, 3)) * 3
        s_true, th = 17.0, np.deg2rad(35)
        R_true = _rot("y", 35)
        t_true = np.array([100.0, 50.0, 10.0])
        T = s_true * (V @ R_true.T) + t_true
        s, R, t = tsync.umeyama_similarity(V, T)
        assert s == pytest.approx(s_true, rel=1e-6)
        assert np.abs(R - R_true).max() < 1e-9
        assert np.abs(t - t_true).max() < 1e-6

    def test_not_normalized_by_n(self):
        # Regression: normalizing H by n divided the fitted scale by n.
        rng = np.random.default_rng(1)
        V = rng.normal(size=(12, 3))
        T = 9.0 * V + 4.0
        s, _R, _t = tsync.umeyama_similarity(V, T)
        assert s == pytest.approx(9.0, rel=1e-9)


class TestProjectionInvariance:
    def test_similarity_preserves_projections(self):
        """Placing the model must not move a single pixel observation."""
        res, _ = _make_result(n=15)
        for i, cam in enumerate(res.cameras.values()):
            cam.rotation = _rot("xyz", [10, -20, 30, -5, 15][i % 5])
        snap = copy.deepcopy(res)  # deep copy: apply_similarity mutates in place

        def uv(cam, X):
            Xc = cam.rotation.T @ (X - cam.position)
            p = cam.intrinsics @ Xc
            return p[:2] / p[2]

        s, R, t = 17.3, _rot("y", 35), np.array([90.0, -40.0, 25.0])
        tsync.apply_similarity_to_reconstruction(res, s, R, t)
        # Compare PRE-state (snap point, snap camera) against POST-state
        # (transformed point, transformed camera) — same physical observation.
        shift = max(
            float(np.linalg.norm(
                uv(snap.cameras[n], snap.points3d[i].position)
                - uv(res.cameras[n], res.points3d[i].position)
            ))
            for n in snap.cameras for i in (0, 5, 20, 39)
        )
        assert shift < 1e-6
        # distances scaled by exactly s
        d0 = np.linalg.norm(snap.points3d[3].position - snap.cameras["frame_000000.jpg"].position)
        d1 = np.linalg.norm(res.points3d[3].position - res.cameras["frame_000000.jpg"].position)
        assert d1 == pytest.approx(s * d0, rel=1e-9)


class TestStaleFixCollapse:
    def test_collapses_repeated_rows_keeps_first(self):
        # 30 Hz log, 6 Hz fixes: positions hold until the next fix.
        # (Positions must never REVERT — each run advances monotonically,
        # matching real GPS logs.)
        t = np.arange(12) / 30.0
        lat = np.array([51.0, 51.0] + [51.0001] * 4 + [51.0002] * 6)
        lon = np.array([7.0, 7.0] + [7.0001] * 4 + [7.0002] * 6)
        alt = np.array([100.0] * 6 + [101.0] * 6)
        t2, lat2, lon2, alt2, info = tsync._collapse_stale_fixes(t, lat, lon, alt)
        assert info["rows_in"] == 12
        assert info["fixes_out"] == 3
        assert info["stale_rows"] == 9
        assert t2[0] == pytest.approx(0.0)      # first row of each run kept
        assert t2[1] == pytest.approx(2 / 30.0)
        assert t2[2] == pytest.approx(6 / 30.0)
        assert lat2[2] == pytest.approx(51.0002)  # altitude moves with its fix

    def test_real_run_shape(self):
        """flight_to_tower_7511dc: 933 rows -> 209 fixes at ~6.7 Hz."""
        csv = Path("data/storage/flight_to_tower_7511dc/telemetry.csv")
        if not csv.is_file():
            pytest.skip("run telemetry not present")
        prior, report = tsync.telemetry_position_prior(csv)
        assert report["rows_in"] == 933
        assert report["unique_gps_fixes"] == 209
        assert report["samples_rejected"] == 0  # no false spike rejections
        span = report["span_m"]
        assert span[0] == pytest.approx(142.37, abs=0.5)
        assert span[1] == pytest.approx(41.17, abs=0.5)


class TestTemporalSync:
    def test_reports_match_and_offsets(self, tmp_path):
        qr = {
            "frames": [
                {"index": i, "kept": True, "filename": f"frame_{i:06d}.jpg",
                 "timestamp_sec": i / 30.0}
                for i in range(0, 10)
            ]
        }
        qp = tmp_path / "quality_report.json"
        qp.write_text(json.dumps(qr))
        # telemetry at 6.7 Hz covering the video span
        tel_rows = ["timestamp,latitude,longitude,altitude"]
        for i in range(70):
            tel_rows.append(f"{i/6.7:.4f},51.0,7.0,100.0")
        tp = tmp_path / "telemetry.csv"
        tp.write_text("\n".join(tel_rows) + "\n")
        rep = tsync.temporal_sync_report(qp, tp, 30.0)
        assert rep["matched_frames"] == 10
        assert rep["pass"] is True
        assert rep["median_frame_to_telemetry_dt_s"] < 0.15


class TestStemMatching:
    """Regression: COLMAP camera names carry the colmap_images extension
    (frame_000012.png) while quality-report filenames carry the selected
    extension (frame_000012.jpg).  Matching must go by STEM — the exact
    failure that produced 0 time-matched correspondences on
    flight_to_tower_7511dc.
    """

    def test_quality_report_keys_are_stems(self, tmp_path):
        qr = {"frames": [
            {"index": 0, "kept": True, "filename": "frame_000000.jpg",
             "timestamp_sec": 0.0},
            {"index": 1, "kept": True, "filename": "frame_000001.jpg",
             "timestamp_sec": 1 / 30.0},
            {"index": 2, "kept": False, "filename": "frame_000002.jpg",
             "timestamp_sec": 2 / 30.0},
        ]}
        qp = tmp_path / "quality_report.json"
        qp.write_text(json.dumps(qr))
        ts = tsync.camera_timestamps_from_quality_report(qp)
        assert set(ts) == {"frame_000000", "frame_000001"}

    def test_alignment_across_extension_mismatch(self):
        # Telemetry: a straight metric flight line, fixes every 0.1 s.
        tel_t = np.arange(10) * 0.1
        tel_pos = np.column_stack([np.linspace(0, 90, 10),
                                   np.zeros(10), np.full(10, 5.0)])
        prior = tsync.PositionPrior(
            positions=tel_pos, timestamps=tel_t,
            anchor=(51.0, 7.0, 100.0),
        )
        # Visual reconstruction: 17x-gauge-collapsed copy of the same line,
        # cameras named with .png while the quality report uses .jpg.
        R_true = _rot("y", 25)
        t_true = np.array([3.0, 1.0, -2.0])
        res = ReconstructionResult()
        vis_centers = {}
        for i in range(10):
            C = 0.06 * (R_true @ tel_pos[i]) + t_true
            vis_centers[f"frame_{i:06d}"] = C
            res.cameras[f"frame_{i:06d}.png"] = CameraPose(
                0, f"frame_{i:06d}.png", C.copy(), np.eye(3),
                np.array([1.0, 0, 0, 0]),
                np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]]),
                np.zeros(5),
            )
        res.num_registered = len(res.cameras)
        camera_ts = {f"frame_{i:06d}": i * (1 / 30.0) * 3 for i in range(10)}
        targets, report = tsync.align_reconstruction_to_telemetry(
            res, prior, camera_ts
        )
        assert "error" not in report
        assert report["cameras_matched"] == 10
        # targets are keyed by COLMAP camera names (with extension)
        assert set(targets) == {f"frame_{i:06d}.png" for i in range(10)}
        # model placed onto the telemetry line: scale ~16.7 recovered
        assert report["similarity_scale_visual_per_telemetry"] == pytest.approx(
            1 / 0.06, rel=1e-3
        )
        assert report["residual_after_alignment_m"]["median"] < 1e-3


class TestRobustFit:
    def test_outlier_segment_does_not_contaminate(self):
        """A drifted camera segment must not drag the global placement."""
        rng = np.random.default_rng(3)
        T = rng.normal(size=(30, 3)) * 20
        V = 0.1 * T  # clean 10x gauge collapse
        V[22:27] += np.array([30.0, -20.0, 10.0])  # drifted segment
        s_ls, R_ls, t_ls = tsync.umeyama_similarity(V, T)
        resid_ls = np.linalg.norm(10 * (V @ R_ls.T) + t_ls - T, axis=1)
        s_rb, R_rb, t_rb, info = tsync.umeyama_similarity_robust(V, T)
        resid_rb = np.linalg.norm(
            s_rb * (V[info["outlier_indices"] or [0]] @ R_rb.T) + t_rb - T[info["outlier_indices"] or [0]],
            axis=1,
        )
        # least-squares residual inflated by the drift; robust fit not
        assert resid_ls.max() > 5.0
        assert info["outliers"] >= 3
        assert info["inlier_residual_m"]["median"] < 0.5
        # inlier-only fit recovers the true scale tightly
        assert s_rb == pytest.approx(10.0, rel=0.05)

    def test_clean_data_no_false_rejection(self):
        rng = np.random.default_rng(5)
        T = rng.normal(size=(20, 3)) * 30
        V = 3.0 * T + 7.0  # visual = 3x telemetry -> visual-per-telemetry scale 1/3
        s, _R, _t, info = tsync.umeyama_similarity_robust(V, T)
        assert info["outliers"] == 0
        assert s == pytest.approx(1 / 3.0, rel=1e-9)


class TestPiecewiseCorrection:
    def _drifted_reconstruction(self):
        """A metric-quality trajectory with two drift bends: locally rigid,
        globally not a similarity — exactly what video-only SfM produces."""
        res = ReconstructionResult()
        camera_ts = {}
        prior_positions = []
        rng = np.random.default_rng(11)
        for i in range(24):
            t = i * (1 / 30.0) * 3  # sparse cadence, matches max_dt 0.75s
            base = np.array([15.0 * t, 0.0, 5.0])
            # drift bends: growing sideways displacement in two bands
            drift = 0.0
            if 8 <= i < 16:
                drift = (i - 8) * 3.0
            elif i >= 16:
                drift = 24.0
            C_t = base + np.array([0.0, drift, 0.0])   # telemetry truth
            C_v = C_t + rng.normal(scale=0.05, size=3)  # visual copy (metres)
            res.cameras[f"frame_{i:06d}.png"] = CameraPose(
                0, f"frame_{i:06d}.png", C_v.copy(), np.eye(3),
                np.array([1.0, 0, 0, 0]),
                np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]]),
                np.zeros(5),
            )
            camera_ts[f"frame_{i:06d}"] = t
            prior_positions.append(C_t)
        res.num_registered = len(res.cameras)
        tel_t = np.arange(0, 24 * 0.1, 0.1)
        n = len(prior_positions)
        tel_pos = np.array([
            np.array([np.interp(tt, [camera_ts[f"frame_{i:06d}"] for i in range(n)],
                                [p[0] for p in prior_positions]),
                      np.interp(tt, [camera_ts[f"frame_{i:06d}"] for i in range(n)],
                                [p[1] for p in prior_positions]),
                      np.interp(tt, [camera_ts[f"frame_{i:06d}"] for i in range(n)],
                                [p[2] for p in prior_positions])])
            for tt in tel_t
        ])
        prior = tsync.PositionPrior(
            positions=tel_pos, timestamps=tel_t, anchor=(51.0, 7.0, 100.0)
        )
        return res, prior, camera_ts

    def test_piecewise_recovers_drifted_trajectory(self):
        res, prior, camera_ts = self._drifted_reconstruction()
        targets, report = tsync.align_reconstruction_to_telemetry(
            res, prior, camera_ts
        )
        # the global robust fit must have failed (drift bends) ...
        assert report["placement_mode"] == "piecewise_rigid"
        assert report["global_robust_fit"]["outlier_fraction"] > 0.25
        # ... and the piecewise correction must place ALL matched cameras
        # near their telemetry positions (each window recovers its own rigid
        # map; scale sanity rejects only genuinely unpinnable windows).
        errs = [
            float(np.linalg.norm(
                np.asarray(res.cameras[f"frame_{i:06d}.png"].position)
                - prior.positions[np.clip(
                    np.searchsorted(prior.timestamps, camera_ts[f"frame_{i:06d}"]),
                    0, len(prior.timestamps) - 1)]
            ))
            for i in range(24)
        ]
        assert report["piecewise"]["applied"]["cameras_transformed"] > 12
        assert np.median(errs) < 2.0
        assert set(targets) == {f"frame_{i:06d}.png" for i in range(24)}


class TestResample:
    def test_uniform_grid(self):
        t = np.linspace(0, 10, 100)
        P = np.column_stack([t, np.zeros(100), np.ones(100)])
        Pg, tg = tsync.resample_path(P, t, 0.5)
        assert len(tg) == 21
        assert np.allclose(np.diff(tg), 0.5)
        assert np.allclose(Pg[:, 0], tg)


class TestCompositeGate:
    def test_all_components_required(self):
        assert tsync.composite_trajectory_gate(
            {"pass": True}, {"pass": True}, {"pass": True}
        )["pass"] is True
        assert tsync.composite_trajectory_gate(
            {"pass": True}, {"pass": False}, {"pass": True}
        )["pass"] is False
        # scale can never fail it
        shape = {
            "pass": True,
            "scale_sanity": {"verdict": "UNIT WARNING", "hard_failure": False},
        }
        g = tsync.composite_trajectory_gate({"pass": True}, shape, {"pass": True})
        assert g["pass"] is True
        assert g["scale_verdict"] == "UNIT WARNING"


class TestContinuity:
    def test_detects_impossible_jump(self):
        t = np.arange(10, dtype=float)
        C = np.column_stack([np.linspace(0, 9, 10), np.zeros(10), np.ones(10)])
        rep = tsync.pose_jump_report(C, t)
        assert rep["pass"] is True and rep["jump_count"] == 0
        C[5] += [50.0, 0, 0]  # teleport
        rep2 = tsync.pose_jump_report(C, t)
        assert rep2["jump_count"] >= 1 and rep2["pass"] is False

    def test_smooth_flight_passes(self):
        # 18 m/s over 30 s — the real flight's speed
        t = np.linspace(0, 30, 200)
        C = np.column_stack([18.0 * t * 0.9, 8.0 * t * 0.1, 226.0 + 0.1 * np.sin(t)])
        rep = tsync.pose_jump_report(C, t)
        assert rep["pass"] is True


def _write_frame_log_csv(
    path: Path,
    n_rows: int = 300,
    *,
    with_frame_id: bool = True,
    fps: float = 30.0,
) -> None:
    """airport1-style telemetry: one GPS row per video frame, no timestamps."""
    rows = ["frame_id,latitude,longitude,altitude"]
    for i in range(n_rows):
        fid = f"{i}" if with_frame_id else ""
        # ~0.1 m/frame at 30 fps -> 3 m/s flight
        rows.append(f"{fid},51.0 + 0,7.0,100.0") if False else rows.append(
            f"{fid},{51.0 + i * 9e-8:.10f},{7.0 + i * 1.2e-7:.10f},{100.0 + i * 0.001:.3f}"
        )
    path.write_text("\n".join(rows) + "\n")


class TestFrameNumberTimeBase:
    """airport1 failure class: telemetry has GPS + frame_id but no timestamps.

    The pipeline must derive the time base from frame_number at the video fps
    (validated against video duration), never fabricate a row-index clock.
    """

    def test_derivation_fills_timestamps(self, tmp_path):
        from app.services.telemetry import derive_timestamps_from_frame_numbers
        from app.services.telemetry_schema import parse_telemetry_csv, dataset_to_samples

        csv = tmp_path / "telemetry.csv"
        _write_frame_log_csv(csv, 90)  # 3 s at 30 fps
        samples = dataset_to_samples(parse_telemetry_csv(csv))
        assert all(s.timestamp_sec is None for s in samples)
        out, note = derive_timestamps_from_frame_numbers(
            samples, 30.0, video_duration_sec=3.0)
        assert all(s.timestamp_sec is not None for s in out)
        assert "30 fps" in note
        assert out[10].timestamp_sec == pytest.approx(10 / 30.0)
        assert out[-1].timestamp_sec == pytest.approx(89 / 30.0)

    def test_derivation_refused_when_span_exceeds_video(self, tmp_path):
        from app.services.telemetry import derive_timestamps_from_frame_numbers
        from app.services.telemetry_schema import parse_telemetry_csv, dataset_to_samples

        csv = tmp_path / "telemetry.csv"
        _write_frame_log_csv(csv, 900)  # 30 s at 30 fps
        samples = dataset_to_samples(parse_telemetry_csv(csv))
        # Video only 10 s long -> frame ids cannot be video frames.
        out, note = derive_timestamps_from_frame_numbers(
            samples, 30.0, video_duration_sec=10.0)
        assert out is samples  # refused: input returned untouched
        assert "not usable" in note
        assert all(s.timestamp_sec is None for s in out)

    def test_prior_uses_frame_number_time_base(self, tmp_path):
        csv = tmp_path / "telemetry.csv"
        _write_frame_log_csv(csv, 300)  # 10 s at 30 fps
        prior, report = tsync.telemetry_position_prior(
            csv, video_fps=30.0, video_duration_sec=10.0)
        assert report["time_base"] == "frame_number_at_video_fps"
        assert report["duration_s"] == pytest.approx(299 / 30.0, abs=0.01)
        assert report["duration_s"] < 11.0  # NOT 299 (the old row-index bug)
        assert prior.n >= 3

    def test_prior_honest_failure_without_time_base(self, tmp_path):
        from app.services.telemetry import TelemetryError

        csv = tmp_path / "telemetry.csv"
        _write_frame_log_csv(csv, 300, with_frame_id=False)
        with pytest.raises(TelemetryError, match="no usable time base"):
            tsync.telemetry_position_prior(csv, video_fps=30.0, video_duration_sec=10.0)

    def test_temporal_sync_passes_with_derived_time_base(self, tmp_path):
        # Full airport1 shape: frames at PTS 0..9.9s, telemetry rows 0..299
        # at 30 fps -> every frame must pair with its own moment.
        qp = tmp_path / "quality_report.json"
        qp.write_text(json.dumps({"frames": [
            {"index": i, "kept": True, "filename": f"frame_{i:06d}.jpg",
             "timestamp_sec": i / 30.0}
            for i in range(0, 300, 10)
        ]}))
        tp = tmp_path / "telemetry.csv"
        _write_frame_log_csv(tp, 300)
        rep = tsync.temporal_sync_report(
            qp, tp, 30.0, video_duration_sec=10.0)
        assert rep["pass"] is True
        assert rep["matched_frames"] == 30
        assert "frame_number" in rep.get("timestamp_provenance", "")

    def test_temporal_sync_honest_failure_without_time_base(self, tmp_path):
        qp = tmp_path / "quality_report.json"
        qp.write_text(json.dumps({"frames": [
            {"index": i, "kept": True, "filename": f"frame_{i:06d}.jpg",
             "timestamp_sec": i / 30.0}
            for i in range(0, 100, 10)
        ]}))
        tp = tmp_path / "telemetry.csv"
        _write_frame_log_csv(tp, 100, with_frame_id=False)
        rep = tsync.temporal_sync_report(qp, tp, 30.0, video_duration_sec=3.3)
        assert rep["pass"] is False
        assert "no usable timestamps" in rep["error"]
