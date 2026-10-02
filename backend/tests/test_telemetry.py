"""Regression tests for the three-mode telemetry model.

Covers the user-facing contract: a manually uploaded MP4 with no GPS is a
normal VIDEO_ONLY run (not an ingestion failure); external telemetry CSVs are
validated and synchronized against frame timestamps; georeferenced artifacts
are produced only when the telemetry track is sufficient. Reconstruction
itself is untouched — every test here drives only ingestion, synchronization,
and georef-stage logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.pipeline_orchestrator import (
    PipelineRequest,
    StageState,
    _stage_georef,
)
from app.services.telemetry import (
    MIN_GEOFERENCE_SAMPLES,
    TelemetryError,
    TelemetryMode,
    load_external_telemetry,
    resolve_mode,
    synchronize,
)

CSV_HEADER = "timestamp,latitude,longitude,altitude"


def _write_csv(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    lines = [CSV_HEADER] + [f"{t},{lat},{lon},{alt}" for t, lat, lon, alt in rows]
    path.write_text("\n".join(lines) + "\n")


def _trajectory_rows(
    n: int = 5, t0: float = 0.0, dt: float = 1.0
) -> list[tuple[float, float, float, float]]:
    """A small drifting trajectory near Shitan, TW (valid WGS84 values)."""
    return [
        (t0 + i * dt, 24.5300 + i * 0.0001, 120.9300 + i * 0.0001, 350.0 + i)
        for i in range(n)
    ]


def _workspace_with_frames(
    tmp_path: Path,
    frame_times: list[float],
) -> Path:
    """Workspace with a kept-frames quality report and minimal poses.json."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    frames = [
        {
            "index": i,
            "timestamp_sec": ts,
            "filename": f"frame_{i:06d}.jpg",
            "kept": True,
        }
        for i, ts in enumerate(frame_times)
    ]
    (workspace / "quality_report.json").write_text(
        json.dumps({"frames": frames, "selected_count": len(frames)})
    )
    (workspace / "poses.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_id": f"frame_{i:06d}", "t": [float(i), 0.0, 0.0]}
                    for i in range(len(frame_times))
                ]
            }
        )
    )
    return workspace


def _run_georef(workspace: Path, request: PipelineRequest | None = None) -> StageState:
    state = StageState(name="georeferencing", status="running")
    _stage_georef("job1", request or PipelineRequest(), workspace, state)
    return state


# ---------------------------------------------------------------------------
# 1. video with no telemetry (VIDEO_ONLY is a normal state, not a failure)
# ---------------------------------------------------------------------------


class TestVideoOnly:
    def test_mode_resolves_to_video_only(self) -> None:
        assert resolve_mode(embedded_gps=None, telemetry_csv_path=None) is TelemetryMode.VIDEO_ONLY

    def test_sync_with_no_samples_is_honest(self) -> None:
        report = synchronize([("frame_000000", 0.0), ("frame_000001", 1.0)], [])
        assert report.mode == TelemetryMode.VIDEO_ONLY.value
        assert report.gps_available is False
        assert report.unmatched_frames == 2
        assert report.has_sufficient_track is False

    def test_georef_stage_completes_without_artifacts(self, tmp_path: Path) -> None:
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0, 2.0])
        state = _run_georef(workspace)
        assert state.status == "completed"  # expected state — NOT a failure
        assert "no GPS telemetry available" in state.detail["note"]
        assert state.detail["telemetry_mode"] == TelemetryMode.VIDEO_ONLY.value
        assert not (workspace / "georef").exists()


# ---------------------------------------------------------------------------
# 2. valid telemetry CSV synchronizes and enables georeferencing
# ---------------------------------------------------------------------------


class TestValidTelemetry:
    def test_load_valid_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "telemetry.csv"
        _write_csv(csv_path, _trajectory_rows(n=5))
        samples = load_external_telemetry(csv_path)
        assert len(samples) == 5
        assert samples[0].timestamp_sec == 0.0
        assert samples[0].latitude == pytest.approx(24.53)
        assert samples[0].altitude_m == pytest.approx(350.0)

    def test_mode_resolves_to_external(self, tmp_path: Path) -> None:
        assert (
            resolve_mode(embedded_gps=None, telemetry_csv_path=tmp_path / "t.csv")
            is TelemetryMode.VIDEO_WITH_EXTERNAL_TELEMETRY
        )

    def test_sync_matches_frames_and_reports_offset(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "telemetry.csv"
        rows = _trajectory_rows(n=4, dt=1.0)
        _write_csv(csv_path, [(t + 0.25, lat, lon, alt) for t, lat, lon, alt in rows])
        samples = load_external_telemetry(csv_path)
        frame_times = [(f"frame_{i:06d}", float(i)) for i in range(4)]
        report = synchronize(frame_times, samples)
        assert report.mode == TelemetryMode.VIDEO_WITH_EXTERNAL_TELEMETRY.value
        assert report.telemetry_samples == 4
        assert report.matched_frames == 4
        assert report.unmatched_frames == 0
        assert report.timestamp_offset_sec == pytest.approx(0.25)
        assert report.gps_available is True
        assert report.has_sufficient_track is True

    def test_georef_stage_writes_artifacts(self, tmp_path: Path) -> None:
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0, 2.0, 3.0, 4.0])
        csv_path = workspace / "telemetry.csv"
        _write_csv(csv_path, _trajectory_rows(n=5, dt=1.0))
        state = _run_georef(workspace, PipelineRequest(telemetry_csv="telemetry.csv"))
        assert state.status == "completed"
        assert state.detail["telemetry_mode"] == TelemetryMode.VIDEO_WITH_EXTERNAL_TELEMETRY.value
        sync = state.detail["sync"]
        assert sync["matched_frames"] == 5
        assert sync["sufficient_for_georeferencing"] is True
        assert (workspace / "georef" / "gps_track.csv").exists()
        assert (workspace / "georef" / "gps_report.json").exists()


# ---------------------------------------------------------------------------
# 3. malformed telemetry CSV
# ---------------------------------------------------------------------------


class TestMalformedCsv:
    def test_missing_header_now_detected_not_rejected(self, tmp_path: Path) -> None:
        """Universal ingestion: 'time,lat,lng' is a VALID variant (renamed
        canonical columns, timestamp-free). Positions work; temporal sync is
        reported unavailable — the old hard rejection is gone (spec §2)."""
        csv_path = tmp_path / "ok.csv"
        csv_path.write_text("time,lat,lng\n0,24.5,120.9\n1,24.5001,120.9001\n")
        samples = load_external_telemetry(csv_path)
        assert len(samples) == 2
        assert samples[0].latitude == pytest.approx(24.5)
        # 'time' with 0,1,... values IS a usable relative timeline (spec §13:
        # never reject merely because it is not Unix epoch) — detected and
        # preserved as relative, not treated as absent.
        assert samples[0].timestamp_sec == pytest.approx(0.0)

    def test_unmappable_csv_rejected_with_diagnostic(self, tmp_path: Path) -> None:
        """Genuine failure: no latitude/longitude columns anywhere — the
        error describes what WAS detected instead of quoting the old header."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
        with pytest.raises(TelemetryError, match="no GPS position columns"):
            load_external_telemetry(csv_path)

    def test_missing_header_error_names_old_error(self, tmp_path: Path) -> None:
        """The forbidden error message must never appear again."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(TelemetryError) as ei:
            load_external_telemetry(csv_path)
        assert "missing required column" not in str(ei.value)
        assert "expected header" not in str(ei.value)

    def test_non_numeric_timestamp_becomes_unavailable_not_rejected(self, tmp_path: Path) -> None:
        """A timestamp column whose values are strings: temporal sync is
        unavailable; GPS rows continue (never invent timestamps, spec §2/§13)."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(f"{CSV_HEADER}\nabc,24.5,120.9,350\n")
        samples = load_external_telemetry(csv_path)
        assert samples[0].timestamp_sec is None
        assert samples[0].latitude == pytest.approx(24.5)

    def test_non_numeric_latitude_row_is_invalid_not_fatal(self, tmp_path: Path) -> None:
        """A row with a non-numeric latitude: that ROW is invalid; the file
        still loads (row-level validation, spec §15)."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(f"{CSV_HEADER}\n0,north,120.9,350\n1,24.5,120.9,351\n")
        samples = load_external_telemetry(csv_path)
        assert len(samples) == 2
        report = synchronize([(f"frame_{i:06d}", float(i)) for i in range(2)], samples)
        assert report.invalid_samples_dropped == 1
        assert report.gps_available is True

    def test_empty_file_rejected(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        with pytest.raises(TelemetryError, match="empty"):
            load_external_telemetry(csv_path)

    def test_georef_stage_survives_malformed_csv(self, tmp_path: Path) -> None:
        """Malformed telemetry must not fail the run or produce artifacts."""
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0, 2.0])
        (workspace / "telemetry.csv").write_text("garbage\nnot,a,csv\n")
        state = _run_georef(workspace, PipelineRequest(telemetry_csv="telemetry.csv"))
        assert state.status == "completed"
        assert "unusable" in state.detail["note"]
        assert state.detail["telemetry_mode"] == TelemetryMode.VIDEO_ONLY.value
        assert not (workspace / "georef").exists()


# ---------------------------------------------------------------------------
# 4. timestamp mismatch (telemetry never overlaps the video timeline)
# ---------------------------------------------------------------------------


class TestTimestampMismatch:
    def test_no_overlap_reports_no_overlap(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "telemetry.csv"
        # Telemetry recorded an hour later — zero overlap with 0-4s frames.
        _write_csv(csv_path, _trajectory_rows(n=5, t0=3600.0, dt=1.0))
        samples = load_external_telemetry(csv_path)
        frame_times = [(f"frame_{i:06d}", float(i)) for i in range(4)]
        report = synchronize(frame_times, samples, max_offset_sec=1.0)
        assert report.matched_frames == 0
        assert report.unmatched_frames == 4
        assert report.telemetry_quality == "no_overlap"
        assert report.has_sufficient_track is False
        assert any("never overlap" in note for note in report.notes)

    def test_georef_skips_on_mismatch(self, tmp_path: Path) -> None:
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0, 2.0, 3.0])
        csv_path = workspace / "telemetry.csv"
        _write_csv(csv_path, _trajectory_rows(n=5, t0=3600.0, dt=1.0))
        state = _run_georef(workspace, PipelineRequest(telemetry_csv="telemetry.csv"))
        assert state.status == "completed"
        assert "insufficient" in state.detail["note"]
        assert not (workspace / "georef").exists()

    def test_partial_overlap_counts_unmatched(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "telemetry.csv"
        _write_csv(csv_path, _trajectory_rows(n=2, t0=3.5, dt=1.0))
        samples = load_external_telemetry(csv_path)
        frame_times = [(f"frame_{i:06d}", float(i)) for i in range(5)]
        report = synchronize(frame_times, samples, max_offset_sec=0.5)
        assert report.matched_frames == 2
        assert report.unmatched_frames == 3


# ---------------------------------------------------------------------------
# 5. insufficient telemetry samples
# ---------------------------------------------------------------------------


class TestInsufficientSamples:
    def test_below_threshold_has_no_sufficient_track(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "telemetry.csv"
        _write_csv(csv_path, _trajectory_rows(n=MIN_GEOFERENCE_SAMPLES - 1))
        samples = load_external_telemetry(csv_path)
        frame_times = [(f"frame_{i:06d}", float(i)) for i in range(2)]
        report = synchronize(frame_times, samples)
        assert report.gps_available is True
        assert report.telemetry_quality == "insufficient"
        assert report.has_sufficient_track is False

    def test_georef_skips_despite_valid_gps_values(self, tmp_path: Path) -> None:
        """GPS values exist — georeferencing still must not claim success."""
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0])
        csv_path = workspace / "telemetry.csv"
        _write_csv(csv_path, _trajectory_rows(n=2))
        state = _run_georef(workspace, PipelineRequest(telemetry_csv="telemetry.csv"))
        assert state.status == "completed"
        assert "insufficient for georeferencing" in state.detail["note"]
        assert not (workspace / "georef").exists()


# ---------------------------------------------------------------------------
# 6. missing GPS values (empty / null-island / out-of-range)
# ---------------------------------------------------------------------------


class TestMissingGpsValues:
    def test_empty_gps_cells_dropped_and_counted(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "telemetry.csv"
        rows = _trajectory_rows(n=3)
        lines = [CSV_HEADER]
        lines += [f"{t},{lat},{lon},{alt}" for t, lat, lon, alt in rows]
        lines.append("9.0,,,")  # no GPS at all
        csv_path.write_text("\n".join(lines) + "\n")
        samples = load_external_telemetry(csv_path)
        assert len(samples) == 4  # row parsed
        assert samples[3].latitude is None
        report = synchronize([(f"frame_{i:06d}", float(i)) for i in range(4)], samples)
        assert report.invalid_samples_dropped == 1
        assert report.matched_frames == 4  # frame still matched in time
        assert report.telemetry_samples == 4
        assert report.gps_available is True

    def test_null_island_dropped(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "telemetry.csv"
        lines = [CSV_HEADER, "0.0,0.0,0.0,0.0"] + [
            f"{t},{lat},{lon},{alt}" for t, lat, lon, alt in _trajectory_rows(n=3, t0=1.0)
        ]
        csv_path.write_text("\n".join(lines) + "\n")
        samples = load_external_telemetry(csv_path)
        report = synchronize([(f"frame_{i:06d}", float(i)) for i in range(4)], samples)
        assert report.invalid_samples_dropped == 1

    def test_out_of_range_latitude_dropped(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "telemetry.csv"
        lines = [CSV_HEADER, "0.0,999.0,120.9,350.0"] + [
            f"{t},{lat},{lon},{alt}" for t, lat, lon, alt in _trajectory_rows(n=3, t0=1.0)
        ]
        csv_path.write_text("\n".join(lines) + "\n")
        samples = load_external_telemetry(csv_path)
        report = synchronize([(f"frame_{i:06d}", float(i)) for i in range(4)], samples)
        assert report.invalid_samples_dropped == 1

    def test_mixed_valid_and_gpsless_rows_never_feed_none_into_track(self, tmp_path: Path) -> None:
        """A frame matched (by time) to a GPS-less sample stays unmatched, so
        the sufficiency gate counts only frames that carry a usable fix and
        the track writer never sees a None coordinate (regression: this used
        to crash the stage with round(None) after passing the gate)."""
        workspace = _workspace_with_frames(tmp_path, [0.0, 2.0, 4.0, 6.0])
        (workspace / "telemetry.csv").write_text(
            "\n".join(
                [CSV_HEADER]
                + [
                    "0.0,,,,",
                    f"2.0,{24.53 + 0.0001:.6f},{120.93 + 0.0001:.6f},351",
                    f"4.0,{24.53 + 0.0002:.6f},{120.93 + 0.0002:.6f},352",
                    f"6.0,{24.53 + 0.0003:.6f},{120.93 + 0.0003:.6f},353",
                ]
            )
            + "\n"
        )
        state = _run_georef(workspace, PipelineRequest(telemetry_csv="telemetry.csv"))
        sync = state.detail["sync"]
        assert sync["matched_frames"] == 3
        assert sync["unmatched_frames"] == 1
        assert sync["invalid_samples_dropped"] == 1
        assert sync["sufficient_for_georeferencing"] is True
        assert state.status == "completed"
        assert (workspace / "georef" / "gps_track.csv").exists()

    def test_all_gps_missing_means_video_only_georef_skip(self, tmp_path: Path) -> None:
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0, 2.0, 3.0])
        csv_path = workspace / "telemetry.csv"
        csv_path.write_text(
            f"{CSV_HEADER}\n" + "\n".join(f"{i},,," for i in range(4)) + "\n"
        )
        state = _run_georef(workspace, PipelineRequest(telemetry_csv="telemetry.csv"))
        assert state.status == "completed"
        assert "no telemetry sample carried valid GPS" in state.detail["note"]
        assert not (workspace / "georef").exists()


# ---------------------------------------------------------------------------
# 7. video-only run never creates fake georeferencing artifacts
# ---------------------------------------------------------------------------


class TestNoFakeArtifacts:
    def test_video_only_leaves_no_georef_dir(self, tmp_path: Path) -> None:
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0, 2.0])
        _run_georef(workspace)
        assert not (workspace / "georef").exists()
        assert not (workspace / "georef" / "gps_track.csv").exists()
        assert not (workspace / "georef" / "gps_report.json").exists()

    def test_missing_telemetry_file_is_video_only_not_error(self, tmp_path: Path) -> None:
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0, 2.0])
        state = _run_georef(workspace, PipelineRequest(telemetry_csv="nope.csv"))
        assert state.status == "completed"
        assert "not found" in state.detail["note"]
        assert "video-only" in state.detail["note"]
        assert not (workspace / "georef").exists()

    def test_single_embedded_anchor_writes_no_track_or_alignment(
        self, tmp_path: Path
    ) -> None:
        """One project-level GPS anchor cannot anchor a trajectory: the stage
        must not emit a fake gps_track.csv or an alignment."""
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0, 2.0])
        state = _run_georef(
            workspace, PipelineRequest(gps={"lat": 24.53, "lon": 120.93, "alt": 350.0})
        )
        assert state.status == "completed"
        assert not (workspace / "georef" / "gps_track.csv").exists()
        assert state.detail["alignment"] is None


# ---------------------------------------------------------------------------
# 8. telemetry-enabled run reports georeferencing only when conditions hold
# ---------------------------------------------------------------------------


class TestConditionalGeoreferencing:
    def test_sync_dict_present_only_for_external_telemetry(self, tmp_path: Path) -> None:
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0, 2.0, 3.0, 4.0])
        csv_path = workspace / "telemetry.csv"
        _write_csv(csv_path, _trajectory_rows(n=5))
        state = _run_georef(workspace, PipelineRequest(telemetry_csv="telemetry.csv"))
        assert state.detail["sync"] is not None
        assert state.detail["sync"]["sufficient_for_georeferencing"] is True

        video_only_ws = _workspace_with_frames(tmp_path / "vo", [0.0, 1.0, 2.0])
        vo_state = _run_georef(video_only_ws)
        assert vo_state.detail["sync"] is None
        assert vo_state.detail["telemetry_mode"] == TelemetryMode.VIDEO_ONLY.value

    def test_sufficiency_flag_matches_artifacts(self, tmp_path: Path) -> None:
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0, 2.0, 3.0, 4.0])
        csv_path = workspace / "telemetry.csv"
        _write_csv(csv_path, _trajectory_rows(n=5))
        state = _run_georef(workspace, PipelineRequest(telemetry_csv="telemetry.csv"))
        sync = state.detail["sync"]
        assert sync["sufficient_for_georeferencing"] is True
        assert (workspace / "georef" / "gps_report.json").exists()
        gps_report = json.loads((workspace / "georef" / "gps_report.json").read_text())
        assert gps_report["telemetry_mode"] == TelemetryMode.VIDEO_WITH_EXTERNAL_TELEMETRY.value

    def test_embedded_only_mode_without_georef_claim(self, tmp_path: Path) -> None:
        """Embedded telemetry on poses (e.g. EXIF-rich imagery): valid GPS
        values exist, but with <3 fixes the stage must not claim georef."""
        workspace = _workspace_with_frames(tmp_path, [0.0, 1.0])
        poses = json.loads((workspace / "poses.json").read_text())
        for frame in poses["frames"]:
            frame["gps"] = {"lat": 24.53, "lon": 120.93, "alt": 350.0}
        (workspace / "poses.json").write_text(json.dumps(poses))
        state = _run_georef(workspace)
        assert state.status == "completed"
        # No track written: 2 identical fixes cannot verify a trajectory.
        assert not (workspace / "georef" / "gps_track.csv").exists()
        assert state.detail["alignment"] is None
