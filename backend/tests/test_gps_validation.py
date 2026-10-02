"""Regression tests for GPS telemetry handling & missing GPS policy in STRATA.

Ensures that missing GPS telemetry returns None / 'GPS unavailable' and NEVER
substitutes (0.0, 0.0, 0.0) or fake coordinates as real georeferencing.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from app.services.metadata_extraction import extract_metadata
from app.services.pipeline_orchestrator import _stage_georef, StageState, PipelineRequest
from app.services import run_service


def test_missing_gps_returns_none(tmp_path: Path):
    """When a video file has no GPS EXIF tags, metadata_extraction returns None for GPS coords."""
    test_avi = Path("backend/data/storage/04b419e5ebb94782aa161b9eb9171f51/test_drone.avi")
    if not test_avi.exists():
        pytest.skip("Test video backend/data/storage/.../test_drone.avi not found")

    metadata = extract_metadata(test_avi)
    assert metadata.gps_lat is None
    assert metadata.gps_lon is None
    assert metadata.gps_alt is None


def test_georef_stage_skips_without_fake_origin(tmp_path: Path):
    """When no GPS telemetry is present, the georeferencing stage skips gracefully without generating fake 0,0 ENU files."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    poses = {"frames": [{"t": [0, 0, 0]}, {"t": [1, 0, 0]}]}
    (workspace / "poses.json").write_text(str(poses).replace("'", '"'))

    state = StageState(name="georeferencing", status="running")
    req = PipelineRequest(gps=None)

    _stage_georef("job1", req, workspace, state)

    assert state.status == "completed"
    assert state.count == 0
    assert "no GPS telemetry available" in state.detail["note"]
    assert not (workspace / "georef" / "gps_track.csv").exists()


def test_georef_autodiscovers_workspace_telemetry_csv(tmp_path: Path, monkeypatch):
    """Parity with the sparse stage: telemetry.csv in the workspace engages
    georeferencing even when the request does not repeat it.

    The UI's retry/resume re-enqueues with an empty request; before this fix
    the georef stage only engaged telemetry when ``request.telemetry_csv`` was
    set, so a resume dropped external telemetry and wrote an honest-but-wrong
    VIDEO_ONLY detail while sparse ran telemetry-placed (measured on
    airport1_test_99ee3c).
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "poses.json").write_text(json.dumps({
        "frames": [{"frame_id": "frame_000000", "t": [0, 0, 0], "K": [[100, 0, 0], [0, 100, 0], [0, 0, 1]], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}],
    }))
    # Untimed one-per-frame telemetry with valid GPS: the airport1 shape.
    (workspace / "telemetry.csv").write_text(
        "frame_id,latitude,longitude,altitude\n"
        "0,52.5173,13.3936,221.3\n"
    )
    (workspace / "source.json").write_text(json.dumps({"fps": 30.0, "duration_sec": 0.033}))
    # quality_report is absent -> _frame_timestamps returns [] -> sync has no
    # frames to match; the mode must still be EXTERNAL, never VIDEO_ONLY.

    monkeypatch.setattr(
        "app.services.pipeline_orchestrator._frame_timestamps", lambda _ws: [])

    state = StageState(name="georeferencing", status="running")
    req = PipelineRequest(gps=None)  # empty request: the resume/retry shape

    _stage_georef("job1", req, workspace, state)

    detail = state.detail
    assert detail.get("telemetry_mode") == "VIDEO_WITH_EXTERNAL_TELEMETRY", detail


def test_georef_video_only_when_no_telemetry_anywhere(tmp_path: Path):
    """No telemetry file + no request key => honest VIDEO_ONLY (unchanged)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "poses.json").write_text(json.dumps({
        "frames": [{"frame_id": "frame_000000", "t": [0, 0, 0]}]}
    ))

    state = StageState(name="georeferencing", status="running")
    _stage_georef("job1", PipelineRequest(gps=None), workspace, state)

    assert state.detail.get("telemetry_mode") == "VIDEO_ONLY"


def test_run_summary_reports_missing_gps_honestly(tmp_path: Path, monkeypatch):
    """A run with no GPS records NO GPS count — never an invented 0,0 fix."""
    monkeypatch.setattr(run_service.settings.storage, "runs_dir", str(tmp_path))
    monkeypatch.setattr(run_service.settings.storage, "base_path", str(tmp_path / "storage"))

    run_dir = tmp_path / "run_nogps"
    run_dir.mkdir()
    manifest = {
        "run_id": "run_nogps",
        "dataset": "run_nogps",
        "mission": "No GPS Mission",
        "status": "completed",
        "stages": {},
    }
    (run_dir / "manifest.json").write_text(str(manifest).replace("'", '"'))

    summary = run_service.get_run("run_nogps")
    # Absent, not zero: a fabricated coordinate would read as a real fix.
    assert not summary.get("gps_points")
    assert "gps_report" not in summary.get("artifacts", [])
