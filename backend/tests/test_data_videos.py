"""Tests for the by-name data-folder video mission flow.

Covers ``app/services/data_video_service.py`` and ``app/routes/data_videos.py``:
name resolution (no traversal), listing, and mission start — the project record
is created and the pipeline is enqueued on the durable job queue with the
caller's tuning fields intact (the ``_pipeline_executor`` contract).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import select

import app.services.data_video_service as data_video_service
from app.config.settings import settings
from app.db.models import Project, QueueJob
from app.services.data_video_service import list_data_videos, resolve_data_video
from app.services.job_queue import _request_from_payload


def _write_real_video(path, frames: int = 6, size: tuple[int, int] = (640, 480)) -> None:
    """A real readable MJPG AVI — validate_all/extract_metadata run on it."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, size)
    assert writer.isOpened()
    rng = np.random.default_rng(7)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), (i * 30) % 255, dtype=np.uint8)
        frame[:, :, 0] = rng.integers(0, 255, (size[1], size[0]), dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# resolve_data_video — path safety
# ---------------------------------------------------------------------------


class TestResolveDataVideo:
    def test_valid_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
        _write_real_video(tmp_path / "london.mp4")
        assert resolve_data_video("london.mp4") == (tmp_path / "london.mp4").resolve()

    def test_rejects_path_traversal(self):
        for bad in ("../secret.mp4", "sub/dir/london.mp4", "..", "."):
            with pytest.raises(FileNotFoundError):
                resolve_data_video(bad)

    def test_rejects_wrong_extension(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
        (tmp_path / "notes.txt").write_text("not a video")
        with pytest.raises(FileNotFoundError):
            resolve_data_video("notes.txt")

    def test_missing_video_lists_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="available: none"):
            resolve_data_video("ghost.mp4")


# ---------------------------------------------------------------------------
# list_data_videos — header metadata
# ---------------------------------------------------------------------------


class TestListDataVideos:
    def test_lists_videos_with_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
        _write_real_video(tmp_path / "base.mp4")
        (tmp_path / "london.mp4").write_bytes(b"junk-not-a-video")  # header read fails, still listed
        (tmp_path / "readme.md").write_text("not a video")

        videos = list_data_videos()
        names = [v["name"] for v in videos]
        assert names == ["base.mp4", "london.mp4"]  # sorted, extension-filtered
        base = videos[0]
        assert base["size_bytes"] > 0
        assert base["width"] == 640 and base["height"] == 480
        assert base["fps"] == 12.0
        assert base["duration_sec"] == 0.5  # 6 frames @ 12 fps
        assert "width" not in videos[1]  # unreadable header → metadata omitted, still listed
        # The mission form requires a GPS source, so the picker has to report
        # one: video-only here, hence none.
        assert base["gps_sources"] == [] and videos[1]["gps_sources"] == []

    def test_reports_the_dataset_gps_source(self, tmp_path, monkeypatch):
        """A dataset log beside the video is what makes the run startable."""
        monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
        dataset = tmp_path / "airport1"
        dataset.mkdir()
        _write_real_video(dataset / "video.mp4")
        assert list_data_videos()[0]["gps_sources"] == []
        (dataset / "video.SRT").write_text("1\n00:00:00,000 --> 00:00:00,033\nGPS(1,2,3)\n")
        assert list_data_videos()[0]["gps_sources"] == ["video.SRT"]
        (dataset / "movingdrone_telemetry.csv").write_text("timestamp,lat,lon,alt\n")
        assert list_data_videos()[0]["gps_sources"] == ["movingdrone_telemetry.csv", "video.SRT"]

    def test_run_store_copies_are_not_offered(self, tmp_path, monkeypatch):
        """The pipeline's own workspace copies must never appear as footage.

        Every run leaves its source video under ``data/storage/<run_id>/``; the
        live picker listed seven duplicate ``london.mp4`` entries and orphaned
        scratch clips that way. The store is output — the picker shows inputs.
        """
        # The store sits BOTH inside data/ (the repo-root launch convention)
        # and at settings.storage.base_path (the backend/ launch convention) —
        # the live backend was writing to the latter while the picker listed
        # the former.
        other_store = tmp_path.parent / "backend_store"
        monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
        monkeypatch.setattr(settings.storage, "base_path", str(other_store))
        _write_real_video(tmp_path / "london.mp4")
        for base in (tmp_path / "storage", other_store):
            store = base / "data_run_20260928_000000_abcdef"
            store.mkdir(parents=True)
            _write_real_video(store / "london.mp4")
        nested = tmp_path / "storage" / "canonical_run" / "base.mp4"
        nested.parent.mkdir(parents=True)
        nested.write_bytes(b"junk")

        assert [v["name"] for v in list_data_videos()] == ["london.mp4"]
        # The resolver applies the same rule, so the offer and the accept agree.
        with pytest.raises(FileNotFoundError, match="run store"):
            resolve_data_video(
                "storage/data_run_20260928_000000_abcdef/london.mp4"
            )


# ---------------------------------------------------------------------------
# Mission start — project + queue payload contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_data_video_mission_enqueues_pipeline(
    client, db_session, tmp_path, monkeypatch
):
    monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path / "storage"))
    _write_real_video(tmp_path / "london.mp4")

    # Listing through the API surface too.
    listed = await client.get("/api/data-videos")
    assert listed.status_code == 200
    assert [v["name"] for v in listed.json()["videos"]] == ["london.mp4"]

    resp = await client.post(
        "/api/data-videos/london.mp4/start",
        json={"target_fps": 4.0, "top_percent": 0.5, "max_depth_views": 12},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    run_id = body["run_id"]
    assert run_id.startswith("data_run_")

    # Project created in storage workspace with a copied video. The copy is
    # the DECODE-READY bytes (identical name unless this OpenCV build needed
    # the transcode fallback), so the frames stage never re-transcodes inside
    # the pipeline.
    from app.services.video_validation import ensure_cv2_readable

    expected_video_name = ensure_cv2_readable(tmp_path / "london.mp4").name
    project = (
        await db_session.execute(select(Project).where(Project.id == run_id))
    ).scalar_one()
    assert project.video_filename == expected_video_name
    assert project.status == "uploaded"
    workspace = settings.storage.project_dir(run_id)
    assert (workspace / expected_video_name).is_file()

    # Exactly one queued pipeline job whose payload carries the tuning fields
    # verbatim (the _pipeline_executor feeds them into PipelineRequest).
    jobs = (await db_session.execute(select(QueueJob).where(QueueJob.project_id == run_id))).scalars().all()
    assert len(jobs) == 1
    assert jobs[0].kind == "pipeline"
    payload = json.loads(jobs[0].payload_json)
    assert payload["target_fps"] == 4.0
    assert payload["top_percent"] == 0.5
    assert payload["max_depth_views"] == 12
    assert payload["force"] == ["frames", "sparse", "depth", "dense", "georef"]

    # Durable run-input provenance: sidecar written at creation with the real
    # fingerprint of the exact bytes that produced this run.
    from app.services.provenance import read_source_record, sha256_of

    src = read_source_record(workspace)
    assert src is not None and src["kind"] == "data_video"
    assert src["original_filename"] == expected_video_name
    assert src["sha256"] == sha256_of(workspace / expected_video_name)
    assert src["file_size_bytes"] == (workspace / expected_video_name).stat().st_size
    assert src["dataset_relative_path"] == "data/london.mp4"


@pytest.mark.asyncio
async def test_start_unknown_video_returns_404(client, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path / "storage"))
    resp = await client.post("/api/data-videos/nope.mp4/start", json={})
    assert resp.status_code == 404
    assert "nope.mp4" in resp.json()["error"]


# ---------------------------------------------------------------------------
# _request_from_payload — the executor's PipelineRequest contract
# ---------------------------------------------------------------------------


def test_request_from_payload_no_duplicate_force():
    """A 'force' key in the payload must not collide with the explicit kwarg
    (regression: PipelineRequest() got multiple values for 'force')."""
    request = _request_from_payload(
        {"force": ["frames", "sparse"], "target_fps": 4.0, "top_percent": 0.5}
    )
    assert request.force == ["frames", "sparse"]
    assert request.target_fps == 4.0
    assert request.top_percent == 0.5
    assert request.plugins == []


def test_request_from_payload_ignores_unknown_keys():
    request = _request_from_payload({"not_a_field": 1, "plugins": ["mesh_generation"]})
    assert request.plugins == ["mesh_generation"]
    assert not hasattr(request, "not_a_field")


# ---------------------------------------------------------------------------
# Decode-ready workspace copy + transcode-sidecar hygiene
# ---------------------------------------------------------------------------


def test_transcode_sidecars_are_never_offered_or_accepted(tmp_path, monkeypatch):
    """A ``*_cv2dec.mp4`` sibling is derived footage for one OpenCV build's

    benefit — never a mission source. Regression: the picker listed it as a
    duplicate ``london.mp4`` entry and the resolver accepted it by name.
    """
    monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
    _write_real_video(tmp_path / "london.mp4")
    sidecar = tmp_path / "london_cv2dec.mp4"
    shutil.copy2(tmp_path / "london.mp4", sidecar)

    assert [v["name"] for v in list_data_videos()] == ["london.mp4"]
    with pytest.raises(FileNotFoundError):
        resolve_data_video("london_cv2dec.mp4")
    # The real source stays resolvable.
    assert resolve_data_video("london.mp4") == (tmp_path / "london.mp4").resolve()


@pytest.mark.asyncio
async def test_start_copies_decode_ready_video_into_workspace(
    client, db_session, tmp_path, monkeypatch
):
    """The frames stage must never transcode: the workspace copy is already

    cv2-readable. Regression: the RAW source was copied, so every mission
    re-transcoded the whole video inside the pipeline — silently, before the
    first Frame Extraction progress tick (~8 measured minutes of dead time).
    """
    from unittest.mock import patch

    from app.services import video_validation as vv

    monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path / "storage"))
    _write_real_video(tmp_path / "london.mp4")

    # Simulate the FFMPEG-less OpenCV build exactly as the fallback suite does:
    # the raw source is refused, anything ffmpeg produces is accepted.
    real_opens = vv.cv2_opens

    def fake_opens(path):
        return Path(path).resolve() != (tmp_path / "london.mp4").resolve() and real_opens(path)

    with patch.object(vv, "cv2_opens", fake_opens):
        resp = await client.post("/api/data-videos/london.mp4/start", json={})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    workspace = settings.storage.project_dir(run_id)
    videos = [p.name for p in workspace.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
    assert videos == ["london_cv2dec.mp4"]
    # The copied bytes are cv2-readable WITHOUT any simulated fallback.
    assert real_opens(workspace / "london_cv2dec.mp4")
    # Provenance names the exact file the pipeline will consume.
    from app.services.provenance import read_source_record

    src = read_source_record(workspace)
    assert src["original_filename"] == "london_cv2dec.mp4"


@pytest.mark.asyncio
async def test_calibration_conversion_failure_warns_without_aborting_mission_start(
    client, db_session, tmp_path, monkeypatch
):
    """Regression (AUD-001): a failing ``cameras.txt`` conversion must warn, not abort.

    The handler called ``get_logger(...)``, but a later
    ``from app.logging_config import get_logger`` in the SAME function made that
    name a function-local for the whole scope. The reference was therefore an
    unbound local, and the raised ``UnboundLocalError`` replaced the handled
    error and escaped ``start_data_video_mission`` -- turning an optional
    calibration problem into a failed mission start.
    """
    monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(settings.storage, "base_path", str(tmp_path / "storage"))
    dataset = tmp_path / "airport1"
    dataset.mkdir()
    _write_real_video(dataset / "video.mp4")
    # Presence of cameras.txt is what selects the conversion branch.
    (dataset / "cameras.txt").write_text("1 PINHOLE 3840 2160 2000 2000 1920 1080\n")

    from app.services import colmap_text

    def boom(*_args, **_kwargs):
        raise RuntimeError("undecodable camera model")

    monkeypatch.setattr(colmap_text, "camera_txt_to_intrinsics", boom)

    resp = await client.post("/api/data-videos/airport1/video.mp4/start", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "queued"
    # The run is still created with its workspace intact.
    run_id = resp.json()["run_id"]
    assert settings.storage.project_dir(run_id).is_dir()


def test_walk_does_not_stat_every_entry(tmp_path, monkeypatch):
    """The picker must not stat every entry in the folders it walks.

    Regression: ``_iter_video_files`` called ``Path.is_file()`` on every entry
    BEFORE checking the depth guard, so a folder holding ~80k images (the
    reference tree has ``AGZ/MAV Images`` with 81,169 files) cost ~80k stat
    syscalls on every listing — 2.4 s of a 2.8 s request. Entry types now come
    from the directory record (``os.scandir`` -> ``d_type``). This pins the
    mechanism: if a per-entry stat returns, the call count stops being zero.
    """
    monkeypatch.setattr(data_video_service, "data_dir", lambda: tmp_path)
    bulk = tmp_path / "MAV Images"
    bulk.mkdir()
    for i in range(200):
        (bulk / f"img_{i:05d}.jpg").write_bytes(b"x")
    # Name-only walk: the files do not need to be decodable videos here.
    (bulk / "clip.mp4").write_bytes(b"x")
    (tmp_path / "root.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")

    calls = {"n": 0}
    real_is_file = Path.is_file

    def counting_is_file(self):
        calls["n"] += 1
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", counting_is_file)
    found = [p.name for p in data_video_service._iter_video_files(tmp_path)]

    assert calls["n"] == 0, "the walk stat-ed entries instead of using d_type"
    assert found == ["clip.mp4", "root.mp4"]  # bulk images never yielded
