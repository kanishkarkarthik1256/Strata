"""Integration tests for the upload API.

Tests upload, validation, metadata extraction, status retrieval, and deletion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_valid_video(client: AsyncClient, synthetic_video: Path):
    """Upload a valid synthetic video and verify response."""
    with open(synthetic_video, "rb") as f:
        response = await client.post(
            "/api/upload",
            files={"file": ("test_drone.avi", f, "video/avi")},
        )

    assert response.status_code == 201
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "uploaded"
    assert "test_drone.avi" in data["message"]

    # Durable run-input provenance sidecar, kind=uploaded, real fingerprint.
    from app.config.settings import settings
    from app.services.provenance import read_source_record, sha256_of

    workspace = settings.storage.project_dir(data["job_id"])
    src = read_source_record(workspace)
    assert src is not None and src["kind"] == "uploaded"
    assert src["original_filename"] == "test_drone.avi"
    stored = next(p for p in workspace.iterdir() if p.suffix == ".avi")
    assert src["sha256"] == sha256_of(stored)
    assert src["file_size_bytes"] == stored.stat().st_size

    return data["job_id"]


@pytest.mark.asyncio
async def test_upload_unsupported_format(client: AsyncClient, tmp_path: Path):
    """Upload an unsupported file type and verify rejection."""
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("not a video")

    with open(bad_file, "rb") as f:
        response = await client.post(
            "/api/upload",
            files={"file": ("test.txt", f, "text/plain")},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_job_status(client: AsyncClient, synthetic_video: Path):
    """Upload a video, then retrieve its status with metadata."""
    # Upload
    with open(synthetic_video, "rb") as f:
        upload_resp = await client.post(
            "/api/upload",
            files={"file": ("test_drone.avi", f, "video/avi")},
        )
    job_id = upload_resp.json()["job_id"]

    # Get status
    status_resp = await client.get(f"/api/upload/{job_id}")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["job_id"] == job_id
    assert data["status"] == "uploaded"
    assert data["metadata"] is not None
    assert data["metadata"]["filename"] == "test_drone.avi"
    assert data["metadata"]["width"] == 640
    assert data["metadata"]["height"] == 480
    assert data["metadata"]["fps"] == 30.0
    assert data["metadata"]["duration_sec"] > 0


@pytest.mark.asyncio
async def test_get_nonexistent_job(client: AsyncClient):
    """Request status for a non-existent job returns 404."""
    response = await client.get("/api/upload/nonexistent123")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_job(client: AsyncClient, synthetic_video: Path):
    """Upload, then delete, then verify it's gone."""
    # Upload
    with open(synthetic_video, "rb") as f:
        upload_resp = await client.post(
            "/api/upload",
            files={"file": ("test_drone.avi", f, "video/avi")},
        )
    job_id = upload_resp.json()["job_id"]

    # Delete
    delete_resp = await client.delete(f"/api/upload/{job_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["status"] == "deleted"

    # Verify gone
    get_resp = await client.get(f"/api/upload/{job_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_job(client: AsyncClient):
    """Delete a non-existent job returns 404."""
    response = await client.delete("/api/upload/nonexistent123")
    assert response.status_code == 404
