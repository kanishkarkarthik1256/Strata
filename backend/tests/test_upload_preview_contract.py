"""Upload preview/analytics contract.

The New Mission page's pre-reconstruction analytics are only honest if the
server measures what it claims: the upload response carries the validated
metadata (fps/codec/bitrate/GPS from the real file), and a corrupt upload
raises WITHOUT deleting the user's bytes (they can re-encode and retry).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

from app.config.settings import settings


@pytest.mark.asyncio
async def test_upload_response_carries_measured_metadata(client: AsyncClient, synthetic_video: Path):
    with open(synthetic_video, "rb") as f:
        response = await client.post(
            "/api/upload",
            files={"file": ("test_drone.avi", f, "video/avi")},
        )
    assert response.status_code == 201
    data = response.json()
    md = data.get("metadata")
    assert md, "upload response must carry server-measured metadata"
    assert md["width"] == 640 and md["height"] == 480
    assert md["fps"] == pytest.approx(30.0)
    assert md["duration_sec"] == pytest.approx(1.0, abs=0.1)
    assert md["file_size_bytes"] > 0


@pytest.mark.asyncio
async def test_corrupt_upload_raises_but_keeps_the_file(client: AsyncClient, tmp_path: Path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"\x00" * 4096)
    with open(corrupt, "rb") as f:
        response = await client.post(
            "/api/upload",
            files={"file": ("corrupt.mp4", f, "video/mp4")},
        )
    assert response.status_code == 400
    body = response.json()
    assert "Cannot open video" in (body.get("detail") or body.get("error") or "")
    # The user's bytes survive for re-encoding / inspection.
    workspace = Path(settings.storage.base_path)
    candidates = list(workspace.glob("**/corrupt.mp4"))
    assert candidates, "corrupt upload must be preserved, not deleted"
    assert candidates[0].stat().st_size == 4096
    # Cleanup: the failed upload left no project record to delete via API.
    for c in candidates:
        c.unlink()


@pytest.mark.asyncio
async def test_upload_leaves_decodable_source_in_workspace(client: AsyncClient, synthetic_video: Path):
    with open(synthetic_video, "rb") as f:
        response = await client.post(
            "/api/upload",
            files={"file": ("test_drone.avi", f, "video/avi")},
        )
    assert response.status_code == 201
    job_id = response.json()["job_id"]
    stored = settings.storage.project_dir(job_id) / "test_drone.avi"
    assert stored.exists() and stored.stat().st_size > 0
