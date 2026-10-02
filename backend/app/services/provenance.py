"""Run input provenance — answers "what exact input generated this run?".

A ``source.json`` sidecar is written in the run workspace at creation time by
each ingestion path (manual upload, data-folder mission). The manifest writer
promotes it to the manifest's ``source`` record so the API/UI can show the
exact input without exposing absolute filesystem paths.

Historical runs without a sidecar are reported as ``kind: "unknown"`` —
never back-filled with guessed provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    """Streaming SHA256 of a file (constant memory for multi-GB videos)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_source_record(
    workspace: Path,
    *,
    kind: str,
    original_filename: str,
    stored_path: Path,
    duration_sec: float | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    dataset_name: str | None = None,
    sequence_id: str | None = None,
    dataset_relative_path: str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``source.json`` in the run workspace and return the record.

    ``kind`` is one of: uploaded | data_video | dataset | synthetic_test.
    Only the original (client-supplied) filename is stored — never absolute
    server paths. ``extras`` carries run-input provenance such as telemetry
    and calibration conversion records.
    """
    size = stored_path.stat().st_size
    record: dict[str, Any] = {
        "kind": kind,
        "original_filename": original_filename,
        "sha256": sha256_of(stored_path),
        "file_size_bytes": size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    for key, value in (
        ("duration_sec", duration_sec),
        ("width", width),
        ("height", height),
        ("fps", fps),
        ("dataset_name", dataset_name),
        ("sequence_id", sequence_id),
        ("dataset_relative_path", dataset_relative_path),
    ):
        if value is not None:
            record[key] = value
    if extras:
        record.update(extras)
    workspace.mkdir(parents=True, exist_ok=True)
    with open(workspace / "source.json", "w") as f:
        json.dump(record, f, indent=2)
    return record


def read_source_record(workspace: Path) -> dict[str, Any] | None:
    """The run's source record, or None when the run predates provenance."""
    path = workspace / "source.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def manifest_source(workspace: Path) -> dict[str, Any]:
    """The manifest ``source`` entry — honest unknown for historical runs."""
    record = read_source_record(workspace)
    if record is None:
        return {"kind": "unknown"}
    return record
