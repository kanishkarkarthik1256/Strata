# STRATA — Phase 11.3 Upload & Processing API Specification

This document specifies the contract between the STRATA frontend (React + Vite) and backend (FastAPI) for real video ingestion, mission creation, pipeline execution, stage tracking, artifact discovery, and grounded AI copilot Q&A.

---

## 1. Video Upload & Ingestion

### `POST /api/upload`
Uploads a drone video file (`.mp4`, `.mov`, `.avi`, `.mkv`). The backend validates file integrity, streams chunks to disk safely without path traversal, extracts video & GPS metadata, and registers a project.

- **Request**: `multipart/form-data` with field `file`
- **Response** `201 Created`:
```json
{
  "job_id": "a1b2c3d4e5f6...",
  "status": "uploaded",
  "message": "Video 'site_a.mp4' uploaded and validated successfully"
}
```

### `GET /api/upload/{job_id}`
Retrieves job status and extracted video/GPS metadata.

- **Response** `200 OK`:
```json
{
  "job_id": "a1b2c3d4e5f6...",
  "status": "uploaded",
  "filename": "site_a.mp4",
  "metadata": {
    "filename": "site_a.mp4",
    "duration_sec": 124.5,
    "fps": 29.97,
    "width": 3840,
    "height": 2160,
    "codec": "h264",
    "frame_count": 3731,
    "bitrate_kbps": 45000.0,
    "file_size_bytes": 700000000,
    "gps_lat": 37.7749,
    "gps_lon": -122.4194,
    "gps_alt": 45.2,
    "camera_model": "DJI FC330"
  },
  "workspace_path": "/path/to/data/storage/a1b2c3d4e5f6...",
  "error": null,
  "created_at": "2026-09-10T21:45:00Z",
  "updated_at": "2026-09-10T21:45:00Z"
}
```

---

## 2. Mission Management

### `POST /api/missions`
Creates a mission wrapper around an uploaded project workspace.

- **Request Body**:
```json
{
  "name": "Site A Mission",
  "project_id": "a1b2c3d4e5f6...",
  "priority": 5,
  "description": "Nadir survey of structure A"
}
```

- **Response** `201 Created`:
```json
{
  "id": "m1234567...",
  "name": "Site A Mission",
  "project_id": "a1b2c3d4e5f6...",
  "status": "CREATED",
  "priority": 5,
  "current_stage": null,
  "stage_progress": 0.0,
  "error": null,
  "created_at": "2026-09-10T21:45:05Z",
  "updated_at": "2026-09-10T21:45:05Z"
}
```

---

## 3. Pipeline Execution & Real-Time Tracking

### `POST /api/pipeline/start/{job_id}`
Starts or resumes the autonomous pipeline for project `job_id`.

- **Request Body**:
```json
{
  "extraction_mode": "every_n",
  "every_n": 10,
  "target_fps": 2.0,
  "depth_backend": "auto",
  "force": []
}
```

- **Response** `200 OK`:
```json
{
  "job_id": "a1b2c3d4e5f6...",
  "status": "completed",
  "run_time_ms": 45200.0,
  "error": "",
  "stages": { ... },
  "message": "Pipeline completed — export the dense model from /api/dense/download/a1b2c3d4e5f6..."
}
```

### `GET /api/pipeline/status/{job_id}`
Returns current stage timeline, performance profile, and artifact resume status.

### `POST /api/pipeline/cancel/{job_id}`
Requests cancellation of the active run.

- **Response** `200 OK`:
```json
{
  "job_id": "a1b2c3d4e5f6...",
  "cancelled": true
}
```

---

## 4. Run & Artifact Discovery

### `GET /api/runs`
Lists all discovered runs (both `outputs/` demo runs and user-uploaded project workspaces in `data/storage/`).

- **Response** `200 OK`:
```json
{
  "runs": [
    {
      "run_id": "a1b2c3d4e5f6...",
      "dataset": "site_a.mp4",
      "mission": "Site A Mission",
      "status": "completed",
      "dense_points": 145200,
      "sparse_points": 2800,
      "cameras": 10,
      "gps_points": 10,
      "mean_reprojection_error_px": 1.1,
      "artifacts": [
        { "path": "dense/dense_model.ply", "size_bytes": 5808000 },
        { "path": "poses.json", "size_bytes": 4500 }
      ]
    }
  ]
}
```

### `GET /api/runs/{run_id}/artifact/{name}`
Serves a validated, traversal-safe artifact file (e.g. `dense/dense_model.ply`, `poses.json`, `reconstruction_report.json`).

---

## 5. Grounded AI Copilot

### `POST /api/runs/{run_id}/copilot`
Answers questions about `run_id` grounded in its real manifest, stage reports, and point counts.

- **Request Body**:
```json
{
  "query": "How many points were reconstructed in this mission?"
}
```

- **Response** `200 OK`:
```json
{
  "query": "How many points were reconstructed in this mission?",
  "intent": "points",
  "answer": "the dense reconstruction contains 145,200 points (sparse model: 2,800 points)",
  "results": [{ "dense_points": 145200, "sparse_points": 2800 }],
  "grounded": true
}
```
