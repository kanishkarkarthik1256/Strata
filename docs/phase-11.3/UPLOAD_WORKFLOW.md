# STRATA Phase 11.3 — Real Footage Ingestion & Upload Workflow

## Overview
This document details the real drone video ingestion architecture and upload workflow in STRATA.

## Architecture

```
[ User Browser ]
       |
       | 1. Select Video (MP4 / MOV) & Fill Mission Details
       v
[ NewMissionModal UI ]
       |
       | 2. POST /api/upload (Streaming Chunked Transfer with Progress Listener)
       v
[ FastAPI backend / upload_service.py ]
       |
       +--> 3. safe_filename() Sanitization & Security Checks
       +--> 4. Stream to disk (`data/storage/<PROJECT_ID>/<filename>`) with size enforcement
       +--> 5. validate_all() (FFmpeg/OpenCV header & frame integrity validation)
       +--> 6. extract_metadata() (Duration, Resolution, FPS, EXIF GPS extraction)
       +--> 7. Persist Project row in SQLite DB
       |
       v
[ Return Project & Metadata Response ]
       |
       | 8. Display Pre-Flight Input Quality Card & Enable CTA
       v
[ User clicks "START RECONSTRUCTION" ]
       |
       | 9. POST /api/projects/<PROJECT_ID>/pipeline/start
       v
[ Pipeline Orchestrator Worker Execution ]
```

## Detailed Ingestion Steps

### 1. File Selection & Frontend Validation
- Supported formats: `.mp4`, `.mov`, `.avi`, `.mkv`.
- Max upload file size: 2 GB (configurable in `settings.py`).
- Zero browser memory bloat: uses native HTTP chunked streaming instead of `FileReader.readAsArrayBuffer()`.

### 2. Streamed Network Transfer
- Native `XMLHttpRequest` / `axios` with `onUploadProgress` handler.
- Byte-accurate progress rendering: `(loaded / total) * 100%`.
- AbortController support for instant user cancellation.

### 3. Server-Side Storage & Security
- Project workspace path: `data/storage/<PROJECT_ID>/`.
- Filename sanitization via `safe_filename()`:
  - Strips path traversal characters (`..`, `/`, `\`).
  - Replaces non-alphanumeric characters with `_`.
  - Enforces extension allowlist.

### 4. Input Validation & Metadata Extraction
- **FFmpeg / OpenCV Validation**:
  - Verifies container integrity and video stream readability.
  - Rejects corrupt or incomplete video files immediately with clean HTTP 400 errors.
- **Exif / GPS Extraction**:
  - Parses embedded drone telemetry / EXIF metadata for reference latitude, longitude, and altitude.
  - Fallbacks gracefully to spatial origin `(0.0, 0.0, 0.0)` if GPS telemetry is absent, retaining full pipeline support.

### 5. Pre-Flight Input Quality Card
- Renders input quality metrics prior to pipeline trigger:
  - Duration, FPS, Resolution.
  - GPS fix status (Green badge if present, Warning badge if missing).
  - Recommended frame extraction interval (e.g. 1 fps for standard velocity).
