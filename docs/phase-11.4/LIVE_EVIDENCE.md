# STRATA Phase 11.4 — Live Evidence Table

## Overview
This document records empirical evidence across every major operational subsystem in STRATA.

## Subsystem Evidence Matrix

| Subsystem / Test | Result | Evidence / Details |
|------------------|--------|--------------------|
| **Application Startup** | **VERIFIED** | FastAPI backend (`localhost:8000`) and Vite frontend started cleanly with live health check API responding `200 OK`. |
| **Upload Stream & Progress** | **VERIFIED** | Chunked upload streamed to `data/storage/<ID>/` with real byte progress (`0%` -> `100%`). |
| **Video Metadata Extraction** | **VERIFIED** | Resolution, FPS, duration, codec, bitrate extracted via OpenCV/FFprobe. |
| **Mission Creation** | **VERIFIED** | Dynamic UUID project ID created in SQLite database and project workspace directory initialized. |
| **Run Creation & Discovery** | **VERIFIED** | `run_service.py` dual-root scanner discovers runs in both `outputs/` and `data/storage/`. |
| **Processing State Telemetry** | **VERIFIED** | Stage progress polled live from `/api/projects/<ID>/pipeline/status` without fake percentage loops. |
| **Reconstruction Execution** | **VERIFIED** | Stages 1–8 execute sequentially; fallback notice displayed when pure OpenCV runs. |
| **Artifact Generation** | **VERIFIED** | `manifest.json`, `dense/dense_model.ply`, `sparse/sparse_model.ply`, and `poses.json` generated in workspace. |
| **3D Viewer Integration** | **VERIFIED** | `ViewerCanvas.tsx` loads binary PLY point cloud and camera poses directly for the active run ID. |
| **GPS Representation** | **VERIFIED** | Missing GPS correctly displays `GPS: Not available`. Zero coordinate substitution prohibited. |
| **Confidence Heatmap** | **VERIFIED** | Displays `Spatial confidence unavailable for this reconstruction` when per-vertex confidence is absent. |
| **Measurements** | **VERIFIED** | Rendered with `Estimated Measurement` label since metric scale is uncalibrated. |
| **Report Generation** | **VERIFIED** | PDF/JSON summary report references active mission ID and metrics without demo data leakage. |
| **Copilot Grounding & Stale Context** | **VERIFIED** | Copilot Q&A queries selected run ID artifacts only; switching missions updates context immediately. |
| **Restart / Reopen** | **VERIFIED** | Completed project workspaces persist across server restarts and reopen in 3D Viewer without re-running reconstruction. |
| **Security & Traversal** | **VERIFIED** | Path traversal attempts (`../../../etc/passwd`) blocked by `safe_filename()` and `resolve_artifact()`. |
