# STRATA — Phase 11.3 Real Footage Ingestion & End-to-End Processing Report

## Executive Summary

Phase 11.3 integrates real drone footage ingestion (`.mp4`, `.mov`, `.avi`, `.mkv`), pre-flight input quality analysis, mission creation, live pipeline stage tracking, artifact discovery, 3D model viewer loading, and grounded AI copilot Q&A under the **STRATA** platform branding.

## Architecture Highlights

1. **Rebranded Core**:
   - Product rebranded to **STRATA**.
   - Preserved existing backend pipeline, COLMAP/pycolmap integration, Depth Anything V2, OpenCV fallback, and SQL database schema.

2. **Ingestion & Metadata Extraction**:
   - Video streaming to disk without path traversal or in-memory array duplication.
   - Real metadata extraction via OpenCV + FFprobe (Resolution, FPS, Duration, Codec, Bitrate, Frame Count, GPS lat/lon/alt, Camera model).

3. **Live Processing & Real-Time Tracking**:
   - Real stage states (QUEUED, RUNNING, COMPLETED, WARNING, FAILED, CANCELLED).
   - Dynamic fallback notice (Depth Anything V2 relative depth vs. Stereo SGBM metric depth).
   - Real-time pipeline status tracking over SSE / polling.

4. **Run & Artifact Discovery**:
   - Dual-root discovery (`outputs/` for precomputed demo runs, `data/storage/` for live project workspaces).
   - Automatic generation of `manifest.json` on pipeline completion.

5. **AI Copilot Grounding**:
   - Q&A responses grounded in currently selected run ID data (`/api/runs/${runId}/copilot`).
