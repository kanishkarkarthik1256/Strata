# STRATA Phase 11.3 — Live Validation Protocol

## Overview
This document specifies the validation procedures for confirming end-to-end real video ingestion and processing in STRATA.

## Live End-to-End Workflow Validation Steps

1. **Backend Server Verification**:
   - Confirm FastAPI server is active: `curl http://localhost:8000/health` (or `/api/runs`).
   - Confirm dual-root directory structures exist: `outputs/` and `data/storage/`.

2. **Frontend UI Verification**:
   - Launch Vite dev server: `npm run dev` in `frontend/`.
   - Verify STRATA header logo, "+ NEW MISSION" button, and status tabs (ALL, COMPLETED, PROCESSING, FAILED).

3. **Video Upload Flow**:
   - Open "+ NEW MISSION" modal.
   - Drag & drop a real drone video file (`.mp4` / `.mov`).
   - Observe progress bar advancing in real time based on raw byte transfer.
   - Confirm Pre-Flight Input Quality card displays:
     - Resolution (e.g. 1920x1080)
     - Duration (e.g. 15.4s)
     - FPS (e.g. 30 fps)
     - GPS Status (Extracted vs Origin fallback)

4. **Pipeline Execution & Telemetry**:
   - Click "START RECONSTRUCTION".
   - Confirm UI transitions automatically to `ProcessingView`.
   - Verify stages update dynamically (frame_extraction -> feature_matching -> ... -> export_artifacts).
   - If OpenCV fallbacks activate, verify the "OpenCV Fallback Engine Active" notice appears.

5. **Artifact Discovery & 3D Viewer Integration**:
   - Upon completion, click "OPEN 3D MODEL".
   - Confirm 3D Viewer loads the generated PLY file from `data/storage/<PROJECT_ID>/dense/dense_model.ply`.
   - Verify AI Copilot correctly answers questions regarding point counts and camera positions for the newly created mission.

6. **Demo Mission Regression Verification**:
   - Click "OPEN DEMO MISSION" (`shitan_ms1_20260909_131251`).
   - Confirm precomputed demo mission loads seamlessly with 2,626,577 points and "PRECOMPUTED DEMO" badge intact.
