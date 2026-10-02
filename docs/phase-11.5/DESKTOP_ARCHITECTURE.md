# STRATA Phase 11.5 — Desktop Architecture Specification

## Overview
This document specifies the current codebase architecture, runtime components, subprocess usage, and desktop application integration for **STRATA**.

## 1. Subsystem Responsibilities

```
+-----------------------------------------------------------------------+
|                              STRATA App                               |
|                                                                       |
|  +-----------------------------------------------------------------+  |
|  |                       Tauri Desktop Shell                       |  |
|  |   - Native Window Management   - Native File Dialogs            |  |
|  |   - Backend Process Lifecycle  - System Tray & OS Integration   |  |
|  +-----------------------------------------------------------------+  |
|                                  |                                    |
|                                  v                                    |
|  +-----------------------------------------------------------------+  |
|  |                      React 18 + Vite UI                         |  |
|  |   - 3D Point Cloud Viewer      - Dynamic Processing Views       |  |
|  |   - Grounded AI Copilot        - Site Survey Reports            |  |
|  +-----------------------------------------------------------------+  |
|                                  |                                    |
|                                  v HTTP / SSE                         |
|  +-----------------------------------------------------------------+  |
|  |                     Python / FastAPI Backend                    |  |
|  |   - Reconstruction Pipeline    - Depth Anything V2 AI           |  |
|  |   - pycolmap SfM Engine        - Geospatial Alignment           |  |
|  +-----------------------------------------------------------------+  |
+-----------------------------------------------------------------------+
```

### Responsibility Breakdown
- **Tauri (Desktop Shell)**: Manages native desktop window lifecycle, spawns/terminates the Python sidecar process, exposes native file pickers, and manages application settings/logs.
- **React + Vite (UI Layer)**: Handles UI rendering, 3D point cloud canvas (Three.js), mission management, AI Copilot interaction, and site report visualization.
- **Python / FastAPI (Reconstruction Engine)**: Executes heavy 3D processing, SfM camera pose estimation, depth map generation, point cloud fusion, metadata extraction, and report generation.

---

## 2. Technical Stack Audit

- **Frontend Framework**: React 18.3, Vite 6.0, TypeScript 5.6.
- **Frontend Build Output**: `frontend/dist/`.
- **Package Manager**: `npm`.
- **Python Environment**: `backend/.venv` (Python 3.9.6, PyTorch 2.2, pycolmap 3.12, OpenCV 4.10, FastAPI 0.115).
- **Backend Entry Point**: `backend/app/main.py` (`uvicorn app.main:app`).
- **Default Port**: `8000` (configurable via `PORT` environment variable or `--port`).
- **Model Checkpoints**: `backend/models/weights/depth_anything_v2_vits.pth` (~98 MB).
- **Storage Locations**:
  - Demo Runs: `outputs/`
  - User Projects: `data/storage/` (configurable via `STORAGE_BASE_PATH`)
- **Subprocess Usage**:
  - `metadata_extraction.py`: Invokes `ffprobe` via `subprocess.run(["ffprobe", ...])`.
  - `video_validation.py`: OpenCV `cv2.VideoCapture`.
  - `sparse_reconstruction.py`: PyCOLMAP in-process C++ bindings (`import pycolmap`).
