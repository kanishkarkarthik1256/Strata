# STRATA Phase 11.5 — Backend Runtime & Bundling Strategy

## Overview
This document specifies the Python backend lifecycle, port management, health checks, shutdown procedure, and bundling strategy for STRATA desktop deployments.

## Backend Process Lifecycle

```
STRATA Application Launch
        |
        v
Tauri Desktop Shell
        |
        | 1. Spawn Python Backend Process (Sidecar / Standalone binary)
        v
Python FastAPI Engine Started
        |
        | 2. Listen on 127.0.0.1 (Default Port: 8000, Fallback Range: 8000-8005)
        v
Frontend Health Discovery (`checkBackendHealth`)
        |
        | 3. GET /api/runs (HTTP 200 OK)
        v
UI Transition to READY State
```

## Detailed Engine Lifecycle Management

1. **Automatic Startup**:
   - Spawns Python backend on desktop app launch.
   - Enforces `127.0.0.1` binding (prohibiting external network exposure).

2. **Port Management**:
   - `desktopRuntime.ts` tests port 8000.
   - If port 8000 is occupied by an unrelated service, scans fallback range `8000–8005` and passes active URL to the API client.

3. **Backend Health Check**:
   - UI displays `STARTING` state until `/api/runs` returns `200 OK`.
   - If startup fails or times out, displays `STRATA Engine failed to start` with options: `Retry`, `Technical Details`, `Open Logs`.

4. **Graceful Shutdown**:
   - On application exit, Tauri shell sends `SIGTERM` / shutdown signal to the Python process and cleans up any child workers.

## Bundling Strategy
- **Bundler**: PyInstaller / PyOxidizer binary packaging.
- **Dependencies Included**: Embedded Python 3.9 interpreter, PyTorch, pycolmap, OpenCV, FastAPI, uvicorn, SQLAlchemy.
- **Exclusions**: Source `.venv` directories, development test files, unused model checkpoints.
