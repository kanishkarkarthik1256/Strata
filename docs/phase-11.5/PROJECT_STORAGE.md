# STRATA Phase 11.5 — Desktop Data Directory & Upgrade Isolation

## Overview
This document specifies user data storage locations, directory layout, and data persistence guarantees across application updates.

## Application Directory Model

```
STRATA Application Directory (Installation Root — READ ONLY)
├── STRATA.exe / STRATA.app
├── frontend/dist/
├── backend/engine/
└── models/weights/

User Data Directory (Persistent User Storage — WRITABLE)
├── projects/ (data/storage/<PROJECT_ID>/)
│   ├── video.mp4
│   ├── manifest.json
│   ├── poses.json
│   ├── sparse/
│   ├── dense/
│   └── reports/
├── outputs/ (Precomputed Demo Runs)
└── logs/ (app.log & backend.log)
```

## Platform Storage Paths

- **macOS**: `~/Library/Application Support/STRATA/`
- **Windows**: `%APPDATA%\STRATA\`
- **Development Default**: `data/storage/` in current workspace directory.

## Upgrade Isolation Guarantees
User missions, drone video uploads, 3D point cloud PLY files, and reports are stored exclusively in the **User Data Directory**. Upgrading or reinstalling the STRATA application binary does NOT modify or erase user project data.
