# STRATA Phase 11.5 — Model Weights Classification & Packaging

## Overview
This document classifies model files and datasets in the repository to establish an efficient desktop package size.

## Model Classification Matrix

| File Path | Size | Classification | Packaging Policy |
|-----------|------|----------------|------------------|
| `backend/models/weights/depth_anything_v2_vits.pth` | **98.2 MB** | **REQUIRED** | Bundle with production installer. |
| Precomputed Demo (`outputs/shitan_ms1_...`) | **31.5 MB** | **DEMO ONLY** | Include in standard release bundle for demo mission. |
| Test Videos (`backend/data/storage/.../test_drone.avi`) | **~2 MB** | **DEVELOPMENT ONLY** | Exclude from production bundle. |
| PyTorch / pycolmap Native Shared Libs | **~180 MB** | **REQUIRED** | Package in backend sidecar bundle. |

## Package Size Estimates

- **Frontend Dist**: ~4.5 MB
- **Tauri Executable Shell**: ~12 MB
- **Python Backend Bundle (PyInstaller)**: ~220 MB
- **AI Models (Depth Anything V2)**: ~98 MB
- **Total Production Package Size**: **~335 MB**
