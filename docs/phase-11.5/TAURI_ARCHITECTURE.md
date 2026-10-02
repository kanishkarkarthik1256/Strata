# STRATA Phase 11.5 — Tauri Shell Architecture & IPC Protocol

## Overview
This document specifies the Tauri desktop shell configuration, native IPC commands, and frontend-backend bridge for STRATA.

## Tauri Configuration (`tauri.conf.json`)

- **Product Name**: `STRATA`
- **Identifier**: `com.strata.drone-recon`
- **Frontend Dist**: `../dist`
- **Development Path**: `http://localhost:5173`
- **Window Specs**: 1280x832, resizable, title: `STRATA — AI Drone Reconstruction & Spatial Intelligence`

## Tauri Native IPC Commands (`main.rs`)

1. `get_app_info()`:
   - Returns application name (`STRATA`), version (`1.0.0`), engine status (`READY`), and default port (`8000`).

2. `get_system_diagnostics()`:
   - Returns host OS (`macos` / `windows`), CPU architecture (`x86_64` / `arm64`), storage path (`data/storage`), and active backend URL (`http://127.0.0.1:8000`).

## Security & IPC Restrictions
- Command permissions restricted via Tauri allowlist.
- Shell sidecar execution scoped strictly to the Python backend binary.
- Filesystem access scoped to application data and user-selected project directories.
