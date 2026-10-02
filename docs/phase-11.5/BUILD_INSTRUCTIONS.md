# STRATA Phase 11.5 — Desktop Build Instructions

## Overview
This document specifies step-by-step build instructions for compiling STRATA desktop packages on macOS and Windows.

## Prerequisites
- Node.js v18+ & npm
- Python 3.9+ virtual environment (`backend/.venv`)
- Rust toolchain (`cargo`, `rustc`) for Tauri build
- PyInstaller (`pip install pyinstaller`)

## Build Steps

### 1. Build Frontend Bundle
```bash
cd frontend
npm install
npm run build
```
Generates production assets in `frontend/dist/`.

### 2. Package Python Backend Binary
```bash
cd backend
source .venv/bin/activate
pyinstaller --onefile --name strata-engine app/main.py
```
Generates executable `backend/dist/strata-engine`.

### 3. Build Desktop Application (Tauri)
```bash
cd frontend
npm run tauri build
```

## Output Artifacts
- **macOS**: `frontend/src-tauri/target/release/bundle/dmg/STRATA_1.0.0_x64.dmg`
- **Windows**: `frontend/src-tauri/target/release/bundle/nsis/STRATA_1.0.0_x64-setup.exe`
