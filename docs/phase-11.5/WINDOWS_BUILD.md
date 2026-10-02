# STRATA Phase 11.5 — Windows Packaging & Build Specification

## Overview
This document specifies the target architecture, packaging configuration, and test status for Windows desktop builds of STRATA.

## Target Configuration

- **Target OS**: Windows 10 / 11 (64-bit).
- **Installer Format**: NSIS / MSI Installer (`.exe` / `.msi`).
- **Binary Architecture**: `x86_64-pc-windows-msvc`.

## Build Status

| Metric | Status | Notes |
|--------|--------|-------|
| **Build Configuration** | **CONFIGURED** | `tauri.conf.json`, `Cargo.toml`, and PyInstaller build targets defined. |
| **Windows Compilation** | **NOT BUILT** | Current build environment is macOS Darwin; cross-compilation requires Windows runner or MSVC toolchain. |
| **Windows Runtime Test** | **NOT TESTED** | Runtime execution not performed on Windows in current macOS environment. |

## Expected Desktop Layout (Windows)

```
C:\Program Files\STRATA\
├── STRATA.exe
├── resources\
│   ├── app\ (Frontend Dist)
│   └── engine\ (PyInstaller PyTorch/pycolmap backend)
└── models\
    └── depth_anything_v2_vits.pth
```

## User Data Directory (Windows)
`%APPDATA%\STRATA\projects\`
