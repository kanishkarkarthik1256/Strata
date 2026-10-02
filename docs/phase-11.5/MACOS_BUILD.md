# STRATA Phase 11.5 — macOS Packaging & Build Specification

## Overview
This document specifies the target architecture, packaging configuration, and test status for macOS desktop builds of STRATA.

## Target Configuration

- **Target OS**: macOS 12+ (Intel `x86_64` & Apple Silicon `aarch64`).
- **Package Format**: App Bundle (`.app`) & Disk Image (`.dmg`).
- **Signing Status**: **NOT SIGNED** (Unsigned development build).

## Build & Test Status

| Architecture | Build Status | Test Status | GPU Status |
|--------------|--------------|-------------|------------|
| **macOS Intel (`x86_64`)** | **CONFIGURED** | **CONFIGURED / TESTED IN DEV** | **CPU Only** (No CUDA/MPS) |
| **macOS Apple Silicon (`aarch64`)** | **CONFIGURED** | **NOT TESTED** | **Apple MPS (Not Tested)** |

> [!WARNING]
> **Apple MPS Reality**: Apple Metal Performance Shaders (MPS) hardware acceleration MUST NOT be reported as READY on Intel macOS machines. On Intel macOS, hardware status correctly reports `GPU: Unavailable`, `Processing: CPU`.

## Expected Application Structure (macOS)

```
STRATA.app/
└── Contents/
    ├── MacOS/
    │   ├── strata-desktop (Tauri binary)
    │   └── strata-engine (Python backend executable)
    └── Resources/
        ├── dist/ (Frontend UI)
        └── models/ (Depth Anything V2 weights)
```

## User Data Directory (macOS)
`~/Library/Application Support/STRATA/projects/`
