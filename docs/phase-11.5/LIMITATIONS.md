# STRATA Phase 11.5 — Desktop System Limitations & Bounds

## Overview
This document specifies technical bounds, build environment constraints, and platform limitations established during Phase 11.5.

## Operational Bounds & Limitations

1. **Rust / Cargo Toolchain Status**:
   - Tauri desktop shell configuration (`tauri.conf.json`, `Cargo.toml`, `main.rs`) is fully defined.
   - Live Tauri binary compilation on the host machine requires installing the Rust toolchain (`cargo`).

2. **Cross-Platform Compilation**:
   - Windows binaries (`.exe` / `.msi`) are **CONFIGURED** but cannot be compiled directly on macOS without a Windows build runner or MSVC toolchain.

3. **Code Signing & Notarization**:
   - Production installer targets are **NOT SIGNED** / **NOT NOTARIZED** as development certificates are not configured in this environment. Builds are documented as `Unsigned development build`.

4. **Hardware Acceleration**:
   - Hardware detection correctly reports `GPU: Unavailable`, `Processing: CPU` on Intel macOS host.
