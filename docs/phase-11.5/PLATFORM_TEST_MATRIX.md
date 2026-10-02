# STRATA Phase 11.5 — Platform Test Matrix

## Overview
This document records test status across target desktop operating systems and hardware configurations.

## Platform Test Matrix

| Capability | Windows (`x86_64`) | macOS Intel (`x86_64`) | macOS Apple Silicon (`aarch64`) |
|------------|-------------------|------------------------|---------------------------------|
| **App Launch** | **CONFIGURED** | **PASS (Dev Shell)** | **CONFIGURED** |
| **Backend Start** | **CONFIGURED** | **PASS** | **CONFIGURED** |
| **Upload Stream** | **CONFIGURED** | **PASS** | **CONFIGURED** |
| **Processing Pipeline** | **CONFIGURED** | **PASS** | **CONFIGURED** |
| **3D Model Viewer** | **CONFIGURED** | **PASS** | **CONFIGURED** |
| **Site Reports** | **CONFIGURED** | **PASS** | **CONFIGURED** |
| **AI Copilot Q&A** | **CONFIGURED** | **PASS** | **CONFIGURED** |
| **GPU Acceleration** | **NOT TESTED** | **UNAVAILABLE (CPU Only)** | **NOT TESTED** |
| **Project Storage** | **CONFIGURED** | **PASS** | **CONFIGURED** |
| **Report Export** | **CONFIGURED** | **PASS** | **CONFIGURED** |
| **Shutdown Handling** | **CONFIGURED** | **PASS** | **CONFIGURED** |

> [!NOTE]
> **State Key**:
> - **PASS**: Verified working in live execution.
> - **FAIL**: Executed and failed.
> - **NOT TESTED**: Environment unavailable for live runtime testing.
> - **CONFIGURED**: Package and runtime configuration built, pending environment compilation.
