# STRATA Ground-Truth Dataset Capability Matrix

**Phase 13 Document — Dataset Inventory & Metric Validation Suitability**

---

## Overview

This matrix evaluates all active and stored datasets in the STRATA environment for their suitability for metric accuracy validation. 

### Critical Distinction Rule
In accordance with Phase 13 guidelines:
- Datasets with GPS telemetry enable **GPS Georeferencing Alignment Validation**.
- Datasets **WITHOUT** independently surveyed Ground Control Points (GCPs), Check Points, known scale dimensions, or high-accuracy 3D reference scans are categorized as **NOT AVAILABLE** for surveyed metric validation.
- Missing ground truth is never inferred or fabricated.

---

## Dataset Capability Matrix

| Dataset | Images / Frames | Video Source | GPS Status | Altitude Status | Camera Intrinsics | RTK/PPK | GCPs | Check Points | 3D Ref Scan | Known Distances | Coordinate System | Metric Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Shitan MS1** (`shitan_ms1_20260909_131251`) | 10 frames extracted | `smoke_clip.mp4` | **AVAILABLE** (Standard Consumer) | **AVAILABLE** ($\pm 10\text{m}$) | Estimated (PINHOLE) | **NOT AVAILABLE** | **NOT AVAILABLE** | **NOT AVAILABLE** | **NOT AVAILABLE** | **NOT AVAILABLE** | WGS84 ($\text{EPSG:4326}$) | **GPS_GEOREFERENCED** (Field GT Unavailable) |
| **Site A** (`site_a.mp4`) | 10 frames extracted | `site_a.mp4` | **AVAILABLE** | **AVAILABLE** | Estimated | **NOT AVAILABLE** | **NOT AVAILABLE** | **NOT AVAILABLE** | **NOT AVAILABLE** | **NOT AVAILABLE** | WGS84 ($\text{EPSG:4326}$) | **GPS_GEOREFERENCED** (Field GT Unavailable) |
| **VisDrone 2019** (`VisDrone2019-DET-test`) | Image directory | Static frames | **NOT AVAILABLE** | **NOT AVAILABLE** | **UNKNOWN** | **NOT AVAILABLE** | **NOT AVAILABLE** | **NOT AVAILABLE** | **NOT AVAILABLE** | **NOT AVAILABLE** | Local Pixel Frame | **RELATIVE_ONLY** |
| **Synthetic Metric Bench** (`synthetic_cube_scene`) | 12 synthetic renders | Rendered trajectory | **AVAILABLE** (Exact 0 error) | **AVAILABLE** (Exact) | Known ($f_x=1000$) | Simulated | **AVAILABLE** (4 GCPs) | **AVAILABLE** (4 Check Points) | **AVAILABLE** (Exact Mesh/Cloud) | **AVAILABLE** (2 Distances) | Local ENU ($\text{m}$) | **METRIC_VALIDATED** (Controlled Synthetic) |

---

## Dataset Field Descriptions & Definitions

1. **GPS Status**: `AVAILABLE` if per-frame or video metadata contains valid non-zero latitude and longitude.
2. **Altitude Status**: `AVAILABLE` if relative or absolute barometric/GPS height is recorded.
3. **RTK/PPK**: `AVAILABLE` if survey-grade carrier-phase differential GPS metadata with sub-centimetric precision is attached.
4. **GCPs (Control Points)**: `AVAILABLE` if ground targets with surveyed 3D coordinates are present to align/scale the reconstruction.
5. **Check Points**: `AVAILABLE` if independent, non-aligned ground points exist for metric error calculation.
6. **3D Reference Scan**: `AVAILABLE` if a high-precision LiDAR point cloud or laser-scanned mesh exists for surface error comparison.
7. **Known Distances**: `AVAILABLE` if surveyed scale bars or physical object dimensions are documented.

---

## Required Data for Final Field Validation

To perform real-world field metric validation (upgrading from `GPS_GEOREFERENCED` to `METRIC_VALIDATED`), the project requires at least one real drone flight dataset containing:
- $\ge 5$ surveyed Ground Control Points (GCPs) measured via RTK-GNSS ($\pm 2\text{ cm}$ accuracy).
- $\ge 3$ independent Check Points measured via total station or RTK-GNSS.
- Recorded camera EXIF metadata with uncompressed imagery.
