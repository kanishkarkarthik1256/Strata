# STRATA Metric Pipeline Audit

**Phase 13 Document — Scale Origin, Georeferencing Flow, and Coordinate Transformation Analysis**

---

## Executive Summary

This document provides a comprehensive scientific audit of the metric scale derivation, coordinate transformations, and georeferencing mechanisms currently implemented within the STRATA reconstruction engine. 

### Key Audit Finding
STRATA's current reconstruction pipeline achieves scale and georeferencing via a global least-squares similarity transformation (Umeyama fit) between arbitrary Structure from Motion (SfM) camera positions and local East-North-Up (ENU) coordinates derived from WGS84 GPS metadata. 

However, prior to Phase 13:
1. **No Ground Control Points (GCPs) or Check Points** were formally ingested or evaluated.
2. **No distinction** was made between *estimated GPS alignment* and *surveyed metric validation*.
3. **No explicit Capability State Machine** existed to enforce honest UI representations.
4. **Depth Anything V2 outputs** were treated as relative depth maps without explicit calibration labeling.

---

## 1. End-to-End Metric Data Flow

```
Input Video/Image
   │
   ├─► Metadata Extraction (metadata_extraction.py)
   │      └─ extract ffprobe GPS (lat, lon, alt, creation_time, make/model)
   │
   ├─► Frame Extraction (frame_extractor.py)
   │      └─ Save JPEGs & attach EXIF/JSON metadata per frame
   │
   ├─► Sparse Reconstruction (camera_pose_estimator.py)
   │      └─ PyCOLMAP / OpenCV SfM → Arbitrary camera poses C_colmap = -R^T t
   │
   ├─► Metric Scale Estimation (metric_scale_validator.py)
   │      └─ WGS84 (lat, lon, alt) → ECEF → Local ENU (wgs84_to_enu)
   │      └─ Umeyama 3D similarity fit: C_enu ≈ s * R * C_colmap + t
   │      └─ Global scale factor s = metres / COLMAP_unit
   │
   ├─► Dense Reconstruction (dense_reconstruction.py)
   │      └─ Multi-view Stereo / Depth Anything V2 (relative depth)
   │      └─ Dense point cloud saved in local/scaled coordinates
   │
   └─► Geospatial Alignment (geo_alignment.py / pipeline_orchestrator.py)
          └─ Apply 4x4 matrix to PLY vertices & GeoJSON footprints
          └─ Output enu_mesh.ply, objects.geojson, crs.json
```

---

## 2. Mathematical Coordinate Transformations

### 2.1 Geodetic to ECEF (WGS84 Ellipsoid)
Given latitude $\phi$, longitude $\lambda$, and ellipsoidal height $h$, position in Earth-Centered Earth-Fixed (ECEF) coordinates is computed using the WGS84 ellipsoid parameters ($a = 6378137.0\text{ m}$, $f = 1/298.257223563$, $e^2 = f(2-f)$):

$$N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}$$

$$X_{ECEF} = (N(\phi) + h) \cos\phi \cos\lambda$$
$$Y_{ECEF} = (N(\phi) + h) \cos\phi \sin\lambda$$
$$Z_{ECEF} = (N(\phi)(1 - e^2) + h) \sin\phi$$

### 2.2 ECEF to Local ENU
Using anchor point $(\phi_0, \lambda_0, h_0)$ with ECEF coordinates $P_{\text{anchor}}$, any ECEF point $P$ is rotated into the local tangent plane:

$$\begin{bmatrix} E \\ N \\ U \end{bmatrix} = \begin{bmatrix} -\sin\lambda_0 & \cos\lambda_0 & 0 \\ -\sin\phi_0\cos\lambda_0 & -\sin\phi_0\sin\lambda_0 & \cos\phi_0 \\ \cos\phi_0\cos\lambda_0 & \cos\phi_0\sin\lambda_0 & \sin\phi_0 \end{bmatrix} (P - P_{\text{anchor}})$$

### 2.3 Umeyama Similarity Scale Fit (COLMAP $\rightarrow$ ENU)
For camera centers in SfM space $S \in \mathbb{R}^{N \times 3}$ and ENU space $T \in \mathbb{R}^{N \times 3}$:

$$s = \frac{\text{Tr}(D S^T T_c)}{\sum \|S_c\|^2}$$

where $S_c$ and $T_c$ are mean-centered coordinates, and $D = \text{diag}(1, 1, \det(UV^T))$ handles reflections via SVD of covariance.

---

## 3. Detailed Parameter & Convention Audit Table

| Parameter / Dimension | Specification in STRATA | Notes & Verification |
| :--- | :--- | :--- |
| **Primary Scale Origin** | GPS camera baseline alignment | Scale $s$ derived from camera motion against GPS |
| **Secondary Scale Origin** | Known object/distance or flight altitude | Fallback when camera GPS baselines are small ($<5\text{ m}$) |
| **Ellipsoid / Datum** | WGS84 ($\text{EPSG:4326}$) | Standard consumer drone GPS format |
| **Local Frame** | Local ENU (East-North-Up) | Anchor set to first valid GPS camera fix |
| **Projected Frame** | UTM via `pyproj` (Optional) | EPSG automatically selected by mean longitude |
| **Linear Units** | Metres ($\text{m}$) | All aligned point clouds, meshes, and volumes in meters |
| **Camera Pose Convention** | $X_{\text{cam}} = R \cdot X_{\text{world}} + t$ | Camera center $C_{\text{world}} = -R^T t$ |
| **Matrix Convention** | Row-vector post-multiplication ($v' = v M^T$) | Consistent across mesh transformation modules |
| **Altitude Source** | Barometric / GPS altitude | Uncalibrated consumer GPS altitude ($\pm 5\text{--}15\text{ m}$ uncertainty) |
| **Camera Intrinsics** | SIFT EXIF focal length / PINHOLE model | Estimated by PyCOLMAP bundle adjustment |
| **RTK / PPK Support** | Supported via metadata fields | Precision weighting applied when RTK flags present |
| **GCP / Check Point Support** | Added in Phase 13 | Separate Control Points from Check Points |

---

## 4. Identified Metric Vulnerabilities & Remediation Plan

1. **Unvalidated Accuracy Claims**:
   - *Issue*: Displaying arbitrary confidence scores as "accuracy".
   - *Fix*: Define explicit capability states (`RELATIVE_ONLY`, `SCALED`, `GPS_GEOREFERENCED`, `METRIC_VALIDATED`). Display `"Metric Accuracy: Not Validated"` unless independent check points or reference 3D data are validated.

2. **GPS `0.0, 0.0` Hazard**:
   - *Issue*: Corrupted or missing GPS metadata resolving to `(0, 0, 0)`.
   - *Fix*: Add strict coordinate validation in `georeferencing.py` rejecting `(0,0)` and flagging `gps_available = false`.

3. **Depth Anything V2 Scale Ambiguity**:
   - *Issue*: Deep depth models output relative scale-free depth maps.
   - *Fix*: Annotate Depth Anything V2 outputs explicitly as `"Relative depth model"` unless calibrated against metric stereo/sparse points.

4. **Control vs Check Point Mixing**:
   - *Issue*: Re-using control points for accuracy reporting overestimates accuracy.
   - *Fix*: Implement strict separation: Control points influence alignment; Check points are isolated for error calculation only.
