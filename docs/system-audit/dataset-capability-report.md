# Dataset Capability Report

Generated: 2026-09-06

## Summary

| Dataset | Video | Images | GPS | Camera Calibration | 3D Ground Truth | Annotations | Suitable Tests |
|---------|-------|--------|-----|--------------------|-----------------|-------------|----------------|
| VisDrone (test-dev) | No | 10,209 JPG | No | No | No | Yes (8,629 bounding boxes) | Frame quality, object detection |
| Aukerman | No | 77 JPG | Yes (lat/lon) | No | No (has orthomosaic PNG) | No | GPS alignment, basic reconstruction |
| Shitan TW | No | 493 JPG (4 missions) | Yes (lat/lon) | No | No | No | Multi-mission, GPS georeferencing |

## Detailed Capabilities

### VisDrone (VisDrone2019-DET-test-dev)

- **Location**: `data/Visdrone/VisDrone2019-DET-test-dev/`
- **Content**: 10,209 aerial drone images from urban environments
- **Resolution**: Variable (1360×765 typical, some 1400×1050)
- **Camera**: Various drone platforms (no DJI metadata)
- **GPS**: Not embedded in images
- **Annotations**: CSV format (x,y,w,h,ignored,class,truncation,occlusion)
  - 10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
- **Use Case**: Frame quality analysis, blur detection, environmental intelligence, object detection validation
- **Limitation**: No video (cannot test frame extraction from video), no GPS (cannot test georeferencing)

### Aukerman

- **Location**: `data/aukerman/`
- **Content**: 77 aerial images from Ohio, USA (Aukerman Farm area)
- **Resolution**: 4896×3672 (Sony camera)
- **Camera**: Sony DSC series
- **GPS**: Yes (lat: ~41.304°, lon: ~-81.750°, no altitude)
- **Annotations**: None (has orthomosaic PNG for reference)
- **Use Case**: GPS alignment, basic SfM, dense reconstruction validation
- **Limitation**: Small dataset (77 images), no camera calibration data, no ground truth 3D

### Shitan TW

- **Location**: `data/shitan_tw/`
- **Content**: 493 aerial images from Shitan, Miaoli, Taiwan (2018-12-21 survey)
- **Resolution**: 4000×3000 (DJI drone)
- **Camera**: DJI drone platform
- **GPS**: Yes (lat: ~24.531°, lon: ~120.931°, no altitude)
- **Missions**: 4 separate flight missions (ms1: 145, ms2: 155, ms3: 92, ms4: 101 images)
- **Annotations**: None
- **Use Case**: Multi-mission comparison, GPS georeferencing, change detection between missions
- **Limitation**: No camera calibration, no ground truth 3D, no altitude data

## Missing Capabilities

### Not Available in Any Dataset

1. **Video files** - All datasets are image sequences, not video
2. **Camera calibration/intrinsics** - No focal length, principal point, distortion coefficients
3. **3D ground truth** - No reference point clouds or meshes for accuracy validation
4. **RTK/PPK data** - No high-precision GPS corrections
5. **IMU data** - No inertial measurement data
6. **Altitude data** - No GPS altitude or barometric data
7. **Timestamps** - No image capture timestamps
8. **Segmentation masks** - No pixel-level annotations

## Recommendations

1. **For video ingestion tests**: Use synthetic video or create video from image sequences
2. **For camera calibration**: Use default DJI/survey camera parameters
3. **For 3D validation**: Compare against orthomosaic (Aukerman) or qualitative assessment
4. **For GPS tests**: Use Aukerman and Shitan datasets (both have coordinates)
5. **For multi-mission tests**: Use Shitan TW (4 separate missions)
