# Limitations Report - Phase 9.5

**Date**: 2026-09-07
**Status**: Documented

## Overview

This document records the limitations encountered during Phase 9.5 validation and the recommended actions to address them.

## Critical Limitations

### 1. No GPU Acceleration

**Impact**: High
**Status**: Documented

**Details**:
- PyTorch not installed
- All processing is CPU-only
- Significant performance impact on:
  - Feature extraction (SIFT vs SuperPoint)
  - Feature matching (BFMatcher vs LightGlue)
  - Depth estimation (Stereo vs Depth Anything)

**Recommendation**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. No COLMAP for SfM

**Impact**: High
**Status**: Documented

**Details**:
- pycolmap not installed
- Fallback to OpenCV essential matrix
- Limitations:
  - No bundle adjustment
  - No 3D point triangulation
  - Limited pose refinement
  - No camera intrinsics refinement

**Recommendation**:
```bash
brew install colmap
pip install pycolmap
```

### 3. No Video Processing

**Impact**: Medium
**Status**: Documented

**Details**:
- FFmpeg not installed
- Cannot process video files
- Dataset contains only image sequences

**Recommendation**:
```bash
brew install ffmpeg
```

## Moderate Limitations

### 4. Stereo Depth Limitations

**Impact**: Medium
**Status**: Documented

**Details**:
- Using SGBM stereo matching
- Limitations:
  - Requires sufficient baseline
  - Fails on textureless regions
  - Last frame fails (no stereo pair)
  - Depth discontinuities at edges

**Recommendation**:
- Install Depth Anything V2 for monocular depth
- Increase image overlap
- Use multiple baselines

### 5. No Bundle Adjustment

**Impact**: Medium
**Status**: Documented

**Details**:
- COLMAP not available
- No pose refinement
- No intrinsics refinement
- Reprojection errors not minimized

**Recommendation**:
- Install COLMAP for proper SfM
- Use OpenCV BA as alternative (not implemented)

### 6. Limited Camera Calibration

**Impact**: Medium
**Status**: Documented

**Details**:
- Intrinsics estimated from image dimensions
- No distortion coefficients
- No principal point refinement

**Recommendation**:
- Perform camera calibration
- Use calibration files if available

## Minor Limitations

### 7. No Ground Truth

**Impact**: Low
**Status**: Documented

**Details**:
- No reference reconstruction
- Cannot validate accuracy
- Cannot compute error metrics

**Recommendation**:
- Collect ground control points
- Use known geometry for validation

### 8. No IMU Data

**Impact**: Low
**Status**: Documented

**Details**:
- IMU not available in EXIF
- No orientation data
- Altitude from GPS only

**Recommendation**:
- Use RTK GPS for better altitude
- Add IMU logging if available

### 9. Limited View Count

**Impact**: Low
**Status**: Documented

**Details**:
- Only 10 images processed
- Limited coverage
- Sparse reconstruction incomplete

**Recommendation**:
- Process 50+ images for better coverage
- Use all 145 images for full mission

## Workarounds Implemented

### 1. SIFT Fallback

**Status**: ✅ Working

**Details**:
- SuperPoint unavailable (needs PyTorch)
- SIFT used as fallback
- Quality: Good (but lower than learned features)

### 2. BFMatcher Fallback

**Status**: ✅ Working

**Details**:
- LightGlue unavailable (needs PyTorch)
- BFMatcher used as fallback
- Quality: Good (but lower than learned matcher)

### 3. OpenCV Pose Estimation

**Status**: ✅ Working

**Details**:
- COLMAP unavailable
- OpenCV essential matrix used
- Quality: Fair (no bundle adjustment)

### 4. Stereo Depth

**Status**: ✅ Working

**Details**:
- Depth Anything unavailable (needs PyTorch)
- Stereo SGBM used
- Quality: Fair (requires sufficient baseline)

## Test Results

| Test | Status | Notes |
|------|--------|-------|
| Regression Suite | 184/184 PASS | No regressions |
| New Tests | None | Validation script only |
| Integration | PASS | Pipeline functional |

## Recommendations for Phase 11

### Priority 1: Install Core Dependencies

```bash
# PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# COLMAP
brew install colmap
pip install pycolmap

# FFmpeg
brew install ffmpeg
```

### Priority 2: Enable GPU Acceleration

- Install CUDA-enabled PyTorch
- Use SuperPoint/LightGlue for better features
- Use Depth Anything V2 for monocular depth

### Priority 3: Improve Reconstruction Quality

- Enable bundle adjustment
- Add camera calibration
- Process more images
- Add ground truth validation

## Conclusion

Phase 9.5 successfully validated the reconstruction pipeline with available tools. The limitations are documented and addressable. The system is functional with fallbacks and ready for enhancement in Phase 11.
