# Known Limitations

Generated: 2026-09-06

## Critical Limitations

### 1. External Tool Dependencies (NOT INSTALLED)

The following external tools are required but not installed:

| Tool | Purpose | Impact |
|------|---------|--------|
| COLMAP | SfM, camera pose estimation, sparse reconstruction | Cannot validate Phases 3-5 |
| OpenMVS | Dense reconstruction, meshing | Cannot validate Phase 5 |
| FFmpeg | Video frame extraction | Cannot test video ingestion |

**Recommendation**: Install COLMAP and OpenMVS for full pipeline validation.

### 2. Missing Dataset Information

| Missing Data | Impact | Affected Phases |
|--------------|--------|-----------------|
| Video files | Cannot test frame extraction | Phases 1-2 |
| Camera calibration | Cannot validate metric accuracy | Phases 3-5 |
| Ground truth 3D | Cannot validate reconstruction accuracy | Phases 5-7 |
| Altitude data | Cannot validate georeferencing | Phase 7 |
| Timestamps | Cannot validate temporal accuracy | All phases |

### 3. AI Models Not Available

| Model | Purpose | Impact |
|-------|---------|--------|
| Depth Anything V2 | Depth estimation | Cannot validate depth generation |
| Grounding DINO | Open-vocabulary detection | Limited object detection |
| SAM 2 | Segmentation | Limited scene understanding |
| FAISS | Vector search | Limited RAG capabilities |
| Sentence Transformers | Embeddings | Limited text understanding |

## Minor Limitations

### 1. Video Writing Issues
OpenCV 5.0 has compatibility issues with certain video codecs on this platform. Synthetic test videos cannot be created using the standard test fixture.

### 2. GPS Data Limitations
- Aukerman: GPS coordinates present but no altitude
- Shitan TW: GPS coordinates present but no altitude
- VisDrone: No GPS data at all

### 3. Resolution Inconsistency
- VisDrone: Variable (1360×765 typical)
- Aukerman: 4896×3672 (Sony)
- Shitan TW: 4000×3000 (DJI)

Cross-dataset testing may produce inconsistent results due to resolution differences.

## What Cannot Be Validated

1. **Metric reconstruction accuracy** - Requires camera calibration + ground truth
2. **GPS georeferencing accuracy** - Requires altitude data
3. **Change detection between missions** - Requires reconstruction first
4. **Depth estimation accuracy** - Requires AI models or stereo pairs
5. **Object detection accuracy** - Requires AI models (YOLO, Grounding DINO)
6. **Semantic segmentation accuracy** - Requires AI models (SAM 2)

## Recommendations for Full Validation

1. **Install external tools**: COLMAP, OpenMVS, FFmpeg
2. **Add calibration data**: Camera intrinsics for each dataset
3. **Add ground truth**: Reference point clouds or meshes
4. **Add video dataset**: Short drone video for frame extraction testing
5. **Install AI models**: Depth Anything V2, Grounding DINO, SAM 2
6. **Add altitude data**: GPS altitude or barometric data for georeferencing

## Current Capability Matrix

| Capability | Status | Validation Level |
|------------|--------|------------------|
| Image quality analysis | WORKS | Real dataset validated |
| Environmental detection | WORKS | Real dataset validated |
| Object detection (classical) | WORKS | Real dataset validated |
| Mission simulation | WORKS | Synthetic data validated |
| Battery estimation | WORKS | Synthetic data validated |
| Path optimization | WORKS | Synthetic data validated |
| Risk assessment | WORKS | Synthetic data validated |
| Mission history | WORKS | Synthetic data validated |
| Report generation | WORKS | Synthetic data validated |
| SfM/camera pose | REQUIRES COLMAP | Not validated |
| Sparse reconstruction | REQUIRES COLMAP | Not validated |
| Dense reconstruction | REQUIRES OpenMVS | Not validated |
| Depth estimation | REQUIRES AI MODEL | Not validated |
| Metric accuracy | REQUIRES CALIBRATION | Not validated |
