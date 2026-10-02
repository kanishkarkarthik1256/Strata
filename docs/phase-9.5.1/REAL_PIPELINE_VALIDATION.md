# Real Pipeline Validation (Phase 9.5.1)

## Functional run

- Run ID: **`shitan_ms1_20260909_131251`**
- Input: 10 real DJI FC330 images, `data/shitan_tw` mission `ms1`, 4000×3000
- Status: **PASS** (all 9 stages)
- Outputs: `backend/outputs/shitan_ms1_20260909_131251/`
- Docs: `backend/docs/shitan_ms1_20260909_131251/`

## Stage-by-stage results

| Stage | Status | Result |
|---|---|---|
| environment | PASS | pycolmap 3.12.5, torch 2.2.2, FFmpeg 7.1, no CUDA/MPS |
| dataset | PASS | 145 total ms1 images, 10 selected, GPS in EXIF (lat 24.5309, lon 120.9307, alt 359.5) |
| images | PASS | 10 frames prepared |
| video | PASS | FFmpeg encode+decode smoke; `original_drone_mp4_validated: false` |
| sparse | PASS | COLMAP: 10/10 registered, 2796 points, mean reproj 1.1381 px |
| depth | PASS | stereo SGBM: 7 generated, 2 skipped (weak texture/baseline, frames 4 & 7) |
| depth_learned | PASS | Depth Anything V2: 10/10, device cpu, `metric=false` |
| dense | PASS | 2,626,577 points → `dense_model.ply` (grade Poor — 10-frame single strip, expected) |
| geospatial | PASS | GPS on 10/10 frames, quality score 91.89/100 Excellent, CRS local_enu |

## Artifacts produced

- `sparse/poses.json` — 10 per-view K/R/t entries
- `sparse/sparse_model.ply` — 2796 points
- `depth/*.npy` — 7 stereo depth maps
- `depth_learned/*.npy|.png|.json` — 10 learned depth maps + `learned_depth_info.json`
- `dense/dense_model.ply` — 2,626,577 points (151 MB)
- `geospatial/gps_quality.json` + `crs_metadata.json`
- `manifest.json` — full stage/artifact/metric record (both `outputs/` and `docs/`)

## Pipeline hierarchy in effect

- CAMERA/SFM: **COLMAP (pycolmap)** primary — validated; SIFT/OpenCV fallback intact
- DEPTH: **Stereo SGBM** for the metric dense path; **Depth Anything V2**
  validated separately as relative depth (never fused as meters)
- VIDEO: **FFmpeg** binary primary; image-sequence pipeline intact
- Selection is dynamic at runtime (`pycolmap` import / checkpoint presence /
  FFmpeg binary detection); no capability is assumed without detection.

## Functional checks covered

1. COLMAP actually runs on project data (not just `--version`) — YES
2. Learned depth actually runs — YES (10 views)
3. Dense reconstruction actually works — YES (2.63 M points)
4. GPS processing still works — YES (91.89 Excellent)
5. Output organization still works — YES (matching `outputs/` + `docs/`)
6. Reports generated — YES (README.md, manifest.json, pipeline_report.md)
7. Existing fallbacks remain operational — YES (184/184 regression tests)