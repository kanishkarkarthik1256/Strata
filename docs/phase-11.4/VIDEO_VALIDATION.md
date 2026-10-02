# STRATA Phase 11.4 — Video Ingestion & Metadata Source Validation

## Overview
This document details the video ingestion testing and metadata extraction analysis in STRATA.

## Video Pipeline Status

> [!NOTE]
> **VIDEO PIPELINE VALIDATED**
> Technical streaming ingestion, chunked upload, format allowlist checking, FFprobe/OpenCV metadata extraction, and pipeline trigger have been fully validated.

> [!WARNING]
> **REAL DRONE VIDEO VALIDATION: NOT YET VALIDATED**
> The project repository currently contains synthetic/test video files (`test_drone.avi`, 320x240 generated AVI files) used for technical pipeline verification. No actual 4K/HD physical drone flight footage (e.g. DJI MOV/MP4 with embedded RTK GPS) is present in the repository.

## Video Metadata Source Breakdown

Metadata extraction in `metadata_extraction.py` derives attributes from specific authoritative sources:

| Attribute | Authoritative Source | Fallback Mechanism |
|-----------|---------------------|--------------------|
| `duration_sec` | FFprobe format container duration (`format.duration`) | OpenCV `frame_count / fps` calculation |
| `width`, `height` | OpenCV `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT` | FFprobe video stream dimensions |
| `fps` | OpenCV `CAP_PROP_FPS` | FFprobe stream frame rate |
| `codec` | FFprobe video stream codec name (`stream.codec_name`) | OpenCV FourCC decoding (`CAP_PROP_FOURCC`) |
| `bitrate_kbps` | FFprobe format bitrate (`format.bit_rate`) | Calculated `(size_bytes * 8) / (duration * 1000)` |
| `creation_time` | FFprobe format tags (`creation_time`) | `None` |
| `camera_make/model` | FFprobe QuickTime/ISO tags (`com.apple.quicktime.make`, `make`, `model`) | `None` |
| `gps_lat/lon/alt` | FFprobe DJI container tags (`location`), QuickTime ISO tags | `None` (Prohibits fake 0,0) |

> [!IMPORTANT]
> **Terminology Policy**: Standard MP4/MOV container tags and FFprobe metadata are documented as **Container Metadata**, NOT **EXIF**, unless the metadata stream explicitly contains TIFF EXIF / XMP headers.
