# STRATA Phase 11.5 — Native Binary & Dependency Policy

## Overview
This document specifies external binary dependencies, discovery mechanisms, and end-user distribution policies in STRATA.

## Native Dependencies Breakdown

| Dependency | Purpose | Implementation Path | Distribution Policy |
|------------|---------|---------------------|---------------------|
| **pycolmap** | SfM Camera Pose Estimation | In-process C++ Python bindings (`import pycolmap`) | Included in Python backend bundle. |
| **OpenCV** | Video validation & image frame extraction | In-process C++ Python bindings (`import cv2`) | Included in Python backend bundle. |
| **FFmpeg / FFprobe** | Video container metadata & stream inspection | System binary or bundled sidecar | Attempt `ffprobe` in PATH; fallback to OpenCV if missing. |
| **COLMAP CLI** | Optional external SfM CLI | Optional system path lookup | **NOT MANDATORY**. pycolmap is the primary validated engine. |

> [!IMPORTANT]
> **No System Package Managers**: End users MUST NOT be instructed or required to run `brew`, `apt`, `choco`, or `pip` to use the STRATA desktop application. All required dependencies are bundled or handled via fallback mechanisms.
