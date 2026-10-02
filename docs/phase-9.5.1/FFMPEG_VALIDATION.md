# FFmpeg Validation (Phase 9.5.1)

## Result

| Item | Value |
|---|---|
| Installed | YES — `imageio-ffmpeg==0.6.0` (bundled static binary) |
| Detected | `imageio_ffmpeg.get_ffmpeg_exe()` resolves to the venv binary |
| Version | `ffmpeg version 7.1` (also `ffprobe` available from the same package) |
| Encode smoke | PASS — generated `testsrc` clip (320×240, 10 fps, 1 s) |
| Decode smoke (ffmpeg) | PASS — full re-decode of the clip |
| Decode smoke (cv2 path) | PASS — `cv2.VideoCapture` opens the clip, 10 frames @ 10 fps, first frame decodes |

## What was executed

- `_run_video_validation` in the functional run (`shitan_ms1_20260909_131251`),
  stage `video: PASS`
- The product's `frame_extractor` / `video_validation` decode path
  (OpenCV `VideoCapture`) reads the FFmpeg-encoded clip correctly

## Honest caveats

- **No original drone MP4/MOV exists in the repository** (`find data -type f
  \( -iname "*.mp4" -o -iname "*.mov" \)` returns nothing). The smoke clip is
  self-generated with FFmpeg itself.
- **Original drone MP4 ingestion: NOT VALIDATED** — explicitly reported as
  `original_drone_mp4_validated: false` in the run manifest, per Phase 9.5.1
  rule 9 (VisDrone-VID frames must not be presented as original drone video).
- A VisDrone-VID MP4 was deliberately NOT manufactured; it would not be
  equivalent to original drone footage.