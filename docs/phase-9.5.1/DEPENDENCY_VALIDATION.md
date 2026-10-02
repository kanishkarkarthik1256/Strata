# Dependency Validation (Phase 9.5.1)

Status levels used throughout (per Phase 9.5.1 rule 25):

- **NOT INSTALLED** — not present on the system
- **INSTALLED** — package present
- **DETECTED** — import/shutil lookup succeeds
- **SMOKE-TESTED** — ran a real operation on real data once
- **USED** — exercised inside the real pipeline run
- **VALIDATED** — performed its intended operation on real project data
  (the strongest level claimed here)

| Dependency | Installed | Detected | Smoke-tested | Used | Validated |
|---|---|---|---|---|---|
| COLMAP (pycolmap 3.12.5) | YES | YES | YES | YES | YES |
| COLMAP CLI binary | NO | — | — | — | NO |
| PyTorch 2.2.2 (CPU) | YES | YES | YES | YES | YES |
| Depth Anything V2 (ViT-Small) | YES (checkpoint) | YES | YES | YES | YES |
| FFmpeg 7.1 (imageio-ffmpeg) | YES | YES | YES | YES (encode/decode smoke) | YES |
| OpenCV 5.0.0 (SIFT/SGBM fallback) | YES | YES | YES | YES | YES |
| OpenMVS | NO | — | — | — | NO |

## Explicit distinctions

- **COLMAP is validated via the pycolmap wheel**, not the `colmap` CLI.
  The Homebrew `colmap` build timed out repeatedly and was abandoned; the
  wheel bundles the real COLMAP 3.12 engine and ran the full
  extract → match → incremental-map pipeline on real Shitan images.
- **FFmpeg is validated via the imageio-ffmpeg static binary**, not a
  Homebrew install. `ffmpeg -version` reports `ffmpeg version 7.1`.
- **Learned depth is validated as relative depth.** The Depth Anything V2
  outputs are explicitly tagged `metric: false` with scale `NOT_VALIDATED`;
  they are never fused into the metric dense cloud.
- **Original drone MP4 ingestion is NOT VALIDATED**: no `.mp4`/`.mov`
  exists anywhere under `data/`. The video stage validates FFmpeg
  encode+decode with a self-generated testsrc clip only, and reports
  `original_drone_mp4_validated: false`.

## Install commands used (for reproducibility)

```bash
# COLMAP engine (Python wheel — real COLMAP 3.12)
pip install pycolmap==3.12.5

# PyTorch CPU-only (this machine has no CUDA/MPS)
pip install torch==2.2.2
pip install "numpy==1.26.4"   # pin — torch 2.2.2 is incompatible with NumPy 2.x

# FFmpeg binary
pip install imageio-ffmpeg==0.6.0

# Depth Anything V2 checkpoint (official, not gated)
pip install huggingface_hub
# downloaded models/weights/depth_anything_v2_vits.pth (98 MB, official repo)
```