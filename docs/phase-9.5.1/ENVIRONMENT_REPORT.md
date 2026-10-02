# Environment Report (Phase 9.5.1)

Audited 2026-09-09 on the development machine. All values measured.

| Item | Value |
|---|---|
| OS | macOS 26.3.1 (build 25D771280a) |
| Architecture | x86_64 |
| CPU | Intel Core i7-1068NG7 @ 2.30 GHz |
| RAM | 32 GB |
| GPU | Intel Iris Plus (integrated, ~1.5 GB, Metal 3) |
| CUDA | NOT AVAILABLE (no NVIDIA GPU) |
| MPS | NOT AVAILABLE (Intel GPU — MPS requires Apple Silicon) |
| Python | 3.9.6 (`.venv`) |
| pip | 26.0.1 |
| OpenCV | 5.0.0 (opencv-python-headless) |
| NumPy | 1.26.4 (pinned — NumPy 2.x incompatible with torch 2.2.2) |
| SciPy | 1.13.1 |
| PyTorch | 2.2.2 |
| pycolmap | 3.12.5 (COLMAP 3.12 engine, Python wheel) |
| imageio-ffmpeg | 0.6.0 (bundles FFmpeg 7.1 static binary) |
| huggingface_hub | installed (checkpoint download) |
| Open3D | NOT INSTALLED |
| CLI `colmap` binary | NOT INSTALLED (Homebrew source build did not complete; pycolmap wheel used instead) |
| CLI `ffmpeg` on PATH | NO — binary provided via `imageio_ffmpeg.get_ffmpeg_exe()` |

## Notes

- The declared `pyproject.toml` dependencies (`torch>=2.5.0`, `open3d>=0.18.0`)
  do not match the working Python 3.9 venv (torch 2.2.2, no open3d). This is a
  pre-existing mismatch; torch 2.5+ wheels do not exist for Python 3.9.
  The venv is the runtime used by every test and pipeline run in this phase.
- All paths are resolved dynamically (venv, `imageio_ffmpeg.get_ffmpeg_exe()`);
  no hard-coded machine paths were added.