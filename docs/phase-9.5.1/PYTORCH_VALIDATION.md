# PyTorch Validation (Phase 9.5.1)

## Result

| Item | Value |
|---|---|
| Version | 2.2.2 |
| CPU | VALIDATED — real tensor ops executed |
| MPS | NOT AVAILABLE (`torch.backends.mps.is_available() == False` — Intel Mac) |
| CUDA | NOT AVAILABLE (`torch.cuda.is_available() == False` — no NVIDIA GPU) |
| NumPy pin | 1.26.4 (NumPy 2.x breaks torch 2.2.2 ABI) |

## What was executed

- `import torch` + version print: OK
- Tensor operation on CPU (create + matmul + reduction): OK
- Depth Anything V2 inference on CPU (see DEPTH_MODEL_VALIDATION.md):
  10 full-resolution (4000×3000) views inferred successfully

## Honest caveats

- This machine (Intel i7, no discrete GPU) has **no accelerated backend**.
  GPU acceleration is not claimed anywhere; the device is reported as `cpu`
  in every run artifact (`learned_depth_info.json` → `device: cpu`).
- `pyproject.toml` declares `torch>=2.5.0`, but no torch ≥2.5 wheel exists
  for Python 3.9; the working venv runs torch 2.2.2. Pre-existing mismatch,
  documented here rather than silently upgraded.