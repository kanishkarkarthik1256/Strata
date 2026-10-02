#!/usr/bin/env bash
set -euo pipefail

# DroneRecon Backend — Docker Entrypoint
# Ensures data directories exist, verifies the reconstruction engine, then
# starts the FastAPI server.

echo "============================================"
echo "  DroneRecon Backend — Starting"
echo "============================================"

# Ensure writable data directories
mkdir -p /app/data/storage
mkdir -p /app/models/weights

# Verify the reconstruction engine. The CLI binary is OPTIONAL: the pipeline
# calls pycolmap — the COLMAP library's own Python bindings — in-process, so a
# missing CLI does not stop reconstruction (only a missing pycolmap degrades
# SfM to the OpenCV fallback, which the import check below reports).
if command -v colmap &>/dev/null; then
    echo "[OK] COLMAP CLI found at: $(which colmap)"
else
    echo "[INFO] COLMAP CLI not on PATH — pycolmap (embedded engine) is the reconstruction path"
fi

# Verify Python imports
python -c "
import sys
print(f'Python {sys.version}')
import fastapi; print(f'[OK] FastAPI {fastapi.__version__}')
import torch; print(f'[OK] PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
try:
    import pycolmap; print(f'[OK] pycolmap {pycolmap.__version__} (embedded COLMAP engine)')
except ImportError:
    print('[WARN] pycolmap missing — SfM would fall back to OpenCV')
import cv2; print(f'[OK] OpenCV {cv2.__version__}')
import numpy; print(f'[OK] NumPy {numpy.__version__}')
"

echo "============================================"
echo "  Starting uvicorn..."
echo "============================================"

exec python -m uvicorn \
    app.main:create_app \
    --factory \
    --host "${SERVER_HOST:-0.0.0.0}" \
    --port "${SERVER_PORT:-8000}" \
    --log-level "${SERVER_LOG_LEVEL:-info}"
