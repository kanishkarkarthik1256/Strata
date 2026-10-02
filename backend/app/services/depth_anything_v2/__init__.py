"""Depth Anything V2 — vendored inference stack (``vits`` variant).

Source: https://github.com/DepthAnything/Depth-Anything-V2

- ``dpt.py`` / ``blocks.py`` / ``transform.py`` are MIT-licensed code from
  the Depth-Anything-V2 repository, trimmed to inference-only behaviour.
- ``dinov2.py`` / ``dinov2_layers.py`` are the Apache-2.0 DINOv2 backbone
  (facebookresearch/dinov2) used as the pretrained encoder.

The stack is vendored so real inference can run with just ``torch`` (no
``torchvision``/``transformers``). Supported encoders: ``vits``,
``vitb`` (default when present) and ``vitl``; ``vitg`` raises
``NotImplementedError``. When the configured variant's checkpoint is
missing, resolution falls back to whichever checkpoint exists — fresh
checkouts without vitb weights keep running on vits.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.depth_anything_v2.dpt import DepthAnythingV2

log = get_logger("drone_recon.services.depth_anything_v2")

CHECKPOINT_PATTERNS = ("depth_anything_v2_*.pth", "depth_anything_*.pth")
SUPPORTED_ENCODERS = ("vits", "vitb", "vitl")


def checkpoint_encoder(name: str) -> str:
    """Encoder variant a checkpoint filename carries (…_vits.pth → 'vits').

    Unknown naming falls back to ``vits`` so legacy layouts keep working.
    """
    m = re.search(r"depth_anything(?:_v2)?_(vit[sblg])\.pth$", name)
    return m.group(1) if m else "vits"


def _configured_encoder() -> str:
    """Encoder requested via settings (AI_DEPTH_ENCODER); falls back safely."""
    enc = getattr(settings.ai, "depth_encoder", "vits") or "vits"
    return enc if enc in SUPPORTED_ENCODERS else "vits"


def find_checkpoint(weights_dir: Path | None = None, encoder: str | None = None) -> Path | None:
    """Locate a Depth Anything V2 checkpoint for *encoder* under the weights dir.

    Defaults to the configured encoder (AI_DEPTH_ENCODER, default vits) so
    a newly downloaded variant never silently changes production behaviour.
    """
    enc = encoder or _configured_encoder()
    candidates = []
    if weights_dir is not None:
        candidates.append(weights_dir)
    else:
        candidates.append(settings.ai.weights_path)
        candidates.append(Path("backend/models/weights"))
        candidates.append(Path(__file__).resolve().parents[2] / "models" / "weights")

    for base in candidates:
        if not base.is_dir():
            continue
        # Exact variant first (depth_anything_v2_vitb.pth), then any
        # checkpoint whose filename names the encoder, then any checkpoint
        # at all (un-named legacy files are assumed to match the default).
        named = [p for pattern in CHECKPOINT_PATTERNS for p in sorted(base.glob(pattern))]
        for hits in (
            [base / f"depth_anything_v2_{enc}.pth"],
            [p for p in named if checkpoint_encoder(p.name) == enc],
            named,
        ):
            for hit in hits:
                if hit.is_file():
                    return hit
    return None

def detect_device() -> str:
    """Best inference device: cuda > mps > cpu (mps only on Apple Silicon)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def resolve_device(requested: str | None = None) -> str:
    """Return a device supported by this host, never a speculative accelerator."""
    requested = (requested or "auto").lower()
    if requested in {"auto", ""}:
        return detect_device()
    if requested == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        log.warning("depth_device_fallback", requested="cuda", selected="cpu")
        return "cpu"
    if requested == "mps":
        try:
            import torch
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        log.warning("depth_device_fallback", requested="mps", selected="cpu")
        return "cpu"
    if requested == "cpu":
        return "cpu"
    log.warning("depth_device_fallback", requested=requested, selected="cpu")
    return "cpu"


_model: DepthAnythingV2 | None = None
_model_device: str = ""
_model_path: str = ""


def load_model(checkpoint: Path | None = None, device: str | None = None) -> tuple[DepthAnythingV2, str, str]:
    """Load a Depth Anything V2 model (cached after first call).

    The encoder is read from the checkpoint filename (``…_vits.pth`` →
    vits, ``…_vitb.pth`` → vitb), so the weights file on disk is the single
    source of truth; ``find_checkpoint`` resolves the variant that
    AI_DEPTH_ENCODER (default vits) asks for.
    ``(model, device, checkpoint_path)``. Raises ``FileNotFoundError``
    with install guidance when no checkpoint is present — callers must not
    treat this as "no model available" without checking ``find_checkpoint``
    first, because a missing checkpoint is a configuration error, not an
    optional fallback.
    """
    global _model, _model_device, _model_path
    if _model is not None:
        return _model, _model_device, _model_path

    import torch

    ckpt = checkpoint or find_checkpoint()
    if ckpt is None or not ckpt.is_file():
        raise FileNotFoundError(
            "No Depth Anything V2 checkpoint found — place depth_anything_v2_vits.pth "
            f"in {settings.ai.weights_path} (AI_WEIGHTS_DIR)"
        )
    encoder = checkpoint_encoder(ckpt.name)
    dev = resolve_device(device)
    log.info("depth_anything_loading", checkpoint=str(ckpt), encoder=encoder, device=dev)

    import cv2

    import torch

    from app.services.cpu_budget import inference_threads

    # Thread budget: the physical core count, not the logical one. This ViT is
    # throughput-bound, and its default (all logical cores) oversubscribed the
    # SIMD units — measured 8.50-9.05 s/frame at 8 threads vs 5.55-5.66 s/frame
    # at 4 on the reference 4-physical/8-logical host. See cpu_budget.
    threads = inference_threads()
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    # OpenCV helpers (resize in transform.py) follow the same limit, so they
    # cannot compete with the encoder for the same cores.
    cv2.setNumThreads(threads)

    model = DepthAnythingV2(encoder=encoder)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint {ckpt.name} missing keys: {missing[:8]}")
    if unexpected:
        log.warning("depth_anything_checkpoint_unused_keys", keys=[str(k) for k in unexpected[:8]])
    model.eval()
    model.to(dev)

    _model, _model_device, _model_path = model, dev, str(ckpt)
    return model, dev, str(ckpt)
