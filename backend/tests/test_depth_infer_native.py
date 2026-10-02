"""Real depth-inference preparation — the path every mocked test skipped.

``image2tensor``/``infer_image`` build the model input from the raw frame and
decide whether the returned map is the model's own grid or the source grid.
Both are on the hot path of every run, and BOTH were invisible to the suite:
every depth test patches inference, so a ``NameError`` inside the tensor prep
(a missing numpy import) and a stunted 518-pixel prediction blown up to
3840x2160 (a 17x pixel inflation carrying no extra information) could each
ship while 500+ tests stayed green.

These tests use a stub ``forward`` — no checkpoint, no weights, no model
load — so they pin the contract, not the numerics:

* the prep runs at all, and produces float32 (the memory contract: float64
  doubles the resize traffic on a 4K frame);
* ``native_resolution=True`` returns the MODEL's grid, i.e. the resized
  input size — not the source size;
* the default returns the SOURCE-shaped map it always returned, so existing
  callers are untouched.
"""

import numpy as np
import pytest
import torch

from app.services.depth_anything_v2.dpt import DepthAnythingV2


class _Stub(DepthAnythingV2):
    """DepthAnythingV2 with a fake network: prep + output shape only."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image):  # type: ignore[override]
        # The real head emits patch_tokens**2 at patch 14 px per side; two
        # scales (1/2) are used, so the honest output grid is that of the
        # resized input itself. Shape is what matters here.
        return torch.zeros(1, image.shape[-2], image.shape[-1])


def _frame(h: int = 360, w: int = 640) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_image2tensor_runs_and_is_float32():
    """The prep must execute against a real frame and stay float32."""
    stub = _Stub()
    tensor, (h, w) = DepthAnythingV2.image2tensor(stub, _frame(), 518)

    assert tensor.dtype == torch.float32
    assert tensor.shape[0] == 1 and tensor.shape[1] == 3
    # Source size is reported back for the caller's own output grid.
    assert (h, w) == (360, 640)
    # keep_aspect_ratio + ensure_multiple_of=14: the short side lands on the
    # model's multiple-of-14 grid, not on a raw 518.
    assert tensor.shape[-2] % 14 == 0 and tensor.shape[-1] % 14 == 0
    assert tensor.shape[-2] >= 518


def test_native_resolution_returns_the_model_grid_not_the_source():
    """``native_resolution=True`` must not interpolate back to source size.

    4K footage is the case that matters: the source grid is 8.3 M pixels
    while the prediction is ~0.5 M, and upsampleing the smaller one was the
    single largest consumer of downstream time in this pipeline.
    """
    stub = _Stub()
    frame = _frame(2160, 3840)

    native = DepthAnythingV2.infer_image(stub, frame, 518, native_resolution=True)
    full = DepthAnythingV2.infer_image(stub, frame, 518)

    assert native.shape[0] < 2160 // 4 and native.shape[1] < 3840 // 4, (
        f"native output {native.shape} looks interpolated to the source grid"
    )
    assert full.shape == (2160, 3840)


def test_default_output_shape_is_unchanged():
    """Backwards compatibility: default callers keep the source grid."""
    stub = _Stub()
    out = DepthAnythingV2.infer_image(stub, _frame(240, 320), 518)
    assert out.shape == (240, 320)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("native", [True, False])
def test_infer_image_returns_numpy_float(native):
    stub = _Stub()
    out = DepthAnythingV2.infer_image(stub, _frame(), 518, native_resolution=native)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
