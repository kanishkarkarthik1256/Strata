"""Depth-model identity — one variant's cache must never serve another.

The depth cache previously keyed only on poses.json, so a second Depth
Anything V2 checkpoint (vitb) on disk could read the small model's maps
(or vice versa): same filenames, different noise and resolution. These
tests pin the identity contract:

1. A sidecar carrying ``model_sha256`` matches only the same checkpoint.
2. A sidecar carrying only ``model_version`` matches by filename.
3. A legacy sidecar (neither field) is compatible ONLY with the vits
   checkpoint — every legacy map was produced by the small model, so a
   vitb run must never serve one.
4. Checkpoint resolution honours the configured encoder (AI_DEPTH_ENCODER)
   and never changes variant just because another .pth appeared on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import depth_generator as dg
from app.services.depth_anything_v2 import checkpoint_encoder, find_checkpoint

# Reuse the sparse-anchor rig (poses + images + sparse cloud seeding).
from tests.test_sparse_anchor_gate import (  # noqa: E402
    _ground_cloud,
    _patch_infer,
    _raw_map,
    _seed,
)


def _sidecar(ws: Path, fid: str, **fields) -> None:
    meta = {"frame_id": fid, "poses_sha256": dg._poses_fingerprint(ws / "poses.json")}
    meta.update(fields)
    (ws / "depth" / f"{fid}.json").write_text(json.dumps(meta))


class TestCheckpointResolution:
    def test_encoder_read_from_filename(self):
        assert checkpoint_encoder("depth_anything_v2_vitb.pth") == "vitb"
        assert checkpoint_encoder("depth_anything_v2_vits.pth") == "vits"
        assert checkpoint_encoder("depth_anything.pth") == "vits"  # legacy name

    def test_find_checkpoint_prefers_configured_encoder(self, tmp_path, monkeypatch):
        (tmp_path / "depth_anything_v2_vits.pth").write_bytes(b"x")
        (tmp_path / "depth_anything_v2_vitb.pth").write_bytes(b"x")
        assert find_checkpoint(weights_dir=tmp_path, encoder="vits").name.endswith("vits.pth")
        assert find_checkpoint(weights_dir=tmp_path, encoder="vitb").name.endswith("vitb.pth")

    def test_find_checkpoint_falls_back_to_any_when_variant_missing(self, tmp_path):
        (tmp_path / "depth_anything_v2_vits.pth").write_bytes(b"x")
        # vitb weights absent: fall through to whatever exists rather than None.
        assert find_checkpoint(weights_dir=tmp_path, encoder="vitb") is not None

    def test_find_checkpoint_none_when_dir_empty(self, tmp_path):
        assert find_checkpoint(weights_dir=tmp_path, encoder="vits") is None

    def test_default_encoder_resolution_on_real_weights_dir(self):
        # Default is vitb (measured quality upgrade); when vitb weights are
        # absent the resolution must fall back to the only checkpoint
        # present so fresh checkouts keep working on vits.
        ckpt = find_checkpoint()
        assert ckpt is not None
        assert ckpt.is_file()
        if (ckpt.parent / "depth_anything_v2_vitb.pth").is_file():
            assert checkpoint_encoder(ckpt.name) == "vitb"
        else:
            assert checkpoint_encoder(ckpt.name) == "vits"


class TestSidecarModelIdentity:
    def test_sha_identity_strict(self, tmp_path):
        fid = "frame_000040"
        ws = _seed(tmp_path, fid, np.zeros((4, 3), np.float64) + 50.0)
        _sidecar(ws, fid, model_sha256="a" * 64, model_version="depth_anything_v2_vits.pth")
        assert dg._sidecar_model_matches(ws / "depth" / f"{fid}.json", "a" * 64, "depth_anything_v2_vits.pth")
        assert not dg._sidecar_model_matches(ws / "depth" / f"{fid}.json", "b" * 64, "depth_anything_v2_vits.pth")

    def test_filename_identity(self, tmp_path):
        fid = "frame_000040"
        ws = _seed(tmp_path, fid, np.zeros((4, 3), np.float64) + 50.0)
        _sidecar(ws, fid, model_version="depth_anything_v2_vitb.pth")
        assert not dg._sidecar_model_matches(ws / "depth" / f"{fid}.json", "a" * 64, "depth_anything_v2_vits.pth")
        assert dg._sidecar_model_matches(ws / "depth" / f"{fid}.json", "a" * 64, "depth_anything_v2_vitb.pth")

    def test_legacy_sidecar_valid_only_for_vits(self, tmp_path):
        fid = "frame_000040"
        ws = _seed(tmp_path, fid, np.zeros((4, 3), np.float64) + 50.0)
        _sidecar(ws, fid)  # pre-model-identity sidecar: no sha, no name
        assert dg._sidecar_model_matches(ws / "depth" / f"{fid}.json", "a" * 64, "depth_anything_v2_vits.pth")
        # A vitb run must never serve a legacy (vits-produced) map.
        assert not dg._sidecar_model_matches(ws / "depth" / f"{fid}.json", "b" * 64, "depth_anything_v2_vitb.pth")

    def test_unreadable_sidecar_never_matches(self, tmp_path):
        fid = "frame_000040"
        ws = _seed(tmp_path, fid, np.zeros((4, 3), np.float64) + 50.0)
        (ws / "depth" / f"{fid}.json").write_text("{not json")
        assert not dg._sidecar_model_matches(ws / "depth" / f"{fid}.json", "a" * 64, "x.pth")


class TestCrossVariantRegeneration:
    def test_vitb_run_regenerates_vits_cache(self, tmp_path, monkeypatch):
        """A vitb checkpoint must NOT serve a vits-produced cache entry: the
        cached map is removed and the view regenerates with the live model."""
        fid = "frame_000041"
        ws = _seed(tmp_path, fid, _ground_cloud())
        good_raw = _raw_map("healthy")
        np.save(ws / "depth" / f"{fid}.npy", (1.0 / (0.0002 * good_raw + 0.005)).astype(np.float32))
        _sidecar(ws, fid, model_version="depth_anything_v2_vits.pth")  # vits-produced

        monkeypatch.setattr(dg, "_ckpt_name", lambda: "depth_anything_v2_vitb.pth")
        monkeypatch.setattr(dg, "_depth_checkpoint_sha256", lambda: "b" * 64)
        _patch_infer(monkeypatch, good_raw)
        summary = dg.generate_view_depths(ws, backend="depth_anything")
        assert fid in summary.generated
        assert fid not in summary.cached
        sidecar = json.loads((ws / "depth" / f"{fid}.json").read_text())
        assert sidecar["model_version"] == "depth_anything_v2_vitb.pth"
        assert sidecar["model_sha256"] == "b" * 64

    def test_vits_run_serves_vits_cache(self, tmp_path, monkeypatch):
        fid = "frame_000041"
        ws = _seed(tmp_path, fid, _ground_cloud())
        good_raw = _raw_map("healthy")
        np.save(ws / "depth" / f"{fid}.npy", (1.0 / (0.0002 * good_raw + 0.005)).astype(np.float32))
        _sidecar(ws, fid, model_version="depth_anything_v2_vits.pth")

        monkeypatch.setattr(dg, "_ckpt_name", lambda: "depth_anything_v2_vits.pth")
        monkeypatch.setattr(dg, "_depth_checkpoint_sha256", lambda: "a" * 64)
        monkeypatch.setattr(
            "app.services.depth_anything_v2.load_model",
            lambda: (_ for _ in ()).throw(AssertionError("cache must be served, not regenerated")),
        )
        summary = dg.generate_view_depths(ws, backend="depth_anything")
        assert fid in summary.cached
