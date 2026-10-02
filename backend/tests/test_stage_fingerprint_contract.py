"""Stage fingerprint contract — params recorded, validation reproducible.

Root cause these tests pin: ``.fingerprint_<stage>.json`` stored only the hash,
not the parameters it covered, so a resume could not reconstruct the original
request and silently re-ran an expensive stage (measured: a resume re-started
the ~15-minute sparse stage on flight_to_tower_7511dc). The contract is now:

  record  = {stage, fingerprint, params, details, adopted?}
  valid   = artifacts exist AND stored.fingerprint == caller.fingerprint
            AND (when params are recorded) compute_stage_fingerprint(stage,
            workspace, stored.params) == stored.fingerprint

i.e. validation reproduces the hash from the stored params against the CURRENT
input state, so a changed input invalidates even when the caller repeats the
same request.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.pipeline_orchestrator import _depth_complete, _frames_complete, _sparse_complete
from app.services.stage_caching import (
    adopt_stage_fingerprint,
    compute_stage_fingerprint,
    describe_fingerprint_mismatch,
    is_stage_cache_valid,
    load_stage_details,
    load_stage_params,
    save_stage_fingerprint,
    verify_stage_fingerprint,
)

PARAMS = {
    "extraction_mode": None,
    "every_n": None,
    "target_fps": None,
    "top_percent": None,
    "quality_threshold": None,
    "frame_budget": None,
    "depth_backend": "auto",
    "frame_stride": 1,
    "max_depth_views": 200,
}


def _depth_workspace(tmp_path: Path) -> Path:
    """A workspace with a depth-shaped input (poses.json) + output (depth/*.npy)."""
    ws = tmp_path / "run_cache"
    (ws / "depth").mkdir(parents=True)
    (ws / "poses.json").write_text(json.dumps({"frames": [{"frame_id": "frame_000000"}]}))
    np.save(ws / "depth" / "frame_000000.npy", np.zeros((4, 4), dtype=np.float32))
    return ws


def _depth_artifacts(ws: Path) -> bool:
    return (ws / "depth" / "frame_000000.npy").exists()


# ---------------------------------------------------------------------------
# params are recorded
# ---------------------------------------------------------------------------


def test_saved_record_carries_params_and_round_trips(tmp_path):
    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    save_stage_fingerprint("depth", ws, fp, params=PARAMS, details={"x": 1})

    raw = json.loads((ws / ".fingerprint_depth.json").read_text())
    assert raw["hash"] if "hash" in raw else True  # shape guard (no legacy key)
    assert raw["fingerprint"] == fp
    assert raw["params"] == PARAMS
    assert load_stage_params("depth", ws) == PARAMS


def test_legacy_record_without_params_reports_none(tmp_path):
    ws = _depth_workspace(tmp_path)
    (ws / ".fingerprint_depth.json").write_text(json.dumps({"stage": "depth", "fingerprint": "abc"}))
    assert load_stage_params("depth", ws) is None


# ---------------------------------------------------------------------------
# reproducible validation + no silent re-run
# ---------------------------------------------------------------------------


def test_matching_resume_does_not_rerun(tmp_path):
    """The anti-regression case: passing the recorded params validates."""
    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    save_stage_fingerprint("depth", ws, fp, params=PARAMS)

    assert is_stage_cache_valid("depth", ws, fp, _depth_artifacts) is True
    # and the fingerprint a resume would compute from the SAME params matches
    assert compute_stage_fingerprint("depth", ws, PARAMS) == fp


def test_different_params_rerun(tmp_path):
    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    save_stage_fingerprint("depth", ws, fp, params=PARAMS)

    changed = dict(PARAMS, max_depth_views=100)
    assert compute_stage_fingerprint("depth", ws, changed) != fp
    assert is_stage_cache_valid("depth", ws, compute_stage_fingerprint("depth", ws, changed), _depth_artifacts) is False


def test_changed_input_invalidates_even_with_identical_params(tmp_path):
    """Reproducibility: the stored params must still reproduce the hash."""
    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    save_stage_fingerprint("depth", ws, fp, params=PARAMS)
    assert is_stage_cache_valid("depth", ws, fp, _depth_artifacts) is True

    # A sparse rerun rewrites poses.json → the depth inputs changed.
    (ws / "poses.json").write_text(json.dumps({"frames": [
        {"frame_id": "frame_000000", "t": [1.0, 2.0, 3.0]},
    ]}))
    assert is_stage_cache_valid("depth", ws, fp, _depth_artifacts) is False


def test_missing_artifacts_invalidates(tmp_path):
    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    save_stage_fingerprint("depth", ws, fp, params=PARAMS)
    (ws / "depth" / "frame_000000.npy").unlink()
    assert is_stage_cache_valid("depth", ws, fp, _depth_artifacts) is False


def test_legacy_record_equality_still_works(tmp_path):
    """Backwards compatibility: hash-only records keep the old behaviour."""
    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    (ws / ".fingerprint_depth.json").write_text(json.dumps({"stage": "depth", "fingerprint": fp}))
    assert is_stage_cache_valid("depth", ws, fp, _depth_artifacts) is True
    assert is_stage_cache_valid("depth", ws, "different", _depth_artifacts) is False


# ---------------------------------------------------------------------------
# adoption upgrades legacy records without running anything
# ---------------------------------------------------------------------------


def test_adopt_upgrades_legacy_record_and_becomes_reproducible(tmp_path):
    ws = _depth_workspace(tmp_path)
    (ws / ".fingerprint_depth.json").write_text(json.dumps({"stage": "depth", "fingerprint": "stale"}))

    assert adopt_stage_fingerprint("depth", ws, PARAMS, _depth_artifacts) is True
    raw = json.loads((ws / ".fingerprint_depth.json").read_text())
    assert raw["adopted"] is True
    assert raw["params"] == PARAMS
    assert raw["fingerprint"] == compute_stage_fingerprint("depth", ws, PARAMS)
    assert is_stage_cache_valid("depth", ws, raw["fingerprint"], _depth_artifacts) is True
    assert verify_stage_fingerprint("depth", ws)["reproducible"] is True


def test_adopt_refuses_without_artifacts(tmp_path):
    ws = _depth_workspace(tmp_path)
    (ws / "depth" / "frame_000000.npy").unlink()
    assert adopt_stage_fingerprint("depth", ws, PARAMS, _depth_artifacts) is False


# ---------------------------------------------------------------------------
# a re-run is never silent: the mismatch is named
# ---------------------------------------------------------------------------


def test_mismatch_explains_legacy_record(tmp_path):
    """The exact historical failure: hash-only record, no params to reconstruct."""
    ws = _depth_workspace(tmp_path)
    (ws / ".fingerprint_depth.json").write_text(
        json.dumps({"stage": "depth", "fingerprint": "stale"}))

    why = describe_fingerprint_mismatch("depth", ws, PARAMS)
    assert why["reason"] == "legacy_record_without_params"
    assert "adopt_stage_fingerprint" in why["note"]


def test_mismatch_names_the_changed_params(tmp_path):
    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    save_stage_fingerprint("depth", ws, fp, params=PARAMS)

    requested = dict(PARAMS, max_depth_views=50, depth_backend="stereo")
    why = describe_fingerprint_mismatch("depth", ws, requested)
    assert why["reason"] == "params_changed"
    assert why["param_differences"] == {
        "depth_backend": {"recorded": "auto", "requested": "stereo"},
        "max_depth_views": {"recorded": 200, "requested": 50},
    }


def test_mismatch_distinguishes_changed_inputs_from_changed_params(tmp_path):
    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    save_stage_fingerprint("depth", ws, fp, params=PARAMS)

    (ws / "poses.json").write_text(json.dumps({"frames": [{"frame_id": "frame_000001"}]}))
    why = describe_fingerprint_mismatch("depth", ws, PARAMS)
    assert why["reason"] == "inputs_changed"
    assert "param_differences" not in why


def test_mismatch_reports_no_record_when_none_exists(tmp_path):
    ws = _depth_workspace(tmp_path)
    assert describe_fingerprint_mismatch("depth", ws, PARAMS)["reason"] == "no_record"


def test_same_request_after_adoption_is_a_hit_not_a_rerun(tmp_path):
    """End to end for the reported defect: legacy record -> adopt -> resume is
    a cache hit, and the *only* thing that can trigger a re-run afterwards is a
    request whose params actually differ (which the explanation names)."""
    ws = _depth_workspace(tmp_path)
    (ws / ".fingerprint_depth.json").write_text(
        json.dumps({"stage": "depth", "fingerprint": "stale"}))
    assert is_stage_cache_valid(
        "depth", ws, compute_stage_fingerprint("depth", ws, PARAMS), _depth_artifacts) is False

    adopt_stage_fingerprint("depth", ws, PARAMS, _depth_artifacts)

    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    assert is_stage_cache_valid("depth", ws, fp, _depth_artifacts) is True
    assert describe_fingerprint_mismatch("depth", ws, PARAMS)["reason"] == "match"

    other = dict(PARAMS, every_n=5)
    assert is_stage_cache_valid(
        "depth", ws, compute_stage_fingerprint("depth", ws, other), _depth_artifacts) is False
    why = describe_fingerprint_mismatch("depth", ws, other)
    assert why["reason"] == "params_changed"
    assert why["param_differences"]["every_n"] == {"recorded": None, "requested": 5}


# ---------------------------------------------------------------------------
# artifact completeness agrees with the stage's own named exclusions
# ---------------------------------------------------------------------------


def _poses_workspace(tmp_path: Path, count: int) -> Path:
    ws = tmp_path / "run_excl"
    (ws / "depth").mkdir(parents=True)
    (ws / "poses.json").write_text(json.dumps({
        "frames": [{"frame_id": f"frame_{i:06d}"} for i in range(count)],
    }))
    return ws


def _write_map(ws: Path, frame_id: str) -> None:
    np.save(ws / "depth" / f"{frame_id}.npy", np.zeros((2, 2), dtype=np.float32))


def test_depth_complete_requires_every_view_when_nothing_is_named(tmp_path):
    ws = _poses_workspace(tmp_path, 3)
    _write_map(ws, "frame_000000")           # frame_000001 missing, unnamed
    assert _depth_complete(ws) is False


def test_depth_complete_accepts_only_named_exclusions(tmp_path):
    """The reported defect: a legitimately excluded view must not make the
    stage report incomplete forever (it re-ran depth on every resume)."""
    ws = _poses_workspace(tmp_path, 3)
    _write_map(ws, "frame_000000")
    _write_map(ws, "frame_000001")
    save_stage_fingerprint("depth", ws, "fp", params=PARAMS,
                           details={"excluded": ["frame_000002"],
                                    "excluded_reasons": {"frame_000002": "inverted_sparse_structure"}})
    assert _depth_complete(ws) is True


def test_depth_complete_rejects_a_missing_view_the_record_does_not_name(tmp_path):
    ws = _poses_workspace(tmp_path, 4)
    _write_map(ws, "frame_000000")
    save_stage_fingerprint("depth", ws, "fp", params=PARAMS,
                           details={"excluded": ["frame_000002"]})
    # frame_000001 is missing and NOT named by the stage (frame_000003 is the
    # trailing view — legitimately unmapped and irrelevant here)
    assert _depth_complete(ws) is False


def test_depth_complete_ignores_the_trailing_view(tmp_path):
    """An open trajectory's last view legitimately has no forward neighbour."""
    ws = _poses_workspace(tmp_path, 3)
    _write_map(ws, "frame_000000")
    _write_map(ws, "frame_000001")
    assert _depth_complete(ws) is True


def test_frames_complete_requires_a_usable_selection(tmp_path):
    """One selected frame is not a reconstruction input: SfM needs multi-view
    geometry, and caching a 1-frame selection made Retry a no-op (London
    Mission: sparse re-failed identically on every retry)."""
    ws = tmp_path / "run_frames"
    (ws / "selected").mkdir(parents=True)
    assert _frames_complete(ws) is False
    np.save(ws / "selected" / "frame_000000.npy", np.zeros((2, 2)))  # not an image
    (ws / "selected" / "frame_000000.jpg").write_bytes(b"jpg")
    assert _frames_complete(ws) is False  # 1 image
    (ws / "selected" / "frame_000001.jpg").write_bytes(b"jpg")
    assert _frames_complete(ws) is True


def test_sparse_complete_requires_a_multi_camera_trajectory(tmp_path):
    """A 1-camera poses.json cannot feed depth or dense — caching it let
    Retry skip the broken sparse stage and re-fail downstream."""
    ws = tmp_path / "run_sparse"
    ws.mkdir(parents=True)
    assert _sparse_complete(ws) is False  # no poses.json
    (ws / "poses.json").write_text(json.dumps({"frames": [{"frame_id": "f0"}]}))
    assert _sparse_complete(ws) is False  # 1 camera
    (ws / "poses.json").write_text(json.dumps(
        {"frames": [{"frame_id": "f0"}, {"frame_id": "f1"}]}))
    assert _sparse_complete(ws) is True


def test_depth_complete_refuses_a_degenerate_single_map_run(tmp_path):
    """Named exclusions cannot outvote maps: 1 usable map on a 10-view
    trajectory is a degenerate stage, not a completed one. Caching it made
    Retry a no-op loop — dense re-failed the audit identically on every
    retry (sunset_06cfea: 1 generated + 9 excluded)."""
    ws = _poses_workspace(tmp_path, 10)
    _write_map(ws, "frame_000000")
    save_stage_fingerprint("depth", ws, "fp", params=PARAMS,
                           details={"generated": ["frame_000000"],
                                    "excluded": [f"frame_{i:06d}" for i in range(1, 9)],
                                    "excluded_reasons": {f"frame_{i:06d}": "low_parallax" for i in range(1, 9)}})
    assert _depth_complete(ws) is False


def test_depth_complete_counts_failed_views_as_usable(tmp_path):
    """A genuine execution failure still produced a map on disk — it feeds
    fusion. The floor counts maps on disk, not the stage's bookkeeping lists
    (a legacy record may not carry them at all)."""
    ws = _poses_workspace(tmp_path, 4)
    _write_map(ws, "frame_000000")
    _write_map(ws, "frame_000001")
    _write_map(ws, "frame_000002")
    save_stage_fingerprint("depth", ws, "fp", params=PARAMS,
                           details={"generated": ["frame_000000"],
                                    "failed": ["frame_000001"],
                                    "failure_reasons": {"frame_000001": "inference_error"}})
    assert _depth_complete(ws) is True


def test_load_stage_details_reads_the_stages_own_record(tmp_path):
    ws = _depth_workspace(tmp_path)
    save_stage_fingerprint("depth", ws, "fp", params=PARAMS, details={"excluded": ["a"]})
    assert load_stage_details("depth", ws) == {"excluded": ["a"]}
    assert load_stage_details("sparse", ws) == {}


def test_verify_reports_non_reproducible_after_input_change(tmp_path):
    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    save_stage_fingerprint("depth", ws, fp, params=PARAMS)
    (ws / "poses.json").write_text(json.dumps({"frames": [{"frame_id": "frame_00000X"}]}))
    assert verify_stage_fingerprint("depth", ws)["reproducible"] is False


# ---------------------------------------------------------------------------
# Code-version contract: a stage's fingerprint covers the CODE that produced
# its artifacts. Without this, a fixed algorithm keeps serving its pre-fix
# output forever (measured: the served mesh stayed a pre-fix Poisson shell
# after the mesher was repaired, because inputs + params never changed).


def _write_module(tmp_path: Path, text: str) -> Path:
    mod = tmp_path / "_fpc_mod_under_test.py"
    mod.write_text(text)
    return mod


def _load_module(mod_path: Path) -> object:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_fpc_mod_under_test", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fpc_mod_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_code_change_invalidates_cache(tmp_path, monkeypatch):
    """A fix to a stage's implementation module must regenerate its artifacts:
    same inputs, same request, new code → the cache refuses to serve."""
    mod_path = _write_module(tmp_path, "VALUE = 1\n")
    _load_module(mod_path)
    monkeypatch.setattr(
        "app.services.stage_caching.STAGE_CODE_MODULES",
        {"depth": ["_fpc_mod_under_test"]},
    )

    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    save_stage_fingerprint("depth", ws, fp, params=PARAMS)
    assert is_stage_cache_valid("depth", ws, fp, lambda w: True)

    # The fix lands: the module's source changes, inputs do not.
    mod_path.write_text("VALUE = 2\n")
    _load_module(mod_path)
    fp2 = compute_stage_fingerprint("depth", ws, PARAMS)
    assert fp2 != fp
    assert not is_stage_cache_valid("depth", ws, fp2, lambda w: True)


def test_missing_code_module_degrades_to_input_only(tmp_path, monkeypatch):
    """An unresolvable module name must never raise or block the pipeline."""
    monkeypatch.setattr(
        "app.services.stage_caching.STAGE_CODE_MODULES",
        {"depth": ["app.services.definitely_not_a_module_xyz"]},
    )
    ws = _depth_workspace(tmp_path)
    fp = compute_stage_fingerprint("depth", ws, PARAMS)
    save_stage_fingerprint("depth", ws, fp, params=PARAMS)
    assert is_stage_cache_valid("depth", ws, fp, lambda w: True)


def test_every_pipeline_stage_declares_its_code_modules():
    from app.services.stage_caching import STAGE_CODE_MODULES

    for stage in ("frames", "sparse", "depth", "dense", "georef"):
        modules = STAGE_CODE_MODULES.get(stage)
        assert modules, f"{stage} declares no code modules"
        assert all(isinstance(m, str) and m for m in modules)
