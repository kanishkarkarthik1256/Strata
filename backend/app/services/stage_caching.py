"""Deterministic Stage Caching & Invalidation Fingerprinting System (Phase 12).

Computes cryptographic hashes of stage input files, configuration parameters,
and runtime settings. Validates cached state to skip redundant execution
while guaranteeing invalidation whenever inputs or parameters change.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger

log = get_logger("drone_recon.services.stage_caching")


def compute_file_hash(path: Path, max_bytes: int = 10 * 1024 * 1024) -> str:
    """Compute MD5 hash of a file (sampling up to max_bytes for speed)."""
    if not path.exists() or not path.is_file():
        return ""
    hasher = hashlib.md5()
    try:
        size = path.stat().st_size
        hasher.update(str(size).encode())
        hasher.update(str(path.stat().st_mtime).encode())
        with open(path, "rb") as f:
            if size <= max_bytes:
                hasher.update(f.read())
            else:
                hasher.update(f.read(max_bytes // 2))
                f.seek(size - max_bytes // 2)
                hasher.update(f.read(max_bytes // 2))
    except Exception:
        pass
    return hasher.hexdigest()


def compute_dir_hash(dir_path: Path, pattern: str = "*", max_files: int = 100) -> str:
    """Compute aggregate hash for files in a directory."""
    if not dir_path.exists() or not dir_path.is_dir():
        return ""
    hasher = hashlib.md5()
    files = sorted(dir_path.glob(pattern))[:max_files]
    hasher.update(str(len(files)).encode())
    for f in files:
        hasher.update(f.name.encode())
        hasher.update(str(f.stat().st_size).encode())
    return hasher.hexdigest()


#: Which implementation modules each stage's OUTPUT depends on. A code
#: change in one of these modules must regenerate the stage's artifacts:
#: the cache previously hashed only inputs + request params, so a fixed
#: algorithm kept serving its PRE-FIX output (measured: a Poisson-shell
#: mesh produced by pre-fix code stayed the served artifact after the
#: mesher was fixed, because the dense-stage fingerprint never changed).
STAGE_CODE_MODULES: Dict[str, List[str]] = {
    "frames": ["app.services.frame_extractor", "app.services.frame_quality"],
    "sparse": ["app.services.sparse_reconstruction", "app.services.camera_pose_estimator",
               "app.services.bundle_adjustment", "app.services.trajectory_sync"],
    "depth": ["app.services.depth_generator", "app.services.depth_alignment",
              "app.services.depth_diagnostics"],
    "dense": ["app.services.dense_reconstruction", "app.services.depth_fusion",
              "app.services.pointcloud_filter", "app.services.pointcloud_optimizer",
              "app.services.mesh_generator", "app.services.mesh_glb", "app.services.textured_glb"],
    "georef": ["app.services.georeferencing"],
}


def _module_digests(modules: List[str]) -> str:
    """Aggregate source digest for a stage's implementation modules."""
    hasher = hashlib.md5()
    for name in modules:
        try:
            mod = importlib.import_module(name)
            path = getattr(mod, "__file__", None)
        except Exception:
            path = None
        if path and Path(path).exists():
            hasher.update(name.encode())
            hasher.update(Path(path).read_bytes())
    return hasher.hexdigest()


def compute_stage_fingerprint(
    stage_name: str,
    workspace: Path,
    params: Dict[str, Any],
) -> str:
    """Generate a deterministic fingerprint for a stage based on workspace state & params.

    Includes the digest of the stage's implementation modules: identical
    inputs and request must still re-run when the CODE that produces the
    artifacts changed — otherwise a fixed algorithm keeps serving the output
    of the broken one.
    """
    hasher = hashlib.md5()
    hasher.update(stage_name.encode())
    hasher.update(json.dumps(params, sort_keys=True, default=str).encode())
    modules = STAGE_CODE_MODULES.get(stage_name)
    if modules:
        hasher.update(_module_digests(modules).encode())

    if stage_name == "frames":
        video_files = list(workspace.glob("*.mp4")) + list(workspace.glob("*.mov")) + list(workspace.glob("*.avi"))
        if video_files:
            hasher.update(compute_file_hash(video_files[0]).encode())

    elif stage_name == "sparse":
        selected_dir = workspace / "selected" if (workspace / "selected").is_dir() else workspace / "frames"
        hasher.update(compute_dir_hash(selected_dir, "*.jpg").encode())

    elif stage_name == "depth":
        poses_path = workspace / "poses.json"
        hasher.update(compute_file_hash(poses_path).encode())

    elif stage_name == "dense":
        depth_dir = workspace / "depth"
        poses_path = workspace / "poses.json"
        hasher.update(compute_dir_hash(depth_dir, "*.npy").encode())
        hasher.update(compute_file_hash(poses_path).encode())

    elif stage_name == "georef":
        poses_path = workspace / "poses.json"
        hasher.update(compute_file_hash(poses_path).encode())

    return hasher.hexdigest()


def is_stage_cache_valid(
    stage_name: str,
    workspace: Path,
    fingerprint: str,
    artifact_check_fn,
) -> bool:
    """Check if output artifacts exist and input fingerprint matches stored state.

    Validation is REPRODUCIBLE: when the stored record carries the ``params``
    it was computed from, the fingerprint is recomputed from those params and
    the current workspace state. A mismatch means a hashed input changed since
    the record was written, so the artifacts are stale even if the caller
    passes an identical request. Legacy records written before the params
    contract (hash only) keep the original equality-only behaviour —
    ``adopt_stage_fingerprint`` upgrades them.
    """
    if not artifact_check_fn(workspace):
        return False

    fp_file = workspace / f".fingerprint_{stage_name}.json"
    if not fp_file.exists():
        return False

    try:
        data = json.loads(fp_file.read_text())
    except Exception:
        return False

    stored = data.get("fingerprint")
    if stored != fingerprint:
        return False

    stored_params = data.get("params")
    if isinstance(stored_params, dict):
        # Reproducible check: the recorded params must still reproduce the
        # recorded fingerprint against the CURRENT input state.
        if compute_stage_fingerprint(stage_name, workspace, stored_params) != stored:
            log.info("stage_fingerprint_inputs_changed", stage=stage_name,
                     note="stored fingerprint no longer reproduces from current inputs — stage will re-run")
            return False
    else:
        log.warning("stage_fingerprint_missing_params", stage=stage_name,
                    note="legacy fingerprint record has no params — validation is not reproducible; "
                         "run adopt_stage_fingerprint to upgrade it")
    return True


def describe_fingerprint_mismatch(
    stage_name: str,
    workspace: Path,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Explain WHY a stage with present artifacts will be recomputed.

    A resume that re-runs an expensive stage must never be silent (measured:
    a resume re-started a 15-minute sparse stage because the legacy record
    kept no params, so the request it hashed could not be reconstructed).
    Called only when the artifacts exist and the cache check failed.
    """
    fp_file = workspace / f".fingerprint_{stage_name}.json"
    if not fp_file.exists():
        return {"reason": "no_record"}
    try:
        data = json.loads(fp_file.read_text())
    except Exception as exc:
        return {"reason": "record_unreadable", "error": str(exc)}

    if compute_stage_fingerprint(stage_name, workspace, params) == data.get("fingerprint"):
        return {"reason": "match"}

    stored_params = data.get("params")
    if not isinstance(stored_params, dict):
        return {
            "reason": "legacy_record_without_params",
            "note": "record predates the params contract, so the request it "
                    "hashed cannot be reconstructed — adopt_stage_fingerprint "
                    "upgrades it so future resumes validate reproducibly",
        }

    differences = {
        key: {"recorded": stored_params.get(key), "requested": params.get(key)}
        for key in sorted(set(stored_params) | set(params))
        if stored_params.get(key) != params.get(key)
    }
    if differences:
        return {"reason": "params_changed", "param_differences": differences}
    # Params identical, so a hashed INPUT changed under the record.
    return {
        "reason": "inputs_changed",
        "note": "recorded params are unchanged but the hashed inputs are not — "
                "the stored artifacts were produced from different inputs",
    }


def load_stage_params(stage_name: str, workspace: Path) -> Optional[Dict[str, Any]]:
    """The params a stored fingerprint covers (None for legacy/absent records)."""
    fp_file = workspace / f".fingerprint_{stage_name}.json"
    if not fp_file.exists():
        return None
    try:
        params = json.loads(fp_file.read_text()).get("params")
    except Exception:
        return None
    return params if isinstance(params, dict) else None


def load_stage_details(stage_name: str, workspace: Path) -> Dict[str, Any]:
    """The per-stage details a stored record carries ({} when absent).

    These are the stage's OWN record of what it produced — e.g. the depth
    stage names every view it deliberately did not map, with the reason, which
    is what lets artifact completeness agree with the conditioning contract
    instead of contradicting it.
    """
    fp_file = workspace / f".fingerprint_{stage_name}.json"
    if not fp_file.exists():
        return {}
    try:
        details = json.loads(fp_file.read_text()).get("details")
    except Exception:
        return {}
    return details if isinstance(details, dict) else {}


def verify_stage_fingerprint(stage_name: str, workspace: Path) -> Dict[str, Any]:
    """Introspect a stored fingerprint: does it reproduce from its own params?"""
    fp_file = workspace / f".fingerprint_{stage_name}.json"
    out: Dict[str, Any] = {"stage": stage_name, "exists": fp_file.exists()}
    if not out["exists"]:
        return out
    try:
        data = json.loads(fp_file.read_text())
    except Exception as exc:
        out["error"] = str(exc)
        return out
    out["fingerprint"] = data.get("fingerprint")
    out["params"] = data.get("params")
    out["adopted"] = bool(data.get("adopted"))
    if isinstance(data.get("params"), dict):
        out["reproducible"] = (
            compute_stage_fingerprint(stage_name, workspace, data["params"]) == data.get("fingerprint")
        )
    else:
        out["reproducible"] = None
    return out


def save_stage_fingerprint(
    stage_name: str,
    workspace: Path,
    fingerprint: str,
    details: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    adopted: bool = False,
) -> None:
    """Save fingerprint on successful stage completion.

    ``params`` is the request-parameter dict the fingerprint was computed
    from. Recording it is what makes a later resume reproducible: without it
    a caller cannot reconstruct the exact request, and any equivalent-but-not-
    identical request re-runs an expensive stage silently (measured: a resume
    re-started a 15-minute sparse stage for exactly this reason).

    ``details`` keeps its original positional slot: a caller written before
    ``params`` existed passes the stage detail fourth, and silently recording
    that as params would make the record claim a request nobody made.
    """
    fp_file = workspace / f".fingerprint_{stage_name}.json"
    data: Dict[str, Any] = {
        "stage": stage_name,
        "fingerprint": fingerprint,
        "details": details or {},
    }
    if params is not None:
        data["params"] = params
    if adopted:
        data["adopted"] = True
    try:
        fp_file.write_text(json.dumps(data, indent=2, default=str))
    except Exception as exc:
        log.warning("failed_to_save_stage_fingerprint", stage=stage_name, error=str(exc))


def adopt_stage_fingerprint(
    stage_name: str,
    workspace: Path,
    params: Dict[str, Any],
    artifact_check_fn,
    reason: str = "legacy fingerprint record had no params",
) -> bool:
    """Record ``params`` for an EXISTING artifact state (opt-in, logged).

    For artifacts produced before the params contract, the historical request
    cannot be reconstructed — but the artifact state and its input hashes CAN
    be verified against the present workspace. Adoption writes a record whose
    fingerprint reproduces from the current inputs, so future resumes validate
    reproducibly instead of silently re-running. It never runs a stage and
    never writes for a stage whose artifacts are absent. Records are marked
    ``adopted`` so the distinction is visible in the file.
    """
    if not artifact_check_fn(workspace):
        return False
    fingerprint = compute_stage_fingerprint(stage_name, workspace, params)
    save_stage_fingerprint(stage_name, workspace, fingerprint, params=params,
                           details={"adoption_reason": reason}, adopted=True)
    log.info("stage_fingerprint_adopted", stage=stage_name,
             fingerprint=fingerprint, reason=reason)
    return True
