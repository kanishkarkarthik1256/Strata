"""Mission history, feature store, similarity and learning gates.

Records live as JSONL at ``<storage base>/mission_history.jsonl`` — one
append-only structured file per deployment (the "structured database" for
this deployment; records carry full metadata). History functions expose
three honesty levels:

* ``measurements`` — values recorded from completed missions,
* ``historical_statistics`` — aggregates over records,
* ``model_prediction`` — only available once ``min_history_for_ml`` records
  exist; before that the engine answers *"Insufficient historical data for
  ML prediction"* instead of inventing a learned model.

No ML is trained here: below the gate the system uses confidence-weighted
averages and nearest-neighbour retrieval, both labelled ``historical``.
Validation compares each new mission's predicted (plan) vs measured outcome
and appends prediction-error rows to the record store.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.config.settings import settings
from app.logging_config import get_logger

log = get_logger("drone_recon.services.mission_history")

_FEATURE_KEYS = ("altitude_m", "speed_m_s", "forward_overlap", "side_overlap",
                 "gps_score", "env_score", "coverage", "confidence",
                 "battery_consumed_pct", "frames", "processing_time_s")


def history_path(base: Path | None = None) -> Path:
    base = Path(base) if base else settings.storage.base_dir
    base.mkdir(parents=True, exist_ok=True)
    return base / "mission_history.jsonl"


def load_history(base: Path | None = None) -> list[dict]:
    """Records, newest last; corrupt lines are skipped with a warning."""
    path = history_path(base)
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            log.warning("history_line_skipped")
    return records


def append_record(record: dict, base: Path | None = None) -> dict:
    """Append one record; a duplicated mission_id overwrites in place."""
    records = load_history(base)
    records = [r for r in records if r.get("mission_id") != record.get("mission_id")]
    records.append(record)
    path = history_path(base)
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return record


def build_features(records: list[dict]) -> list[dict]:
    """ML-ready numeric features; missing values become None (not imputed)."""
    feats = []
    for r in records:
        f = {k: r.get(k) for k in _FEATURE_KEYS}
        f["mission_id"] = r.get("mission_id")
        f["timestamp"] = r.get("timestamp")
        feats.append(f)
    return feats


def _norm(value: float | None, lo: float | None, hi: float | None) -> float | None:
    if value is None or lo is None or hi is None:
        return None
    if hi <= lo:  # degenerate range (single-value pool): only an exact match scores
        return 0.0 if float(value) == float(lo) else None
    return (float(value) - lo) / (hi - lo)


def similar_missions(query: dict, records: list[dict], k: int | None = None) -> dict:
    """Nearest historical missions by normalized feature distance.

    Every returned neighbour is labelled ``historical``; the aggregate
    outcome columns are *observations*, never predictions.
    """
    k = k or settings.planning.similar_top_k
    pool = [r for r in records if r.get("mission_id") != query.get("mission_id")]
    if not pool:
        return {"status": "insufficient_data",
                "message": "no previous missions in history for similarity search",
                "neighbours": [], "min_required": settings.planning.min_history_for_similar}
    keys = [k2 for k2 in ("altitude_m", "speed_m_s", "forward_overlap",
                          "env_score", "gps_score") if _has(pool, k2)]
    ranges = {k2: (min(r[k2] for r in pool if r.get(k2) is not None),
                   max(r[k2] for r in pool if r.get(k2) is not None))
              for k2 in keys}
    scored = []
    for r in pool:
        dist = 0.0
        used = 0
        for k2 in keys:
            qv, rv = query.get(k2), r.get(k2)
            if qv is None or rv is None:
                continue
            qn = _norm(qv, *ranges[k2])
            rn = _norm(rv, *ranges[k2])
            if qn is None or rn is None:
                continue
            dist += (qn - rn) ** 2
            used += 1
        if used:
            scored.append((dist / max(used, 1), r))
    scored.sort(key=lambda x: x[0])
    neighbours = scored[:k]
    out = []
    for dist, r in neighbours:
        out.append({"mission_id": r.get("mission_id"),
                    "distance": round(dist, 4),
                    "source": "historical",
                    "coverage": r.get("coverage"),
                    "confidence": r.get("confidence"),
                    "battery_consumed_pct": r.get("battery_consumed_pct"),
                    "env_score": r.get("env_score"),
                    "gps_score": r.get("gps_score")})
    if not out:
        return {"status": "insufficient_data",
                "message": "no comparable historical features found",
                "neighbours": [], "min_required": 1}
    cov = [n["coverage"] for n in out if n["coverage"] is not None]
    conf = [n["confidence"] for n in out if n["confidence"] is not None]
    bat = [n["battery_consumed_pct"] for n in out if n["battery_consumed_pct"] is not None]
    avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
    return {"status": "historical", "count": len(out),
            "average_coverage": avg(cov), "average_confidence": avg(conf),
            "average_battery_pct": avg(bat),
            "label": "historical observations from similar missions — not predictions",
            "neighbours": out}


def _has(records: list[dict], key: str) -> bool:
    return any(r.get(key) is not None for r in records)


def learning_summary(records: list[dict]) -> dict:
    """Aggregate learning view with an explicit data-gate.

    Below ``min_history_for_ml`` nothing is labelled a model prediction; the
    summary still offers honest historical statistics when at least one
    record exists.
    """
    n = len(records)
    if n == 0:
        return {"status": "no_history", "missions": 0,
                "message": "no completed missions in history yet — run a "
                           "mission to start the feedback loop"}
    cov = [r["coverage"] for r in records if r.get("coverage") is not None]
    conf = [r["confidence"] for r in records if r.get("confidence") is not None]
    bat = [r["battery_consumed_pct"] for r in records if r.get("battery_consumed_pct") is not None]
    proc = [r["processing_time_s"] for r in records if r.get("processing_time_s") is not None]
    avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
    ok = sum(1 for r in records if r.get("mission_success", True) is not False)
    ml_gate = n >= settings.planning.min_history_for_ml
    best = None
    if cov:
        top = max(records, key=lambda r: r.get("coverage") or 0.0)
        best = {"coverage_best": {"mission_id": top.get("mission_id"),
                                  "coverage": top.get("coverage"),
                                  "confidence": top.get("confidence"),
                                  "altitude_m": top.get("altitude_m"),
                                  "speed_m_s": top.get("speed_m_s")},
                "label": "historical observation — the record with the highest measured coverage"}
    return {
        "status": "ready" if ml_gate else "insufficient_for_ml",
        "missions": n, "successful": ok,
        "average_coverage": avg(cov), "average_confidence": avg(conf),
        "average_battery_consumed_pct": avg(bat),
        "average_processing_time_s": avg(proc),
        "best_performer": best,
        "ml_gate": {"required": settings.planning.min_history_for_ml,
                    "message": None if ml_gate else
                    "Insufficient historical data for ML prediction "
                    f"({n}/{settings.planning.min_history_for_ml} missions). "
                    "Historical statistics above are observations, not learned predictions."},
        "label": "historical statistics" if not ml_gate else "historical statistics + model-ready",
    }


def record_from_workspace(workspace: Path) -> dict:
    """Assemble a measured-outcome record from a completed mission workspace.

    Only values actually present in artifacts are recorded (fields stay None
    otherwise); nothing is estimated here.
    """
    rec: dict = {"mission_id": workspace.name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "coverage": None, "confidence": None, "env_score": None,
                 "gps_score": None, "battery_consumed_pct": None,
                 "processing_time_s": None, "frames": None, "dense_points": None,
                 "mesh_faces": None, "mission_success": None,
                 "measured": True}

    def _load(*parts: str) -> dict | None:
        p = workspace.joinpath(*parts)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    conf = _load("confidence", "confidence_report.json")
    if conf:
        # outcome columns are percentages (0-100), matching the predicted
        # plan metrics below so validation compares like for like
        rec["coverage"] = round(100.0 * (1.0 - float(conf.get("low_share", 0.0) or 0.0)), 1)
        rec["confidence"] = round(100.0 * float(conf.get("mean_confidence", 0.0) or 0.0), 1)
    env = _load("intel", "environment_report.json")
    if env:
        rec["env_score"] = env.get("quality_score")
    gps = _load("georef", "gps_report.json")
    if gps:
        rec["gps_score"] = (gps.get("gps_quality") or {}).get("gps_score")
    dense = _load("dense_report.json")
    if dense:
        q = dense.get("quality") or {}
        rec["dense_points"] = q.get("point_count") or q.get("points")
    plan = _load("intel", "mission_plan.json")
    if plan:
        chosen = plan.get("chosen_plan") or {}
        rec["altitude_m"] = (chosen.get("plan") or {}).get("altitude_m")
        rec["speed_m_s"] = (chosen.get("plan") or {}).get("speed_m_s")
        rec["forward_overlap"] = (chosen.get("plan") or {}).get("forward_overlap")
        metrics = chosen.get("metrics", {}) or {}
        # plan metrics are 0-1 fractions; scale outcome rates to percent so
        # they validate against the measured columns above
        predicted = {}
        for key in ("coverage_est", "quality_index"):
            v = metrics.get(key)
            predicted[key] = round(100.0 * float(v), 1) if v is not None else None
        predicted["duration_min"] = metrics.get("duration_min")
        rec["plan_predicted"] = predicted
    frames_dir = workspace / "selected"
    if frames_dir.is_dir():
        rec["frames"] = sum(1 for p in frames_dir.iterdir() if p.suffix.lower() in
                            (".jpg", ".png", ".jpeg"))
    report = _load("pipeline_report.json")
    if report:
        rec["processing_time_s"] = round(float(report.get("run_time_ms", 0.0)) / 1000.0, 1)
        rec["mission_success"] = report.get("status") == "completed"
    mesh_path = workspace / "mesh" / "repaired_mesh.ply"
    if mesh_path.exists():
        try:
            header = mesh_path.read_bytes().split(b"end_header")[0].decode("ascii")
            for line in header.splitlines():
                if line.startswith("element face "):
                    rec["mesh_faces"] = int(line.split()[-1])
        except (OSError, ValueError):
            pass
    return rec


def validate_mission(record: dict, base: Path | None = None) -> dict:
    """Compare predicted vs measured outcome and log prediction error.

    Returns the prediction-error row for this mission; the comparison is only
    meaningful when the mission carried a plan with predictions.
    """
    pred = record.get("plan_predicted") or {}
    if not pred:
        return {"mission_id": record.get("mission_id"),
                "status": "no_prediction_recorded",
                "message": "this mission had no recorded plan predictions to validate"}
    measured = {"coverage_est": record.get("coverage"),
                "quality_index": record.get("confidence")}
    # flight duration is only validated when a measured duration exists;
    # ``processing_time_s`` is compute time, not flight time, so it is never
    # substituted as a measured duration.
    if record.get("duration_measured_min") is not None:
        measured["duration_min"] = record["duration_measured_min"]
    errors = {}
    for key, mv in measured.items():
        pv = pred.get(key)
        if pv is None or mv is None:
            continue
        errors[key] = round(float(mv) - float(pv), 3)
    row = {"mission_id": record.get("mission_id"),
           "timestamp": record.get("timestamp"),
           "predicted": {k: v for k, v in pred.items() if k in measured},
           "measured": {k: v for k, v in measured.items() if v is not None},
           "prediction_error": errors,
           "label": "prediction-vs-actual validation"}
    return row


def recent(records: list[dict], limit: int = 20) -> list[dict]:
    return list(reversed(records[-limit:]))
