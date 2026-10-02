"""Tests for Phase 12 Performance Profiler, Hardware Discovery, and Stage Caching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.performance_profiler import PerformanceProfiler, get_system_hardware_info
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.stage_caching import (
    compute_stage_fingerprint,
    is_stage_cache_valid,
    save_stage_fingerprint,
)


def test_hardware_info_discovery():
    hw = get_system_hardware_info()
    assert "cpu_cores" in hw
    assert hw["cpu_cores"] >= 1
    assert "ram_gb" in hw
    assert hw["ram_gb"] > 0
    assert "device_type" in hw
    assert hw["device_type"] in ("cpu", "cuda", "mps")
    assert "python_version" in hw


def test_performance_profiler_lifecycle(tmp_path: Path):
    profiler = PerformanceProfiler("job_test_12", tmp_path)
    profiler.start_stage("frames")
    profiler.complete_stage(
        "frames",
        input_count=100,
        output_count=20,
        throughput_unit="frames/sec",
        details={"mode": "target_fps"},
    )
    profiler.start_stage("sparse")
    profiler.complete_stage(
        "sparse",
        input_count=20,
        output_count=18,
        details={"num_cameras": 18},
    )

    report = profiler.finalize_and_save(status="completed")

    assert report["job_id"] == "job_test_12"
    assert report["status"] == "completed"
    assert report["total_wall_clock_sec"] >= 0.0
    assert "hardware" in report
    assert "bottleneck" in report
    assert "stages" in report
    assert "frames" in report["stages"]
    assert report["stages"]["frames"]["output_count"] == 20

    perf_file = tmp_path / "performance.json"
    assert perf_file.exists()
    saved = json.loads(perf_file.read_text())
    assert saved["job_id"] == "job_test_12"


def test_stage_caching_and_fingerprinting(tmp_path: Path):
    stage = "frames"
    params = {"mode": "every_n", "every_n": 5}
    fp1 = compute_stage_fingerprint(stage, tmp_path, params)
    assert len(fp1) == 32

    # Artifact check returns False initially
    check_fn = lambda w: (w / "selected").is_dir()
    assert not is_stage_cache_valid(stage, tmp_path, fp1, check_fn)

    # Create dummy artifact and save fingerprint
    (tmp_path / "selected").mkdir(parents=True, exist_ok=True)
    (tmp_path / "selected" / "frame_000.jpg").write_text("dummy")
    save_stage_fingerprint(stage, tmp_path, fp1, {"count": 1})

    assert is_stage_cache_valid(stage, tmp_path, fp1, check_fn)

    # Differing params must invalidate cache
    params_diff = {"mode": "every_n", "every_n": 10}
    fp2 = compute_stage_fingerprint(stage, tmp_path, params_diff)
    assert not is_stage_cache_valid(stage, tmp_path, fp2, check_fn)
