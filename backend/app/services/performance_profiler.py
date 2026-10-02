"""STRATA Performance Profiler & Benchmark System (Phase 12).

Provides hardware capability detection, per-stage wall-clock performance metrics,
input/output throughput measurements, execution device logging, cache hit/miss tracking,
fallback auditing, bottleneck detection, and standardized performance.json output generation.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger

log = get_logger("drone_recon.services.performance_profiler")


def get_system_hardware_info() -> Dict[str, Any]:
    """Inspect and report hardware capabilities of the host system."""
    cpu_cores = os.cpu_count() or 1
    ram_gb = 8.0  # standard fallback
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            ram_gb = round(ram_bytes / (1024 ** 3), 2)
    except Exception:
        pass

    device_type = "cpu"
    gpu_name = "CPU Only"

    try:
        import torch

        if torch.cuda.is_available():
            device_type = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            gpu_name = "Apple Silicon MPS"
    except Exception:
        pass

    opencv_ver = "unavailable"
    try:
        import cv2

        opencv_ver = cv2.__version__
    except Exception:
        pass

    pycolmap_ver = "unavailable"
    try:
        import pycolmap

        pycolmap_ver = getattr(pycolmap, "__version__", "installed")
    except Exception:
        pass

    pytorch_ver = "unavailable"
    try:
        import torch

        pytorch_ver = torch.__version__
    except Exception:
        pass

    return {
        "os_system": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": sys.version.split()[0],
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb,
        "device_type": device_type,
        "gpu_name": gpu_name,
        "opencv_version": opencv_ver,
        "pycolmap_version": pycolmap_ver,
        "pytorch_version": pytorch_ver,
    }


@dataclass
class StageMetric:
    """Performance metrics for a single pipeline stage."""

    name: str
    status: str = "pending"  # pending|running|completed|skipped|failed|cancelled
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    duration_sec: float = 0.0
    input_count: int = 0
    output_count: int = 0
    throughput_value: float = 0.0
    throughput_unit: str = "items/sec"
    device: str = "cpu"
    cache_hit: bool = False
    cache_key: Optional[str] = None
    fallback_used: bool = False
    fallback_name: str = ""
    error: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def finish(
        self,
        *,
        status: str = "completed",
        input_count: int = 0,
        output_count: int = 0,
        throughput_unit: str = "items/sec",
        device: str = "cpu",
        cache_hit: bool = False,
        cache_key: Optional[str] = None,
        fallback_used: bool = False,
        fallback_name: str = "",
        error: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.end_time = time.perf_counter()
        self.duration_sec = max(0.0, self.end_time - self.start_time)
        self.duration_ms = round(self.duration_sec * 1000.0, 2)
        self.status = status
        self.input_count = input_count
        self.output_count = output_count
        self.device = device
        self.cache_hit = cache_hit
        self.cache_key = cache_key
        self.fallback_used = fallback_used
        self.fallback_name = fallback_name
        self.error = error
        self.throughput_unit = throughput_unit

        if self.duration_sec > 0:
            count_for_tp = output_count if output_count > 0 else input_count
            self.throughput_value = round(count_for_tp / self.duration_sec, 2)
        else:
            self.throughput_value = 0.0

        if details:
            self.details.update(details)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "duration_sec": round(self.duration_sec, 3),
            "input_count": self.input_count,
            "output_count": self.output_count,
            "throughput": {
                "value": self.throughput_value,
                "unit": self.throughput_unit,
            },
            "device": self.device,
            "cache": {
                "hit": self.cache_hit,
                "key": self.cache_key,
            },
            "fallback": {
                "used": self.fallback_used,
                "name": self.fallback_name,
            },
            "error": self.error,
            "details": self.details,
        }


class PerformanceProfiler:
    """Manages end-to-end reconstruction profiling and benchmark report generation."""

    def __init__(self, job_id: str, workspace_dir: Path):
        self.job_id = job_id
        self.workspace_dir = workspace_dir
        self.start_wall_time = time.perf_counter()
        self.end_wall_time = 0.0
        self.total_duration_sec = 0.0
        self.hardware = get_system_hardware_info()
        self.stages: Dict[str, StageMetric] = {}

    def start_stage(self, stage_name: str) -> StageMetric:
        metric = StageMetric(
            name=stage_name,
            status="running",
            start_time=time.perf_counter(),
            device=self.hardware.get("device_type", "cpu"),
        )
        self.stages[stage_name] = metric
        return metric

    def complete_stage(
        self,
        stage_name: str,
        *,
        input_count: int = 0,
        output_count: int = 0,
        throughput_unit: str = "items/sec",
        device: Optional[str] = None,
        cache_hit: bool = False,
        cache_key: Optional[str] = None,
        fallback_used: bool = False,
        fallback_name: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> StageMetric:
        metric = self.stages.get(stage_name)
        if not metric:
            metric = StageMetric(name=stage_name, start_time=time.perf_counter())
            self.stages[stage_name] = metric

        metric.finish(
            status="completed" if not cache_hit else "skipped",
            input_count=input_count,
            output_count=output_count,
            throughput_unit=throughput_unit,
            device=device or self.hardware.get("device_type", "cpu"),
            cache_hit=cache_hit,
            cache_key=cache_key,
            fallback_used=fallback_used,
            fallback_name=fallback_name,
            details=details,
        )
        return metric

    def skip_stage(self, stage_name: str, reason: str = "artifact_present", cache_key: Optional[str] = None) -> StageMetric:
        metric = self.stages.get(stage_name)
        if not metric:
            metric = StageMetric(name=stage_name, start_time=time.perf_counter())
            self.stages[stage_name] = metric

        metric.finish(
            status="skipped",
            cache_hit=True,
            cache_key=cache_key,
            details={"reason": reason},
        )
        return metric

    def fail_stage(self, stage_name: str, error: str) -> StageMetric:
        metric = self.stages.get(stage_name)
        if not metric:
            metric = StageMetric(name=stage_name, start_time=time.perf_counter())
            self.stages[stage_name] = metric

        metric.finish(
            status="failed",
            error=error,
        )
        return metric

    def analyze_bottleneck(self) -> Dict[str, Any]:
        active_durations = {
            name: stage.duration_sec
            for name, stage in self.stages.items()
            if stage.status == "completed" and stage.duration_sec > 0
        }
        if not active_durations:
            return {
                "bottleneck_stage": "none",
                "bottleneck_percentage": 0.0,
                "recommendation": "No stages completed wall-clock timing.",
            }

        bottleneck = max(active_durations, key=active_durations.get)
        b_time = active_durations[bottleneck]
        total_stage_time = sum(active_durations.values())
        pct = round((b_time / total_stage_time) * 100.0, 1) if total_stage_time > 0 else 0.0

        recommendations = {
            "frames": "Consider setting target_fps or quality_threshold to reduce frame candidate volume.",
            "sparse": "Use feature extraction limits or GPU acceleration for pycolmap feature matching.",
            "depth": "Consider stride downsampling or batch processing for Depth Anything V2 / SGBM.",
            "dense": "Increase voxel_size or reduce max_points_per_view for faster point cloud fusion.",
            "georef": "Check telemetry formatting or reduce alignment points count.",
        }

        rec = recommendations.get(bottleneck, f"Optimize algorithms or resources in {bottleneck} stage.")
        return {
            "bottleneck_stage": bottleneck,
            "bottleneck_duration_sec": round(b_time, 3),
            "bottleneck_percentage": pct,
            "recommendation": rec,
        }

    def finalize_and_save(self, status: str = "completed", error: str = "") -> Dict[str, Any]:
        self.end_wall_time = time.perf_counter()
        self.total_duration_sec = max(0.0, self.end_wall_time - self.start_wall_time)
        bottleneck_info = self.analyze_bottleneck()

        report = {
            "schema_version": "1.0",
            "job_id": self.job_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "error": error,
            "total_wall_clock_sec": round(self.total_duration_sec, 3),
            "total_wall_clock_ms": round(self.total_duration_sec * 1000.0, 2),
            "hardware": self.hardware,
            "bottleneck": bottleneck_info,
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
        }

        # Save performance.json in workspace
        try:
            perf_file = self.workspace_dir / "performance.json"
            perf_file.parent.mkdir(parents=True, exist_ok=True)
            with open(perf_file, "w") as f:
                json.dump(report, f, indent=2)
            log.info("performance_report_written", path=str(perf_file), total_sec=report["total_wall_clock_sec"])
        except Exception as exc:
            log.warning("performance_report_write_failed", error=str(exc))

        return report
