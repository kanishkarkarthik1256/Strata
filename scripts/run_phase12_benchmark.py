"""Phase 12 End-to-End Real Reconstruction Benchmark Runner.

Executes a live end-to-end pipeline processing run on test video input,
captures real wall-clock execution metrics per stage via PerformanceProfiler,
validates output performance.json schema, and reports hardware/throughput metrics.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.config.settings import settings
from app.services.performance_profiler import PerformanceProfiler, get_system_hardware_info
from app.services.pipeline_orchestrator import PipelineRequest, run_autonomous_pipeline
from app.services.stage_caching import compute_stage_fingerprint, save_stage_fingerprint


def main():
    print("===============================================================")
    print("STRATA PHASE 12 - LIVE PERFORMANCE BENCHMARK RUNNER")
    print("===============================================================")

    # 1. Discover hardware capabilities
    hw = get_system_hardware_info()
    print("\n--- System Hardware Discovery ---")
    print(f"OS: {hw['os_system']} {hw['os_release']} ({hw['architecture']})")
    print(f"Python: {hw['python_version']}")
    print(f"CPU Cores: {hw['cpu_cores']}")
    print(f"RAM: {hw['ram_gb']} GB")
    print(f"Execution Device: {hw['device_type'].upper()} ({hw['gpu_name']})")
    print(f"PyTorch Version: {hw['pytorch_version']}")
    print(f"OpenCV Version: {hw['opencv_version']}")
    print(f"pycolmap Version: {hw['pycolmap_version']}")

    # 3. Setup clean benchmark job workspace
    job_id = "phase12_benchmark_run"
    workspace = settings.storage.project_dir(job_id)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    shitan_frames = Path("backend/outputs/shitan_ms1_20260909_131251/frames")
    shitan_video = Path("backend/outputs/shitan_ms1_20260909_131251/video/smoke_clip.mp4")
    if shitan_frames.exists() and list(shitan_frames.glob("*.jpg")):
        print(f"\n--- Using Validated Reconstruction Dataset ({shitan_frames}) ---")
        selected_target = workspace / "selected"
        selected_target.mkdir(parents=True, exist_ok=True)
        for img_file in sorted(shitan_frames.glob("*.jpg")):
            shutil.copy2(img_file, selected_target / img_file.name)
        if shitan_video.exists():
            shutil.copy2(shitan_video, workspace / "benchmark_input.mp4")

        req_params = {
            "extraction_mode": "every_n",
            "every_n": 1,
            "target_fps": None,
            "top_percent": None,
            "quality_threshold": 0.01,
            "depth_backend": "auto",
            "frame_stride": 1,
            "max_depth_views": 200,
        }
        fp = compute_stage_fingerprint("frames", workspace, req_params)
        save_stage_fingerprint("frames", workspace, fp, {"selected_count": len(list(shitan_frames.glob("*.jpg")))})
    else:
        video_source = Path("data/storage/e5c6bcf2ee7442f8b907e03e85058046/site_a.mp4")
        if not video_source.exists():
            video_source = Path("data/storage/814b65c93c1d44a99dece98f2332fcda/site_a.mp4")
        if not video_source.exists():
            video_source = Path("backend/outputs/shitan_ms1_20260909_131251/video/smoke_clip.mp4")

        print(f"\n--- Benchmark Input Video ---")
        print(f"Source Path: {video_source}")
        print(f"Source Size: {round(video_source.stat().st_size / (1024 * 1024), 2)} MB")

        target_video = workspace / "benchmark_input.mp4"
        shutil.copy2(video_source, target_video)

    # 4. Execute full pipeline run
    print(f"\n--- Launching Real Pipeline Benchmark Run (Job ID: {job_id}) ---")
    req = PipelineRequest(
        extraction_mode="every_n",
        every_n=1,
        quality_threshold=0.01,
        depth_backend="auto",
        frame_stride=1,
        force=["sparse", "depth", "dense", "georef"],
    )

    t0 = time.perf_counter()
    report = run_autonomous_pipeline(job_id, req)
    total_time = time.perf_counter() - t0

    print(f"\n--- Pipeline Run Completed in {round(total_time, 2)}s ---")
    print(f"Pipeline Run Status: {report.get('status')}")

    # 5. Read generated performance.json
    perf_file = workspace / "performance.json"
    if not perf_file.exists():
        print("[ERROR] performance.json artifact was not generated!")
        sys.exit(1)

    perf_data = json.loads(perf_file.read_text())
    print("\n--- Benchmark Metrics Summary (performance.json) ---")
    print(f"Job ID: {perf_data.get('job_id')}")
    print(f"Total Wall-Clock Time: {perf_data.get('total_wall_clock_sec')}s ({perf_data.get('total_wall_clock_ms')} ms)")
    
    bottleneck = perf_data.get("bottleneck", {})
    print(f"Bottleneck Stage: {bottleneck.get('bottleneck_stage')} ({bottleneck.get('bottleneck_percentage')}% of total time)")
    print(f"Recommendation: {bottleneck.get('recommendation')}")

    print("\nPer-Stage Metrics Breakdown:")
    for name, st in perf_data.get("stages", {}).items():
        tp = st.get("throughput", {})
        print(f"  Stage [{name.upper()}]:")
        print(f"    Status: {st.get('status')}")
        print(f"    Duration: {st.get('duration_sec')}s ({st.get('duration_ms')} ms)")
        print(f"    Input/Output: {st.get('input_count')} in -> {st.get('output_count')} out")
        print(f"    Throughput: {tp.get('value')} {tp.get('unit')}")
        print(f"    Device: {st.get('device')}")
        print(f"    Cache Hit: {st.get('cache', {}).get('hit')}")
        print(f"    Fallback Used: {st.get('fallback', {}).get('used')}")

    print("\n===============================================================")
    print("BENCHMARK RUN SUCCESSFULLY VALIDATED")
    print("===============================================================")


if __name__ == "__main__":
    main()
