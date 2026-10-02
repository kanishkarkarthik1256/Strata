#!/usr/bin/env python3
"""Run Controlled Synthetic Metric Accuracy Benchmark for STRATA.

Generates synthetic 3D ground-truth geometry, camera trajectories, check points,
and distance constraints, then computes and verifies metric accuracy calculations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.synthetic_benchmark import run_synthetic_benchmark


def main() -> None:
    print("--- Running Controlled Synthetic Metric Accuracy Benchmark ---")
    output_dir = backend_dir.parent / "data" / "storage" / "synthetic_metric_benchmark"
    report = run_synthetic_benchmark(output_dir)

    print(f"\nBenchmark Output Path: {output_dir}")
    print(f"Capability State:      {report.get('capability_state')}")
    print(f"Is Validated:          {report.get('is_validated')}")
    print(f"Is Synthetic:          {report.get('is_synthetic')}")
    print(f"Status Note:           {report.get('validation_status_text')}\n")

    print("Summary Metrics Table:")
    print("-" * 80)
    print(f"{'Metric':<35} | {'Result':<20} | {'Status':<15}")
    print("-" * 80)
    for row in report.get("summary_table", []):
        metric = row.get("metric", "")
        result = row.get("result", "")
        status = row.get("status", "")
        print(f"{metric:<35} | {result:<20} | {status:<15}")
    print("-" * 80)

    # Verification assertions
    assert report.get("is_synthetic") is True, "Synthetic benchmark must be labeled is_synthetic=True"
    assert report.get("capability_state") == "METRIC_VALIDATED", "Synthetic benchmark must reach METRIC_VALIDATED state"
    assert len(report.get("summary_table", [])) > 0, "Summary table must not be empty"

    print("\nSUCCESS: Controlled Synthetic Metric Benchmark validated successfully.")


if __name__ == "__main__":
    main()
