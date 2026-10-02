"""CPU thread budget — how many threads a numerical kernel may use.

Every kernel here (torch/OpenMP BLAS, OpenCV, Ceres inside pycolmap) defaults
to the LOGICAL processor count. On an SMT machine that oversubscribes: the
extra threads contend for the same SIMD units and memory bandwidth, so a
throughput-bound kernel gets measurably *slower*. Measured on the reference
host (4 physical / 8 logical cores, Depth Anything V2 vitb, 518x924 input):

    threads=8   8.50-9.05 s/frame      <- os.cpu_count() default
    threads=6   5.92-6.08 s/frame
    threads=4   5.55-5.66 s/frame      <- physical cores

COLMAP independently warns about the same default ("Your current options use
the maximum number of threads on the machine to extract features"). So the
budget is the PHYSICAL core count, not the logical one.

This module is deliberately stdlib-only: it is imported by ``app.main`` to set
the BLAS thread environment *before* torch/OpenMP initialise, and importing
torch first would freeze the wrong defaults.

Set ``AI_CPU_THREADS`` to override (an explicit operator decision always wins).
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

#: Environment variables honoured as an explicit operator override, in the
#: order of specificity. ``AI_CPU_THREADS`` is this module's own knob; the
#: others are the conventional knobs operators already set.
OVERRIDE_VARS = ("AI_CPU_THREADS", "TORCH_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _darwin_physical_cores() -> int | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            capture_output=True, text=True, timeout=5,
        )
        n = int((out.stdout or "").strip())
        return n if n > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _linux_physical_cores() -> int | None:
    """Distinct (physical id, core id) pairs in /proc/cpuinfo."""
    try:
        text = Path("/proc/cpuinfo").read_text(errors="ignore")
    except OSError:
        return None
    pairs: set[tuple[str, str]] = set()
    current: dict[str, str] = {}
    for line in [*text.splitlines(), ""]:
        if not line.strip():
            if "physical id" in current and "core id" in current:
                pairs.add((current["physical id"], current["core id"]))
            current = {}
            continue
        key, _, value = line.partition(":")
        current[key.strip()] = value.strip()
    return len(pairs) or None


@lru_cache(maxsize=1)
def physical_cpu_count() -> int:
    """Physical cores when the OS can prove it, else the logical count.

    Falling back to the logical count keeps behaviour identical on platforms
    where the physical topology is not exposed — never a guessed halving.
    """
    detected = None
    if sys.platform == "darwin":
        detected = _darwin_physical_cores()
    elif sys.platform.startswith("linux"):
        detected = _linux_physical_cores()
    return detected or os.cpu_count() or 1


@lru_cache(maxsize=1)
def inference_threads() -> int:
    """Threads one numerical kernel should use: operator override, else physical."""
    for var in OVERRIDE_VARS:
        raw = (os.environ.get(var) or "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return physical_cpu_count()


def apply_thread_environment() -> int:
    """Publish the budget to the BLAS/OpenMP environment. Returns the budget.

    Must run before torch (or anything that initialises OpenMP) is imported,
    because those runtimes read these variables once, at load time.
    """
    budget = inference_threads()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(budget))
    return budget
