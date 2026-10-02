"""Minimal in-process metrics registry (Phase 10, no external deps).

Exposes counters/gauges as Prometheus text format at ``/api/metrics``.
Values are only ever derived from real events (HTTP requests observed by
middleware, queue jobs processed, DB reachability probes) — nothing is
fabricated.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Optional


class MetricsRegistry:
    """Thread-safe flat registry of counters + gauges."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict] = defaultdict(dict)  # name → {(labels_tuple): value}
        self._gauges: dict[str, dict] = defaultdict(dict)

    # -- counters -----------------------------------------------------------

    def inc(self, name: str, labels: Optional[dict] = None, value: int = 1) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._counters[name][key] = self._counters[name].get(key, 0) + value

    # -- gauges -------------------------------------------------------------

    def set_gauge(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._gauges[name][key] = value

    # -- render -------------------------------------------------------------

    def render(self) -> str:
        out: list[str] = []
        with self._lock:
            for name, series in sorted(self._counters.items()):
                out.append(f"# TYPE {name} counter")
                for key, value in sorted(series.items()):
                    out.append(_format_sample(name, key, value))
            for name, series in sorted(self._gauges.items()):
                out.append(f"# TYPE {name} gauge")
                for key, value in sorted(series.items()):
                    out.append(_format_sample(name, key, value))
        return "\n".join(out) + "\n"


def _format_sample(name: str, key: tuple, value) -> str:
    if key:
        labels = ",".join(f'{k}="{v}"' for k, v in key)
        return f"{name}{{{labels}}} {value}"
    return f"{name} {value}"


metrics = MetricsRegistry()
