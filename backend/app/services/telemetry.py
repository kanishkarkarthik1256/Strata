"""Telemetry ingestion, mode classification, and video-frame synchronization.

Three-mode telemetry model:

- ``VIDEO_ONLY`` — no embedded, no external telemetry. Reconstruction runs in
  the local SfM coordinate system; GPS/georeferencing are reported as
  unavailable. This is an expected state for manually uploaded MP4s, not an
  ingestion failure.
- ``VIDEO_WITH_EMBEDDED_TELEMETRY`` — GPS found in the video container
  metadata (DJI / QuickTime tags via ffprobe).
- ``VIDEO_WITH_EXTERNAL_TELEMETRY`` — an external telemetry file (CSV) was
  supplied alongside the video.

This module never invents coordinates: samples lacking valid GPS values are
dropped and *counted*, never fabricated. Georeferencing consumers must check
``SyncReport`` sufficiency (``has_sufficient_track``) before claiming success.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from app.logging_config import get_logger

log = get_logger("drone_recon.services.telemetry")

#: Georeferencing requires at least this many *distinct matched* telemetry
#: samples spanning a real trajectory; fewer cannot anchor a spatial
#: alignment and must not produce georeferenced outputs.
MIN_GEOFERENCE_SAMPLES = 3

#: Legacy canonical header (still supported; now one case of the general
#: detection path in ``telemetry_schema``). Kept for diagnostics/reporting.
CSV_COLUMNS = ("timestamp", "latitude", "longitude", "altitude")

#: Sanity bounds for WGS84 input.
LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (-180.0, 180.0)
ALT_RANGE = (-500.0, 20000.0)  # metres; consumer drones never leave this


class TelemetryError(ValueError):
    """Unusable external telemetry input (HTTP 422 class).

    Raised only when the file genuinely lacks the information required for
    the requested operation — never merely because column names differ from
    the canonical schema (detection handles that; see
    ``app.services.telemetry_schema``)."""


class TelemetryMode(str, Enum):
    VIDEO_ONLY = "VIDEO_ONLY"
    VIDEO_WITH_EMBEDDED_TELEMETRY = "VIDEO_WITH_EMBEDDED_TELEMETRY"
    VIDEO_WITH_EXTERNAL_TELEMETRY = "VIDEO_WITH_EXTERNAL_TELEMETRY"


@dataclass(frozen=True)
class TelemetrySample:
    """One external telemetry record.

    ``timestamp_sec``/``altitude_m`` may be None (temporal sync / altitude
    unavailable — capabilities are independent). The remaining fields are
    optional enrichments populated by schema detection when the source CSV
    carries them; absent values stay None (never zero-filled).
    """

    timestamp_sec: float | None
    latitude: float | None
    longitude: float | None
    altitude_m: float | None
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    heading: float | None = None
    gps_accuracy_m: float | None = None
    satellites: float | None = None
    hdop: float | None = None
    vdop: float | None = None
    frame_number: float | None = None
    sequence: float | None = None


@dataclass
class SyncReport:
    """Outcome of matching video frames against telemetry samples.

    Serializes directly into manifests / pipeline reports.
    """

    mode: str = TelemetryMode.VIDEO_ONLY.value
    telemetry_samples: int = 0
    matched_frames: int = 0
    unmatched_frames: int = 0
    timestamp_offset_sec: float = 0.0
    max_offset_sec: float = 1.0
    gps_available: bool = False
    telemetry_quality: str = "unavailable"
    invalid_samples_dropped: int = 0
    # Median gap between consecutive valid samples (seconds; 0 when fewer
    # than two samples). Consumers use it to judge whether the clock offset
    # is large *relative to* sample density, not just in absolute terms.
    median_sample_spacing_sec: float = 0.0
    matched_frame_ids: list[str] = field(default_factory=list)
    # Runtime-only: the sample object matched to each id, same order. Not
    # serialized (the manifest gets counts, not payloads).
    matched_samples: list[TelemetrySample] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_sufficient_track(self) -> bool:
        """True when enough distinct, matched GPS samples exist to attempt
        georeferencing. Merely having GPS values does NOT satisfy this."""
        return (
            self.gps_available
            and self.matched_frames >= MIN_GEOFERENCE_SAMPLES
            and len(set(self.matched_frame_ids)) >= MIN_GEOFERENCE_SAMPLES
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "telemetry_samples": self.telemetry_samples,
            "matched_frames": self.matched_frames,
            "unmatched_frames": self.unmatched_frames,
            "timestamp_offset_sec": round(self.timestamp_offset_sec, 3),
            "max_offset_sec": self.max_offset_sec,
            "gps_available": self.gps_available,
            "telemetry_quality": self.telemetry_quality,
            "invalid_samples_dropped": self.invalid_samples_dropped,
            "median_sample_spacing_sec": round(self.median_sample_spacing_sec, 3),
            "sufficient_for_georeferencing": self.has_sufficient_track,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# External CSV ingestion
# ---------------------------------------------------------------------------


def load_external_telemetry(csv_path: Path) -> list[TelemetrySample]:
    """Parse any reasonable UAV telemetry CSV into canonical samples.

    Detection-driven (delimiter, encoding, header row, column aliases,
    units, timestamp interpretation — see ``telemetry_schema``); the fixed
    ``timestamp,latitude,longitude,altitude`` header remains supported as
    one case of the general path. Raises :class:`TelemetryError` ONLY when
    the file genuinely lacks the information needed for ingestion (no GPS
    columns with plausible values anywhere, or no data rows) — never merely
    because names differ from the canonical schema. Rows with invalid GPS
    are kept and flagged; :func:`synchronize` drops+counts them.
    """
    samples, _report = load_telemetry_with_schema(csv_path)
    return samples


def load_telemetry_with_schema(csv_path: Path) -> tuple[list[TelemetrySample], dict[str, Any]]:
    """Parse and also return the schema-detection report.

    The second element carries ``schema`` (column mapping, delimiter, header
    row, timestamp interpretation, capabilities), ``quality`` (row/GPS
    statistics) and the artifact paths written next to the source file
    (``telemetry_normalized.csv`` / ``telemetry_schema.json`` /
    ``telemetry_quality.json``) for upload-time diagnostics.

    Acceptance is capability-based: the file is rejected ONLY when no
    latitude/longitude *columns* could be detected at all (genuine failure —
    the requested GPS operation cannot proceed). Files whose GPS columns
    exist but carry degenerate values (all empty, all null-island) are
    accepted and reported; synchronize() drops+counts the rows and the
    georef stage skips honestly — quality problems are reported, not
    dressed up as schema failures.
    """
    from app.services.telemetry_schema import (
        dataset_to_samples,
        parse_telemetry_csv,
        write_telemetry_artifacts,
    )

    with open(csv_path, "rb") as probe:  # unreadable/empty file → real error
        if not probe.read(1):
            raise TelemetryError("Telemetry CSV is empty") from None
    dataset = parse_telemetry_csv(Path(csv_path))
    if not dataset.rows:
        raise TelemetryError("Telemetry CSV contains no data rows")
    if not dataset.has_gps:
        raise TelemetryError(
            "Telemetry CSV carries no GPS position columns — "
            f"{dataset.failure_reason()}"
        )
    report = {
        "schema": dataset.detection,
        "quality": dataset.quality,
        "artifacts": write_telemetry_artifacts(
            dataset, Path(csv_path).resolve().parent),
    }
    return dataset_to_samples(dataset), report


def derive_timestamps_from_frame_numbers(
    samples: list[TelemetrySample],
    video_fps: float,
    *,
    video_duration_sec: float | None = None,
) -> tuple[list[TelemetrySample], str]:
    """Give untimed telemetry rows a time base from their video frame index.

    Some drone logs record one row per video frame (``frame_id``/``frame_number``
    0..N) with GPS fixes but no timestamp column.  At a known constant fps the
    frame index IS a valid time base: ``t = frame_number / fps``.  This is not
    a fabricated clock — it is the video's own presentation timeline, measured
    from the extractor (the same timeline ``quality_report.json`` carries as
    ``timestamp_sec``).

    Returns (samples, provenance_note): the input list unchanged when nothing
    can be derived; otherwise a new list where every previously-untimed row
    carries ``t = frame_number / fps``.  Derivation is refused (input returned
    untouched + refusal reason) unless ALL of:
    * at least two untimed rows carry frame numbers (no partial timelines);
    * frame numbers are non-negative, finite and non-decreasing;
    * when the video duration is known, the derived span fits it — otherwise
      the mapping between frame ids and video frames is an assumption, not a
      measurement, and temporal sync would pair frames with the wrong flight
      moment (exactly the failure this derivation exists to prevent).
    """
    def _refuse(note: str) -> tuple[list[TelemetrySample], str]:
        return samples, note

    if not samples or video_fps <= 0:
        return _refuse("" if samples else "no telemetry rows")
    untyped = [s for s in samples if s.timestamp_sec is None]
    numbered = [s for s in untyped if s.frame_number is not None]
    if len(numbered) < 2:
        return _refuse("no frame_number column")
    # Partial coverage would build a two-speed clock — refuse.
    if len(numbered) != len(untyped):
        return _refuse(
            f"only {len(numbered)}/{len(untyped)} untimed rows carry frame numbers"
        )
    fns = np.array([float(s.frame_number) for s in numbered])
    if not np.all(np.isfinite(fns)) or np.any(fns < 0) or np.any(np.diff(fns) < 0):
        return _refuse("frame numbers are not finite/non-negative/non-decreasing")
    if video_duration_sec is not None and video_duration_sec > 0:
        span = float(fns[-1]) / video_fps
        if span > video_duration_sec * 1.05 + 0.5:
            return _refuse(
                f"frame numbers span {span:.1f}s at {video_fps:g} fps but the video "
                f"is only {video_duration_sec:.1f}s — frame index not usable as time base"
            )
    out = [
        replace(s, timestamp_sec=float(s.frame_number) / video_fps)
        if s.timestamp_sec is None else s
        for s in samples
    ]
    note = (
        f"timestamps derived from frame_number at {video_fps:g} fps "
        f"({len(numbered)} rows; video-frame timeline)"
    )
    log.info("telemetry_timestamps_derived_from_frame_numbers",
             rows=len(numbered), fps=video_fps)
    return out, note


def _valid_gps(sample: TelemetrySample) -> bool:
    """A sample carries a usable GPS fix only when lat/lon are present and in
    WGS84 range; altitude is optional — a fix whose altitude is implausible
    stays a valid 2D fix (the altitude value is simply not trustworthy)."""
    if sample.latitude is None or sample.longitude is None:
        return False
    if sample.latitude == 0.0 and sample.longitude == 0.0:
        return False
    if not (LAT_RANGE[0] <= sample.latitude <= LAT_RANGE[1]):
        return False
    return LON_RANGE[0] <= sample.longitude <= LON_RANGE[1]


# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------


def synchronize(
    frame_timestamps: list[tuple[str, float]],
    samples: list[TelemetrySample],
    *,
    max_offset_sec: float = 1.0,
) -> SyncReport:
    """Match ``(frame_id, timestamp_sec)`` pairs to nearest telemetry sample.

    Nearest-neighbour with per-frame tolerance ``max_offset_sec``; the
    reported ``timestamp_offset_sec`` is the median frame→sample delta (a
    robust clock-offset estimate). Frames match only against samples with
    valid GPS — a frame whose nearest sample has missing/out-of-range GPS
    stays unmatched and is counted, so a matched frame always carries a
    usable fix.
    """
    report = SyncReport(max_offset_sec=max_offset_sec)
    if samples:
        valid = [s for s in samples if _valid_gps(s)]
        report.invalid_samples_dropped = len(samples) - len(valid)
        report.telemetry_samples = len(samples)
        report.gps_available = bool(valid)
        timed = sorted(
            (s for s in valid if s.timestamp_sec is not None),
            key=lambda s: s.timestamp_sec,
        )
        if valid:
            report.telemetry_quality = _grade(valid)
            if len(timed) >= 2:
                gaps = sorted(
                    b.timestamp_sec - a.timestamp_sec
                    for a, b in zip(timed, timed[1:])
                    if b.timestamp_sec > a.timestamp_sec
                )
                if gaps:
                    report.median_sample_spacing_sec = gaps[len(gaps) // 2]
            if len(timed) < len(valid):
                report.notes.append(
                    f"{len(valid) - len(timed)} GPS sample(s) carry no usable timestamp — "
                    "positions remain available but temporal synchronization does not"
                )
        if report.invalid_samples_dropped:
            report.notes.append(
                f"{report.invalid_samples_dropped} sample(s) dropped for missing or out-of-range GPS values"
            )
    if not frame_timestamps:
        if samples:
            report.mode = TelemetryMode.VIDEO_WITH_EXTERNAL_TELEMETRY.value
            report.notes.append("no frames to match against telemetry")
        return report

    if not samples:
        report.unmatched_frames = len(frame_timestamps)
        return report

    report.mode = TelemetryMode.VIDEO_WITH_EXTERNAL_TELEMETRY.value
    timed_sorted = sorted(
        (s for s in valid if s.timestamp_sec is not None),
        key=lambda s: s.timestamp_sec,
    )
    if not timed_sorted:
        report.notes.append(
            "temporal synchronization unavailable — no timestamped GPS samples; "
            "GPS positions remain available for spatial use"
        )
        return report
    sample_times = [s.timestamp_sec for s in timed_sorted]
    deltas: list[float] = []
    for frame_id, ts in frame_timestamps:
        # binary search for nearest sample
        lo, hi = 0, len(sample_times)
        while lo < hi:
            mid = (lo + hi) // 2
            if sample_times[mid] < ts:
                lo = mid + 1
            else:
                hi = mid
        best_idx = None
        best_dist = None
        for cand in (lo - 1, lo):
            if 0 <= cand < len(sample_times):
                dist = abs(sample_times[cand] - ts)
                if best_dist is None or dist < best_dist:
                    best_dist, best_idx = dist, cand
        if best_idx is None or best_dist > max_offset_sec:
            report.unmatched_frames += 1
            continue
        deltas.append(sample_times[best_idx] - ts)
        report.matched_frames += 1
        report.matched_frame_ids.append(frame_id)
        report.matched_samples.append(timed_sorted[best_idx])

    if deltas:
        deltas.sort()
        mid = len(deltas) // 2
        report.timestamp_offset_sec = (
            deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2
        )
    if report.unmatched_frames:
        report.notes.append(
            f"{report.unmatched_frames} frame(s) had no telemetry sample within {max_offset_sec:.2f}s"
        )
    if report.matched_frames == 0:
        report.telemetry_quality = "no_overlap"
        report.notes.append(
            "telemetry timestamps never overlap the video timeline — "
            "check that the telemetry was recorded for this video"
        )
    return report


def _grade(valid: list[TelemetrySample]) -> str:
    """Coarse quality grade from sample count and span.

    Timestamp-less samples cap the grade at ``low``: positions may be real,
    but without a timeline the track cannot be synchronized honestly."""
    if len(valid) < MIN_GEOFERENCE_SAMPLES:
        return "insufficient"
    timed = [s for s in valid if s.timestamp_sec is not None]
    if len(timed) < 2:
        return "low"
    span = timed[-1].timestamp_sec - timed[0].timestamp_sec
    if span <= 0:
        return "insufficient"
    if len(valid) < 10 or span < 5.0:
        return "low"
    return "good"


def format_telemetry_detection(report: dict[str, Any]) -> str:
    """Human-readable TELEMETRY DETECTED summary (spec §17).

    Rendered from the schema report: detected mappings with source columns,
    timestamp interpretation, GPS coverage, schema confidence. Ambiguous
    fields surface as ``[review mapping]`` hints. This is diagnostics only —
    the canonical sync report remains the georef stage's ``sync`` block.
    """
    schema = report.get("schema", {})
    quality = report.get("quality", {})
    fields = schema.get("fields", {})
    caps = schema.get("capabilities", {})
    lines = ["TELEMETRY DETECTED"]
    order = ["timestamp", "latitude", "longitude", "altitude",
             "roll", "pitch", "yaw", "heading"]
    for canon in order:
        m = fields.get(canon)
        if m:
            lines.append(
                f"✓ {canon} → '{m['source_column']}'"
                + (f" ({m['detected_unit']})" if m.get("detected_unit") not in (None, "native") else "")
            )
    for canon, m in fields.items():
        if canon not in order:
            lines.append(f"✓ {canon} → '{m['source_column']}'")
    ts = schema.get("timestamp_interpretation") or {}
    if ts:
        lines.append(
            f"Timestamp format: {ts.get('kind')}"
            + (" / relative flight time" if ts.get("epoch_based") is False else "")
        )
    total = quality.get("total_rows") or 0
    if total:
        lines.append(f"Rows: {total:,}")
    valid = caps.get("gps_position", {}).get("valid_rows") or 0
    if total and valid:
        lines.append(f"GPS coverage: {100.0 * valid / total:.1f}%")
    sync = caps.get("temporal_sync", {})
    lines.append(
        "Temporal synchronization: "
        + ("Available" if sync.get("available") else "UNAVAILABLE — positions still usable spatially")
    )
    conf = (quality.get("mapping_confidence") or {}).get("mean")
    if conf is not None:
        label = "High" if conf >= 0.8 else "Medium" if conf >= 0.6 else "Low"
        lines.append(f"Schema confidence: {label} ({conf:.2f})")
    for amb in schema.get("ambiguous", [])[:4]:
        if amb.get("ranked"):
            names = " vs ".join(f"'{r['column']}'" for r in amb["ranked"])
            lines.append(f"⚠ {amb['field']}: {names} [review mapping]")
    return " | ".join(lines)


def resolve_mode(
    *,
    embedded_gps: dict | None = None,
    telemetry_csv_path: Path | None = None,
) -> TelemetryMode:
    """Classify the run's telemetry mode from its inputs.

    ``embedded_gps`` is the video-container GPS dict (or None). External
    telemetry wins when present (it is per-sample and richer than a single
    embedded fix).
    """
    if telemetry_csv_path is not None:
        return TelemetryMode.VIDEO_WITH_EXTERNAL_TELEMETRY
    if embedded_gps:
        return TelemetryMode.VIDEO_WITH_EMBEDDED_TELEMETRY
    return TelemetryMode.VIDEO_ONLY
