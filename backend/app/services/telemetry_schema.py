"""Universal UAV telemetry CSV ingestion (schema detection + normalization).

Replaces the fixed ``timestamp,latitude,longitude,altitude`` contract with an
inspect → detect → map → validate → normalize pipeline:

1. **Format detection** — delimiter (`,` ``;`` tab ``|``), encoding (UTF-8
   BOM / UTF-8 / latin-1), quoted fields, and the *actual* header row (some
   exporters prepend metadata blocks).
2. **Column mapping** — a canonical-field alias dictionary + normalized-token
   fuzzy matching. Names propose; sampled values validate (lat/lon ranges,
   timestamp magnitude/intervals). Value evidence never *creates* a mapping,
   it only confirms or rejects a name-based one.
3. **Unit interpretation** — timestamps (ISO-8601 / unix s / ms / µs /
   relative s / ms / µs), angles (deg default, radians converted), altitude
   (m default, explicit ft converted, ambiguity flagged, never silently
   guessed), longitude 0–360 shift. Every conversion is recorded.
4. **Capability model** — GPS position (lat+lon), altitude, and temporal sync
   are assessed INDEPENDENTLY. A CSV with GPS but no timestamp is accepted:
   positions are real, temporal synchronization is reported unavailable.
5. **Artifacts** — ``telemetry_normalized.csv``, ``telemetry_schema.json``,
   ``telemetry_quality.json``.

This module never invents telemetry: unmapped fields stay absent, ambiguous
units stay flagged, rows failing validation keep their raw values and a flag.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.logging_config import get_logger

log = get_logger("drone_recon.services.telemetry_schema")

# ---------------------------------------------------------------------------
# Canonical fields + alias dictionary
# ---------------------------------------------------------------------------

#: All canonical fields this layer can recognize (superset of what the
#: reconstruction consumes). Each entry: canonical name → ordered aliases.
#: Matching is token-based after normalization (see ``_name_scores``).
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": (
        "timestamp", "time", "datetime", "date_time", "utc", "utc_time",
        "gps_time", "gps_timestamp", "unix_time", "unix_timestamp", "epoch",
        "epoch_time", "time_ms", "timestamp_ms", "timestamp_us",
        "timestamp_usec", "time_us", "time_usec", "time_boot_ms",
        "time_boot_us", "flight_time", "elapsed_time", "time_s",
        # NOTE: generic single-letter aliases ("t", "x", "y", "z") are
        # deliberately NOT aliases: in UAV workflows those are overwhelmingly
        # LOCAL pose coordinates / short tokens, and mapping them as GPS
        # lat/lon would fabricate geographic meaning (spec: "Do NOT map
        # arbitrary columns just because the names are vaguely similar").
        # Pose CSVs (frame_id,x,y,z,qw,...) have their own dedicated adapter.
    ),
    "latitude": (
        "latitude", "lat", "gps_latitude", "gps_lat", "latitude_deg",
        "lat_deg", "latitude_degrees", "lat_degrees", "gps_lat_deg",
        "gps_latitude_deg", "lat_dd",
    ),
    "longitude": (
        "longitude", "lon", "lng", "long", "gps_longitude", "gps_lon",
        "longitude_deg", "lon_deg", "longitude_degrees", "lon_degrees",
        "gps_lon_deg", "gps_longitude_deg", "lon_dd",
    ),
    "altitude": (
        "altitude", "alt", "gps_altitude", "gps_alt", "altitude_m", "alt_m",
        "altitude_meter", "altitude_meters", "height", "height_m",
        "elevation", "elevation_m", "gps_height", "global_alt",
        "global_altitude", "alt_msl", "altitude_msl",
    ),
    # Relative/AGL altitudes are recognized but NEVER treated as absolute
    # GPS altitude (different reference — see ``_altitude_reference``).
    "altitude_agl": (
        "relative_alt", "relative_altitude", "rel_alt", "agl", "height_agl",
        "relative_alt_m", "alt_relative", "altitude_rel",
    ),
    "roll": ("roll", "roll_deg", "bank", "bank_angle"),
    "pitch": ("pitch", "pitch_deg"),
    "yaw": ("yaw", "yaw_deg", "heading_deg", "course_over_ground"),
    "heading": ("heading", "heading_deg", "compass", "course", "cog"),
    "velocity_x": ("velocity_x", "vx", "vel_x", "speed_north", "vn"),
    "velocity_y": ("velocity_y", "vy", "vel_y", "speed_east", "ve"),
    "velocity_z": ("velocity_z", "vz", "vel_z", "speed_down", "vd"),
    "speed": ("speed", "speed_kmh", "speed_mph", "ground_speed", "gs"),
    "gps_accuracy": ("gps_accuracy", "accuracy", "pos_accuracy", "eph"),
    "horizontal_accuracy": ("horizontal_accuracy", "h_acc", "hacc", "acc_h"),
    "vertical_accuracy": ("vertical_accuracy", "v_acc", "vacc", "acc_v"),
    "satellites": ("satellites", "sats", "num_sats", "sat", "satellites_visible"),
    "hdop": ("hdop", "h_dop"),
    "vdop": ("vdop", "v_dop"),
    "frame_number": ("frame_number", "frame", "frame_id", "frame_idx", "fn"),
    "sequence": ("sequence", "seq", "sample_number", "index", "row"),
}

#: Fields the reconstruction layer can consume per-row (built samples).
SAMPLE_FIELDS = (
    "roll", "pitch", "yaw", "heading", "gps_accuracy", "satellites",
    "hdop", "vdop", "frame_number", "sequence",
)

#: Delimiters considered during format detection (most likely first).
DELIMITERS = (",", ";", "\t", "|")

LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (-180.0, 180.0)
ALT_RANGE = (-500.0, 20000.0)  # metres

#: Name-based mapping score below this ⇒ the column is not mapped even when
#: its values look plausible (prevents ``x``/``y`` pose columns masquerading
#: as coordinates). Value evidence multiplies confidence, never substitutes
#: for name evidence.
MIN_NAME_SCORE = 0.5

#: Fraction of sampled values that must be plausible for a mapped column's
#: validation to pass (rows failing remain flagged individually).
VALUE_PASS_FRACTION = 0.6

_HEADER_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _norm_header(name: str) -> str:
    """Normalize a raw header cell: lowercase, trim, drop BOM/units in
    parentheses/degree suffixes, collapse separators to ``_``."""
    s = name.strip().lstrip("﻿").strip()
    s = re.sub(r"\((?:[^()]*)\)", " ", s)          # "(deg)", "(ms)" hints kept separately
    s = s.lower()
    s = s.replace("°", " ")  # bare degree mark carries no token of its own
    tokens = [t for t in _HEADER_TOKEN_SPLIT.split(s) if t]
    # Drop bare unit tokens for *matching* (they are unit hints, not identity).
    unit_tokens = {"deg", "degree", "degrees", "m", "meter", "meters", "ft",
                   "feet", "ms", "millis", "milliseconds", "us", "usec",
                   "micros", "microseconds", "s", "sec", "secs", "seconds",
                   "kmh", "kph", "mph"}
    core = [t for t in tokens if t not in unit_tokens]
    return "_".join(core) if core else "_".join(tokens)


# ---------------------------------------------------------------------------
# Name-based scoring
# ---------------------------------------------------------------------------

def _name_scores(header: str) -> dict[str, float]:
    """Score how well a header cell names each canonical field (0..1)."""
    norm = _norm_header(header)
    if not norm or norm == "_":
        return {}
    tokens = set(norm.split("_"))
    # Pose/quaternion/geometry tokens disqualify pose columns from telemetry
    # semantics: x/y/z/qw…/fov_* in UAV exports are camera-pose or intrinsics
    # columns (local metric frame), never telemetry. Without this guard the
    # fuzzy matcher claims x→velocity_x, z→velocity_z, fov_vertical→acc_v.
    pose_tokens = tokens & {"qw", "qx", "qy", "qz", "quat", "quaternion",
                            "fov", "focal", "cx", "cy", "intrinsics"}
    if tokens <= {"x", "y", "z"} or pose_tokens:
        return {}
    scores: dict[str, float] = {}
    for canon, aliases in FIELD_ALIASES.items():
        best = 0.0
        for alias in aliases:
            a_norm = _norm_header(alias)
            a_tokens = set(a_norm.split("_"))
            if not a_tokens:
                continue
            if norm == a_norm:
                best = max(best, 1.0)
            elif a_tokens <= tokens:
                # Multi-token alias fully present ("gps latitude" ⊆ header).
                best = max(best, 0.85 + 0.05 * (len(a_tokens) - 1))
            elif a_tokens & tokens:
                # Single shared token ("time" in "time_boot_ms").
                best = max(best, 0.6)
            elif len(a_norm) >= 4 and (a_norm in norm or norm in a_norm):
                # Substring without token boundary — weak evidence only.
                best = max(best, 0.45)
        # Disambiguation penalties among similar fields.
        if canon == "yaw":
            for rel in ("heading", "course", "compass", "cog"):
                if rel in tokens and "yaw" not in tokens:
                    best = 0.0
        if canon == "heading" and "yaw" in tokens:
            best = 0.0
        # Altitude reference is semantic, not spelling: a column is AGL
        # only when its name says relative/AGL, absolute otherwise. Without
        # this guard, 'altitude_rel' shares the 'altitude' token and a
        # relative_alt column would score 1.0 as absolute altitude.
        agl_tokens = tokens & {"relative", "rel", "agl"}
        if canon == "altitude" and agl_tokens:
            best = 0.0
        if canon == "altitude_agl" and not agl_tokens:
            best = 0.0
        scores[canon] = best
    return scores


# ---------------------------------------------------------------------------
# Value validation helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

_DMS_RE = re.compile(
    r"^\s*(?P<sign>[NSEWnsew]?[-+]?)(?P<d>\d{1,3})(?:[°d:\s]+(?P<m>\d{1,2}(?:\.\d+)?))?"
    r"(?:['m:\s]+(?P<s>\d{1,2}(?:\.\d+)?))?\s*[°\"″]?\s*(?P<hem>[NSEWnsew])?\s*$"
)


def _parse_number(cell: str) -> float | None:
    """Parse a plain decimal number; ``None`` when not numeric."""
    c = cell.strip()
    if not c:
        return None
    if _NUM_RE.match(c):
        try:
            v = float(c)
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def _parse_dms(cell: str) -> float | None:
    """Parse a DMS string like ``11°01'00.5"N`` into signed decimal degrees.
    Returns None when the cell is not DMS-shaped."""
    m = _DMS_RE.match(cell.strip())
    if not m:
        return None
    hemi = (m.group("hem") or m.group("sign") or "").upper()
    if not hemi or hemi not in "NSEW":
        return None  # required hemisphere — plain numbers are not DMS
    d = float(m.group("d"))
    mnt = float(m.group("m") or 0.0)
    sec = float(m.group("s") or 0.0)
    dec = d + mnt / 60.0 + sec / 3600.0
    if hemi in "SW":
        dec = -dec
    return dec


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def _parse_iso(cell: str) -> float | None:
    """Parse an ISO-8601 / UTC datetime into unix seconds (UTC)."""
    c = cell.strip()
    if not _ISO_RE.match(c):
        return None
    try:
        txt = c.replace("Z", "+00:00")
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)  # naive treated as UTC, recorded
        return dt.timestamp()
    except ValueError:
        return None


def _in_range(v: float, rng: tuple[float, float]) -> bool:
    return rng[0] <= v <= rng[1]


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2


# ---------------------------------------------------------------------------
# Timestamp interpretation
# ---------------------------------------------------------------------------

def _classify_timestamps(
    numeric: list[float], header: str
) -> dict[str, Any]:
    """Determine the timestamp representation from name hints, magnitudes and
    the sampling-interval sanity of each candidate unit.

    Returns ``{kind, unit_seconds_factor, epoch_based, confidence, basis}``.
    """
    if not numeric:
        return {"kind": "unknown", "unit_seconds_factor": None,
                "epoch_based": None, "confidence": 0.0, "basis": "no values"}

    h = header.lower()
    hints_ms = any(k in h for k in ("_ms", "milli", "time_boot_ms"))
    hints_us = any(k in h for k in ("_us", "_usec", "micro"))
    hints_epoch = any(k in h for k in ("unix", "epoch", "utc", "gps_time", "timestamp"))
    hints_rel = any(k in h for k in ("boot", "elapsed", "flight", "relative", "mission"))

    vals = sorted(abs(v) for v in numeric)
    med_mag = _median(vals)
    pos = sorted(b - a for a, b in zip(sorted(numeric), sorted(numeric)[1:]) if b > a)
    med_dt = _median(pos) if pos else None

    def dt_sane(seconds: float) -> bool:
        return seconds is not None and 1e-3 <= seconds <= 60.0

    def dt_in(factor: float) -> bool:
        # med_dt is in NATIVE units; the candidate interpretation converts it
        # to seconds by MULTIPLYING by the factor (1000 ms × 1e-3 = 1 s).
        return med_dt is not None and dt_sane(med_dt * factor)

    # ISO handled before this call. Numeric families, most specific first:
    candidates: list[tuple[str, float, bool, list[str]]] = []  # kind, factor, epoch, basis
    if med_mag >= 1e14:
        candidates.append(("unix_microseconds", 1e-6, True, ["magnitude≥1e14"]))
        candidates.append(("relative_microseconds", 1e-6, False, ["magnitude≥1e14"]))
    elif med_mag >= 1e11:
        candidates.append(("unix_milliseconds", 1e-3, True, ["magnitude~1e11–1e14"]))
        candidates.append(("relative_milliseconds", 1e-3, False, ["magnitude~1e11–1e14"]))
    elif med_mag >= 1e9:
        candidates.append(("unix_seconds", 1.0, True, ["magnitude~1e9 (epoch range)"]))
        candidates.append(("unix_milliseconds", 1e-3, True, ["magnitude~1e9"]))
    else:
        candidates.append(("relative_seconds", 1.0, False, ["magnitude<1e9"]))
        candidates.append(("relative_milliseconds", 1e-3, False, ["magnitude<1e9"]))
        candidates.append(("relative_microseconds", 1e-6, False, ["magnitude<1e9"]))

    # Name-hint reordering.
    def boost(kind: str, basis: list[str]) -> list[str]:
        if hints_us and "micro" in kind: return basis + ["name hint: µs"]
        if hints_ms and "milli" in kind: return basis + ["name hint: ms"]
        if hints_epoch and kind.startswith("unix"): return basis + ["name hint: epoch/unix"]
        if hints_rel and kind.startswith("relative"): return basis + ["name hint: relative"]
        return basis

    scored: list[tuple[float, str, float, bool, list[str]]] = []
    for kind, factor, epoch, basis in candidates:
        basis = boost(kind, basis)
        if not dt_in(factor):
            continue
        conf = 0.7 + 0.2 * bool(basis and basis[-1].startswith("name"))
        scored.append((conf, kind, factor, epoch, basis))

    # Epoch vs relative preference within the winning magnitude family.
    if scored:
        scored.sort(key=lambda t: (-t[0],))
        conf, kind, factor, epoch, basis = scored[0]
        return {"kind": kind, "unit_seconds_factor": factor, "epoch_based": epoch,
                "confidence": round(min(conf, 0.95), 2), "basis": basis,
                "median_interval_native": med_dt}
    # No unit produced a sane sampling interval — pick by magnitude alone,
    # flagged low-confidence.
    kind, factor, epoch = candidates[0][0], candidates[0][1], candidates[0][2]
    return {"kind": kind, "unit_seconds_factor": factor, "epoch_based": epoch,
            "confidence": 0.4,
            "basis": ["no consistent sampling interval; magnitude heuristic only"],
            "median_interval_native": med_dt}


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _decode(path: Path) -> tuple[str, str]:
    """Read the file as text; returns (text, encoding). UTF-8 BOM handled,
    latin-1 as lossless fallback."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def _detect_delimiter(lines: list[str]) -> tuple[str, float]:
    """Pick the delimiter that appears most consistently across lines.

    Comma-as-decimal-marker (``0,000000;52,35...``) puts a comma in EVERY
    value; a naive count then picks ',' and shreds the file. Real delimiters
    also split the HEADER into multiple cells, so candidates are validated
    against the header line's cell count — a delimiter that cannot split the
    header is a decimal marker, not a separator."""
    header_line = next((ln for ln in lines if ln.strip()), "")
    best, best_score = ",", -1.0
    for d in DELIMITERS:
        counts = [ln.count(d) for ln in lines[:40] if ln.strip()]
        if not counts:
            continue
        modal = max(set(counts), key=counts.count)
        if modal == 0:
            continue
        consistency = counts.count(modal) / len(counts)
        # The delimiter must actually split the header into ≥2 cells.
        if d not in header_line or len(_split_row(header_line, d)) < 2:
            continue
        score = consistency * min(modal, 8)
        if score > best_score:
            best, best_score = d, score
    if best_score < 0:
        best, best_score = ",", 0.0
    conf = min(0.99, 0.6 + best_score / 20)
    return best, round(conf, 2)


def _split_row(line: str, delimiter: str) -> list[str]:
    return next(csv.reader(io.StringIO(line), delimiter=delimiter))


@dataclass
class _Table:
    header: list[str]
    rows: list[list[str]]
    header_row_index: int  # 0-based line index of the header
    delimiter: str
    encoding: str
    decimal_comma: bool = False


def _load_table(path: Path) -> _Table:
    """Detect encoding, delimiter and the real header row; parse the table.

    The header is the first line that name-maps at least two distinct
    canonical fields (metadata lines like ``Aircraft: ABC`` map to none).
    """
    text, encoding = _decode(path)
    lines = text.splitlines()
    if not any(ln.strip() for ln in lines):
        raise ValueError("empty")
    delimiter, _ = _detect_delimiter([ln for ln in lines if ln.strip()][:40])

    header_idx = None
    header: list[str] = []
    best_count = 0
    for i, ln in enumerate(lines[:60]):
        if not ln.strip():
            continue
        cells = _split_row(ln, delimiter)
        mapped = {c for cell in cells for c, s in _name_scores(cell).items()
                  if s >= MIN_NAME_SCORE}
        # A header maps several fields; a data row is mostly numeric.
        numeric_like = sum(1 for c in cells if _parse_number(c) is not None)
        count = len(mapped)
        if count >= 2 and numeric_like <= len(cells) / 2 and count > best_count:
            best_count, header_idx, header = count, i, cells
    if header_idx is None:
        # Fall back to line 1 (legacy strict behaviour would also have failed).
        header_idx = next(i for i, ln in enumerate(lines) if ln.strip())
        header = _split_row(lines[header_idx], delimiter)

    rows: list[list[str]] = []
    for ln in lines[header_idx + 1:]:
        if not ln.strip():
            continue  # blank lines are not data
        cells = _split_row(ln, delimiter)
        if len(cells) == 1 and not cells[0].strip():
            continue
        rows.append(cells)

    # European decimal comma: with a non-comma delimiter, values like
    # '24,5301' are decimals, not lists. Detect on sampled data cells and
    # normalize in place (recorded — never a silent rewrite).
    decimal_comma = False
    if delimiter != ",":
        sample_cells = [c for row in rows[:: max(1, len(rows) // 100)] for c in row]
        dc = sum(1 for c in sample_cells if re.fullmatch(r"-?\d+,\d+", c.strip()))
        if sample_cells and dc / len(sample_cells) >= 0.3:
            decimal_comma = True
            rows = [
                [re.sub(r"(\d),(\d)", r"\1.\2", c) for c in row]
                for row in rows
            ]
    return _Table(header, rows, header_idx, delimiter, encoding, decimal_comma)


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

@dataclass
class FieldMapping:
    source_column: str
    canonical: str
    detected_unit: str
    canonical_unit: str
    conversion_applied: bool = False
    conversion_note: str | None = None
    reference: str | None = None           # altitude: MSL / AGL
    confidence: float = 0.0
    validation_status: str = "VALIDATED"   # VALIDATED / AMBIGUOUS / UNVALIDATED
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_column": self.source_column,
            "canonical_field": self.canonical,
            "detected_unit": self.detected_unit,
            "canonical_unit": self.canonical_unit,
            "conversion_applied": self.conversion_applied,
            "conversion_note": self.conversion_note,
            "reference": self.reference,
            "confidence": self.confidence,
            "validation_status": self.validation_status,
            "notes": self.notes,
        }


def _altitude_reference(header: str) -> str:
    h = header.lower()
    if any(k in h for k in ("relative", "rel_", "rel ", "agl", "above_ground")):
        return "AGL"
    if "msl" in h or "gps" in h or "absolute" in h:
        return "MSL"
    return "MSL (assumed — no reference in column name)"


def _map_columns(
    header: list[str], column_samples: dict[int, list[str]]
) -> tuple[dict[str, int], dict[str, FieldMapping], list[dict]]:
    """Assign canonical fields to column indices.

    For each canonical field: rank columns by name score, then validate the
    top candidate's sampled values. Competing candidates with the same
    semantics resolve by score (recorded); a rejected candidate demotes to
    the next one.
    """
    # Precompute per-column candidate scores once.
    per_column: list[dict[str, float]] = []
    for cell in header:
        per_column.append(_name_scores(cell))

    claims: dict[str, int] = {}
    mappings: dict[str, FieldMapping] = {}
    ambiguity: list[dict] = []

    # Order matters: specific fields before generic ones that share aliases
    # (heading vs yaw, altitude vs altitude_agl, speed vs velocity_*).
    order = [
        "latitude", "longitude", "altitude_agl", "altitude", "timestamp",
        "heading", "yaw", "roll", "pitch", "velocity_x", "velocity_y",
        "velocity_z", "speed", "gps_accuracy", "horizontal_accuracy",
        "vertical_accuracy", "satellites", "hdop", "vdop",
        "frame_number", "sequence",
    ]
    used: set[int] = set()
    for canon in order:
        ranked = sorted(
            ((s.get(canon, 0), i) for i, s in enumerate(per_column)
             if s.get(canon, 0) >= MIN_NAME_SCORE and i not in used),
            key=lambda t: -t[0],
        )
        if not ranked:
            continue
        top_score, top_idx = ranked[0]
        header_text = header[top_idx]
        values = [v for v in column_samples.get(top_idx, []) if v.strip()]

        # --- value validation per field family ----------------------------
        # Names propose, values disambiguate: when the top-scoring candidate's
        # values fail validation, promote the next candidate that passes —
        # regardless of its name score (a real 'latitude' column must beat a
        # same-score pose 'y' column whose values are out of geographic range).
        val_ok, unit, conv_note, conf_val = _validate_values(canon, values)
        if not val_ok:
            promoted = False
            for s2, i2 in ranked[1:]:
                v2 = [v for v in column_samples.get(i2, []) if v.strip()]
                ok2, unit2, note2, conf2 = _validate_values(canon, v2)
                if ok2:
                    ambiguity.append({
                        "field": canon, "rejected": header[top_idx].strip(),
                        "chosen": header[i2].strip(),
                        "reason": "first candidate failed value validation",
                    })
                    top_score, top_idx, header_text, values = s2, i2, header[i2], v2
                    unit, conv_note, conf_val = unit2, note2, conf2
                    val_ok = True
                    promoted = True
                    break
            if not promoted and canon in ("latitude", "longitude"):
                # Keep the name-mapped column but mark it unvalidated: rows
                # with out-of-range values are flagged individually downstream
                # (never fabricate, never silently drop the field).
                ambiguity.append({
                    "field": canon, "kept_unvalidated": header_text.strip(),
                    "reason": "sampled values failed validation; no better candidate",
                })
        conf = round(min(0.99, top_score * 0.7 + conf_val * 0.3), 2)
        canonical_unit = {
            "timestamp": "seconds", "latitude": "degrees", "longitude": "degrees",
            "altitude": "meters", "altitude_agl": "meters",
        }.get(canon, "native")
        if canon in ("altitude", "altitude_agl"):
            numeric_vals = [v for v in (_parse_number(x) for x in values) if v is not None]
            unit, ft_note = _altitude_unit_from_header(header_text, numeric_vals)
            if ft_note:
                conv_note = ft_note
                conf_val = max(conf_val, 0.85)
        mapping = FieldMapping(
            source_column=header_text.strip(),
            canonical=canon,
            detected_unit=unit,
            canonical_unit=canonical_unit,
            conversion_applied=bool(conv_note),
            conversion_note=conv_note,
            confidence=conf,
            validation_status="VALIDATED" if val_ok else "UNVALIDATED",
            notes=[],
        )
        if canon == "altitude":
            mapping.reference = _altitude_reference(header_text)
            if "assumed" in mapping.reference:
                mapping.notes.append("altitude reference not stated in source")
        if canon == "altitude_agl":
            mapping.reference = "AGL"
        used.add(top_idx)
        claims[canon] = top_idx
        mappings[canon] = mapping

        # Record competing candidates as ranked alternatives (§10) without
        # blocking: any second candidate above the mapping floor competes
        # (e.g. 'GPS Altitude' vs 'Relative Alt') — the choice is recorded.
        ranked_alts = [
            {"column": header[i2].strip(), "score": s2}
            for s2, i2 in ranked[1:] if s2 >= MIN_NAME_SCORE
        ]
        if ranked_alts:
            ambiguity.append({
                "field": canon,
                "ranked": [{"column": header[top_idx].strip(), "score": top_score}]
                          + ranked_alts[:2],
                "chosen": header[top_idx].strip(),
            })
    return claims, mappings, ambiguity


def _validate_values(canon: str, values: list[str]) -> tuple[bool, str, str | None, float]:
    """Validate sampled values for a canonical field.

    Returns (passed, detected_unit, conversion_note, value_confidence).
    """
    if not values:
        return False, "unknown", None, 0.0
    numeric = [_parse_number(v) for v in values]
    nums = [v for v in numeric if v is not None]
    numeric_frac = len(nums) / len(values)

    if canon == "timestamp":
        iso = [_parse_iso(v) for v in values]
        iso_hits = sum(1 for v in iso if v is not None)
        if iso_hits / len(values) >= 0.8:
            return True, "iso8601", "iso8601 → unix seconds (UTC)", 0.95
        if numeric_frac < 0.6:
            return False, "unknown", None, 0.0
        info = _classify_timestamps(nums, "")
        ok = info["confidence"] >= 0.4
        return ok, info["kind"], None, info["confidence"]

    if canon in ("latitude", "longitude"):
        dms = [_parse_dms(v) for v in values]
        dms_hits = sum(1 for v in dms if v is not None)
        if dms_hits / len(values) >= 0.6:
            return True, "DMS", "DMS → decimal degrees", 0.9
        rng = LAT_RANGE if canon == "latitude" else LON_RANGE
        inr = [v for v in nums if _in_range(v, rng)]
        if not nums:
            return False, "unknown", None, 0.0
        frac = len(inr) / len(nums)
        if frac >= VALUE_PASS_FRACTION:
            return True, "degrees", None, 0.6 + 0.35 * frac
        # Longitude 0..360 convention?
        if canon == "longitude":
            shifted = [v for v in nums if 0.0 <= v <= 360.0 and not _in_range(v, (-180, 180))]
            if len(shifted) / len(nums) >= VALUE_PASS_FRACTION:
                return True, "degrees_0_360", "longitude 0–360 → ±180 (−360)", 0.85
        return False, "degrees", None, frac

    if canon in ("altitude", "altitude_agl"):
        if numeric_frac < 0.6:
            return False, "unknown", None, 0.0
        plausible = [v for v in nums if _in_range(v, ALT_RANGE)]
        frac = len(plausible) / len(nums)
        h = ""  # unit hint comes from the caller-visible header; handled below
        return frac >= VALUE_PASS_FRACTION, "meters", None, 0.5 + 0.4 * frac

    # Generic numeric optional fields.
    if numeric_frac >= VALUE_PASS_FRACTION:
        return True, "native", None, 0.6 + 0.3 * numeric_frac
    return False, "unknown", None, numeric_frac


def _altitude_unit_from_header(header: str, values: list[float]) -> tuple[str, str | None]:
    """Explicit ft/feet in the header converts; otherwise meters, flagged
    ambiguous when magnitudes look like feet but the name is silent."""
    h = header.lower()
    if any(k in h for k in ("ft", "feet")) and "ft_" not in h + "_":
        pass
    if re.search(r"(?:^|_)(ft|feet)(?:_|$)", h):
        return "feet", "feet → meters (× 0.3048)"
    return "meters", None


# ---------------------------------------------------------------------------
# Full-file parsing
# ---------------------------------------------------------------------------

@dataclass
class TelemetryDataset:
    """Detection outcome + normalized rows for one telemetry CSV."""
    source_file: str
    detection: dict[str, Any]
    rows: list[dict[str, Any]]
    quality: dict[str, Any]

    @property
    def has_gps(self) -> bool:
        return bool(self.detection["capabilities"]["gps_position"]["available"])

    def failure_reason(self) -> str:
        caps = self.detection["capabilities"]
        det = self.detection.get("fields", {})
        mapped = ", ".join(
            f"{k}←'{v['source_column']}'" for k, v in det.items()) or "none"
        if not caps["gps_position"]["available"]:
            unmapped = self.detection.get("unmapped_columns", [])
            hint = f"; unmapped headers: {unmapped[:8]}" if unmapped else ""
            return (
                "no latitude/longitude column with valid coordinate values "
                f"could be detected (mapped: {mapped}){hint}"
            )
        return "telemetry lacks required capability"


def _sample_rows(rows: list[list[str]], per_column_values: dict[int, list[str]], n: int = 200) -> None:
    """Fill ``per_column_values`` with up to n sampled cells per column."""
    step = max(1, len(rows) // n)
    for row in rows[::step][:n]:
        for i, cell in enumerate(row):
            per_column_values.setdefault(i, []).append(cell)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    p1, p2 = radians(lat1), radians(lat2)
    dp = p2 - p1
    dl = radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * 6371008.8 * asin(min(1.0, sqrt(a)))


def parse_telemetry_csv(path: Path) -> TelemetryDataset:
    """Inspect, detect, validate and normalize one telemetry CSV.

    Never raises for schema-shape reasons: whatever cannot be mapped is
    reported through ``detection``/``quality`` (callers decide whether the
    requested operation can proceed). Raises only for unreadable files.
    """
    table = _load_table(path)
    per_column: dict[int, list[str]] = {}
    _sample_rows(table.rows, per_column)
    claims, mappings, ambiguity = _map_columns(table.header, per_column)

    detection: dict[str, Any] = {
        "source_file": path.name,
        "detected_delimiter": table.delimiter,
        "delimiter": table.delimiter,
        "decimal_comma": table.decimal_comma,
        "header_row": table.header_row_index + 1,  # 1-based, as reported to users
        "header_row_index": table.header_row_index,
        "encoding": table.encoding,
        "columns": table.header,
        "fields": {k: m.to_dict() for k, m in mappings.items()},
        "ambiguous": ambiguity,
        "unmapped_columns": [
            h.strip() for i, h in enumerate(table.header)
            if i not in claims.values() and h.strip()
        ],
    }

    ts_idx = claims.get("timestamp")
    ts_map = mappings.get("timestamp")
    ts_kind = ts_map.detected_unit if ts_map else "absent"
    ts_factor = None
    if ts_map and ts_map.detected_unit not in ("iso8601",):
        kind_info = _classify_timestamps(
            [v for v in (_parse_number(table.rows[r][ts_idx]) for r in range(len(table.rows))) if v is not None][:500],
            table.header[ts_idx] if ts_idx is not None else "",
        ) if ts_idx is not None else {}
        ts_factor = kind_info.get("unit_seconds_factor")
        detection["timestamp_interpretation"] = {
            "kind": ts_kind,
            "unit_seconds_factor": ts_factor,
            "epoch_based": kind_info.get("epoch_based"),
            "confidence": kind_info.get("confidence"),
            "basis": kind_info.get("basis"),
        }
    elif ts_map:
        detection["timestamp_interpretation"] = {
            "kind": "iso8601", "unit_seconds_factor": 1.0, "epoch_based": True,
            "confidence": ts_map.confidence, "basis": ["ISO-8601 strings"],
        }

    # ---- per-row normalization -------------------------------------------
    def cell(row: list[str], idx: int | None) -> str:
        if idx is None or idx >= len(row):
            return ""
        return row[idx]

    def num_at(row: list[str], canon: str) -> float | None:
        idx = claims.get(canon)
        if idx is None:
            return None
        raw = cell(row, idx)
        m = mappings.get(canon)
        if m and m.detected_unit == "DMS":
            return _parse_dms(raw)
        if m and m.detected_unit == "iso8601":
            return _parse_iso(raw)
        v = _parse_number(raw)
        if v is None:
            return None
        if m and m.conversion_note and "× 0.3048" in m.conversion_note:
            v = v * 0.3048
        if m and m.detected_unit == "radians":
            v = math.degrees(v)
        if canon == "longitude" and m and m.detected_unit == "degrees_0_360" and v > 180.0:
            v = v - 360.0
        return v

    ts_is_relative = ts_kind.startswith("relative")
    rows_out: list[dict[str, Any]] = []
    for n, row in enumerate(table.rows):
        entry: dict[str, Any] = {"source_row": table.header_row_index + 2 + n}
        for canon in ("latitude", "longitude", "altitude", "altitude_agl",
                      "roll", "pitch", "yaw", "heading", "speed",
                      "gps_accuracy", "horizontal_accuracy", "vertical_accuracy",
                      "satellites", "hdop", "vdop", "frame_number", "sequence"):
            entry[canon] = num_at(row, canon)
        ts_raw = cell(row, ts_idx) if ts_idx is not None else ""
        ts_val: float | None = None
        if ts_map:
            ts_val = _parse_iso(ts_raw) if ts_map.detected_unit == "iso8601" else _parse_number(ts_raw)
            if ts_val is not None and ts_factor:
                ts_val = ts_val * ts_factor
        entry["timestamp"] = ts_val
        # GPS validity (§15): VALID / INVALID / MISSING / SUSPECT.
        lat, lon = entry["latitude"], entry["longitude"]
        if lat is None or lon is None:
            entry["gps_status"] = "MISSING"
            entry["gps_valid"] = False
        elif (
            (lat == 0.0 and lon == 0.0)
            or not _in_range(lat, LAT_RANGE)
            or not _in_range(lon, LON_RANGE)
        ):
            entry["gps_status"] = "INVALID"
            entry["gps_valid"] = False
        else:
            entry["gps_status"] = "VALID"
            entry["gps_valid"] = True
        rows_out.append(entry)

    # SUSPECT flags: physically impossible jumps between consecutive fixes.
    prev: dict[str, Any] | None = None
    for entry in rows_out:
        if not entry["gps_valid"]:
            continue
        if prev is not None and entry["timestamp"] is not None and prev["timestamp"] is not None:
            dt = entry["timestamp"] - prev["timestamp"]
            if dt > 0:
                dist = _haversine_m(prev["latitude"], prev["longitude"],
                                    entry["latitude"], entry["longitude"])
                speed = dist / dt
                if dist > 200.0 and speed > 200.0:  # >200 m AND implied >200 m/s
                    entry["gps_status"] = "SUSPECT"
        prev = entry

    # ---- capabilities (§2) -------------------------------------------------
    valid_rows = [r for r in rows_out if r["gps_valid"]]
    alt_abs = [r["altitude"] for r in rows_out if r["altitude"] is not None]
    alt_agl = [r["altitude_agl"] for r in rows_out if r["altitude_agl"] is not None]
    ts_present = [r["timestamp"] for r in rows_out if r["timestamp"] is not None]
    # Capability = the SCHEMA provides the field (mapping exists). Whether the
    # values are any good is measured separately (valid_rows) — a file whose
    # GPS column exists but holds sentinels is a quality problem, not a
    # missing-capability problem, and downstream reports say which.
    detection["capabilities"] = {
        "gps_position": {
            "available": "latitude" in claims and "longitude" in claims,
            "valid_rows": len(valid_rows),
            "schema_mapped": {"latitude": claims.get("latitude") is not None,
                              "longitude": claims.get("longitude") is not None},
        },
        "altitude": {
            "available": bool(alt_abs or alt_agl),
            "absolute_rows": len(alt_abs),
            "agl_rows": len(alt_agl),
            "reference": (mappings.get("altitude").reference
                          if mappings.get("altitude") else
                          (mappings.get("altitude_agl").reference
                           if mappings.get("altitude_agl") else None)),
        },
        "temporal_sync": {
            "available": bool(ts_present),
            "timestamp_kind": ts_kind,
            "relative": ts_is_relative,
        },
    }

    quality = _quality(rows_out, detection, ts_is_relative, mappings)
    dataset = TelemetryDataset(
        source_file=path.name, detection=detection, rows=rows_out, quality=quality)
    return dataset


def _quality(
    rows: list[dict[str, Any]],
    detection: dict[str, Any],
    ts_is_relative: bool,
    mappings: dict[str, FieldMapping],
) -> dict[str, Any]:
    valid = [r for r in rows if r["gps_valid"]]
    invalid = [r for r in rows if r["gps_status"] == "INVALID"]
    missing = [r for r in rows if r["gps_status"] == "MISSING"]
    suspect = [r for r in rows if r["gps_status"] == "SUSPECT"]
    ts_vals = [r["timestamp"] for r in rows if r["timestamp"] is not None]
    monotonic = None
    interval = None
    duration = None
    if len(ts_vals) >= 2:
        diffs = [b - a for a, b in zip(ts_vals, ts_vals[1:])]
        monotonic = all(d > 0 for d in diffs)
        pos = [d for d in diffs if d > 0]
        interval = round(_median(pos), 6) if pos else None
        duration = round(ts_vals[-1] - ts_vals[0], 3)
    traj = 0.0
    max_jump = 0.0
    for a, b in zip(valid, valid[1:]):
        d = _haversine_m(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
        traj += d
        max_jump = max(max_jump, d)
    lats = [r["latitude"] for r in valid]
    lons = [r["longitude"] for r in valid]
    confs = [m.confidence for m in mappings.values()]
    return {
        "total_rows": len(rows),
        "valid_gps_rows": len(valid),
        "invalid_gps_rows": len(invalid),
        "missing_gps_rows": len(missing),
        "suspect_gps_rows": len(suspect),
        "timestamp": {
            "availability": "available" if ts_vals else "unavailable",
            "type": detection.get("timestamp_interpretation", {}).get("kind", "absent"),
            "relative": ts_is_relative,
            "monotonic": monotonic,
            "median_interval_sec": interval,
            "duration_sec": duration,
        },
        "altitude": {
            "availability": "available" if (mappings.get("altitude") or mappings.get("altitude_agl")) else "unavailable",
            "reference": detection["capabilities"]["altitude"]["reference"],
        },
        "units": {k: {"detected": m.detected_unit, "canonical": m.canonical_unit,
                      "converted": m.conversion_applied}
                  for k, m in mappings.items()},
        "detected_columns": {k: m.source_column for k, m in mappings.items()},
        "mapping_confidence": {
            "min": round(min(confs), 2) if confs else None,
            "mean": round(sum(confs) / len(confs), 2) if confs else None,
        },
        "coordinate_bounds": {
            "lat_min": min(lats) if lats else None,
            "lat_max": max(lats) if lats else None,
            "lon_min": min(lons) if lons else None,
            "lon_max": max(lons) if lons else None,
        },
        "trajectory_distance_m": round(traj, 1) if valid else 0.0,
        "max_instantaneous_jump_m": round(max_jump, 1) if valid else 0.0,
        "telemetry_duration_sec": duration,
    }


# ---------------------------------------------------------------------------
# Artifacts (§12/§16)
# ---------------------------------------------------------------------------

NORMALIZED_COLUMNS = (
    "timestamp", "latitude", "longitude", "altitude", "altitude_agl",
    "roll", "pitch", "yaw", "heading", "speed", "gps_accuracy",
    "horizontal_accuracy", "vertical_accuracy", "satellites", "hdop",
    "vdop", "frame_number", "sequence", "gps_status", "source_row",
)


def write_telemetry_artifacts(dataset: TelemetryDataset, out_dir: Path) -> dict[str, str]:
    """Write ``telemetry_normalized.csv``, ``telemetry_schema.json`` and
    ``telemetry_quality.json``. Best-effort: failures are logged, never
    raised (artifacts must not break ingestion)."""
    written: dict[str, str] = {}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        norm_path = out_dir / "telemetry_normalized.csv"
        with open(norm_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(NORMALIZED_COLUMNS)
            for r in dataset.rows:
                w.writerow([
                    "" if r.get(c) is None else
                    (f"{r[c]:.9f}" if isinstance(r[c], float) and c in
                     ("timestamp", "latitude", "longitude") else r[c])
                    for c in NORMALIZED_COLUMNS
                ])
        written["normalized_csv"] = str(norm_path)
        schema_path = out_dir / "telemetry_schema.json"
        schema_path.write_text(json.dumps(dataset.detection, indent=2, default=str))
        written["schema_json"] = str(schema_path)
        quality_path = out_dir / "telemetry_quality.json"
        quality_path.write_text(json.dumps(dataset.quality, indent=2, default=str))
        written["quality_json"] = str(quality_path)
    except OSError as exc:
        log.warning("telemetry_artifacts_write_failed", error=str(exc))
    return written


def dataset_to_samples(dataset: TelemetryDataset) -> list:
    """Project normalized rows onto the canonical TelemetrySample sequence.

    All rows are returned (valid and invalid GPS alike — downstream filtering
    drops+counts them). Rows carry ``timestamp=None`` when the file has no
    usable timestamps; positions are never fabricated. The timeline handed
    to synchronization is file-relative: absolute epochs are shifted by the
    first sample's time (the video↔telemetry clock offset is a separate,
    reported concern)."""
    from app.services.telemetry import TelemetrySample

    # Timestamps stay ABSOLUTE (epoch-based) or AS-RECORDED (relative): the
    # video↔telemetry clock relationship is synchronize()'s concern — it
    # reports honest no-overlap instead of silently re-basing timelines.
    samples: list[TelemetrySample] = []
    for r in dataset.rows:
        ts = r["timestamp"]
        samples.append(TelemetrySample(
            timestamp_sec=ts,
            latitude=r["latitude"],
            longitude=r["longitude"],
            altitude_m=r["altitude"],
            roll=r.get("roll"), pitch=r.get("pitch"), yaw=r.get("yaw"),
            heading=r.get("heading"),
            gps_accuracy_m=r.get("gps_accuracy"),
            satellites=r.get("satellites"),
            hdop=r.get("hdop"), vdop=r.get("vdop"),
            frame_number=r.get("frame_number"), sequence=r.get("sequence"),
        ))
    if any(s.timestamp_sec is not None for s in samples):
        samples.sort(key=lambda s: (s.timestamp_sec is None, s.timestamp_sec))
    return samples
