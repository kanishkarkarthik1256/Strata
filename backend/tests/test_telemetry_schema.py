"""Universal telemetry ingestion regression tests (spec §21).

Covers the full matrix: exact canonical CSV, renamed/uppercased/whitespace
columns, units in headers, ms/µs/unix/relative/ISO timestamps, missing
timestamp, missing altitude, extra/reordered columns, semicolon delimiter,
BOM, metadata before the header, invalid GPS rows, ambiguous altitude,
duplicate columns, DMS coordinates and 0–360 longitudes. The parser must
never silently invent telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.telemetry import (
    TelemetryError,
    format_telemetry_detection,
    load_external_telemetry,
    load_telemetry_with_schema,
    synchronize,
)
from app.services.telemetry_schema import (
    _name_scores,
    parse_telemetry_csv,
    write_telemetry_artifacts,
)

# A small valid trajectory near Shitan, TW.
_ROWS = [
    (0.0, 24.5300, 120.9300, 350.0),
    (1.0, 24.5301, 120.9301, 351.0),
    (2.0, 24.5302, 120.9302, 352.0),
    (3.0, 24.5303, 120.9303, 353.0),
    (4.0, 24.5304, 120.9304, 354.0),
]


def _csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _canonical(rows=_ROWS) -> str:
    return "timestamp,latitude,longitude,altitude\n" + "".join(
        f"{t},{la},{lo},{al}\n" for t, la, lo, al in rows)


# ---------------------------------------------------------------------------
# §19 test matrix — formats A–K
# ---------------------------------------------------------------------------


class TestSpecMatrix:
    def test_A_canonical(self, tmp_path):
        samples = load_external_telemetry(_csv(tmp_path / "a.csv", _canonical()))
        assert len(samples) == 5
        assert samples[3].timestamp_sec == pytest.approx(3.0)
        assert samples[3].latitude == pytest.approx(24.5303)
        assert samples[3].altitude_m == pytest.approx(353.0)

    def test_B_renamed_short(self, tmp_path):
        text = "time,lat,lon,alt\n" + "".join(
            f"{t},{la},{lo},{al}\n" for t, la, lo, al in _ROWS)
        samples = load_external_telemetry(_csv(tmp_path / "b.csv", text))
        assert samples[2].timestamp_sec == pytest.approx(2.0)
        assert samples[2].latitude == pytest.approx(24.5302)
        assert samples[2].altitude_m == pytest.approx(352.0)

    def test_C_ms_columns_and_epoch_ms(self, tmp_path):
        # Column names with GPS/units + unix-millisecond values (~1.7e12).
        text = "timestamp_ms,GPS_Latitude,GPS_Longitude,GPS_Altitude\n" + "".join(
            f"{1700000000000 + int(t * 1000)},{la},{lo},{al}\n"
            for t, la, lo, al in _ROWS)
        samples = load_external_telemetry(_csv(tmp_path / "c.csv", text))
        assert samples[1].timestamp_sec == pytest.approx(1700000000001 * 1e-3)
        assert samples[1].latitude == pytest.approx(24.5301)

    def test_D_mixed_case_plus_orientation(self, tmp_path):
        text = (
            "Time,Lat,Lon,Alt,Roll,Pitch,Yaw\n"
            "0.0,24.53,120.93,350.0,1.0,2.0,90.0\n"
            "1.0,24.5301,120.9301,351.0,1.1,2.1,91.0\n"
            "2.0,24.5302,120.9302,352.0,1.2,2.2,92.0\n"
        )
        samples = load_external_telemetry(_csv(tmp_path / "d.csv", text))
        assert samples[1].timestamp_sec == pytest.approx(1.0)
        assert samples[1].roll == pytest.approx(1.1)
        assert samples[1].pitch == pytest.approx(2.1)
        assert samples[1].yaw == pytest.approx(91.0)

    def test_E_many_unrelated_columns(self, tmp_path):
        text = (
            "timestamp,lat,lon,alt,roll,pitch,yaw,velocity_x,velocity_y,"
            "velocity_z,battery_pct,flight_mode,satellites,hdop\n"
            "0.0,24.53,120.93,350.0,1,2,90,0.5,0.1,-0.2,98,stabilize,12,0.9\n"
            "1.0,24.5301,120.9301,351.0,1,2,90,0.5,0.1,-0.2,97,stabilize,12,0.9\n"
            "2.0,24.5302,120.9302,352.0,1,2,90,0.5,0.1,-0.2,96,stabilize,12,0.9\n"
        )
        samples = load_external_telemetry(_csv(tmp_path / "e.csv", text))
        assert samples[0].satellites == pytest.approx(12.0)
        assert samples[0].hdop == pytest.approx(0.9)
        assert samples[0].latitude == pytest.approx(24.53)

    def test_F_metadata_before_header(self, tmp_path):
        text = (
            "Flight exported by SuperPilot v3\n"
            "Generated: 2026-09-20\n"
            "Aircraft: ABC-123\n"
            "--------------------------------\n"
            "Time, GPS Lat, GPS Lon, GPS Alt\n"
            "0.0,24.53,120.93,350.0\n"
            "1.0,24.5301,120.9301,351.0\n"
        )
        samples = load_external_telemetry(_csv(tmp_path / "f.csv", text))
        assert len(samples) == 2
        assert samples[0].latitude == pytest.approx(24.53)
        dataset = parse_telemetry_csv(tmp_path / "f.csv")
        assert dataset.detection["header_row"] == 5

    def test_G_semicolon_delimiter(self, tmp_path):
        text = "timestamp;latitude;longitude;altitude\n" + "".join(
            f"{t};{la};{lo};{al}\n".replace(".", ",")  # decimal commas too
            for t, la, lo, al in _ROWS[:3])
        samples = load_external_telemetry(_csv(tmp_path / "g.csv", text))
        assert len(samples) == 3
        assert samples[1].latitude == pytest.approx(24.5301)

    def test_H_no_timestamp_but_gps(self, tmp_path):
        text = "latitude,longitude,altitude\n" + "".join(
            f"{la},{lo},{al}\n" for _, la, lo, al in _ROWS)
        samples = load_external_telemetry(_csv(tmp_path / "h.csv", text))
        assert len(samples) == 5
        assert all(s.timestamp_sec is None for s in samples)
        assert samples[4].latitude == pytest.approx(24.5304)
        # Capabilities: position YES, temporal sync NO.
        _, report = load_telemetry_with_schema(tmp_path / "h.csv")
        caps = report["schema"]["capabilities"]
        assert caps["gps_position"]["available"] is True
        assert caps["temporal_sync"]["available"] is False
        # synchronize() stays honest: nothing matched, GPS still available.
        report_sync = synchronize(
            [(f"frame_{i:06d}", float(i)) for i in range(3)], samples)
        assert report_sync.matched_frames == 0
        assert report_sync.gps_available is True
        assert any("temporal synchronization unavailable" in n
                   for n in report_sync.notes)

    def test_I_relative_timestamps(self, tmp_path):
        # 0/33/66/99 ms-style relative clock (native milliseconds).
        text = "time_ms,latitude,longitude\n" + "".join(
            f"{int(t * 1000)},{la},{lo}\n" for t, la, lo, _ in _ROWS)
        samples = load_external_telemetry(_csv(tmp_path / "i.csv", text))
        assert samples[2].timestamp_sec == pytest.approx(2.0)
        _, report = load_telemetry_with_schema(tmp_path / "i.csv")
        assert report["schema"]["timestamp_interpretation"]["kind"] == "relative_milliseconds"
        assert report["schema"]["timestamp_interpretation"]["epoch_based"] is False

    def test_J_unix_milliseconds(self, tmp_path):
        text = "timestamp,latitude,longitude\n" + "".join(
            f"{1700000000000 + int(t * 1000)},{la},{lo}\n"
            for t, la, lo, _ in _ROWS)
        samples = load_external_telemetry(_csv(tmp_path / "j.csv", text))
        assert samples[0].timestamp_sec == pytest.approx(1700000000000 * 1e-3)

    def test_K_unix_seconds(self, tmp_path):
        text = "timestamp,latitude,longitude\n" + "".join(
            f"{1700000000 + t},{la},{lo}\n" for t, la, lo, _ in _ROWS)
        samples = load_external_telemetry(_csv(tmp_path / "k.csv", text))
        assert samples[0].timestamp_sec == pytest.approx(1700000000.0)

    def test_iso8601_timestamps(self, tmp_path):
        text = (
            "timestamp,latitude,longitude\n"
            "2026-09-20T10:00:00Z,24.53,120.93\n"
            "2026-09-20T10:00:01Z,24.5301,120.9301\n"
            "2026-09-20T10:00:02Z,24.5302,120.9302\n"
        )
        samples = load_external_telemetry(_csv(tmp_path / "iso.csv", text))
        assert samples[1].timestamp_sec - samples[0].timestamp_sec == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Header normalization + fuzzy matching
# ---------------------------------------------------------------------------


class TestHeaderNormalization:
    def test_whitespace_units_and_parentheses(self, tmp_path):
        text = (
            '  Timestamp (ms) ,  Latitude (deg), Longitude (deg), Altitude (m)\n'
            '0,24.53,120.93,350\n1,24.5301,120.9301,351\n'
        )
        samples = load_external_telemetry(_csv(tmp_path / "w.csv", text))
        assert samples[1].latitude == pytest.approx(24.5301)

    def test_bom_survives(self, tmp_path):
        p = tmp_path / "bom.csv"
        p.write_bytes(("﻿" + _canonical()).encode("utf-8"))
        samples = load_external_telemetry(p)
        assert len(samples) == 5
        d = parse_telemetry_csv(p)
        assert d.detection["encoding"] == "utf-8-sig"

    def test_degree_symbols(self, tmp_path):
        s = _name_scores("Latitude°")
        assert s["latitude"] >= 0.85

    def test_reordered_columns(self, tmp_path):
        text = "alt,lon,lat,timestamp\n" + "".join(
            f"{al},{lo},{la},{t}\n" for t, la, lo, al in _ROWS[:2])
        samples = load_external_telemetry(_csv(tmp_path / "r.csv", text))
        assert samples[0].latitude == pytest.approx(24.53)
        assert samples[0].longitude == pytest.approx(120.93)
        assert samples[0].timestamp_sec == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Units + coordinate formats (§6/§7)
# ---------------------------------------------------------------------------


class TestUnitsAndFormats:
    def test_altitude_feet_converted_and_recorded(self, tmp_path):
        text = "timestamp,latitude,longitude,altitude_ft\n0,24.53,120.93,1148.29\n"
        samples = load_external_telemetry(_csv(tmp_path / "ft.csv", text))
        assert samples[0].altitude_m == pytest.approx(350.0, rel=1e-3)
        d = parse_telemetry_csv(tmp_path / "ft.csv")
        m = d.detection["fields"]["altitude"]
        assert m["detected_unit"] == "feet"
        assert m["conversion_applied"] is True
        assert "0.3048" in m["conversion_note"]

    def test_dms_coordinates_normalized(self, tmp_path):
        text = (
            "timestamp,latitude,longitude\n"
            "0,24°31'48.0\"N,120°55'48.0\"E\n"
            "1,24°31'48.4\"N,120°55'48.4\"E\n"
        )
        samples = load_external_telemetry(_csv(tmp_path / "dms.csv", text))
        assert samples[0].latitude == pytest.approx(24.53, abs=1e-6)
        assert samples[0].longitude == pytest.approx(120.93, abs=1e-6)

    def test_longitude_0_360_shifted(self, tmp_path):
        text = "timestamp,latitude,longitude\n0,24.53,240.93\n1,24.5301,240.9301\n"
        samples = load_external_telemetry(_csv(tmp_path / "l360.csv", text))
        assert samples[0].longitude == pytest.approx(-119.07, abs=1e-9)

    def test_relative_alt_not_treated_as_absolute(self, tmp_path):
        text = "timestamp,latitude,longitude,relative_alt\n0,24.53,120.93,50\n"
        samples = load_external_telemetry(_csv(tmp_path / "ra.csv", text))
        assert samples[0].altitude_m is None  # relative altitude ≠ absolute
        d = parse_telemetry_csv(tmp_path / "ra.csv")
        caps = d.detection["capabilities"]["altitude"]
        assert caps["available"] is True and caps["absolute_rows"] == 0
        assert caps["reference"] == "AGL"

    def test_agl_and_absolute_both_present(self, tmp_path):
        text = (
            "timestamp,latitude,longitude,altitude,relative_alt\n"
            "0,24.53,120.93,490.0,50.0\n"
        )
        samples = load_external_telemetry(_csv(tmp_path / "both.csv", text))
        assert samples[0].altitude_m == pytest.approx(490.0)  # GPS altitude wins


# ---------------------------------------------------------------------------
# Row-level validation (§15) — flag, never silently drop or invent
# ---------------------------------------------------------------------------


class TestRowValidation:
    def test_invalid_rows_flagged_individually(self, tmp_path):
        text = (
            "timestamp,latitude,longitude,altitude\n"
            "0,24.53,120.93,350\n"
            "1,999.0,120.93,351\n"     # invalid latitude
            "2,24.5302,120.9302,352\n"
            "3,0,0,353\n"              # null island
            "4,,120.93,354\n"          # missing latitude
        )
        samples = load_external_telemetry(_csv(tmp_path / "v.csv", text))
        assert len(samples) == 5                      # rows preserved
        d = parse_telemetry_csv(tmp_path / "v.csv")
        statuses = [r["gps_status"] for r in d.rows]
        assert statuses == ["VALID", "INVALID", "VALID", "INVALID", "MISSING"]
        q = d.quality
        assert q["valid_gps_rows"] == 2 and q["invalid_gps_rows"] == 2
        assert q["missing_gps_rows"] == 1

    def test_suspect_jump_flagged(self, tmp_path):
        # Two fixes >200 m apart with >200 m/s implied speed → SUSPECT.
        text = (
            "timestamp,latitude,longitude\n"
            "0,24.53,120.93\n"
            "1,24.55,120.95\n"
        )
        d = parse_telemetry_csv(_csv(tmp_path / "jump.csv", text))
        assert [r["gps_status"] for r in d.rows] == ["VALID", "SUSPECT"]

    def test_quality_report_contents(self, tmp_path):
        d = parse_telemetry_csv(_csv(tmp_path / "q.csv", _canonical()))
        q = d.quality
        for key in ("total_rows", "valid_gps_rows", "timestamp", "altitude",
                    "units", "detected_columns", "mapping_confidence",
                    "coordinate_bounds", "trajectory_distance_m",
                    "max_instantaneous_jump_m", "telemetry_duration_sec"):
            assert key in q, key
        assert q["timestamp"]["monotonic"] is True
        assert q["timestamp"]["median_interval_sec"] == pytest.approx(1.0)
        assert q["trajectory_distance_m"] > 0


# ---------------------------------------------------------------------------
# Ambiguity handling (§10) + adversarial schemas
# ---------------------------------------------------------------------------


class TestAmbiguity:
    def test_altitude_candidates_ranked_not_fabricated(self, tmp_path):
        text = (
            "timestamp,latitude,longitude,GPS Altitude,Relative Alt\n"
            "0,24.53,120.93,490.0,50.0\n"
            "1,24.5301,120.9301,491.0,51.0\n"
        )
        samples = load_external_telemetry(_csv(tmp_path / "amb.csv", text))
        # Absolute GPS altitude wins; relative altitude is AGL, not a rival.
        assert samples[0].altitude_m == pytest.approx(490.0)
        d = parse_telemetry_csv(tmp_path / "amb.csv")
        caps = d.detection["capabilities"]["altitude"]
        assert caps["reference"] == "MSL"
        assert caps["agl_rows"] == 2  # both carried, references kept apart

    def test_same_reference_altitude_competitors_ranked(self, tmp_path):
        """Two same-reference candidates (typo variants) → ranked, chosen by
        score, choice recorded — no silent pick (spec §10)."""
        text = (
            "timestamp,latitude,longitude,GPS Altitude,GPS_Alt\n"
            "0,24.53,120.93,490.0,490.0\n"
            "1,24.5301,120.9301,491.0,491.0\n"
        )
        samples = load_external_telemetry(_csv(tmp_path / "amb2.csv", text))
        assert samples[0].altitude_m == pytest.approx(490.0)
        d = parse_telemetry_csv(tmp_path / "amb2.csv")
        assert any(a["field"] == "altitude" and "ranked" in a
                   for a in d.detection["ambiguous"])

    def test_duplicate_columns_last_or_flagged(self, tmp_path):
        text = (
            "timestamp,latitude,latitude,longitude\n"
            "0,24.53,999.0,120.93\n"   # second 'latitude' holds garbage
            "1,24.5301,999.0,120.9301\n"
        )
        # Must not crash and must not map the garbage copy.
        samples = load_external_telemetry(_csv(tmp_path / "dup.csv", text))
        assert samples[0].latitude == pytest.approx(24.53)

    def test_two_time_columns_value_validation_decides(self, tmp_path):
        # 'Time' holds strings (unusable); 'time_ms' holds real values.
        text = (
            "Time,time_ms,latitude,longitude\n"
            "abc,0,24.53,120.93\n"
            "def,1000,24.5301,120.9301\n"
        )
        samples = load_external_telemetry(_csv(tmp_path / "t2.csv", text))
        assert samples[1].timestamp_sec == pytest.approx(1.0)
        d = parse_telemetry_csv(tmp_path / "t2.csv")
        assert d.detection["fields"]["timestamp"]["source_column"] == "time_ms"

    def test_pose_csv_xyz_never_mapped_as_gps(self, tmp_path):
        """Pose columns (x/y/z) must not fabricate geographic coordinates."""
        text = (
            "frame_id,x,y,z,qw,qx,qy,qz,latitude,longitude,altitude,fov_vertical\n"
            "0,-172.5,94.7,188.2,0.12,-0.46,0.85,-0.22,52.365,13.521,188.2,24.12\n"
        )
        samples = load_external_telemetry(_csv(tmp_path / "pose.csv", text))
        assert samples[0].latitude == pytest.approx(52.365)   # real GPS column
        assert samples[0].longitude == pytest.approx(13.521)
        d = parse_telemetry_csv(tmp_path / "pose.csv")
        assert "x" not in d.detection["fields"]       # x ≠ longitude
        assert "z" not in d.detection["fields"]       # z ≠ altitude (metric pose)
        # Pose/intrinsics columns stay UNMAPPED (x/y/z/qw..qz/fov_*), never
        # repurposed as velocities or accuracies.
        for pose_col in ("x", "y", "z", "qw", "qx", "qy", "qz", "fov_vertical"):
            assert pose_col in d.detection["unmapped_columns"], pose_col
        assert not (set(d.detection["fields"]) & {"velocity_x", "velocity_y",
                                                   "velocity_z", "vertical_accuracy"})
        # frame_id is a frame hint, not a clock: mapped as frame_number,
        # timestamp stays ABSENT (never fabricate temporal sync).
        assert d.detection["fields"]["frame_number"]["source_column"] == "frame_id"
        assert samples[0].timestamp_sec is None


# ---------------------------------------------------------------------------
# Artifacts (§12/§16) + detection UX (§17)
# ---------------------------------------------------------------------------


class TestArtifactsAndUX:
    def test_three_artifacts_written(self, tmp_path):
        p = _csv(tmp_path / "art.csv", _canonical())
        _, report = load_telemetry_with_schema(p)
        art = report["artifacts"]
        assert set(art) == {"normalized_csv", "schema_json", "quality_json"}
        norm = Path(art["normalized_csv"]).read_text().splitlines()
        assert norm[0] == (
            "timestamp,latitude,longitude,altitude,altitude_agl,roll,pitch,yaw,"
            "heading,speed,gps_accuracy,horizontal_accuracy,vertical_accuracy,"
            "satellites,hdop,vdop,frame_number,sequence,gps_status,source_row"
        )
        assert len(norm) == 6  # header + 5 rows
        schema = json.loads(Path(art["schema_json"]).read_text())
        assert schema["fields"]["latitude"]["source_column"] == "latitude"
        assert schema["detected_delimiter"] == ","
        quality = json.loads(Path(art["quality_json"]).read_text())
        assert quality["total_rows"] == 5

    def test_normalized_nulls_for_missing(self, tmp_path):
        text = "latitude,longitude\n24.53,120.93\n"
        p = _csv(tmp_path / "null.csv", text)
        _, report = load_telemetry_with_schema(p)
        rows = Path(report["artifacts"]["normalized_csv"]).read_text().splitlines()
        data = rows[1].split(",")
        assert data[0] == ""       # timestamp absent → empty, never 0
        assert data[4] == ""       # altitude_agl absent

    def test_schema_report_fields(self, tmp_path):
        _, report = load_telemetry_with_schema(_csv(tmp_path / "sr.csv", _canonical()))
        f = report["schema"]["fields"]["timestamp"]
        for key in ("source_column", "canonical_field", "detected_unit",
                    "canonical_unit", "confidence", "validation_status"):
            assert key in f, key

    def test_detection_summary_format(self, tmp_path):
        _, report = load_telemetry_with_schema(_csv(tmp_path / "u.csv", _canonical()))
        text = format_telemetry_detection(report)
        assert "TELEMETRY DETECTED" in text
        assert "latitude → 'latitude'" in text
        assert "Temporal synchronization: Available" in text
        assert "Schema confidence" in text

    def test_detection_summary_honest_when_no_timestamp(self, tmp_path):
        text = "latitude,longitude\n24.53,120.93\n"
        _, report = load_telemetry_with_schema(_csv(tmp_path / "nt.csv", text))
        text_out = format_telemetry_detection(report)
        assert "Temporal synchronization: UNAVAILABLE" in text_out
        assert "positions still usable" in text_out


# ---------------------------------------------------------------------------
# Failure honesty + never-invent guarantees
# ---------------------------------------------------------------------------


class TestFailureHonesty:
    def test_no_gps_anywhere_is_the_only_rejection(self, tmp_path):
        p = _csv(tmp_path / "x.csv", "a,b,c\n1,2,3\n")
        with pytest.raises(TelemetryError, match="no GPS position columns") as ei:
            load_external_telemetry(p)
        msg = str(ei.value)
        assert "timestamp, latitude, longitude, altitude" not in msg
        assert "missing required column" not in msg

    def test_timestamp_never_invented_when_column_absent(self, tmp_path):
        samples = load_external_telemetry(
            _csv(tmp_path / "ni.csv", "latitude,longitude\n24.53,120.93\n"))
        assert samples[0].timestamp_sec is None

    def test_garbage_latitude_rows_flagged_not_invented(self, tmp_path):
        """A row holding a non-numeric latitude is INVALID per §15 (row-level
        flagging); the file is not 'fixed' by inventing coordinates."""
        text = "timestamp,latitude,longitude\n0,hello,120.93\n"
        samples = load_external_telemetry(_csv(tmp_path / "g.csv", text))
        assert samples[0].latitude is None   # nothing invented
        d = parse_telemetry_csv(tmp_path / "g.csv")
        assert d.rows[0]["gps_status"] == "MISSING"
        assert d.quality["missing_gps_rows"] == 1

    def test_empty_file_still_rejected(self, tmp_path):
        with pytest.raises(TelemetryError, match="empty"):
            load_external_telemetry(_csv(tmp_path / "e.csv", ""))

    def test_write_artifacts_never_raises(self, tmp_path):
        d = parse_telemetry_csv(_csv(tmp_path / "ok.csv", _canonical()))
        # Read-only directory → best-effort, no exception.
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o444)
        try:
            write_telemetry_artifacts(d, ro / "sub" / "dir")
        finally:
            ro.chmod(0o755)
