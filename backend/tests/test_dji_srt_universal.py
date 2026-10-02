"""Universal DJI SRT telemetry — the format matrix must all parse.

DJI shipped (at least) three SRT format families. Before this suite, the
parser accepted only the bracketed family, so legacy GPS-function logs
(Mavic Pro / Phantom 4 era, Format 2 — the ``data/dji`` sample) were
rejected with "no DJI GPS blocks found" and the run silently degraded to
slow video-only SfM. These tests pin every family plus the honest
dispatch:

* Format 1 (bracketed, gimbal-bearing) → flight_poses.csv.
* Formats 2/2b/2c (GPS-function legacy: bare, Matrice-300 ``M`` suffix,
  P4-RTK space-before-paren) → telemetry.csv, position-only, never
  fabricated quaternions.
* Formats 3/3b (HTML comprehensive: SrtCnt/FrameCnt counters) → parsed.
* The measured lat/lon swap guard and the real ``data/dji`` sample file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.dji_srt_telemetry import (
    FORMAT_BRACKETED,
    FORMAT_GPS_FUNCTION,
    convert_srt_for_run,
    parse_srt,
    srt_has_orientation,
    srt_to_telemetry_csv,
)

REAL_DJI_SRT = Path(__file__).resolve().parents[2] / "data" / "dji" / "DJI_0501.SRT"
REAL_DJI_MP4 = Path(__file__).resolve().parents[2] / "data" / "dji" / "DJI_0501.MP4"


# ---------------------------------------------------------------- fixtures

GPS_FUNCTION_BLOCK = (
    "{n}\r\n"
    "00:00:{s:02d},{ms:03d} --> 00:00:{s2:02d},{ms2:03d}\r\n"
    "HOME(-4.0067,57.9816) 2017.07.29 11:58:52\r\n"
    "GPS({lat},{lon},{alt}) BAROMETER:57.2\r\n"
    "ISO:100 Shutter:320 EV:-2/3 Fnum:F2.2 \r\n"
    "\r\n"
)


def _write_gps_function(path: Path, n: int = 10) -> None:
    """Legacy Format 2 (the data/dji sample's family), CRLF line endings."""
    blocks = []
    for i in range(1, n + 1):
        lat = -4.0071 + i * 1e-6  # crawl north a hair
        blocks.append(GPS_FUNCTION_BLOCK.format(
            n=i, s=(i - 1) // 30, ms=((i - 1) % 30) * 33,
            s2=(i - 1) // 30, ms2=((i - 1) % 30) * 33 + 30,
            lat=f"{lat:.4f}", lon="57.9811", alt="18",
        ))
    path.write_bytes("".join(blocks).encode("utf-8"))


def _write_matrice300(path: Path) -> None:
    """Format 2b: unit-suffixed altitude, colon BAROMETER."""
    blocks = []
    for i in range(1, 6):
        blocks.append(
            f"{i}\r\n00:00:0{i-1},000 --> 00:00:0{i},000\r\n"
            f"GPS(36.6146,-6.1120,{5 + i}.0M) BAROMETER:0.3M\r\nD 5.2m H 1.5m\r\n\r\n"
        )
    path.write_bytes("".join(blocks).encode("utf-8"))


def _write_p4rtk(path: Path) -> None:
    """Format 2c: compact single-line, space before paren, int altitude."""
    lines = []
    for i in range(1, 6):
        lat = -58.851741 - i * 1e-6
        lines.append(
            f"{i}\r\n00:00:0{i-1},000 --> 00:00:0{i},000\r\n"
            f"F/5.6, SS 400, ISO 100, EV 0, GPS ({lat:.6f}, -34.23792, 15), "
            "HOME (-58.8475, -34.2327, -57.98m), D 698.70m, H 85.80m\r\n\r\n"
        )
    path.write_bytes("".join(lines).encode("utf-8"))


def _write_html_comprehensive(path: Path) -> None:
    """Format 3: HTML-wrapped SrtCnt + bracketed fields."""
    blocks = []
    for i in range(1, 6):
        blocks.append(
            f'{i}\r\n00:00:0{i-1},000 --> 00:00:0{i},000\r\n'
            f'<font size="36">SrtCnt : {i}, DiffTime : 33ms\r\n'
            f"2024-01-15 14:30:2{i-1},000\r\n"
            "[iso : 100] [shutter : 1/1000] [fnum : 280] [focal_len : 240] "
            f"[latitude: 59.3023{i:02d}] [longitude: 18.203059] "
            "[rel_alt: 10.200 abs_alt: 142.760]</font>\r\n\r\n"
        )
    path.write_bytes("".join(blocks).encode("utf-8"))


def _write_bracketed(path: Path, n: int = 40) -> None:
    """Format 1 with gimbal fields (the previously-only supported shape)."""
    blocks = []
    for i in range(1, n + 1):
        blocks.append(
            f"{i}\r\n00:00:{(i-1)//30:02d},{((i-1)%30)*33:03d} --> "
            f"00:00:{(i-1)//30:02d},{((i-1)%30)*33+30:03d}\r\n"
            f"[latitude: 59.302335] [longitude: 18.203059] "
            f"[rel_alt: {1.0 + i*0.01:.3f} abs_alt: 132.860] "
            f"[gb_yaw: 45.0 gb_pitch: -30.0 gb_roll: 0.5] [focal_len: 240]\r\n\r\n"
        )
    path.write_bytes("".join(blocks).encode("utf-8"))


# ------------------------------------------------------------- format matrix

def test_gps_function_legacy_parses(tmp_path: Path):
    p = tmp_path / "DJI_0501.SRT"
    _write_gps_function(p)
    frames = parse_srt(p)
    assert len(frames) == 10
    assert not srt_has_orientation(frames)
    assert frames[0].latitude == pytest.approx(-4.0071)
    assert frames[0].rel_alt == pytest.approx(18.0)
    assert frames[0].time_s == pytest.approx(0.0)


def test_matrice300_unit_suffix_parses(tmp_path: Path):
    p = tmp_path / "M300.SRT"
    _write_matrice300(p)
    frames = parse_srt(p)
    assert len(frames) == 5
    assert frames[2].rel_alt == pytest.approx(8.0)  # "8.0M" suffix tolerated


def test_p4rtk_compact_parses(tmp_path: Path):
    p = tmp_path / "P4RTK.SRT"
    _write_p4rtk(p)
    frames = parse_srt(p)
    assert len(frames) == 5
    assert not srt_has_orientation(frames)
    assert frames[0].latitude == pytest.approx(-58.851741)


def test_html_comprehensive_parses(tmp_path: Path):
    p = tmp_path / "Mavic3.SRT"
    _write_html_comprehensive(p)
    frames = parse_srt(p)
    assert len(frames) == 5
    assert frames[4].frame_cnt == 5  # SrtCnt, not the subtitle index
    assert frames[0].latitude == pytest.approx(59.302301)


def test_bracketed_with_gimbal_still_parses(tmp_path: Path):
    p = tmp_path / "Mini3.SRT"
    _write_bracketed(p)
    frames = parse_srt(p)
    assert len(frames) == 40
    assert srt_has_orientation(frames)


# -------------------------------------------------- position-only conversion

def test_position_only_converts_to_telemetry_csv(tmp_path: Path):
    p = tmp_path / "DJI_0501.SRT"
    _write_gps_function(p, n=60)
    out = tmp_path / "telemetry.csv"
    prov = srt_to_telemetry_csv(p, out)
    text = out.read_text().splitlines()
    assert text[0] == "timestamp,latitude,longitude,altitude"
    rows = [r.split(",") for r in text[1:]]
    assert len(rows) == 60
    # Timecodes come from the SRT timeline, not invented indices.
    assert rows[0][0] == "0.000"
    assert rows[59][0] == "1.957"  # block 60: 1 s carry + 29*33 ms
    assert prov["mode"] == "DJI_SRT_TELEMETRY_POSITION_ONLY"
    assert prov["samples"] == 60
    assert prov["blocks_with_timecode"] == 60


def test_orientation_log_refuses_telemetry_csv_target(tmp_path: Path):
    p = tmp_path / "Mini3.SRT"
    _write_bracketed(p, n=5)
    with pytest.raises(ValueError, match="gimbal orientation"):
        srt_to_telemetry_csv(p, tmp_path / "telemetry.csv")


def test_non_dji_file_rejected(tmp_path: Path):
    p = tmp_path / "movie.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:00,500\nJust some dialogue.\n\n")
    with pytest.raises(ValueError, match="no DJI GPS blocks"):
        parse_srt(p)


# ------------------------------------------------------------ swap guard

def test_swapped_latlon_corrected_and_flagged(tmp_path: Path):
    """A GPS(lat,lon) pair whose first slot is >|90| is impossible — the
    parser corrects the swapped order once instead of folding the
    trajectory or dropping real telemetry. (A merely *plausible* swap of
    two in-range coordinates is not detectable and is trusted as logged.)"""
    blocks = []
    for i in range(1, 6):
        # 98.7654 cannot be a latitude; the real lat sits in the lon slot
        blocks.append(
            f"{i}\r\n00:00:0{i-1},000 --> 00:00:0{i},000\r\n"
            f"GPS(98.7654,-4.0071,18) BAROMETER:57.2\r\n\r\n"
        )
    p = tmp_path / "swapped.SRT"
    p.write_bytes("".join(blocks).encode("utf-8"))
    frames = parse_srt(p)
    assert frames[0].latitude == pytest.approx(-4.0071)
    assert frames[0].longitude == pytest.approx(98.7654)


# ------------------------------------------------------------ dispatch

def test_convert_srt_for_run_dispatches_by_format(tmp_path: Path):
    legacy = tmp_path / "legacy.SRT"
    _write_gps_function(legacy, n=30)
    ws = tmp_path / "run_a"
    ws.mkdir()
    prov = convert_srt_for_run(legacy, ws, video_fps=30.0)
    assert prov["artifact"] == "telemetry.csv"
    assert (ws / "telemetry.csv").is_file()
    assert not (ws / "flight_poses.csv").exists()

    gimbal = tmp_path / "gimbal.SRT"
    _write_bracketed(gimbal)
    ws2 = tmp_path / "run_b"
    ws2.mkdir()
    prov2 = convert_srt_for_run(gimbal, ws2, video_fps=30.0)
    assert prov2["artifact"] == "flight_poses.csv"
    assert (ws2 / "flight_poses.csv").is_file()


def test_convert_srt_for_run_rejects_non_dji(tmp_path: Path):
    junk = tmp_path / "junk.SRT"
    junk.write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n\n")
    ws = tmp_path / "run"
    ws.mkdir()
    with pytest.raises(ValueError, match="no DJI GPS blocks"):
        convert_srt_for_run(junk, ws, video_fps=30.0)
    assert not (ws / "telemetry.csv").exists()
    assert not (ws / "flight_poses.csv").exists()


# ------------------------------------------------------------ real sample

@pytest.mark.skipif(not REAL_DJI_SRT.is_file(), reason="data/dji sample not present")
def test_real_data_dji_sample_parses(tmp_path: Path):
    """The actual file that failed in production (legacy format, CRLF).

    ``parse_srt`` reports what the log WROTE — this family writes
    ``GPS(longitude, latitude, altitude)``, so the first slot is the
    longitude here. The coordinate ORDER is settled downstream against the
    video's own fix (see the order tests below); the parser never guesses.
    """
    frames = parse_srt(REAL_DJI_SRT)
    assert len(frames) == 276
    assert not srt_has_orientation(frames)
    assert frames[0].latitude == pytest.approx(-4.0071)
    assert frames[0].longitude == pytest.approx(57.9811)
    assert frames[-1].time_s == pytest.approx(8.28)
    out = tmp_path / "telemetry.csv"
    prov = srt_to_telemetry_csv(REAL_DJI_SRT, out)
    rows = out.read_text().splitlines()[1:]
    assert len(rows) == 276
    assert prov["blocks_with_timecode"] == 276


@pytest.mark.skipif(
    not (REAL_DJI_SRT.is_file() and REAL_DJI_MP4.is_file()),
    reason="data/dji sample not present",
)
def test_real_dji_sample_order_settled_by_the_video_fix(tmp_path: Path):
    """The real pair: the log's written order is wrong for this flight.

    Ground truth is the video's own ISO 6709 atom: ``+57.981100-4.007092``
    (lat 57.9811, lon -4.007092 — Scotland). The log writes the same fix as
    ``GPS(-4.0071,57.9811,18)``, i.e. (lon, lat). Before this, the assumed
    order placed the whole reconstruction at lat -4.0 / lon 58.0 (Indian
    Ocean) with every internal check still passing, because everything
    downstream agreed with itself.
    """
    from app.services.metadata_extraction import extract_metadata, read_container_gps

    container = read_container_gps(REAL_DJI_MP4)
    assert container is not None
    assert container["lat"] == pytest.approx(57.9811, abs=1e-4)
    assert container["lon"] == pytest.approx(-4.007092, abs=1e-5)

    meta = extract_metadata(REAL_DJI_MP4)
    ws = tmp_path / "run"
    ws.mkdir()
    prov = convert_srt_for_run(
        REAL_DJI_SRT, ws, video_fps=meta.fps, video_gps=(meta.gps_lat, meta.gps_lon)
    )
    assert prov["coordinate_order"] == "longitude_latitude"
    assert prov["coordinate_order_source"] == "video_gps"
    evidence = prov["coordinate_order_evidence"]
    assert evidence["swapped_delta_deg"] < evidence["direct_delta_deg"] / 100.0

    header, first = (ws / "telemetry.csv").read_text().splitlines()[:2]
    assert header == "timestamp,latitude,longitude,altitude"
    ts, lat, lon, alt = first.split(",")
    assert float(lat) == pytest.approx(57.9811, abs=1e-4)
    assert float(lon) == pytest.approx(-4.0071, abs=1e-4)
    assert float(alt) == pytest.approx(18.0)


@pytest.mark.skipif(not REAL_DJI_SRT.is_file(), reason="data/dji sample not present")
def test_without_video_evidence_the_written_order_is_kept_and_named(tmp_path: Path):
    """No independent fix -> keep the log's order, and SAY it is convention.

    The honest-degradation path: no evidence is never presented as measured.
    """
    ws = tmp_path / "run"
    ws.mkdir()
    prov = convert_srt_for_run(REAL_DJI_SRT, ws, video_fps=30.0, video_gps=None)
    assert prov["coordinate_order"] == "latitude_longitude"
    assert prov["coordinate_order_source"] == "format_convention"
    assert "coordinate_order_evidence" not in prov
    ts, lat, lon, _alt = (ws / "telemetry.csv").read_text().splitlines()[1].split(",")
    assert float(lat) == pytest.approx(-4.0071)
    assert float(lon) == pytest.approx(57.9811)


class TestCoordinateOrderEvidence:
    """`resolve_positional_order` decides only on evidence, never assumption."""

    def _frames(self, lat: float, lon: float, n: int = 5):
        from app.services.dji_srt_telemetry import SrtFrame

        return [SrtFrame(frame_cnt=i + 1, latitude=lat, longitude=lon, rel_alt=10.0,
                         abs_alt=10.0, gimbal_yaw=0.0, gimbal_pitch=0.0,
                         gimbal_roll=0.0, focal_len_mm=0.0) for i in range(n)]

    def test_swapped_reading_matching_the_video_fix_is_corrected(self):
        from app.services.dji_srt_telemetry import resolve_positional_order

        frames, prov = resolve_positional_order(self._frames(-4.0071, 57.9811), (57.9811, -4.007092))
        assert prov["coordinate_order"] == "longitude_latitude"
        assert prov["coordinate_order_source"] == "video_gps"
        assert frames[0].latitude == pytest.approx(57.9811)
        assert frames[0].longitude == pytest.approx(-4.0071)

    def test_correct_reading_is_left_alone(self):
        from app.services.dji_srt_telemetry import resolve_positional_order

        frames, prov = resolve_positional_order(self._frames(57.9811, -4.007092), (57.9811, -4.007092))
        assert prov["coordinate_order"] == "latitude_longitude"
        assert prov["coordinate_order_source"] == "video_gps_agrees"
        assert frames[0].latitude == pytest.approx(57.9811)
        assert frames[0].longitude == pytest.approx(-4.007092)

    def test_a_far_away_video_fix_does_not_trigger_a_swap(self):
        """Evidence must MATCH, not merely exist: a drone flown in one place
        cannot justify reinterpreting an unrelated fix."""
        from app.services.dji_srt_telemetry import resolve_positional_order

        frames, prov = resolve_positional_order(self._frames(-4.0071, 57.9811), (10.0, 20.0))
        assert prov["coordinate_order"] == "latitude_longitude"
        assert frames[0].latitude == pytest.approx(-4.0071)
        assert frames[0].longitude == pytest.approx(57.9811)

    def test_iso6709_renderings_all_parse(self):
        from app.services.metadata_extraction import parse_iso6709

        for text in ("+57.981100-4.007092+57.200/", "+57.9811-004.0071/", "+57.9811-4.00709/"):
            iso = parse_iso6709(text)
            assert iso is not None, text
            assert iso["lat"] == pytest.approx(57.9811)
            assert iso["lon"] == pytest.approx(-4.0071, abs=1e-4)
        # No coordinate pair -> None (never a guessed fix).
        assert parse_iso6709("not a location") is None
        assert parse_iso6709("") is None
