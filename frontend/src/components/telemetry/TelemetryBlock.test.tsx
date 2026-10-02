import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import TelemetryBlock, {
  formatOffset,
  hasLargeClockOffset,
  telemetryViewFromManifest,
  LARGE_OFFSET_FRACTION,
  type TelemetrySyncView,
} from "./TelemetryBlock";

function externalView(overrides: Partial<NonNullable<TelemetrySyncView["sync"]>> = {}): TelemetrySyncView {
  return {
    telemetry_mode: "VIDEO_WITH_EXTERNAL_TELEMETRY",
    sync: {
      telemetry_samples: 120,
      matched_frames: 96,
      unmatched_frames: 4,
      timestamp_offset_sec: 0.25,
      gps_available: true,
      telemetry_quality: "good",
      invalid_samples_dropped: 0,
      median_sample_spacing_sec: 5,
      sufficient_for_georeferencing: true,
      ...overrides,
    },
    note: null,
  };
}

afterEach(() => vi.restoreAllMocks());

describe("formatOffset", () => {
  it("signs non-zero offsets and trims trailing zeros", () => {
    expect(formatOffset(0.25)).toBe("+0.25s");
    expect(formatOffset(-1.5)).toBe("−1.5s");
    expect(formatOffset(0)).toBe("0s");
    expect(formatOffset(123.4)).toBe("+123s");
  });
});

describe("hasLargeClockOffset", () => {
  it("warns when offset exceeds half the sample spacing", () => {
    expect(hasLargeClockOffset(3, 5)).toBe(true); // 3 > 5 * 0.5
    expect(hasLargeClockOffset(2.5, 5)).toBe(false); // exactly at threshold
    expect(hasLargeClockOffset(LARGE_OFFSET_FRACTION * 5 + 0.01, 5)).toBe(true);
  });

  it("ignores tiny absolute offsets and unknown density", () => {
    expect(hasLargeClockOffset(0.25, undefined)).toBe(false); // below floor
    expect(hasLargeClockOffset(0.9, 0)).toBe(true); // unknown density, big offset
    expect(hasLargeClockOffset(NaN, 5)).toBe(false);
  });
});

describe("telemetryViewFromManifest", () => {
  it("reads the flat sync dict and the mode-only stub", () => {
    const flat = telemetryViewFromManifest({
      mode: "VIDEO_WITH_EXTERNAL_TELEMETRY",
      matched_frames: 9,
      telemetry_samples: 10,
    });
    expect(flat?.telemetry_mode).toBe("VIDEO_WITH_EXTERNAL_TELEMETRY");
    expect(flat?.sync?.matched_frames).toBe(9);

    const stub = telemetryViewFromManifest({ mode: "VIDEO_ONLY" });
    expect(stub?.telemetry_mode).toBe("VIDEO_ONLY");
    expect(stub?.sync).toBeNull();

    expect(telemetryViewFromManifest(undefined)).toBeNull();
  });
});

describe("TelemetryBlock", () => {
  it("video-only shows GPS Not available with no sync grid", () => {
    render(<TelemetryBlock data={{ telemetry_mode: "VIDEO_ONLY" }} />);
    expect(screen.getByText("Not available")).toBeTruthy();
    expect(screen.getByText("video-only")).toBeTruthy();
    expect(screen.queryByText("Samples")).toBeNull();
  });

  it("external telemetry shows samples, matched/unmatched, offset, quality", () => {
    render(<TelemetryBlock data={externalView()} />);
    const block = screen.getByLabelText("GPS telemetry");
    expect(within(block).getByText("Available")).toBeTruthy();
    expect(within(block).getByText("120")).toBeTruthy(); // samples
    expect(within(block).getByText("+0.25s")).toBeTruthy();
    expect(within(block).getByText("good")).toBeTruthy();
    expect(within(block).getByText("4")).toBeTruthy(); // unmatched
    expect(screen.queryByText(/CSV clock likely/)).toBeNull();
  });

  it("warns when the clock offset is large relative to sample spacing", () => {
    render(
      <TelemetryBlock
        data={externalView({ timestamp_offset_sec: 4, median_sample_spacing_sec: 5 })}
      />,
    );
    expect(screen.getByText(/CSV clock likely doesn't match the video/)).toBeTruthy();
  });

  it("counts dropped samples next to matched frames", () => {
    render(<TelemetryBlock data={externalView({ invalid_samples_dropped: 2 })} />);
    expect(screen.getByText(/2 dropped/)).toBeTruthy();
  });
});
