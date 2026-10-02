import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import GroundLevelBlock, { levelSourceLabel } from "./GroundLevelBlock";
import type { ViewerAlignment } from "../../lib/types";

function alignment(over: Partial<ViewerAlignment> = {}): ViewerAlignment {
  return {
    version: 2,
    run_id: "run",
    source_coordinate_convention: "x",
    viewer_coordinate_convention: "y",
    R_view: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    t_view: [0, 0, 0],
    scale: 1,
    ground_normal_world: [0, 0, 1],
    ground_reference_point_world: [0, 0, 0],
    horizontal_direction_world: [1, 0, 0],
    method: "geometry_plane_fit",
    confidence: "high",
    created_at: "2026-01-01T00:00:00Z",
    ...over,
  };
}

describe("GroundLevelBlock — levelling provenance", () => {
  it("names the metric ENU vertical and the measured plane", () => {
    render(
      <GroundLevelBlock
        alignment={alignment({
          leveling: {
            up_prior_source: "enu_vertical",
            confidence: "high",
            plane_measured: true,
          },
        })}
      />,
    );
    expect(screen.getByText(/TELEMETRY ENU VERTICAL/)).toBeTruthy();
    expect(screen.getByText("HIGH")).toBeTruthy();
    expect(screen.getByText("YES")).toBeTruthy();
  });

  it("does not pass off an unverified camera prior as a measured ground plane", () => {
    render(
      <GroundLevelBlock
        alignment={alignment({
          method: "up_prior:camera_image_up",
          confidence: "low",
          leveling: {
            up_prior_source: "camera_image_up",
            confidence: "low",
            plane_measured: false,
          },
        })}
      />,
    );
    expect(screen.getByText(/CAMERA ORIENTATION PRIOR/)).toBeTruthy();
    expect(screen.getByText("LOW")).toBeTruthy();
    expect(screen.getByText("NO")).toBeTruthy();
    expect(screen.getByText(/may be tilted relative to the ground/)).toBeTruthy();
  });

  it("says so when a run has no levelling at all", () => {
    render(
      <GroundLevelBlock
        alignment={alignment({
          method: "identity_fallback",
          confidence: "low",
          fallback_reason: "ground plane could not be estimated",
          leveling: null,
        })}
      />,
    );
    expect(screen.getByText(/NOT ESTIMATED/)).toBeTruthy();
    expect(screen.getByText(/ground plane could not be estimated/)).toBeTruthy();
  });

  it("handles a missing alignment payload", () => {
    render(<GroundLevelBlock alignment={null} />);
    expect(screen.getByText(/NOT ESTIMATED/)).toBeTruthy();
  });

  it("falls back to the raw source id for an unknown source", () => {
    expect(levelSourceLabel("some_new_source")).toBe("some_new_source");
    expect(levelSourceLabel(null)).toBeNull();
  });
});
