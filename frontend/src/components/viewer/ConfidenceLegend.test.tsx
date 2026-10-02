import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ConfidenceLegend from "./ConfidenceLegend";

/**
 * The confidence legend must reflect the MEASURED state of the loaded cloud:
 * the measured band when the pipeline produced per-point confidence, and the
 * honest "not available" state only when it did not — never the reverse.
 */

afterEach(() => vi.unstubAllGlobals());

describe("ConfidenceLegend", () => {
  it("shows the measured confidence band when the cloud carries confidence", () => {
    render(
      <ConfidenceLegend threshold={60} onThreshold={() => {}} available range={{ min: 0.468, max: 1.0 }} />,
    );
    expect(screen.getByText(/Measured per-point confidence: 0.47\u20131.00/)).toBeInTheDocument();
    expect(screen.queryByText(/Spatial confidence data is not available/)).not.toBeInTheDocument();
  });

  it("keeps the honest not-available state for clouds without confidence", () => {
    render(
      <ConfidenceLegend threshold={60} onThreshold={() => {}} available={false} range={null} />,
    );
    expect(screen.getByText(/Spatial confidence data is not available/)).toBeInTheDocument();
    expect(screen.queryByText(/Measured per-point confidence/)).not.toBeInTheDocument();
  });

  it("shows a checking state while loading, then reports threshold changes", () => {
    const onThreshold = vi.fn();
    const { rerender } = render(
      <ConfidenceLegend threshold={60} onThreshold={onThreshold} available={null} range={null} />,
    );
    expect(screen.getByText(/Checking confidence data/)).toBeInTheDocument();
    rerender(
      <ConfidenceLegend threshold={75} onThreshold={onThreshold} available={false} range={null} />,
    );
    expect(screen.getByText("Threshold: 75%")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Confidence threshold"), { target: { value: "80" } });
    expect(onThreshold).toHaveBeenCalledWith(80);
  });
});
