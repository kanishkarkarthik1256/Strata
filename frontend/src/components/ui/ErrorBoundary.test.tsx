import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ErrorBoundary from "./ErrorBoundary";

function Bomb(): never {
  throw new Error("boom: caps.depth_model is undefined");
}

describe("ErrorBoundary", () => {
  it("renders children when nothing throws", () => {
    render(
      <ErrorBoundary>
        <div>shell content</div>
      </ErrorBoundary>,
    );
    expect(screen.getByText("shell content")).toBeInTheDocument();
  });

  it("catches a render crash instead of blanking the app", () => {
    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>,
    );
    expect(screen.getByText("Something went wrong.")).toBeInTheDocument();
    expect(screen.getByText("Retry")).toBeInTheDocument();
    expect(screen.queryByText("shell content")).not.toBeInTheDocument();
  });

  it("reveals technical details on demand", () => {
    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>,
    );
    fireEvent.click(screen.getByText("Technical details"));
    expect(screen.getByText(/boom: caps\.depth_model is undefined/)).toBeInTheDocument();
  });

  it("recovers after Retry when the crash is transient", () => {
    let shouldThrow = true;
    function Flaky() {
      if (shouldThrow) throw new Error("transient");
      return <div>recovered</div>;
    }
    render(
      <ErrorBoundary>
        <Flaky />
      </ErrorBoundary>,
    );
    expect(screen.getByText("Something went wrong.")).toBeInTheDocument();
    shouldThrow = false;
    fireEvent.click(screen.getByText("Retry"));
    expect(screen.getByText("recovered")).toBeInTheDocument();
  });
});