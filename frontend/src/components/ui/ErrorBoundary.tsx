import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
  showDetails: boolean;
}

/**
 * A page/component crash must never blank the whole application: the shell
 * keeps rendering and this boundary shows a retry + expandable technical
 * details instead.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, showDetails: false };

  static getDerivedStateFromError(error: Error): State {
    return { error, showDetails: false };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("ErrorBoundary caught:", error, info.componentStack);
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children;
    return (
      <div className="card error-boundary">
        <div className="card-title">Something went wrong.</div>
        <p style={{ color: "var(--gray-600)", fontSize: 14, margin: "8px 0 16px" }}>
          This part of the application failed to render. The rest of STRATA
          is still available.
        </p>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            className="btn btn-primary"
            onClick={() => this.setState({ error: null, showDetails: false })}
          >
            Retry
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => this.setState((s) => ({ ...s, showDetails: !s.showDetails }))}
          >
            {this.state.showDetails ? "Hide technical details" : "Technical details"}
          </button>
        </div>
        {this.state.showDetails && (
          <pre
            style={{
              marginTop: 12,
              padding: 12,
              background: "var(--gray-50)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
              overflow: "auto",
              whiteSpace: "pre-wrap",
            }}
          >
            {this.state.error.name}: {this.state.error.message}
            {"\n"}
            {this.state.error.stack}
          </pre>
        )}
      </div>
    );
  }
}