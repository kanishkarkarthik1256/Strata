/**
 * API error handling — distinguishes HTTP status classes and network failure,
 * and maps each to a useful user-facing message (never a bare "Something went
 * wrong").
 */

export class ApiError extends Error {
  /** HTTP status, or null when the request never reached the backend. */
  readonly status: number | null;
  /** Machine-readable detail from the backend, when present. */
  readonly detail: unknown;

  constructor(message: string, status: number | null, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function isApiError(err: unknown): err is ApiError {
  return err instanceof ApiError;
}

/** User-facing message for a thrown error, by status class. */
export function friendlyMessage(err: unknown): string {
  if (!isApiError(err)) {
    return err instanceof Error ? err.message : "An unexpected error occurred.";
  }
  if (typeof err.detail === "string" && err.detail.trim().length > 0) {
    return err.detail;
  }
  switch (err.status) {
    case 401:
      return "Authentication required. Sign in to continue.";
    case 403:
      return "You don't have permission to perform this action.";
    case 404:
      return "This item is not available for the selected run.";
    case 422:
      return "The request format was rejected by the backend.";
    case 400:
      return typeof err.detail === "string" ? err.detail : "The request was rejected by the backend.";
    default:
      if (err.status !== null && err.status >= 500) {
        return "The reconstruction service returned an internal error.";
      }
      if (err.status === null) {
        // Network failure — api.ts already set the engine-offline message.
        return err.message;
      }
      return "The request could not be completed.";
  }
}