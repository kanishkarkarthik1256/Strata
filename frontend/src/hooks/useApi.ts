import { useCallback, useEffect, useRef, useState } from "react";

export type RequestState<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; error: unknown };

/**
 * Run an async loader once (or on dependency change) and expose the
 * loading / ready / error states. `reload` re-runs the loader.
 */
export function useApi<T>(loader: () => Promise<T>, deps: unknown[]): {
  state: RequestState<T>;
  reload: () => void;
} {
  const [state, setState] = useState<RequestState<T>>({ status: "loading" });
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(async () => {
    setState({ status: "loading" });
    try {
      setState({ status: "ready", data: await loaderRef.current() });
    } catch (error) {
      setState({ status: "error", error });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    void run();
  }, [run]);

  return { state, reload: run };
}