/**
 * STRATA Desktop Runtime Lifecycle & Backend Connection Manager.
 *
 * Handles dynamic backend discovery, health checks, lifecycle states,
 * and system diagnostics for the desktop app shell.
 */

export type EngineStatus = "STARTING" | "READY" | "FAILED" | "STOPPING";

export interface DesktopDiagnostics {
  os: string;
  arch: string;
  backendUrl: string;
  port: number;
  engineReady: boolean;
  storagePath: string;
  appVersion: string;
}

const DEFAULT_PORT = Number(import.meta.env.VITE_API_PORT || 8000);
let activeBackendUrl = `http://127.0.0.1:${DEFAULT_PORT}`;

export function getActiveBackendUrl(): string {
  return activeBackendUrl;
}

export function setActiveBackendUrl(url: string): void {
  activeBackendUrl = url;
}

export async function checkBackendHealth(targetUrl = activeBackendUrl): Promise<boolean> {
  try {
    const res = await fetch(`${targetUrl}/api/runs`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function discoverBackendPort(): Promise<{ status: EngineStatus; url: string; port: number }> {
  // Try default port 8000 first
  if (await checkBackendHealth(activeBackendUrl)) {
    return { status: "READY", url: activeBackendUrl, port: DEFAULT_PORT };
  }

  // Scan fallback local ports (8000-8005)
  for (let port = 8000; port <= 8005; port++) {
    const candidate = `http://127.0.0.1:${port}`;
    if (await checkBackendHealth(candidate)) {
      setActiveBackendUrl(candidate);
      return { status: "READY", url: candidate, port };
    }
  }

  return { status: "FAILED", url: activeBackendUrl, port: DEFAULT_PORT };
}

export function getDesktopDiagnostics(): DesktopDiagnostics {
  const ua = navigator.userAgent;
  let os = "macOS";
  if (ua.includes("Win")) os = "Windows";
  else if (ua.includes("Linux")) os = "Linux";

  return {
    os,
    arch: ua.includes("x86_64") || ua.includes("Intel") ? "x86_64" : "arm64",
    backendUrl: activeBackendUrl,
    port: DEFAULT_PORT,
    engineReady: true,
    storagePath: "data/storage",
    appVersion: "1.0.0",
  };
}
