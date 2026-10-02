/**
 * STRATA — API client. Single source of truth for backend communication.
 *
 * All calls go through Vite's dev-server proxy (/api → FastAPI). The client
 * attaches the bearer token when present and maps HTTP/network failures to
 * ApiError with status-aware messages (see lib/errors.ts).
 */

import { ApiError } from "./errors";
import type {
  AuthUser,
  DataVideoListResponse,
  DataVideoStartResponse,
  DsmAccuracy,
  HostCapabilities,
  JobStatusResponse,
  LoginResponse,
  Mission,
  MissionListResponse,
  PipelineStartResponse,
  PipelineStatusResponse,
  PosesJson,
  RegisterResponse,
  RunAnalysis,
  RunDetail,
  RunListResponse,
  UploadResponse,
  ViewerAlignment,
} from "./types";

import { getActiveBackendUrl } from "./desktopRuntime";

const TOKEN_KEY = "strata.token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* storage unavailable — session-only auth */
  }
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* noop */
  }
}

function resolveUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  if ((window as any).__TAURI__ || window.location.protocol === "file:" || window.location.protocol === "tauri:") {
    return `${getActiveBackendUrl()}${path}`;
  }
  return path;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const isFormData = typeof FormData !== "undefined" && init?.body instanceof FormData;
  if (init?.body && !isFormData) headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const targetUrl = resolveUrl(path);
  let res: Response;
  try {
    res = await fetch(targetUrl, { ...init, headers });
  } catch {
    throw new ApiError("Cannot connect to the local STRATA engine.", null);
  }

  if (!res.ok) {
    let detail: unknown;
    try {
      const body = await res.json();
      detail = body?.detail ?? body?.error ?? body?.message ?? null;
    } catch {
      detail = null;
    }
    throw new ApiError(
      typeof detail === "string" ? detail : `Request failed (${res.status})`,
      res.status,
      detail,
    );
  }
  return (await res.json()) as T;
}

export const api = {
  /* ---- system ---- */
  getHealth: () => request<{ status: string }>("/health"),
  getCapabilities: () => request<HostCapabilities>("/api/system/capabilities"),

  /* ---- runs (Phase 9.5 filesystem artifacts) ---- */
  getRuns: () => request<RunListResponse>("/api/runs"),
  getRun: (runId: string) => request<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`),
  getRunManifest: (runId: string) =>
    request<Record<string, unknown>>(`/api/runs/${encodeURIComponent(runId)}/manifest`),
  /** URL for a run artifact file (PLY/JSON/…). Not fetched through `request`
   *  because the response is binary or used directly as a loader URL. */
  artifactUrl: (runId: string, name: string) =>
    `/api/runs/${encodeURIComponent(runId)}/artifact/${name}`,
  getRunJsonArtifact: (runId: string, name: string) =>
    request<Record<string, unknown>>(
      `/api/runs/${encodeURIComponent(runId)}/artifact/${name}`,
    ),
  getPoses: (runId: string) =>
    request<PosesJson>(`/api/runs/${encodeURIComponent(runId)}/artifact/poses.json`),
  getViewerAlignment: (runId: string) =>
    request<ViewerAlignment>(`/api/runs/${encodeURIComponent(runId)}/viewer-alignment`),
  /**
   * Accuracy report for a run, served by the single accuracy engine
   * (app.services.metric_validation). When the run has no measurement artifact
   * the backend returns an honest NOT CERTIFIED envelope in the same shape —
   * never a fabricated figure. Accuracy is deliberately read through this
   * endpoint rather than as a raw artifact path, so the UI can only ever show
   * what the engine measured.
   */
  getMetricValidation: (runId: string) =>
    request<Record<string, unknown>>(
      `/api/reconstruction/metric-validation/${encodeURIComponent(runId)}`,
    ),
  /** Ground-truth DSM accuracy (404 when the run has no reference grid). */
  getDsmAccuracy: (runId: string) =>
    request<DsmAccuracy>(
      `/api/reconstruction/dsm-accuracy/${encodeURIComponent(runId)}`,
    ),
  /** Area/elevation/flight/error-map analysis for a run (backend engine). */
  getRunAnalysis: (runId: string) =>
    request<RunAnalysis>(`/api/runs/${encodeURIComponent(runId)}/analysis`),

  /* ---- upload & ingestion (Phase 11.3) ---- */
  uploadVideo: (
    file: File,
    onProgress?: (pct: number) => void,
    telemetryFile?: File | null,
    missionName?: string,
    lidarFile?: File | null,
  ): Promise<UploadResponse> => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", resolveUrl("/api/upload"));
      const token = getToken();
      if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      if (onProgress && xhr.upload) {
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && e.total > 0) {
            onProgress(Math.round((e.loaded / e.total) * 100));
          }
        };
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new ApiError("Failed to parse upload response", xhr.status));
          }
        } else {
          let detail: string | null = null;
          try {
            const res = JSON.parse(xhr.responseText);
            detail = res?.detail ?? res?.error ?? res?.message ?? null;
          } catch {
            /* noop */
          }
          reject(new ApiError(detail || `Upload failed (${xhr.status})`, xhr.status, detail));
        }
      };
      xhr.onerror = () => reject(new ApiError("Network error during file upload", 0));
      const formData = new FormData();
      formData.append("file", file);
      if (telemetryFile) formData.append("telemetry", telemetryFile);
      // Held out from reconstruction: stored in the run's reference/ directory
      // and only ever read back by the accuracy comparison.
      if (lidarFile) formData.append("lidar", lidarFile);
      if (missionName && missionName.trim()) formData.append("mission_name", missionName.trim());
      xhr.send(formData);
    });
  },

  getJobStatus: (jobId: string) =>
    request<JobStatusResponse>(`/api/upload/${encodeURIComponent(jobId)}`),

  createMission: (data: { name: string; project_id: string; description?: string }) =>
    request<Mission>("/api/missions", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  startPipeline: (jobId: string, options: Record<string, unknown> = {}) =>
    request<PipelineStartResponse>(`/api/pipeline/start/${encodeURIComponent(jobId)}`, {
      method: "POST",
      body: JSON.stringify({
        extraction_mode: "every_n",
        every_n: 10,
        target_fps: 2.0,
        depth_backend: "auto",
        force: [],
        ...options,
      }),
    }),

  getPipelineStatus: (jobId: string) =>
    request<PipelineStatusResponse>(`/api/pipeline/status/${encodeURIComponent(jobId)}`),

  cancelPipeline: (jobId: string) =>
    request<{ job_id: string; cancelled: boolean }>(
      `/api/pipeline/cancel/${encodeURIComponent(jobId)}`,
      { method: "POST" },
    ),

  /* ---- canonical & fast demo ---- */
  listDataVideos: () => request<DataVideoListResponse>("/api/data-videos"),

  startDataVideoMissionWithTelemetry: (name: string, telemetry: File) => {
    const formData = new FormData();
    formData.append("telemetry", telemetry);
    return request<{ run_id: string; video: string; status: string; message?: string }>(
      `/api/data-videos/${encodeURIComponent(name)}/start-with-telemetry`,
      { method: "POST", body: formData },
    );
  },

  startDataVideoMission: (name: string, payload: Record<string, unknown> = {}) =>
    request<DataVideoStartResponse>(
      `/api/data-videos/${encodeURIComponent(name)}/start`,
      { method: "POST", body: JSON.stringify(payload) }
    ),

  /* ---- missions (auth required) ---- */
  getMissions: () => request<MissionListResponse>("/api/missions"),

  /* ---- auth ---- */
  me: () => request<AuthUser>("/api/auth/me"),
  login: (email: string, password: string) =>
    request<LoginResponse>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  register: (email: string, password: string) =>
    request<RegisterResponse>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  logout: () =>
    request<{ revoked: boolean }>("/api/auth/logout", { method: "POST" }).catch(() => ({
      revoked: false,
    })),
};