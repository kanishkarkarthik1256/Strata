# Backend ↔ UI Integration Map

## System

| UI Feature | Backend Service | Endpoint | Data | Status |
|---|---|---|---|---|
| System status panel | `resources.py` | `GET /api/system/capabilities` | SystemCapabilities | ✅ Wired |
| Backend health | health check | `GET /health` | `{status: "ok"}` | ✅ Wired |
| Queue status | `job_queue.py` | `GET /api/system/queue` | QueueStatus | ✅ Type defined |

## Missions

| UI Feature | Backend Service | Endpoint | Data | Status |
|---|---|---|---|---|
| Mission list | `mission_service.py` | `GET /api/missions` | Mission[] | ✅ Wired |
| Create mission | `mission_service.py` | `POST /api/missions` | Mission | ✅ Type defined |
| Start mission | `mission_service.py` | `POST /api/missions/{id}/start` | — | ✅ Type defined |
| Cancel mission | `mission_service.py` | `POST /api/missions/{id}/cancel` | — | ✅ Type defined |
| Run history | run organization | `GET /api/missions/runs` | Run[] | ✅ Type defined |
| Run manifest | run organization | `GET /api/missions/runs/{id}/manifest` | RunManifest | ✅ Wired |

## 3D Viewer

| UI Feature | Backend Service | Endpoint | Data | Status |
|---|---|---|---|---|
| Dense PLY | artifact storage | `GET /api/intel/artifact/{run}/dense_model.ply` | Binary PLY | ✅ Wired |
| Sparse PLY | artifact storage | `GET /api/intel/artifact/{run}/sparse_model.ply` | Binary PLY | ✅ Wired |
| Camera poses | artifact storage | `GET /api/intel/artifact/{run}/poses.json` | PosesJson | ✅ Wired |
| Camera trajectory | poses.json `t` vectors | (same endpoint) | Vector3[] | ✅ Computed in viewer |
| GPS markers | poses.json `gps` fields | (same endpoint) | GPS markers | ✅ Computed in viewer |
| Confidence heatmap | reconstruction confidence | — | Per-point confidence | ⚠️ Legend only (no per-point data yet) |

## Analysis

| UI Feature | Backend Service | Endpoint | Data | Status |
|---|---|---|---|---|
| Dashboard data | intel pipeline | `GET /api/intel/dashboard/{job}` | IntelDashboard | ✅ Wired |
| Suggestions | intel pipeline | (same endpoint) | string[] | ✅ Wired |

## Reports

| UI Feature | Backend Service | Endpoint | Data | Status |
|---|---|---|---|---|
| Report data | intel pipeline | `GET /api/intel/report/{job}` | IntelReport | ✅ Type defined |
| Run manifest | run organization | `GET /api/missions/runs/{id}/manifest` | RunManifest | ✅ Wired |
| PDF export | — | — | — | ⚠️ Button only |

## AI Copilot

| UI Feature | Backend Service | Endpoint | Data | Status |
|---|---|---|---|---|
| Chat | intel copilot | `POST /api/intel/copilot/{job}` | CopilotResponse | ✅ Wired |
| Structured actions | intel copilot | (same response) | CopilotAction[] | ⚠️ Type defined, UI not wired |

## New endpoints needed

None. The frontend consumes all existing backend endpoints.
