import { useCallback, useRef, useState } from "react";
import { api } from "../lib/api";
import { friendlyMessage } from "../lib/errors";
import type { DataVideoInfo, VideoMetadata } from "../lib/types";

/** The two ways a run can start, plus the form values the backend reads. */
export interface LaunchRequest {
  /** Upload path: the local file. Null when starting a server-side video. */
  file: File | null;
  /** Data-folder path: the server-side video. Null when uploading a file. */
  dataVideo: DataVideoInfo | null;
  telemetryFile: File | null;
  /** Upload path only: an optional held-out LiDAR reference (LAS/LAZ/NPZ) the
   *  accuracy comparison measures against. The pipeline never reads it. */
  lidarFile: File | null;
  missionName: string;
  description: string;
}

export type LaunchPhase = "idle" | "uploading" | "starting";

export interface MissionLaunch {
  /**
   * Start the run once. A second call while the first is in flight is
   * ignored; after a failure the latch is released so a retry works.
   */
  launch: (req: LaunchRequest) => Promise<void>;
  /** One launch at a time — drives both the button's disabled state and rows. */
  busy: boolean;
  phase: LaunchPhase;
  /** 0-100 while `phase === "uploading"`. */
  uploadProgress: number;
  /** Name of the server-side video being started (for its row label). */
  startedVideoName: string | null;
  /** Server-measured metadata from the upload; null until uploaded. */
  metadata: VideoMetadata | null;
  /** Job id once the upload landed — a retry reuses it instead of re-uploading. */
  uploadedJobId: string | null;
  error: string | null;
  clearError: () => void;
  /** Drop the uploaded job + metadata (the footage changed). */
  reset: () => void;
}

/**
 * The New Mission launch workflow: upload → server metadata → mission record →
 * pipeline → hand the new run id back to the page, or start a server-side
 * video by name. Nothing here renders; the page owns the form and the routing.
 *
 * `onStarted` must be stable (wrap it in useCallback): it is how the page
 * learns a run exists, and the page owns selecting and routing to it.
 */
export function useMissionLaunch(onStarted: (jobId: string, name: string) => void): MissionLaunch {
  const [phase, setPhase] = useState<LaunchPhase>("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [startedVideoName, setStartedVideoName] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [uploadedJobId, setUploadedJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const latch = useRef(false);
  const started = useRef(onStarted);
  started.current = onStarted;

  const clearError = useCallback(() => setError(null), []);
  const reset = useCallback(() => {
    setUploadedJobId(null);
    setMetadata(null);
  }, []);

  const launch = useCallback(
    async (req: LaunchRequest) => {
      // The start button's `disabled` attribute is a render-time signal:
      // clicks dispatched before the next render all pass it, so a careless
      // double/triple click used to start one run per click. Latch the first
      // intent, and release it only on failure — where a retry is meaningful.
      if (latch.current) return;
      const { file, dataVideo, telemetryFile, lidarFile, missionName, description } = req;
      if (!file && !dataVideo) {
        setError("Select a drone video first.");
        return;
      }
      latch.current = true;
      setError(null);
      try {
        if (dataVideo) {
          setStartedVideoName(dataVideo.name);
          setPhase("starting");
          // The backend picks up telemetry sitting beside the video; an
          // explicitly attached file (CSV or a converted DJI SRT) wins.
          const res = telemetryFile
            ? await api.startDataVideoMissionWithTelemetry(dataVideo.name, telemetryFile)
            : await api.startDataVideoMission(dataVideo.name);
          started.current(res.run_id, dataVideo.name);
          return;
        }

        setPhase("uploading");
        let jobId = uploadedJobId;
        let meta = metadata;
        const source = file!;

        if (!jobId) {
          const uploadRes = await api.uploadVideo(
            source,
            (pct) => setUploadProgress(pct),
            telemetryFile,
            missionName,
            lidarFile,
          );
          jobId = uploadRes.job_id;
          setUploadedJobId(jobId);
          setPhase("starting");

          // Prefer the server-measured metadata carried in the upload response
          // (fps, codec, bitrate, GPS tags — things the browser cannot read).
          if (uploadRes.metadata) {
            meta = uploadRes.metadata;
            setMetadata(meta);
          } else {
            const statusRes = await api.getJobStatus(jobId);
            if (statusRes.metadata) {
              meta = statusRes.metadata;
              setMetadata(meta);
            }
          }
        } else {
          setPhase("starting");
        }

        if (!jobId) throw new Error("Failed to initialize project ID for reconstruction.");

        // Mission registration is best-effort: it needs a role that may
        // mutate missions, and reconstruction does not depend on it.
        await api
          .createMission({
            name: missionName || `Mission ${jobId.substring(0, 8)}`,
            project_id: jobId,
            description: description || `Uploaded ${source.name} for 3D reconstruction`,
          })
          .catch(() => {});

        // CSV telemetry lands as telemetry.csv (schema-detected path); DJI
        // SRT is converted to flight_poses.csv at upload and consumed
        // automatically — no payload key for it.
        await api.startPipeline(
          jobId,
          telemetryFile && !telemetryFile.name.toLowerCase().endsWith(".srt")
            ? { telemetry_csv: "telemetry.csv" }
            : {},
        );
        started.current(jobId, missionName || `Mission ${jobId.substring(0, 8)}`);
      } catch (err) {
        latch.current = false;
        setPhase("idle");
        setStartedVideoName(null);
        setUploadProgress(0);
        setError(friendlyMessage(err));
      }
    },
    [metadata, uploadedJobId],
  );

  return {
    launch,
    busy: phase !== "idle",
    phase,
    uploadProgress,
    startedVideoName,
    metadata,
    uploadedJobId,
    error,
    clearError,
    reset,
  };
}
