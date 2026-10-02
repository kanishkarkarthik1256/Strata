import React, { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { useRuns } from "../hooks/useRuns";
import { useFootageProbe } from "../hooks/useFootageProbe";
import { useMissionLaunch } from "../hooks/useMissionLaunch";
import { Icon } from "../components/ui/Icon";
import type { DataVideoInfo } from "../lib/types";
import { formatBytes, formatClock } from "../lib/format";
import { uploadEstimate } from "../lib/videoInsights";

const VALID_EXTS = ["mp4", "mov", "avi", "mkv", "webm", "m4v", "ts", "mts"];

/** Held-out LiDAR reference formats — must match the backend's
 * `lidar.LIDAR_EXTENSIONS`. `.npz` is the pre-baked form: returns, a height
grid, or both. */
const LIDAR_EXTENSIONS = [".las", ".laz", ".npz"];
const LIDAR_ACCEPT = LIDAR_EXTENSIONS.join(",");

type Source = "upload" | "data";

/**
 * Footage selection and the mission form. This page renders and owns the form
 * values; the two things it does with them belong to hooks:
 *
 * - `useFootageProbe` — what the browser measured about the picked file
 *   (preview URL, decoder stats, overlap/sharpness probe, readiness score).
 * - `useMissionLaunch` — the launch workflow and its async state (upload →
 *   server metadata → mission record → pipeline), including the single-launch
 *   latch that a render-time `disabled` attribute cannot provide.
 */
export default function NewMission() {
  const navigate = useNavigate();
  // Selecting the new job immediately pins the shared run context to THIS
  // upload — the processing screen and viewer can then never display a
  // previously selected run for the user's mission.
  const { selectRun } = useRuns();

  const [source, setSource] = useState<Source>("upload");
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  const [missionName, setMissionName] = useState("");
  const [location, setLocation] = useState("Site Sector 1");
  const [missionType, setMissionType] = useState("Nadir Survey");
  const [description, setDescription] = useState("");

  const [dataVideos, setDataVideos] = useState<DataVideoInfo[]>([]);
  // Picking a server-side video only SELECTS it: the page's promise is
  // "review what STRATA can read, then start", and a row used to launch a
  // full reconstruction on the first click (no confirmation, no way to read
  // the row without starting a run). Launch lives on the start bar alone.
  const [dataVideoSelected, setDataVideoSelected] = useState<DataVideoInfo | null>(null);

  // Optional external telemetry (CSV or DJI SRT flight log). Absence is
  // normal: video-only runs.
  const [telemetryFile, setTelemetryFile] = useState<File | null>(null);
  const [telemetryError, setTelemetryError] = useState<string | null>(null);

  // Optional held-out LiDAR reference for the upload path — the data/ path
  // takes whatever tile sits beside the chosen video. It is never an input to
  // reconstruction: it is the independent ground truth the finished model is
  // measured against. `.npz` is the pre-baked form and may carry returns, a
  // height grid, or both.
  const [lidarFile, setLidarFile] = useState<File | null>(null);
  const [lidarError, setLidarError] = useState<string | null>(null);

  const probe = useFootageProbe();

  // Where a launched run goes: the page owns routing, the launch hook owns the
  // workflow and hands the new run id back through this stable callback.
  const toProcessing = useCallback(
    (jobId: string, name: string) => {
      if (jobId) selectRun(jobId);
      navigate(`/processing/${encodeURIComponent(jobId)}`, { state: { missionName: name } });
    },
    [navigate, selectRun],
  );
  const launch = useMissionLaunch(toProcessing);

  const { clearError, reset } = launch;
  const { clear: clearProbe, probe: runProbe } = probe;

  useEffect(() => {
    let cancelled = false;
    api
      .listDataVideos()
      .then((res) => {
        if (!cancelled) setDataVideos(res.videos ?? []);
      })
      .catch(() => {
        /* picker is optional — upload still works */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const pickFile = (file: File) => {
    setFileError(null);
    clearError();
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!ext || (!VALID_EXTS.includes(ext) && !file.type.startsWith("video/"))) {
      setFileError("Unsupported file type. Please upload a drone video file (MP4, MOV, AVI, MKV, WEBM).");
      setSelectedFile(null);
      clearProbe();
      return;
    }
    setSelectedFile(file);
    // Local preview + measured stats from the browser's own decoder — no
    // upload required. A clip the browser cannot decode still uploads fine
    // (the server's ffmpeg fallback handles what the browser cannot).
    runProbe(file);
    // Server-measured metadata belongs to the file that produced it.
    reset();
    const stem = file.name.substring(0, file.name.lastIndexOf(".")) || file.name;
    const readable = stem.replace(/[-_]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    setMissionName(`${readable} Mission`);
  };

  const clearFile = () => {
    setSelectedFile(null);
    clearProbe();
    reset();
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  }, []);
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  }, []);
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (e.dataTransfer.files?.[0]) pickFile(e.dataTransfer.files[0]);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) pickFile(e.target.files[0]);
  };

  const handleTelemetryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    const isSrt = !!file && file.name.toLowerCase().endsWith(".srt");
    setTelemetryError(null);
    if (file && !file.name.toLowerCase().endsWith(".csv") && !isSrt) {
      setTelemetryFile(null);
      setTelemetryError("Telemetry must be a .csv file (timestamp, latitude, longitude, altitude) or a DJI .srt flight log.");
      return;
    }
    setTelemetryFile(file);
  };

  const handleLidarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    setLidarError(null);
    const name = (file?.name ?? "").toLowerCase();
    if (file && !LIDAR_EXTENSIONS.some((ext) => name.endsWith(ext))) {
      setLidarFile(null);
      setLidarError(`A LiDAR reference must be a ${LIDAR_EXTENSIONS.join(", ")} file.`);
      return;
    }
    setLidarFile(file);
  };

  const clearLidar = () => {
    setLidarFile(null);
    setLidarError(null);
  };
  const clearTelemetry = () => {
    setTelemetryFile(null);
    setTelemetryError(null);
  };

  const {
    previewUrl,
    stats,
    previewError,
    motion,
    motionProgress,
    insights,
    blurInsight,
    readiness,
  } = probe;
  const { metadata, uploadProgress, startedVideoName, error: launchError } = launch;

  const showUpload = source === "upload";
  const footageChosen = showUpload ? !!selectedFile : !!dataVideoSelected;
  // GPS IS REQUIRED. A run without a GPS/telemetry source reconstructs in an
  // arbitrary frame: it cannot be georeferenced, so no measurement taken from
  // it is metric and no accuracy can be validated. The two ways to satisfy it
  // are the two the pipeline actually consumes: an attached log (CSV or DJI
  // SRT), or — for a server-side video — a log sitting beside it, which the
  // picker reports from the backend's own discovery rule. Nothing here guesses
  // at a video's embedded GPS tags: they are unknown until the server reads
  // the file, so they cannot unlock the start button.
  const datasetGps = dataVideoSelected?.gps_sources ?? [];
  const gpsReady = !!telemetryFile || (!showUpload && datasetGps.length > 0);

  // LiDAR is never required — it only decides whether absolute accuracy is
  // measurable at all, so its absence is stated rather than gated on.
  const datasetLidar = dataVideoSelected?.lidar_sources ?? [];
  const lidarReady = showUpload ? !!lidarFile : datasetLidar.length > 0;
  const lidarHint = lidarReady
    ? "Held out from reconstruction — compared against the finished model to measure absolute accuracy."
    : "Optional. Without a LiDAR reference the run still reconstructs, but absolute accuracy stays NOT MEASURED: there is nothing independent to measure it against.";
  const gpsReason = telemetryFile
    ? `Attached: ${telemetryFile.name}`
    : !showUpload && datasetGps.length > 0
      ? `Found on the server beside this video: ${datasetGps.join(", ")}`
      : "No GPS source — attach a telemetry log to start";

  // The start bar owns the launch for both sources, so neither one can fire a
  // run from a stray click on a row, and neither can fire one without GPS.
  const startDisabled =
    launch.busy || (showUpload ? !selectedFile : !dataVideoSelected) || !gpsReady;
  const startLabel = launch.busy
    ? "STARTING…"
    : showUpload && readiness && selectedFile && readiness.verdict === "unlikely"
      ? "START ANYWAY — LIKELY TO FAIL"
      : showUpload && readiness && selectedFile && readiness.verdict === "risky"
        ? "START DESPITE RISK"
        : "START RECONSTRUCTION";
  const handleStart = () => {
    void launch.launch({
      file: showUpload ? selectedFile : null,
      dataVideo: showUpload ? null : dataVideoSelected,
      telemetryFile,
      lidarFile,
      missionName,
      description,
    });
  };

  return (
    <div className="nm-wrap">
      <button className="nm-back" onClick={() => navigate(-1)}>
        ← Back
      </button>
      <h1 className="nm-title">New mission</h1>
      <p className="nm-sub">Choose footage, review what STRATA can read from it, then start the pipeline.</p>

      {/* Source selection */}
      <div className="nm-sources">
        <button type="button" className={`nm-source ${showUpload ? "selected" : ""}`} onClick={() => setSource("upload")}>
          <div className="nm-source-icon"><Icon name="video" size={22} /></div>
          <div className="nm-source-name">Upload a video</div>
          <div className="nm-source-desc">Drag in or browse local drone footage (up to 10 GB).</div>
        </button>
        <button type="button" className={`nm-source ${source === "data" ? "selected" : ""}`} onClick={() => setSource("data")}>
          <div className="nm-source-icon"><Icon name="folder" size={22} /></div>
          <div className="nm-source-name">From data/ folder</div>
          <div className="nm-source-desc">Select a video already on the server, then start it.</div>
        </button>
      </div>

      {source === "data" && (
        <div style={{ marginBottom: 24 }}>
          {dataVideos.length === 0 ? (
            <div className="empty-state">No videos found in data/. Add one on the server and refresh.</div>
          ) : (
            dataVideos.map((video) => {
              const picked = dataVideoSelected?.name === video.name;
              return (
                <button
                  key={video.name}
                  type="button"
                  className={`nm-video-card${picked ? " selected" : ""}`}
                  style={{ width: "100%", cursor: "pointer", textAlign: "left", marginBottom: 10 }}
                  disabled={launch.busy}
                  aria-pressed={picked}
                  onClick={() => {
                    clearError();
                    setDataVideoSelected(video);
                  }}
                >
                  <div>
                    <div className="nm-source-name"><Icon name="play" size={12} /> {video.name}</div>
                    <div className="nm-source-desc">
                      {formatClock(video.duration_sec ?? 0)} · {formatBytes(video.size_bytes)} ·{" "}
                      GPS: {video.gps_sources?.length ? video.gps_sources.join(", ") : "none found"}
                      {video.lidar_sources?.length ? ` · LiDAR: ${video.lidar_sources.join(", ")}` : ""}
                    </div>
                  </div>
                  <span className={`badge ${picked ? "badge-completed" : "badge-info"}`}>
                    {startedVideoName === video.name ? "STARTING…" : picked ? "SELECTED" : "SELECT"}
                  </span>
                </button>
              );
            })
          )}
          {dataVideoSelected && (
            <p className="nm-source-desc">                      Selected {dataVideoSelected.name} — start the pipeline below. GPS: {gpsReason}
            </p>
          )}
        </div>
      )}

      {showUpload && !selectedFile && (
        <div
          className={`nm-drop ${dragActive ? "drag" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v,.MP4,.MOV,.AVI,.MKV"
            onChange={handleFileChange}
            aria-label="Choose a drone video file"
          />
          <div className="nm-drop-icon"><Icon name="video" size={34} /></div>
          <div className="nm-drop-title">Drop your drone footage here</div>
          <div className="nm-drop-hint">MP4, MOV, AVI, MKV, WEBM supported (up to 10 GB)</div>
        </div>
      )}

      {showUpload && selectedFile && (
        <div className="nm-video-card">
          {previewUrl && (
            <video
              className="nm-preview"
              src={previewUrl}
              controls
              muted
              playsInline
              aria-label="Selected footage preview"
            />
          )}
          <div>
            <div className="nm-file-row">
              <div className="nm-source-name"><Icon name="film" size={14} /> {selectedFile.name}</div>
              <button className="btn-ghost btn" onClick={clearFile}>
                Change file
              </button>
            </div>
            {stats && (
              <div className="nm-insights">
                {insights.map((ins) => (
                  <div key={ins.label} className={`nm-insight nm-insight-${ins.level}`}>
                    <span className={`nm-insight-dot dot-${ins.level}`} />
                    <span className="nm-insight-label">{ins.label}</span>
                    <span className="nm-insight-detail">{ins.detail}</span>
                  </div>
                ))}
                <div className="nm-insight">
                  <span className="nm-insight-dot dot-neutral" />
                  <span className="nm-insight-label">Upload</span>
                  <span className="nm-insight-detail">
                    {formatBytes(selectedFile.size)} — {uploadEstimate(selectedFile.size)} on a typical connection.
                  </span>
                </div>
                {blurInsight && (
                  <div className={`nm-insight nm-insight-${blurInsight.level}`}>
                    <span className={`nm-insight-dot dot-${blurInsight.level}`} />
                    <span className="nm-insight-label">{blurInsight.label}</span>
                    <span className="nm-insight-detail">{blurInsight.detail}</span>
                  </div>
                )}
              </div>
            )}
            {readiness && (
              <div className={`nm-readiness nm-readiness-${readiness.verdict}`}>
                <div className="nm-readiness-head">
                  <span className="nm-readiness-score">{readiness.score}</span>
                  <div>
                    <div className="nm-readiness-verdict">
                      {readiness.verdict === "ready"
                        ? "Ready to reconstruct"
                        : readiness.verdict === "risky"
                          ? "Risky — may produce a broken model"
                          : "Unlikely to reconstruct"}
                    </div>
                    <div className="nm-readiness-sub">
                      {motion
                        ? motion.pairsUnusable
                          ? `Measured from ${motion.pairs}/${motion.pairsAttempted ?? motion.pairs} frame pairs (${motion.pairsUnusable} undecodable) — duration, detail and view overlap`
                          : `Measured from ${motion.pairs} frame pairs — duration, detail and view overlap`
                        : motionProgress
                          ? `Measuring view overlap… ${motionProgress.done}/${motionProgress.total}`
                          : "Scored from duration and detail; overlap not measured"}
                    </div>
                  </div>
                </div>
                <div className="nm-readiness-bars">
                  {readiness.components.map((c) => (
                    <div key={c.key} className="nm-readiness-comp" title={c.detail}>
                      <span className="nm-readiness-comp-label">{c.label}</span>
                      <span className="nm-readiness-bar">
                        <span
                          className={`nm-readiness-fill fill-${c.level}`}
                          style={{ width: `${Math.round((c.points / c.max) * 100)}%` }}
                        />
                      </span>
                      <span className="nm-readiness-comp-pts">
                        {c.level === "unmeasured" ? "—" : `${Math.round(c.points)}/${c.max}`}
                      </span>
                    </div>
                  ))}
                </div>
                {readiness.verdict === "unlikely" && (
                  <div className="nm-readiness-warn">
                    This footage matches the measured signature of runs that failed to reconstruct. You can still start, but expect a failed or broken result.
                  </div>
                )}
              </div>
            )}
            {previewError && (
              <div className="nm-source-desc" style={{ marginTop: 6 }}>
                <Icon name="warning" size={12} /> {previewError}
              </div>
            )}
            {metadata && (
              <div className="nm-source-desc" style={{ marginTop: 6 }}>
                Server-measured: {metadata.codec || "codec n/a"} · {metadata.fps.toFixed(1)} fps ·
                {" "}{Math.round(metadata.bitrate_kbps / 1000)} Mbps · {metadata.frame_count} frames
                {metadata.gps_lat != null && metadata.gps_lon != null
                  ? ` · GPS ${metadata.gps_lat.toFixed(4)}, ${metadata.gps_lon.toFixed(4)}`
                  : " · no embedded GPS"}
              </div>
            )}
            <div className="nm-video-meta">
            <div className="nm-meta-cell">
              <span className="nm-meta-label">Resolution</span>
              {metadata
                ? `${metadata.width} × ${metadata.height}`
                : stats
                  ? `${stats.width} × ${stats.height}`
                  : "—"}
            </div>
            <div className="nm-meta-cell">
              <span className="nm-meta-label">Framerate</span>
              {metadata ? `${metadata.fps} FPS` : "—"}
            </div>
            <div className="nm-meta-cell">
              <span className="nm-meta-label">Duration</span>
              {metadata
                ? formatClock(metadata.duration_sec)
                : stats
                  ? formatClock(stats.duration_sec)
                  : "—"}
            </div>
              <div className="nm-meta-cell">
                <span className="nm-meta-label">GPS Telemetry</span>
                {metadata?.gps_lat != null || telemetryFile ? (
                  <span className="sys-value ok">Available</span>
                ) : (
                  <span style={{ color: "var(--amber-700, #b45309)" }}>— Required</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* GPS is one requirement for both footage sources, so it has one
          control: the log the pipeline actually consumes. */}
      {footageChosen && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-title" style={{ marginBottom: 10 }}>
            GPS / telemetry log <span style={{ color: "#b45309" }}>— required</span>
          </div>
          <div className="nm-meta-cell">
            {telemetryFile ? (
              <span className="flex items-center justify-between" style={{ gap: 8 }}>
                <span><Icon name="check" size={12} /> {telemetryFile.name}</span>
                <button type="button" className="btn-ghost btn" onClick={clearTelemetry}>
                  Remove
                </button>
              </span>
            ) : (
              <label className="btn-ghost btn" style={{ cursor: "pointer" }}>
                Attach telemetry log (CSV or DJI SRT)…
                <input
                  type="file"
                  accept=".csv,.srt,text/csv"
                  onChange={handleTelemetryChange}
                  style={{ display: "none" }}
                />
              </label>
            )}
          </div>
          <p className="nm-source-desc" style={{ marginTop: 8 }}>
            {gpsReady
              ? gpsReason
              : "Without GPS a run reconstructs in an arbitrary frame: no length, area or height taken from it is metric and its accuracy cannot be validated, so STRATA will not start it. DJI drones export the flight log as an .SRT beside the video; a CSV needs timestamp, latitude, longitude, altitude."}
          </p>
          {telemetryError && <div className="nm-error"><Icon name="warning" size={14} /> {telemetryError}</div>}
        </div>
      )}

      {/* Optional for both footage sources, and the only place the two paths
          differ: a server video uses the tile the backend found beside it, an
          upload supplies one. */}
      {footageChosen && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-title" style={{ marginBottom: 10 }}>
            LiDAR reference <span style={{ color: "var(--gray-400)" }}>— optional</span>
          </div>
          <div className="nm-meta-cell">
            {!showUpload ? (
              datasetLidar.length > 0 ? (
                <span><Icon name="check" size={12} /> {datasetLidar.join(", ")} on the server</span>
              ) : (
                <span style={{ color: "var(--gray-400)" }}>None found beside this video</span>
              )
            ) : lidarFile ? (
              <span className="flex items-center justify-between" style={{ gap: 8 }}>
                <span><Icon name="check" size={12} /> {lidarFile.name}</span>
                <button type="button" className="btn-ghost btn" onClick={clearLidar}>
                  Remove
                </button>
              </span>
            ) : (
              <label className="btn-ghost btn" style={{ cursor: "pointer" }}>
                Attach LiDAR reference (LAS, LAZ or NPZ)…
                <input
                  type="file"
                  accept={LIDAR_ACCEPT}
                  onChange={handleLidarChange}
                  style={{ display: "none" }}
                />
              </label>
            )}
          </div>
          <p className="nm-source-desc" style={{ marginTop: 8 }}>{lidarHint}</p>
          {lidarError && <div className="nm-error"><Icon name="warning" size={14} /> {lidarError}</div>}
        </div>
      )}

      {fileError && <div className="nm-error"><Icon name="warning" size={14} /> {fileError}</div>}

      {/* Mission details */}
      {showUpload && (
        <>
          <hr className="strata-divider" />
          <h3 className="section-title">Mission information</h3>
          <div className="nm-form-grid">
            <div className="nm-field">
              <label htmlFor="nm-name">Mission name</label>
              <input
                id="nm-name"
                type="text"
                value={missionName}
                onChange={(e) => setMissionName(e.target.value)}
                placeholder="e.g. Site Alpha Survey"
              />
            </div>
            <div className="nm-field">
              <label htmlFor="nm-location">Location / sector</label>
              <input
                id="nm-location"
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="e.g. Sector 4, North Ridge"
              />
            </div>
            <div className="nm-field">
              <label htmlFor="nm-type">Mission type</label>
              <select id="nm-type" value={missionType} onChange={(e) => setMissionType(e.target.value)}>
                <option value="Nadir Survey">Nadir Survey (Terrain / Ground)</option>
                <option value="Oblique Mapping">Oblique Mapping (3D Building)</option>
                <option value="Infrastructure Inspection">Infrastructure Inspection</option>
                <option value="General Recon">General Aerial Recon</option>
              </select>
            </div>
            <div className="nm-field">
              <label htmlFor="nm-desc">Description</label>
              <input
                id="nm-desc"
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Optional mission notes…"
              />
            </div>
          </div>

          {launch.phase === "uploading" && (
            <div className="nm-upload-bar">
              <div className="sys-row">
                <span className="sys-label">
                  {uploadProgress >= 100 ? "Validating footage & extracting metadata…" : "Uploading drone footage…"}
                </span>
                <span className="sys-value ok">{uploadProgress}%</span>
              </div>
              <div className="nm-upload-track">
                <div className="nm-upload-fill" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          )}
        </>
      )}

      {launchError && <div className="nm-error"><Icon name="warning" size={14} /> {launchError}</div>}

      {/* Sticky start bar */}
      <div className="nm-start-bar">
        <button className="btn btn-secondary" onClick={() => navigate("/missions")} disabled={launch.busy}>
          Cancel
        </button>
        <button
          className={`nm-start-btn${
            showUpload && readiness && readiness.verdict !== "ready" && selectedFile ? " nm-start-warn" : ""
          }`}
          onClick={handleStart}
          disabled={startDisabled}
        >
          <span><Icon name="play" size={12} /></span>
          <span>{startLabel}</span>
        </button>
      </div>
    </div>
  );
}
