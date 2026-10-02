import { Icon } from "../ui/Icon";

/**
 * View of the backend sync report (`stages.georef.detail.sync` from
 * /api/pipeline/status, or the manifest's `telemetry` block). Only
 * `telemetry_mode` is guaranteed — `sync` is present only for
 * external-telemetry runs, so every field here is optional.
 */
export interface TelemetrySyncView {
  telemetry_mode: string;
  sync?: {
    telemetry_samples: number;
    matched_frames: number;
    unmatched_frames: number;
    timestamp_offset_sec: number;
    gps_available: boolean;
    telemetry_quality: string;
    invalid_samples_dropped: number;
    median_sample_spacing_sec?: number;
    sufficient_for_georeferencing?: boolean;
  } | null;
  note?: string | null;
}

/**
 * Nearest-neighbour matching becomes ambiguous when the clock offset is a
 * large fraction of the gap between telemetry samples: a frame can bind to
 * the wrong sample. The absolute offset alone is not the criterion — 0.9s
 * is harmless for 5s-spaced samples, dangerous for 1s-spaced ones — so the
 * threshold is relative to the median sample spacing. 50% is the point
 * where a frame is closer to a *neighbouring* sample's midpoint than to its
 * own; past it, the assignment is not trustworthy.
 */
export const LARGE_OFFSET_FRACTION = 0.5;
/** Absolute floor (seconds) so dense CSVs never warn over tiny offsets. */
export const LARGE_OFFSET_MIN_SEC = 0.5;

export function hasLargeClockOffset(
  offsetSec: number,
  medianSpacingSec: number | undefined,
): boolean {
  if (!Number.isFinite(offsetSec) || Math.abs(offsetSec) < LARGE_OFFSET_MIN_SEC) {
    return false;
  }
  if (medianSpacingSec == null || medianSpacingSec <= 0) {
    return true; // unknown density — do not stay silent about a big offset
  }
  return Math.abs(offsetSec) > medianSpacingSec * LARGE_OFFSET_FRACTION;
}

export function formatOffset(offsetSec: number): string {
  const sign = offsetSec > 0 ? "+" : offsetSec < 0 ? "−" : "";
  const abs = Math.abs(offsetSec);
  const body = abs >= 100 ? String(Math.round(abs)) : abs.toFixed(2).replace(/\.?0+$/, "");
  return `${sign}${body || "0"}s`;
}

/**
 * Normalize the manifest's `telemetry` block, which is either the flat sync
 * dict itself (external runs — carries `mode` plus the sync fields) or a
 * mode-only stub (`{mode: "VIDEO_ONLY"}` / `...EMBEDDED...`). Returns null
 * when absent so callers skip the block entirely.
 */
export function telemetryViewFromManifest(
  t: Record<string, unknown> | undefined | null,
): TelemetrySyncView | null {
  if (!t || typeof t !== "object") return null;
  const mode = (t.mode as string) ?? "VIDEO_ONLY";
  const isFlat = "matched_frames" in t || "telemetry_samples" in t;
  return {
    telemetry_mode: mode,
    sync: isFlat ? (t as TelemetrySyncView["sync"]) : null,
    note: (t.note as string | undefined) ?? null,
  };
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="tel-stat">
      <span className="tel-stat-label">{label}</span>
      <span className="tel-stat-value">{children}</span>
    </div>
  );
}

/**
 * Compact, honest telemetry summary. GPS Telemetry is the headline:
 * Available / Not available — never fabricated. The sync stats appear only
 * for external telemetry; the clock-offset warning fires when the median
 * offset is large relative to sample spacing (see hasLargeClockOffset).
 */
export default function TelemetryBlock({ data }: { data: TelemetrySyncView }) {
  const external = data.telemetry_mode === "VIDEO_WITH_EXTERNAL_TELEMETRY";
  const embedded = data.telemetry_mode === "VIDEO_WITH_EMBEDDED_TELEMETRY";
  const s = data.sync;

  const gpsAvailable = external ? (s?.gps_available ?? false) : embedded;
  const gpsLine = gpsAvailable ? (
    <span className="tel-ok">
      <Icon name="check" size={12} /> Available
    </span>
  ) : (
    <span className="tel-off">
      <Icon name="x" size={12} /> Not available
    </span>
  );

  return (
    <section className="tel-block" aria-label="GPS telemetry">
      <div className="tel-head">
        <Icon name="pin" size={14} />
        <span className="tel-title">GPS Telemetry</span>
        {gpsLine}
        <span className="tel-mode">
          {external ? "external CSV" : embedded ? "embedded" : "video-only"}
        </span>
      </div>

      {external && s && (
        <div className="tel-grid">
          <Stat label="Samples">{s.telemetry_samples.toLocaleString()}</Stat>
          <Stat label="Matched frames">
            {s.matched_frames.toLocaleString()}
            {s.invalid_samples_dropped > 0 && (
              <span className="tel-dropped"> · {s.invalid_samples_dropped} dropped</span>
            )}
          </Stat>
          <Stat label="Unmatched frames">{s.unmatched_frames.toLocaleString()}</Stat>
          <Stat label="Clock offset">
            {formatOffset(s.timestamp_offset_sec)}
            {hasLargeClockOffset(s.timestamp_offset_sec, s.median_sample_spacing_sec) && (
              <span className="tel-warn">
                <Icon name="warning" size={12} /> CSV clock likely doesn't match the video —
                frames may pair with wrong samples
              </span>
            )}
          </Stat>
          <Stat label="Quality">{s.telemetry_quality}</Stat>
        </div>
      )}

      {data.note && <p className="tel-note">{data.note}</p>}
    </section>
  );
}
