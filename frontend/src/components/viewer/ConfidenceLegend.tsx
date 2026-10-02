/**
 * Reconstruction-confidence legend. Terminology stays "Reconstruction
 * Confidence" (never absolute accuracy). When the loaded cloud carries a
 * measured per-point confidence attribute the legend reflects its real
 * measured range; when it does not, the panel says so explicitly instead of
 * pretending the gradient applies to the cloud.
 */

interface Props {
  threshold: number;
  onThreshold: (t: number) => void;
  /** null = still loading; false = cloud has no confidence attribute. */
  available: boolean | null;
  /** Measured confidence band of the loaded cloud (min .. 1.0). */
  range: { min: number; max: number } | null;
}

export default function ConfidenceLegend({ threshold, onThreshold, available, range }: Props) {
  return (
    <div>
      <div className="section-title">Reconstruction Confidence</div>
      <div className="heatmap-legend" style={{ position: "static" }}>
        <div className="heatmap-gradient" />
        <div className="heatmap-labels">
          <span>High (&gt;90%)</span>
          <span>Medium</span>
          <span>Low (&lt;60%)</span>
        </div>
      </div>
      <div style={{ marginTop: 12, fontSize: 12, color: "var(--gray-500)" }}>
        <div>Threshold: {threshold}%</div>
        <input
          type="range"
          min={10}
          max={100}
          value={threshold}
          onChange={(e) => onThreshold(Number(e.target.value))}
          style={{ width: "100%" }}
          aria-label="Confidence threshold"
        />
      </div>
      {available === false ? (
        <div
          className="heatmap-unavailable"
          style={{
            marginTop: 10,
            padding: 10,
            background: "var(--gray-50)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            fontSize: 12,
            color: "var(--gray-600)",
          }}
        >
          Spatial confidence data is not available for this reconstruction. The
          gradient will apply once per-point confidence is produced by the
          pipeline.
        </div>
      ) : available && range ? (
        <div
          style={{
            marginTop: 10,
            padding: 10,
            background: "var(--gray-50)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            fontSize: 12,
            color: "var(--gray-600)",
          }}
        >
          Measured per-point confidence: {range.min.toFixed(2)}–{range.max.toFixed(2)}.
          Heatmap colors the cloud by observation count and fusion agreement;
          points below the threshold are dimmed.
        </div>
      ) : (
        <div style={{ marginTop: 10, fontSize: 12, color: "var(--gray-500)" }}>
          Checking confidence data…
        </div>
      )}
    </div>
  );
}
