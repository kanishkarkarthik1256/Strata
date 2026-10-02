export type ViewMode = "pointcloud" | "mesh" | "heatmap";
import type { ViewCommand, ViewPreset } from "./ViewerCanvas";

const PRESETS: { preset: ViewPreset; label: string }[] = [
  { preset: "full", label: "Full Model" },
  { preset: "isometric", label: "Isometric" },
  { preset: "top", label: "Top" },
  { preset: "front", label: "Front" },
  { preset: "side", label: "Side" },
];

interface Props {
  mode: ViewMode;
  onChange: (m: ViewMode) => void;
  onView: (cmd: ViewCommand) => void;
}

export default function ViewerToolbar({ mode, onChange, onView }: Props) {
  return (
    <div className="viewer-toolbar">
      <button className={mode === "pointcloud" ? "active" : ""} onClick={() => onChange("pointcloud")}>
        Point Cloud
      </button>
      <button className={mode === "mesh" ? "active" : ""} onClick={() => onChange("mesh")}>
        Mesh
      </button>
      <button className={mode === "heatmap" ? "active" : ""} onClick={() => onChange("heatmap")}>
        Heatmap
      </button>
      <span className="viewer-toolbar-divider" aria-hidden="true" />
      {PRESETS.map(({ preset, label }) => (
        <button
          key={preset}
          title={`Frame the model — ${label}`}
          onClick={() => onView({ preset, nonce: performance.now() })}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
