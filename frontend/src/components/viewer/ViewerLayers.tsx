interface Layer {
  name: string;
  visible: boolean;
  enabled: boolean;
  icon: string;
}

export default function ViewerLayers({
  layers,
  onToggle,
}: {
  layers: Layer[];
  onToggle: (index: number) => void;
}) {
  return (
    <div>
      <div className="section-title">Layers</div>
      {layers.map((l, i) => (
        <label
          key={l.name}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 0",
            fontSize: 13,
            cursor: l.enabled ? "pointer" : "not-allowed",
            opacity: l.enabled ? 1 : 0.55,
          }}
        >
          <input
            type="checkbox"
            checked={l.visible}
            disabled={!l.enabled}
            onChange={() => onToggle(i)}
          />
          <span>{l.icon}</span> {l.name}
        </label>
      ))}
    </div>
  );
}