/**
 * Flat stroke SVG icons — no emoji anywhere in the UI.
 * 16×16 viewBox, currentColor stroke, 1.5px — sized via font-size.
 */

export type IconName =
  | "dashboard"
  | "missions"
  | "cube"
  | "chart"
  | "report"
  | "gear"
  | "play"
  | "video"
  | "film"
  | "folder"
  | "check"
  | "x"
  | "clock"
  | "warning"
  | "info"
  | "refresh"
  | "cancel"
  | "arrow-up-right"
  | "camera"
  | "pin"
  | "map"
  | "sun"
  | "moon";

const PATHS: Record<IconName, React.ReactNode> = {
  // Theme control: the icon shows the mode you would switch TO, matching the
  // button's title so the affordance is never ambiguous.
  sun: (
    <>
      <circle cx="8" cy="8" r="3.2" />
      <path d="M8 1v1.8M8 13.2V15M1 8h1.8M13.2 8H15M3.1 3.1l1.3 1.3M11.6 11.6l1.3 1.3M12.9 3.1l-1.3 1.3M4.4 11.6l-1.3 1.3" />
    </>
  ),
  moon: <path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8Z" />,
  dashboard: (
    <>
      <rect x="2" y="2" width="5" height="5" rx="1" />
      <rect x="9" y="2" width="5" height="5" rx="1" />
      <rect x="2" y="9" width="5" height="5" rx="1" />
      <rect x="9" y="9" width="5" height="5" rx="1" />
    </>
  ),
  missions: (
    <>
      <path d="M2 4.5A1.5 1.5 0 0 1 3.5 3h3l1.5 2h4.5A1.5 1.5 0 0 1 14 6.5v5A1.5 1.5 0 0 1 12.5 13h-9A1.5 1.5 0 0 1 2 11.5v-7Z" />
    </>
  ),
  cube: (
    <>
      <path d="M8 1.8 14 5v6l-6 3.2L2 11V5l6-3.2Z" />
      <path d="M2 5l6 3.2L14 5M8 8.2V14" />
    </>
  ),
  chart: (
    <>
      <path d="M2.5 13.5v-5M8 13.5V2.5M13.5 13.5V7" />
    </>
  ),
  report: (
    <>
      <path d="M4 1.5h5.5L13 5v9.5H4v-13Z" />
      <path d="M9.5 1.5V5H13M6 8h4M6 10.5h4" />
    </>
  ),
  gear: (
    <>
      <circle cx="8" cy="8" r="2.2" />
      <path d="M8 1.8v2M8 12.2v2M1.8 8h2M12.2 8h2M3.6 3.6l1.4 1.4M11 11l1.4 1.4M12.4 3.6 11 5M5 11l-1.4 1.4" />
    </>
  ),
  play: <path d="M5 3.2v9.6L12.4 8 5 3.2Z" />,
  video: (
    <>
      <rect x="1.8" y="4" width="8.4" height="8" rx="1.5" />
      <path d="m10.2 8 4-2.6v5.2l-4-2.6Z" />
    </>
  ),
  film: (
    <>
      <rect x="2" y="2.5" width="12" height="11" rx="1.5" />
      <path d="M5 2.5v11M11 2.5v11M2 6h3M2 10h3M11 6h3M11 10h3" />
    </>
  ),
  folder: (
    <>
      <path d="M2 4.5A1.5 1.5 0 0 1 3.5 3h3l1.5 2h4.5A1.5 1.5 0 0 1 14 6.5v5A1.5 1.5 0 0 1 12.5 13h-9A1.5 1.5 0 0 1 2 11.5v-7Z" />
    </>
  ),
  check: <path d="m3 8.5 3.2 3L13 4.5" />,
  x: <path d="M4 4l8 8M12 4l-8 8" />,
  clock: (
    <>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 4.5V8l2.5 1.5" />
    </>
  ),
  warning: (
    <>
      <path d="M8 2 14.5 13.5h-13L8 2Z" />
      <path d="M8 6.5v3M8 11.5v.01" />
    </>
  ),
  info: (
    <>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 7.5v3.5M8 5v.01" />
    </>
  ),
  refresh: (
    <>
      <path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9" />
      <path d="M13.5 1.5v3h-3" />
    </>
  ),
  cancel: (
    <>
      <circle cx="8" cy="8" r="6" />
      <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" />
    </>
  ),
  "arrow-up-right": <path d="M4.5 11.5 11.5 4.5M6 4.5h5.5V10" />,
  camera: (
    <>
      <path d="M2 5.5A1.5 1.5 0 0 1 3.5 4h1.7l1-1.5h3.6l1 1.5h1.7A1.5 1.5 0 0 1 14 5.5v6a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 11.5v-6Z" />
      <circle cx="8" cy="8.3" r="2.2" />
    </>
  ),
  pin: (
    <>
      <path d="M8 14s4.5-4 4.5-7.5a4.5 4.5 0 1 0-9 0C3.5 10 8 14 8 14Z" />
      <circle cx="8" cy="6.5" r="1.6" />
    </>
  ),
  map: (
    <>
      <path d="m2 4 4-1.5L10 4l4-1.5v9.5L10 13.5 6 12l-4 1.5V4Z" />
      <path d="M6 2.5V12M10 4v9.5" />
    </>
  ),
};

export function Icon({
  name,
  size = 16,
  className,
}: {
  name: IconName;
  size?: number;
  className?: string;
}) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {PATHS[name]}
    </svg>
  );
}
