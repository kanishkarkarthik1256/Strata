# Phase 11 Limitations

## Not yet implemented

1. **Tauri desktop wrapper** — requires Rust toolchain installation; the
   app currently runs as a web application only.
2. **Windows/macOS native builds** — not produced; the build system is
   configured but not executed.
3. **Backend process lifecycle** — the desktop app does not yet launch or
   manage the Python backend; users must start it manually.
4. **Frontend tests** — no Vitest or React Testing Library tests exist yet.
5. **PDF export** — the button exists but PDF generation is not wired.
6. **Dark mode** — light theme only; dark mode CSS not implemented.
7. **Per-point confidence heatmap** — the viewer shows a legend and
   threshold slider but the actual per-point confidence data is not yet
   computed or loaded into the viewer. This requires a confidence
   computation pass over the dense point cloud.
8. **Copilot structured actions** — the `CopilotAction` type is defined
   but the viewer does not yet react to action payloads (e.g.,
   `ENABLE_HEATMAP`, `FOCUS_REGION`).
9. **Keyboard shortcuts** — minimal; only Enter to send copilot messages.
10. **Code splitting** — Three.js creates a 700 KB bundle; lazy loading
    not implemented.

## Honesty items (must not be hidden)

- Metric 3D accuracy is NOT validated — shown as "Not validated" in UI
- Depth Anything V2 is relative depth — labeled as such
- Objects/Damage tabs show "data not available" for this validation run
- No original drone MP4/MOV validated
- OpenMVS unavailable
- GPU unavailable (Intel Mac, CPU-only)
- Metric scale is estimated from GPS-vs-COLMAP alignment, not surveyed ground truth
