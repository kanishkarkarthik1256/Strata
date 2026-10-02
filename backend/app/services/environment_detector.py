"""Environmental intelligence — per-frame adverse-condition detection.

Every detector below computes a *physical* image statistic with numpy/OpenCV
and labels its method provenance; no learned model is assumed. Conditions that
cannot be separated from a single still (rain vs sensor noise) are reported as
low-confidence proxies rather than fabrications: the report carries a
``confidence`` per condition and an explicit ``method`` string so consumers
know exactly what evidence each score rests on.

Frame metrics feed three aggregated outputs written to
``intel/environment_report.json``:

* per-frame condition table (CSV-ready) and scores,
* an ``EnvironmentalQualityScore`` (0-100) with grade + per-condition severity,
* ``adaptive_strategies`` — structured, parameter-level suggestions a caller
  can fold into a re-run of the reconstruction pipeline.

The heavy learned classifiers the spec permits (DINO/SAM2 for scene content)
live elsewhere; this stage is deliberately input-agnostic (works on raw
frames before any reconstruction) and can run as its own plugin stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.config.settings import settings
from app.logging_config import get_logger
from app.services.image_files import list_image_files
from app.services.pipeline_stage import PipelineStage, StageNotApplicable, register

log = get_logger("drone_recon.services.environment_detector")


# ---------------------------------------------------------------------------
# Per-condition detectors (single-image, classical)
# ---------------------------------------------------------------------------


def _luminance(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _dark_channel(img: np.ndarray, patch: int = 7) -> float:
    """He & Sun dark-channel prior: min over a local patch of the per-pixel
    minimum channel. High mean ⇒ haze/fog/smoke veiling the scene."""
    minch = np.min(img.astype(np.float32) / 255.0, axis=2)
    if patch > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
        minch = cv2.erode(minch, k)
    return float(np.mean(minch))


def _contrast(gray: np.ndarray) -> float:
    return float(np.std(gray) / max(np.mean(gray), 1e-6))


def _laplacian_focus(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard no-reference focus measure."""
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _saturation(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    return hsv[:, :, 1] / 255.0


def _gaussian_highpass(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    base = cv2.GaussianBlur(gray, (0, 0), sigma)
    return np.abs(gray - base)


def detect_fog_haze(img: np.ndarray) -> dict:
    """Fog/smog/smoke: a raised dark channel (path-scattered veil). The veil
    is the primary cue; contrast loss only *modulates* it once the veil is
    real, so blur or low light alone never reads as fog."""
    dc = _dark_channel(img)
    c = _contrast(_luminance(img))
    veil = float(np.clip((dc - 0.30) / 0.45, 0.0, 1.0))
    contrast_loss = float(np.clip((0.50 - c) / 0.50, 0.0, 1.0))
    score = float(veil * (0.65 + 0.35 * contrast_loss))
    return {
        "condition": "fog_haze", "score": round(score, 3),
        "confidence": round(0.7 + 0.3 * veil, 3),  # veil is the strongest cue
        "evidence": {"dark_channel": round(dc, 3), "contrast_loss": round(contrast_loss, 3)},
        "method": "dark_channel_prior",
    }


def detect_smoke_dust(img: np.ndarray) -> dict:
    """Smoke/dust: grey veil with *low* saturation (unlike white fog we allow
    a tan/grey cast) plus high dark channel. Uses colour statistics only."""
    dc = _dark_channel(img)
    sat = float(np.mean(_saturation(img)))
    greyness = float(np.clip((0.35 - sat) / 0.35, 0.0, 1.0))
    score = float(np.clip(0.6 * greyness + 0.4 * np.clip((dc - 0.3) / 0.5, 0.0, 1.0), 0, 1))
    return {
        "condition": "smoke_dust", "score": round(score, 3),
        "confidence": round(0.6 + 0.3 * score, 3),
        "evidence": {"mean_saturation": round(sat, 3), "dark_channel": round(dc, 3)},
        "method": "desaturation_grey_veil",
    }


def detect_lowlight(img: np.ndarray) -> dict:
    lum = _luminance(img)
    mean_lum = float(np.mean(lum))
    p5 = float(np.percentile(lum, 5))
    th = settings.intel.lowlight_luma_threshold
    score = float(np.clip(1.0 - mean_lum / th, 0.0, 1.0)) if th else 0.0
    return {
        "condition": "lowlight", "score": round(score, 3),
        "confidence": round(0.85 + 0.15 * score, 3),
        "evidence": {"mean_luminance": round(mean_lum, 1), "p5_luminance": round(p5, 1)},
        "method": "luminance_statistics",
    }


def detect_motion_blur(img: np.ndarray) -> dict:
    """Blur via Laplacian focus; directional elongation separates motion blur
    from defocus (motion smears energy along one axis)."""
    gray = _luminance(img)
    focus = _laplacian_focus(gray)
    min_focus = settings.intel.blur_min_focus
    blur = float(np.clip(1.0 - focus / min_focus, 0.0, 1.0))
    hp = _gaussian_highpass(gray, sigma=1.5)
    gy, gx = np.gradient(hp)
    energy = gx**2 + gy**2 + 1e-9
    # Orientation anisotropy: var of direction-weighted energy along the axis.
    theta = np.arctan2(gy, gx)
    dir_energy = np.histogram(theta, bins=18, range=(-np.pi, np.pi), weights=energy)[0]
    anisotropy = float(np.clip((dir_energy.max() - dir_energy.mean()) / (dir_energy.max() + 1e-6), 0, 1))
    score = float(np.clip(0.65 * blur + 0.35 * anisotropy * blur, 0.0, 1.0))
    return {
        "condition": "motion_blur", "score": round(score, 3),
        "confidence": round(0.7 + 0.3 * anisotropy, 3),
        "evidence": {"laplacian_variance": round(focus, 1), "anisotropy": round(anisotropy, 3)},
        "method": "laplacian_focus_directional_energy",
    }


def detect_sun_glare(img: np.ndarray) -> dict:
    """Glare: a compact, near-saturated bright region (bloom/lens flare)."""
    gray = _luminance(img)
    hot = gray > 235
    if not hot.any():
        return {"condition": "sun_glare", "score": 0.0, "confidence": 0.9,
                "evidence": {"hot_fraction": 0.0}, "method": "saturation_fraction"}
    frac = float(np.mean(hot))
    # Compactness: fraction of hot pixels inside their bounding box.
    ys, xs = np.nonzero(hot)
    box_area = max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    fill = float(hot.sum() / box_area)  # how solidly the bloom fills its bbox
    glare_frac = float(np.clip((frac - settings.intel.glare_high_pct) / 0.2, 0.0, 1.0))
    score = float(np.clip(glare_frac * (0.4 + 0.6 * min(fill, 1.0)), 0.0, 1.0))
    return {
        "condition": "sun_glare", "score": round(score, 3),
        "confidence": round(0.6 + 0.4 * min(fill, 1.0), 3),
        "evidence": {"hot_fraction": round(frac, 4), "fill_ratio": round(fill, 3)},
        "method": "saturation_fraction",
    }


def detect_strong_shadows(img: np.ndarray) -> dict:
    """Strong shadows: deep, well-separated dark regions (high dynamic range)."""
    lum = _luminance(img)
    dark = lum < np.percentile(lum, 25) - 15  # clear lower tail
    frac = float(np.mean(dark))
    dr = float(np.percentile(lum, 98) - np.percentile(lum, 2))
    score = float(np.clip((frac - settings.intel.shadow_low_pct) / 0.4, 0.0, 1.0)
                   * np.clip(dr / 200.0, 0.2, 1.0))
    return {
        "condition": "strong_shadows", "score": round(score, 3),
        "confidence": round(0.6 + 0.4 * np.clip(frac * 3, 0, 1), 3),
        "evidence": {"dark_fraction": round(frac, 4), "dynamic_range": round(dr, 1)},
        "method": "low_tail_dynamic_range",
    }


def detect_lens_artifacts(img: np.ndarray) -> dict:
    """Lens water droplets / dirt: localised patches that are *sharp on the
    rim and smooth inside* (defocus blobs) or persistent smudges — detected as
    small low-texture islands inside a textured scene."""
    gray = _luminance(img)
    h, w = gray.shape
    cell = 16
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    # Local RMS of the Laplacian in each cell.
    gy, gx = np.mgrid[0:h, 0:w]
    rows, cols = h // cell, w // cell
    rms = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            blk = lap[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            rms[i, j] = float(np.sqrt(np.mean(blk**2)))
    if rms.size < 4 or rms.max() < 1.0:
        return {"condition": "lens_artifacts", "score": 0.0, "confidence": 0.9,
                "evidence": {"droplet_patch_fraction": 0.0}, "method": "local_defocus_islands"}
    med = float(np.median(rms))
    # Isolated near-zero cells flanked by texture ⇒ defocus blobs (droplets).
    islands = (rms < med * 0.25) & (rms > 0)
    frac = float(np.mean(islands))
    score = float(np.clip(frac * 6.0, 0.0, 1.0))  # even a few % is significant
    return {
        "condition": "lens_artifacts", "score": round(score, 3),
        "confidence": round(float(np.clip(0.5 + frac * 4, 0, 0.95)), 3),
        "evidence": {"droplet_patch_fraction": round(frac, 5)},
        "method": "local_defocus_islands",
    }


def detect_cloud_cover(img: np.ndarray) -> dict:
    """Cloud cover: only meaningful when sky occupies the top band (oblique
    drone footage). Top-band whiteness + smoothness; zero evidence when the
    band is not bright/smooth (nadir scenes → not measurable, low confidence)."""
    gray = _luminance(img)
    h = gray.shape[0]
    band = gray[: max(2, int(h * 0.12)), :]
    mean_b = float(np.mean(band))
    smooth = float(np.std(cv2.GaussianBlur(band, (0, 0), 5)))
    if mean_b < 160 or smooth > 25:
        return {"condition": "cloud_cover", "score": 0.0, "confidence": 0.35,
                "evidence": {"sky_band_mean": round(mean_b, 1),
                             "note": "no bright sky band detected (nadir or overcast-indeterminate)"},
                "method": "top_band_whiteness"}
    score = float(np.clip((mean_b - 160) / 95.0 * np.clip(1 - smooth / 40.0, 0.3, 1.0), 0, 1))
    return {
        "condition": "cloud_cover", "score": round(score, 3),
        "confidence": round(0.7 + 0.3 * score, 3),
        "evidence": {"sky_band_mean": round(mean_b, 1), "sky_band_std": round(smooth, 1)},
        "method": "top_band_whiteness",
    }


def detect_rain_streaks(img: np.ndarray) -> dict:
    """Rain streaks: strong directional high-frequency energy. A still frame
    cannot separate streaks from sensor noise, so confidence stays low and the
    score is explicitly a proxy; consecutive-frame flicker would firm it up."""
    gray = _luminance(img)
    hp = _gaussian_highpass(gray, sigma=1.0)
    gy, gx = np.gradient(hp)
    theta = np.arctan2(gy, gx).ravel()
    mag = np.sqrt(gx**2 + gy**2).ravel()
    keep = mag > np.percentile(mag, 90)
    if keep.sum() < 20:
        return {"condition": "rain_streaks", "score": 0.0, "confidence": 0.5,
                "evidence": {"note": "insufficient high-frequency energy"}, "method": "streak_proxy"}
    hist = np.histogram(theta[keep], bins=36, range=(-np.pi, np.pi), weights=mag[keep])[0]
    peak = float(np.max(hist) / max(np.sum(hist), 1e-6))
    score = float(np.clip((peak - 0.12) / 0.2, 0.0, 1.0))
    return {
        "condition": "rain_streaks", "score": round(score, 3),
        "confidence": 0.4,  # single-frame proxy — deliberately conservative
        "evidence": {"directional_peak_share": round(peak, 3),
                     "note": "still-frame proxy; confirm with temporal flicker"},
        "method": "streak_proxy",
    }


_DETECTORS = {
    "fog_haze": detect_fog_haze,
    "smoke_dust": detect_smoke_dust,
    "lowlight": detect_lowlight,
    "motion_blur": detect_motion_blur,
    "sun_glare": detect_sun_glare,
    "strong_shadows": detect_strong_shadows,
    "lens_artifacts": detect_lens_artifacts,
    "cloud_cover": detect_cloud_cover,
    "rain_streaks": detect_rain_streaks,
}


# ---------------------------------------------------------------------------
# Aggregation + scoring
# ---------------------------------------------------------------------------


def analyze_frame(img: np.ndarray) -> dict:
    """Run every detector on one BGR frame; returns condition → result map."""
    return {name: fn(img) for name, fn in _DETECTORS.items()}


_CONDITION_WEIGHTS = {
    "fog_haze": 1.4, "smoke_dust": 1.2, "lowlight": 1.0, "motion_blur": 1.5,
    "sun_glare": 1.0, "strong_shadows": 0.8, "lens_artifacts": 1.3,
    "cloud_cover": 0.7, "rain_streaks": 1.2,
}


def score_environment(frame_results: list[dict]) -> dict:
    """Aggregate per-frame detector results into quality score + severity.

    Quality is driven down by whichever condition is *dominant* per frame
    (the 90th percentile of the per-condition maximum, weighted) — a mission
    degraded by one bad condition is degraded even if others are clean.
    """
    if not frame_results:
        return {"quality_score": 100.0, "grade": "Excellent", "conditions": {}}
    cond_series: dict[str, list[float]] = {c: [] for c in _DETECTORS}
    for res in frame_results:
        for cond, det in res.items():
            if not isinstance(det, dict) or "score" not in det:
                continue  # metadata keys (frame_id, …) are not conditions
            cond_series.setdefault(cond, []).append(det["score"])
    summary = {}
    for cond, scores in cond_series.items():
        if not scores:
            continue
        arr = np.asarray(scores)
        summary[cond] = {
            "mean_score": round(float(arr.mean()), 3),
            "p90_score": round(float(np.percentile(arr, 90)), 3),
            "max_score": round(float(arr.max()), 3),
            "affected_frames_pct": round(float((arr > 0.35).mean()) * 100.0, 1),
            "weight": _CONDITION_WEIGHTS.get(cond, 1.0),
        }
    # Dominant degradation across frames, weighted.
    degradation = 0.0
    for cond, s in summary.items():
        peak = max(s["p90_score"], s["max_score"] * 0.8)
        degradation = max(degradation, peak * s["weight"])
    raw = 100.0 * (1.0 - degradation / 2.2)
    q = float(np.clip(raw, 0.0, 100.0))
    grade = ("Excellent" if q >= 85 else "Good" if q >= 70 else "Fair"
             if q >= 50 else "Poor" if q >= 30 else "Unusable")
    return {
        "quality_score": round(q, 1),
        "grade": grade,
        "conditions": summary,
        "dominant_condition": max(summary, key=lambda c: summary[c]["p90_score"] * summary[c]["weight"])
        if summary else "clear",
    }


def visibility_estimate(frame_results: list[dict]) -> dict:
    """Visibility proxy (m): a contrast/extinction heuristic derived from the
    fog veil — high veil ⇒ short visible range. Calibrated against
    ``visibility_max_m`` (no fog, clear air). Honest: it is a monotone proxy,
    not a LIDAR measurement."""
    if not frame_results:
        return {"estimated_visibility_m": settings.intel.visibility_max_m, "method": "clear_air_nominal"}
    veils = [r.get("fog_haze", {}).get("score", 0.0) for r in frame_results]
    veil = float(np.mean(veils))
    vmax = settings.intel.visibility_max_m
    vis = float(vmax * np.exp(-6.0 * veil))
    return {
        "estimated_visibility_m": round(vis, 1),
        "veil_proxy": round(veil, 3),
        "method": "contrast_extinction_proxy",
    }


def adaptive_strategies(environment: dict) -> list[dict]:
    """Map detected conditions to concrete, parameter-level reconstruction
    adjustments (module 2 of the Phase 8 spec). Each entry names the pipeline
    parameter it touches so a caller can apply it without guesswork."""
    cond = environment.get("conditions", {})
    strategies: list[dict] = []
    if cond.get("fog_haze", {}).get("mean_score", 0) > 0.3:
        strategies.append({"condition": "fog_haze", "action": "enable_dehaze", "target": "processing",
                           "params": {"CLAHE_clip": 3.0, "gamma": 1.15},
                           "rationale": "recover local contrast lost to the atmospheric veil"})
    if cond.get("rain_streaks", {}).get("mean_score", 0) > 0.3 or cond.get("motion_blur", {}).get("mean_score", 0) > 0.3:
        strategies.append({"condition": "rain_streaks_or_motion_blur", "action": "robustify_matching",
                           "target": "pipeline", "params": {"max_depth_views": 40, "frame_stride": 2},
                           "rationale": "fewer, better-conditioned views reduce streak/blur matching noise"})
    if cond.get("lowlight", {}).get("mean_score", 0) > 0.4:
        strategies.append({"condition": "lowlight", "action": "raise_confidence_floor",
                           "target": "dense", "params": {"min_confidence": 0.12, "normal_k": 30},
                           "rationale": "noisy low-light depth needs a higher acceptance floor and larger normals"})
    if cond.get("sun_glare", {}).get("mean_score", 0) > 0.4:
        strategies.append({"condition": "sun_glare", "action": "mask_glare_frames",
                           "target": "selection", "params": {"quality_threshold": 0.65},
                           "rationale": "drop frames where bloom overwhelms the sensor"})
    if cond.get("lens_artifacts", {}).get("mean_score", 0) > 0.3:
        strategies.append({"condition": "lens_artifacts", "action": "clean_optics",
                           "target": "hardware", "params": {},
                           "rationale": "droplets/dirt are a physical artifact — clean before re-fly"})
    return strategies


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def load_frames(workspace: Path, limit: int = 24) -> list[tuple[str, np.ndarray]]:
    """Load up to *limit* registered/selected frames as BGR arrays."""
    import itertools

    img_dir = None
    for cand in ("selected", "frames"):
        d = workspace / cand
        if d.is_dir():
            img_dir = d
            break
    if img_dir is None:
        return []
    paths = sorted(
        itertools.chain(list_image_files(img_dir), img_dir.glob("*.png")),
        key=lambda p: p.name,
    )
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        out.append((p.stem, img))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@register
class EnvironmentalIntelligenceStage(PipelineStage):
    name = "environmental_intelligence"
    description = "Adverse-condition detection, environmental quality score and adaptive strategies"
    artifact_rel = "intel/environment_report.json"
    version = "1.0.0"

    def validate_inputs(self) -> None:
        frames = load_frames(self.workspace, limit=1)
        if not frames:
            raise StageNotApplicable("no selected frames — run the frames stage first")

    def execute(self) -> None:
        frames = load_frames(self.workspace, limit=settings.intel.env_sample_frames)
        frame_results = []
        for fid, img in frames:
            res = analyze_frame(img)
            res["frame_id"] = fid
            frame_results.append(res)
        env = score_environment(frame_results)
        vis = visibility_estimate(frame_results)
        env["visibility"] = vis
        env["adaptive_strategies"] = adaptive_strategies(env)
        env["frame_count"] = len(frames)
        env["sample"] = frame_results[:8]  # bounded payload; full table on disk
        intel_dir = self.workspace / "intel"
        intel_dir.mkdir(parents=True, exist_ok=True)
        (intel_dir / "environment_report.json").write_text(json.dumps(env, indent=2))
        with open(intel_dir / "environment_frames.csv", "w") as f:
            f.write("frame_id,condition,score,confidence\n")
            for res in frame_results:
                for cond, det in res.items():
                    if isinstance(det, dict) and "score" in det:
                        f.write(f"{res['frame_id']},{cond},{det['score']},{det.get('confidence', '')}\n")
        self._count = len(frames)
        self._detail = {"quality_score": env["quality_score"], "grade": env["grade"],
                        "dominant_condition": env["dominant_condition"]}
        self._outputs = [
            {"kind": "data", "name": "environment_report", "path": str(intel_dir / "environment_report.json")},
            {"kind": "data", "name": "environment_frames", "path": str(intel_dir / "environment_frames.csv")},
        ]
        self.progress(1.0, {"quality_score": env["quality_score"]})
