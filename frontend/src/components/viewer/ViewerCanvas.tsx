import { useEffect, useRef } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import type { PosesJson, ViewerAlignment } from "../../lib/types";
import { withSmoothPositions } from "./meshSmooth";

/**
 * Corner budget for in-tab normal smoothing (measured at ~0.9 s for 3.3M
 * corners on the 1.1M-face furnerhem_6 GLB). Larger geometry keeps its
 * authored normals rather than stalling the tab.
 */
const SMOOTH_NORMAL_MAX_CORNERS = 5_000_000;

/** Quantization grid per axis for the `withSmoothNormals` position key. */
const SMOOTH_NORMAL_GRID = 131071; // 2^17 - 1

/**
 * Smooth normals that leave the UV atlas alone.
 *
 * A photogrammetric GLB stores one vertex per triangle corner (measured: 3.3M
 * corners for 1.1M faces), so `computeVertexNormals()` can only ever produce
 * FLAT shading — no two corners are shared, so nothing is ever averaged. That
 * faceted look is a large part of why the same file reads better in a viewer
 * app than in here.
 *
 * Welding vertices is not the answer: corners that share a position almost
 * always differ in UV (atlas seams), so a full weld merges nothing at all
 * (measured: 0 of 600k corners) and a position-only weld would destroy the
 * texture mapping. Instead, each triangle's normal is accumulated at its
 * corners' shared POSITION and the average is written back to every corner:
 * the surface shades continuously while every UV stays exactly as authored.
 * Accumulation is unnormalized, so larger triangles contribute more — the
 * same area weighting `computeVertexNormals` uses.
 */
export function withSmoothNormals(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  const pos = geometry.getAttribute("position") as THREE.BufferAttribute | undefined;
  if (!pos) return geometry;
  const corners = pos.count;
  const index = geometry.index ? (geometry.index.array as ArrayLike<number>) : null;
  const faces = Math.floor((index ? index.length : corners) / 3);
  if (corners > SMOOTH_NORMAL_MAX_CORNERS || faces < 1) {
    geometry.computeVertexNormals();
    return geometry;
  }
  const p = pos.array as ArrayLike<number>;

  // Quantize positions onto a per-axis grid and pack the three cells into ONE
  // number. The grid is the scene's span / 131071, which measured 6.2 mm on
  // an 813 m scene — far below the ~1 m sampling spacing, so no distinct
  // surfaces are merged — and the pack `qx·2^34 + qy·2^17 + qz` stays under
  // 2^51, so it is exact in a float64. Numeric keys hash 2.5x faster than the
  // string form they replaced (measured 728 ms vs 1849 ms over 3.3M corners)
  // and resolve to the same 898,079 shared positions on that mesh.
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < corners; i++) {
    const x = p[i * 3];
    const y = p[i * 3 + 1];
    const z = p[i * 3 + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const step = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6) / SMOOTH_NORMAL_GRID;
  const cell = (v: number, min: number) =>
    Math.min(SMOOTH_NORMAL_GRID, Math.max(0, Math.round((v - min) / step)));

  const groupOfCorner = new Int32Array(corners);
  const groups = new Map<number, number>();
  for (let c = 0; c < corners; c++) {
    const vi = index ? index[c] : c;
    const key =
      (cell(p[vi * 3], minX) * 131072 + cell(p[vi * 3 + 1], minY)) * 131072 +
      cell(p[vi * 3 + 2], minZ);
    let g = groups.get(key);
    if (g === undefined) {
      g = groups.size;
      groups.set(key, g);
    }
    groupOfCorner[c] = g;
  }
  const groupCount = Math.max(1, groups.size);
  groups.clear(); // release the key strings before the accumulation pass

  const acc = new Float32Array(groupCount * 3);
  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const nrm = new THREE.Vector3();
  for (let f = 0; f < faces; f++) {
    const i0 = index ? index[f * 3] : f * 3;
    const i1 = index ? index[f * 3 + 1] : f * 3 + 1;
    const i2 = index ? index[f * 3 + 2] : f * 3 + 2;
    vA.fromArray(p, i0 * 3);
    vB.fromArray(p, i1 * 3);
    vC.fromArray(p, i2 * 3);
    nrm.crossVectors(e1.subVectors(vB, vA), e2.subVectors(vC, vA));
    for (let k = 0; k < 3; k++) {
      const g = groupOfCorner[f * 3 + k];
      acc[g * 3] += nrm.x;
      acc[g * 3 + 1] += nrm.y;
      acc[g * 3 + 2] += nrm.z;
    }
  }

  const normals = new Float32Array(corners * 3);
  for (let c = 0; c < corners; c++) {
    const g = groupOfCorner[c];
    const x = acc[g * 3];
    const y = acc[g * 3 + 1];
    const z = acc[g * 3 + 2];
    const len = Math.hypot(x, y, z) || 1;
    normals[c * 3] = x / len;
    normals[c * 3 + 1] = y / len;
    normals[c * 3 + 2] = z / len;
  }
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  return geometry;
}

export type ViewPreset = "full" | "top" | "front" | "side" | "isometric";

export interface ViewCommand {
  preset: ViewPreset;
  nonce: number;
}

/**
 * Point-sprite sizing: with a FIXED world size, zooming in opens gaps
 * between sprites (the cloud 'loses detail' — measured on airport1 where a
 * 0.35 m sprite on a 1.07 m-spaced cloud covers ~10% of the surface up
 * close). Shader-derived screen-space sizing keeps sprites covering the
 * surface at every distance: the vertex shader computes the pixel size from
 * actual per-frame projection scale (fov + viewport height + distance), so
 * no uniform updates are needed when the camera moves.
 */
const POINT_SHADER = {
  vertex: `
    uniform float uBaseSize;
    uniform float uMinPx;
    uniform float uViewportH;
    uniform float uFovFactor;
    varying float vClampFade;
    #ifdef USE_CONFIDENCE
    attribute float confidence;  // per-point confidence from the pipeline PLY
    varying float vConfidence;
    #endif
    #ifdef USE_COLOR
    varying vec3 vColor;
    #endif
    void main() {
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      float dist = max(0.01, -mvPosition.z);
      // Pixel size of a uBaseSize-world-unit disc at this distance.
      float px = uViewportH * uFovFactor * uBaseSize / dist;
      gl_PointSize = max(px, uMinPx);
      // Fade tiny far sprites (alpha scaled by the clamped fraction) instead
      // of popping between the clamped and unclamped size.
      vClampFade = clamp(px / max(uMinPx, 0.001), 0.0, 1.0);
      #ifdef USE_COLOR
      vColor = color;
      #endif
      #ifdef USE_CONFIDENCE
      vConfidence = confidence;
      #endif
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragment: `
    uniform vec3 uColor;
    uniform bool uHasColors;
    uniform bool uHeatmap;
    uniform float uConfMin;   // ramp low end (below = dim gray)
    uniform float uConfHigh;  // threshold: below this = suppressed, above = full ramp
    varying float vClampFade;
    varying float vConfidence;
    #ifdef USE_COLOR
    varying vec3 vColor;
    #endif
    // Confidence ramp green (1.0) -> yellow -> red (0.0).
    vec3 ramp(float t) {
      t = clamp(t, 0.0, 1.0);
      if (t < 0.5) return mix(vec3(0.902, 0.494, 0.133), vec3(0.918, 0.702, 0.063), t * 2.0);
      return mix(vec3(0.918, 0.702, 0.063), vec3(0.180, 0.800, 0.443), (t - 0.5) * 2.0);
    }
    void main() {
      vec4 color = uHasColors ? vec4(vColor, 1.0) : vec4(uColor, 1.0);
      #ifdef USE_CONFIDENCE
      if (uHeatmap) {
        // Normalize the measured band so the ramp spans the run's actual range.
        float t = clamp((vConfidence - uConfMin) / max(1e-6, 1.0 - uConfMin), 0.0, 1.0);
        color = vec4(ramp(t), vConfidence >= uConfHigh ? 1.0 : 0.18);
      }
      #endif
      // Circular mask kills the square-sprite look when zoomed in.
      vec2 c = gl_PointCoord - 0.5;
      if (dot(c, c) > 0.25) discard;
      gl_FragColor = vec4(color.rgb, color.a * vClampFade);
    }
  `,
};

interface Props {
  meshUrl?: string | null;
  denseUrl: string | null;
  sparseUrl: string | null;
  combinedUrl?: string | null;
  poses: PosesJson | null;
  /** Rigid presentation transform from the backend (null → identity). */
  alignment: ViewerAlignment | null;
  /** External request to move the camera (preset buttons). */
  viewCommand: ViewCommand | null;
  showCameraPath: boolean;
  showGps: boolean;
  /** Heatmap mode: recolor points by their measured per-point confidence. */
  heatmap: boolean;
  /** Measured dense-cloud spacing (m) from the run's analysis — point sprites
   *  are sized from it so a zoomed-in cloud stays solid. Falls back to the
   *  legacy fixed size when the run has no analysis. */
  meanSpacingM?: number | null;
  /** Heatmap suppression threshold in percent (points below are dimmed). */
  confidenceThreshold: number;
  /** Fired once confidence data is (or isn't) found in the loaded cloud. */
  onConfidenceAvailable: (available: boolean, min: number, max: number) => void;
  onLoaded: (points: number, cameras: number, gps: number) => void;
  onError: (message: string) => void;
}

const OVERLAY_COLORS = {
  path: 0x38bdf8, // sky blue — clearly secondary to the reconstruction
  camera: 0x38bdf8,
  gps: 0x3b82f6,
};

/** Direction each view preset looks from, in viewer coordinates. */
function presetDirection(preset: ViewPreset): THREE.Vector3 {
  switch (preset) {
    case "top": return new THREE.Vector3(0, 1, 0.001);
    case "front": return new THREE.Vector3(0, 0.18, 1);
    case "side": return new THREE.Vector3(1, 0.18, 0);
    case "isometric":
    case "full":
    default: return new THREE.Vector3(1, 0.62, 1);
  }
}

/**
 * Framing box from vertex quantiles rather than the raw bounding box.
 *
 * A photogrammetric surface's bbox is set by isolated debris and by the empty
 * corners of a box around a mostly flat scene: measured on furnerhem_6 the raw
 * bbox spans 813 x 777 m where the 2nd-98th percentile envelope spans
 * 558 x 619 m, so framing the bbox rendered the model at 52% of the stage
 * width while the outliers it was framed around are not the thing anyone wants
 * to see. Nothing is hidden by this — the outliers simply do not set the zoom.
 */
export function robustFramingBox(obj: THREE.Object3D): THREE.Box3 | null {
  const pos = (obj as THREE.Mesh).geometry?.attributes?.position;
  if (!pos || pos.count < 64) return null;
  obj.updateWorldMatrix(true, false);
  const step = Math.max(1, Math.floor(pos.count / 50_000));
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += step) {
    v.fromBufferAttribute(pos, i).applyMatrix4(obj.matrixWorld);
    xs.push(v.x);
    ys.push(v.y);
    zs.push(v.z);
  }
  const at = (a: number[], f: number) => {
    a.sort((p, q) => p - q);
    return a[Math.round(f * (a.length - 1))];
  };
  return new THREE.Box3(
    new THREE.Vector3(at(xs, 0.02), at(ys, 0.02), at(zs, 0.02)),
    new THREE.Vector3(at(xs, 0.98), at(ys, 0.98), at(zs, 0.98)),
  );
}

/** Move camera+controls to frame `box` from a named preset. */
function frameBox(
  box: THREE.Box3,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  preset: ViewPreset,
) {
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const dir = presetDirection(preset).normalize();

  // Distance that puts every box corner inside BOTH frustum angles.
  //
  // The old rule fit the largest dimension to the vertical angle, which
  // ignores the stage's aspect (horizontal FOV = vertical x aspect) and the
  // box's depth toward the camera: in a portrait stage the model overflowed
  // sideways, and a wide-but-shallow model thrown in the same box sat small
  // in frame. Framing the box's circumscribed sphere instead is safe but
  // wastes the frame on a flat scene (measured 50% width fill). For a camera
  // at center + dir*d, a corner at offset o has depth (d - o.dir) along the
  // view axis, so it is inside the frustum when
  //   |o.right| <= tan(halfH) * (d - o.dir)  and  |o.up| <= tan(halfV) * (d - o.dir).
  const vFov = (camera.fov * Math.PI) / 180;
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * Math.max(camera.aspect, 1e-3));
  const tanV = Math.tan(vFov / 2);
  const tanH = Math.tan(hFov / 2);
  const forward = dir.clone().negate();
  const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0));
  if (right.lengthSq() < 1e-8) right.set(1, 0, 0); // looking straight down
  right.normalize();
  const up = new THREE.Vector3().crossVectors(right, forward).normalize();

  let fitDist = 0;
  const corner = new THREE.Vector3();
  for (let i = 0; i < 8; i++) {
    corner.set(
      i & 1 ? box.max.x : box.min.x,
      i & 2 ? box.max.y : box.min.y,
      i & 4 ? box.max.z : box.min.z,
    ).sub(center);
    const along = corner.dot(dir);
    fitDist = Math.max(
      fitDist,
      Math.abs(corner.dot(right)) / tanH + along,
      Math.abs(corner.dot(up)) / tanV + along,
    );
  }
  // Never crowd the frame tighter than the scene's own scale.
  fitDist = Math.max(fitDist, maxDim / 2) * 1.06;
  camera.position.copy(center).addScaledVector(dir, fitDist);
  controls.target.copy(center);
  controls.update();
}

/**
 * Renders the REAL run artifacts: the surface mesh (or dense/sparse point
 * cloud) from the run directory, the camera trajectory and GPS markers from
 * poses.json. Nothing here is decorative — every object comes from an actual
 * artifact.
 *
 * Coordinate handling: artifact data stays in original reconstruction
 * coordinates; the optional backend `alignment` (X_view = R·X_world + t) is
 * applied as a rigid GROUP transform so mesh, clouds, trajectory and GPS all
 * share one presentation frame with the ground on the y=0 grid. The grid is
 * sized to the scene, and the default camera is an isometric framing of the
 * whole model.
 */
export default function ViewerCanvas({
  meshUrl,
  denseUrl,
  sparseUrl,
  combinedUrl,
  poses,
  alignment,
  meanSpacingM,
  viewCommand,
  showCameraPath,
  showGps,
  heatmap,
  confidenceThreshold,
  onConfidenceAvailable,
  onLoaded,
  onError,
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    alignedGroup: THREE.Group;
    cameraPath: THREE.Object3D;
    gpsMarkers: THREE.Object3D;
    contentBox: THREE.Box3 | null;
    animId: number;
  } | null>(null);
  type SceneRef = NonNullable<typeof sceneRef.current>;

  /** Cloud materials registered by the last load — heatmap uniforms update live. */
  const cloudMaterialsRef = useRef<THREE.ShaderMaterial[]>([]);
  /** Measured confidence minimum of the loaded cloud (ramp normalization). */
  const confArrRef = useRef<{ min: number } | null>(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;
    cloudMaterialsRef.current = [];
    confArrRef.current = null;

    const w = container.clientWidth || 800;
    const h = container.clientHeight || 600;
    const scene = new THREE.Scene();
    // Backdrop is the CSS gradient behind the canvas (alpha: true) — a soft
    // studio falloff reads far better around a model than a flat fill.

    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 200000);
    camera.position.set(0, 60, 150);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // Photogrammetric appearance matched to how macOS Preview shows the same
    // GLB: sRGB output, filmic tone mapping, and image-based lighting so the
    // baked-in albedo is what reaches the screen instead of being crushed by
    // a few hard lights. Exposure is kept close to neutral so a correctly
    // exposed atlas (the scene's own photography) is not brightened into
    // clipping the way a pure preview-style tone curve would.
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    container.appendChild(renderer.domElement);

    // Image-based lighting: a neutral studio probe, pre-filtered for the
    // standard material's roughness response. This is the single biggest
    // difference from a hard-lit scene — soft wrap-around light plus real
    // specular response on the surface.
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTexture;
    scene.environmentIntensity = 1.35;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // A photogrammetric mesh's texture already CONTAINS the scene's own
    // illumination (it is baked from the source photographs), so dimming it
    // under directional shading makes a correctly built model read as "no
    // colour". Measured on the furnerhem_6_34be62 GLB: ambient 0.6 +
    // directional 0.8 rendered the mesh at L 37 while its atlas albedo is
    // L 75. three.js applies the Lambert BRDF (albedo / PI) to these
    // intensities, so full-albedo presentation needs the total to reach ~PI.
    // The studio environment supplies most of that now; ambient + a single
    // directional keep relief definition without flattening the model.
    //
    // For a Preview-like presentation the goal is softer wrap, not a hard
    // key light: a dim ambient fill plus a less aggressive directional keeps
    // the atlas readable while still giving the surface a faint 3D fold.
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.55);
    scene.add(dirLight);

    // Every run artifact lives under one aligned group so a single rigid
    // transform orients mesh, clouds, trajectory and GPS identically.
    const alignedGroup = new THREE.Group();
    alignedGroup.name = "aligned_scene";
    scene.add(alignedGroup);

    let grid: THREE.GridHelper | null = null;

    const cameraPath = new THREE.Group();
    cameraPath.name = "camera_path_group";
    alignedGroup.add(cameraPath);
    const gpsMarkers = new THREE.Group();
    gpsMarkers.name = "gps_markers_group";
    alignedGroup.add(gpsMarkers);

    const ref = {
      scene, camera, renderer, controls, alignedGroup,
      cameraPath, gpsMarkers, contentBox: null as THREE.Box3 | null, animId: 0,
    };
    sceneRef.current = ref;
    if (import.meta.env.DEV) {
      // dev-only inspection handle (tests/debugging); stripped in production builds
      (window as unknown as Record<string, unknown>).__strataViewer = { scene, camera, renderer, controls, alignedGroup };
    }
    ref.animId = requestAnimationFrame(function animate() {
      controls.update();
      renderer.render(scene, camera);
      ref.animId = requestAnimationFrame(animate);
    });

    const onResize = () => {
      const cw = container.clientWidth || w;
      const ch = container.clientHeight || h;
      camera.aspect = cw / ch;
      camera.updateProjectionMatrix();
      renderer.setSize(cw, ch);
    };
    window.addEventListener("resize", onResize);
    // Collapsible side panels change the stage's width without a window
    // resize — observe the container itself so the canvas re-fits.
    const ro = new ResizeObserver(onResize);
    ro.observe(container);

    // ---- presentation state -------------------------------------------------

    const applyAlignment = (a: ViewerAlignment | null) => {
      if (a && Array.isArray(a.R_view) && a.R_view.length === 3) {
        // X_view = R_view · X_world + t_view  (row-major JSON → column-major THREE)
        const m = new THREE.Matrix4().set(
          a.R_view[0][0], a.R_view[0][1], a.R_view[0][2], a.t_view?.[0] ?? 0,
          a.R_view[1][0], a.R_view[1][1], a.R_view[1][2], a.t_view?.[1] ?? 0,
          a.R_view[2][0], a.R_view[2][1], a.R_view[2][2], a.t_view?.[2] ?? 0,
          0, 0, 0, 1,
        );
        alignedGroup.matrix.copy(m);
      } else {
        alignedGroup.matrix.identity();
      }
      alignedGroup.matrixAutoUpdate = false;
      alignedGroup.matrixWorldNeedsUpdate = true;
    };

    const refreshPresentation = () => {
      // bbox of all aligned content (grid + lighting are sized from this)
      const box = new THREE.Box3().setFromObject(alignedGroup);
      if (box.isEmpty()) return;
      // The camera frames the SURFACE, not every overlay: the camera
      // trajectory runs well outside the reconstructed area, and including it
      // shrank the model to a small patch in the middle of the stage. Overlays
      // stay subordinate, which is what they are.
      const surfaceBox = new THREE.Box3();
      alignedGroup.traverse((o) => {
        if (o.name !== "run_mesh" && o.name !== "run_cloud") return;
        const robust = robustFramingBox(o);
        if (robust) surfaceBox.union(robust);
      });
      ref.contentBox = surfaceBox.isEmpty() ? box : surfaceBox;
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;

      // scene-sized ground grid on the y=0 plane
      const niceCeil = (v: number) => {
        const p = Math.pow(10, Math.floor(Math.log10(v)));
        for (const m of [1, 2, 2.5, 5, 10]) {
          if (m * p >= v) return m * p;
        }
        return 10 * p;
      };
      const gridSize = niceCeil(maxDim * 2.2);
      if (grid) {
        scene.remove(grid);
        grid.geometry.dispose();
        (grid.material as THREE.Material).dispose();
      }
      grid = new THREE.GridHelper(gridSize, 40, 0x475569, 0x334155);
      (grid.material as THREE.Material).transparent = true;
      (grid.material as THREE.Material).opacity = 0.35;
      grid.name = "ground_grid";
      scene.add(grid);

      // scale lights + clipping to the scene
      dirLight.position.set(center.x + maxDim, center.y + maxDim * 1.6, center.z + maxDim);
      camera.near = Math.max(maxDim / 1000, 0.01);
      camera.far = Math.max(maxDim * 60, 2000);
      camera.updateProjectionMatrix();

      frameView("full");
    };

    const frameView = (preset: ViewPreset) => {
      if (!ref.contentBox) return;
      frameBox(ref.contentBox, camera, controls, preset);
    };

    // ---- artifact loading ---------------------------------------------------

    const loader = new PLYLoader();
    // The pipeline's PLYs carry a per-point `confidence` scalar (observations +
    // fusion residual, see app.services.confidence_estimator) — map it into a
    // shader attribute so the heatmap mode colors from measured data.
    loader.setCustomPropertyNameMapping({ confidence: ["confidence"] });
    let cancelled = false;
    let loadedPoints = 0;

    const makeCloudMaterial = (hasColors: boolean, baseWorldSize: number, hasConfidence: boolean) => {
      const uniforms = {
        uBaseSize: { value: baseWorldSize },
        uMinPx: { value: 1.5 },
        uViewportH: { value: container.clientHeight || h },
        uFovFactor: { value: 1 / (2 * Math.tan((camera.fov * Math.PI) / 360)) },
        uColor: { value: new THREE.Color(0x94a3b8) },
        uHasColors: { value: hasColors },
        uHeatmap: { value: false },
        uConfMin: { value: 0.0 },
        uConfHigh: { value: 0.6 },
      };
      const mat = new THREE.ShaderMaterial({
        uniforms,
        vertexShader: POINT_SHADER.vertex,
        fragmentShader: POINT_SHADER.fragment,
        defines: hasConfidence ? { USE_CONFIDENCE: "" } : {},
        vertexColors: hasColors,
        transparent: true,
        depthWrite: true,
      });
      const onResize = () => {
        mat.uniforms.uViewportH.value = container.clientHeight || h;
      };
      window.addEventListener("resize", onResize);
      (mat as THREE.ShaderMaterial & { __cleanup?: () => void }).__cleanup = () =>
        window.removeEventListener("resize", onResize);
      return mat;
    };

    const addCloud = (geometry: THREE.BufferGeometry, hasColors: boolean, baseWorldSize: number, name: string) => {
      const confAttr = geometry.getAttribute("confidence") as THREE.BufferAttribute | undefined;
      const hasConfidence = Boolean(confAttr && confAttr.count === geometry.attributes.position.count);
      if (hasConfidence && confAttr) {
        // Single pass over the measured band (once per load) for ramp bounds.
        const arr = confAttr.array as ArrayLike<number>;
        let mn = Infinity;
        for (let i = 0; i < confAttr.count; i++) { const v = arr[i]; if (v < mn) mn = v; }
        confArrRef.current = { min: mn };
        onConfidenceAvailable(true, mn, 1.0);
      } else {
        confArrRef.current = null;
        onConfidenceAvailable(false, 0, 1);
      }
      const material = makeCloudMaterial(hasColors, baseWorldSize, hasConfidence);
      cloudMaterialsRef.current.push(material);
      const pointsObj = new THREE.Points(geometry, material);
      pointsObj.name = name;
      alignedGroup.add(pointsObj);
      loadedPoints = geometry.attributes.position.count;
      refreshPresentation();
      onLoaded(loadedPoints, pts.length, gpsFrames.length);
    };

    const MESH_UNLIT_VS = `
      varying vec3 vColor;
      void main() {
        vColor = color.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;
    const MESH_UNLIT_FS = `
      varying vec3 vColor;
      void main() { gl_FragColor = vec4(vColor, 1.0); }
    `;

    const addMesh = (geometry: THREE.BufferGeometry, hasColors: boolean, map?: THREE.Texture | null) => {
      geometry = withSmoothNormals(geometry);

      // Photogrammetric color comes from one of two sources, never both:
      //  - a baked texture (textured GLB loader gave us `map`),
      //  - or per-vertex rgb (PLY/untextured GLB with a color attribute).
      // For the color case the frame-level photography is the appearance —
      // lighting it under a BRDF (Lambert albedo / PI, plus the studio
      // environment) crushes the values below what the source frames store
      // and what macOS Preview shows for the same PLY. So: texture lit
      // matte surface, vertex colors unlit passthrough.
      //
      // A second viewer-side pass softens the remaining hard triangle edges
      // on blobby photogrammetry surfaces (Poisson/BPA) without disturbing
      // the authored UVs: it is bounded, rollback-safe, and skipped for large
      // or degenerate inputs so the tab stays responsive.
      if (map || geometry.attributes.color) {
        const smoothed = withSmoothPositions(geometry, { passes: 1 });
        if (smoothed !== geometry) {
          geometry.dispose();
          geometry = smoothed;
        }
      }
      if (map) map.anisotropy = renderer.capabilities.getMaxAnisotropy();

      const hasVertexColors = Boolean(geometry.attributes.color);
      if (map) {
        // Preview presents the baked atlas as a photographed surface with
        // only a faint 3D fold, not a PBR object under a studio probe. A
        // higher roughness / lower env intensity keeps the atlas albedo
        // dominant while the environment adds just enough wrap to stop the
        // mesh reading as flat colour.
        const material = new THREE.MeshStandardMaterial({
          vertexColors: false,
          map,
          side: THREE.DoubleSide,
          roughness: 0.92,
          metalness: 0.0,
          envMapIntensity: 0.55,
          color: new THREE.Color(0xffffff),
        });
        const meshObj = new THREE.Mesh(geometry, material);
        meshObj.name = "run_mesh";
        alignedGroup.add(meshObj);
        loadedPoints = geometry.attributes.position.count;
        refreshPresentation();
        onLoaded(loadedPoints, pts.length, gpsFrames.length);
        return;
      }
      if (hasVertexColors) {
        const c = geometry.attributes.color as THREE.BufferAttribute;
        const arr = new Float32Array(c.count * 3);
        for (let i = 0; i < c.count; i++) {
          arr[i * 3] = c.getX(i);
          arr[i * 3 + 1] = c.getY(i);
          arr[i * 3 + 2] = c.getZ(i);
        }
        geometry.setAttribute("color", new THREE.BufferAttribute(arr, 3));
        const material = new THREE.ShaderMaterial({
          vertexShader: MESH_UNLIT_VS,
          fragmentShader: MESH_UNLIT_FS,
          vertexColors: false,
          side: THREE.DoubleSide,
          depthWrite: true,
        });
        const meshObj = new THREE.Mesh(geometry, material);
        meshObj.name = "run_mesh";
        alignedGroup.add(meshObj);
        loadedPoints = geometry.attributes.position.count;
        refreshPresentation();
        onLoaded(loadedPoints, pts.length, gpsFrames.length);
        return;
      }
      // No colors, no texture — neutral fallback matte surface.
      // Smooth the raw (non-textured, non-colored) fallback mesh a touch too,
      // since a hard unlit soup still reads as faceted against the studio
      // backdrop even when there is no atlas to preserve.
      const fallbackSmoothed = withSmoothPositions(geometry, { passes: 1 });
      const fbGeo = fallbackSmoothed !== geometry ? fallbackSmoothed : geometry;
      if (fbGeo !== geometry) geometry.dispose();
      const material = new THREE.MeshStandardMaterial({
        vertexColors: false,
        map: null,
        side: THREE.DoubleSide,
        roughness: 0.85,
        metalness: 0.0,
        color: new THREE.Color(0xb0bec5),
      });
      const meshObj = new THREE.Mesh(fbGeo, material);
      meshObj.name = "run_mesh";
      alignedGroup.add(meshObj);
      loadedPoints = fbGeo.attributes.position.count;
      refreshPresentation();
      onLoaded(loadedPoints, pts.length, gpsFrames.length);
    };

    const loadMesh = (url: string) => {
      if (url.endsWith(".glb")) {
        // Viewer LOD: decimated GLB (the full mesh.ply is never shipped).
        new GLTFLoader().load(
          url,
          (gltf) => {
            if (cancelled) return;
            const obj = gltf.scene.getObjectByProperty("isMesh", true) as THREE.Mesh | undefined;
            if (!obj) {
              onError("Viewer LOD contains no mesh.");
              return;
            }
            const geometry = obj.geometry.clone();
            const hasColors = Boolean(geometry.attributes.color);
            // Textured GLB: the loader decoded the embedded PNG into the
            // material map (UVs arrive as TEXCOORD_0 → 'uv'); reuse it so the
            // atlas colors the LOD. Untextured GLB falls back to vertex colors.
            const map = (obj.material as THREE.MeshStandardMaterial | undefined)?.map ?? null;
            if (geometry.attributes.color && geometry.attributes.color.itemSize === 4 && !map) {
              const c = geometry.attributes.color;
              const arr = new Float32Array(c.count * 3);
              for (let i = 0; i < c.count; i++) {
                arr[i * 3] = c.getX(i);
                arr[i * 3 + 1] = c.getY(i);
                arr[i * 3 + 2] = c.getZ(i);
              }
              geometry.setAttribute("color", new THREE.BufferAttribute(arr, 3));
            }
            if (map) geometry.deleteAttribute("color");
            addMesh(geometry, Boolean(geometry.attributes.color), map);
          },
          undefined,
          () => {
            if (cancelled) return;
            onError("Failed to load the viewer mesh.");
          },
        );
        return;
      }
      loader.load(
        url,
        (geometry) => {
          if (cancelled) return;
          const hasColors = Boolean(geometry.attributes.color);
          const isTriangleMesh = Boolean(geometry.index) || (geometry.attributes.position && geometry.attributes.position.count >= 3);
          if (isTriangleMesh && geometry.index) {
            addMesh(geometry, hasColors);
          } else {
            addCloud(geometry, hasColors, 0.35, "run_cloud");
          }
          loadedPoints = geometry.attributes.position.count;
          refreshPresentation();
          onLoaded(loadedPoints, pts.length, gpsFrames.length);
        },
        undefined,
        () => {
          if (cancelled) return;
          if (denseUrl) loadCloud(denseUrl, 0.35);
          else if (sparseUrl) loadCloud(sparseUrl, 0.6);
          else onError("Surface reconstruction unavailable for this run.");
        },
      );
    };

    const loadCloud = (url: string, size: number) => {
      loader.load(
        url,
        (geometry) => {
          if (cancelled) return;
          addCloud(geometry, Boolean(geometry.attributes.color), size, "run_cloud");
        },
        undefined,
        () => {
          if (cancelled) return;
          onError(`Failed to load ${url.split("/").pop()}.`);
        },
      );
    };

    // Point-sprite world size from the cloud's MEASURED spacing: a sprite
    // ~1.15x spacing keeps discs overlapping at every zoom (fixed 0.35 m on
    // a 0.55 m-spaced cloud covered only ~1/3 of the surface up close).
    const denseSpriteSize = meanSpacingM ? meanSpacingM * 1.15 : 0.35;
    if (meshUrl) loadMesh(meshUrl);
    else if (combinedUrl) loadCloud(combinedUrl, denseSpriteSize);
    else if (denseUrl) loadCloud(denseUrl, denseSpriteSize);
    else if (sparseUrl) loadCloud(sparseUrl, 0.6);
    else onError("Surface reconstruction unavailable for this run.");

    // Camera trajectory + GPS markers from the real poses.json. Overlays are
    // deliberately subordinate: thin line, small semi-transparent markers.
    const frames = poses?.frames ?? [];
    const pts = frames.filter((f) => Array.isArray(f.t) && f.t.length === 3).map((f) => new THREE.Vector3(f.t[0], f.t[1], f.t[2]));
    const gpsFrames = frames.filter((f) => f.gps);

    // scene radius for marker sizing (poses span ∝ scene span)
    let sceneRadius = 10;
    if (pts.length >= 2) {
      const span = new THREE.Vector3().subVectors(pts[pts.length - 1], pts[0]).length();
      sceneRadius = Math.max(span, 5);
    }
    const markerR = Math.max(0.004 * sceneRadius, 0.02);
    // world-space "up" for the GPS offset — same direction the alignment maps to +Y
    const upWorld = new THREE.Vector3(
      alignment?.R_view?.[1]?.[0] ?? 0,
      alignment?.R_view?.[1]?.[1] ?? 1,
      alignment?.R_view?.[1]?.[2] ?? 0,
    ).normalize();

    if (pts.length >= 2) {
      const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const line = new THREE.Line(
        lineGeo,
        new THREE.LineBasicMaterial({ color: OVERLAY_COLORS.path, transparent: true, opacity: 0.85 }),
      );
      line.name = "camera_path";
      cameraPath.add(line);
    }
    pts.forEach((p) => {
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(markerR, 8, 8),
        new THREE.MeshBasicMaterial({ color: OVERLAY_COLORS.camera, transparent: true, opacity: 0.75 }),
      );
      dot.position.copy(p);
      dot.name = "camera_marker";
      cameraPath.add(dot);
    });
    gpsFrames.forEach((f) => {
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(markerR * 1.5, 10, 10),
        new THREE.MeshBasicMaterial({ color: OVERLAY_COLORS.gps }),
      );
      dot.position.set(f.t[0], f.t[1], f.t[2]).addScaledVector(upWorld, markerR * 6);
      dot.name = "gps_marker";
      gpsMarkers.add(dot);
    });
    onLoaded(loadedPoints, pts.length, gpsFrames.length);

    applyAlignment(alignment);
    refreshPresentation();

    return () => {
      cancelled = true;
      window.removeEventListener("resize", onResize);
      ro.disconnect();
      cancelAnimationFrame(ref.animId);
      alignedGroup.traverse((obj) => {
        const mat = (obj as THREE.Mesh).material as THREE.Material | undefined;
        const cleanup = (mat as (THREE.Material & { __cleanup?: () => void }) | undefined)?.__cleanup;
        if (cleanup) cleanup();
      });
      renderer.dispose();
      envTexture.dispose();
      pmrem.dispose();
      container.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meshUrl, denseUrl, sparseUrl, combinedUrl, poses, alignment, meanSpacingM]);

  // Layer visibility — toggled without reloading the cloud.
  useEffect(() => {
    const ref = sceneRef.current;
    if (!ref) return;
    ref.cameraPath.visible = showCameraPath;
    ref.gpsMarkers.visible = showGps;
  }, [showCameraPath, showGps]);

  // Heatmap mode + threshold — uniform updates only, no reload.
  useEffect(() => {
    const confMin = confArrRef.current?.min ?? 0;
    for (const mat of cloudMaterialsRef.current) {
      mat.uniforms.uHeatmap.value = heatmap && confArrRef.current != null;
      mat.uniforms.uConfMin.value = confMin;
      mat.uniforms.uConfHigh.value = confidenceThreshold / 100;
    }
  }, [heatmap, confidenceThreshold]);

  // View presets requested from outside (toolbar buttons).
  useEffect(() => {
    if (!viewCommand) return;
    const ref = sceneRef.current;
    if (!ref?.contentBox) return;
    frameBox(ref.contentBox, ref.camera, ref.controls, viewCommand.preset);
  }, [viewCommand]);

  return <div className="viewer-3d" ref={mountRef} />;
}
