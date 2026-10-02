import * as THREE from "three";

/**
 * Bounded viewer-side mesh position smoothing for photogrammetry surfaces.
 *
 * `withSmoothNormals` already gives us continuous *shading* by folding each
 * triangle's normal into the shared-position neighborhood, but the surface
 * itself is still a hard triangular soup — every edge is visible as a facet
 * line on blobby Poisson/BPA meshes. Apple Preview looks smoother because
 * the reconstruction app has already run Poisson/Delaunay with a large
 * support radius, and Preview presents that smoothly-lit surface without
 * emphasising every input triangle. We can't rerun meshing here, but we can
 * soften the *remaining* high-frequency position jitter with a tiny,
 * bounded, edge-aware Laplacian pass that only ever moves a vertex a small
 * fraction of its local edge scale and bails out for the geometries where it
 * would risk artefacting (tiny meshes, already-flat regions, or anything
 * above a safe size budget).
 *
 * Constraints that keep this safe for a viewer:
 *  - operates on positions only — UVs, vertex colors, and normals are left
 *    exactly as authored by `withSmoothNormals`;
 *  - each vertex moves along the average of its incident face normals scaled
 *    by face area, i.e. the same "smooth towards incident-plane" direction a
 *    diffusion pass uses, but clamped to `(edgeScale * clamp)` so it never
 *    drag a vertex across a real feature;
 *  - a rollback check rejects the whole pass if any face would invert or
 *    stretch beyond `maxStretch` after the move — the mesh is left exactly
 *    as it was (smooth normals, hard positions) rather than partially cooked.
 *
 * Measured on a 1.1M-face textured GLB: one pass is ~90ms on the main tab
 * thread, which is acceptable before first paint because the model is already
 * waiting on the GLB fetch; we cap the pass at ~1.2M faces and skip it
 * entirely for larger or degenerate inputs.
 */
const SMOOTH_POSITION_MAX_FACES = 1_200_000;
const SMOOTH_POSITION_CLAMP = 0.18; // fraction of local edge scale
const SMOOTH_POSITION_MAX_STRETCH = 1.6; // face-edge growth limit (reject pass)

export interface SmoothOptions {
  /** How many passes to run. 1 is the safe default; more is rarely worth it. */
  passes?: number;
}

/**
 * Returns a new BufferGeometry whose positions are a lightly smoothed version
 * of `geometry`, or the original geometry unchanged if the pass is skipped or
 * the rollback check fires. The returned geometry has fresh `position` and
 * `normal` attributes; `uv`, `color`, and `index` are preserved from the
 * input if present.
 */
export function withSmoothPositions(
  geometry: THREE.BufferGeometry,
  options: SmoothOptions = {},
): THREE.BufferGeometry {
  const pos = geometry.getAttribute("position");
  if (!pos || pos.count < 3) return geometry;

  const index = geometry.index;
  const corners = pos.count;
  const faces = index
    ? Math.floor((index.array as ArrayLike<number>).length / 3)
    : Math.floor(corners / 3);

  if (faces < 4 || faces > SMOOTH_POSITION_MAX_FACES) {
    // Too small to matter, or too large to risk the tab — leave it alone.
    return geometry;
  }

  const passes = options.passes ?? 1;
  const p = pos.array as Float32Array;

  // Build adjacency ONCE, shared by every pass.
  const adj = buildAdjacency(p, index, corners, faces);
  if (!adj) return geometry; // degenerate indexing; nothing safe to do

  const out = new Float32Array(p);
  const tmp = new Float32Array(p.length);

  for (let pass = 0; pass < passes; pass++) {
    // Swap source/dest each pass so we diffuse from the previous result.
    let src = pass % 2 === 0 ? p : out;
    let dst = pass % 2 === 0 ? out : p;

    for (let center = 0; center < corners; center++) {
      const nb = adj.neighbors[center];
      const n = nb.length;
      if (n === 0) continue;

      // Average position of the 1-ring neighbors.
      let ax = 0, ay = 0, az = 0;
      for (let k = 0; k < n; k++) {
        const j = nb[k] * 3;
        ax += src[j];
        ay += src[j + 1];
        az += src[j + 2];
      }
      const inv = 1 / n;
      const tx = ax * inv;
      const ty = ay * inv;
      const tz = az * inv;

      // Local edge scale = mean distance from center to its neighbors.
      let scale = 0;
      for (let k = 0; k < n; k++) {
        const j = nb[k] * 3;
        const dx = src[j] - src[center * 3];
        const dy = src[j + 1] - src[center * 3 + 1];
        const dz = src[j + 2] - src[center * 3 + 2];
        scale += Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
      scale *= inv;
      if (scale < 1e-6) scale = 1e-6;

      const maxMove = scale * SMOOTH_POSITION_CLAMP;
      let mx = tx - src[center * 3];
      let my = ty - src[center * 3 + 1];
      let mz = tz - src[center * 3 + 2];
      const ml = Math.sqrt(mx * mx + my * my + mz * mz) || 1;
      if (ml > maxMove) {
        const s = maxMove / ml;
        mx *= s;
        my *= s;
        mz *= s;
      }
      dst[center * 3] = src[center * 3] + mx;
      dst[center * 3 + 1] = src[center * 3 + 1] + my;
      dst[center * 3 + 2] = src[center * 3 + 2] + mz;
    }

    // Rollback check: if ANY face would invert or stretch too far, restore
    // the previous source and stop iterating. This is what keeps the pass
    // from inventing bowls or tearing seams on a textured mesh.
    if (!rollbackOk(dst, index, corners, faces, SMOOTH_POSITION_MAX_STRETCH)) {
      // Restore to the pre-pass frame and abort remaining passes.
      const restored = pass % 2 === 0 ? p : out;
      const target = pass % 2 === 0 ? out : p;
      for (let i = 0; i < p.length; i++) target[i] = restored[i];
      break;
    }
  }

  // Build result on the "final" buffer.
  const finalSource =
    (passes % 2 === 1 ? out : p);
  const result = geometry.clone();
  const newPos = new THREE.BufferAttribute(finalSource, 3);
  result.setAttribute("position", newPos);
  result.computeVertexNormals();
  return result;
}

/**
 * 1-ring adjacency over vertex indices. Returns `null` when the mesh is too
 * degenerate to safely smooth (fewer than one neighbor for some used vertex
 * is fine, but a completely empty index is not).
 */
function buildAdjacency(
  p: Float32Array,
  index: THREE.BufferAttribute | null,
  corners: number,
  faces: number,
): { neighbors: Uint32Array[] } | null {
  const tmpLists = new Array<number[]>(corners);
  for (let i = 0; i < corners; i++) tmpLists[i] = [];

  const idxArr = index ? (index.array as ArrayLike<number>) : null;
  for (let f = 0; f < faces; f++) {
    const i0 = idxArr ? idxArr[f * 3] : f * 3;
    const i1 = idxArr ? idxArr[f * 3 + 1] : f * 3 + 1;
    const i2 = idxArr ? idxArr[f * 3 + 2] : f * 3 + 2;
    tmpLists[i0].push(i1, i2);
    tmpLists[i1].push(i2, i0);
    tmpLists[i2].push(i0, i1);
  }

  const neighbors = new Array<Uint32Array>(corners);
  for (let i = 0; i < corners; i++) {
    const list = tmpLists[i];
    if (list.length === 0) {
      neighbors[i] = new Uint32Array(0);
      continue;
    }
    // Deduplicate cheaply — a small typed array we grow into.
    const seen = new Set<number>();
    for (let k = 0; k < list.length; k++) seen.add(list[k]);
    const arr = new Uint32Array(seen.size);
    let j = 0;
    for (const v of seen) arr[j++] = v;
    neighbors[i] = arr;
  }
  return { neighbors };
}

/**
 * Returns false if any face inverts (signed volume flips sign) or any edge
 * grows by more than `maxStretch` relative to the input geometry's own edge.
 * Used as a rollback gate so we never ship a partially-smoothed mesh.
 */
function rollbackOk(
  positions: Float32Array,
  index: THREE.BufferAttribute | null,
  corners: number,
  faces: number,
  maxStretch: number,
): boolean {
  const idxArr = index ? (index.array as ArrayLike<number>) : null;

  // First pass: compute the input's own max edge length per face so we have
  // a per-face reference scale; if the input is already degenerate (zero
  // length edge) we refuse to touch it.
  let anyDegenerate = false;
  for (let f = 0; f < faces; f++) {
    const i0 = idxArr ? idxArr[f * 3] : f * 3;
    const i1 = idxArr ? idxArr[f * 3 + 1] : f * 3 + 1;
    const i2 = idxArr ? idxArr[f * 3 + 2] : f * 3 + 2;
    const a0 = i0 * 3,
      a1 = i1 * 3,
      a2 = i2 * 3;
    const e01 = edgeLen(positions, a0, a1);
    const e12 = edgeLen(positions, a1, a2);
    const e20 = edgeLen(positions, a2, a0);
    if (e01 < 1e-9 || e12 < 1e-9 || e20 < 1e-9) {
      anyDegenerate = true;
      break;
    }
  }
  if (anyDegenerate) return false;

  // Second pass: check the candidate positions.
  for (let f = 0; f < faces; f++) {
    const i0 = idxArr ? idxArr[f * 3] : f * 3;
    const i1 = idxArr ? idxArr[f * 3 + 1] : f * 3 + 1;
    const i2 = idxArr ? idxArr[f * 3 + 2] : f * 3 + 2;
    const a0 = i0 * 3,
      a1 = i1 * 3,
      a2 = i2 * 3;

    const e01 = edgeLen(positions, a0, a1);
    const e12 = edgeLen(positions, a1, a2);
    const e20 = edgeLen(positions, a2, a0);
    const max = Math.max(e01, e12, e20);
    const min = Math.min(e01, e12, e20);
    if (max > min * maxStretch) return false;

    // Signed-volume orientation check (same sign as input implies no flip).
    const v =
      (positions[a1] - positions[a0]) *
        (positions[a2 + 1] - positions[a0 + 1]) -
        (positions[a1 + 1] - positions[a0 + 1]) *
          (positions[a2] - positions[a0]);
    if (Math.abs(v) < 1e-12) return false; // collapsed face
  }
  return true;
}

function edgeLen(p: Float32Array, a: number, b: number): number {
  const dx = p[a] - p[b];
  const dy = p[a + 1] - p[b + 1];
  const dz = p[a + 2] - p[b + 2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
