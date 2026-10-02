/**
 * Smooth-normal contract for the textured mesh path.
 *
 * A photogrammetric GLB stores one vertex per triangle corner, so
 * `computeVertexNormals()` can only ever flat-shade it — the measured reason
 * the same file reads better in a native viewer app than in the browser tab.
 * These tests pin the three properties the smoothing must hold, in the order
 * they matter: the texture mapping is untouched, the surface shades
 * continuously across shared positions, and the shading is area-weighted like
 * `computeVertexNormals` rather than an unweighted average.
 */

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { robustFramingBox, withSmoothNormals } from "./ViewerCanvas";

/**
 * Two triangles sharing an edge, laid out as the triangle soup a GLB contains:
 * every corner is its own vertex even where positions coincide (4 shared
 * positions, 6 corners). UVs differ at the shared positions on purpose — that
 * is what a texture atlas seam looks like.
 */
function foldedQuadSoup(): THREE.BufferGeometry {
  const positions = new Float32Array([
    // triangle A
    0, 0, 0,
    1, 0, 0,
    1, 0, 0, // deliberately duplicated corner (degenerate sliver, as real LODs contain)
    // triangle B
    1, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  const uvs = new Float32Array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]);
  const index = new Uint16Array([0, 1, 2, 3, 4, 5]);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  g.setIndex(new THREE.BufferAttribute(index, 1));
  return g;
}

/** A square built from two triangles that share their diagonal corners. */
function sharedEdgeQuad(): THREE.BufferGeometry {
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    1, 0, 1,
    // second triangle re-states two positions with different UVs (atlas seam)
    0, 0, 0,
    1, 0, 1,
    0, 0, 1,
  ]);
  const uvs = new Float32Array([0, 0, 1, 0, 1, 1, 0.5, 0.5, 0.5, 1, 0, 1]);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  return g;
}

describe("withSmoothNormals", () => {
  it("leaves the UV atlas and the vertex count exactly as authored", () => {
    const g = sharedEdgeQuad();
    const uvBefore = Array.from((g.getAttribute("uv") as THREE.BufferAttribute).array);
    const posBefore = Array.from((g.getAttribute("position") as THREE.BufferAttribute).array);

    const out = withSmoothNormals(g);

    // Welding would merge the two seam corners and remap the atlas; this path
    // must not touch either the counts or the coordinates.
    expect(out).toBe(g);
    expect(out.getAttribute("position").count).toBe(6);
    expect(Array.from((out.getAttribute("position") as THREE.BufferAttribute).array)).toEqual(posBefore);
    expect(Array.from((out.getAttribute("uv") as THREE.BufferAttribute).array)).toEqual(uvBefore);
  });

  it("gives every corner at one position the SAME normal, so the surface shades continuously", () => {
    const out = withSmoothNormals(sharedEdgeQuad());
    const pos = out.getAttribute("position") as THREE.BufferAttribute;
    const nrm = out.getAttribute("normal") as THREE.BufferAttribute;

    // The two triangles share positions 0 and 2 (corners 0/3 and 2/4).
    const cornerNormal = (c: number) => [nrm.getX(c), nrm.getY(c), nrm.getZ(c)];
    expect(cornerNormal(3)).toEqual(cornerNormal(0));
    expect(cornerNormal(4)).toEqual(cornerNormal(2));

    // And the smoothed normal is unit length at every corner.
    for (let c = 0; c < pos.count; c++) {
      expect(Math.hypot(cornerNormal(c)[0], cornerNormal(c)[1], cornerNormal(c)[2])).toBeCloseTo(1, 5);
    }
  });

  it("weights by area: a 100x larger neighbour dominates the shared normal", () => {
    // A large +Y triangle (cross magnitude 100) and a small +X triangle (cross
    // magnitude 1) meeting at the origin. An UNWEIGHTED mean would land on
    // (0.707, 0.707, 0); area weighting must keep the shared normal at
    // ~(0.01, 0.99995, 0). This is the assertion that separates the two.
    const positions = new Float32Array([
      0, 0, 0,
      0, 0, 10,
      10, 0, 0,
      // small triangle sharing the origin
      0, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 3, 4, 5]), 1));

    const out = withSmoothNormals(g);
    const nrm = out.getAttribute("normal") as THREE.BufferAttribute;
    expect(nrm.getY(0)).toBeGreaterThan(0.99);
    expect(Math.abs(nrm.getX(0))).toBeLessThan(0.05);
  });

  it("falls back to flat normals instead of throwing on degenerate input", () => {
    const noPos = new THREE.BufferGeometry();
    expect(() => withSmoothNormals(noPos)).not.toThrow();

    const tooFew = new THREE.BufferGeometry();
    tooFew.setAttribute("position", new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 1, 1]), 3));
    const out = withSmoothNormals(tooFew);
    expect(out.getAttribute("normal")).toBeTruthy();
  });

  it("frames the measured surface, not its debris, when choosing the zoom", () => {
    // A dense 40x40 m patch plus three stray faces thrown 1 km out — the shape
    // a photogrammetric mesh actually has. The raw bbox is 2 km wide and would
    // zoom the model down to a speck (measured 52% of the stage width before
    // this rule existed); the 2-98% envelope must stay on the patch.
    const verts: number[] = [];
    for (let i = 0; i <= 40; i++) {
      for (let j = 0; j <= 40; j++) verts.push(i, 0, j);
    }
    for (const x of [-1000, 1000, 1000]) verts.push(x, 0, 0);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(verts), 3));
    const obj = new THREE.Mesh(g);
    obj.updateMatrixWorld(true);

    const box = robustFramingBox(obj);
    expect(box).not.toBeNull();
    expect(box!.min.x).toBeGreaterThan(-100);
    expect(box!.max.x).toBeLessThan(100);
    expect(box!.max.z).toBeGreaterThan(30);
  });

  it("keeps a soup's duplicated corners in agreement (the real GLB shape)", () => {
    const out = withSmoothNormals(foldedQuadSoup());
    const pos = out.getAttribute("position") as THREE.BufferAttribute;
    const nrm = out.getAttribute("normal") as THREE.BufferAttribute;
    const byPosition = new Map<string, string>();
    let violations = 0;
    for (let c = 0; c < pos.count; c++) {
      const key = `${pos.getX(c)},${pos.getY(c)},${pos.getZ(c)}`;
      const value = `${nrm.getX(c)},${nrm.getY(c)},${nrm.getZ(c)}`;
      const seen = byPosition.get(key);
      if (seen === undefined) byPosition.set(key, value);
      else if (seen !== value) violations++;
    }
    expect(violations).toBe(0);
  });
});
