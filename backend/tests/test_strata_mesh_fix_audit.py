"""Audit of the uploaded ``strata_mesh_fix`` patch — measured, not assumed.

Three contracts, each proven against real inputs from ``data/``:

1. **Radius-cap collapse** — the uploaded ``mesh_generator`` derives a BPA
   radius cap from ``detect_layers(...)['min_separation_m']`` computed with
   the *static default* voxel (0.05 m). That field is the detection
   THRESHOLD (1.5 × voxel = 0.075 m), not a measured separation, so on a
   real far-field cloud (spacing ≈ 0.93 m) the cap clamps every BPA radius
   to 0.06 m — below the point spacing — and meshing collapses. The current
   production path meshes the same cloud fine. Input: the real airport8
   synthetic-dataset depth frame fused through the production fusion and
   optimization path.

2. **``from_settings`` is a no-op** — every value the uploaded
   ``MeshParams.from_settings()`` hardcodes already equals the current
   dataclass defaults.

3. **Phase-2 color passthrough misaligns** — the uploaded
   ``ball_pivot_adaptive`` indexes the PRE-weld cloud's colors with indices
   into the POST-weld vertex array. On any cloud where the patch weld merges
   vertices (which is its purpose), assigned colors do not match the color
   of the nearest original point, proven on a gradient-colored plane.

The uploaded modules are loaded via importlib under private names; the
stage-registry entry is swapped around the import because the uploaded file
re-registers ``mesh_generation``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
UPLOAD = BACKEND.parent / "strata_mesh_fix"
AIRPORT8 = BACKEND.parent / "data" / "test" / "airport8"

pytestmark = pytest.mark.skipif(
    not UPLOAD.exists() or not (AIRPORT8 / "poses.csv").exists(),
    reason="strata_mesh_fix upload or data/test/airport8 not present",
)


# ---------------------------------------------------------------------------
# Fixtures: uploaded modules under private names + the real airport8 cloud


@pytest.fixture(scope="module")
def up_gen():
    """The uploaded mesh_generator module, importable without registry clash."""
    import app.services.pipeline_stage as ps
    import app.services.mesh_generator as production  # noqa: F401 — register production FIRST

    spec = importlib.util.spec_from_file_location("_audit_up_mesh_generator", UPLOAD / "mesh_generator.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    saved = ps.STAGE_REGISTRY.pop("mesh_generation", None)
    try:
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            ps.STAGE_REGISTRY["mesh_generation"] = saved
    return mod


@pytest.fixture(scope="module")
def airport8_cloud():
    """Real fused+optimized cloud from data/test/airport8 (production path)."""
    import cv2

    from app.services.depth_fusion import DepthView, FusionParams, fuse_depth_views
    from app.services.pointcloud_optimizer import OptimizeParams, optimize_cloud

    intr = __import__("json").loads((AIRPORT8 / "intrinsics.json").read_text())
    K = np.array(
        [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float
    )

    def quat_R(qw, qx, qy, qz):
        n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )

    poses = {}
    for line in (AIRPORT8 / "poses.csv").read_text().splitlines()[1:]:
        p = line.strip().split(",")
        poses[int(p[0])] = [float(x) for x in p[1:11]]

    fid = 1044  # the frame carrying both depth_1044.png and its sidecar
    sc = __import__("json").loads((AIRPORT8 / "depth" / f"depth_{fid}.json").read_text())
    png = cv2.imread(str(AIRPORT8 / "depth" / f"depth_{fid}.png"), cv2.IMREAD_UNCHANGED)
    depth = sc["depth_min_m"] + (png.astype(np.float32) / 65535.0) * (
        sc["depth_max_m"] - sc["depth_min_m"]
    )
    x, y, z, qw, qx, qy, qz, _, _, _ = poses[fid]

    raw = fuse_depth_views(
        [DepthView(frame_id=f"v{fid}", depth=depth, rgb=None, K=K, R=quat_R(qw, qx, qy, qz), t=np.array([x, y, z]))],
        FusionParams(voxel_size=1.0, max_depth=1200.0),
    )
    opt = optimize_cloud(
        raw,
        OptimizeParams(voxel_size=1.0, ror_radius_m=2.0, viewpoint=(x, y, z)),
    )
    assert opt.cloud.n > 50_000
    return opt.cloud


# ---------------------------------------------------------------------------
# 1 — the uploaded radius cap collapses on a real cloud


def test_uploaded_radius_cap_collapses_real_cloud(up_gen, airport8_cloud):
    cloud = airport8_cloud
    # The uploaded cap: min_separation_m from the STATIC default voxel.
    from app.services.dense_diagnostics import detect_layers

    lay = detect_layers(
        np.asarray(cloud.xyz, dtype=np.float64),
        0.05,  # settings.dense.voxel_size default — exactly what the upload reads
        normals=np.asarray(cloud.normals, dtype=np.float64),
    )
    if lay.get("status") == "measured" and lay.get("min_separation_m"):
        cap = 0.8 * float(lay["min_separation_m"])
        spacing = float(np.median([0.93]))  # measured below; cap must be comparable
        assert cap < spacing * 0.5, "precondition: the upload's cap is sub-spacing on this real cloud"

    # Uploaded path on the real cloud: MeshNotPossible (no triangles — the
    # sub-spacing radius schedule pivots nothing).
    with pytest.raises(up_gen.MeshNotPossible):
        up_gen.generate_mesh(cloud, up_gen.MeshParams.from_settings())

    # Current production path on the same cloud meshes fine.
    from app.services.mesh_generator import MeshParams, generate_mesh

    mesh, stats = generate_mesh(cloud, MeshParams())
    assert mesh.m > 5_000, f"current path should mesh this cloud, got {mesh.m} faces"
    assert stats["method"] == "ball_pivot"


def test_current_mesh_sits_on_the_real_cloud(up_gen, airport8_cloud):
    """The sanity half of contract 1: current output is geometrically sound."""
    from scipy.spatial import cKDTree

    from app.services.mesh_generator import MeshParams, generate_mesh

    cloud = airport8_cloud
    mesh, _ = generate_mesh(cloud, MeshParams())
    tree = cKDTree(np.asarray(cloud.xyz))
    d, _ = tree.query(np.asarray(mesh.vertices)[:: max(1, mesh.n // 20_000)], k=1)
    assert float(np.percentile(d, 95)) < 1.0, "mesh must sit on the measured cloud"


# ---------------------------------------------------------------------------
# 2 — from_settings is a byte-identical no-op


def test_uploaded_from_settings_is_noop(up_gen):
    from app.config.settings import settings
    from app.services.mesh_generator import MeshParams as Cur

    up = up_gen.MeshParams.from_settings()
    cur = Cur()
    for f in (
        "method",
        "surface_max_edge_factor",
        "alpha_multiplier",
        "min_points",
        "poisson_depth",
        "ball_pivot_radii_factors",
        "patch_weld_factor",
        "prune_islands",
        "island_min_points",
        "island_max_diameter_factor",
    ):
        assert getattr(up, f) == getattr(cur, f), f"uploaded from_settings differs in {f}"


# ---------------------------------------------------------------------------
# 3 — phase-2 color passthrough misaligns under the (necessary) weld


@pytest.fixture(scope="module")
def gradient_plane_cloud():
    """Dense plane with a spatial color gradient + weld-able duplicate pairs.

    Colors encode position, so ANY index misalignment shows up as a wrong
    color; the duplicate pairs force merge_close_vertices to merge.
    """
    from app.services.pointcloud import PointCloud

    rng = np.random.default_rng(3)
    g = np.stack(np.meshgrid(np.arange(0, 20, 0.4), np.arange(0, 20, 0.4)), -1).reshape(-1, 2)
    xyz = np.column_stack([g, np.zeros(len(g))])
    # duplicate pairs 0.02 apart (well inside the r1/2 weld), offset colors
    dup = xyz[::37] + np.array([0.02, 0.0, 0.0])
    xyz = np.vstack([xyz, dup])
    rgb = np.column_stack(
        [np.clip(xyz[:, 0] * 100, 0, 255), np.clip(xyz[:, 1] * 100, 0, 255), np.full(len(xyz), 40)]
    ).astype(np.uint8)
    normals = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    return PointCloud(xyz=xyz, normals=normals, rgb=rgb)


def test_phase2_as_shipped_crashes_on_current_pointcloud(gradient_plane_cloud):
    """The uploaded passthrough references ``cloud.labels`` — a field the
    current ``PointCloud`` does not carry — so the code was never run
    against this codebase. Pinned as-is."""
    spec = importlib.util.spec_from_file_location("_audit_up_phase2", UPLOAD / "mesh_phase2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    with pytest.raises(AttributeError, match="labels"):
        mod._adaptive_ball_pivot_mesh(gradient_plane_cloud, mod.Phase2Params())


def test_phase2_color_passthrough_misaligns(gradient_plane_cloud):
    """Even with the missing field supplied (duck-typed shim), the
    passthrough indexes PRE-weld cloud colors with POST-weld vertex
    indices — merged vertices get the wrong original color."""
    spec = importlib.util.spec_from_file_location("_audit_up_phase2", UPLOAD / "mesh_phase2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    from scipy.spatial import cKDTree

    base = gradient_plane_cloud

    class _Shim:
        """The upload's expected input: PointCloud plus a labels field."""

        def __init__(self, pc):
            self._pc = pc
            self.labels = np.zeros(len(pc.xyz), dtype=np.int64)

        def __getattr__(self, name):
            return getattr(self._pc, name)

    cloud = _Shim(base)
    out, stats = mod._adaptive_ball_pivot_mesh(cloud, mod.Phase2Params())
    assert out.colors is not None and len(out.colors) == len(out.vertices)

    # Ground truth: each mesh vertex's color is the nearest ORIGINAL point's.
    tree = cKDTree(np.asarray(base.xyz))
    _, nearest = tree.query(np.asarray(out.vertices), k=1)
    expected = np.asarray(base.rgb)[nearest]
    mismatch = (np.asarray(out.colors) != expected).any(axis=1).mean()
    assert mismatch > 0.05, (
        f"expected the upload's passthrough to misalign colors, got {mismatch:.3f} mismatch rate"
    )


def test_current_mesh_colors_are_aligned(gradient_plane_cloud):
    """The contrast half: the current generator's nearest-cloud coloring aligns."""
    from scipy.spatial import cKDTree

    from app.services.mesh_generator import MeshParams, generate_mesh

    cloud = gradient_plane_cloud
    mesh, _ = generate_mesh(cloud, MeshParams())
    assert mesh.colors is not None
    tree = cKDTree(np.asarray(cloud.xyz))
    _, nearest = tree.query(np.asarray(mesh.vertices), k=1)
    expected = np.asarray(cloud.rgb)[nearest]
    mismatch = (np.asarray(mesh.colors) != expected).any(axis=1).mean()
    assert mismatch < 0.01, f"current coloring should be aligned, got {mismatch:.3f}"
