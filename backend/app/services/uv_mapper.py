"""UV atlas layout: one texture tile per face (row-major grid).

Each face is mapped onto its own square tile so any triangle can carry a
bilinearly sampled patch from the view that sees it best. Tiles are padded
to avoid bleeding between neighbours. The layout is purely geometric and
independent of any source image, which keeps UV generation deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AtlasLayout:
    """A rectangular texture atlas with per-face tiles.

    ``uv_px[i]`` are the pixel coordinates (x, y) of face *i*'s three corners
    within the atlas image; ``uv_norm[i]`` the same normalised to [0, 1].
    """

    width: int
    height: int
    cols: int
    rows: int
    cell: int
    padding: int
    uv_px: np.ndarray  # (M, 3, 2) float — pixel corner coordinates
    uv_norm: np.ndarray  # (M, 3, 2) float — normalised corner coordinates

    @property
    def m(self) -> int:
        return len(self.uv_px)


def build_atlas_layout(
    face_count: int,
    atlas_width: int = 8192,
    padding: int = 2,
    max_cell: int | None = None,
) -> AtlasLayout:
    """Layout *face_count* triangle tiles in a square-ish atlas grid."""
    if face_count < 1:
        raise ValueError("cannot build an atlas for an empty mesh")
    cols = max(1, int(np.ceil(np.sqrt(face_count))))
    rows = max(1, int(np.ceil(face_count / cols)))
    cell = (atlas_width - (cols + 1) * padding) // cols
    if cell < 3:
        raise ValueError(f"atlas width {atlas_width} too small for {face_count} faces")
    if max_cell is not None:
        cell = min(cell, int(max_cell))
    height = (rows + 1) * padding + rows * cell
    width = (cols + 1) * padding + cols * cell

    uv_px = np.empty((face_count, 3, 2), dtype=np.float64)
    for i in range(face_count):
        col = i % cols
        row = i // cols
        x0 = padding + col * (cell + padding)
        y0 = padding + row * (cell + padding)
        # Triangle corners inside the tile: (0,0) (1,0) (0,1) of the cell.
        uv_px[i] = [[x0, y0], [x0 + cell, y0], [x0, y0 + cell]]
    return AtlasLayout(
        width=width, height=height, cols=cols, rows=rows,
        cell=cell, padding=padding, uv_px=uv_px,
        uv_norm=uv_px / np.array([width, height]),
    )


def face_uv(layout: AtlasLayout, face_id: int) -> np.ndarray:
    """Pixel corner coordinates of a single face's tile."""
    return layout.uv_px[face_id]
