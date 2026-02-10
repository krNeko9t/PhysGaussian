"""Convex hull computation from point clouds.

Phase 1 of the geometry extraction pipeline. Later phases will add:
  - surface_mesh.py   (Poisson / Alpha Shape reconstruction)
  - tetrahedralize.py (surface mesh → tetrahedral mesh for FEM)
  - sdf.py            (voxel density → signed distance field)
"""

import numpy as np
from scipy.spatial import ConvexHull


def compute_convex_hull(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the convex hull of a 3D point cloud.

    Args:
        positions: (N, 3) point positions.

    Returns:
        vertices: (V, 3) float32 convex hull vertices.
        faces:    (F, 3) int32 triangle face indices (into *vertices*).

    Raises:
        ValueError: If the point cloud has fewer than 4 non-coplanar points.
    """
    if positions.shape[0] < 4:
        raise ValueError(
            f"Need at least 4 points for a convex hull, got {positions.shape[0]}"
        )

    hull = ConvexHull(positions)

    # hull.vertices: sorted array of unique vertex indices (into *positions*)
    # hull.simplices: (F, 3) face indices (also into *positions*)
    unique_idx = hull.vertices
    vertices = positions[unique_idx]

    # Remap face indices from the original array to the compact vertex array
    idx_map = np.full(positions.shape[0], -1, dtype=np.int32)
    idx_map[unique_idx] = np.arange(len(unique_idx), dtype=np.int32)
    faces = idx_map[hull.simplices]

    return vertices.astype(np.float32), faces.astype(np.int32)
