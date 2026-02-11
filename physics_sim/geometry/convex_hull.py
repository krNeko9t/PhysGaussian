"""Convex hull computation from point clouds.

Face winding is guaranteed to produce **outward-facing normals** (CCW
when viewed from outside), which is required by Newton's collision
pipeline.
"""

import numpy as np
from scipy.spatial import ConvexHull


def compute_convex_hull(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the convex hull of a 3D point cloud.

    The returned faces have **outward-facing normals** (verified against
    ``hull.equations`` from scipy, which are guaranteed outward).

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

    # hull.simplices: (F, 3) face indices — winding order is ARBITRARY.
    # hull.equations: (F, 4) plane equations [nx, ny, nz, d] with normals
    #                 guaranteed to point OUTWARD.
    # We fix the winding so that cross(e1, e2) aligns with the outward normal.
    simplices = hull.simplices.copy()
    for i in range(simplices.shape[0]):
        v0, v1, v2 = positions[simplices[i]]
        face_normal = np.cross(v1 - v0, v2 - v0)
        outward_normal = hull.equations[i, :3]
        if np.dot(face_normal, outward_normal) < 0:
            # Flip winding: swap v1 and v2
            simplices[i, 1], simplices[i, 2] = (
                simplices[i, 2],
                simplices[i, 1],
            )

    # Remap to compact vertex array
    unique_idx = hull.vertices
    vertices = positions[unique_idx]
    idx_map = np.full(positions.shape[0], -1, dtype=np.int32)
    idx_map[unique_idx] = np.arange(len(unique_idx), dtype=np.int32)
    faces = idx_map[simplices]

    return vertices.astype(np.float32), faces.astype(np.int32)
