"""Alpha Shape surface reconstruction from 3D point clouds.

An Alpha Shape is a generalization of the convex hull: with a finite
*alpha* parameter it "shrinks" the hull to follow concavities, producing
a much tighter surface mesh than a plain convex hull.  This makes it
ideal as a collision proxy for organic shapes (animals, food, etc.)
where the convex hull fills in too much empty space.

Requires: ``open3d >= 0.15``
"""

from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]


def compute_alpha_shape(
    positions: np.ndarray,
    alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Alpha Shape mesh of a 3D point cloud.

    Args:
        positions: (N, 3) point positions (float32 or float64).
        alpha:     Alpha parameter controlling the tightness of the mesh.
                   - *Smaller* alpha → tighter mesh (more concavities).
                   - *Larger*  alpha → closer to convex hull.
                   - ``None`` → auto-estimate from point cloud extent.

    Returns:
        vertices: (V, 3) float32 mesh vertices.
        faces:    (F, 3) int32   triangle face indices (into *vertices*).

    Raises:
        ImportError: If Open3D is not installed.
        ValueError:  If the result mesh has zero triangles (alpha too small).
    """
    if o3d is None:
        raise ImportError(
            "Open3D is required for Alpha Shape reconstruction. "
            "Install it with: pip install open3d"
        )

    if positions.shape[0] < 4:
        raise ValueError(
            f"Need >= 4 points for alpha shape, got {positions.shape[0]}"
        )

    # ── Auto-estimate alpha if not provided ─────────────────────────
    if alpha is None:
        # Heuristic: use ~10% of the bounding-box diagonal.
        # This works well for 3DGS objects that are roughly unit-scale.
        extent = positions.max(axis=0) - positions.min(axis=0)
        diag = np.linalg.norm(extent)
        alpha = float(diag * 0.1)
        # Clamp to reasonable range
        alpha = max(alpha, 0.01)

    # ── Build Open3D point cloud and run alpha shape ────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=alpha
    )

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if faces.shape[0] == 0:
        raise ValueError(
            f"Alpha shape produced 0 triangles (alpha={alpha:.4f} is too "
            f"small for this point cloud). Try a larger alpha value."
        )

    # ── Clean up the mesh ───────────────────────────────────────────
    # Remove degenerate / duplicate triangles; ensure consistent winding.
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    print(
        f"[AlphaShape] alpha={alpha:.4f} → "
        f"{verts.shape[0]} verts, {faces.shape[0]} faces "
        f"(watertight={mesh.is_watertight()})"
    )

    return verts.astype(np.float32), faces.astype(np.int32)
