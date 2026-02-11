"""Alpha Shape surface reconstruction from 3D point clouds.

An Alpha Shape is a generalization of the convex hull: with a finite
*alpha* parameter it "shrinks" the hull to follow concavities, producing
a much tighter surface mesh than a plain convex hull.  This makes it
ideal as a collision proxy for organic shapes (animals, food, etc.)
where the convex hull fills in too much empty space.

The output mesh is **decimated** to a physics-friendly triangle count
(default 300) so that collision detection remains fast and XPBD can
converge within a reasonable number of iterations.

Requires: ``open3d >= 0.15``
"""

from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]

# Default target triangle count for the physics collision mesh.
# Game engines typically use 100–500 triangles for physics proxies.
_DEFAULT_MAX_TRIANGLES = 300


def compute_alpha_shape(
    positions: np.ndarray,
    alpha: float | None = None,
    max_triangles: int = _DEFAULT_MAX_TRIANGLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simplified Alpha Shape mesh from a 3D point cloud.

    The raw alpha shape is decimated to *max_triangles* so that it can
    serve as a lightweight collision proxy for physics engines.

    Args:
        positions:     (N, 3) point positions (float32 or float64).
        alpha:         Alpha parameter controlling the tightness.
                       - *Smaller* → tighter mesh (more concavities).
                       - *Larger*  → closer to convex hull.
                       - ``None``  → auto-estimate from point cloud extent.
        max_triangles: Maximum number of triangles in the output mesh.
                       The raw alpha shape is decimated to this count.
                       Set to 0 or negative to skip decimation.

    Returns:
        vertices: (V, 3) float32 mesh vertices.
        faces:    (F, 3) int32   triangle face indices (into *vertices*).

    Raises:
        ImportError: If Open3D is not installed.
        ValueError:  If the result mesh has zero triangles.
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
        extent = positions.max(axis=0) - positions.min(axis=0)
        diag = np.linalg.norm(extent)
        alpha = float(diag * 0.1)
        alpha = max(alpha, 0.01)

    # ── Build Open3D point cloud and run alpha shape ────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=alpha
    )

    if len(mesh.triangles) == 0:
        raise ValueError(
            f"Alpha shape produced 0 triangles (alpha={alpha:.4f} is too "
            f"small for this point cloud). Try a larger alpha value."
        )

    raw_faces = len(mesh.triangles)

    # ── Clean up topology ───────────────────────────────────────────
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    # ── Decimate to physics-friendly triangle count ─────────────────
    if max_triangles > 0 and len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=max_triangles
        )
        # Clean up again after decimation
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()

    # ── Ensure consistent outward-facing normals ────────────────────
    # orient_triangles() makes face winding consistent across the mesh.
    # Without this, Newton gets mixed inward/outward normals → explosion.
    mesh.orient_triangles()
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    print(
        f"[AlphaShape] alpha={alpha:.4f}, raw={raw_faces} faces → "
        f"decimated to {verts.shape[0]} verts, {faces.shape[0]} faces "
        f"(watertight={mesh.is_watertight()})"
    )

    return verts.astype(np.float32), faces.astype(np.int32)
