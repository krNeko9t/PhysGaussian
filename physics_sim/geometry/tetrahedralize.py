"""Surface mesh → tetrahedral mesh via TetGen.

Takes a closed (watertight) triangle surface mesh and produces a
volumetric tetrahedral mesh suitable for FEM simulation (Newton
``add_soft_mesh``).

If the input mesh is *not* watertight, the function attempts to repair
it (hole-filling via Open3D) before tetrahedralizing.  If repair fails,
a convex-hull fallback is used to guarantee a valid closed surface.

Requires: ``tetgen >= 0.6``  (``pip install tetgen``)
"""

from __future__ import annotations

import numpy as np


def tetrahedralize(
    verts: np.ndarray,
    faces: np.ndarray,
    max_volume: float | None = None,
    quality: float = 1.5,
    ensure_watertight: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a surface mesh to a tetrahedral mesh.

    Args:
        verts:  (V, 3) surface vertices (float32/float64).
        faces:  (F, 3) triangle face indices (int32).
        max_volume: Maximum tetrahedron volume.  Smaller → denser mesh.
                    ``None`` → TetGen decides automatically.
        quality:    Minimum radius-to-edge ratio (TetGen ``-q`` switch).
                    Default 1.5 is a good balance between quality and
                    element count.
        ensure_watertight: If ``True`` (default), attempt to close holes
                    in a non-watertight mesh before tetrahedralization.

    Returns:
        tet_verts: (V', 3) float32 — tet mesh vertices (may include
                   internal Steiner points added by TetGen).
        tet_cells: (T, 4)  int32   — tetrahedron vertex indices.

    Raises:
        ImportError: If ``tetgen`` is not installed.
        RuntimeError: If tetrahedralization fails even after fallbacks.
    """
    try:
        import tetgen as _tetgen
    except ImportError:
        raise ImportError(
            "TetGen is required for tetrahedralization. "
            "Install with:  pip install tetgen"
        )

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    # ── Optionally repair non-watertight meshes ──────────────────────
    if ensure_watertight:
        verts, faces = _ensure_watertight(verts, faces)

    # ── Run TetGen ───────────────────────────────────────────────────
    tet = _tetgen.TetGen(verts, faces)

    # Build switch string:  p = PLC input,  q = quality
    switches = f"pq{quality}"
    if max_volume is not None:
        switches += f"a{max_volume}"

    try:
        tet.tetrahedralize(switches=switches)
    except RuntimeError as e:
        raise RuntimeError(
            f"TetGen failed: {e}.  The surface mesh may have "
            "self-intersections or degenerate triangles."
        ) from e

    tet_verts = tet.node   # (V', 3)
    tet_cells = tet.elem   # (T, 4)

    print(
        f"[Tetrahedralize] {verts.shape[0]} surface verts, "
        f"{faces.shape[0]} faces → "
        f"{tet_verts.shape[0]} tet verts, {tet_cells.shape[0]} tets"
    )

    return tet_verts.astype(np.float32), tet_cells.astype(np.int32)


# ── Internal helpers ─────────────────────────────────────────────────


def _ensure_watertight(
    verts: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Try to close holes in the mesh.  Falls back to convex hull."""
    try:
        import open3d as o3d
    except ImportError:
        # Can't repair without Open3D; hope for the best.
        return verts, faces

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    if mesh.is_watertight():
        return verts, faces

    # Attempt repair
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.orient_triangles()

    if mesh.is_watertight():
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        print("[Tetrahedralize] Repaired mesh → watertight")
        return v, f

    # Last resort: convex hull (always watertight)
    from physics_sim.geometry.convex_hull import compute_convex_hull

    hull_v, hull_f = compute_convex_hull(verts.astype(np.float32))
    print(
        "[Tetrahedralize] WARNING: mesh not watertight after repair. "
        "Using convex hull fallback."
    )
    return hull_v.astype(np.float64), hull_f
