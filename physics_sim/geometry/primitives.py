"""Primitive shape fitting from 3D point clouds.

Fits simple collision proxies (OBB, ellipsoid) to point clouds using PCA.
These shapes are analytically handled by physics engines — no triangle
meshes, no topology issues, always stable and fast.
"""

from __future__ import annotations

import numpy as np


def fit_obb(positions: np.ndarray) -> dict:
    """Fit an Oriented Bounding Box (OBB) to a 3D point cloud via PCA.

    Args:
        positions: (N, 3) point positions.

    Returns:
        dict with keys:
            ``half_extents``: (3,) float32 — half-sizes along principal axes
                              (hx, hy, hz), sorted largest-first.
            ``axes``:         (3, 3) float32 — rotation matrix whose rows are
                              the principal axes (columns = basis vectors).
            ``center``:       (3,) float32 — centroid of the point cloud.
    """
    center = positions.mean(axis=0)
    centered = positions - center

    # PCA via covariance eigen-decomposition
    cov = np.cov(centered, rowvar=False)  # (3, 3)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh returns ascending order; flip to descending (largest axis first)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]  # columns = principal axes

    # Project points onto principal axes to get tight extents
    projected = centered @ eigvecs  # (N, 3)
    lo = projected.min(axis=0)
    hi = projected.max(axis=0)
    half_extents = (hi - lo) / 2.0

    # Adjust center to the OBB center (not centroid)
    obb_center_local = (hi + lo) / 2.0  # in PCA space
    center = center + eigvecs @ obb_center_local

    # Ensure right-handed coordinate system
    axes = eigvecs.T  # rows = principal axes
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]

    print(
        f"[OBB] half_extents=({half_extents[0]:.4f}, "
        f"{half_extents[1]:.4f}, {half_extents[2]:.4f})"
    )

    return {
        "half_extents": half_extents.astype(np.float32),
        "axes": axes.astype(np.float32),
        "center": center.astype(np.float32),
    }


def fit_ellipsoid(positions: np.ndarray) -> dict:
    """Fit an axis-aligned ellipsoid to a 3D point cloud via PCA.

    The ellipsoid semi-axes (a, b, c) are chosen so that the ellipsoid
    tightly encloses the point cloud along each principal direction.

    Args:
        positions: (N, 3) point positions.

    Returns:
        dict with keys:
            ``semi_axes``:  (3,) float32 — semi-axis lengths (a, b, c),
                            sorted largest-first.
            ``axes``:       (3, 3) float32 — rotation matrix (rows = axes).
            ``center``:     (3,) float32 — centroid of the point cloud.
    """
    center = positions.mean(axis=0)
    centered = positions - center

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Project to get actual extents (not just std-dev)
    projected = centered @ eigvecs
    lo = projected.min(axis=0)
    hi = projected.max(axis=0)
    semi_axes = (hi - lo) / 2.0

    # Adjust center
    ellipsoid_center_local = (hi + lo) / 2.0
    center = center + eigvecs @ ellipsoid_center_local

    axes = eigvecs.T
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]

    print(
        f"[Ellipsoid] semi_axes=({semi_axes[0]:.4f}, "
        f"{semi_axes[1]:.4f}, {semi_axes[2]:.4f})"
    )

    return {
        "semi_axes": semi_axes.astype(np.float32),
        "axes": axes.astype(np.float32),
        "center": center.astype(np.float32),
    }
