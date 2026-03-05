"""
Plane fitting utilities for collider generation.

This module is intentionally lightweight (NumPy-only) so it can be used in
pipeline preprocessing without heavy physics/render dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PlaneFitResult:
    point: np.ndarray   # (3,) point on plane (centroid)
    normal: np.ndarray  # (3,) unit normal
    rms: float          # root-mean-square point-to-plane distance


def fit_plane_svd(
    points: np.ndarray,
    *,
    sample_max: Optional[int] = None,
    seed: int = 0,
    prefer_up: Optional[np.ndarray] = None,
) -> PlaneFitResult:
    """Fit a plane to 3D points using SVD/PCA.

    Args:
        points: (N,3) float array.
        sample_max: optional cap on points used (uniform random sample).
        seed: RNG seed for sampling.
        prefer_up: optional (3,) vector to stabilize normal direction.
                   If provided, the output normal will be flipped so
                   dot(normal, prefer_up) >= 0.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got shape {pts.shape}")
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    if sample_max is not None and pts.shape[0] > int(sample_max):
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=int(sample_max), replace=False)
        pts = pts[idx]

    centroid = pts.mean(axis=0)
    X = pts - centroid

    # PCA: normal is eigenvector of smallest eigenvalue of covariance.
    # Use SVD for numerical stability.
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[-1]
    n_norm = np.linalg.norm(normal)
    if not np.isfinite(n_norm) or n_norm < 1e-12:
        raise ValueError("Degenerate plane fit (normal has near-zero norm).")
    normal = normal / n_norm

    if prefer_up is not None:
        up = np.asarray(prefer_up, dtype=np.float64).reshape(3)
        up_norm = np.linalg.norm(up)
        if np.isfinite(up_norm) and up_norm > 1e-12:
            up = up / up_norm
            if float(np.dot(normal, up)) < 0.0:
                normal = -normal

    # RMS point-to-plane distance
    d = X @ normal
    rms = float(np.sqrt(np.mean(d * d)))

    return PlaneFitResult(
        point=centroid.astype(np.float32),
        normal=normal.astype(np.float32),
        rms=rms,
    )

