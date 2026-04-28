"""Convert quantile-based filling thresholds to absolute grid_density cutoffs."""

from __future__ import annotations

import numpy as np


def resolve_absolute_thresholds_from_quantiles(
    grid_counts: np.ndarray,
    grid_density: np.ndarray,
    density_quantile: float,
    search_quantile: float,
) -> tuple[float, float]:
    """Map dimensionless quantiles to absolute ``grid_density`` thresholds.

    Statistics use voxels with ``grid_counts > 0`` (particle-center occupancy
    after densify). If that set is empty, fall back to voxels with
    ``grid_density > 0``.

    Args:
        grid_counts: Integer occupancies, shape ``(n, n, n)``.
        grid_density: Accumulated density field, same shape, float.
        density_quantile: ``q`` in ``(0, 1)`` for dense-fill cutoff.
        search_quantile: ``q`` in ``(0, 1)`` for internal ray collision cutoff.

    Returns:
        ``(density_thres, search_thres)`` in original ``grid_density`` units.

    Raises:
        ValueError: No voxels to sample after both masks.
    """
    if grid_counts.shape != grid_density.shape:
        raise ValueError(
            f"grid_counts shape {grid_counts.shape} != grid_density shape {grid_density.shape}"
        )

    mask = grid_counts > 0
    samples = grid_density[mask]

    if samples.size == 0:
        mask = grid_density > 0.0
        samples = grid_density[mask]

    if samples.size == 0:
        raise ValueError(
            "[particle_filling] quantile calibration: no occupied or positive-density voxels"
        )

    density_thres, search_thres = np.quantile(
        samples.astype(np.float64, copy=False),
        [density_quantile, search_quantile],
    )
    return float(density_thres), float(search_thres)
