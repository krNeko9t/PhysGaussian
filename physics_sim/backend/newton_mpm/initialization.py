"""Initialization helpers for NewtonMPM backend."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM


@dataclass
class InitializationResult:
    builder: newton.ModelBuilder
    voxel_size: float
    bbox_lo: list[float]
    bbox_hi: list[float]
    grid_lim: float
    cov_flat: np.ndarray
    volumes: np.ndarray


def build_mpm_initialization(
    *,
    positions,
    volumes,
    covariances,
    n_grid: int,
) -> InitializationResult:
    """Build builder and derived arrays from particle tensors."""
    pos_np = positions.detach().cpu().numpy().astype(np.float32)
    vol_np = volumes.detach().cpu().numpy().astype(np.float32)
    cov_np = covariances.detach().cpu().numpy().astype(np.float32).reshape(-1)
    n_particles = int(pos_np.shape[0])

    lo = pos_np.min(axis=0)
    hi = pos_np.max(axis=0)
    max_extent = float((hi - lo).max())
    if max_extent < 1e-8:
        max_extent = 1.0
    voxel_size = max_extent / max(n_grid, 1)

    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)

    mass_np = vol_np.copy()
    radius_np = np.cbrt(vol_np * 3.0 / (4.0 * math.pi)).astype(np.float32)
    radius_np = np.maximum(radius_np, 1e-6)

    pos_list = [
        wp.vec3(float(pos_np[i, 0]), float(pos_np[i, 1]), float(pos_np[i, 2]))
        for i in range(n_particles)
    ]
    builder.add_particles(
        pos=pos_list,
        vel=[wp.vec3(0.0, 0.0, 0.0)] * n_particles,
        mass=[float(mass_np[i]) for i in range(n_particles)],
        radius=[float(radius_np[i]) for i in range(n_particles)],
        flags=[int(newton.ParticleFlags.ACTIVE)] * n_particles,
    )

    return InitializationResult(
        builder=builder,
        voxel_size=voxel_size,
        bbox_lo=lo.tolist(),
        bbox_hi=hi.tolist(),
        grid_lim=max_extent,
        cov_flat=cov_np,
        volumes=vol_np,
    )
