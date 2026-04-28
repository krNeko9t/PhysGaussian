"""Per-particle reference-volume estimators (V₀).

V₀,ᵢ is the volume of the material region represented by particle i in
its reference configuration.  It is an estimator of a local quantity
— the value depends on the particle cloud's density and the estimator
chosen, but the semantics are always the same.

Each stage that mutates the particle cloud (source load, filling, and in
the future downsample / driving-point selection) is responsible for
producing V₀ for its output.  Downstream consumers (MPM backend) must
not re-estimate.

Two estimators live here:

* :func:`estimate_volume_ellipsoid` — per-particle from Gaussian scales
  (treating each GS as a 1σ ellipsoid).  Appropriate when the cloud is
  sparse and per-particle scales still carry geometric meaning
  (typically right after PLY load).
* :func:`estimate_volume_occupancy` — ``dx³/cells_occupancy`` via
  Taichi.  Appropriate once the cloud has been densely resampled (e.g.
  after filling), where ellipsoid scales are no longer the best
  geometric proxy.
"""

from __future__ import annotations

import math

import torch

from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)

_FOUR_THIRDS_PI = 4.0 / 3.0 * math.pi


def estimate_volume_ellipsoid(scales: torch.Tensor) -> torch.Tensor:
    """Return per-particle V₀ = (4π/3)·s₁·s₂·s₃.

    ``scales`` is (N, 3) for 3DGS (1σ semi-axes directly).  For 2DGS,
    ``scales`` is (N, 2); we use a pancake heuristic with the shorter
    in-plane axis as the effective thickness::

        V = (4π/3) · s₁ · s₂ · min(s₁, s₂)

    Returns a (N,) float tensor on the same device as ``scales``.
    """
    if scales.ndim != 2:
        raise ValueError(f"scales must be 2D (N, D); got shape {tuple(scales.shape)}")
    s = scales.abs()
    if s.shape[-1] == 3:
        return _FOUR_THIRDS_PI * s[:, 0] * s[:, 1] * s[:, 2]
    if s.shape[-1] == 2:
        thickness = torch.minimum(s[:, 0], s[:, 1])
        return _FOUR_THIRDS_PI * s[:, 0] * s[:, 1] * thickness
    raise ValueError(
        f"scales must have last dim 2 (2DGS) or 3 (3DGS); got {s.shape[-1]}"
    )


def estimate_volume_occupancy(pos: torch.Tensor, n_grid: int) -> torch.Tensor:
    """Return per-particle V₀ = ``dx³ / cells_occupancy``.

    Builds a uniform grid spanning the cloud's AABB (largest extent
    divided by ``n_grid``) and assigns each particle ``cell_volume /
    particles_in_cell``.  ΣV₀ ≈ (occupied-cell count) · dx³, which
    approximates the body volume when the cloud is densely filled.
    """
    from physics_sim.preprocessing.particle_filling import get_particle_volume

    lo = pos.min(dim=0).values
    hi = pos.max(dim=0).values
    extent = (hi - lo).max().item()
    if extent < 1e-8:
        extent = 1.0
    dx = extent / max(n_grid, 1)
    vol = get_particle_volume(pos - lo, grid_n=n_grid, grid_dx=dx)
    return vol.to(pos.device)


def sanity_check_volumes(
    volumes: torch.Tensor,
    pos: torch.Tensor,
    *,
    name: str,
    tol: float = 3.0,
) -> None:
    """Warn if ΣV₀ is off from a crude bbox-volume estimate by > ``tol``×.

    The bbox reference (product of extents) is a loose upper bound; a
    ratio outside ``[1/tol, tol]`` usually means the estimator picked
    is mismatched to the cloud (e.g. ellipsoid on a densely-filled
    cloud under-counts because ellipsoids overlap).

    This is intentionally a warning, not an assertion — the estimators
    are approximate and many scenes legitimately deviate.
    """
    if volumes.numel() == 0:
        return
    total = float(volumes.sum().item())
    lo = pos.min(dim=0).values
    hi = pos.max(dim=0).values
    bbox = float((hi - lo).prod().item())
    if bbox <= 0.0 or total <= 0.0:
        LOGGER.warning(
            "[volumes:%s] degenerate: ΣV₀=%.3e bbox_vol=%.3e (N=%d)",
            name, total, bbox, volumes.numel(),
        )
        return
    ratio = total / bbox
    if ratio > tol or ratio < 1.0 / tol:
        LOGGER.warning(
            "[volumes:%s] ΣV₀=%.3e vs bbox=%.3e (ratio=%.3f, N=%d) — "
            "estimator may be mismatched to the cloud",
            name, total, bbox, ratio, volumes.numel(),
        )
