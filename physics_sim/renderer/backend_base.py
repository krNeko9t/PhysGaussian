"""Abstract base class for 3DGS rasterization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physics_sim.render.camera import SimpleCamera


class RasterBackend(ABC):
    """Rasterization backend interface.

    Every backend receives the same canonical inputs and must return a
    rendered image in ``[3, H, W]`` layout together with a metadata dict.

    Callers must pass **exactly one** of the following shape representations
    as keyword arguments:

    * ``cov6`` — (N, 6) upper-triangle covariance (3DGS path)
    * ``quats`` **and** ``scales`` — (N, 4) quaternion + (N, 2|3) scales (2DGS path)
    """

    @abstractmethod
    def render(
        self,
        camera: SimpleCamera,
        means: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        bg_color: torch.Tensor | None = None,
        *,
        cov6: torch.Tensor | None = None,
        quats: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Rasterize one frame.

        Args:
            camera: Camera with intrinsics / extrinsics.
            means: (N, 3) Gaussian centres in world space.
            colors: (N, 3) pre-computed RGB colours.
            opacities: (N, 1) per-Gaussian opacity.
            bg_color: (3,) background colour, default black.
            cov6: (N, 6) upper-triangle covariance (3DGS).
            quats: (N, 4) wxyz quaternions (2DGS).
            scales: (N, 2|3) activated scales (2DGS).

        Returns:
            rendered: (3, H, W) float32 rendered image.
            meta: backend-specific metadata (e.g. radii, alpha map).
        """
        ...


def create_raster_backend(name: str, sh_degree: int = 3) -> RasterBackend:
    """Factory: instantiate a raster backend by name.

    Args:
        name: ``"gsplat"`` or ``"diffrast"``.
        sh_degree: SH degree forwarded to backends that need it.
    """
    if name == "gsplat":
        from physics_sim.render.rasterizers.gsplat import GsplatBackend

        return GsplatBackend()
    if name == "diffrast":
        from physics_sim.render.rasterizers.diffrast import DiffRastBackend

        return DiffRastBackend(sh_degree=sh_degree)
    raise ValueError(f"Unknown raster backend: {name!r}. Choose 'gsplat' or 'diffrast'.")
