"""Abstract base class and factory for rasterization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physics_sim.render.camera import SimpleCamera


class RasterBackend(ABC):
    """Rasterization backend interface."""

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
        """Rasterize one frame."""
        ...


def create_raster_backend(name: str) -> RasterBackend:
    """Factory: instantiate a raster backend by name."""
    if name == "gsplat":
        from physics_sim.render.rasterizers.gsplat import GsplatBackend

        return GsplatBackend()
    if name == "diffrast":
        from physics_sim.render.rasterizers.diffrast import DiffRastBackend

        return DiffRastBackend()
    raise ValueError(f"Unknown raster backend: {name!r}. Choose 'gsplat' or 'diffrast'.")
