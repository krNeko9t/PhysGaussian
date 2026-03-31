"""Rasterization backend using diff_gaussian_rasterization (Inria)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from physics_sim.renderer.backend_base import RasterBackend

if TYPE_CHECKING:
    from physics_sim.renderer.gs_renderer import SimpleCamera


class DiffRastBackend(RasterBackend):
    """Wraps the original ``diff_gaussian_rasterization`` CUDA extension."""

    def __init__(self, sh_degree: int = 3):
        self.sh_degree = sh_degree

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
        if cov6 is None:
            raise ValueError(
                "DiffRastBackend requires cov6.  Native 2DGS (quats+scales) "
                "is not supported — use the gsplat backend instead."
            )

        try:
            from diff_gaussian_rasterization import (
                GaussianRasterizationSettings,
                GaussianRasterizer,
            )
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "diff_gaussian_rasterization is required for the 'diffrast' backend. "
                "Install it or use --raster_backend gsplat."
            ) from e

        if bg_color is None:
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)

        settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=self.sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)

        screen_points = torch.zeros(
            (means.shape[0], 3), device=means.device, requires_grad=False
        )
        rendering, radii = rasterizer(
            means3D=means,
            means2D=screen_points,
            shs=None,
            colors_precomp=colors,
            opacities=opacities,
            scales=None,
            rotations=None,
            cov3D_precomp=cov6,
        )
        return rendering, {"radii": radii}
