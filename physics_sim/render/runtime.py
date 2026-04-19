"""Composable render runtime (camera + color + rasterizer)."""

from __future__ import annotations

from typing import Optional

import torch

from physics_sim.render.camera import SimpleCamera
from physics_sim.render.interfaces import RenderRuntime
from physics_sim.render.sh_colorizer import ShColorizer
from physics_sim.render.gaussian_asset_loader import GaussianAssetLoader
from physics_sim.render.registries import (
    create_asset_loader,
    create_camera_builder,
    create_rasterizer,
)


class GaussianRenderRuntime(RenderRuntime):
    """Runtime composed from focused render-domain components."""

    def __init__(self, sh_degree: int = 3, raster_backend: str = "gsplat"):
        self._camera = create_camera_builder("default")
        self._colorizer = ShColorizer(sh_degree=sh_degree)
        self._rasterizer = create_rasterizer(raster_backend)

    def build_camera_external(
        self,
        *,
        camera_params: dict,
        center_view_world_space=None,
        observant_coordinates=None,
        current_frame: int = 0,
        source_axes=None,
    ) -> SimpleCamera:
        return self._camera.build_camera_external(
            camera_params=camera_params,
            center_view_world_space=center_view_world_space,
            observant_coordinates=observant_coordinates,
            current_frame=current_frame,
            source_axes=source_axes,
        )

    def build_camera_orbit(
        self,
        *,
        camera_params: dict,
        center_view_world_space,
        observant_coordinates,
        current_frame: int = 0,
    ) -> SimpleCamera:
        return self._camera.build_camera_orbit(
            camera_params=camera_params,
            center_view_world_space=center_view_world_space,
            observant_coordinates=observant_coordinates,
            current_frame=current_frame,
        )

    def build_camera_fixed(
        self,
        *,
        camera_params: dict,
    ) -> SimpleCamera:
        return self._camera.build_camera_fixed(camera_params=camera_params)

    def convert_sh(
        self,
        shs: torch.Tensor,
        camera: SimpleCamera,
        position: torch.Tensor,
        *,
        view_rotations: Optional[torch.Tensor] = None,
        alignment_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._colorizer.convert_sh(
            shs=shs,
            camera=camera,
            position=position,
            view_rotations=view_rotations,
            alignment_inv=alignment_inv,
        )

    def render(
        self,
        camera: SimpleCamera,
        means: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        bg_color: Optional[torch.Tensor] = None,
        *,
        cov6: Optional[torch.Tensor] = None,
        quats: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        return self._rasterizer.render(
            camera=camera,
            means=means,
            colors=colors,
            opacities=opacities,
            bg_color=bg_color,
            cov6=cov6,
            quats=quats,
            scales=scales,
        )


def create_scene_asset_loader(*, sh_degree: int) -> GaussianAssetLoader:
    """Factory function for scene-stage asset loader."""
    return create_asset_loader("ply", sh_degree=sh_degree)


def create_render_runtime(*, sh_degree: int, raster_backend: str) -> RenderRuntime:
    """Factory function for render-stage runtime."""
    return GaussianRenderRuntime(
        sh_degree=sh_degree,
        raster_backend=raster_backend,
    )
