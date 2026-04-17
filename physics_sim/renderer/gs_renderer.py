"""Deprecated compatibility facade for legacy GaussianRenderer imports."""

from __future__ import annotations

import warnings
from typing import Optional

import torch

from physics_sim.render.camera import SimpleCamera
from physics_sim.render.gaussian_asset_loader import GaussianAssetLoader
from physics_sim.render.runtime import GaussianRenderRuntime


class GaussianRenderer:
    """Deprecated adapter. New code should use render-domain modules directly."""

    def __init__(self, sh_degree: int = 3, raster_backend: str = "gsplat"):
        warnings.warn(
            "GaussianRenderer is deprecated. "
            "Use GaussianAssetLoader + GaussianRenderRuntime instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._loader = GaussianAssetLoader(sh_degree=sh_degree)
        self._runtime = GaussianRenderRuntime(
            sh_degree=sh_degree,
            raster_backend=raster_backend,
        )

    def load_ply(self, ply_path: str) -> dict:
        return self._loader.load_ply(ply_path).to_dict()

    def convert_sh(
        self,
        shs: torch.Tensor,
        camera: SimpleCamera,
        position: torch.Tensor,
        rotation: Optional[torch.Tensor] = None,
        alignment_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._runtime.convert_sh(
            shs=shs,
            camera=camera,
            position=position,
            rotation=rotation,
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
    ) -> tuple:
        return self._runtime.render(
            camera=camera,
            means=means,
            colors=colors,
            opacities=opacities,
            bg_color=bg_color,
            cov6=cov6,
            quats=quats,
            scales=scales,
        )

    def build_camera_from_json(
        self,
        cameras_json_path: str,
        camera_params: dict,
        center_view_world_space=None,
        observant_coordinates=None,
        current_frame: int = 0,
        source_axes=None,
    ) -> SimpleCamera:
        return self._runtime.build_camera_from_json(
            cameras_json_path=cameras_json_path,
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
        return self._runtime.build_camera_orbit(
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
        return self._runtime.build_camera_fixed(camera_params=camera_params)
