"""Stable protocol boundaries for scene and rendering pipeline."""

from __future__ import annotations

from typing import Optional, Protocol

import torch

from physics_sim.render.camera import SimpleCamera
from physics_sim.render.types import GaussianAsset


class SceneAssetLoader(Protocol):
    """Load gaussian asset tensors from persistent storage."""

    def load_ply(self, ply_path: str) -> GaussianAsset:
        ...


class CameraBuilder(Protocol):
    """Build a runtime camera from scene and config state."""

    def build_camera_external(
        self,
        *,
        camera_params: dict,
        center_view_world_space=None,
        observant_coordinates=None,
        current_frame: int = 0,
        source_axes=None,
    ) -> SimpleCamera:
        ...

    def build_camera_orbit(
        self,
        *,
        camera_params: dict,
        center_view_world_space,
        observant_coordinates,
        current_frame: int = 0,
    ) -> SimpleCamera:
        ...

    def build_camera_fixed(
        self,
        *,
        camera_params: dict,
    ) -> SimpleCamera:
        ...


class ColorComputer(Protocol):
    """Convert per-Gaussian SH features into precomputed RGB colors."""

    def convert_sh(
        self,
        shs: torch.Tensor,
        camera: SimpleCamera,
        position: torch.Tensor,
        *,
        view_rotations: Optional[torch.Tensor] = None,
        alignment_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...


class Rasterizer(Protocol):
    """Render gaussian tensors into an image frame."""

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
        ...


class RenderRuntime(CameraBuilder, ColorComputer, Rasterizer, Protocol):
    """Unified rendering runtime contract used by simulation loop."""

