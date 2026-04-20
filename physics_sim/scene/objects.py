"""Scene object data structure for the physics simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from physics_sim.config.models import (
    ColliderConfig,
    FillingConfig,
    MaterialSpec,
)


@dataclass
class SceneObject:
    """Self-contained representation of one object in the scene.

    All GS tensor fields are already preprocessed (opacity-filtered,
    axis-permuted, rotated into the backend's operating space) by the
    time this object is constructed by the assembler.
    """

    name: str
    role: str  # "dynamic" | "render_only" | "collider_only"

    positions: torch.Tensor       # (N, 3)
    covariances: torch.Tensor     # (N, 6) upper-triangle
    opacities: torch.Tensor       # (N, 1)
    shs: torch.Tensor             # (N, C, 3)
    quats: torch.Tensor           # (N, 4) wxyz
    scales: torch.Tensor          # (N, 2|3)

    material: Optional[MaterialSpec] = None
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    particle_filling: Optional[FillingConfig] = None
    collider: Optional[ColliderConfig] = None

    gs_type: str = "3dgs"

    @property
    def n_particles(self) -> int:
        return self.positions.shape[0]

