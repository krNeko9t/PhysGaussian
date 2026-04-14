"""Scene object data structure for the physics simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class SceneObject:
    """Self-contained representation of one object in the scene.

    All GS tensor fields are already preprocessed (opacity-filtered,
    axis-permuted, rotated into the backend's operating space) by the
    time this object is constructed by the assembler.
    """

    name: str
    mode: str  # "simulate" | "render_only" | "collider_only"

    # GS data (preprocessed — rotated world space for simulate objects,
    # world space for static/collider objects).
    positions: torch.Tensor       # (N, 3)
    covariances: torch.Tensor     # (N, 6) upper-triangle
    opacities: torch.Tensor       # (N, 1)
    shs: torch.Tensor             # (N, C, 3)
    quats: torch.Tensor           # (N, 4) wxyz
    scales: torch.Tensor          # (N, 2|3)

    # Material params for this object (top-level defaults merged with
    # per-object overrides).
    material: dict = field(default_factory=dict)

    # Optional position offset applied in rotated world space.
    position_offset: Optional[list[float]] = None

    # Particle filling config (used by some backends).
    particle_filling: Optional[dict] = None

    # Collision proxy definition (used by collider_only and some
    # simulate backends like newton_rigid).
    collider: Optional[dict] = None

    # PLY type detected at load time.
    gs_type: str = "3dgs"

    @property
    def n_particles(self) -> int:
        return self.positions.shape[0]
