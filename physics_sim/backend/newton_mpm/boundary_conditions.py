"""Boundary-condition parsing and per-step application for Newton MPM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import newton
import warp as wp

from physics_sim.backend.newton_common.boundary import (
    build_bounding_box_planes,
    surface_plane_from_bc,
)
from physics_sim.config.models import (
    BoundaryCondition,
    BoundingBox,
    ReleaseParticlesSequentially,
    SurfaceCollider,
)
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class BoundaryConditionRuntime:
    """Runtime BC state for NewtonMPM backend.

    MPM only consumes the runtime-path BCs in this dataclass
    (release-style).  Shape-path BCs (surface_collider, bounding_box)
    are registered directly into the ``ModelBuilder``.
    """

    release_bcs: list[ReleaseParticlesSequentially] = field(default_factory=list)

    def apply_velocity_bcs(
        self,
        *,
        state: Any,
        time_now: float,
        dt: float,
        device: str,
    ) -> None:
        # Placeholder for future per-step velocity BCs — currently no-op.
        return

    def apply_impulse_bcs(
        self,
        *,
        state: Any,
        particle_mass: Any,
        time_now: float,
        dt: float,
        device: str,
    ) -> None:
        # Placeholder for future per-step impulse BCs — currently no-op.
        return

    def log_unimplemented(self) -> None:
        if self.release_bcs:
            LOGGER.warning(
                "[NewtonMPM] release_particles_sequentially currently degrades to no-op count=%s",
                len(self.release_bcs),
            )


def register_boundary_conditions(
    *,
    builder: newton.ModelBuilder,
    bc_params: list[BoundaryCondition],
    voxel_size: float,
    bbox_lo: list[float],
    bbox_hi: list[float],
) -> BoundaryConditionRuntime:
    """Register shape BCs in builder and return runtime BC handlers."""
    runtime = BoundaryConditionRuntime()
    if not isinstance(bc_params, list):
        return runtime

    for bc in bc_params:
        if isinstance(bc, BoundingBox):
            margin = max(0.01, voxel_size * 2.0)
            wall_cfg = newton.ModelBuilder.ShapeConfig(ke=0.0, kd=0.0, mu=0.3)
            planes = build_bounding_box_planes(lo=bbox_lo, hi=bbox_hi, margin=margin)
            for p in planes:
                builder.add_shape_plane(plane=p, cfg=wall_cfg)
            continue

        if isinstance(bc, SurfaceCollider):
            plane, mu = surface_plane_from_bc(bc)
            cfg = newton.ModelBuilder.ShapeConfig(ke=0.0, kd=0.0, mu=mu)
            builder.add_shape_plane(plane=plane, cfg=cfg)
            continue

        if isinstance(bc, ReleaseParticlesSequentially):
            runtime.release_bcs.append(bc)
            continue

        raise ValueError(
            f"[NewtonMPM] unsupported boundary condition type: {type(bc).__name__}"
        )

    return runtime


