"""Rigid mesh builders used by NewtonVBD backend."""

from __future__ import annotations

import newton
import torch

from physics_sim.backend.newton_common import (
    BuiltRigidGeometry,
    create_rigid_body_geometry,
)
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)
RigidBuildResult = BuiltRigidGeometry


def create_rigid_body(
    *,
    builder: newton.ModelBuilder,
    init_positions: torch.Tensor,
    init_covariances: torch.Tensor,
    init_quats: torch.Tensor | None,
    init_scales: torch.Tensor | None,
    device: str,
    particle_indices: list[int],
    shape_cfg: newton.ModelBuilder.ShapeConfig,
    name: str,
    collision_geo: str | None,
    default_collision_geo: str,
    alpha: float | None,
    max_triangles: int,
) -> RigidBuildResult:
    return create_rigid_body_geometry(
        builder=builder,
        init_positions=init_positions,
        init_covariances=init_covariances,
        init_quats=init_quats,
        init_scales=init_scales,
        device=device,
        particle_indices=particle_indices,
        shape_cfg=shape_cfg,
        name=name,
        collision_geo=collision_geo,
        default_collision_geo=default_collision_geo,
        alpha=alpha,
        max_triangles=max_triangles,
        log_prefix="NewtonVBD",
        logger=LOGGER,
    )
