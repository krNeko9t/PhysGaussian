"""Boundary-condition registration for NewtonRigid backend."""

from __future__ import annotations

import numpy as np
import newton

from physics_sim.backend.newton_common.boundary import build_bounding_box_planes, surface_plane_from_bc
from physics_sim.config.models import (
    BoundaryCondition,
    BoundingBox,
    SurfaceCollider,
)
from physics_sim.errors import configuration_error, lifecycle_error


def register_boundary_conditions(
    *,
    builder: newton.ModelBuilder,
    bc_params: list[BoundaryCondition],
    bbox_lo: np.ndarray | None,
    bbox_hi: np.ndarray | None,
) -> list[tuple[list[float], float]]:
    if not isinstance(bc_params, list):
        detail = f"bc_params_type={type(bc_params).__name__}"
        raise configuration_error(
            owner="newton_rigid",
            operation="set_boundary_conditions",
            expected="bc_params must be list",
            detail=detail,
        )

    plane_equations: list[tuple[list[float], float]] = []
    for bc in bc_params:
        if isinstance(bc, SurfaceCollider):
            plane, mu = surface_plane_from_bc(bc)
            plane_cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
            builder.add_shape_plane(plane=plane, cfg=plane_cfg)
            n_list = [plane[0], plane[1], plane[2]]
            d_f = plane[3]
            plane_equations.append((n_list, d_f))
            continue

        if isinstance(bc, BoundingBox):
            if bbox_lo is None or bbox_hi is None:
                detail = "bounding box unavailable; initialize() did not set bbox"
                raise lifecycle_error(
                    owner="newton_rigid",
                    operation="set_boundary_conditions",
                    expected="initialize() must set bbox before bounding_box BC",
                    detail=detail,
                )
            planes = build_bounding_box_planes(lo=bbox_lo, hi=bbox_hi, margin=0.01)
            wall_cfg = newton.ModelBuilder.ShapeConfig(mu=0.3)
            for plane in planes:
                builder.add_shape_plane(plane=plane, cfg=wall_cfg)
            continue

        # Other BC types (e.g. release_particles_sequentially) not supported by rigid backend.

    return plane_equations

