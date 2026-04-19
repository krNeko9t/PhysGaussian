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
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class BoundaryConditionRuntime:
    """Runtime BC state for NewtonMPM backend."""

    velocity_bcs: list[dict[str, Any]] = field(default_factory=list)
    impulse_bcs: list[dict[str, Any]] = field(default_factory=list)
    release_bcs: list[dict[str, Any]] = field(default_factory=list)
    velocity_rotation_bcs: list[dict[str, Any]] = field(default_factory=list)

    def apply_velocity_bcs(
        self,
        *,
        state: Any,
        time_now: float,
        dt: float,
        device: str,
    ) -> None:
        if not self.velocity_bcs:
            return
        pos_np = state.particle_q.numpy()
        vel_np = state.particle_qd.numpy()

        for bc in self.velocity_bcs:
            start = float(bc.get("start_time", 0.0))
            end = float(bc.get("end_time", 1e3))
            if not (start <= time_now <= end):
                continue
            point = np.array(bc["point"], dtype=np.float32)
            size = np.array(bc["size"], dtype=np.float32)
            velocity = np.array(bc["velocity"], dtype=np.float32)
            lo = point - size
            hi = point + size
            mask = np.all((pos_np >= lo) & (pos_np <= hi), axis=1)
            if np.any(mask):
                vel_np[mask] = velocity

        state.particle_qd.assign(
            wp.from_numpy(vel_np.astype(np.float32), dtype=wp.vec3, device=device)
        )

    def apply_impulse_bcs(
        self,
        *,
        state: Any,
        particle_mass: Any,
        time_now: float,
        dt: float,
        device: str,
    ) -> None:
        if not self.impulse_bcs:
            return
        pos_np = state.particle_q.numpy()
        vel_np = state.particle_qd.numpy()
        mass_np = particle_mass.numpy()

        for bc in self.impulse_bcs:
            start = float(bc.get("start_time", 0.0))
            num_dt = int(bc.get("num_dt", 1))
            if not (start <= time_now < start + num_dt * dt + 1e-10):
                continue
            force = np.array(bc["force"], dtype=np.float32)
            point = np.array(bc.get("point", [1, 1, 1]), dtype=np.float32)
            size = np.array(bc.get("size", [1, 1, 1]), dtype=np.float32)
            lo = point - size
            hi = point + size
            mask = np.all((pos_np >= lo) & (pos_np <= hi), axis=1)
            if np.any(mask):
                for i in np.where(mask)[0]:
                    m = mass_np[i]
                    if m > 0:
                        vel_np[i] += force * dt / m

        state.particle_qd.assign(
            wp.from_numpy(vel_np.astype(np.float32), dtype=wp.vec3, device=device)
        )

    def log_unimplemented(self) -> None:
        if self.release_bcs:
            LOGGER.warning(
                "[NewtonMPM] release_particles_sequentially currently degrades to no-op count=%s",
                len(self.release_bcs),
            )
        if self.velocity_rotation_bcs:
            LOGGER.warning(
                "[NewtonMPM] enforce_particle_velocity_rotation currently unsupported count=%s",
                len(self.velocity_rotation_bcs),
            )


def register_boundary_conditions(
    *,
    builder: newton.ModelBuilder,
    bc_params: list[dict[str, Any]],
    voxel_size: float,
    bbox_lo: list[float],
    bbox_hi: list[float],
) -> BoundaryConditionRuntime:
    """Register shape BCs in builder and return runtime BC handlers."""
    runtime = BoundaryConditionRuntime()
    if not isinstance(bc_params, list):
        return runtime

    for bc in bc_params:
        bc_type = bc.get("type")
        if bc_type == "bounding_box":
            margin = float(bc.get("margin", max(0.01, voxel_size * 2.0)))
            wall_mu = float(bc.get("friction", 0.3))
            wall_cfg = newton.ModelBuilder.ShapeConfig(ke=0.0, kd=0.0, mu=wall_mu)
            planes = build_bounding_box_planes(lo=bbox_lo, hi=bbox_hi, margin=margin)
            for p in planes:
                builder.add_shape_plane(plane=p, cfg=wall_cfg)
            continue

        if bc_type == "surface_collider":
            plane, mu = surface_plane_from_bc(bc)
            cfg = newton.ModelBuilder.ShapeConfig(ke=0.0, kd=0.0, mu=mu)
            builder.add_shape_plane(plane=plane, cfg=cfg)
            continue

        if bc_type in ("cuboid", "enforce_particle_translation"):
            runtime.velocity_bcs.append(bc)
            continue
        if bc_type == "particle_impulse":
            runtime.impulse_bcs.append(bc)
            continue
        if bc_type == "release_particles_sequentially":
            runtime.release_bcs.append(bc)
            continue
        if bc_type == "enforce_particle_velocity_rotation":
            runtime.velocity_rotation_bcs.append(bc)
            continue

        raise ValueError(f"[NewtonMPM] unsupported boundary condition type: {bc_type!r}")

    return runtime

