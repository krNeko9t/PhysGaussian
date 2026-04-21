"""Material and body-spec helpers for NewtonRigid backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import newton

from physics_sim.backend.spec import ObjectRuntimeInfo
from physics_sim.config.models import RigidMaterial
from physics_sim.errors import configuration_error


@dataclass
class BodySpec:
    """One rigid body creation request resolved from material config."""

    particle_indices: list[int]
    name: str
    material: RigidMaterial
    initial_velocity: tuple[float, float, float] | None = None


def build_body_shape_config(
    *,
    material: RigidMaterial,
    use_sdf: bool,
    sdf_resolution: int,
    sdf_narrow_band: tuple[float, float] | tuple[Any, Any],
) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig(
        density=float(material.density),
        mu=float(material.mu),
    )
    if material.ke is not None:
        cfg.ke = float(material.ke)
    if material.kd is not None:
        cfg.kd = float(material.kd)
    if use_sdf:
        cfg.sdf_max_resolution = sdf_resolution
        cfg.sdf_narrow_band_range = tuple(sdf_narrow_band)
        cfg.contact_margin = 0.01
    return cfg


def iter_body_specs(
    *,
    per_object: list[ObjectRuntimeInfo],
    n_particles: int,
) -> list[BodySpec]:
    if not per_object:
        # Empty per_object → single body covering all particles with default material.
        return [
            BodySpec(
                particle_indices=list(range(n_particles)),
                name="single_body",
                material=RigidMaterial(),
            )
        ]

    specs: list[BodySpec] = []
    for info in per_object:
        material = info.material
        if not isinstance(material, RigidMaterial):
            detail = f"object={info.name} material_type={type(material).__name__}"
            raise configuration_error(
                owner="newton_rigid",
                operation="set_material",
                expected="newton_rigid requires RigidMaterial per object",
                detail=detail,
            )
        iv_raw = info.initial_velocity
        iv: tuple[float, float, float] | None = None
        if any(float(v) != 0.0 for v in iv_raw):
            iv = (float(iv_raw[0]), float(iv_raw[1]), float(iv_raw[2]))
        specs.append(
            BodySpec(
                particle_indices=list(info.particle_indices),
                name=info.name,
                material=material,
                initial_velocity=iv,
            )
        )
    return specs
