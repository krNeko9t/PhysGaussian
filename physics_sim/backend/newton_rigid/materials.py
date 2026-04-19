"""Material and body-spec helpers for NewtonRigid backend."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any

import newton

from physics_sim.coord import E_GRAVITY_MISSING, gravity_contract_error, normalize_internal_gravity
from physics_sim.errors import configuration_error


@dataclass
class BodySpec:
    """One rigid body creation request resolved from material config."""

    particle_indices: list[int]
    name: str
    material: dict[str, Any]


def resolve_gravity(material_params: dict[str, Any]) -> tuple[float, float, float]:
    if "g" not in material_params:
        raise gravity_contract_error(
            E_GRAVITY_MISSING,
            backend="newton_rigid",
            config_path="material.g",
            detail="set_material() missing required gravity vector",
            suggestion="pass g as [0, -|g|, 0], usually from backend_init._resolve_gravity",
        )
    return normalize_internal_gravity(
        material_params.get("g"),
        backend="newton_rigid",
        config_path="material.g",
        allow_scalar=False,
    )


def resolve_solver_options(material_params: dict[str, Any]) -> tuple[int, float]:
    solver_opts = material_params.get("newton_solver_opts", {})
    iterations = int(solver_opts.get("iterations", 10))
    relaxation = float(solver_opts.get("contact_relaxation", 0.8))
    return iterations, relaxation


def build_base_shape_config(
    *,
    material_params: dict[str, Any],
    use_sdf: bool,
    sdf_resolution: int,
    sdf_narrow_band: tuple[float, float] | tuple[Any, Any],
) -> newton.ModelBuilder.ShapeConfig:
    default_mu = float(material_params.get("mu", 0.5))
    default_density = float(material_params.get("density", 1000.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=default_density, mu=default_mu)
    if "ke" in material_params:
        cfg.ke = float(material_params["ke"])
    if "kd" in material_params:
        cfg.kd = float(material_params["kd"])
    if use_sdf:
        cfg.sdf_max_resolution = sdf_resolution
        cfg.sdf_narrow_band_range = tuple(sdf_narrow_band)
        cfg.contact_margin = 0.01
    return cfg


def iter_body_specs(
    material_params: dict[str, Any],
    *,
    n_particles: int,
) -> list[BodySpec]:
    per_object = material_params.get("per_object")
    if per_object is not None and not isinstance(per_object, list):
        detail = f"per_object_type={type(per_object).__name__}"
        raise configuration_error(
            owner="newton_rigid",
            operation="set_material",
            expected="material.per_object must be a list",
            detail=detail,
        )
    if per_object is None:
        return [
            BodySpec(
                particle_indices=list(range(n_particles)),
                name="single_body",
                material={},
            )
        ]

    specs: list[BodySpec] = []
    for obj in per_object:
        if not isinstance(obj, dict):
            detail = f"per_object_item_type={type(obj).__name__}"
            raise configuration_error(
                owner="newton_rigid",
                operation="set_material",
                expected="each material.per_object item must be dict",
                detail=detail,
            )
        specs.append(
            BodySpec(
                particle_indices=obj["particle_indices"],
                name=obj.get("name", "?"),
                material=obj.get("material", {}),
            )
        )
    return specs


def build_shape_config_for_body(
    *,
    base_cfg: newton.ModelBuilder.ShapeConfig,
    body_material: dict[str, Any],
) -> newton.ModelBuilder.ShapeConfig:
    cfg = copy(base_cfg)
    cfg.density = float(body_material.get("density", cfg.density))
    if "mu" in body_material:
        cfg.mu = float(body_material["mu"])
    elif "friction" in body_material:
        cfg.mu = float(body_material["friction"])
    if "ke" in body_material:
        cfg.ke = float(body_material["ke"])
    if "kd" in body_material:
        cfg.kd = float(body_material["kd"])
    return cfg
