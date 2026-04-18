"""Material resolution and application for Newton MPM backend."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from physics_sim.coord import (
    E_GRAVITY_MISSING,
    gravity_contract_error,
    normalize_internal_gravity,
)
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)

MATERIAL_PRESETS: dict[str, dict[str, float | None]] = {
    "sand": dict(
        friction=None,
        yield_pressure=1.0e12,
        yield_stress=0.0,
        tensile_yield_ratio=0.0,
        hardening=0.0,
    ),
    "jelly": dict(
        friction=0.0,
        yield_pressure=1.0e6,
        yield_stress=5.0e4,
        tensile_yield_ratio=0.1,
        hardening=3.0,
    ),
    "snow": dict(
        friction=0.1,
        yield_pressure=2.0e4,
        yield_stress=1.0e3,
        tensile_yield_ratio=0.05,
        hardening=10.0,
    ),
    "mud": dict(
        friction=0.0,
        yield_pressure=1.0e10,
        yield_stress=3.0e2,
        tensile_yield_ratio=1.0,
        hardening=2.0,
    ),
    "metal": dict(
        friction=0.3,
        yield_pressure=1.0e12,
        yield_stress=1.0e8,
        tensile_yield_ratio=0.0,
        hardening=0.0,
    ),
    "foam": dict(
        friction=0.5,
        yield_pressure=1.0e6,
        yield_stress=1.0e4,
        tensile_yield_ratio=0.1,
        hardening=5.0,
    ),
    "plasticine": dict(
        friction=0.5,
        yield_pressure=1.0e6,
        yield_stress=5.0e3,
        tensile_yield_ratio=0.1,
        hardening=3.0,
    ),
}


def friction_from_angle(friction_angle_deg: float) -> float:
    """Convert friction angle in degrees to Coulomb friction coefficient."""
    rad = friction_angle_deg / 180.0 * math.pi
    return math.tan(rad)


def resolve_friction(
    *,
    mat_name: str,
    preset: dict[str, Any],
    material_cfg: dict[str, Any],
) -> float:
    """Resolve friction in a single consistent path."""
    if "friction" in material_cfg:
        return float(material_cfg["friction"])

    if "friction_angle" in material_cfg:
        return friction_from_angle(float(material_cfg["friction_angle"]))

    friction = preset.get("friction")
    if friction is not None:
        return float(friction)

    if mat_name == "sand":
        return 0.68
    return 0.0


def apply_material_to_model(
    *,
    model: Any,
    volumes: np.ndarray,
    material_params: dict[str, Any],
    device: str,
) -> None:
    """Apply global and per-object material parameters to finalized model."""
    if "g" not in material_params:
        raise gravity_contract_error(
            E_GRAVITY_MISSING,
            backend="newton_mpm",
            config_path="material.g",
            detail="set_material() missing required gravity vector",
            suggestion="pass g as [0, -|g|, 0], usually from backend_init._resolve_gravity",
        )
    g = normalize_internal_gravity(
        material_params.get("g"),
        backend="newton_mpm",
        config_path="material.g",
        allow_scalar=False,
    )
    model.set_gravity(g)

    mat_name = material_params.get("material", "jelly")
    preset = MATERIAL_PRESETS.get(mat_name, MATERIAL_PRESETS["jelly"])

    E = float(material_params.get("E", 1e5))
    nu = float(material_params.get("nu", 0.3))
    friction = resolve_friction(
        mat_name=mat_name,
        preset=preset,
        material_cfg=material_params,
    )
    yield_pressure = float(material_params.get("yield_pressure", preset.get("yield_pressure", 1e12)))
    yield_stress = float(material_params.get("yield_stress", preset.get("yield_stress", 0.0)))
    tensile_ratio = float(material_params.get("tensile_yield_ratio", preset.get("tensile_yield_ratio", 0.0)))
    hardening = float(material_params.get("hardening", preset.get("hardening", 0.0)))

    model.mpm.young_modulus.fill_(E)
    model.mpm.poisson_ratio.fill_(nu)
    model.mpm.friction.fill_(friction)
    model.mpm.yield_pressure.fill_(yield_pressure)
    model.mpm.yield_stress.fill_(yield_stress)
    model.mpm.tensile_yield_ratio.fill_(tensile_ratio)
    model.mpm.hardening.fill_(hardening)

    base_density = float(material_params.get("density", 200.0))
    _assign_density_for_indices(
        model=model,
        volumes=volumes,
        density=base_density,
        particle_indices=np.arange(volumes.shape[0]),
        device=device,
    )

    per_object = material_params.get("per_object")
    if per_object:
        apply_per_object_materials(
            model=model,
            volumes=volumes,
            per_object=per_object,
            device=device,
        )


def apply_per_object_materials(
    *,
    model: Any,
    volumes: np.ndarray,
    per_object: list[dict[str, Any]],
    device: str,
) -> None:
    """Apply per-object overrides on top of the global material."""
    for obj in per_object:
        raw_idx = obj["particle_indices"]
        if len(raw_idx) == 0:
            continue
        mat = obj.get("material", {})
        name = obj.get("name", "?")
        idx_wp = wp.array(np.array(raw_idx, dtype=np.int32), dtype=int, device=device)

        mat_name = mat.get("material", "jelly")
        preset = MATERIAL_PRESETS.get(mat_name, MATERIAL_PRESETS["jelly"])
        if "E" in mat:
            model.mpm.young_modulus[idx_wp].fill_(float(mat["E"]))
        if "nu" in mat:
            model.mpm.poisson_ratio[idx_wp].fill_(float(mat["nu"]))

        friction = resolve_friction(mat_name=mat_name, preset=preset, material_cfg=mat)
        model.mpm.friction[idx_wp].fill_(friction)
        model.mpm.yield_pressure[idx_wp].fill_(float(mat.get("yield_pressure", preset.get("yield_pressure", 1e12))))
        model.mpm.yield_stress[idx_wp].fill_(float(mat.get("yield_stress", preset.get("yield_stress", 0.0))))
        model.mpm.tensile_yield_ratio[idx_wp].fill_(
            float(mat.get("tensile_yield_ratio", preset.get("tensile_yield_ratio", 0.0)))
        )
        model.mpm.hardening[idx_wp].fill_(float(mat.get("hardening", preset.get("hardening", 0.0))))

        if "density" in mat:
            _assign_density_for_indices(
                model=model,
                volumes=volumes,
                density=float(mat["density"]),
                particle_indices=np.array(raw_idx, dtype=np.int32),
                device=device,
            )

        LOGGER.info(
            "[NewtonMPM] object='%s' count=%s preset=%s friction=%.4f",
            name,
            len(raw_idx),
            mat_name,
            friction,
        )


def _assign_density_for_indices(
    *,
    model: Any,
    volumes: np.ndarray,
    density: float,
    particle_indices: np.ndarray,
    device: str,
) -> None:
    mass_np = model.particle_mass.numpy()
    inv_mass_np = model.particle_inv_mass.numpy()
    m = (volumes[particle_indices] * density).astype(np.float32)
    mass_np[particle_indices] = m
    inv_mass_np[particle_indices] = np.where(m > 0.0, 1.0 / m, 0.0).astype(np.float32)
    model.particle_mass.assign(
        wp.from_numpy(mass_np.astype(np.float32), dtype=float, device=device)
    )
    model.particle_inv_mass.assign(
        wp.from_numpy(inv_mass_np.astype(np.float32), dtype=float, device=device)
    )
