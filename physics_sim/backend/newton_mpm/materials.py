"""Material application for Newton MPM backend."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from physics_sim.backend.spec import MaterialSetupSpec
from physics_sim.config.models import MPMMaterial
from physics_sim.errors import configuration_error
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


def friction_from_angle(friction_angle_deg: float) -> float:
    """Convert friction angle in degrees to Coulomb friction coefficient."""
    rad = friction_angle_deg / 180.0 * math.pi
    return math.tan(rad)


def resolve_friction(material: MPMMaterial) -> float:
    """Resolve friction coefficient from explicit value or friction_angle."""
    if material.friction is not None:
        return float(material.friction)
    if material.friction_angle is not None:
        return friction_from_angle(float(material.friction_angle))
    return 0.0


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


def _apply_mpm_params_to_indices(
    *,
    model: Any,
    volumes: np.ndarray,
    material: MPMMaterial,
    particle_indices: np.ndarray,
    device: str,
) -> None:
    """Write MPM per-particle fields for the given index subset."""
    if particle_indices.size == 0:
        return
    friction = resolve_friction(material)
    if particle_indices.size == volumes.shape[0]:
        model.mpm.young_modulus.fill_(float(material.E))
        model.mpm.poisson_ratio.fill_(float(material.nu))
        model.mpm.friction.fill_(friction)
        model.mpm.yield_pressure.fill_(float(material.yield_pressure))
        model.mpm.yield_stress.fill_(float(material.yield_stress))
        model.mpm.tensile_yield_ratio.fill_(float(material.tensile_yield_ratio))
        model.mpm.hardening.fill_(float(material.hardening))
    else:
        idx_wp = wp.array(np.asarray(particle_indices, dtype=np.int32), dtype=int, device=device)
        model.mpm.young_modulus[idx_wp].fill_(float(material.E))
        model.mpm.poisson_ratio[idx_wp].fill_(float(material.nu))
        model.mpm.friction[idx_wp].fill_(friction)
        model.mpm.yield_pressure[idx_wp].fill_(float(material.yield_pressure))
        model.mpm.yield_stress[idx_wp].fill_(float(material.yield_stress))
        model.mpm.tensile_yield_ratio[idx_wp].fill_(float(material.tensile_yield_ratio))
        model.mpm.hardening[idx_wp].fill_(float(material.hardening))

    _assign_density_for_indices(
        model=model,
        volumes=volumes,
        density=float(material.density),
        particle_indices=np.asarray(particle_indices, dtype=np.int32),
        device=device,
    )


def apply_material_to_model(
    *,
    model: Any,
    volumes: np.ndarray,
    spec: MaterialSetupSpec,
    device: str,
) -> None:
    """Apply gravity and per-particle MPM material fields.

    Particles not covered by ``per_part`` are filled with
    :class:`MPMMaterial` defaults so that filled particles (from
    particle_filling) still have valid material parameters.
    """
    model.set_gravity(tuple(float(x) for x in spec.gravity))

    # Fill defaults for all particles first.
    default = MPMMaterial()
    _apply_mpm_params_to_indices(
        model=model,
        volumes=volumes,
        material=default,
        particle_indices=np.arange(volumes.shape[0], dtype=np.int32),
        device=device,
    )

    for info in spec.per_part:
        if not isinstance(info.material, MPMMaterial):
            detail = f"part={info.name} material_type={type(info.material).__name__}"
            raise configuration_error(
                owner="newton_mpm",
                operation="set_material",
                expected="newton_mpm requires MPMMaterial per object",
                detail=detail,
            )
        if len(info.particle_indices) == 0:
            continue
        _apply_mpm_params_to_indices(
            model=model,
            volumes=volumes,
            material=info.material,
            particle_indices=np.asarray(info.particle_indices, dtype=np.int32),
            device=device,
        )
        LOGGER.info(
            "[NewtonMPM] object='%s' count=%s E=%.3g nu=%.3g friction=%.4f",
            info.name,
            len(info.particle_indices),
            float(info.material.E),
            float(info.material.nu),
            resolve_friction(info.material),
        )
