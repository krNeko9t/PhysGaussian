"""Backend factory and per-backend material defaults.

``resolve_material`` fills in missing material fields using the backend's
default values, so users only need to specify what they care about.

``create_backend`` is a simple dispatch that returns the right
``PhysicsBackend`` instance for a given ``BackendConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physics_sim.errors import unknown_registry_error

if TYPE_CHECKING:
    from physics_sim.backend.base import PhysicsBackend
    from physics_sim.config.models import BackendConfig


BACKEND_MATERIAL_DEFAULTS: dict[str, dict] = {
    "newton_rigid": dict(
        density=1000.0,
        mu=0.5,
        E=1e5,
        nu=0.3,
        collision_geometry="convex_hull",
    ),
    "newton_vbd": dict(
        density=1000.0,
        mu=0.5,
        physics="rigid",
        collision_geometry="convex_hull",
        k_mu=1e5,
        k_lambda=1e5,
        k_damp=1e-3,
    ),
    "newton_mpm": dict(
        density=200.0,
        mu=0.5,
        E=1e5,
        nu=0.3,
        material="jelly",
    ),
}


def resolve_material(backend_type: str, material: dict) -> dict:
    """Merge backend defaults underneath user-supplied material values."""
    defaults = BACKEND_MATERIAL_DEFAULTS.get(backend_type, {})
    return {**defaults, **material}


def create_backend(cfg: BackendConfig, device: str = "cuda:0") -> PhysicsBackend:
    """Instantiate the physics backend matching *cfg.type*."""
    if cfg.type == "newton_rigid":
        from physics_sim.backend.newton_rigid import NewtonRigidBackend
        return NewtonRigidBackend(device=device)
    if cfg.type == "newton_vbd":
        from physics_sim.backend.newton_vbd import NewtonVBDBackend
        return NewtonVBDBackend(device=device)
    if cfg.type == "newton_mpm":
        from physics_sim.backend.newton_mpm import NewtonMPMBackend
        return NewtonMPMBackend(device=device)
    if cfg.type == "none":
        from physics_sim.backend.none import NoneBackend
        return NoneBackend(device=device)
    raise unknown_registry_error(
        registry="backend",
        key=cfg.type,
        available=BACKEND_MATERIAL_DEFAULTS.keys(),
    )
