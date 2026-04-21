"""Backend factory.

Each backend consumes its typed :class:`BackendConfig` directly at
construction time — solver numerical options (``n_grid``, ``grid_lim``,
``solver_iterations``, ...) never flow through ``set_material`` or
``initialize``.  The previous ``BACKEND_MATERIAL_DEFAULTS`` /
``resolve_material`` helpers are gone; MaterialSpec Pydantic defaults
replace them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physics_sim.errors import unknown_registry_error

if TYPE_CHECKING:
    from physics_sim.backend.base import PhysicsBackend
    from physics_sim.config.models import BackendConfig


_AVAILABLE_BACKENDS = ("newton_rigid", "newton_vbd", "newton_mpm", "none")


def create_backend(cfg: BackendConfig, device: str = "cuda:0") -> PhysicsBackend:
    """Instantiate the physics backend matching ``cfg.type``.

    The typed ``cfg`` is passed straight to the backend constructor so
    that each backend can read its own fields via attribute access.
    """
    if cfg.type == "newton_rigid":
        from physics_sim.backend.newton_rigid import NewtonRigidBackend
        return NewtonRigidBackend(cfg=cfg, device=device)
    if cfg.type == "newton_vbd":
        from physics_sim.backend.newton_vbd import NewtonVBDBackend
        return NewtonVBDBackend(cfg=cfg, device=device)
    if cfg.type == "newton_mpm":
        from physics_sim.backend.newton_mpm import NewtonMPMBackend
        return NewtonMPMBackend(cfg=cfg, device=device)
    if cfg.type == "none":
        from physics_sim.backend.none import NoneBackend
        return NoneBackend(cfg=cfg, device=device)
    raise unknown_registry_error(
        registry="backend",
        key=cfg.type,
        available=_AVAILABLE_BACKENDS,
    )

