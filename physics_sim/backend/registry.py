"""Backend factory.

Each :class:`MaterialSpec` carries its own Pydantic defaults, so there is
no global defaults merge here anymore — the previous
``BACKEND_MATERIAL_DEFAULTS`` dict is gone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physics_sim.errors import unknown_registry_error

if TYPE_CHECKING:
    from physics_sim.backend.base import PhysicsBackend
    from physics_sim.config.models import BackendConfig


_AVAILABLE_BACKENDS = ("newton_rigid", "newton_vbd", "newton_mpm", "none")


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
        available=_AVAILABLE_BACKENDS,
    )

