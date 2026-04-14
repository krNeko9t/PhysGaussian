"""Backend configuration preset factories."""

from physics_sim.config.models import (
    NewtonMPMConfig,
    NewtonRigidConfig,
    NewtonVBDConfig,
    NoneBackendConfig,
)


def newton_rigid(**kw) -> NewtonRigidConfig:
    return NewtonRigidConfig(**kw)


def newton_vbd(**kw) -> NewtonVBDConfig:
    return NewtonVBDConfig(**kw)


def newton_mpm(**kw) -> NewtonMPMConfig:
    return NewtonMPMConfig(**kw)


def none_backend() -> NoneBackendConfig:
    return NoneBackendConfig()
