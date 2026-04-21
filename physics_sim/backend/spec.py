"""Shared spec types at the scene↔backend boundary.

These types live outside individual backend modules so that both
``physics_sim.stages.scene_setup`` (producer) and
``physics_sim.backend.base`` (consumer) can reference them without
circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from physics_sim.config.models import MaterialSpec


@dataclass
class PartRuntimeInfo:
    """Per-part runtime descriptor.

    Single source of truth for downstream "by-part" lookups.
    ``particle_indices`` is a contiguous range built from the concatenation
    order of ``sim_objects`` in ``DynamicSceneInit``.
    """
    name: str
    particle_indices: list[int]
    material: MaterialSpec
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class MaterialSetupSpec:
    """Typed payload passed to ``PhysicsBackend.set_material``.

    Contains only material/physics concerns.  Solver numerical options
    (``n_grid``, ``grid_lim``, ``solver_iterations``, ...) live on
    ``BackendConfig`` and are consumed directly at backend construction.
    """
    gravity: tuple[float, float, float]
    per_part: list[PartRuntimeInfo] = field(default_factory=list)
