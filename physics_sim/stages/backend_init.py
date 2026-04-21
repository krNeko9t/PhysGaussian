"""Stage 2: Backend initialization, material setup, and boundary conditions.

All positions, directions, and boundary conditions are already in the
internal Y-up coordinate system after :func:`setup_scene`.  This stage
transforms user-specified "world" BCs (which are in source coordinates)
into internal coordinates using :mod:`physics_sim.coord`.

Gravity
-------
Direction is always ``[0, -g_magnitude, 0]`` in internal Y-up space.
Each :class:`MaterialSpec` carries its own ``g_magnitude`` (default 9.8);
this stage reads the first dynamic object's value and forms the vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from physics_sim.backend.registry import create_backend
from physics_sim.backend.spec import MaterialSetupSpec, ObjectRuntimeInfo
from physics_sim.config.models import SimConfig
from physics_sim.coord import (
    E_GRAVITY_SHAPE,
    gravity_contract_error,
    gravity_vector,
)
from physics_sim.logging_utils import get_logger
from physics_sim.stages.boundary_normalization import normalize_boundary_conditions

if TYPE_CHECKING:
    from physics_sim.backend.base import PhysicsBackend
    from physics_sim.stages.scene_setup import SceneData

LOGGER = get_logger(__name__)


def _resolve_gravity(objects_runtime: list[ObjectRuntimeInfo]) -> tuple[float, float, float]:
    """Read ``g_magnitude`` from the first dynamic object and form ``[0, -g, 0]``."""
    default_magnitude = 9.8
    if not objects_runtime:
        return tuple(gravity_vector(default_magnitude, device="cpu").tolist())

    info = objects_runtime[0]
    magnitude = abs(float(info.material.g_magnitude))
    if not np.isfinite(magnitude):
        raise gravity_contract_error(
            E_GRAVITY_SHAPE,
            backend="backend_init",
            config_path=f"objects_runtime[0].material.g_magnitude",
            material_name=info.name,
            raw_g=info.material.g_magnitude,
            detail="g_magnitude must be finite",
            suggestion="set g_magnitude to a finite number, e.g. 9.8",
        )
    return tuple(gravity_vector(magnitude, device="cpu").tolist())


def init_backend(
    cfg: SimConfig,
    scene_data: SceneData,
    device: str = "cuda:0",
) -> PhysicsBackend:
    """Create, configure, and finalize the physics backend."""
    bt = cfg.backend.type
    LOGGER.info("Initialising backend: %s", bt)
    backend = create_backend(cfg.backend, device=device)

    dyn = scene_data.dynamic_init
    if scene_data.gs_num > 0:
        init_quats = dyn.quats if scene_data.gs_type == "2dgs" else None
        init_scales = dyn.scales if scene_data.gs_type == "2dgs" else None
        backend.initialize(
            dyn.pos,
            dyn.vol,
            dyn.cov,
            init_quats=init_quats,
            init_scales=init_scales,
        )

    material_spec = MaterialSetupSpec(
        gravity=_resolve_gravity(scene_data.objects_runtime),
        per_object=list(scene_data.objects_runtime),
    )
    backend.set_material(material_spec)

    normalized_bcs = normalize_boundary_conditions(
        raw_boundary_conditions=list(cfg.boundary_conditions),
        collider_objects=scene_data.collider_objects,
        source_axes=scene_data.coord.source_axes,
    )
    backend.set_boundary_conditions(normalized_bcs, cfg.time)
    backend.finalize()
    backend.apply_constraints(scene_data.resolved_constraints)

    return backend
